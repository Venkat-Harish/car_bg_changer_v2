"""Grounding in the room path: tyre bottoms on the car's outline, the contact lock, and the shadow layers
drawn on the re-rendered floor (never on the car).

Tyre bottoms. A predicted tyre counts when it is on the side facing the camera (both tyres of the facing end
in end views) and no other proxy piece hides its contact patch. Its bottom is the lowest point of the levelled car's outline within one tyre radius of
the predicted column, if that point is a local bump (not a slope), lies 0-15% of the car's height below the
prediction, and is not on a cut edge. End views use the same rule without the proxy's occlusion test (its
full-width bumper slab hides tyres whose bottoms really show below the bumper corners).

Contact lock. With two or more bottoms, the virtual camera's height and pitch are adjusted (regularised,
within +-15% and +-3 degrees) so the predicted contacts land on them; with one, the height only. The car's
pixels never move: the floor is brought to the tyres.

Shadow layers, in linear light on floor pixels only, with the background's room ShadowStyle:
  core and skirt: the footprint (grounding.footprint_layer) looked up through each floor pixel's own floor
      point, then shifted per column onto the found tyre bottoms (grounding.contact_offsets);
  tyre patches: at the found bottoms, about the size of the real contact patch;
  contact, gap, band: from the car's own outline (grounding.outline_floor / silhouette_layers), which
      also grounds end views whose tyres hide behind the bumpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from bg_changer.carmodel import CarShape, _wheel_visible
from bg_changer.grounding import (Contact, ReflectionStyle, ShadowStyle, add_reflection, contact_offsets,
                                  footprint_layer, outline_floor, silhouette_layers)

from .pose import PIECES, TYRE_PIECE, ZN, tyre_width

TYRE_NAMES = ("FL", "FR", "RL", "RR")
LOCK_IOU = 0.90
LOCK_BOUNDS = (0.15, 3.0)  # relative camera height, pitch (deg)
PATCH_HALF_LEN_M = 0.07  # the tyre's contact shadow: about the real contact patch, centred on the found bottom
PATCH_WIDEN_M = 0.0


def project_rc(P, R, C, K):
    """Car points -> (u, v, z) through a camera given as rotation R and centre C (car coordinates)."""
    q = (np.atleast_2d(P) - C) @ R.T
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * q[:, 0] / z + K[0, 2], K[1, 1] * q[:, 1] / z + K[1, 2], z


def rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


@dataclass
class TyreBottom:
    name: str
    u_pred: float  # levelled grid, full resolution
    v_pred: float
    r_px: float
    status: str  # found | hidden | occluded | cut | off | implausible
    u: float | None = None
    v: float | None = None


def _occluded(i, Q, qc, K, margin) -> bool:
    """Contact i hidden behind another proxy piece by more than margin px (Q: camera coordinates)."""
    zc = qc[2]
    uc, vc = K[0, 0] * qc[0] / zc + K[0, 2], K[1, 1] * qc[1] / zc + K[1, 2]
    for j, idx in enumerate(PIECES):
        if j == TYRE_PIECE[i]:
            continue
        q = Q[idx]
        if (q[:, 2] <= 0.05).any() or q[:, 2].min() >= zc - 0.02:
            continue
        hull = cv2.convexHull(np.stack([K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]],
                                       1).astype(np.float32))
        if cv2.pointPolygonTest(hull, (float(uc), float(vc)), True) > margin:
            return True
    return False


def find_tyre_bottoms(alpha: np.ndarray, origin, pts: np.ndarray, shape: CarShape, yaw: float, R: np.ndarray,
                      C: np.ndarray, K: np.ndarray, cut_edges: dict) -> list[TyreBottom]:
    """alpha: the levelled car canvas (full resolution) at `origin` on the levelled grid; (R, C, K): the
    levelled camera (car -> camera rotation, camera centre in car metres). cut_edges: {side: grid position}
    of each cut border on the levelled grid (columns for left/right, rows for top/bottom)."""
    ox, oy = origin
    solid = alpha >= 0.5
    has = solid.any(0)
    if has.sum() < 4:
        return []
    bottom = np.where(has, solid.shape[0] - 1 - np.argmax(solid[::-1], 0), -1).astype(float)
    b = median_filter(bottom, size=5, mode="nearest")
    b[~has] = -1
    rows = np.flatnonzero(solid.any(1))
    car_h = float(rows.max() + 1 - rows.min())
    con = shape.contacts()
    u, v, z = project_rc(con, R, C, K)
    facing = _wheel_visible(yaw)
    sy = math.sin(math.radians(yaw))
    end_view = abs(sy) <= 0.17
    Q = (pts - C) @ R.T
    tw = tyre_width(shape)
    out = []
    for i in range(4):
        if not facing[i] or not np.isfinite(z[i]) or z[i] <= ZN:
            out.append(TyreBottom(TYRE_NAMES[i], float(u[i]), float(v[i]), 0.0, "hidden"))
            continue
        ppm = K[0, 0] / z[i]
        r_px = max(2.0, shape.tyre_radius * ppm)
        tb = TyreBottom(TYRE_NAMES[i], float(u[i]), float(v[i]), r_px, "off")
        near = (sy > 0.17 and i in (1, 3)) or (sy < -0.17 and i in (0, 2))
        if not end_view and not near:  # a far-side tyre: its bottom hides behind the body; a bumper corner or
            tb.status = "far"           # sill there would be mistaken for it
            out.append(tb)
            continue
        cp = con[i].copy()
        if near:
            cp[1] += math.copysign(0.5 * tw, cp[1])  # the outer edge of the contact patch
        # End views: the proxy's full-width bumper "hides" the tyres, but real tyre bottoms show below the
        # bumper corners, so the bump rule below decides there.
        if not end_view and _occluded(i, Q, (cp - C) @ R.T, K, max(2.0, 0.06 * ppm)):
            tb.status = "occluded"
            out.append(tb)
            continue
        uc, vc = u[i] - ox, v[i] - oy  # canvas coordinates
        lo_c, hi_c = int(math.floor(uc - r_px)), int(math.ceil(uc + r_px))
        cols = [j for j in range(max(0, lo_c), min(len(b), hi_c + 1)) if b[j] >= 0]
        guard = []
        for side, pos in cut_edges.items():  # stay clear of a cut edge
            if side in ("left", "right"):
                guard.append((side, pos - ox))
        cols = [j for j in cols if all(abs(j - g) > 4 for _, g in guard)]
        if not cols:
            tb.status = "cut" if guard else "off"
            out.append(tb)
            continue
        j = max(cols, key=lambda q: b[q])
        bv = b[j] + 0.5  # the outline's bottom edge (pixel-centre convention)
        dv = bv - vc
        wide = max(3, int(1.5 * r_px))
        seg = b[max(0, j - wide):j + wide + 1]
        seg = seg[seg >= 0]
        prominent = len(seg) and b[j] - seg.min() >= 0.12 * r_px
        if prominent and -2.0 <= dv <= 0.15 * car_h and abs(j - uc) <= r_px:
            tb.status, tb.u, tb.v = "found", float(j + ox), float(bv + oy)
        else:
            tb.status = "implausible" if prominent else "hidden"
        out.append(tb)
    return out


def contact_lock(bottoms: list[TyreBottom], shape: CarShape, R: np.ndarray, C: np.ndarray, K: np.ndarray):
    """Adjust the camera's height (relative) and pitch so the predicted contacts meet the found bottoms.
    Returns (R, C, info)."""
    con = shape.contacts()
    idx = [i for i, t in enumerate(bottoms) if t.status == "found"]
    info = dict(n=len(idx), mode="none", dh=0.0, dp=0.0, res_before=[], res_after=[], at_bound=False)
    if not idx:
        return R, C, info
    tgt = np.array([bottoms[i].v for i in idx])

    def mod(x):
        C2 = C.copy()
        C2[2] *= 1 + x[0]
        return rx(x[1]) @ R, C2

    def rows(x):
        R2, C2 = mod(x)
        return project_rc(con[idx], R2, C2, K)[1] - tgt

    info["res_before"] = [float(r) for r in rows([0.0, 0.0])]
    if len(idx) >= 2:
        sol = least_squares(lambda x: np.r_[rows(x), 2.0 * x[0] / 0.05, 2.0 * x[1]], np.zeros(2),
                            bounds=([-LOCK_BOUNDS[0], -LOCK_BOUNDS[1]], [LOCK_BOUNDS[0], LOCK_BOUNDS[1]])).x
        info["mode"] = "height+pitch"
    else:
        h = least_squares(lambda x: np.r_[rows([x[0], 0.0]), 2.0 * x[0] / 0.05], np.zeros(1),
                          bounds=([-LOCK_BOUNDS[0]], [LOCK_BOUNDS[0]])).x
        sol = np.r_[h, 0.0]
        info["mode"] = "height"
    R2, C2 = mod(sol)
    info.update(dh=float(sol[0]), dp=float(sol[1]), res_after=[float(r) for r in rows(sol)],
                at_bound=bool(abs(sol[0]) > 0.99 * LOCK_BOUNDS[0] or abs(sol[1]) > 0.99 * LOCK_BOUNDS[1]))
    return R2, C2, info


# ---------------------------------------------------------------- shadow layers on the output floor
def _raster_lookup(layer: np.ndarray, s_raster: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Sample a top-down car-frame raster (s_raster: raster px -> car metres) at car-floor points (X, Y)."""
    inv = np.linalg.inv(s_raster)
    mx = (inv[0, 0] * X + inv[0, 1] * Y + inv[0, 2]).astype(np.float32)
    my = (inv[1, 0] * X + inv[1, 1] * Y + inv[1, 2]).astype(np.float32)
    mx = np.nan_to_num(mx, nan=-1e4)
    my = np.nan_to_num(my, nan=-1e4)
    return cv2.remap(layer, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _patch_raster(shape: CarShape, centres, style: ShadowStyle, s_raster: np.ndarray, size) -> np.ndarray:
    rows, cols = size
    L = np.zeros((rows, cols), np.float32)
    inv = np.linalg.inv(s_raster)
    half_w = 0.5 * tyre_width(shape) + PATCH_WIDEN_M
    res = s_raster[0, 0]
    for X, Y in centres:
        c = inv @ np.array([X, Y, 1.0])
        cv2.ellipse(L, (int(round(c[0] * 16)), int(round(c[1] * 16))),
                    (int(round(PATCH_HALF_LEN_M / res * 16)), int(round(half_w / res * 16))), 0, 0, 360, 1.0, -1,
                    cv2.LINE_8, 4)
    if centres:
        L = cv2.GaussianBlur(L, (0, 0), max(0.5, style.tyre_sigma_m / res))
        L /= max(1e-6, float(L.max()))
    return L


def floor_points_of(bottoms, R, C, K) -> list[tuple[float, float]]:
    """The car-floor points (car metres) under the found tyre bottoms, through camera (R, C, K)."""
    out = []
    Ki = np.linalg.inv(K)
    for t in bottoms:
        if t.status != "found":
            continue
        d = R.T @ (Ki @ np.array([t.u, t.v, 1.0]))
        if d[2] < -1e-9:
            P = C + (-C[2] / d[2]) * d
            out.append((float(P[0]), float(P[1])))
    return out


def shadow_layers(shape: CarShape, style: ShadowStyle, car_floor_xy: np.ndarray, patch_xy, a_out: np.ndarray,
                  px_per_m: float, yaw: float, offsets_cols: np.ndarray | None, with_3d: bool = True) -> dict:
    """All layers (0..1) on the output grid. car_floor_xy: (h, w, 2) the car-frame floor point seen at each
    output pixel (NaN off the floor); a_out: the pasted car's alpha on the output grid."""
    h, w = a_out.shape
    table = SimpleNamespace(shape=shape)
    th = math.radians(yaw)
    toward = (style.shift_m * math.cos(th), -style.shift_m * math.sin(th))
    X, Y = car_floor_xy[..., 0], car_floor_xy[..., 1]
    L = {}
    if with_3d:
        core, s_core = footprint_layer(table, -style.core_inset_m, style.core_sigma_m, toward)
        skirt, s_skirt = footprint_layer(table, style.skirt_grow_m, style.skirt_sigma_m, toward)
        L["core"] = _raster_lookup(core, s_core, X, Y)
        L["skirt"] = _raster_lookup(skirt, s_skirt, X, Y)
        patches = _patch_raster(shape, patch_xy, style, s_core, core.shape)
        L["tyre"] = _raster_lookup(patches, s_core, X, Y)
        if offsets_cols is not None and np.any(offsets_cols != 0):
            mxs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            mys = np.arange(h, dtype=np.float32)[:, None] - offsets_cols[None, :].astype(np.float32)
            for k in ("core", "skirt"):
                L[k] = cv2.remap(L[k], mxs, mys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fl = outline_floor(a_out, (0, 0), px_per_m)
    L["contact"], L["gap"], L["band"] = silhouette_layers(fl, (0, 0, w, h), px_per_m)
    L["_floor"] = fl
    return L


def shade(lin: np.ndarray, layers: dict, style: ShadowStyle, floor_mask: np.ndarray, gain: float) -> np.ndarray:
    """Multiply the floor by the layers, in linear light (in place); returns the darkening (0..1)."""
    m = np.ones(floor_mask.shape, np.float32)
    for key, opacity in (("core", style.core), ("skirt", style.skirt), ("tyre", style.tyre), ("contact", style.contact),
                         ("gap", style.gap), ("band", style.band)):
        if key in layers:
            m *= 1 - np.clip(gain * opacity * layers[key], 0, 1)
    dark = (1.0 - m) * floor_mask
    mult = np.power(np.clip(1.0 - dark, 1e-4, 1.0)[..., None], np.array(style.tint_gamma_bgr, np.float32)[None, None])
    lin *= mult
    return dark


def pin_offsets_from(found_out: list[tuple[float, float, float, float]], cols: np.ndarray, px_per_m: float) -> np.ndarray:
    """found_out: (u_found, v_found, u_pred, v_pred) on the output grid."""
    cs = [Contact(u=u, v=v, u_pred=up, v_pred=vp, scale=1.0, visible=True, snapped=True) for u, v, up, vp in found_out]
    return contact_offsets(cs, cols, px_per_m)


# ---------------------------------------------------------------- reflection (glossy floors)
def schlick_ratio(delta_deg: float, ref_deg: float = 16.0, n: float = 1.5) -> float:
    """Fresnel reflectance at a view `delta` degrees above the floor, relative to the flat path's 16 deg."""
    r0 = ((n - 1) / (n + 1)) ** 2

    def R(d):
        return r0 + (1 - r0) * (1 - math.cos(math.radians(90 - d))) ** 5

    return R(max(0.5, delta_deg)) / R(ref_deg)


def upright_rotation(R: np.ndarray) -> np.ndarray:
    """Camera -> camera rotation to a view at the same heading with zero pitch and zero roll relative to the
    car's floor (R: car -> camera). In that view the car's verticals are parallel and upright."""
    g = -R[:, 2]  # the car's down in camera coordinates
    g = g / np.linalg.norm(g)
    z = np.array([0.0, 0.0, 1.0]) - g[2] * g  # the optical axis made horizontal
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, -1.0, 0.0]) - g[1] * g
    z /= np.linalg.norm(z)
    x = np.cross(g, z)
    return np.stack([x, g, z])


def room_reflection(lin: np.ndarray, car_lin: np.ndarray, car_alpha: np.ndarray, top_left, px_per_m: float,
                    style: ReflectionStyle, gloss: np.ndarray, delta_deg: float, K: np.ndarray, R: np.ndarray) -> bool:
    """A faint mirror image of the car on a glossy floor (added to `lin` in place; the car is pasted later).
    Mirrored about the car's own floor line along the true vertical: the car's region (and the reflection's
    reach below it) is re-projected into an upright view of the same camera (no pitch, no roll), mirrored per
    column there, and warped back. Returns False when the region cannot be drawn upright (then nothing is
    added)."""
    h, w = lin.shape[:2]
    tx, ty = top_left
    ch, cw = car_alpha.shape
    ys, xs = np.nonzero(car_alpha >= 0.5)
    if len(xs) < 20:
        return False
    bx0, bx1 = tx + xs.min(), tx + xs.max() + 1
    by0, by1 = ty + ys.min(), ty + ys.max() + 1
    bw, bh = bx1 - bx0, by1 - by0
    rx0, rx1 = max(0, int(bx0 - 0.3 * bw)), min(w, int(bx1 + 0.3 * bw))
    ry0, ry1 = max(0, int(by0 - 0.05 * bh)), min(h, int(by1 + 1.2 * bh))
    if rx1 <= rx0 or ry1 <= ry0:
        return False
    opacity = style.opacity * min(1.0, schlick_ratio(delta_deg))
    st = ReflectionStyle(opacity=opacity, fade_m=style.fade_m * max(0.3, math.cos(math.radians(delta_deg))),
                         compress=style.compress, blur_px=style.blur_px, env_linear=style.env_linear,
                         outside_factor=style.outside_factor)
    Hm = K @ upright_rotation(R) @ np.linalg.inv(K)
    corners = np.array([[rx0, ry0, 1], [rx1, ry0, 1], [rx1, ry1, 1], [rx0, ry1, 1]], float).T
    q = Hm @ corners
    if (q[2] <= 1e-6).any():
        return False
    q = q[:2] / q[2]
    x0, y0 = np.floor(q.min(1)) - 2
    x1, y1 = np.ceil(q.max(1)) + 2
    if not np.isfinite([x0, y0, x1, y1]).all():
        return False
    area = (x1 - x0) * (y1 - y0)
    cap = min(6.0 * (rx1 - rx0) * (ry1 - ry0), 4e6)
    k = 1.0 if area <= cap else math.sqrt(cap / area)  # a steep view stretches the upright copy: draw it smaller
    T = np.diag([k, k, 1.0]) @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ Hm
    size = (max(1, int((x1 - x0) * k)), max(1, int((y1 - y0) * k)))
    px_per_m = px_per_m * k
    canvas_c = np.zeros((h, w, 3), np.float32)
    canvas_a = np.zeros((h, w), np.float32)
    ys0, xs0 = max(0, ty), max(0, tx)
    ys1, xs1 = min(h, ty + ch), min(w, tx + cw)
    canvas_c[ys0:ys1, xs0:xs1] = car_lin[ys0 - ty:ys1 - ty, xs0 - tx:xs1 - tx]
    canvas_a[ys0:ys1, xs0:xs1] = car_alpha[ys0 - ty:ys1 - ty, xs0 - tx:xs1 - tx]
    rc = cv2.warpPerspective(canvas_c, T, size, flags=cv2.INTER_LINEAR)
    ra = cv2.warpPerspective(canvas_a, T, size, flags=cv2.INTER_LINEAR)
    rfl = outline_floor(ra, (0, 0), px_per_m)
    if rfl is None:
        return False
    base = np.full((size[1], size[0], 3), 0.5, np.float32)
    add_reflection(base, (0, 0, size[0], size[1]), rc, ra, SimpleNamespace(top_left=(0, 0), px_per_m=px_per_m),
                   rfl, st, np.ones((size[1], size[0]), np.float32), None, math.radians(delta_deg))
    delta = cv2.warpPerspective(base - 0.5, np.linalg.inv(T), (w, h), flags=cv2.INTER_LINEAR)
    lin += delta * gloss[..., None]
    np.clip(lin, 0.0, 1.0, out=lin)
    return True


__all__ = ["TYRE_NAMES", "TyreBottom", "contact_lock", "find_tyre_bottoms", "floor_points_of", "pin_offsets_from",
           "project_rc", "room_reflection", "rx", "schlick_ratio", "shade", "shadow_layers"]
