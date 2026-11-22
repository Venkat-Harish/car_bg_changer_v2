"""Shadows and reflections under the car. Background pixels only, in linear light; the car is pasted
afterwards, so its own pixels never change.

Floor geometry. Under the car the floor follows the capture camera V, so the shadow touches the pasted
tyres. In front of the car's lowest row, depth follows the background's own camera (the "plate"),
which foreshortens the floor more, so the shadow stays compact ("pinned floor map").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .carmodel import CarTable, REF_TYRE_W, REF_WIDTH
from .placement import Plan
from .probe import CarProbe

# ---------------------------------------------------------------- colour

_LIN_LUT = np.array([((i / 255) / 12.92) if i / 255 <= 0.04045 else (((i / 255) + 0.055) / 1.055) ** 2.4
                     for i in range(256)], dtype=np.float32)


def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    return _LIN_LUT[img_u8]


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- styles

@dataclass(frozen=True)
class ShadowStyle:
    """Opacities 0-1; sizes in metres. Layers multiply: floor * prod(1 - opacity * layer)."""

    tyre: float = 0.85  # patches under matched tyres
    tyre_sigma_m: float = 0.035
    core: float = 0.55  # under the footprint (seen through the gap under the body)
    core_inset_m: float = 0.10
    core_sigma_m: float = 0.20
    skirt: float = 0.25  # soft ambient shadow around the footprint
    skirt_grow_m: float = 0.30
    skirt_sigma_m: float = 0.25
    contact: float = 0.80  # dark line right under the car's own outline, where it touches the floor
    gap: float = 0.60  # the floor seen under the body, between its underside and the floor line
    band: float = 0.35  # soft band under the whole outline
    shift_m: float = 0.0  # move core and skirt toward the camera
    tint_gamma_bgr: tuple[float, float, float] = (1.0, 1.0, 1.0)  # shadow colour: M_c = M ** gamma_c


@dataclass(frozen=True)
class ReflectionStyle:
    """Faint mirror image of the car on a glossy floor (added to the floor, never to the car)."""

    opacity: float = 0.18  # at the mirror line
    fade_m: float = 0.6  # e-fold distance in mirrored height
    compress: float = 0.90  # vertical compression of the mirror image
    blur_px: tuple[float, float, float] = (3.0, 10.0, 25.0)  # sharp near the line -> soft higher up
    env_linear: float = 0.60  # what the floor reflects without the car (bright back wall)
    outside_factor: float = 0.6  # strength outside the glossy disc


# ---------------------------------------------------------------- plate floors

class PlateFloor:
    """The background camera's floor along the anchor's line of sight: depth s (metres beyond the
    anchor, away from the camera) <-> image row."""

    def row(self, s):
        raise NotImplementedError

    def depth(self, v):
        raise NotImplementedError


class LevelFloor(PlateFloor):
    """Level pinhole camera: row = horizon + k / (z0 + s)."""

    def __init__(self, horizon: float, k: float, z0: float):
        self.horizon, self.k, self.z0 = horizon, k, z0

    @classmethod
    def from_anchor(cls, anchor_y: float, horizon: float, z0: float) -> "LevelFloor":
        return cls(horizon, (anchor_y - horizon) * z0, z0)

    def row(self, s):
        return self.horizon + self.k / (self.z0 + np.asarray(s, dtype=np.float64))

    def depth(self, v):
        return self.k / np.maximum(np.asarray(v, dtype=np.float64) - self.horizon, 1e-3) - self.z0


class HomographyFloor(PlateFloor):
    """Floor homography H: (lateral X, forward Z, 1) metres -> image. The line of sight is +Z
    through the anchor's floor point."""

    def __init__(self, h: np.ndarray, anchor_px: tuple[float, float]):
        self.h = np.asarray(h, dtype=np.float64)
        p = np.linalg.solve(self.h, np.array([anchor_px[0], anchor_px[1], 1.0]))
        self.xa, self.za = p[0] / p[2], p[1] / p[2]
        s = np.concatenate([np.linspace(-0.9 * self.za, 0, 400, endpoint=False), np.geomspace(1e-3, 500, 1200)])
        s = np.unique(np.concatenate([s, [0.0]]))
        self._s, self._v = s, self.row(s)

    def row(self, s):
        s = np.asarray(s, dtype=np.float64)
        q = self.h @ np.stack([np.full_like(s, self.xa), self.za + s, np.ones_like(s)]).reshape(3, -1)
        return (q[1] / q[2]).reshape(s.shape)

    def depth(self, v):
        # rows fall as s grows: interpolate on the reversed arrays
        return np.interp(np.asarray(v, dtype=np.float64), self._v[::-1], self._s[::-1])


class CompressionFloor(PlateFloor):
    """No plate calibration: in front of the car, depth is compressed by a constant factor vs V."""

    def __init__(self, factor: float, table: CarTable, anchor_y: float, px_per_m: float):
        self.factor, self.table, self.anchor_y, self.p = factor, table, anchor_y, px_per_m

    def row(self, s):
        return self.anchor_y + self.factor * (self.table.floor_row(s, self.anchor_y, self.p) - self.anchor_y)

    def depth(self, v):
        v_v = self.anchor_y + (np.asarray(v, dtype=np.float64) - self.anchor_y) / self.factor
        return self.table.floor_depth(v_v, self.anchor_y, self.p)


class PinnedFloorMap:
    """Output row -> row of a layer drawn through V: identity down to the car's lowest row, then
    stepped with the plate's foreshortening."""

    def __init__(self, table: CarTable, plan: Plan, plate_floor: PlateFloor):
        self.table, self.plan, self.plate = table, plan, plate_floor
        self.v_att = float(plan.box_out[3])
        ay, p = plan.anchor[1], plan.px_per_m
        self.s_att = float(table.floor_depth(self.v_att, ay, p))
        self.base = float(plate_floor.row(self.s_att))

    def source_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        out = rows.copy()
        below = rows > self.v_att
        if below.any():
            s_star = self.plate.depth(self.base + (rows[below] - self.v_att))
            s_star = np.maximum(s_star, -0.95 * self.table.distance)
            out[below] = self.table.floor_row(s_star, self.plan.anchor[1], self.plan.px_per_m)
        return out

    def output_rows(self, v_rows: np.ndarray) -> np.ndarray:
        """Inverse of source_rows (for drawing V-projected outlines on the result)."""
        grid = np.arange(int(self.v_att) - 1, int(self.v_att) + 4000, dtype=np.float64)
        src = self.source_rows(grid)
        v_rows = np.asarray(v_rows, dtype=np.float64)
        return np.where(v_rows > self.v_att, np.interp(v_rows, src, grid), v_rows)


# ---------------------------------------------------------------- tyre contacts

@dataclass(frozen=True)
class Contact:
    u: float  # where the tyre touches the floor on the background (snapped when matched)
    v: float
    u_pred: float  # where V predicts it
    v_pred: float
    scale: float  # local px per metre relative to px_per_m
    visible: bool
    snapped: bool


SNAP_X_M = 0.45  # match window around the predicted contact, metres (sideways)
SNAP_Y_M = 0.45  # and vertically (as image px at the anchor scale)
TOUCH_M = 0.06  # a point of the outline this close to the floor line touches the floor
BODY_DROP_M = 1.3  # outline columns more than this above the car's lowest point are not underbody (mirrors)
GAP_MAX_M = 0.45  # the dark gap is only drawn under parts at most this high above the floor line
OFFSET_TAPER_M = 0.8  # a matched tyre's vertical correction fades out over about this distance


@dataclass
class OutlineFloor:
    """The pasted car's own bottom edge, and the floor line under it: the lower envelope (hull)
    through its lowest points. Built from the real silhouette, so it passes under every tyre that
    touches the floor, whatever the angle estimate says."""

    u: np.ndarray  # background columns that hold car pixels
    b: np.ndarray  # bottom edge row of the car in each of them
    body: np.ndarray  # column belongs to the underbody (not a mirror or speck far above the floor)
    line: np.ndarray  # floor line under each column

    @property
    def height(self) -> np.ndarray:
        return np.maximum(0.0, self.line - self.b)

    def line_at(self, cols) -> np.ndarray:
        """Floor line at any columns (flat beyond the car's ends)."""
        return np.interp(np.asarray(cols, dtype=np.float64), self.u[self.body], self.line[self.body])


def _lower_envelope(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Floor line under an outline: the hull below its points (largest rows) between the outermost
    points where it touches (hull corners inside the outline, i.e. tyres or the lowest bumper part),
    kept level with them beyond (a hull would otherwise climb to the bumper corners at the ends)."""
    hull: list[tuple[float, float]] = []
    for x, y in zip(u.tolist(), (-b).tolist()):  # lower hull of (u, -b) = envelope below the car
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    hu = np.array([h[0] for h in hull])
    hv = -np.array([h[1] for h in hull])
    span = u[-1] - u[0]
    inner = (hu > u[0] + 0.03 * span) & (hu < u[-1] - 0.03 * span)
    if inner.any():  # the touching points; np.interp keeps the line level beyond the outer ones
        return np.interp(u, hu[inner], hv[inner])
    return np.interp(u, hu, hv)


def outline_floor(car_alpha: np.ndarray, top_left: tuple[int, int], px_per_m: float) -> OutlineFloor | None:
    x0, y0 = top_left
    nh, nw = car_alpha.shape
    solid = car_alpha >= 0.5
    has = solid.any(axis=0)
    if has.sum() < 2:
        return None
    cols = np.flatnonzero(has)
    bottom = (nh - 1) - np.argmax(solid[::-1, cols], axis=0)
    u = (x0 + cols).astype(np.float64)
    b = (y0 + bottom + 1).astype(np.float64)
    body = b >= b.max() - BODY_DROP_M * px_per_m
    line = np.full_like(b, np.nan)
    line[body] = _lower_envelope(u[body], b[body])
    line[~body] = np.interp(u[~body], u[body], line[body])
    return OutlineFloor(u=u, b=b, body=body, line=np.maximum(line, b))


def snap_contacts(plan: Plan, probe: CarProbe, floor: OutlineFloor | None = None) -> list[Contact]:
    """Project the 4 tyre contacts through V (mapped onto the pasted car). In side and 3/4 views,
    match each near-side tyre to the pasted car's tyre there: a local lowest point of its outline
    that touches the floor line, away from the bumper ends. End views are left alone: their tyres
    hide behind the bumpers."""
    p, (ax, ay) = plan.px_per_m, plan.anchor
    row = plan.view
    s = math.sin(math.radians(plan.yaw.deg))
    side_ish = abs(s) >= 0.5
    cands: list[tuple[float, float]] = []
    if side_ish:
        x_lo, x_hi = plan.box_out[0], plan.box_out[2]
        margin = 0.06 * (x_hi - x_lo)
        for x, y in probe.contacts:
            cu, cv = (float(v) for v in plan.to_out(x, y + 1))
            if not (x_lo + margin <= cu <= x_hi - margin):
                continue  # bumper corners at the car's ends
            if floor is not None and float(floor.line_at(cu)) - cv > TOUCH_M * p:
                continue  # above the floor line: an apron or bumper edge, not a tyre
            cands.append((cu, cv))
    near_side = [s < -0.17, s > 0.17, s < -0.17, s > 0.17]  # FL, FR, RL, RR: their side faces the camera
    wx, wy = SNAP_X_M * p, SNAP_Y_M * p
    out, used = [], set()
    for i in range(4):
        u0 = float(plan.xmap.fwd(ax + p * row.contacts_du[i]))
        v0 = float(ay + p * row.contacts_dv[i])
        visible = bool(row.contacts_visible[i])
        u, v, snapped = u0, v0, False
        if visible and near_side[i] and cands:
            scored = [((c[0] - u0) / wx) ** 2 + ((c[1] - v0) / wy) ** 2 for c in cands]
            j = int(np.argmin(scored))
            if scored[j] <= 1.0 and j not in used:
                (u, v), snapped = cands[j], True
                used.add(j)
        out.append(Contact(u=u, v=v, u_pred=u0, v_pred=v0, scale=float(row.contacts_scale[i]),
                           visible=visible, snapped=snapped))
    return out


def contact_offsets(contacts: list[Contact], cols: np.ndarray, px_per_m: float) -> np.ndarray:
    """Per-column vertical shift that moves V's floor drawing onto the matched tyres: interpolated
    between them, fading to 0 beyond them (a lone tyre says nothing about the far end of the car)."""
    pts = sorted((c.u, c.v - c.v_pred) for c in contacts if c.snapped)
    cols = np.asarray(cols, dtype=np.float64)
    if not pts:
        return np.zeros_like(cols)
    us, ds = np.array([q[0] for q in pts]), np.array([q[1] for q in pts])
    out = np.interp(cols, us, ds)
    taper = OFFSET_TAPER_M * px_per_m
    out = np.where(cols < us[0], ds[0] * np.exp(-0.5 * ((cols - us[0]) / taper) ** 2), out)
    return np.where(cols > us[-1], ds[-1] * np.exp(-0.5 * ((cols - us[-1]) / taper) ** 2), out)


# ---------------------------------------------------------------- shadow

RASTER_M = 0.02  # top-down raster resolution, metres per px


def _footprint_layer(table: CarTable, grow: float, sigma_m: float, shift: tuple[float, float]):
    shape = table.shape
    x_min, x_max = shape.rear_tip - 1.2, shape.front_tip + 1.2
    y_max = shape.width / 2 + 1.2
    cols = int(math.ceil((x_max - x_min) / RASTER_M))
    rows = int(math.ceil(2 * y_max / RASTER_M))
    poly = shape.footprint(grow=grow) + np.array(shift)
    px = np.stack([(poly[:, 0] - x_min) / RASTER_M, (y_max - poly[:, 1]) / RASTER_M], axis=1)
    layer = np.zeros((rows, cols), np.float32)
    cv2.fillPoly(layer, [np.round(px * 16).astype(np.int32)], 1.0, lineType=cv2.LINE_8, shift=4)
    if sigma_m > 0:
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma_m / RASTER_M)
    s_raster = np.array([[RASTER_M, 0.0, x_min], [0.0, -RASTER_M, y_max], [0.0, 0.0, 1.0]])
    return layer, s_raster


def _warp_floor_layer(layer, s_raster, plan: Plan, table: CarTable, roi, floor_map: PinnedFloorMap,
                      shift_cols: np.ndarray):
    rx1, ry1, rx2, ry2 = roi
    w, hgt = rx2 - rx1, ry2 - ry1
    # Draw through V (in V's own columns, with room above and below); then pull it onto the pasted
    # car: columns through plan.xmap, rows shifted onto the matched tyres, and rows below the car
    # through the pinned floor map.
    src_cols = plan.xmap.inv(np.arange(rx1, rx2, dtype=np.float64))
    vx1 = int(math.floor(src_cols.min())) - 2
    vx2 = int(math.ceil(src_cols.max())) + 3
    top_pad = int(max(0.0, float(shift_cols.max()))) + 2
    h_floor = table.floor_homography(plan.yaw.deg, plan.anchor, plan.px_per_m)
    to_layer = np.array([[1.0, 0.0, -vx1], [0.0, 1.0, -(ry1 - top_pad)], [0.0, 0.0, 1.0]])
    extra = hgt * 2
    v_layer = cv2.warpPerspective(layer, to_layer @ h_floor @ s_raster, (vx2 - vx1, top_pad + hgt + extra),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    rows = np.arange(ry1, ry2, dtype=np.float64)[:, None] - shift_cols[None, :]
    src_rows = floor_map.source_rows(rows.ravel()).reshape(rows.shape) - (ry1 - top_pad)
    map_y = src_rows.astype(np.float32)
    map_x = np.repeat((src_cols - vx1).astype(np.float32)[None, :], hgt, axis=0)
    return cv2.remap(v_layer, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _silhouette_layers(floor: OutlineFloor | None, roi, px_per_m: float):
    """Layers drawn from the pasted car's own bottom outline and the floor line under it, so they
    follow the real car whatever the angle estimate says:
      contact - a dark line right under the outline where it touches the floor (the tyres);
      gap     - the floor seen between the underbody and the floor line (dark under sills and bumpers);
      band    - a soft shadow just under the outline, fading for parts high above the floor.
    Mirrors and other parts far above the floor get none of these."""
    rx1, ry1, rx2, ry2 = roi
    shape = (ry2 - ry1, rx2 - rx1)
    band, contact, gap = (np.zeros(shape, np.float32) for _ in range(3))
    if floor is None:
        return contact, gap, band
    keep = floor.body & (floor.u >= rx1) & (floor.u < rx2)
    if not keep.any():
        return contact, gap, band
    c_out = (floor.u[keep] - rx1).astype(int)
    b_out, line, height = floor.b[keep], floor.line[keep], floor.height[keep]
    thick = max(2.0, 0.06 * px_per_m)
    rows = np.arange(ry1, ry2, dtype=np.float64)[:, None]
    band[:, c_out] = ((rows >= b_out[None, :] - thick) & (rows <= b_out[None, :] + 2.5 * thick)).astype(np.float32)
    contact[:, c_out] = ((rows >= b_out[None, :] - 2) & (rows <= b_out[None, :] + 0.5 * thick)).astype(np.float32)
    low_enough = (height > 2) & (height <= GAP_MAX_M * px_per_m)
    gap[:, c_out] = ((rows >= b_out[None, :]) & (rows <= line[None, :] + 0.3 * thick)
                     & low_enough[None, :]).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=2.0 * thick, sigmaY=1.2 * thick)
    contact = cv2.GaussianBlur(contact, (0, 0), sigmaX=0.6 * thick, sigmaY=0.35 * thick)
    gap = cv2.GaussianBlur(gap, (0, 0), sigmaX=1.2 * thick, sigmaY=0.6 * thick)

    def column_weight(values):
        wcol = np.zeros(shape[1], np.float32)
        wcol[c_out] = values
        return cv2.GaussianBlur(wcol[None, :], (0, 0), sigmaX=thick).ravel()[None, :]

    touching = column_weight(np.exp(-height / (0.10 * px_per_m)))
    near = column_weight(np.exp(-height / (0.30 * px_per_m)))
    norm = lambda a: np.clip(a / max(1e-6, float(a.max())), 0, 1)  # noqa: E731
    return norm(contact) * touching, np.clip(gap, 0, 1), norm(band) * near


def _tyre_layer(contacts: list[Contact], plan: Plan, table: CarTable, style: ShadowStyle, roi):
    """Contact patches, only under tyres that were matched on the pasted car."""
    rx1, ry1, rx2, ry2 = roi
    layer = np.zeros((ry2 - ry1, rx2 - rx1), np.float32)
    tyre_w = REF_TYRE_W * table.shape.width / REF_WIDTH
    sig = []
    for c in contacts:
        if not c.snapped:
            continue
        lam = plan.px_per_m * c.scale
        half_w = max(1.0, 0.5 * tyre_w * lam)
        half_h = max(1.0, 0.5 * 0.16 * lam * 0.3)
        centre = (int(round((c.u - rx1) * 16)), int(round((c.v - ry1) * 16)))
        axes = (int(round(half_w * 16)), int(round(half_h * 16)))
        cv2.ellipse(layer, centre, axes, 0, 0, 360, 1.0, -1, cv2.LINE_8, shift=4)
        sig.append(style.tyre_sigma_m * lam)
    if sig:
        s = float(np.mean(sig))
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(1.0, s), sigmaY=max(1.0, 0.4 * s))
        layer /= max(1e-6, float(layer.max()))
    return layer


def shadow_multiplier(plan: Plan, table: CarTable, style: ShadowStyle, floor_map: PinnedFloorMap,
                      contacts: list[Contact], mask_roi: np.ndarray, roi, floor: OutlineFloor | None,
                      gain: float = 1.0) -> np.ndarray:
    """(roi_h, roi_w, 3) float32 multiplier for the floor in linear light; exactly 1 where no shadow."""
    rx1, ry1, rx2, ry2 = roi
    th = math.radians(plan.yaw.deg)
    toward_cam = (style.shift_m * math.cos(th), -style.shift_m * math.sin(th))
    shift = contact_offsets(contacts, np.arange(rx1, rx2, dtype=np.float64), plan.px_per_m)
    core, s_core = _footprint_layer(table, -style.core_inset_m, style.core_sigma_m, toward_cam)
    skirt, s_skirt = _footprint_layer(table, style.skirt_grow_m, style.skirt_sigma_m, toward_cam)
    m_core = _warp_floor_layer(core, s_core, plan, table, roi, floor_map, shift)
    m_skirt = _warp_floor_layer(skirt, s_skirt, plan, table, roi, floor_map, shift)
    m_tyre = _tyre_layer(contacts, plan, table, style, roi)
    m_contact, m_gap, m_band = _silhouette_layers(floor, roi, plan.px_per_m)
    g = float(gain)
    m = np.ones((ry2 - ry1, rx2 - rx1), np.float32)
    for opacity, layer in ((style.core, m_core), (style.skirt, m_skirt), (style.tyre, m_tyre),
                           (style.contact, m_contact), (style.gap, m_gap), (style.band, m_band)):
        m *= 1 - np.clip(g * opacity * layer, 0, 1)
    dark = 1.0 - m
    # Cut frames: fade out past the slice, where the missing part of the car would be.
    if plan.cut:
        x = np.arange(rx1, rx2, dtype=np.float32)
        ramp = np.ones_like(x)
        fade = max(1.0, 0.25 * plan.px_per_m)
        if "right" in plan.cut:
            ramp = np.minimum(ramp, np.clip(1 - (x - plan.box_out[2]) / fade, 0, 1))
        if "left" in plan.cut:
            ramp = np.minimum(ramp, np.clip(1 - (plan.box_out[0] - x) / fade, 0, 1))
        dark *= ramp[None, :]
    dark *= mask_roi
    m = 1.0 - dark
    gamma = np.array(style.tint_gamma_bgr, dtype=np.float32)
    return np.power(np.clip(m, 0, 1)[..., None], gamma[None, None, :]).astype(np.float32)


# ---------------------------------------------------------------- reflection

def add_reflection(lin_roi: np.ndarray, roi, car_lin: np.ndarray, car_alpha: np.ndarray, plan: Plan,
                   floor: OutlineFloor | None, style: ReflectionStyle, mask_roi: np.ndarray,
                   inside_roi: np.ndarray | None, elevation_rad: float) -> None:
    """Add a faint, blurred mirror image of the car about the floor line under each column (in place).
    Computed over a strip wider than the car and blurred with a zero border, so it fades out
    smoothly at the car's ends instead of stopping in a seam."""
    if floor is None:
        return
    rx1, ry1, rx2, ry2 = roi
    x0, y0 = plan.top_left
    nh, nw = car_alpha.shape
    p = plan.px_per_m
    pad = int(math.ceil(3 * max(style.blur_px)))
    left, right = int(max(rx1, x0 - pad)), int(min(rx2, x0 + nw + pad))
    if right <= left:
        return
    cols = np.arange(left, right, dtype=np.float64)
    axis = floor.line_at(cols)
    reach = int(math.ceil(3.0 * style.fade_m * style.compress * p * math.cos(elevation_rad)))
    top = int(max(ry1, math.floor(axis.min())))
    bot = int(min(ry2, math.ceil(axis.max()) + reach))
    if bot <= top:
        return
    rows = np.arange(top, bot, dtype=np.float64)[:, None]
    ax_c = axis[None, :]
    below = rows > ax_c
    src_y = ax_c - (rows - ax_c) / style.compress - y0
    map_x = np.repeat((cols - x0).astype(np.float32)[None, :], len(rows), axis=0)
    map_y = src_y.astype(np.float32)
    prem = np.dstack([car_lin * car_alpha[..., None], car_alpha]).astype(np.float32)
    mirrored = cv2.remap(prem, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mirrored[~below] = 0
    h_m = np.where(below, (rows - ax_c) / (style.compress * p * math.cos(elevation_rad)), 0.0)
    b = np.clip(h_m / (2 * style.fade_m), 0, 1)[..., None]
    blurs = [cv2.GaussianBlur(mirrored, (0, 0), sigmaX=s, borderType=cv2.BORDER_CONSTANT) for s in style.blur_px]
    layer = np.where(b < 0.5, blurs[0] * (1 - 2 * b) + blurs[1] * (2 * b), blurs[1] * (2 - 2 * b) + blurs[2] * (2 * b - 1))
    fade = style.opacity * np.exp(-h_m / style.fade_m) * below
    delta = fade[..., None] * (layer[..., :3] - layer[..., 3:4] * style.env_linear)
    sl = (slice(top - ry1, bot - ry1), slice(int(left) - rx1, int(right) - rx1))
    weight = mask_roi[sl]
    if inside_roi is not None:
        weight = weight * (inside_roi[sl] + (1 - inside_roi[sl]) * style.outside_factor)
    lin_roi[sl] = np.clip(lin_roi[sl] + delta * weight[..., None], 0.0, 1.0)


# Public names for the room path (car_fit_to_bg), which draws the same layers through its own camera.
footprint_layer = _footprint_layer
silhouette_layers = _silhouette_layers
