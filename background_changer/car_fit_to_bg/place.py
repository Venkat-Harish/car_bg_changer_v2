"""Per background: put the photo's levelled camera into the background's 3D room, render the room, shade
its floor and paste the car last.

The virtual camera keeps its pose relative to the car (from the fit, levelled, contact-locked). The car's
footprint centre stands on the scene's stage point and the camera sits on the plate camera's side of it,
so the background is seen from the front at the photo's own height and angle. The room is scaled by the
stage rule: a side view of any car spans the scene's side_px at the stage, as on the flat path.

Output (decision 6): the photo's own frame on the levelled grid, scaled by s_out = min(1, max_output_size /
long side, s_tex), s_tex keeping >= 1 plate texel per output pixel on the floor at the stage (the background
is never magnified there). A cut side's border sits on the innermost point of the slanted cut edge and the
window keeps its size by sliding toward the uncut side; an uncut car is never cropped (the window grows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from bg_changer.compose import paste
from bg_changer.scenes.base import SceneSpec
from room3d import CEIL, FLOOR, WALL, draw_room, linear_to_srgb, plane_h, render, srgb_to_linear, texels_per_px

from .ground import (pin_offsets_from, project_rc, room_reflection, schlick_ratio, shade, shadow_layers)
from .level import downscale
from .rooms import DEFAULT_ROOM_SHADOW, ROOM_SHADOW, build_room

MIN_FLOOR_TEXELS = 1.0
CAMERA_MARGIN = 0.3  # room units: the virtual camera keeps this far inside the room
DISC_RADIUS = 3.0  # background_1's glossy turntable, in its room units (6.0 m disc)
CONTACT_MISMATCH = 0.01  # share of the output height: lock residual above this is flagged


def rz2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Placement:
    image: np.ndarray | None  # BGR uint8, or None when skipped
    size: tuple[int, int] = (0, 0)
    s_out: float = 1.0
    window: tuple[int, int] = (0, 0)  # origin on the scaled levelled grid
    metrics: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    debug: np.ndarray | None = None
    note: str = ""
    steps: dict = field(default_factory=dict)  # debug only: the step pictures, {name: BGR uint8}


def stage_scale(spec: SceneSpec, side_length: float, room, S) -> float:
    """Room units per car metre: a side view of the car spans side_px x size at the stage."""
    return (spec.side_px * spec.size / side_length) / room.px_per_unit_at(S)


def room_camera(fit, room, S, sigma):
    """The levelled, locked photo camera placed in the room. Returns (R_room, C_room, A) with A the car ->
    room rotation about the vertical (its heading puts the camera on the plate camera's side)."""
    C_car = fit.C_lock
    u_plate = room.C[:2] - S
    psi = math.degrees(math.atan2(u_plate[1], u_plate[0]) - math.atan2(C_car[1], C_car[0]))
    A = rz2(psi)
    R_room = fit.R_lock @ A.T
    C_room = np.array([S[0], S[1], 0.0]) + sigma * (A @ C_car)
    return R_room, C_room, A


def output_window(fit, s: float, W: int, H: int) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) on the scaled levelled grid: the photo frame, cut borders on the slanted cut edges,
    uncut car never cropped."""
    Wo, Ho = max(1, int(round(W * s))), max(1, int(round(H * s)))
    x0, y0 = 0, 0
    cuts = fit.cut_edges
    if "right" in cuts:
        right = int(math.floor((cuts["right"]["lo"] + 0.5) * s))
        x0 = right - Wo
    elif "left" in cuts:
        x0 = int(math.ceil((cuts["left"]["hi"] + 0.5) * s))
    if "top" in cuts:
        y0 = int(math.ceil((cuts["top"]["hi"] + 0.5) * s))
    # keep every uncut side of the car inside
    ys, xs = np.nonzero(fit.car.alpha > 0)
    bx0, by0 = (fit.car.origin[0] + xs.min()) * s, (fit.car.origin[1] + ys.min()) * s
    bx1, by1 = (fit.car.origin[0] + xs.max() + 1) * s, (fit.car.origin[1] + ys.max() + 1) * s
    x1, y1 = x0 + Wo, y0 + Ho
    if "left" not in cuts and bx0 < x0:
        x0 = int(math.floor(bx0))
    if "right" not in cuts and bx1 > x1:
        x1 = int(math.ceil(bx1))
    if "top" not in cuts and by0 < y0:
        y0 = int(math.floor(by0))
    if "bottom" not in cuts and by1 > y1:
        y1 = int(math.ceil(by1))
    return x0, y0, x1 - x0, y1 - y0


def place_in_room(fit, name: str, plate: np.ndarray, spec: SceneSpec, shadow_gain: float = 1.0,
                  max_output_size: int = 1920, debug: bool = False,
                  min_floor_texels: float = MIN_FLOOR_TEXELS, room=None, generic: bool = False) -> Placement:
    """Render one background for one fitted photo. spec: the background's scene spec WITH the config's
    [scene] overrides and WITHOUT the flat path's auto-fit shrink. room: the background's room (default: its
    own module's); generic: it is a generic room built from horizon_y (flagged)."""
    room = room if room is not None else build_room(name)
    if room is None:
        return Placement(None, note="no room model for this background", flags=["no-room"])
    flags: list[str] = ["generic-room"] if generic else []
    S = room.floor_point(spec.anchor)
    sigma = stage_scale(spec, fit.side_length, room, S)
    R_room, C_room, A = room_camera(fit, room, S, sigma)
    if not room.contains(C_room, CAMERA_MARGIN):
        C_room = room.clamp_inside(C_room, CAMERA_MARGIN, toward=S)
        flags.append("camera-outside-room")
    W, H = fit.frame_size
    f, (cx, cy) = fit.f, fit.c
    K1 = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    q = plane_h(K1, R_room, C_room) @ np.array([S[0], S[1], 1.0])  # where the stage appears (texels measured there)
    t1 = texels_per_px(room, K1, R_room, C_room, q[:2] / q[2] if abs(q[2]) > 1e-9 else None)[2]
    s_tex = t1 / min_floor_texels if min_floor_texels > 0 else 1.0
    s = min(1.0, max_output_size / max(W, H) if max_output_size else 1.0, s_tex)
    if t1 / s < MIN_FLOOR_TEXELS:
        flags.append("soft-floor")
    x0, y0, Wo, Ho = output_window(fit, s, W, H)
    Ko = np.array([[f * s, 0.0, (cx + 0.5) * s - 0.5 - x0], [0.0, f * s, (cy + 0.5) * s - 0.5 - y0], [0.0, 0.0, 1.0]])
    rr = render(plate, room, Ko, R_room, C_room, (Wo, Ho))
    lin = rr.lin
    plain = linear_to_srgb(lin) if debug else None  # the room before shadows, reflection and the car
    # The car on the output grid.
    pixels, a_s, corner = downscale(fit.car, s)
    px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
    a_out = np.zeros((Ho, Wo), np.float32)
    hh, ww = a_s.shape
    ya, xa = max(0, py), max(0, px)
    yb, xb = min(Ho, py + hh), min(Wo, px + ww)
    if yb > ya and xb > xa:
        a_out[ya:yb, xa:xb] = a_s[ya - py:yb - py, xa - px:xb - px]
    # Floor points (room units) -> car floor (metres).
    fxy = rr.floor_xy
    rel = (fxy - np.array([S[0], S[1]], np.float32)) / sigma
    Ai = A[:2, :2].T
    car_xy = np.stack([Ai[0, 0] * rel[..., 0] + Ai[0, 1] * rel[..., 1], Ai[1, 0] * rel[..., 0] + Ai[1, 1] * rel[..., 1]], -1)
    # Found tyre bottoms and the locked camera's contacts on the output grid.
    con = fit.shape.contacts()
    uc, vc, _ = project_rc(con, fit.R_lock, fit.C_lock, Ko)
    found_out = []
    for i, tb in enumerate(fit.bottoms):
        if tb.status == "found":
            found_out.append(((tb.u + 0.5) * s - 0.5 - x0, (tb.v + 0.5) * s - 0.5 - y0, float(uc[i]), float(vc[i])))
    ppm = f * s / fit.pose.D
    offsets = pin_offsets_from(found_out, np.arange(Wo, dtype=np.float64), ppm) if found_out else None
    patch_xy = fit.patch_xy
    style = ROOM_SHADOW.get(name) or DEFAULT_ROOM_SHADOW
    layers = shadow_layers(fit.shape, style, car_xy, patch_xy, a_out, ppm, fit.pose.yaw, offsets,
                           with_3d=not fit.pose.failed)
    floor = cv2.GaussianBlur((rr.kinds == FLOOR).astype(np.float32), (0, 0), 1.0)
    gain = shadow_gain * spec.shadow_gain
    if gain > 0:
        shade(lin, layers, style, floor, gain)
    if spec.reflection is not None:
        if room.name == "background_1":  # its glossy turntable; the floor around it is less glossy
            dist = np.hypot(fxy[..., 0], fxy[..., 1])
            inside = np.where(np.isfinite(dist), (dist <= DISC_RADIUS).astype(np.float32), 0.0)
            inside = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 2.0)
        else:
            inside = np.ones(floor.shape, np.float32)
        gloss = floor * (inside + (1 - inside) * spec.reflection.outside_factor)
        if not room_reflection(lin, srgb_to_linear(pixels), a_s, (px, py), ppm, spec.reflection, gloss,
                               fit.pose.delta, Ko, fit.R_lock):
            flags.append("reflection-skipped")
    out = linear_to_srgb(lin)
    shaded = out.copy() if debug else None
    paste(out, pixels, a_s, px, py)
    # Metrics.
    car_px = a_out >= 0.5
    hole = rr.hole & ~car_px
    ys, xs = np.nonzero(car_px)
    near = 0.0
    if len(xs):
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        nx0, ny0 = max(0, int(xs.min() - 0.25 * bw)), max(0, int(ys.min() - 0.25 * bh))
        nx1, ny1 = min(Wo, int(xs.max() + 1 + 0.25 * bw)), min(Ho, int(ys.max() + 1 + 0.25 * bh))
        sub = hole[ny0:ny1, nx0:nx1]
        near = float(sub.mean()) if sub.size else 0.0
    invented = float(hole[~car_px].mean()) if (~car_px).any() else 0.0
    if invented > 0.35 or near > 0.01:
        flags.append("invented")
    floor_tex = float(t1 / s)
    res = [r for r in fit.lock.get("res_after", [])] or fit.lock.get("res_before", [])
    resid = float(max(abs(r) for r in res) * s) if res else 0.0
    metrics = dict(out_w=Wo, out_h=Ho, out_scale=round(s, 4), invented_pct=round(100 * invented, 1),
                   invented_near_pct=round(100 * near, 2), floor_texels=round(floor_tex, 2), render_s=round(rr.seconds, 2),
                   room_units_per_m=round(sigma, 4), cam_room=[round(v, 3) for v in C_room],
                   contact_residual_px=round(resid, 1), reflection_scale=round(schlick_ratio(fit.pose.delta), 2)
                   if spec.reflection is not None else "")
    if resid > CONTACT_MISMATCH * Ho:  # (the shadows are pinned to the found tyre bottoms anyway)
        flags.append("contact-mismatch")
    dbg, steps = None, {}
    if debug:
        dbg = _debug_overlay(out, fit, Ko, R_room, C_room, room, S, sigma, A, found_out, rr, car_px)
        steps = _debug_steps(plain, shaded, a_out, rr, room, Ko, R_room, C_room, S)
    return Placement(out, (Wo, Ho), s, (x0, y0), metrics, flags, dbg, steps=steps)


def _debug_steps(plain, shaded, a_out, rr, room, Ko, R_room, C_room, S) -> dict:
    """The per-background step pictures: the re-rendered room with its edges, what each pixel shows (floor,
    wall, ceiling; areas the background picture never showed hatched), the shadows and reflection drawn on
    the floor, and where the car goes."""
    steps = {}
    room_img = draw_room(plain, room, Ko, R_room, C_room, (255, 0, 255), thick=2)
    q = plane_h(Ko, R_room, C_room) @ np.array([S[0], S[1], 1.0])
    if q[2] > 0:
        cv2.drawMarker(room_img, (int(round(q[0] / q[2])), int(round(q[1] / q[2]))), (0, 0, 255), cv2.MARKER_CROSS,
                       24, 2)
    steps["room"] = room_img
    colours = {FLOOR: (60, 180, 60), WALL: (200, 140, 40), CEIL: (170, 60, 170)}
    tint = np.zeros_like(plain)
    for kind, col in colours.items():
        tint[rr.kinds == kind] = col
    surf = cv2.addWeighted(plain, 0.5, tint, 0.5, 0)
    hatch = ((np.add.outer(np.arange(plain.shape[0]), np.arange(plain.shape[1])) // 6) % 2 == 0) & rr.hole
    surf[hatch] = (0, 0, 230)
    steps["surfaces"] = surf
    # Shadows (darker) and reflection (brighter) as the change they made to the room, on a grey ground.
    before = plain.astype(np.float32).mean(2)
    after = shaded.astype(np.float32).mean(2)
    diff = np.clip((after - before) * 3.0, -255, 255)
    vis = np.full(plain.shape, 128, np.float32)
    vis[diff < 0] = 128 + diff[diff < 0, None] * np.array([0.5, 0.5, 0.5])  # shadow: towards black
    vis[diff > 0] = 128 + diff[diff > 0, None] * np.array([0.0, 0.35, 0.5])  # reflection: towards orange
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    cs, _ = cv2.findContours((a_out >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (0, 220, 0), 2, cv2.LINE_AA)
    steps["shadows"] = vis
    steps["background_ready"] = shaded
    return steps


def _debug_overlay(img, fit, Ko, R_room, C_room, room, S, sigma, A, found_out, rr, car_px) -> np.ndarray:
    """Footprint (orange), stage (red +), horizon (cyan), projected tyres (cyan), contacts (green; grey =
    hidden side), found tyre bottoms (magenta x), invented areas tinted blue."""
    out = img.copy()
    tint = rr.hole & ~car_px
    out[tint] = (0.6 * out[tint] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    shape = fit.shape
    fp = shape.footprint(n_arc=16)
    u, v, _ = project_rc(np.c_[fp, np.zeros(len(fp))], fit.R_lock, fit.C_lock, Ko)
    if np.isfinite(u).all():
        cv2.polylines(out, [np.round(np.c_[u, v] * 4).astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA, 2)
    Hroom = plane_h(Ko, R_room, C_room)
    q = Hroom @ np.array([S[0], S[1], 1.0])
    if q[2] > 0:
        cv2.drawMarker(out, (int(round(q[0] / q[2])), int(round(q[1] / q[2]))), (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
    hz = Ko @ R_room @ np.array([[math.cos(t), math.sin(t), 0.0] for t in np.linspace(0, 2 * math.pi, 1441)]).T
    ok = hz[2] > 1e-6
    hp = (hz[:2, ok] / hz[2, ok]).T
    hp = hp[(hp[:, 0] > -50) & (hp[:, 0] < out.shape[1] + 50)]
    if len(hp) > 1:
        hp = hp[np.argsort(hp[:, 0])]
        cv2.polylines(out, [np.round(hp * 4).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA, 2)
    con = shape.contacts()
    uc, vc, zc = project_rc(con, fit.R_lock, fit.C_lock, Ko)
    for i, tb in enumerate(fit.bottoms):
        if not np.isfinite(uc[i]):
            continue
        col = (0, 255, 0) if tb.status == "found" else (160, 160, 160)
        cv2.circle(out, (int(round(uc[i])), int(round(vc[i]))), 4, col, 1, cv2.LINE_AA)
    for u_f, v_f, _, _ in found_out:
        cv2.drawMarker(out, (int(round(u_f)), int(round(v_f))), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 9, 1)
    return out


__all__ = ["Placement", "output_window", "place_in_room", "room_camera", "stage_scale"]
