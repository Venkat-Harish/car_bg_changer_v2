"""The one allowed change to the car's pixels: a rigid rotation about the frame centre that levels the photo
(camera roll plus the car's own roll on a sloped road), followed later by one INTER_AREA downscale.

Rotation, at the photo's own resolution, onto a canvas holding just the car:
  colour = Lanczos4(alpha * c) / Lanczos4(alpha)   (the same kernel; 0 where that alpha < 1/255),
           clamped to the min/max of the 3x3 source pixels with alpha > 0 (no ringing, no new colours);
  alpha  = bilinear(alpha), values above 1 - 1e-4 set to 1.
Skipped below 0.5 degrees: the canvas is then a plain crop and the car's pixels stay bit-identical.

Grid convention: a photo pixel p goes to c + Rot(theta) (p - c) on the "levelled grid" (c = the frame centre,
Rot(theta) = [[cos, -sin], [sin, cos]] in image axes x right, y down). This is the image of the camera
rotated by Rz(theta) about its optical axis, so the levelled camera is Rz(theta) @ R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from bg_changer.segment import fill_holes

MIN_DEG = 0.5
MAX_CUT_CROP = 0.02  # share of the visible car width a slanted cut edge may lose to levelling
CAR_ROLL_RANGE = (1.0, 8.0)  # deg: the car's own roll is applied in this range
END_SIDE_TOL = 15.0


def rz(theta_deg: float) -> np.ndarray:
    a = math.radians(theta_deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def to_levelled(p, c, theta_deg: float) -> np.ndarray:
    """Photo px -> levelled grid px (full resolution)."""
    p = np.atleast_2d(np.asarray(p, float))
    a = math.radians(theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    d = p - np.asarray(c, float)
    return np.stack([c[0] + ca * d[:, 0] - sa * d[:, 1], c[1] + sa * d[:, 0] + ca * d[:, 1]], 1)


@dataclass
class LevelledCar:
    pixels: np.ndarray  # (h, w, 3) uint8 BGR
    alpha: np.ndarray  # (h, w) float32 0..1
    origin: tuple[int, int]  # the canvas's top-left on the levelled grid (full resolution)
    theta: float  # applied rotation (deg); 0 = untouched


def level_car(frame: np.ndarray, alpha: np.ndarray, theta: float, c, pad: int = 6) -> LevelledCar:
    """Rotate the car (frame + full-frame alpha) by theta about c onto a canvas holding just the car."""
    ys, xs = np.nonzero(alpha > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if abs(theta) < MIN_DEG:
        return LevelledCar(frame[y1:y2, x1:x2].copy(), alpha[y1:y2, x1:x2].astype(np.float32).copy(), (x1, y1), 0.0)
    corners = to_levelled(np.array([[x1 - 0.5, y1 - 0.5], [x2 - 0.5, y1 - 0.5], [x2 - 0.5, y2 - 0.5],
                                    [x1 - 0.5, y2 - 0.5]]), c, theta)
    ox, oy = int(math.floor(corners[:, 0].min())) - pad, int(math.floor(corners[:, 1].min())) - pad
    ex, ey = int(math.ceil(corners[:, 0].max())) + pad, int(math.ceil(corners[:, 1].max())) + pad
    gx, gy = np.meshgrid(np.arange(ox, ex, dtype=np.float32), np.arange(oy, ey, dtype=np.float32))
    src = to_levelled(np.stack([gx.ravel(), gy.ravel()], 1), c, -theta)  # inverse rotation
    sx = src[:, 0].reshape(gx.shape).astype(np.float32)
    sy = src[:, 1].reshape(gx.shape).astype(np.float32)
    # Work on a source crop around the car (+ Lanczos support) to keep the remaps small.
    m = 8
    cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
    cx2, cy2 = min(frame.shape[1], x2 + m), min(frame.shape[0], y2 + m)
    f32 = frame[cy1:cy2, cx1:cx2].astype(np.float32)
    a = alpha[cy1:cy2, cx1:cx2].astype(np.float32)
    sx -= cx1
    sy -= cy1
    border = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    num = cv2.remap(f32 * a[..., None], sx, sy, cv2.INTER_LANCZOS4, **border)
    den = cv2.remap(a, sx, sy, cv2.INTER_LANCZOS4, **border)
    a_out = np.clip(cv2.remap(a, sx, sy, cv2.INTER_LINEAR, **border), 0.0, 1.0)
    car = a > 0
    big = np.float32(1e6)
    lo_src = cv2.erode(np.where(car[..., None], f32, big), np.ones((3, 3), np.uint8))
    hi_src = cv2.dilate(np.where(car[..., None], f32, -big), np.ones((3, 3), np.uint8))
    near = dict(interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    lo = cv2.remap(lo_src, sx, sy, **near)
    hi = cv2.remap(hi_src, sx, sy, **near)
    with np.errstate(divide="ignore", invalid="ignore"):
        col = np.where(den[..., None] >= 1 / 255, num / np.maximum(den[..., None], 1e-6), 0.0)
    valid = hi >= lo
    col = np.where(valid, np.clip(col, lo, hi), 0.0)
    a_out[a_out < 1 / 255] = 0.0
    a_out[a_out > 1 - 1e-4] = 1.0
    pixels = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    pixels[a_out <= 0] = 0
    return LevelledCar(pixels, a_out.astype(np.float32), (ox, oy), float(theta))


def downscale(car: LevelledCar, s: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """The car's one INTER_AREA downscale (never up). Opaque pixels: plain INTER_AREA of the colour, as the flat
    path; soft-edge pixels: INTER_AREA(colour x alpha) / INTER_AREA(alpha). Returns (pixels, alpha, top-left corner
    on the scaled levelled grid)."""
    h, w = car.alpha.shape
    corner = (car.origin[0] * s, car.origin[1] * s)  # pixel corners scale; centres go to (p + 0.5) s - 0.5
    if s >= 1.0:
        a = car.alpha.copy()
        a[a > 1 - 1e-4] = 1.0
        return car.pixels, a, corner
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    pixels = cv2.resize(car.pixels, size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(car.alpha, size, interpolation=cv2.INTER_AREA)
    # Soft edge: average the colour weighted by alpha, so pixels outside the car (black after a rotation, the
    # old background otherwise) never darken or tint the outline. Opaque pixels keep the plain INTER_AREA.
    edge = (a > 1 / 255) & (a <= 1 - 1e-4)
    if edge.any():
        prem = cv2.resize(car.pixels.astype(np.float32) * car.alpha[..., None], size, interpolation=cv2.INTER_AREA)
        col = prem[edge] / np.maximum(a[edge], 1e-6)[:, None]
        pixels[edge] = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    a[a > 1 - 1e-4] = 1.0
    return pixels, a.astype(np.float32), corner


# ---------------------------------------------------------------- the levelling angle
def roll_of(R: np.ndarray) -> float:
    """The image rotation that makes the car's floor level: the roll of the car's down axis in the camera."""
    g = -R[:, 2]
    return math.degrees(math.atan2(g[0], g[1]))


def cut_crop(alpha: np.ndarray, cut, c, theta: float) -> dict:
    """Where a slanted cut edge lands on the levelled grid. For each cut side: the innermost and outermost
    position of the photo border rows/columns the car touches, and the crop that keeps the cut edge on the
    output border, as px and share of the visible car width (height for top/bottom cuts)."""
    H, W = alpha.shape
    solid = alpha >= 0.5
    out = {}
    ys, xs = np.nonzero(solid)
    vis_w, vis_h = (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) if len(xs) else (1, 1)
    for side in cut:
        if side in ("left", "right"):
            rows = np.flatnonzero(solid[:, :3].any(1) if side == "left" else solid[:, W - 3:].any(1))
            if not len(rows):
                continue
            xb = -0.5 if side == "left" else W - 0.5
            q = to_levelled(np.c_[np.full(len(rows), xb), rows], c, theta)[:, 0]
            lo, hi = float(q.min()), float(q.max())
            crop = hi - lo
            out[side] = dict(lo=lo, hi=hi, crop_px=crop, crop_pct=100.0 * crop / vis_w)
        else:
            cols = np.flatnonzero(solid[:3].any(0) if side == "top" else solid[H - 3:].any(0))
            if not len(cols):
                continue
            yb = -0.5 if side == "top" else H - 0.5
            q = to_levelled(np.c_[cols, np.full(len(cols), yb)], c, theta)[:, 1]
            lo, hi = float(q.min()), float(q.max())
            crop = hi - lo
            out[side] = dict(lo=lo, hi=hi, crop_px=crop, crop_pct=100.0 * crop / vis_h)
    return out


def limit_for_cut(alpha: np.ndarray, cut, c, theta: float) -> tuple[float, bool]:
    """Reduce |theta| until no slanted cut edge loses more than MAX_CUT_CROP of the visible car. Returns
    (theta to apply, reduced?)."""
    if not cut or abs(theta) < MIN_DEG:
        return theta, False
    for k in np.linspace(1.0, 0.0, 41):
        th = theta * k
        info = cut_crop(alpha, cut, c, th)
        if all(v["crop_pct"] <= 100 * MAX_CUT_CROP for v in info.values()):
            return (th if abs(th) >= MIN_DEG else 0.0), k < 1.0
    return 0.0, True


def thirds_tilt(u, v) -> float:
    """Tilt (deg, + rises to the right) of the line from the lowest point of the left third of a bottom
    outline to the lowest point of its right third."""
    o = np.argsort(u)
    u, v = np.asarray(u, float)[o], np.asarray(v, float)[o]
    n = len(u) // 3
    if n < 2:
        return 0.0
    i, j = int(np.argmax(v[:n])), len(u) - n + int(np.argmax(v[-n:]))
    return math.degrees(math.atan2(v[i] - v[j], u[j] - u[i]))


def bottom_profile(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns holding the mask and the lowest mask row in each (holes filled)."""
    s = fill_holes(mask.astype(np.uint8))
    has = s.any(0)
    cols = np.flatnonzero(has)
    bottom = s.shape[0] - 1 - np.argmax(s[::-1, cols], axis=0)
    return cols.astype(float), bottom.astype(float)


def near_end_or_side(yaw: float) -> bool:
    return min(abs((yaw - a + 180) % 360 - 180) for a in (0, 90, 180, 270)) <= END_SIDE_TOL


__all__ = ["CAR_ROLL_RANGE", "LevelledCar", "MIN_DEG", "bottom_profile", "cut_crop", "downscale", "level_car",
           "limit_for_cut", "near_end_or_side", "roll_of", "rz", "thirds_tilt", "to_levelled"]
