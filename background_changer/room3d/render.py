"""Re-render a Room's plate through any pinhole camera (CPU: numpy + OpenCV).

Per output pixel: ray from the camera -> nearest hit with the floor, walls or ceiling (hits behind the
camera never count) -> project that point into the plate's camera -> sample the plate. Sampling is
mip-mapped (Gaussian pyramid, trilinear by the local footprint), so minified floors don't shimmer and
magnified ones are plain bicubic. Points the plate never shows are filled in their surface's own
coordinates (room.Room.fill_plane / fill_wall):
  floor   "hybrid": lighting from a blurred plate at the nearest shown point, fine texture tiled from a
          clean floor patch (falls back to "band" when the room has no patch);
  walls   "band": same height, folded back within a band along the wall, then the nearest shown height;
  ceiling "smooth": a heavily low-passed plate at the nearest shown point (no repeated light bars);
  sky above the plate's top row (walls moved in height): the same low-passed plate (no streaks or beams).
Everything is in linear light; the caller shades the floor and converts to sRGB.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .room import CEIL, FLOOR, NONE, WALL, Room, plane_h

FILL_DEFAULT = {"floor": "hybrid", "wall": "band", "ceil": "smooth"}


@dataclass
class Render:
    lin: np.ndarray  # (h, w, 3) float32, linear light, BGR
    hole: np.ndarray  # (h, w) bool: the plate never showed this surface point (it was filled)
    kinds: np.ndarray  # (h, w) int8: FLOOR, WALL, CEIL or NONE
    floor_xy: np.ndarray  # (h, w, 2) float32: the floor point (room units) seen at each floor pixel, NaN elsewhere
    texels: np.ndarray  # (h, w) float32: plate texels per output pixel (square root of the area ratio)
    seconds: float


def _mip_sample(pyr: list[np.ndarray], mx: np.ndarray, my: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Gaussian pyramid at plate coordinates (mx, my), choosing the level by the local footprint.
    Returns (image, plate texels per output pixel)."""
    gx_r, gx_c = np.gradient(mx)
    gy_r, gy_c = np.gradient(my)
    fp = np.sqrt(np.maximum(gx_c ** 2 + gy_c ** 2, gx_r ** 2 + gy_r ** 2))  # the footprint's longer axis
    texels = np.sqrt(np.abs(gx_c * gy_r - gx_r * gy_c)).astype(np.float32)
    n = len(pyr)
    lev = np.clip(np.log2(np.maximum(fp, 1e-6)), 0.0, n - 1.001).astype(np.float32)
    lev = cv2.erode(lev, np.ones((3, 3), np.uint8))  # keep edges next to a discontinuity sharp
    l0 = np.floor(lev).astype(np.int32)
    fr = lev - l0
    out = np.zeros(mx.shape + (pyr[0].shape[2],), np.float32)
    for k in range(min(n, int(l0.max()) + 2)):
        wk = np.where(l0 == k, 1.0 - fr, 0.0) + np.where(l0 == k - 1, fr, 0.0)
        if not (wk > 0).any():
            continue
        sc = 2.0 ** k
        # cv2.pyrDown keeps every second sample: level-k pixel i sits at level-0 position 2^k i.
        smp = cv2.remap(pyr[k], mx / sc, my / sc, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out += smp * wk[..., None].astype(np.float32)
    return np.clip(out, 0.0, None), texels


def render(plate: np.ndarray, room: Room, K, R, C, size, fill: dict | None = None, bands=(0.45, 1.7),
           hp_sigma: float = 10.0, sky_sigma: float = 25.0, chunk_px: int = 1 << 20) -> Render:
    """Re-render `room`'s plate through the pinhole camera (K, R, C) at size = (w, h).
    plate: the full-size picture the room was measured on (BGR uint8). bands: band widths (room units) for
    floor/ceiling and wall fills."""
    t0 = time.perf_counter()
    if (plate.shape[1], plate.shape[0]) != tuple(room.size):
        raise ValueError(f"{room.name}: the plate is {plate.shape[1]}x{plate.shape[0]} but the room was measured on "
                         f"{room.size[0]}x{room.size[1]} (the room path always uses the full-size background)")
    fills = dict(FILL_DEFAULT, **(fill or {}))
    hybrid = fills["floor"] == "hybrid" and room.texture_box is not None
    floor_mode = "clamp" if fills["floor"] == "hybrid" else fills["floor"]
    if fills["floor"] == "hybrid" and room.texture_box is None:
        floor_mode = "band"
    w, h = int(size[0]), int(size[1])
    K, R, C = (np.asarray(a, float) for a in (K, R, C))
    B = R.T @ np.linalg.inv(K)  # pixel (u, v, 1) -> world ray direction
    fwd = math.atan2(R[2, 1], R[2, 0])
    ca, sa = math.cos(fwd), math.sin(fwd)

    def rays(u, v):
        return (B[0, 0] * u + B[0, 1] * v + B[0, 2], B[1, 0] * u + B[1, 1] * v + B[1, 2],
                B[2, 0] * u + B[2, 1] * v + B[2, 2])

    def rel_phi(dx, dy):  # horizontal azimuth relative to the camera's forward azimuth
        return np.arctan2(-sa * dx + ca * dy, ca * dx + sa * dy)

    # Walls: exact nearest hits on a dense azimuth table (they depend only on the ray's azimuth).
    gu, gv = np.meshgrid(np.r_[np.arange(0, w, 8), w - 1].astype(float), np.r_[np.arange(0, h, 8), h - 1].astype(float))
    ph = rel_phi(*rays(gu, gv)[:2])
    lo_phi, hi_phi = float(ph.min()), float(ph.max())
    pad = 0.002 * (hi_phi - lo_phi) + 1e-6
    for zdir in (-1.0, 1.0):  # nadir or zenith in view: every azimuth occurs
        q = K @ (R @ np.array([0.0, 0.0, zdir]))
        if q[2] > 0 and 0 <= q[0] / q[2] <= w - 1 and 0 <= q[1] / q[2] <= h - 1:
            lo_phi, hi_phi, pad = -math.pi, math.pi, 0.0
    n_tab = 1 << 16
    g0, dg = lo_phi - pad, (hi_phi - lo_phi + 2 * pad) / (n_tab - 1)
    r_tab, s_tab = room.wall_hits(C[:2], g0 + dg * np.arange(n_tab) + fwd)

    def lookup(p, table):  # linear interpolation on the uniform azimuth grid
        f = np.clip((p - g0) / dg, 0.0, n_tab - 1.000001)
        i = f.astype(np.intp)
        f -= i
        return table[i] * (1.0 - f) + table[i + 1] * f

    map_x, map_y = np.empty((h, w), np.float32), np.empty((h, w), np.float32)
    hole, kinds, smooth = np.zeros((h, w), bool), np.empty((h, w), np.int8), np.zeros((h, w), bool)
    floor_xy = np.full((h, w, 2), np.nan, np.float32)
    if hybrid:
        tex_x, tex_y, hyb = np.empty((h, w), np.float32), np.empty((h, w), np.float32), np.zeros((h, w), bool)
    rows_per = max(1, chunk_px // w)
    for y0 in range(0, h, rows_per):
        y1 = min(h, y0 + rows_per)
        u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(y0, y1, dtype=float))
        dx, dy, dz = rays(u, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_f = np.where(dz < 0, -C[2] / dz, np.inf)
            t_c = (np.where(dz > 0, (room.ceiling - C[2]) / dz, np.inf) if room.ceiling is not None
                   else np.full(dz.shape, np.inf))
            phi = rel_phi(dx, dy)
            r_h = lookup(phi, r_tab)
            hxy = np.hypot(dx, dy)
            t_w = np.where(hxy > 1e-12, r_h / hxy, np.inf)
            z_w = C[2] + t_w * dz
        t_f = np.where(t_f > 1e-9, t_f, np.inf)  # hits behind the camera never count
        t_c = np.where(t_c > 1e-9, t_c, np.inf)
        t_w = np.where((t_w > 1e-9) & (z_w >= -1e-9) & (z_w <= room.wall_top), t_w, np.inf)
        T = np.stack([t_f, t_w, t_c])
        kind = np.argmin(T, axis=0).astype(np.int8)  # FLOOR 0, WALL 1, CEIL 2
        t = np.take_along_axis(T, kind[None].astype(np.intp), 0)[0]
        kind[~np.isfinite(t)] = NONE
        t = np.where(np.isfinite(t), t, 0.0)
        X, Y, Z = C[0] + t * dx, C[1] + t * dy, C[2] + t * dz
        Z = np.where(kind == FLOOR, 0.0, Z)
        if room.ceiling is not None:
            Z = np.where(kind == CEIL, room.ceiling, Z)
        fm = kind == FLOOR
        fxy = floor_xy[y0:y1]
        fxy[fm, 0], fxy[fm, 1] = X[fm], Y[fm]
        up, vp, dep = room.to_plate(X, Y, Z)
        bad = ~room.inside(up, vp, dep) | (kind == NONE)
        sm = np.zeros_like(bad)
        tu = tv = hb = None
        if bad.any():
            for kd in (FLOOR, CEIL, WALL):
                m = bad & (kind == kd)
                if not m.any():
                    continue
                if kd == WALL:
                    fx, fy, fz, vert = room.fill_wall(lookup(phi[m], s_tab), Z[m], fills["wall"], bands[1])
                    tmp = np.zeros_like(m)
                    tmp[m] = vert
                    sm |= tmp
                elif kd == CEIL:
                    fx, fy, fz = room.fill_plane(X[m], Y[m], room.ceiling, "clamp", bands[0])
                    if fills["ceil"] == "smooth":
                        sm |= m
                else:
                    fx, fy, fz = room.fill_plane(X[m], Y[m], 0.0, floor_mode, bands[0])
                    if hybrid:
                        tx, ty = room.tile(X[m], Y[m])
                        tu, tv = up.copy(), vp.copy()
                        tu[m], tv[m], _ = room.to_plate(tx, ty, np.zeros_like(tx))
                        hb = m
                up[m], vp[m], dep[m] = room.to_plate(fx, fy, fz)
            still = ~room.inside(up, vp, dep, margin=0.0)
            if still.any():  # last resort (e.g. no hit at all): clamp in the plate's image space
                pw, phh = room.size
                up[still] = np.clip(np.nan_to_num(up[still], nan=pw / 2), 2.0, pw - 3.0)
                vp[still] = np.clip(np.nan_to_num(vp[still], nan=phh / 2), 2.0, phh - 3.0)
        map_x[y0:y1], map_y[y0:y1] = up, vp
        hole[y0:y1], kinds[y0:y1], smooth[y0:y1] = bad, kind, sm
        if hybrid:
            tex_x[y0:y1], tex_y[y0:y1] = (tu, tv) if tu is not None else (up, vp)
            hyb[y0:y1] = hb if hb is not None else False
    layers = room.plate_layers(plate, hp_sigma, sky_sigma)
    lin, texels = _mip_sample(layers["pyr"], map_x, map_y)
    if hybrid and hyb.any():
        low = cv2.remap(layers["low"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        high = cv2.remap(layers["high"], tex_x, tex_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        lin[hyb] = np.clip(low[hyb] + high[hyb], 0.0, 1.0)
    if smooth.any():
        sky = cv2.remap(layers["sky"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        lin[smooth] = sky[smooth]
    return Render(lin=lin, hole=hole, kinds=kinds, floor_xy=floor_xy, texels=texels, seconds=time.perf_counter() - t0)


def texels_per_px(room: Room, K, R, C, px=None) -> tuple[float, float, float]:
    """Plate texels per output pixel on the floor at output pixel px (default: where the room origin, the
    stage point, appears): (along output x, along output y, square root of the area ratio). Below 1 the
    plate is magnified there (soft)."""
    Hout = plane_h(np.asarray(K, float), np.asarray(R, float), np.asarray(C, float))
    G = plane_h(room.K, room.R, room.C) @ np.linalg.inv(Hout)
    if px is None:
        q = Hout @ np.array([0.0, 0.0, 1.0])
        px = q[:2] / q[2]

    def g(p):
        q = G @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]

    e = 0.5
    jx = (g((px[0] + e, px[1])) - g((px[0] - e, px[1]))) / (2 * e)
    jy = (g((px[0], px[1] + e)) - g((px[0], px[1] - e))) / (2 * e)
    return float(np.linalg.norm(jx)), float(np.linalg.norm(jy)), float(math.sqrt(abs(jx[0] * jy[1] - jx[1] * jy[0])))


__all__ = ["FILL_DEFAULT", "Render", "render", "texels_per_px", "CEIL", "FLOOR", "NONE", "WALL"]
