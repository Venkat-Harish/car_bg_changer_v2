"""A background picture ("plate") rebuilt as a 3D room: floor, walls, ceiling or sky backdrop.

World frame of a room: origin = the stage point on the floor; x right, y forward (away from the plate
camera), z up; the floor is z = 0. A camera is (K, R, C): K 3x3 with pixel centres at integer coordinates,
R world -> camera (rows: right, down, forward), C its centre in world units. A plate pixel is K R (P - C).
Room units are the plate's own calibrated units; callers scale car metres into them.

Walls are a closed floor-plan polygon extruded from the floor up to `wall_top` (the part behind the plate
camera is never shown). The plate only shows part of each surface. Points it does not show are filled in
that surface's own coordinates (see `Room.fill_plane` / `Room.fill_wall`), so floor seams, LED strips and
plinths run on straight instead of smearing.

Nothing here knows about cars: this is the background side only.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

FLOOR, WALL, CEIL, NONE = 0, 1, 2, 3
FILL_MARGIN = 2.0  # px: fill targets stay this far inside the plate (Lanczos support)
FRONT_EPS = 0.05  # room units: points this close to the plate camera's image plane count as not shown

_LIN_LUT = np.array([(i / 255) / 12.92 if i / 255 <= 0.04045 else ((i / 255 + 0.055) / 1.055) ** 2.4
                     for i in range(256)], dtype=np.float32)


def srgb_to_linear(img_u8: np.ndarray) -> np.ndarray:
    return _LIN_LUT[img_u8]


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def look_at(C, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World -> camera rotation looking from C at target with no roll (camera x axis horizontal)."""
    f = np.asarray(target, float) - np.asarray(C, float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(r) < 1e-9 else r / np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f])


def kmat(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def plane_h(K, R, C, z0: float = 0.0) -> np.ndarray:
    """Homography of the horizontal plane z = z0: (x, y, 1) -> pixels."""
    K, R, C = (np.asarray(a, float) for a in (K, R, C))
    return K @ R @ np.array([[1.0, 0.0, -C[0]], [0.0, 1.0, -C[1]], [0.0, 0.0, z0 - C[2]]])


def project(K, R, C, X, Y, Z):
    """World points -> (u, v, depth along the camera's axis)."""
    M = np.asarray(K, float) @ np.asarray(R, float)
    a, b, c = X - C[0], Y - C[1], Z - C[2]
    q2 = M[2, 0] * a + M[2, 1] * b + M[2, 2] * c
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (M[0, 0] * a + M[0, 1] * b + M[0, 2] * c) / q2
        v = (M[1, 0] * a + M[1, 1] * b + M[1, 2] * c) / q2
    return u, v, q2


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _distance_to_polygon(x: float, y: float, poly: np.ndarray) -> float:
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    t = np.clip(((x - a[:, 0]) * e[:, 0] + (y - a[:, 1]) * e[:, 1]) / np.maximum((e ** 2).sum(1), 1e-12), 0, 1)
    d = np.hypot(a[:, 0] + t * e[:, 0] - x, a[:, 1] + t * e[:, 1] - y)
    return float(d.min())


class Room:
    """Floor z = 0, walls = vertical extrusion (0 <= z <= wall_top) of a closed floor-plan polygon, and an
    optional ceiling plane. Textured by projecting into the plate's calibrated camera (K, R, C).

    texture_box (x0, y0, w, h): a clean floor patch the plate shows sharply; it supplies fine texture where
    the floor has to be filled. devignette (a, r): the plate's brightness falls off as 1 - a (d / r)^2 with
    the distance d from the principal point; it is undone before projecting, so darker plate edges do not
    move into the middle of a new view."""

    def __init__(self, name, K, R, C, size, poly, wall_top, ceiling=None, texture_box=None, devignette=None,
                 notes=""):
        self.name, self.size, self.notes = name, (int(size[0]), int(size[1])), notes
        self.K, self.R, self.C = (np.asarray(a, float) for a in (K, R, C))
        self.M = self.K @ self.R
        self.wall_top = float(wall_top)
        self.ceiling = None if ceiling is None else float(ceiling)
        self.texture_box = texture_box
        self.devignette = devignette
        P = np.asarray(poly, float)
        self.poly, self.p0, self.p1 = P, P, np.roll(P, -1, axis=0)
        self.seg_len = np.linalg.norm(self.p1 - self.p0, axis=1)
        self.s0 = np.r_[0.0, np.cumsum(self.seg_len)[:-1]]
        self.perimeter = float(self.seg_len.sum())
        self._vis = None
        self._plate_cache: dict = {}

    # ------------------------------------------------------------------ the plate camera
    def to_plate(self, X, Y, Z):
        return project(self.K, self.R, self.C, X, Y, Z)

    def inside(self, u, v, depth, margin=-0.5):
        w, h = self.size
        return (depth > FRONT_EPS) & (u >= margin) & (u <= w - 1 - margin) & (v >= margin) & (v <= h - 1 - margin)

    def floor_point(self, px) -> np.ndarray:
        """The floor point (x, y) under plate pixel px."""
        q = np.linalg.solve(plane_h(self.K, self.R, self.C), np.array([px[0], px[1], 1.0]))
        return q[:2] / q[2]

    def px_per_unit_at(self, xy=(0.0, 0.0)) -> float:
        """Lateral floor scale (plate px per room unit, along x) at floor point xy."""
        H, e = plane_h(self.K, self.R, self.C), 1e-4
        p = [H @ np.array([xy[0] + s, xy[1], 1.0]) for s in (-e, e)]
        p = [q[:2] / q[2] for q in p]
        return float(np.linalg.norm(p[1] - p[0]) / (2 * e))

    def contains(self, C, margin: float = 0.3) -> bool:
        """Is a camera at C inside the room, at least `margin` units from the floor, walls and ceiling?"""
        top = self.wall_top if self.ceiling is None else min(self.wall_top, self.ceiling)
        return bool(margin <= C[2] <= top - margin and _point_in_polygon(C[0], C[1], self.poly)
                    and _distance_to_polygon(C[0], C[1], self.poly) >= margin)

    def clamp_inside(self, C, margin: float = 0.3, toward=(0.0, 0.0)) -> np.ndarray:
        """A point near C that `contains` accepts: height clamped, then pulled toward the floor point `toward`
        (the stage)."""
        top = self.wall_top if self.ceiling is None else min(self.wall_top, self.ceiling)
        out = np.array(C, float)
        out[2] = min(max(out[2], margin), top - margin)
        t = np.asarray(toward, float)
        for _ in range(60):
            if self.contains(out, margin):
                break
            out[:2] = t + 0.95 * (out[:2] - t)
        return out

    # ------------------------------------------------------------------ plate pixels (linear light)
    def plate_layers(self, plate: np.ndarray, hp_sigma: float, sky_sigma: float, levels: int = 6) -> dict:
        """Linear-light plate (de-vignetted), its Gaussian pyramid and two low-passed copies (cached)."""
        sample = np.ascontiguousarray(plate[::7, ::7])
        key = (hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest(), plate.shape, hp_sigma, sky_sigma, levels)
        if key not in self._plate_cache:
            if len(self._plate_cache) > 2:
                self._plate_cache.clear()
            if self.devignette is None:
                lin = srgb_to_linear(plate)
            else:  # measured on the sRGB values, so undone there
                a, r = self.devignette
                h, w = plate.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                d2 = ((xx - self.K[0, 2]) ** 2 + (yy - self.K[1, 2]) ** 2) / float(r) ** 2
                x = np.clip(plate.astype(np.float32) / 255.0 / np.maximum(1.0 - a * d2, 0.2)[..., None], 0.0, 1.0)
                lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
            pyr = [lin]
            for _ in range(levels):
                pyr.append(cv2.pyrDown(pyr[-1]))
            low = cv2.GaussianBlur(lin, (0, 0), hp_sigma)
            self._plate_cache[key] = dict(lin=lin, pyr=pyr, low=low, high=lin - low,
                                          sky=cv2.GaussianBlur(lin, (0, 0), sky_sigma))
        return self._plate_cache[key]

    # ------------------------------------------------------------------ walls
    def wall_xy(self, s):
        s = np.mod(s, self.perimeter)
        i = np.clip(np.searchsorted(self.s0, s, side="right") - 1, 0, len(self.s0) - 1)
        t = (s - self.s0[i]) / self.seg_len[i]
        return (self.p0[i, 0] + t * (self.p1[i, 0] - self.p0[i, 0]), self.p0[i, 1] + t * (self.p1[i, 1] - self.p0[i, 1]))

    def wall_hits(self, cxy, phi):
        """Nearest wall along horizontal directions phi (rad) from cxy: (distance, arc length s)."""
        dx, dy = np.cos(phi), np.sin(phi)
        best_r, best_s = np.full(len(phi), np.inf), np.zeros(len(phi))
        for a, b, s0, L in zip(self.p0, self.p1, self.s0, self.seg_len):
            ex, ey = b - a
            wx, wy = a - cxy
            den = dx * ey - dy * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (wx * ey - wy * ex) / den
                t = (wx * dy - wy * dx) / den
            ok = (r > 1e-9) & (t >= -1e-9) & (t <= 1 + 1e-9) & (r < best_r)
            best_r[ok], best_s[ok] = r[ok], s0 + np.clip(t[ok], 0.0, 1.0) * L
        return best_r, best_s

    def _frustum(self, margin):
        """Rows c with c . (P - C) >= 0 for points P inside the plate (margin px); the last row is depth."""
        w, h = self.size
        M = self.M
        return np.stack([M[0] - margin * M[2], (w - 1 - margin) * M[2] - M[0],
                         M[1] - margin * M[2], (h - 1 - margin) * M[2] - M[1], M[2]])

    def wall_visibility(self):
        """Per height z: the arc-length interval [s_lo, s_hi] of the walls the plate shows."""
        if self._vis is None:
            top = min(self.ceiling if self.ceiling is not None else self.wall_top, 200.0)
            zs = np.unique(np.r_[np.linspace(0.0, min(top, 30.0), 1201), np.linspace(min(top, 30.0), top, 120)])
            F = self._frustum(FILL_MARGIN)
            n = len(self.p0)
            lo, hi = np.zeros((len(zs), n)), np.ones((len(zs), n))
            A = np.c_[self.p0 - self.C[:2], np.full(n, -self.C[2])]
            E = np.c_[self.p1 - self.p0, np.zeros(n)]
            for k, c in enumerate(F):
                g = (A @ c)[None, :] + zs[:, None] * c[2] - (FRONT_EPS if k == 4 else 0.0)
                hv = np.broadcast_to((E @ c)[None, :], g.shape)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = -g / hv
                lo = np.where(hv > 0, np.maximum(lo, t), lo)
                hi = np.where(hv < 0, np.minimum(hi, t), hi)
                hi = np.where((hv == 0) & (g < 0), -1.0, hi)
            ok = hi > lo
            s_lo = np.where(ok, self.s0 + lo * self.seg_len, np.inf).min(axis=1)
            s_hi = np.where(ok, self.s0 + hi * self.seg_len, -np.inf).max(axis=1)
            valid = np.isfinite(s_lo)
            self._vis = (zs[valid], s_lo[valid], s_hi[valid])
        return self._vis

    # ------------------------------------------------------------------ hole fills, in surface coordinates
    @staticmethod
    def _fold(s, lo, hi, mode, band):
        """Bring s into [lo, hi]. clamp: nearest end; mirror: reflect about it; band: a triangle wave that only
        re-uses the strip of width `band` next to the end (features deeper inside are never copied)."""
        if mode == "mirror":
            s = np.where(s < lo, 2 * lo - s, np.where(s > hi, 2 * hi - s, s))
        elif mode == "band":
            w = np.maximum(np.minimum(band, hi - lo), 1e-9)
            d = np.where(s < lo, lo - s, np.where(s > hi, s - hi, 0.0))
            t = np.mod(d, 2 * w)
            t = np.where(t > w, 2 * w - t, t)
            s = np.where(s < lo, lo + t, np.where(s > hi, hi - t, s))
        return np.clip(s, lo, hi)

    def fill_wall(self, s, z, mode, band=1.0):
        """Wall points the plate does not show. Off its left/right side (or behind it): same height, nearest
        shown point along the wall (or folded back, see _fold), so strips and plinths run on straight. Then,
        if still above/below the plate: the same wall column, nearest shown height. Returns (x, y, z, vert),
        vert = points moved in height (sampled from a low-passed plate by the renderer: no streaks)."""
        zs, s_lo, s_hi = self.wall_visibility()
        w, h = self.size
        x, y = self.wall_xy(s)
        u, v, d = self.to_plate(x, y, z)
        side = (d <= FRONT_EPS) | ~(u >= FILL_MARGIN) | ~(u <= w - 1 - FILL_MARGIN)
        if side.any():
            zc = np.clip(z[side], zs[0], zs[-1])
            lo, hi = np.interp(zc, zs, s_lo), np.interp(zc, zs, s_hi)
            s = s.copy()
            s[side] = self._fold(s[side], lo, hi, mode, band)
            x, y = self.wall_xy(s)
            z = z.copy()
            z[side] = zc
            u, v, d = self.to_plate(x, y, z)
        vert = ~((v >= FILL_MARGIN) & (v <= h - 1 - FILL_MARGIN))
        if vert.any():  # heights the plate shows on this wall column: solve v(z) on the vertical line
            M, C = self.M, self.C
            q0 = (np.stack([x[vert], y[vert], np.zeros(vert.sum())], -1) - C) @ M.T
            m = M[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                z_top = (FILL_MARGIN * q0[:, 2] - q0[:, 1]) / (m[1] - FILL_MARGIN * m[2])
                z_bot = ((h - 1 - FILL_MARGIN) * q0[:, 2] - q0[:, 1]) / (m[1] - (h - 1 - FILL_MARGIN) * m[2])
            lo = np.clip(np.nan_to_num(np.minimum(z_top, z_bot), nan=0.0), 0.0, self.wall_top)
            hi = np.clip(np.nan_to_num(np.maximum(z_top, z_bot), nan=0.0), 0.0, self.wall_top)
            z = z.copy()
            z[vert] = self._fold(z[vert], lo, np.maximum(hi, lo), "clamp", band)
        return x, y, z, vert

    def tile(self, x, y):
        """Mirror-periodic tiling of the floor plane into texture_box (continuous; never leaves the box)."""
        x0, y0, bw, bh = self.texture_box

        def tri(a, a0, L):
            t = np.mod(a - a0, 2 * L)
            return a0 + np.where(t > L, 2 * L - t, t)

        return tri(x, x0, bw), tri(y, y0, bh)

    def fill_plane(self, x, y, z0, mode, band=0.5):
        """Floor / ceiling points the plate does not show: same x, nearest shown y (along the plate
        camera's forward axis), or folded back (see _fold)."""
        B = np.array([[1.0, 0.0, -self.C[0]], [0.0, 1.0, -self.C[1]], [0.0, 0.0, z0 - self.C[2]]])
        y_lo, y_hi = np.full_like(y, -np.inf), np.full_like(y, np.inf)
        for k, c in enumerate(self._frustum(FILL_MARGIN)):
            cc = c @ B
            c0 = cc[2] - (FRONT_EPS if k == 4 else 0.0)
            if abs(cc[1]) < 1e-12:
                continue
            yb = -(cc[0] * x + c0) / cc[1]
            if cc[1] > 0:
                y_lo = np.maximum(y_lo, yb)
            else:
                y_hi = np.minimum(y_hi, yb)
        ok = y_lo <= y_hi
        y = np.where(ok, self._fold(y, y_lo, np.where(ok, y_hi, y_lo + 1.0), mode, band), y)
        return x, y, np.full_like(x, z0)


def draw_room(img, room: Room, K, R, C, colour=(255, 0, 255), heights=None, thick=1):
    """Overlay the room's edges (polygon at each height, vertical corners) through camera (K, R, C)."""
    out = img.copy()
    top = room.ceiling if room.ceiling is not None else min(room.wall_top, 10.0)
    heights = [0.0, top] if heights is None else heights

    def poly_line(u, v, ok):
        for i in range(len(u) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(out, (int(round(u[i] * 4)), int(round(v[i] * 4))), (int(round(u[i + 1] * 4)),
                         int(round(v[i + 1] * 4))), colour, thick, cv2.LINE_AA, shift=2)

    for z in heights:
        for a, b in zip(room.p0, room.p1):
            t = np.linspace(0.0, 1.0, 400)
            u, v, d = project(K, R, C, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), np.full_like(t, z))
            poly_line(u, v, (d > 0.05) & (np.abs(u) < 2e4) & (np.abs(v) < 2e4))
    for a in room.p0:
        z = np.linspace(0.0, top, 50)
        u, v, d = project(K, R, C, np.full_like(z, a[0]), np.full_like(z, a[1]), z)
        poly_line(u, v, (d > 0.05) & (np.abs(u) < 2e4) & (np.abs(v) < 2e4))
    return out


__all__ = ["CEIL", "FLOOR", "NONE", "WALL", "Room", "draw_room", "kmat", "linear_to_srgb", "look_at",
           "plane_h", "project", "srgb_to_linear"]
