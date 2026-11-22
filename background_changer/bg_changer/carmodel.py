"""A 3D proxy of the car, built from its dimensions, and its view through the capture camera V.

Why a capture camera. The car's pixels may only be scaled down and moved, so the car always keeps the
walkaround's own viewpoint: phone photos taken close to the car (about 3.4-4.6 m) and looking down
about 16 degrees. V models that viewpoint: 4.70 m from the car's footprint centre, 16.08 degrees down
(about 1.30 m high). Projecting the EQB-shaped proxy through V reproduces Spyne's spin of the same
video to about 6 px RMS, while a background's own (farther, flatter) camera would make the car's
height and width disagree by up to 23%. V is a constant of the method, not an input: nobody has to
supply camera details.

The proxy is the EQB shape measured from the frames (lower body with apron and bumpers, greenhouse
with a sloped windscreen, roof rails, mirrors, four tyres). Other cars use the same shape stretched
to their length, width, height and wheelbase (track = 0.864 * width, tyre diameter = 0.43 * height,
front overhang = 48.5% of the overhangs).

Conventions (as the view classifier): yaw 0 = the car's front faces the camera, 90 = its right side,
180 = its rear, 270 = its left side. Car frame: X forward, Y left, Z up, origin = the middle of the
wheelbase on the floor ("footprint centre").

Everything in CarTable is in "anchor metres": image px = anchor + px_per_m * value, where px_per_m
is the scale at the footprint centre. The focal length of V is then px_per_m * distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CAPTURE_DISTANCE_M = 4.70  # slant distance from V to the footprint centre
CAPTURE_ELEVATION_DEG = 16.08  # V looks down this much at the footprint centre
ASPECT_BIAS = 1.05  # photos are ~5% narrower than V predicts at equal height (1.053 +- 0.038, 15 frames)

# Reference car: 2024 Mercedes-Benz EQB. The shape below is in its metres.
REF_LENGTH, REF_WIDTH, REF_HEIGHT, REF_WHEELBASE = 4.684, 1.834, 1.667, 2.829
REF_XF, REF_XR = 2.3135, -2.3705  # front / rear tips (overhangs 0.899 / 0.956)
REF_HALF_W = 0.917
REF_HALF_TRACK = 0.7925
REF_TYRE_R, REF_TYRE_W = 0.358, 0.235
FRONT_OVERHANG_SHARE = (REF_XF - REF_WHEELBASE / 2) / (REF_LENGTH - REF_WHEELBASE)  # 0.4846
FOOTPRINT_CORNER_R = 0.35
SECTION_STEP = 0.08  # m between sampled body sections
TYRE_POINTS = 48


def _ref_half_width(x: np.ndarray) -> np.ndarray:
    """Plan half-width of the reference body, with rounded corners (r 0.40 front, 0.30 rear)."""
    w = np.full_like(x, REF_HALF_W)
    for centre, r, side in ((REF_XF - 0.40, 0.40, 1), (REF_XR + 0.30, 0.30, -1)):
        m = (x - centre) * side > 0
        w[m] = REF_HALF_W - r + np.sqrt(np.maximum(r * r - (x[m] - centre) ** 2, 0.0))
    return w


def _ref_body_points() -> np.ndarray:
    """Vertices of the reference body's convex pieces (lower body, greenhouse, rails, mirrors)."""
    breaks = [1.95, -1.95, 2.15, 1.10, REF_XF - 0.40, REF_XR + 0.30]
    xs = np.unique(np.concatenate([
        np.linspace(REF_XR, REF_XF, int(math.ceil((REF_XF - REF_XR) / SECTION_STEP)) + 1),
        np.array([b + d for b in breaks for d in (-1e-4, 1e-4)]),
    ]))
    xs = xs[(xs >= REF_XR) & (xs <= REF_XF)]
    w = _ref_half_width(xs)
    zb = np.where(xs > 1.95, 0.30, np.where(xs < -1.95, 0.40, 0.20))  # apron / rear bumper / sills
    zt = np.full_like(xs, 1.08)  # beltline
    m = (xs >= 1.10) & (xs < 2.15)
    zt[m] = 1.08 - 0.16 * (xs[m] - 1.10) / 1.05  # bonnet
    m = xs >= 2.15
    zt[m] = 0.92 - 0.17 * (xs[m] - 2.15) / (REF_XF - 2.15)  # nose
    pts = []
    for y_sign in (1, -1):
        for z in (zb, zt):
            pts.append(np.stack([xs, y_sign * w, z], axis=1))
    # Greenhouse: base at the beltline, top at the roof (sloped windscreen in front of X = 0.35).
    g = xs[(xs >= -2.32) & (xs <= 1.10)]
    gw = np.minimum(0.84, _ref_half_width(g))
    zr = np.where(g > 0.35, 1.08 + 0.56 * (1.10 - g) / 0.75, 1.64)
    top_w = 0.84 - 0.20 * (zr - 1.08) / 0.56
    for y_sign in (1, -1):
        pts.append(np.stack([g, y_sign * gw, np.full_like(g, 1.08)], axis=1))
        pts.append(np.stack([g, y_sign * top_w, zr], axis=1))
    # Roof rails and mirrors (boxes).
    for x_rng, y_rng, z_rng in (((-2.0, 0.25), (0.53, 0.59), (1.64, 1.667)),
                                ((0.85, 1.05), (0.88, 1.01), (0.98, 1.16))):
        for y_sign in (1, -1):
            for x in x_rng:
                for y in y_rng:
                    for z in z_rng:
                        pts.append(np.array([[x, y_sign * y, z]]))
    return np.concatenate(pts, axis=0)


_REF_BODY = _ref_body_points()


@dataclass(frozen=True)
class CarShape:
    """Car dimensions in metres (width without mirrors)."""

    length: float = REF_LENGTH
    width: float = REF_WIDTH
    height: float = REF_HEIGHT
    wheelbase: float = REF_WHEELBASE
    name: str = "Mercedes-Benz EQB 2024"
    body: str | None = None  # suv | hatch | sedan (cars.json "body"); None = guess from the dimensions

    def __post_init__(self):
        for key in ("length", "width", "height", "wheelbase"):
            v = getattr(self, key)
            if not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"car {key} must be a positive number of metres, got {v!r}")
        if self.wheelbase >= self.length:
            raise ValueError("car wheelbase must be shorter than its length")
        if not 1.0 <= self.length <= 20.0 or not 0.5 <= self.height <= 6.0:
            raise ValueError(f"car dimensions are in metres here; got length {self.length}, height {self.height}")

    @property
    def front_tip(self) -> float:
        return self.wheelbase / 2 + FRONT_OVERHANG_SHARE * (self.length - self.wheelbase)

    @property
    def rear_tip(self) -> float:
        return self.front_tip - self.length

    @property
    def half_track(self) -> float:
        return REF_HALF_TRACK * self.width / REF_WIDTH

    @property
    def tyre_radius(self) -> float:
        return REF_TYRE_R * self.height / REF_HEIGHT

    def map_x(self, x_ref: np.ndarray) -> np.ndarray:
        """Reference X -> this car's X: overhangs and wheelbase stretch separately."""
        ref = [REF_XR, -REF_WHEELBASE / 2, REF_WHEELBASE / 2, REF_XF]
        new = [self.rear_tip, -self.wheelbase / 2, self.wheelbase / 2, self.front_tip]
        return np.interp(x_ref, ref, new)

    def points(self) -> np.ndarray:
        """(N, 3) vertices of the proxy's convex pieces; silhouette extremes occur at vertices."""
        body = _REF_BODY.copy()
        body[:, 0] = self.map_x(body[:, 0])
        body[:, 1] *= self.width / REF_WIDTH
        body[:, 2] *= self.height / REF_HEIGHT
        r = self.tyre_radius
        half_tw = 0.5 * REF_TYRE_W * self.width / REF_WIDTH
        phi = np.linspace(0, 2 * np.pi, TYRE_POINTS, endpoint=False)
        tyres = []
        for cx in (self.wheelbase / 2, -self.wheelbase / 2):
            for cy in (self.half_track, -self.half_track):
                for y in (cy - half_tw, cy + half_tw):
                    tyres.append(np.stack([cx + r * np.cos(phi), np.full_like(phi, y), r + r * np.sin(phi)], axis=1))
        return np.concatenate([body] + tyres, axis=0)

    def contacts(self) -> np.ndarray:
        """(4, 3) tyre contact points: front-left, front-right, rear-left, rear-right."""
        a, t = self.wheelbase / 2, self.half_track
        return np.array([[a, t, 0.0], [a, -t, 0.0], [-a, t, 0.0], [-a, -t, 0.0]])

    def footprint(self, grow: float = 0.0, corner_r: float = FOOTPRINT_CORNER_R, n_arc: int = 8) -> np.ndarray:
        """(M, 2) polygon of the body's footprint (X, Y), rounded corners, grown (or inset if < 0)."""
        hx_f, hx_r = self.front_tip + grow, self.rear_tip - grow
        hy = self.width / 2 + grow
        r = max(0.01, min(corner_r * self.width / REF_WIDTH + grow, hy - 0.01, (hx_f - hx_r) / 2 - 0.01))
        pts = []
        corners = ((hx_f - r, hy - r, 0.0), (hx_r + r, hy - r, 90.0), (hx_r + r, -hy + r, 180.0), (hx_f - r, -hy + r, 270.0))
        for cx, cy, start in corners:
            for a in np.linspace(start, start + 90.0, n_arc):
                pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
        return np.array(pts)


@dataclass(frozen=True)
class ViewRow:
    """The proxy seen through V at one yaw, in anchor metres (px = anchor + px_per_m * value)."""

    yaw: float
    top: float  # roof row (negative = above the anchor)
    low: float  # lowest row
    x0: float
    x1: float
    low_x: float  # column of the lowest point (near tyre or bumper)
    contacts_du: np.ndarray  # (4,) contact columns
    contacts_dv: np.ndarray  # (4,) contact rows
    contacts_scale: np.ndarray  # (4,) local scale at each contact relative to the anchor (D / depth)
    contacts_visible: np.ndarray  # (4,) bool

    @property
    def h(self) -> float:
        return self.low - self.top

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def aspect(self) -> float:
        return self.w / self.h


def _wheel_visible(yaw_deg: float) -> np.ndarray:
    """Which contacts (FL, FR, RL, RR) the camera can see: their side or their end faces it."""
    th = math.radians(yaw_deg)
    s, c = math.sin(th), math.cos(th)
    right_faces, left_faces = s > 0.17, s < -0.17
    front_faces, rear_faces = c > 0.17, c < -0.17
    return np.array([left_faces or front_faces, right_faces or front_faces,
                     left_faces or rear_faces, right_faces or rear_faces])


class CarTable:
    """The proxy projected through V for every yaw (0.5 degree grid), interpolated in between."""

    def __init__(self, shape: CarShape | None = None, step_deg: float = 0.5,
                 distance: float = CAPTURE_DISTANCE_M, elevation_deg: float = CAPTURE_ELEVATION_DEG):
        self.shape = shape or CarShape()
        self.distance = distance
        self.elevation = math.radians(elevation_deg)
        self.step = step_deg
        self.yaws = np.arange(0.0, 360.0, step_deg)
        du, dv, _ = self.project(self.shape.points(), self.yaws)
        self._top, self._low = dv.min(axis=1), dv.max(axis=1)
        self._x0, self._x1 = du.min(axis=1), du.max(axis=1)
        self._low_x = du[np.arange(len(self.yaws)), dv.argmax(axis=1)]  # column of the lowest point
        cdu, cdv, cdep = self.project(self.shape.contacts(), self.yaws)
        self._cdu, self._cdv, self._cscale = cdu, cdv, distance / cdep

    def project(self, pts: np.ndarray, yaws_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Car points (N, 3) at yaws (Y,) -> (du, dv, depth), each (Y, N), in anchor metres."""
        th = np.radians(np.atleast_1d(np.asarray(yaws_deg, dtype=np.float64)))[:, None]
        X, Y, Z = (pts[:, i][None, :] for i in range(3))
        cd, sd = math.cos(self.elevation), math.sin(self.elevation)
        s = -X * np.cos(th) + Y * np.sin(th)  # metres beyond the footprint centre, away from V
        t = X * np.sin(th) + Y * np.cos(th)  # metres to the right
        up = Z * cd + s * sd
        dep = self.distance + s * cd - Z * sd
        return self.distance * t / dep, -self.distance * up / dep, dep

    def at(self, yaw_deg: float) -> ViewRow:
        y = float(yaw_deg) % 360.0
        i = y / self.step
        i0 = int(math.floor(i)) % len(self.yaws)
        i1 = (i0 + 1) % len(self.yaws)
        f = i - math.floor(i)

        def lerp(a):
            return (1 - f) * a[i0] + f * a[i1]

        return ViewRow(yaw=y, top=float(lerp(self._top)), low=float(lerp(self._low)),
                       x0=float(lerp(self._x0)), x1=float(lerp(self._x1)), low_x=float(lerp(self._low_x)),
                       contacts_du=lerp(self._cdu), contacts_dv=lerp(self._cdv),
                       contacts_scale=lerp(self._cscale), contacts_visible=_wheel_visible(y))

    def side_length(self) -> float:
        """Silhouette length of a pure side view (yaw 90), in anchor metres (EQB: 5.809)."""
        return self.at(90.0).w

    def cx_smoothed(self, yaw_deg: float, sigma_deg: float) -> float:
        """Expected box-centre offset when the yaw is only known to +-sigma (Gaussian-weighted)."""
        if sigma_deg <= 0:
            return self.at(yaw_deg).cx
        offs = np.arange(-2 * sigma_deg, 2 * sigma_deg + 1e-9, self.step)
        g = np.exp(-(offs ** 2) / (2 * sigma_deg ** 2))
        cx = np.array([self.at(yaw_deg + o).cx for o in offs])
        return float((g * cx).sum() / g.sum())

    # --- the floor as V sees it (s = metres beyond the footprint centre, along V's view) ---

    def focal(self, px_per_m: float) -> float:
        return px_per_m * self.distance

    def floor_row(self, s, anchor_y: float, px_per_m: float):
        cd, sd = math.cos(self.elevation), math.sin(self.elevation)
        s = np.asarray(s, dtype=np.float64)
        return anchor_y - self.focal(px_per_m) * s * sd / (self.distance + s * cd)

    def floor_depth(self, v, anchor_y: float, px_per_m: float):
        cd, sd = math.cos(self.elevation), math.sin(self.elevation)
        k = (anchor_y - np.asarray(v, dtype=np.float64)) / (self.focal(px_per_m) * sd)
        return k * self.distance / (1 - k * cd)

    def floor_homography(self, yaw_deg: float, anchor: tuple[float, float], px_per_m: float) -> np.ndarray:
        """3x3 map from the car's floor plane (X, Y, 1) in metres to image px, through V."""
        ax, ay = anchor
        f, d = self.focal(px_per_m), self.distance
        cd, sd = math.cos(self.elevation), math.sin(self.elevation)
        h_v = np.array([[f, ax * cd, ax * d], [0.0, ay * cd - f * sd, ay * d], [0.0, cd, d]])
        th = math.radians(yaw_deg)
        rot = np.array([[math.sin(th), math.cos(th), 0.0], [-math.cos(th), math.sin(th), 0.0], [0.0, 0.0, 1.0]])
        return h_v @ rot
