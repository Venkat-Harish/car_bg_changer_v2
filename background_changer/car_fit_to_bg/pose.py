"""The car's pose in its photo, from the silhouette: yaw, distance, camera height, aim and roll.

A 3D proxy of the car (cars.json dimensions + a body preset) is drawn as convex pieces through the photo's
pinhole camera (focal from the background lines, principal point = the frame centre) and compared with the
cut-out mask (IoU). Unknowns: yaw, camera roll, distance D to the footprint centre, camera height h (so the
angle down to the car is asin(h / D)) and the aim (the pixel where the footprint centre appears).

Search, per photo, with no hand seeds (it was tested on 34 frames without a front/back flip):
  1. sweep: 36 yaws at h = 1.5 m, each a short fit from a box-aligned start (camera outside the car);
  2. candidates: the best IoU modes, the prior's best yaw, and their 180-degree partners;
  3. refine each (yaw, roll, D, h in [1.0, 2.2] m, aim) and score it in nats:
       score = -IoU / 0.02 - classifier log-prior + red-tail-lamp penalties;
  4. a final refit of the winner with the background lines' roll and pitch as priors (sigma 3 degrees);
  5. flags: yaw-unsure (margin < 2 nats; the label is used when it fits as well, decision 3),
     flip-ambiguous, fit-failed (IoU < 0.75 or an implausible camera height: placed with the label's yaw),
     and residual blobs (merged neighbour car, merged object, reflection, occluder).
Conventions: car frame X forward, Y left, Z up, origin = footprint centre (bg_changer.carmodel); yaw 0 =
the front faces the camera, 90 = its right side; camera frame x right, y down, z forward.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import minimize

from bg_changer.carmodel import (REF_HEIGHT, REF_TYRE_W, REF_WIDTH, REF_XF, REF_XR, SECTION_STEP, CarShape, CarTable,
                                 _ref_half_width, _wheel_visible)
from bg_changer.probe import LABELS
from bg_changer.yaw import BIN_DEG

TAU = 0.02  # IoU per nat
ZN = 0.08  # m: near plane for drawing the proxy (close, cut frames put parts of the car behind the camera)
H_SWEEP, H_LIMITS = 1.5, (1.0, 2.2)  # camera height (m) for the sweep / allowed during refinement
FAIL_IOU, WEAK_IOU, UNSURE_NATS = 0.75, 0.90, 2.0
H_PLAUSIBLE = (0.3, 3.0)
H_LIMITS_STRONG = (0.5, 3.0)  # with strong lines the pitch prior pins the height, so a wider range is safe
H_DEFAULT_PRIOR = (1.5, 0.25)  # weak lines and no folder to learn the camera height from
SWEEP_LONG, REFINE_LONG = 320, 480  # px: the car's long side in the working rasters

# ---------------------------------------------------------------- proxy: convex pieces of CarShape.points()
_br = [1.95, -1.95, 2.15, 1.10, REF_XF - 0.40, REF_XR + 0.30]
XS = np.unique(np.r_[np.linspace(REF_XR, REF_XF, int(math.ceil((REF_XF - REF_XR) / SECTION_STEP)) + 1),
                     [b + d for b in _br for d in (-1e-4, 1e-4)]])
XS = XS[(XS >= REF_XR) & (XS <= REF_XF)]
G = XS[(XS >= -2.32) & (XS <= 1.10)]
N, NG = len(XS), len(G)
B0 = 4 * N + 4 * NG
PIECES = ([np.array([b + k * m + j for k in range(4) for j in (i, i + 1)]) for b, m in ((0, N), (4 * N, NG))
           for i in range(m - 1)] + [np.arange(B0 + 8 * q, B0 + 8 * q + 8) for q in range(4)]
          + [np.arange(B0 + 32 + 96 * q, B0 + 128 + 96 * q) for q in range(4)])
TYRE_PIECE = [len(PIECES) - 4 + i for i in range(4)]  # FL, FR, RL, RR (CarShape.contacts order)
ROOF = {"hatch": ([-2.32, -1.90, -0.90, 0.10, 1.10], [1.12, 1.50, 1.64, 1.64, 1.08]),  # (x, roof z), reference m
        "sedan": ([-2.32, -1.75, -1.20, -0.80, 0.10, 1.10], [1.08, 1.12, 1.55, 1.64, 1.64, 1.08])}
BODIES = ("suv", "hatch", "sedan")


def body_points(shape: CarShape, body: str = "suv") -> np.ndarray:
    """CarShape.points() with a body preset's roof line (same vertex layout; 'suv' = the EQB proxy)."""
    pts = shape.points().copy()
    if body in ROOF:
        zr = np.interp(G, *ROOF[body])
        tw = np.minimum(_ref_half_width(G), 0.84 - 0.20 * (np.clip(zr, 1.08, 1.64) - 1.08) / 0.56)
        for blk, sgn in ((1, 1), (3, -1)):  # greenhouse top rows, left / right
            i = 4 * N + blk * NG
            pts[i:i + NG, 2] = zr * shape.height / REF_HEIGHT
            pts[i:i + NG, 1] = sgn * tw * shape.width / REF_WIDTH
        r = slice(B0, B0 + 16)  # no roof rails
        pts[r, 2] = np.minimum(pts[r, 2], 1.60 * shape.height / REF_HEIGHT)
        pts[r, 0] = np.clip(pts[r, 0], *shape.map_x(np.array([-0.9, 0.10])))
    return pts


def guess_body(shape: CarShape) -> str:
    if shape.height >= 1.60:
        return "suv"
    return "hatch" if shape.length < 4.10 else "sedan"


def body_ambiguous(shape: CarShape) -> bool:
    return shape.height < 1.60 and shape.length < 4.30


# ---------------------------------------------------------------- the camera model
def mv(yaw: float, delta: float) -> np.ndarray:
    """Car (X fwd, Y left, Z up) -> V's camera axes (x right, y down, z fwd); as CarTable.project."""
    th, dl = math.radians(yaw), math.radians(delta)
    st, ct, sd, cd = math.sin(th), math.cos(th), math.sin(dl), math.cos(dl)
    return np.array([[st, ct, 0.0], [ct * sd, -st * sd, -cd], [-ct * cd, st * cd, -sd]])


def camera_rotation(aim, c, f, roll, delta) -> np.ndarray:
    """V's camera frame -> the photo's: the footprint centre's ray goes through pixel `aim`; the car's
    vertical shows `roll` degrees off upright at the image centre (0 = level)."""
    a = np.array([aim[0] - c[0], aim[1] - c[1], f], float)
    a /= np.linalg.norm(a)
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    A, B = a[0] * sr + a[1] * cr, a[2]
    phi = math.asin(max(-1.0, min(1.0, math.sin(math.radians(delta)) / math.hypot(A, B)))) - math.atan2(A, B)
    g = np.array([sr * math.cos(phi), cr * math.cos(phi), math.sin(phi)])  # the car's 'down' in the photo frame
    gp = g - (g @ a) * a
    gp /= np.linalg.norm(gp)
    return np.stack([a, gp, np.cross(a, gp)], 1) @ np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])


def car_to_camera(yaw, D, delta, aim, roll, f, c) -> tuple[np.ndarray, np.ndarray]:
    """(R, t): a car point P maps to camera coordinates R @ P + t."""
    Rc = camera_rotation(aim, c, f, roll, delta)
    return Rc @ mv(yaw, delta), Rc @ np.array([0.0, 0.0, D])


def cam_angles(R: np.ndarray) -> tuple[float, float]:
    """(pitch, roll) in degrees of a car->camera rotation: pitch > 0 looks down at the car's floor."""
    g = -R[:, 2]  # the car's down in camera coordinates
    return math.degrees(math.asin(max(-1.0, min(1.0, g[2])))), math.degrees(math.atan2(g[0], g[1]))


def implied_pitch(aim, c, f, roll, delta) -> float:
    a = np.array([aim[0] - c[0], aim[1] - c[1], f], float)
    a /= np.linalg.norm(a)
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    A, B = a[0] * sr + a[1] * cr, a[2]
    return math.degrees(math.asin(max(-1.0, min(1.0, math.sin(math.radians(delta)) / math.hypot(A, B)))) - math.atan2(A, B))


def project(P: np.ndarray, R: np.ndarray, t: np.ndarray, f: float, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.atleast_2d(P) @ R.T + t
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return c[0] + f * q[:, 0] / z, c[1] + f * q[:, 1] / z, z


def render_proxy(Q: np.ndarray, f: float, c, wh) -> np.ndarray:
    """Anti-aliased silhouette (uint8) of the convex pieces given in camera coordinates Q (N, 3); pieces
    crossing the near plane z = ZN are clipped there."""
    img = np.zeros((wh[1], wh[0]), np.uint8)
    lim = 4e5
    for idx in PIECES:
        q = Q[idx]
        fr = q[:, 2] >= ZN
        if not fr.any():
            continue
        if not fr.all():
            a, b = q[fr], q[~fr]
            t = (ZN - a[:, 2][:, None]) / (b[:, 2][None, :] - a[:, 2][:, None])
            q = np.vstack([a, (a[:, None, :] + t[..., None] * (b[None, :, :] - a[:, None, :])).reshape(-1, 3)])
        P = np.round(np.clip(np.stack([c[0] + f * q[:, 0] / q[:, 2], c[1] + f * q[:, 1] / q[:, 2]], 1), -lim, lim) * 16)
        cv2.fillConvexPoly(img, cv2.convexHull(P.astype(np.int32)), 255, cv2.LINE_AA, 4)
    return img


def iou(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    if keep is not None:
        a, b = a * keep, b * keep
    return float(np.minimum(a, b).sum() / max(1.0, np.maximum(a, b).sum()))


def ext_view(shape: CarShape, yaw: float) -> float:
    t = math.radians(yaw)
    return abs(math.cos(t)) * shape.length / 2 + abs(math.sin(t)) * shape.width / 2


def wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- hints: classifier, label, red tail lamps
def label_logprior(probs: dict | None, yaws, sigma: float = 25.0, floor: float = 0.05) -> np.ndarray:
    """log of a von Mises mixture over the 8 view bins (classifier probabilities); 0 when unknown."""
    yaws = np.atleast_1d(np.asarray(yaws, float))
    if not probs:
        return np.zeros(len(yaws))
    k = 1 / math.radians(sigma) ** 2
    return np.log(floor + sum(p * np.exp(k * (np.cos(np.radians(yaws - BIN_DEG[lab])) - 1))
                              for lab, p in probs.items() if lab in BIN_DEG))


def read_view_results(path: Path) -> dict[str, dict]:
    """frame_view_type's results.csv -> {file stem: {'probs': {label: p}, 'angle': deg, 'time': s}}."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                probs = {lab: float(r[f"prob_{lab}"]) for lab in LABELS if r.get(f"prob_{lab}") not in (None, "")}
                out[Path(r["file"]).stem] = dict(probs=probs, angle=float(r.get("angle_deg") or "nan"),
                                                 time=float(r.get("time_s") or "nan"))
            except (KeyError, ValueError):
                continue
    return out


def probs_from_label(label: str | None, conf_pct: float | None) -> dict | None:
    if label not in BIN_DEG:
        return None
    p = min(0.95, max(0.3, (conf_pct or 60) / 100))
    out = {lab: (1 - p) / 7 for lab in BIN_DEG}
    out[label] = p
    return out


@dataclass
class RedCue:
    frac: float = 0.0  # red share of the car's pixels
    end: int = 0  # -1 red only at the left end, +1 right end, 0 unknown
    body: bool = False  # the car itself is red: the cue says nothing


def red_cue(frame: np.ndarray, alpha: np.ndarray) -> RedCue:
    """Red tail lamps on the source frame's colours (Lab a* > 25 inside the car), car scaled to 420 px tall."""
    M = alpha >= 0.5
    ys, xs = np.nonzero(M)
    if len(xs) < 50:
        return RedCue()
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    sc = min(1.0, 420.0 / (y2 - y1))
    bgr = cv2.resize(frame[y1:y2, x1:x2], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    m = (cv2.resize(M[y1:y2, x1:x2].astype(np.float32), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA) >= 0.5)
    m = m.astype(np.uint8)
    n, lab, stt, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return RedCue()
    m = (lab == 1 + np.argmax(stt[1:, 4])).astype(np.uint8)
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1].astype(int) - 128
    red = (a > 25) & (m > 0)
    frac = float(red.sum() / max(1, m.sum()))
    mx = np.nonzero(m)[1]
    xa, xb = mx.min(), mx.max()
    end = 0
    if red.sum() >= 0.002 * m.sum():
        rx = np.nonzero(red)[1]
        fl, frr = (rx < xa + 0.3 * (xb - xa)).mean(), (rx > xb - 0.3 * (xb - xa)).mean()
        end = -1 if fl > 0.8 else (1 if frr > 0.8 else 0)
    return RedCue(frac=frac, end=end, body=frac > 0.15)


def red_penalty(cue: RedCue, shape: CarShape, yaw: float, R, t, f, c, car_w_px: float) -> float:
    """Nats against a pose: red seen and the front facing the camera (+3); red at the end that is not the
    predicted rear (+4); no red at all while the rear faces the camera (+1.5)."""
    if cue.body:
        return 0.0
    toward = math.cos(math.radians(yaw))  # > 0.5: the front faces the camera
    pen = 0.0
    if cue.frac >= 0.002:
        if toward > 0.5:
            pen += 3.0
        if cue.end != 0:
            pts = np.array([[shape.front_tip, 0.0, 0.5 * shape.height], [shape.rear_tip, 0.0, 0.5 * shape.height]])
            u, _, z = project(pts, R, t, f, c)
            if np.all(z > ZN):
                dx = u[1] - u[0]  # rear minus front
                if cue.end * dx < 0 and abs(dx) > 0.05 * car_w_px:
                    pen += 4.0
    elif cue.frac < 0.0005 and toward < -0.5:
        pen += 1.5
    return pen


# ---------------------------------------------------------------- working rasters
@dataclass
class Work:
    """The car mask on a car-centred window of the frame, scaled; principal point and focal in its px."""

    mask: np.ndarray  # uint8 0..255
    s: float  # working px per frame px
    origin: tuple[float, float]  # the window's top-left in frame px
    c: tuple[float, float]
    f: float
    cut: frozenset

    def to_frame(self, p):
        return ((p[0] + 0.5) / self.s - 0.5 + self.origin[0], (p[1] + 0.5) / self.s - 0.5 + self.origin[1])

    def from_frame(self, p):
        return ((p[0] - self.origin[0] + 0.5) * self.s - 0.5, (p[1] - self.origin[1] + 0.5) * self.s - 0.5)


def make_work(alpha: np.ndarray, cut, f: float, long_px: int, pad: float = 0.30) -> Work:
    H, W = alpha.shape
    ys, xs = np.nonzero(alpha >= 0.5)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = x2 - x1, y2 - y1
    wx0, wy0 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    wx1, wy1 = min(W, int(math.ceil(x2 + pad * bw))), min(H, int(math.ceil(y2 + pad * bh)))
    s = long_px / max(bw, bh)
    size = (max(8, int(round((wx1 - wx0) * s))), max(8, int(round((wy1 - wy0) * s))))
    m = cv2.resize(alpha[wy0:wy1, wx0:wx1].astype(np.float32), size, interpolation=cv2.INTER_AREA)
    mask = np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)
    c_full = ((W - 1) / 2, (H - 1) / 2)
    work = Work(mask=mask, s=s, origin=(wx0, wy0), c=(0.0, 0.0), f=f * s, cut=frozenset(cut))
    work.c = work.from_frame(c_full)
    return work


# ---------------------------------------------------------------- one local fit
def _start(work: Work, pts, shape, yaw, h):
    """Box-aligned start: the distance whose V-projection, scaled to the mask box, matches f / D; the camera
    stays outside the car."""
    m, cut, f = work.mask, work.cut, work.f
    ys, xs = np.nonzero(m > 127)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    best = None
    for D in np.geomspace(max(math.hypot(h, ext_view(shape, yaw) + 0.25), h / 0.95), 40.0, 48):
        du, dv, _ = CarTable.project(SimpleNamespace(distance=D, elevation=math.asin(h / D)), pts, [yaw])
        du, dv = du[0], dv[0]
        k = (y2 - y1) / max(np.ptp(dv), 1e-6) if not {"top", "bottom"} & cut else (x2 - x1) / max(np.ptp(du), 1e-6)
        e = abs(math.log(f / k / D))
        if best is None or e < best[0]:
            best = (e, D, k, du, dv)
    _, D, k, du, dv = best
    side = cut & {"left", "right"}
    if side == {"right"}:
        ax = x1 - k * du.min()
    elif side == {"left"}:
        ax = x2 - k * du.max()
    else:
        ax = (x1 + x2) / 2 - k * (du.min() + du.max()) / 2
    ay = y2 - k * dv.max() if "bottom" not in cut else y1 - k * dv.min()
    return D, (float(ax), float(ay))


@dataclass
class Pose:
    yaw: float
    roll: float
    D: float
    h: float
    delta: float
    aim: tuple[float, float]  # working px
    iou: float
    e: float = 0.0
    pitch: float = 0.0

    def rt(self, work: Work):
        return car_to_camera(self.yaw, self.D, self.delta, self.aim, self.roll, work.f, work.c)


def fit_local(work: Work, pts, shape, yaw0, h0=H_SWEEP, D0=None, aim0=None, roll0=0.0, roll=None, fix_yaw=False,
              fix_h=False, n_eval=300, hlim=H_LIMITS, pitch=None, sp=3.0, rollp=None, hprior=None,
              keep=None) -> Pose:
    """Nelder-Mead on IoU over yaw, roll (unless given), log D, camera height h and the aim, at a fixed focal.
    pitch: line pitch (deg) -> Gaussian prior (sigma sp) on the camera pitch implied by aim, roll and the
    angle down; rollp: (roll, sigma) prior; hprior: (h0, sigma) prior; keep: pixels that count in the IoU."""
    m, f, c, wh = work.mask, work.f, work.c, (work.mask.shape[1], work.mask.shape[0])
    m_sum = int(m.sum(dtype=np.int64))

    def fast_iou(r: np.ndarray) -> float:  # soft IoU on uint8 rasters: sum(min) / (sum(a) + sum(b) - sum(min))
        if keep is not None:
            return iou(r, m, keep)
        inter = int(np.minimum(r, m).sum(dtype=np.int64))
        return inter / max(1.0, int(r.sum(dtype=np.int64)) + m_sum - inter)

    if D0 is None:
        D0, aim0 = _start(work, pts, shape, yaw0, h0)
    x_all = np.array([yaw0, roll if roll is not None else roll0, math.log(max(D0, h0 / 0.9)), h0, *aim0], float)
    free = np.array([not fix_yaw, roll is None, True, not fix_h, True, True], bool)

    def full(p):
        x = x_all.copy()
        x[free] = p
        return x

    def energy(x):
        yaw, rl, lD, h, ax, ay = x
        D = math.exp(lD)
        if not ((fix_h or hlim[0] <= h <= hlim[1]) and h / 0.97 < D < 80 and abs(rl) < 30):
            return 1.0, 0.0, 0.0
        need = ext_view(shape, yaw) + 0.15
        dh = math.sqrt(max(D * D - h * h, 0.0))
        if dh < need:
            return 1.0 + need - dh, 0.0, 0.0
        dl = math.degrees(math.asin(h / D))
        R, t = car_to_camera(yaw, D, dl, (ax, ay), rl, f, c)
        I = fast_iou(render_proxy(pts @ R.T + t, f, c, wh))
        ip = implied_pitch((ax, ay), c, f, rl, dl)
        e = -I
        if pitch is not None:
            e += TAU * 0.5 * ((ip - pitch) / sp) ** 2
        if rollp is not None:
            e += TAU * 0.5 * ((rl - rollp[0]) / rollp[1]) ** 2
        if hprior is not None:
            e += TAU * 0.5 * ((h - hprior[0]) / hprior[1]) ** 2
        return e, I, ip

    x0 = x_all[free]
    st = np.array([3.0, 1.0, 0.03, 0.08, 1.5, 1.5])[free]
    r = minimize(lambda p: energy(full(p))[0], x0, method="Nelder-Mead",
                 options=dict(initial_simplex=np.vstack([x0, x0 + np.diag(st)]), maxfev=n_eval, xatol=1e-3, fatol=1e-5))
    x = full(r.x)
    e, I, ip = energy(x)
    yaw, rl, lD, h, ax, ay = x
    D = math.exp(lD)
    return Pose(yaw=float(yaw % 360), roll=float(rl), D=float(D), h=float(h),
                delta=float(math.degrees(math.asin(min(1.0, h / D)))), aim=(float(ax), float(ay)), iou=float(I),
                e=float(e), pitch=float(ip))


def rescale_pose(p: Pose, a: Work, b: Work) -> Pose:
    """The same pose expressed in another working raster (same frame, other window/scale)."""
    fx, fy = a.to_frame(p.aim)
    return Pose(**{**p.__dict__, "aim": b.from_frame((fx, fy))})


# ---------------------------------------------------------------- residual blobs
def residual_blobs(work: Work, mask_proxy: np.ndarray, top_row: float, bottom_row: float, x_range) -> list[dict]:
    """Connected leftovers between the mask and the fitted proxy (each > 1% of the car's area), classified."""
    m = work.mask >= 128
    pr = mask_proxy >= 128
    area = max(1, m.sum())
    k = max(3, int(round(0.02 * (bottom_row - top_row))) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for kind_src, img in (("extra", m & ~(cv2.dilate(pr.astype(np.uint8), ker) > 0)),
                          ("missing", pr & ~(cv2.dilate(m.astype(np.uint8), ker) > 0))):
        n, lab, st, cen = cv2.connectedComponentsWithStats(img.astype(np.uint8), 8)
        for i in range(1, n):
            share = st[i, cv2.CC_STAT_AREA] / area
            if share < 0.01:
                continue
            cx, cy = cen[i]
            if kind_src == "missing":
                kind = "occluder"
            elif cy < top_row + 0.15 * (bottom_row - top_row):
                kind = "merged-object"
            elif cy > bottom_row - 0.02 * (bottom_row - top_row):
                kind = "reflection"
            elif cx < x_range[0] or cx > x_range[1]:
                kind = "neighbour-car"
            else:
                kind = "merged-object"
            out.append(dict(kind=kind, share=float(share), label=i, src=kind_src, mask=(lab == i)))
    return out


# ---------------------------------------------------------------- the per-photo search
@dataclass
class PoseFit:
    yaw: float
    roll: float  # camera roll relative to the car's floor (deg)
    D: float  # m, camera to footprint centre
    h: float  # m, camera height above the car's floor
    delta: float  # deg, angle down to the footprint centre
    aim: tuple[float, float]  # frame px where the footprint centre appears
    pitch: float  # deg, camera pitch relative to the car's floor
    iou: float
    margin: float  # nats to the best flipped pose (yaw + 180 +- 45)
    f: float  # focal (frame px) used
    body: str
    yaw_source: str  # fit | label | label-fallback
    flags: list[str] = field(default_factory=list)
    failed: bool = False
    alternatives: list[tuple[float, float, float]] = field(default_factory=list)  # (yaw, iou, score)
    blobs: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    work: Work | None = None  # the refinement raster (for overlays)
    sweep: list[tuple[float, float, float]] = field(default_factory=list)  # (yaw, iou, score) every 10 deg (debug)
    candidates: list[Pose] = field(default_factory=list)  # the refined candidates on `work`, as `alternatives`

    def rt(self, c_frame) -> tuple[np.ndarray, np.ndarray]:
        """(R, t) car -> photo camera at full frame resolution (principal point c_frame)."""
        return car_to_camera(self.yaw, self.D, self.delta, self.aim, self.roll, self.f, c_frame)


def _local_maxima(C: np.ndarray) -> list[int]:
    n = len(C)
    return [j for j in range(n) if C[j] >= C[j - 1] and C[j] >= C[(j + 1) % n]]


def fit_pose(alpha: np.ndarray, cut, shape: CarShape, f: float, body: str, probs: dict | None = None,
             cue: RedCue | None = None, line_roll: float | None = None, line_pitch: float | None = None,
             strong_lines: bool = True, h_prior: tuple[float, float] | None = None, label: str | None = None,
             max_seconds: float = 120.0, alt_body: str | None = None) -> PoseFit:
    """Fit the car's pose to its full-frame alpha (H x W) at focal f (frame px). probs: classifier (or
    label) probabilities by view bin; cue: red tail lamps; line roll/pitch: priors from background lines
    (strong_lines False -> h_prior is used and roll is left to the silhouette)."""
    t0 = time.perf_counter()
    cut = frozenset(cut)
    cue = cue or RedCue()
    pts = body_points(shape, body)
    w1 = make_work(alpha, cut, f, SWEEP_LONG)
    w2 = make_work(alpha, cut, f, REFINE_LONG)
    car_w2 = float(np.ptp(np.nonzero(w2.mask > 127)[1]) + 1)
    flags: list[str] = []

    # Priors from the background lines (strong) or a camera-height prior (weak lines); used throughout, so
    # every candidate is judged by the same energy.
    if strong_lines and line_roll is not None:
        prior_kw = dict(pitch=line_pitch, sp=3.0, rollp=(line_roll, 3.0), hlim=H_LIMITS_STRONG)
    else:
        prior_kw = dict(hprior=h_prior or H_DEFAULT_PRIOR)

    def score(p: Pose, work: Work, cw: float) -> float:
        R, t = p.rt(work)
        return p.e / TAU - float(label_logprior(probs, [p.yaw])[0]) + red_penalty(cue, shape, p.yaw, R, t, work.f, work.c, cw)

    # 1. sweep (roll fixed to the lines' roll when they are strong)
    yaws = np.arange(0.0, 360.0, 10.0)
    sweep_roll = line_roll if (strong_lines and line_roll is not None) else None
    sweep = [fit_local(w1, pts, shape, y, h0=H_SWEEP, fix_yaw=True, fix_h=True, n_eval=40, roll=sweep_roll,
                       roll0=sweep_roll or 0.0) for y in yaws]
    C = np.array([p.iou for p in sweep])
    car_w1 = float(np.ptp(np.nonzero(w1.mask > 127)[1]) + 1)
    U = np.array([score(p, w1, car_w1) for p in sweep])

    # 2. candidates: best IoU modes, the prior-informed best, and their 180-degree partners
    cand = sorted(_local_maxima(C), key=lambda j: -C[j])[:2]
    cand.append(int(np.argmin(U)))
    for j in list(cand):
        near = np.flatnonzero(np.abs(wrap(yaws - yaws[j] - 180)) <= 40)
        cand.append(int(near[np.argmin(U[near])]))
    order = []
    for j in sorted(set(cand), key=lambda j: U[j]):
        if all(abs(wrap(yaws[j] - yaws[k])) > 15 for k in order):
            order.append(j)
    order = order[:5]

    # 3. refine (same priors for every candidate)
    refined = []
    for n_c, j in enumerate(order):
        base = rescale_pose(sweep[j], w1, w2)
        starts = [dict(yaw0=base.yaw, D0=base.D, aim0=base.aim, roll0=base.roll)]
        if n_c == 0:
            starts.append(dict(yaw0=base.yaw + 8.0, roll0=base.roll))
        best = None
        for st in starts:
            q = fit_local(w2, pts, shape, n_eval=250, **st, **prior_kw)
            if best is None or q.e < best.e:
                best = q
        refined.append(best)
    if time.perf_counter() - t0 > max_seconds:  # reported only: the search itself never depends on the clock
        flags.append("fit-slow")
    scores = [score(q, w2, car_w2) for q in refined]
    i_best = int(np.argmin(scores))
    win = refined[i_best]
    flips = [i for i, q in enumerate(refined) if abs(wrap(q.yaw - win.yaw - 180)) <= 45]
    margin = float(min(scores[i] - scores[i_best] for i in flips)) if flips else 99.0  # nats to the flipped pose
    near_flip = [refined[i] for i in flips if abs(wrap(refined[i].yaw - win.yaw - 180)) <= 20]
    if near_flip and max(q.iou for q in near_flip) >= win.iou - 0.01:
        flags.append("flip-ambiguous")

    # 4. final polish of the winner with the same priors
    kw = dict(yaw0=win.yaw, D0=win.D, aim0=win.aim, roll0=win.roll, h0=min(2.15, max(1.05, win.h)), n_eval=300)
    final = fit_local(w2, pts, shape, **kw, **prior_kw)
    if final.e > win.e:
        final = win
    if alt_body is not None and alt_body != body:  # body preset guessed from the dimensions: try the other one
        pts_alt = body_points(shape, alt_body)
        alt = fit_local(w2, pts_alt, shape, **dict(kw, yaw0=final.yaw, D0=final.D, aim0=final.aim, roll0=final.roll),
                        **prior_kw)
        if alt.iou > final.iou + 0.01:
            final, body, pts = alt, alt_body, pts_alt
        elif alt.iou >= final.iou - 0.01:
            flags.append("body-ambiguous")
    yaw_source = "fit"

    # 5. unsure: the label decides when it fits about as well (decision 3)
    if margin < UNSURE_NATS:
        flags.append("yaw-unsure")
        if label in BIN_DEG:
            lp = fit_local(w2, pts, shape, yaw0=BIN_DEG[label], n_eval=300, roll0=final.roll, **prior_kw)
            if lp.iou >= final.iou - 0.02:
                final, yaw_source = lp, "label"

    # residual blobs: merged neighbour cars / objects, reflections, occluders
    R, t = final.rt(w2)
    pr = render_proxy(pts @ R.T + t, w2.f, w2.c, (w2.mask.shape[1], w2.mask.shape[0]))
    rows = np.flatnonzero((pr > 127).any(1))
    colz = np.flatnonzero((pr > 127).any(0))
    blobs = []
    if len(rows) and len(colz):
        blobs = residual_blobs(w2, pr, rows.min(), rows.max(), (colz.min(), colz.max()))
        big = [b for b in blobs if b["share"] > 0.08 and b["src"] == "extra"]
        if big:
            ignore = np.zeros(w2.mask.shape, bool)
            for b in big:
                ignore |= b["mask"]
            k = max(3, int(round(0.03 * (rows.max() - rows.min()))) | 1)
            ignore = cv2.dilate(ignore.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
            kw2 = dict(kw, yaw0=final.yaw, D0=final.D, aim0=final.aim, roll0=final.roll)
            refit = fit_local(w2, pts, shape, keep=(~ignore).astype(np.float32), **kw2, **prior_kw)
            final = refit
        for b in blobs:
            if b["share"] > 0.03:
                tag = b["kind"] if b["kind"] != "reflection" else "mask-reflection"
                if tag not in flags:
                    flags.append(tag)

    hl = prior_kw.get("hlim", H_LIMITS)
    if not (hl[0] + 0.02 < final.h < hl[1] - 0.02):
        flags.append("camera-height-at-bound")
    failed = final.iou < FAIL_IOU or not (H_PLAUSIBLE[0] <= final.h <= H_PLAUSIBLE[1])
    if failed:
        flags.append("fit-failed")
        y = BIN_DEG[label] if label in BIN_DEG else final.yaw
        D, aim = _start(w2, pts, shape, y, H_SWEEP)
        final = Pose(yaw=y, roll=line_roll if (strong_lines and line_roll is not None) else final.roll, D=D, h=H_SWEEP,
                     delta=math.degrees(math.asin(H_SWEEP / D)), aim=aim, iou=final.iou)
        yaw_source = "label-fallback" if label in BIN_DEG else "fit"
    elif final.iou < WEAK_IOU:
        flags.append("weak-fit")

    R, t = final.rt(w2)
    pitch, _ = cam_angles(R)
    return PoseFit(yaw=final.yaw % 360, roll=final.roll, D=final.D, h=final.h, delta=final.delta,
                   aim=w2.to_frame(final.aim), pitch=pitch, iou=final.iou, margin=margin, f=f, body=body,
                   yaw_source=yaw_source, flags=flags, failed=failed,
                   alternatives=[(p.yaw, p.iou, s) for p, s in zip(refined, scores)], blobs=blobs,
                   seconds=time.perf_counter() - t0, work=w2,
                   sweep=[(float(y), float(i), float(u)) for y, i, u in zip(yaws, C, U)], candidates=list(refined))


def refit_with_roll(pf: PoseFit, shape: CarShape, roll: float, line_pitch: float | None = None) -> PoseFit:
    """The same photo refitted with the camera's roll (relative to the car's floor) fixed, e.g. after the car
    was found to stand on a cross-slope. Yaw, distance, height and aim re-adapt, so the proxy still matches
    the silhouette."""
    w = pf.work
    kw = dict(yaw0=pf.yaw, D0=pf.D, aim0=w.from_frame(pf.aim), roll=roll, roll0=roll,
              h0=min(H_LIMITS[1] - 0.05, max(H_LIMITS[0] + 0.05, pf.h)), n_eval=300)
    if line_pitch is not None:
        kw.update(pitch=line_pitch, sp=3.0)
    p = fit_local(w, body_points(shape, pf.body), shape, **kw)
    R, t = p.rt(w)
    pitch, _ = cam_angles(R)
    return PoseFit(yaw=p.yaw % 360, roll=p.roll, D=p.D, h=p.h, delta=p.delta, aim=w.to_frame(p.aim), pitch=pitch,
                   iou=p.iou, margin=pf.margin, f=pf.f, body=pf.body, yaw_source=pf.yaw_source, flags=list(pf.flags),
                   failed=pf.failed, alternatives=pf.alternatives, blobs=pf.blobs, seconds=pf.seconds, work=w,
                   sweep=pf.sweep, candidates=pf.candidates)


def tyre_names() -> tuple[str, ...]:
    return ("FL", "FR", "RL", "RR")


def tyre_width(shape: CarShape) -> float:
    return REF_TYRE_W * shape.width / REF_WIDTH


__all__ = ["BODIES", "PIECES", "Pose", "PoseFit", "RedCue", "TYRE_PIECE", "Work", "body_ambiguous", "body_points",
           "cam_angles", "camera_rotation", "car_to_camera", "fit_local", "fit_pose", "guess_body", "iou",
           "label_logprior", "make_work", "probs_from_label", "project", "read_view_results", "red_cue",
           "red_penalty", "render_proxy", "tyre_names", "tyre_width", "_wheel_visible"]
