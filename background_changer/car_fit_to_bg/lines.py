"""The photo's camera from its own background lines: roll, pitch and focal length (the car masked out).

Building edges, kerbs, walls and LED strips are straight and mostly vertical or horizontal. Their
vanishing points give the camera's tilt and, with a Manhattan scene, its focal length:
  1. LSD line segments, those on the car dropped;
  2. the vertical vanishing point by MSAC over near-vertical segments, refined by eigen-fit;
  3. the focal length from a search over Manhattan directions (skipped when f is given);
  4. a soft_l1 least-squares refit of roll, pitch, the horizontal direction and f, and bootstrap spreads.
Conventions: pitch > 0 looks down; roll > 0 = the horizon rises to the right. Principal point = the frame
centre. `conf` (roll/pitch) and `conf_f` (focal) are 0-1.

Focal choice for a run (`choose_focals`), in order: a number from the config; the folder's pooled focal
(>= 3 confident photos of one size, clustered within +-8%; a photo whose own confident focal is more than
20% off keeps it, flag lens-change); the photo's own confident focal; EXIF; a 70-degree prior.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

D2R, R2D = math.pi / 180.0, 180.0 / math.pi
POOL_CONF = 0.5  # conf_f a photo needs for its focal to count (alone or in the pool)
POOL_MIN = 3
POOL_TOL = math.log(1.08)  # +-8%: one lens's frames cluster this tightly
LENS_TOL = math.log(1.20)  # a photo's own confident focal this far from the pool = another lens or zoom
PRIOR_HFOV_LONG = 70.0  # degrees across the long side when nothing else is known
WEAK_FOCAL_CONF = 0.15  # below this a line focal is noise (EQB 0024: -48%), above it better than the prior
WEAK_LINES = 0.5  # conf below this: roll/pitch are only a soft hint
NEIGHBOUR_CONF, NEIGHBOUR_WINDOW_S, NEIGHBOUR_TAU_S = 0.3, 8.0, 4.0
VIDEO_NAME = re.compile(r"^(\d+(?:\.\d+)?)s_")  # frame_view_type stills: <time>s_<label>_<conf>pct

_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


@dataclass
class LineCamera:
    roll: float | None = None  # degrees; None = unknown
    pitch: float | None = None
    f: float | None = None  # px at the photo's full size (this solve's focal)
    conf: float = 0.0
    conf_f: float = 0.0
    sd_logf: float = 1.0
    n_vert: int = 0
    source: str = "lines"  # lines | neighbours | none
    flags: list[str] = field(default_factory=list)

    @property
    def strong(self) -> bool:
        return self.roll is not None and self.conf >= WEAK_LINES


# ---------------------------------------------------------------- vanishing-point machinery
def _segments(img, car_mask=None):
    H, W = img.shape[:2]
    S = max(W, H) / 2
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    s = _lsd().detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]
    s = np.zeros((0, 4)) if s is None else s.reshape(-1, 4).astype(float)
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= max(15, 0.05 * S)]
    if car_mask is not None and len(s):
        md = cv2.dilate((car_mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
        t = np.linspace(0, 1, 5)
        xi = np.clip((s[:, :1] * (1 - t) + s[:, 2:3] * t).round().astype(int), 0, W - 1)
        yi = np.clip((s[:, 1:2] * (1 - t) + s[:, 3:4] * t).round().astype(int), 0, H - 1)
        s = s[(md[yi, xi] > 0).mean(1) < 0.4]
    a, b = (s[:, :2] - c) / S, (s[:, 2:] - c) / S
    d = b - a
    L = np.linalg.norm(d, axis=1) * S
    ln = np.cross(np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))])
    with np.errstate(divide="ignore", invalid="ignore"):
        D = dict(a=a, b=b, mid=(a + b) / 2, dir=d / np.linalg.norm(d, axis=1, keepdims=True), L=L,
                 line=ln / np.linalg.norm(ln, axis=1, keepdims=True), th=np.clip(2 / np.maximum(L, 1e-9), 0.75 * D2R, 3 * D2R))
    return D, c, S


def _take(D, m):
    return {k: v[m] for k, v in D.items()}


def _err(D, V):
    V = np.atleast_2d(V)
    u = V[None, :, :2] - D["mid"][:, None, :] * V[None, :, 2:3]
    s = np.abs(D["dir"][:, None, 0] * u[..., 1] - D["dir"][:, None, 1] * u[..., 0]) / (np.linalg.norm(u, axis=2) + 1e-18)
    return np.arcsin(np.clip(s, 0, 1))


def _score(D, E):
    return (D["L"][:, None] * np.clip(1 - (E / D["th"][:, None]) ** 2, 0, None)).sum(0)


def _vert_ok(v):
    v = v / np.linalg.norm(v)
    if abs(v[2]) < 1e-9:
        q, r = (v[:2] if v[1] >= 0 else -v[:2]), np.inf
    else:
        q = v[:2] / v[2]
        r = np.hypot(*q)
    lim = 0.5
    if q[1] < 0:
        q, lim = -q, 3.0
    return abs(np.arctan2(q[0], q[1])) < 40 * D2R and r > lim


def _down(r, p):
    return np.array([np.sin(r) * np.cos(p), np.cos(r) * np.cos(p), np.sin(p)])


def _rp(v, f):
    d = np.array([v[0], v[1], v[2] * f])
    d = (d if d[1] >= 0 else -d) / np.linalg.norm(d)
    return np.arctan2(d[0], d[1]), np.arcsin(d[2])


def _vps(roll, pitch, alpha, f):
    d = _down(roll, pitch)
    e1 = np.array([1.0, 0, 0]) - d[0] * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    al = np.atleast_1d(alpha)

    def k(Q):
        return np.c_[f * Q[:, 0], f * Q[:, 1], Q[:, 2]]

    return (k(d[None]), k(np.cos(al)[:, None] * e1 + np.sin(al)[:, None] * e2),
            k(-np.sin(al)[:, None] * e1 + np.cos(al)[:, None] * e2))


def _vertical_vp(D, n_hyp=800):
    vc = np.abs(D["dir"][:, 1]) > np.cos(40 * D2R)
    Cd = _take(D, vc)
    if vc.sum() < 3:
        return None, vc
    rng = np.random.default_rng(0)
    p = Cd["L"] / Cd["L"].sum()
    idx = rng.choice(len(p), (n_hyp, 2), p=p)
    idx = idx[idx[:, 0] != idx[:, 1]]
    V = np.cross(Cd["line"][idx[:, 0]], Cd["line"][idx[:, 1]])
    V = V[np.array([_vert_ok(v) for v in V], bool)] if len(V) else V
    if not len(V):
        return None, vc
    v = V[np.argmax(_score(Cd, _err(Cd, V)))]
    for _ in range(3):
        m = _err(Cd, v)[:, 0] < Cd["th"]
        if m.sum() < 2:
            break
        Nn = np.cross(np.c_[Cd["a"][m], np.full(m.sum(), 1.2)], np.c_[Cd["b"][m], np.full(m.sum(), 1.2)])
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
        d = np.linalg.eigh((Nn * Cd["L"][m, None]).T @ Nn)[1][:, 0]
        v2 = np.array([d[0], d[1], d[2] / 1.2])
        if _vert_ok(v2):
            v = v2 / np.linalg.norm(v2)
    return v, vc


def _manhattan(v, Hs, fs, al):
    out = []
    for f in fs:
        _, V1, V2 = _vps(*_rp(v, f), al, f)
        sc = _score(Hs, np.minimum(_err(Hs, V1), _err(Hs, V2)))
        out.append((sc.max(), al[sc.argmax()]))
    return np.array(out)


def _refine(D, vc, x, fix_f):
    def labels(x):
        E = np.stack([_err(D, V)[:, 0] for V in _vps(x[0], x[1], x[2], np.exp(x[3]))], 1)
        E[~vc, 0] = np.inf
        lab = E.argmin(1)
        return np.where(E[np.arange(len(lab)), lab] < D["th"], lab, -1)

    def res(p, lab, Dd):
        q = np.r_[p, x[3]] if fix_f else p
        V = _vps(q[0], q[1], q[2], np.exp(q[3]))
        out = []
        for k in range(3):
            m = lab == k
            u = V[k][0, :2][None] - Dd["mid"][m] * V[k][0, 2]
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-18
            out.append((Dd["dir"][m, 0] * u[:, 1] - Dd["dir"][m, 1] * u[:, 0]) * Dd["L"][m] / 1.41)
        r = np.concatenate(out)
        return r if len(r) else np.zeros(1)

    def solve(Dd, lab, x0):
        return least_squares(res, x0[:3] if fix_f else x0, args=(lab, Dd), loss="soft_l1", f_scale=2.0).x

    for _ in range(2):
        lab = labels(x)
        p = solve(D, lab, x)
        x = np.r_[p, x[3]] if fix_f else p
    return x, labels(x), solve


def estimate_camera(img, car_mask=None, f_px=None, n_boot=20, detail: dict | None = None):
    """roll, pitch (deg), f (px at img's resolution) and confidences, or None (no vertical vanishing point).
    detail: a dict that, when given, receives the segments used (x1, y1, x2, y2 px), each one's direction
    label (0 = vertical, 1 and 2 = the two horizontal directions, -1 = none) and the three vanishing points
    (px, or None at infinity), for the debug pictures; the result is the same."""
    D, c, S = _segments(img, car_mask)
    if detail is not None:
        detail.update(segments=np.c_[D["a"] * S + c, D["b"] * S + c], labels=np.full(len(D["L"]), -1), vps=[])
    if len(D["L"]) < 3:
        return None
    v, vc = _vertical_vp(D)
    if v is None:
        return None
    e = _err(D, v)[:, 0]
    Hs = _take(D, ~((e < D["th"]) & vc) & (e > 5 * D2R))
    Hs = _take(Hs, np.argsort(-Hs["L"])[:500])
    al = np.arange(0, 90, 0.5) * D2R
    fix = f_px is not None or len(Hs["L"]) < 5
    sharp = 0.0
    if not fix:
        fg = 1 / np.tan(np.linspace(25, 120, 60) * D2R / 2)
        Rm = _manhattan(v, Hs, fg, al)
        k = Rm[:, 0].argmax()
        ff = fg[k] * np.linspace(0.92, 1.08, 17)
        R2 = _manhattan(v, Hs, ff, Rm[k, 1] + np.arange(-1.5, 1.5, 0.1) * D2R)
        f, alpha = ff[R2[:, 0].argmax()], R2[R2[:, 0].argmax(), 1]
        lfg, rsc = np.log(fg)[::-1], Rm[::-1, 0]  # np.interp needs increasing x (fg falls as the field of view grows)
        side = max(np.interp(np.log(f / 1.25), lfg, rsc), np.interp(np.log(f * 1.25), lfg, rsc))
        sharp = float(np.clip((Rm[k, 0] / max(side, 1e-9) - 1) / 0.3, 0, 1))
    else:
        f = (f_px or S / np.tan(35 * D2R)) / S
        alpha = _manhattan(v, Hs, [f], al)[0, 1] if len(Hs["L"]) else 0.0
    x, lab, solve = _refine(D, vc, np.r_[_rp(v, f), alpha, np.log(f)], fix)
    if detail is not None:
        vps = [V[0] for V in _vps(x[0], x[1], x[2], np.exp(x[3]))]
        detail.update(labels=lab.copy(), vps=[(V[:2] / V[2] * S + c) if abs(V[2]) > 1e-9 else None for V in vps])
    rng = np.random.default_rng(1)
    idx = np.arange(len(lab))
    xb = []
    for _ in range(n_boot):
        parts = [rng.choice(idx[lab == k], (lab == k).sum()) for k in range(3) if (lab == k).any()]
        if not parts:
            break
        sel = np.concatenate(parts)
        p = solve(_take(D, sel), lab[sel], x)
        xb.append(np.r_[p, x[3]] if fix else p)
    sd = np.std(xb, 0) if len(xb) > 1 else np.full(4, 0.5)
    n = [int((lab == k).sum()) for k in range(3)]
    Lv = D["L"][lab == 0].sum() / (2 * S)
    sup = min(1, n[0] / 8) * min(1, Lv)
    return dict(roll=float(x[0] * R2D), pitch=float(x[1] * R2D), f=float(np.exp(x[3]) * S), n_vert=n[0], sd_logf=float(sd[3]),
                conf=float(sup * np.exp(-(sd[0] * R2D / 1.0) ** 2 - (sd[1] * R2D / 2.0) ** 2)),
                conf_f=0.0 if fix else float(sup * sharp * min(1, (n[1] + n[2]) / 15) * np.exp(-(sd[3] / 0.08) ** 2)))


def line_camera(image: np.ndarray, car_alpha: np.ndarray, f_full: float | None = None, work_long: int = 1280,
                n_boot: int = 20, detail: dict | None = None) -> LineCamera:
    """estimate_camera on the photo (car masked), at most `work_long` px; f in full-resolution px.
    detail: see estimate_camera, in full-resolution px (debug pictures only)."""
    k = 1.0
    try:
        k = min(1.0, work_long / max(image.shape[:2]))
        img = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image
        m = (car_alpha >= 0.5).astype(np.uint8)
        if k < 1:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        r = estimate_camera(img, m, None if f_full is None else f_full * k, n_boot, detail)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        r = None
    if detail is not None and "segments" in detail:
        detail["segments"] = (detail["segments"] + 0.5) / k - 0.5
        detail["vps"] = [None if p is None else (np.asarray(p) + 0.5) / k - 0.5 for p in detail["vps"]]
    if r is None:
        return LineCamera(source="none", flags=["no-lines"])
    return LineCamera(roll=r["roll"], pitch=r["pitch"], f=r["f"] / k, conf=r["conf"], conf_f=r["conf_f"],
                      sd_logf=r["sd_logf"], n_vert=r["n_vert"])


# ---------------------------------------------------------------- focal choice for a run
def exif_focal_px(path, width: int, height: int) -> float | None:
    """35 mm-equivalent focal from EXIF -> px at (width, height). Video stills usually have none."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
            f35 = sub.get(0xA405) or ex.get(0xA405)
    except Exception:  # noqa: BLE001 - any unreadable or odd file simply has no EXIF focal
        return None
    if not f35:
        return None
    return float(f35) * math.hypot(width, height) / 43.27


def prior_focal(width: int, height: int) -> float:
    return 0.5 * max(width, height) / math.tan(math.radians(PRIOR_HFOV_LONG / 2))


def pool_focals(cams: list[LineCamera], sizes: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Per image size: the weighted median of log f over its biggest +-8% cluster of confident photos
    (weight conf_f / sd_logf^2), when it holds at least POOL_MIN photos."""
    out = {}
    for size in set(sizes):
        ok = [c for c, s in zip(cams, sizes) if s == size and c.f and c.conf_f >= POOL_CONF]
        if len(ok) < POOL_MIN:
            continue
        lf = np.log([c.f for c in ok])
        w = np.array([c.conf_f / max(c.sd_logf, 0.01) ** 2 for c in ok])
        best = max(range(len(lf)), key=lambda i: (w[np.abs(lf - lf[i]) <= POOL_TOL].sum(), -i))
        m = np.abs(lf - lf[best]) <= POOL_TOL
        if m.sum() < POOL_MIN:
            continue
        o = np.argsort(lf[m])
        cw = np.cumsum(w[m][o])
        out[size] = float(np.exp(lf[m][o][np.searchsorted(cw, cw[-1] / 2)]))
    return out


def choose_focal(cam: LineCamera, size: tuple[int, int], pooled: dict, config_focal: float | None,
                 exif: float | None) -> tuple[float, str, list[str]]:
    """(focal px, source, flags) for one photo; see the module docstring for the order."""
    W, H = size
    if config_focal:
        return float(config_focal), "config", []
    own = cam.f if (cam.f and cam.conf_f >= POOL_CONF) else None
    if size in pooled:
        fp = pooled[size]
        if own is not None and abs(math.log(own / fp)) > LENS_TOL:
            return own, "lines", ["lens-change"]
        return fp, "pooled", []
    if own is not None:
        return own, "lines", []
    if exif:
        return float(exif), "exif", []
    if cam.f and cam.conf_f >= WEAK_FOCAL_CONF:  # an uncertain line focal usually beats a generic field of view
        return float(cam.f), "lines-weak", ["focal-weak"]
    return prior_focal(W, H), "prior", ["focal-prior"]


def video_time(stem: str) -> float | None:
    m = VIDEO_NAME.match(stem)
    return float(m.group(1)) if m else None


def neighbour_roll_pitch(cams: list[LineCamera], times: list[float | None], i: int) -> tuple[float, float] | None:
    """Roll/pitch for a weak frame of a video: conf * exp(-(dt/4 s)^2) average over confident frames
    within +-8 s, or None."""
    if times[i] is None:
        return None
    acc = []
    for j, (c, t) in enumerate(zip(cams, times)):
        if j == i or t is None or c.roll is None or c.conf < NEIGHBOUR_CONF or abs(t - times[i]) > NEIGHBOUR_WINDOW_S:
            continue
        acc.append((c.conf * math.exp(-((t - times[i]) / NEIGHBOUR_TAU_S) ** 2), c.roll, c.pitch))
    sw = sum(a for a, _, _ in acc)
    if sw <= 0:
        return None
    return sum(a * r for a, r, _ in acc) / sw, sum(a * p for a, _, p in acc) / sw


__all__ = ["LineCamera", "choose_focal", "estimate_camera", "exif_focal_px", "line_camera", "neighbour_roll_pitch",
           "pool_focals", "prior_focal", "video_time"]
