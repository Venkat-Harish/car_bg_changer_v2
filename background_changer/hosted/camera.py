"""Hosted camera: MoGe-3 per frame (focal + floor plane -> camera height, pitch, roll) and VGGT per car (all frames
solved together: shared focal, relative yaw between shots). The result, HostedCamera, is what
car_fit_to_bg.batch.Measured.external carries; fit_measured makes it the photo's camera unless strong background
lines disagree (then the line camera stays, flag camera-lines-fallback).

Conventions (car_fit_to_bg/CLAUDE.md, pose.cam_angles): camera axes x right, y down, z forward (OpenCV); pitch > 0
looks down, roll > 0 = horizon rises to the right, degrees; with g = the floor's "down" in camera coordinates,
pitch = asin(g_z), roll = atan2(g_x, g_y). Focal in px at the photo's FULL size; principal point = frame centre;
height in metres above the floor. rel_yaw_deg uses the car-yaw convention: car_yaw(frame i) ~= car_yaw(frame 0) +
rel_yaw_deg[i] for a car standing still while the camera walks around it (pose.mv: yaw 0 = front, 90 = right side).

Endpoint facts (read 2026-11-23; never called by tests):

MoGe-3, `microsoft/moge-3` (https://fal.ai/models/microsoft/moge-3/api, .../llms.txt), $0.015 per megapixel of the
input, rounded up, minimum 1 MP:
- input: `image_url` (one image), `resolution_level` 0-9 (9), `num_refinement_steps` 0-5 (3), `fov_x` (deg, optional
  known FoV), `apply_mask` (true: drops sky/invalid), `edge_threshold` (0.04), `export_glb` (true), `export_ply`
  (true), `enable_safety_checker`. No mask input. Output size = input size.
- output: `depth_map`, `normal_map` (colourised PNG visualisations only, not data), `mask` (PNG, white = valid),
  `fov_x`, `fov_y` (deg), `intrinsics` (3x3 "normalised" [[fx,0,cx],[0,fy,cy],[0,0,1]]: MoGe's convention is fx/W,
  cy/H, cx = cy = 0.5), `point_cloud` (coloured .ply), `model_mesh` (.glb).
- There is no raw point-map field, so the 3D points come from the PLY. Not documented: per-pixel or not, and the
  axes. MoGe's own exporter writes the PLY in OpenGL axes (x right, y up, z backward); we detect that from the sign
  of z (OpenGL: points have z < 0) and flip y and z. Pixel positions come from re-projecting each point with the
  intrinsics (so the car can be masked out). Units: metric (MoGe-2/3 predict metres).
- focal at full size = (W / 2) / tan(fov_x / 2).

VGGT, `fal-ai/vggt-1b` (https://fal.ai/models/fal-ai/vggt-1b/llms.txt), $0.000975 per GPU second (we book
$0.001 per frame):
- input: `image_urls` (list; frames "padded to a square and resized so the longest side is 518 pixels"),
  `video_url`, `frame_sampling_rate`, `alpha_blend_onto` (keep|white|black|mean), `export_prediction_data`,
  `export_depth_maps`, `export_point_cloud` (all true by default; we turn them off), `array_encoding`,
  `exclude_keys`, `confidence_percentile`, `max_points`, `enable_safety_checker`. No mask input: we paint the
  (dilated) car mid-grey before sending, because shiny paint and glass hurt the multi-view solve.
- output: `num_frames`, `depth_range`, `extrinsics` (per frame 3x4 "camera-from-world", OpenCV axes x right, y down,
  z forward; the world is frame 0's camera, translation scale arbitrary), `intrinsics` (per frame 3x3 in pixels of
  the 518x518 model frame), optional `depth_maps` (16-bit PNG), `prediction_data` (JSON), `point_cloud` (GLB).
- focal at full size: square padding (cx ~= cy) -> fx * max(W, H) / (2 cx); otherwise (cropped to 518 wide)
  fx * W / (2 cx).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .fal import FalClient

MOGE_ENDPOINT = "microsoft/moge-3"
VGGT_ENDPOINT = "fal-ai/vggt-1b"
MOGE_USD_PER_MP = 0.015
VGGT_USD_PER_FRAME = 0.001

# floor fit
RANSAC_ITERS = 400
RANSAC_SAMPLE = 20000  # points kept for the fit (seeded subsample)
MIN_FLOOR_POINTS = 200
MIN_INLIER_SHARE = 0.25
MAX_FLOOR_TILT_DEG = 50.0  # the floor normal must be within this of the camera's "up" (-y)
HEIGHT_RANGE = (0.2, 6.0)  # m
CAR_DILATE = 0.02  # of the long side

# VGGT
GREY = 128
VGGT_LEVEL_DEG = 3.0  # a camera x axis tilted more than this from horizontal lowers that frame's conf


@dataclass
class FrameCam:
    f_px: float | None = None  # full-size px
    pitch: float | None = None
    roll: float | None = None
    height_m: float | None = None
    rel_yaw_deg: float | None = None  # VGGT: camera yaw relative to the first frame, degrees
    conf: float = 0.0  # 0..1
    source: str = ""  # moge | vggt


@dataclass
class HostedCamera:
    """What the fit consumes for one photo (car_fit_to_bg.batch.Measured.external)."""
    f_px: float | None = None
    pitch: float | None = None
    roll: float | None = None
    height_m: float | None = None
    rel_yaw_deg: float | None = None
    conf: float = 0.0
    source: str = "hosted"  # hosted | moge | vggt
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- geometry helpers
def angles_from_up(n_up: np.ndarray) -> tuple[float, float]:
    """(pitch, roll) in degrees from the world's up direction in camera coordinates (as pose.cam_angles)."""
    g = -np.asarray(n_up, float)
    g = g / np.linalg.norm(g)
    return math.degrees(math.asin(max(-1.0, min(1.0, g[2])))), math.degrees(math.atan2(g[0], g[1]))


def pitch_at_focal(pitch: float, f_from: float | None, f_to: float | None) -> float:
    """Pitch moved to another focal as car_fit_to_bg.batch._at_focal does (tan(pitch) scales with the focal)."""
    if not f_from or not f_to:
        return pitch
    return math.degrees(math.atan(math.tan(math.radians(pitch)) * f_to / f_from))


def _dilate(mask: np.ndarray, frac: float) -> np.ndarray:
    m = np.asarray(mask)
    m = (m > (0.5 if m.dtype.kind == "f" else 127 if m.dtype == np.uint8 else 0)).astype(np.uint8)
    k = max(3, int(round(frac * max(m.shape))) | 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def fit_floor(points: np.ndarray, seed: int = 0, iters: int = RANSAC_ITERS) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane through camera-frame points (N, 3, OpenCV axes). Returns (unit normal pointing to the camera's
    side, distance of the camera to the plane, inlier share) or None. Only planes whose normal is within
    MAX_FLOOR_TILT_DEG of -y (up in the image) count."""
    P = np.asarray(points, float)
    P = P[np.isfinite(P).all(1)]
    if len(P) < MIN_FLOOR_POINTS:
        return None
    rng = np.random.default_rng(seed)
    if len(P) > RANSAC_SAMPLE:
        P = P[np.sort(rng.choice(len(P), RANSAC_SAMPLE, replace=False))]
    thr = max(0.01, 0.01 * float(np.median(np.linalg.norm(P, axis=1))))
    cos_max = math.cos(math.radians(MAX_FLOOR_TILT_DEG))
    best_n, best_cnt = None, 0
    idx = rng.integers(0, len(P), (iters, 3))
    for i, j, k in idx:
        n = np.cross(P[j] - P[i], P[k] - P[i])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        d = -float(n @ P[i])
        if d < 0:
            n, d = -n, -d
        if -n[1] < cos_max:
            continue
        cnt = int((np.abs(P @ n + d) < thr).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    if best_n is None:
        return None
    n, d = best_n, best_d
    for _ in range(3):  # least-squares refinement on the inliers
        inl = P[np.abs(P @ n + d) < thr]
        if len(inl) < 3:
            return None
        cen = inl.mean(0)
        n2 = np.linalg.svd(inl - cen, full_matrices=False)[2][-1]
        if n2 @ n < 0:
            n2 = -n2
        n, d = n2, -float(n2 @ cen)
    if d < 0 or -n[1] < cos_max:
        return None
    share = float((np.abs(P @ n + d) < thr).mean())
    return n, d, share


# ---------------------------------------------------------------- PLY
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_points(data: bytes) -> np.ndarray:
    """(N, 3) float64 x, y, z of a PLY's vertex element (ascii or binary; vertex must be the first element and
    must not hold list properties)."""
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("not a PLY file")
    body_start = data.index(b"\n", end) + 1
    lines = data[:end].decode("ascii", "replace").splitlines()
    fmt, n, props, in_vertex, seen_element = None, 0, [], False, False
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            if seen_element and not in_vertex and t[1] == "vertex":
                raise ValueError("PLY: vertex is not the first element")
            in_vertex = t[1] == "vertex"
            if in_vertex:
                n = int(t[2])
            seen_element = True
        elif t[0] == "property" and in_vertex:
            if t[1] == "list":
                raise ValueError("PLY: list property in vertex")
            props.append((t[2], _PLY_TYPES[t[1]]))
    names = [p[0] for p in props]
    if not {"x", "y", "z"} <= set(names):
        raise ValueError("PLY: no x/y/z")
    if fmt == "ascii":
        rows = data[body_start:].decode("ascii").split("\n")
        arr = np.array([r.split()[:len(props)] for r in rows[:n] if r.strip()], float)
        return arr[:, [names.index("x"), names.index("y"), names.index("z")]]
    order = "<" if fmt == "binary_little_endian" else ">"
    dt = np.dtype([(nm, order + ty) for nm, ty in props])
    v = np.frombuffer(data, dt, count=n, offset=body_start)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


# ---------------------------------------------------------------- MoGe-3 per frame
def moge_frame(client: FalClient, image: np.ndarray, car_mask: np.ndarray, max_side: int = 1024) -> FrameCam | None:
    """Focal (full-size px) and the floor plane from MoGe-3's metric point cloud: RANSAC on the points that land in
    the lower half of the frame, off the (dilated) car. None when MoGe gives nothing usable or no floor is found."""
    H, W = image.shape[:2]
    s = min(1.0, max_side / max(H, W)) if max_side else 1.0
    mp = (round(W * s) * round(H * s)) / 1e6
    out = client.run(MOGE_ENDPOINT, {"export_glb": False, "export_ply": True, "apply_mask": True},
                     images={"image_url": image}, cost=MOGE_USD_PER_MP * max(1, math.ceil(mp - 1e-9)),
                     max_side=max_side)
    fov_x = out.get("fov_x")
    K = np.asarray(out.get("intrinsics") or np.eye(3), float)
    if fov_x and 0 < float(fov_x) < 179:
        f = (W / 2) / math.tan(math.radians(float(fov_x)) / 2)
    elif K[0, 0] > 0 and K[0, 2] > 0:
        f = K[0, 0] * W if K[0, 2] < 2 else K[0, 0] / s  # normalised, else px of the sent image
    else:
        return None
    cam = FrameCam(f_px=float(f), source="moge")
    pc = out.get("point_cloud")
    if not pc or not pc.get("url"):
        return cam
    P = read_ply_points(client.fetch_bytes(pc["url"]))
    P = P[np.isfinite(P).all(1)]
    if len(P) == 0:
        return cam
    if np.median(P[:, 2]) < 0:  # OpenGL axes (y up, z backward) -> OpenCV
        P = P * np.array([1.0, -1.0, -1.0])
    P = P[P[:, 2] > 1e-6]
    # pixel of every point at full size (normalised intrinsics when cx ~ 0.5, else derived from the focal)
    if 0 < K[0, 2] < 2 and K[0, 0] > 0 and K[1, 1] > 0:
        fx, fy, cx, cy = K[0, 0] * W, K[1, 1] * H, K[0, 2] * W - 0.5, K[1, 2] * H - 0.5
    else:
        fx = fy = f
        cx, cy = (W - 1) / 2, (H - 1) / 2
    u = np.round(fx * P[:, 0] / P[:, 2] + cx).astype(np.int64)
    v = np.round(fy * P[:, 1] / P[:, 2] + cy).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= H / 2) & (v < H)
    car = _dilate(car_mask, CAR_DILATE)
    ok[ok] &= ~car[v[ok], u[ok]]
    fl = fit_floor(P[ok])
    if fl is None:
        return cam
    n, d, share = fl
    if share < MIN_INLIER_SHARE or not (HEIGHT_RANGE[0] <= d <= HEIGHT_RANGE[1]):
        return cam
    cam.pitch, cam.roll = angles_from_up(n)
    cam.height_m = float(d)
    cam.conf = float(min(1.0, max(0.0, (share - MIN_INLIER_SHARE) / (0.8 - MIN_INLIER_SHARE))) * 0.7 + 0.3)
    return cam


# ---------------------------------------------------------------- VGGT per car
def _grey_car(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(image[..., :3]).copy()
    out[_dilate(mask, CAR_DILATE)] = GREY
    return out


def _vggt_focal(Ki: np.ndarray, W: int, H: int) -> float | None:
    fx, cx, cy = float(Ki[0, 0]), float(Ki[0, 2]), float(Ki[1, 2])
    if fx <= 0:
        return None
    if cx <= 0:
        cx = cy = 259.0
    side = max(W, H) if abs(cx - cy) < 0.05 * cx else W
    return fx * side / (2 * cx)


def estimate_up(R_list: list[np.ndarray]) -> np.ndarray:
    """World up (unit) from world->camera rotations: the direction most perpendicular to every camera's x axis
    (handheld cameras keep x roughly level; robust to pitch), signed like the cameras' mean -y. With too little
    spread in the x axes (one shot, or all shots parallel) the mean -y alone."""
    X = np.array([R[0] for R in R_list])  # camera x axes in world coordinates
    U = np.array([-R[1] for R in R_list])  # camera "up" (-y) in world coordinates
    mean_up = U.sum(0)
    mean_up /= np.linalg.norm(mean_up)
    if len(R_list) >= 3:
        w, V = np.linalg.eigh(X.T @ X)
        if w[1] > 0.02 * len(R_list):  # the x axes span a plane
            up = V[:, 0]
            return up if up @ mean_up >= 0 else -up
    return mean_up


def vggt_car(client: FalClient, images: list[np.ndarray], car_masks: list[np.ndarray],
             max_side: int = 518) -> list[FrameCam | None]:
    """All frames of one car in one VGGT call (car painted grey). Per frame: focal at full size, yaw relative to
    frame 0 about the estimated world up (car-yaw sign, measured on the ray through the car mask's centroid so an
    off-centre car does not bias it), conf from how level the camera's x axis is."""
    if not images:
        return []
    sent = [_grey_car(im, mk) for im, mk in zip(images, car_masks)]
    out = client.run(VGGT_ENDPOINT, {"export_point_cloud": False, "export_depth_maps": False,
                                     "export_prediction_data": False},
                     images={"image_urls": sent}, cost=VGGT_USD_PER_FRAME * len(images), max_side=max_side)
    E = out.get("extrinsics") or []
    Ks = out.get("intrinsics") or []
    if len(E) != len(images) or len(Ks) != len(images):
        return [None] * len(images)
    R_list, cams = [], []
    for i, (im, mk) in enumerate(zip(images, car_masks)):
        H, W = im.shape[:2]
        R = np.asarray(E[i], float)[:3, :3]
        R_list.append(R)
        cams.append(FrameCam(f_px=_vggt_focal(np.asarray(Ks[i], float), W, H), source="vggt"))
    up = estimate_up(R_list)
    az = []
    for i, (im, mk) in enumerate(zip(images, car_masks)):
        H, W = im.shape[:2]
        m = _dilate(mk, 0.0) if mk is not None else np.zeros((H, W), bool)
        f = cams[i].f_px or max(W, H)
        if m.any():
            ys, xs = np.nonzero(m)
            ray = np.array([(xs.mean() - (W - 1) / 2) / f, (ys.mean() - (H - 1) / 2) / f, 1.0])
        else:
            ray = np.array([0.0, 0.0, 1.0])
        d = R_list[i].T @ ray  # world direction from the camera to the car
        d -= (d @ up) * up
        az.append(d / max(np.linalg.norm(d), 1e-12))
    for i, cam in enumerate(cams):
        a, b = az[0], az[i]
        ang = math.degrees(math.atan2(float(up @ np.cross(a, b)), float(a @ b)))
        cam.rel_yaw_deg = float((-ang + 180.0) % 360.0 - 180.0)  # car yaw turns the other way to the camera
        tilt = math.degrees(math.asin(min(1.0, abs(float(R_list[i][0] @ up)))))
        base = 0.8 if len(images) >= 2 else 0.3
        cam.conf = float(base * math.exp(-(max(0.0, tilt - VGGT_LEVEL_DEG) / 5.0) ** 2))
    return cams


# ---------------------------------------------------------------- combine
def combine(moge: list[FrameCam | None], vggt: list[FrameCam | None] | None,
            sizes: list[tuple[int, int]]) -> list[HostedCamera | None]:
    """One HostedCamera per photo: focal = median VGGT focal over the car's photos of the same size (else MoGe's),
    pitch/roll/height from MoGe (pitch moved to that focal), rel_yaw from VGGT. Flags: camera-moge, camera-vggt,
    vggt-missing, moge-missing, focal-vggt / focal-moge."""
    n = len(sizes)
    vggt_list = list(vggt) if vggt is not None else [None] * n
    group_f: dict[tuple[int, int], float] = {}
    for size in set(map(tuple, sizes)):
        fs = [v.f_px for v, sz in zip(vggt_list, sizes) if v is not None and v.f_px and tuple(sz) == size]
        if fs:
            group_f[size] = float(np.median(fs))
    out: list[HostedCamera | None] = []
    for i in range(n):
        mo = moge[i] if i < len(moge) else None
        vg = vggt_list[i] if i < len(vggt_list) else None
        if mo is None and vg is None:
            out.append(None)
            continue
        flags: list[str] = []
        f = group_f.get(tuple(sizes[i]))
        if f is not None:
            flags.append("focal-vggt")
        elif mo is not None and mo.f_px:
            f = mo.f_px
            flags.append("focal-moge")
        hc = HostedCamera(f_px=f, flags=flags)
        if mo is not None:
            flags.insert(0, "camera-moge")
            if mo.pitch is not None and mo.roll is not None:
                hc.pitch = pitch_at_focal(mo.pitch, mo.f_px, f)
                hc.roll = mo.roll
            hc.height_m = mo.height_m
            hc.conf = mo.conf
        else:
            flags.insert(0, "moge-missing")
        if vg is not None:
            flags.insert(1, "camera-vggt")
            hc.rel_yaw_deg = vg.rel_yaw_deg
            if mo is None:
                hc.conf = 0.5 * vg.conf
        else:
            flags.insert(1, "vggt-missing")
        hc.source = "hosted" if (mo is not None and vg is not None) else ("moge" if mo is not None else "vggt")
        out.append(hc)
    return out


# ---------------------------------------------------------------- yaw hints
def _bins() -> dict[str, float]:
    from bg_changer.yaw import BIN_DEG  # noqa: PLC0415 - keeps hosted importable without bg_changer loaded first

    return BIN_DEG


def yaw_probs(yaw_deg: float, sigma: float = 15.0) -> dict[str, float]:
    """A peaked distribution over the view bins (pose.probs_from_label's keys) around a car yaw in degrees, for
    view_hints / fit_photo(probs=...). von Mises weights, normalised to 1."""
    bins = _bins()
    k = 1.0 / math.radians(sigma) ** 2
    w = {lab: math.exp(k * (math.cos(math.radians(yaw_deg - deg)) - 1.0)) for lab, deg in bins.items()}
    tot = sum(w.values())
    return {lab: v / tot for lab, v in w.items()}


def anchor_yaws(rel_yaws: list[float | None], fitted_yaws: list[float | None]) -> float | None:
    """Offset (deg, in [0, 360)) so that fitted ~= offset + rel, as the circular median of fitted - rel over the
    frames that have both (e.g. the strong, well-fitted ones). None when no frame has both."""
    diffs = [((fy - ry) % 360.0) for ry, fy in zip(rel_yaws, fitted_yaws) if ry is not None and fy is not None]
    if not diffs:
        return None
    d = np.array(diffs)
    cost = [float(np.abs((d - c + 180.0) % 360.0 - 180.0).sum()) for c in d]
    return float(d[int(np.argmin(cost))] % 360.0)


__all__ = ["MOGE_ENDPOINT", "VGGT_ENDPOINT", "FrameCam", "HostedCamera", "anchor_yaws", "angles_from_up", "combine",
           "estimate_up", "fit_floor", "moge_frame", "pitch_at_focal", "read_ply_points", "vggt_car", "yaw_probs"]
