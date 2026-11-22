"""car_fit_to_bg: fit every car photo into a background re-rendered from that photo's own camera.

Every car goes through this step (room mode). Per photo, independent of the background:
  1. the camera's roll, pitch and focal from the photo's own background lines (lines.py);
  2. the car's pose from its silhouette: yaw, distance, camera height, aim (pose.py);
  3. the level angle: camera roll plus the car's own roll on a sloped road, limited so a slanted cut edge
     loses at most 2% of the car; the car's pixels get this one rigid rotation (level.py);
  4. tyre bottoms on the levelled car and the contact lock of the virtual camera (ground.py).
Per background (place.py): the background's 3D room (room3d, rooms.py) is rendered from that camera, its
floor shaded, and the car pasted last.

    fit = fit_photo(frame, alpha, cut, shape, focal, focal_source, cam, probs=..., label=...)
    placed = place_in_room(fit, "background_1", plate, spec)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from bg_changer.carmodel import CarShape, CarTable

from .ground import TyreBottom, contact_lock, find_tyre_bottoms, floor_points_of
from .level import (CAR_ROLL_RANGE, MIN_DEG, LevelledCar, bottom_profile, cut_crop, level_car, limit_for_cut,
                    near_end_or_side,
                    roll_of, rz, thirds_tilt)
from .lines import LineCamera
from .place import Placement, place_in_room
from .pose import (PoseFit, RedCue, body_ambiguous, body_points, guess_body, red_cue, refit_with_roll, render_proxy)

LOCK_IOU = 0.90
MIN_CAR_PX = 400


class FitError(ValueError):
    """The photo cannot be fitted at all (no car, or a car filling the whole frame)."""


@dataclass
class PhotoFit:
    stem: str
    frame_size: tuple[int, int]
    c: tuple[float, float]
    f: float
    focal_source: str
    shape: CarShape
    side_length: float
    cam: LineCamera
    pose: PoseFit
    body_source: str
    theta: float  # deg: the rotation applied to the car's pixels
    theta_full: float  # deg: the full level angle (before the cut-edge limit)
    car_roll: float  # deg: the car's own roll on its floor that was applied (0 = none)
    R_lev: np.ndarray  # car -> levelled camera (before the contact lock)
    R_lock: np.ndarray  # car -> levelled, locked camera
    C_lock: np.ndarray  # the camera centre in car metres
    lock: dict
    bottoms: list[TyreBottom]
    patch_xy: list[tuple[float, float]]
    car: LevelledCar
    cut: frozenset
    cut_edges: dict
    cue: RedCue
    flags: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return self.pose.failed


def _levelled_tilt(alpha: np.ndarray, c, theta: float) -> float | None:
    """Bottom-line tilt (deg, + rises right) of a mask after rotating it by theta about c."""
    H, W = alpha.shape
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -theta, 1.0)  # cv2: + = counter-clockwise
    big = int(1.3 * max(W, H))
    M[0, 2] += (big - W) / 2
    M[1, 2] += (big - H) / 2
    rot = cv2.warpAffine((alpha >= 0.5).astype(np.uint8), M, (big, big), flags=cv2.INTER_NEAREST)
    if rot.sum() < 50:
        return None
    u, v = bottom_profile(rot > 0)
    return thirds_tilt(u, v)


def _expected_tilt(shape, body, R, t, f, c, size) -> float | None:
    """The same tilt for the proxy drawn through the levelled camera (a level car at the fitted yaw)."""
    pts = body_points(shape, body)
    W, H = size
    big = int(1.3 * max(W, H))
    cc = (c[0] + (big - W) / 2, c[1] + (big - H) / 2)
    m = render_proxy(pts @ R.T + t, f, cc, (big, big))
    if (m > 127).sum() < 50:
        return None
    u, v = bottom_profile(m > 127)
    return thirds_tilt(u, v)


def fit_photo(frame: np.ndarray, alpha: np.ndarray, cut, shape: CarShape, focal: float, focal_source: str,
              cam: LineCamera, probs: dict | None = None, label: str | None = None,
              h_prior: tuple[float, float] | None = None, max_seconds: float = 120.0, stem: str = "",
              strong_h_prior: tuple[float, float] | None = None) -> PhotoFit:
    """Everything about one photo that does not depend on the background. alpha: full-frame car alpha.
    strong_h_prior: a measured camera height (h, sigma) used even with strong lines (pose.fit_pose)."""
    from .pose import fit_pose
    t0 = time.perf_counter()
    H, W = alpha.shape
    solid = int((alpha >= 0.5).sum())
    if solid < MIN_CAR_PX:
        raise FitError(f"no car to fit ({solid} car pixels)")
    if solid > 0.9 * H * W:
        raise FitError("the car fills the whole photo: nothing to fit it against")
    c = ((W - 1) / 2, (H - 1) / 2)
    cut = frozenset(cut)
    flags: list[str] = list(cam.flags)
    body = shape.body or guess_body(shape)
    body_source = "cars.json" if shape.body else "guess"
    alt = None
    if not shape.body:
        flags.append("body-guessed")
        if body_ambiguous(shape):
            alt = "sedan" if body == "hatch" else "hatch"
    cue = red_cue(frame, alpha)
    pose = fit_pose(alpha, cut, shape, focal, body, probs=probs, cue=cue, line_roll=cam.roll, line_pitch=cam.pitch,
                    strong_lines=cam.strong, h_prior=h_prior, label=label, max_seconds=max_seconds, alt_body=alt,
                    strong_h_prior=strong_h_prior)
    flags += [fl for fl in pose.flags if fl not in flags]
    R_pc, t_pc = pose.rt(c)
    C_car = -R_pc.T @ t_pc

    # The level angle: the car's floor level in the output. Weak lines: the silhouette's roll is not trusted
    # with the car's pixels; the room is rendered with that roll instead.
    car_roll = 0.0
    if cam.strong:
        theta_full = roll_of(R_pc)
        if not cut and not pose.failed and near_end_or_side(pose.yaw):
            actual = _levelled_tilt(alpha, c, theta_full)
            R_lev0 = rz(theta_full) @ R_pc
            expected = _expected_tilt(shape, pose.body, R_lev0, rz(theta_full) @ t_pc, focal, c, (W, H))
            if actual is not None and expected is not None:
                tau = actual - expected
                if CAR_ROLL_RANGE[0] <= abs(tau) <= CAR_ROLL_RANGE[1]:
                    # The car stands on a cross-slope: the lines' gravity pulled the fitted roll off the car's own
                    # floor. Refit with the camera's roll relative to the car's floor fixed, so the car's pixels, the
                    # proxy and the room all agree on that floor.
                    refit = refit_with_roll(pose, shape, pose.roll + tau, cam.pitch)
                    if refit.iou >= pose.iou - 0.02:
                        car_roll = tau
                        pose = refit
                        R_pc, t_pc = pose.rt(c)
                        C_car = -R_pc.T @ t_pc
                        theta_full = roll_of(R_pc)
                        flags.append("car-on-slope")
                    else:
                        flags.append("possible-slope")
                elif abs(tau) > CAR_ROLL_RANGE[1]:
                    flags.append("possible-slope")
        elif cam.roll is not None and abs(pose.roll - cam.roll) > 3.0:
            flags.append("possible-slope")
    else:
        theta_full = 0.0
        flags.append("roll-from-silhouette")
    theta, reduced = limit_for_cut(alpha, cut, c, theta_full)
    if abs(theta) < MIN_DEG:  # not applied to the pixels, so not assumed anywhere else either
        theta = 0.0
    if reduced:
        flags.append("partial-level")
    R_lev = rz(theta) @ R_pc  # the camera of the levelled image: the car's pixels and the proxy stay together
    car = level_car(frame, alpha, theta, c)
    edges = cut_crop(alpha, cut, c, theta)
    guard = {s: (v["lo"] if s in ("right", "bottom") else v["hi"]) for s, v in edges.items()}
    K = np.array([[focal, 0.0, c[0]], [0.0, focal, c[1]], [0.0, 0.0, 1.0]])
    pts = body_points(shape, pose.body)
    bottoms = find_tyre_bottoms(car.alpha, car.origin, pts, shape, pose.yaw, R_lev, C_car, K, guard)
    if pose.iou >= LOCK_IOU and not pose.failed:
        R_lock, C_lock, lock = contact_lock(bottoms, shape, R_lev, C_car, K)
    else:
        R_lock, C_lock = R_lev, C_car.copy()
        lock = dict(n=0, mode="none", dh=0.0, dp=0.0, res_before=[], res_after=[], at_bound=False)
    patch_xy = floor_points_of(bottoms, R_lock, C_lock, K) if not pose.failed else []
    return PhotoFit(stem=stem, frame_size=(W, H), c=c, f=focal, focal_source=focal_source, shape=shape,
                    side_length=CarTable(shape).side_length(), cam=cam, pose=pose, body_source=body_source,
                    theta=theta, theta_full=theta_full, car_roll=car_roll, R_lev=R_lev, R_lock=R_lock, C_lock=C_lock, lock=lock,
                    bottoms=bottoms, patch_xy=patch_xy, car=car, cut=cut, cut_edges=edges, cue=cue, flags=flags,
                    seconds=time.perf_counter() - t0)


def fit_overlay(frame: np.ndarray, fit: PhotoFit, alpha: np.ndarray | None = None) -> np.ndarray:
    """The photo with the car's cut-out outline (red; when its full-frame alpha is given), the fitted proxy
    (green), its tyre contacts (yellow), the aim (magenta +) and the background lines' horizon (cyan)."""
    out = frame.copy()
    W, H = fit.frame_size
    R, t = fit.pose.rt(fit.c)
    pts = body_points(fit.shape, fit.pose.body)
    prox = render_proxy(pts @ R.T + t, fit.f, fit.c, (W, H))
    mask = None if alpha is None else ((alpha >= 0.5) * 255).astype(np.uint8)
    for m, col in ((mask, (0, 0, 255)), (prox, (0, 255, 0))):
        if m is None:
            continue
        cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 1, cv2.LINE_AA)
    cv2.drawMarker(out, (int(round(fit.pose.aim[0])), int(round(fit.pose.aim[1]))), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
    con = fit.shape.contacts()
    q = con @ R.T + t
    for p in q:
        if p[2] > 0.05:
            cv2.circle(out, (int(round(fit.c[0] + fit.f * p[0] / p[2])), int(round(fit.c[1] + fit.f * p[1] / p[2]))), 3,
                       (0, 255, 255), 1, cv2.LINE_AA)
    if fit.cam.roll is not None and fit.cam.pitch is not None:
        r, pch = math.radians(fit.cam.roll), math.radians(fit.cam.pitch)
        y_h = fit.c[1] - fit.f * math.tan(pch)
        dx = 2 * W
        cv2.line(out, (int(fit.c[0] - dx), int(round(y_h + dx * math.tan(r)))), (int(fit.c[0] + dx), int(round(y_h - dx * math.tan(r)))),
                 (255, 255, 0), 1, cv2.LINE_AA)
    return out


__all__ = ["FitError", "LineCamera", "PhotoFit", "Placement", "fit_overlay", "fit_photo", "place_in_room"]
