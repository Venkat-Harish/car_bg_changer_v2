"""The room-mode flow over a set of photos, in three stages (used by change_background.py and
tools/check_room.py):

  1. measure (per photo): the line camera at a free focal, on the full frame with the car masked;
  2. choose focals (per run): the config focal, else the folder pool per image size (>= 3 confident
     photos), else each photo's own lines, EXIF or the 70-degree prior; weak roll/pitch of time-stamped
     video frames borrowed from confident neighbours;
  3. fit (per photo): roll/pitch re-solved at the chosen focal, then car_fit_to_bg.fit_photo.
Stage 3 re-reads nothing itself: the caller passes the frame (re-read from disk) and the alpha it kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bg_changer.carmodel import CarShape
from bg_changer.probe import parse_name

from . import PhotoFit, fit_photo
from .lines import (LineCamera, choose_focal, exif_focal_px, line_camera, neighbour_roll_pitch, pool_focals,
                    video_time)
from .pose import probs_from_label, read_view_results

if TYPE_CHECKING:  # hosted is only imported when [hosted] is on; fit_measured reads the fields by name
    from hosted.camera import HostedCamera

HOSTED_CONF = 0.8  # LineCamera.conf of a hosted camera (strong: it levels the car like strong lines)
HOSTED_DISAGREE_DEG = 5.0  # strong lines that differ by more than this in pitch or roll win over the hosted camera
HOSTED_H_SIGMA = 0.15  # m: camera-height prior from the hosted floor fit


@dataclass
class Measured:
    path: Path
    stem: str
    size: tuple[int, int]  # (W, H)
    cam: LineCamera  # free-focal line camera
    exif_f: float | None
    time_s: float | None
    focal: float = 0.0
    focal_source: str = ""
    flags: list[str] = field(default_factory=list)
    external: HostedCamera | None = None  # hosted camera (MoGe floor / VGGT); None = the line camera alone


def measure(path: Path, frame: np.ndarray, alpha: np.ndarray) -> Measured:
    H, W = alpha.shape
    return Measured(path=path, stem=path.stem, size=(W, H), cam=line_camera(frame, alpha),
                    exif_f=exif_focal_px(path, W, H), time_s=video_time(path.stem))


def choose_focals(items: list[Measured], config_focal: float | None = None, pool: bool = True) -> None:
    """Fill in focal and focal_source for every measured photo (in place)."""
    pooled = pool_focals([m.cam for m in items], [m.size for m in items]) if pool else {}
    for m in items:
        m.focal, m.focal_source, fl = choose_focal(m.cam, m.size, pooled, config_focal, m.exif_f)
        m.flags += fl


def find_view_results(images_folder: Path) -> Path | None:
    """frame_view_type's results.csv for the same folder name, next to background_changer, if it exists."""
    root = Path(__file__).resolve().parents[2]
    p = root / "frame_view_type" / "output" / images_folder.name / "results.csv"
    return p if p.is_file() else None


def view_hints(stem: str, view_rows: dict | None) -> tuple[dict | None, str | None]:
    """(probabilities by view bin, label) for a photo: the classifier's row if any, else its file-name label."""
    info = parse_name(stem)
    label = info.label if info else None
    if view_rows and stem in view_rows and view_rows[stem]["probs"]:
        return view_rows[stem]["probs"], label
    return probs_from_label(label, info.conf if info else None), label


def fit_measured(m: Measured, frame: np.ndarray, alpha: np.ndarray, cut, shape: CarShape, all_items: list[Measured],
                 view_rows: dict | None, video: bool = True, h_prior: tuple[float, float] | None = None,
                 max_seconds: float = 120.0) -> PhotoFit:
    """Stage 3 for one photo: lines re-solved at the chosen focal, weak roll/pitch from video neighbours,
    then the pose fit and everything background-independent."""
    try:
        cam = line_camera(frame, alpha, f_full=m.focal)
    except Exception:  # noqa: BLE001 - no usable lines (stage 1 may have failed the same way): fit without them
        cam = LineCamera(source="none", flags=["lines-error"])
    strong_h = None
    if m.external is not None:
        cam, h_prior, strong_h = _with_external(cam, m.external, m.focal, h_prior)
    if cam.roll is None or cam.conf < 0.3:
        nb = None
        if video and m.time_s is not None:
            idx = all_items.index(m)
            nb = neighbour_roll_pitch([_at_focal(x.cam, m.focal) for x in all_items], [x.time_s for x in all_items], idx)
        if nb is not None:
            cam = LineCamera(roll=nb[0], pitch=nb[1], f=m.focal, conf=0.5, conf_f=0.0, source="neighbours",
                             flags=["roll-from-neighbours"])
    probs, label = view_hints(m.stem, view_rows)
    extra = {} if strong_h is None else {"strong_h_prior": strong_h}
    fit = fit_photo(frame, alpha, cut, shape, m.focal, m.focal_source, cam, probs=probs, label=label,
                    h_prior=h_prior, max_seconds=max_seconds, stem=m.stem, **extra)
    if probs is None and label is None:  # no classifier row and no label in the file name: the outline alone
        fit.flags.append("no-view-label")
    for fl in m.flags:
        if fl not in fit.flags:
            fit.flags.append(fl)
    return fit


def _with_external(cam: LineCamera, ext, focal: float, h_prior):
    """(camera, h_prior, strong_h_prior) once a hosted camera is known. Its roll/pitch (moved to the chosen focal
    like _at_focal) replace the lines unless the lines are strong and disagree by more than HOSTED_DISAGREE_DEG
    (then the lines stay, flag camera-lines-fallback, and the hosted height is not used either). Its height
    becomes the camera-height prior, for strong and weak lines alike."""
    use_height = True
    if ext.pitch is not None and ext.roll is not None:
        hosted = _at_focal(LineCamera(roll=float(ext.roll), pitch=float(ext.pitch), f=ext.f_px or focal), focal)
        lines_win = False
        if cam.strong and cam.source == "lines":
            d_roll = abs((cam.roll - hosted.roll + 180.0) % 360.0 - 180.0)
            d_pitch = abs(cam.pitch - hosted.pitch) if cam.pitch is not None else 0.0
            lines_win = max(d_roll, d_pitch) > HOSTED_DISAGREE_DEG
        if lines_win:
            cam = LineCamera(roll=cam.roll, pitch=cam.pitch, f=cam.f, conf=cam.conf, conf_f=cam.conf_f,
                             sd_logf=cam.sd_logf, n_vert=cam.n_vert, source=cam.source,
                             flags=list(cam.flags) + ["camera-lines-fallback"])
            use_height = False
        else:
            flags = ["camera-hosted"] + [fl for fl in ext.flags if fl != "camera-hosted"]
            cam = LineCamera(roll=hosted.roll, pitch=hosted.pitch, f=focal, conf=HOSTED_CONF, conf_f=0.0,
                             source="hosted", flags=flags)
    if use_height and ext.height_m is not None and ext.height_m > 0:
        hp = (float(ext.height_m), HOSTED_H_SIGMA)
        return cam, hp, hp
    return cam, h_prior, None


def _at_focal(cam: LineCamera, f: float) -> LineCamera:
    """A stage-1 (free-focal) line camera with its pitch moved to focal f: tan(pitch) scales with the focal."""
    if cam.pitch is None or not cam.f or not f:
        return cam
    p = math.degrees(math.atan(math.tan(math.radians(cam.pitch)) * f / cam.f))
    return LineCamera(roll=cam.roll, pitch=p, f=f, conf=cam.conf, conf_f=cam.conf_f, sd_logf=cam.sd_logf,
                      n_vert=cam.n_vert, source=cam.source, flags=list(cam.flags))


def folder_height_prior(fits: list[PhotoFit]) -> tuple[float, float] | None:
    """N(median camera height of the confident fits, 0.25 m), from at least 3 photos."""
    hs = [f.pose.h for f in fits if not f.pose.failed and f.cam.strong and f.pose.iou >= 0.9]
    if len(hs) < 3:
        return None
    return float(np.median(hs)), 0.25


def cut_sides(alpha: np.ndarray, tol: int = 3) -> set[str]:
    """Which photo borders the solid car touches (as bg_changer.segment's BORDER_TOL)."""
    sm = alpha >= 0.5
    return {n for n, v in (("left", sm[:, :tol].any()), ("right", sm[:, -tol:].any()), ("top", sm[:tol].any()),
                           ("bottom", sm[-tol:].any())) if v}


def skip_reason(cut) -> str:
    """Decision 1: bottom cuts and cuts on two sides are not placed."""
    cut = set(cut)
    if "bottom" in cut:
        return "car cut off at the bottom of the photo: no ground contact to stand it on"
    if len(cut) >= 2:
        return f"car cut off on {len(cut)} sides ({', '.join(sorted(cut))})"
    return ""


__all__ = ["Measured", "choose_focals", "cut_sides", "find_view_results", "fit_measured", "folder_height_prior",
           "measure", "read_view_results", "skip_reason", "view_hints"]

