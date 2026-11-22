"""Measure the cut-out car and read the view label from the file name.

Everything is measured on the solid outline behind `CarMask.box` (alpha >= 0.5, holes filled, posts
removed), in source-photo pixels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .segment import CarMask, bottom_profile

LABELS = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")
NAME_RE = re.compile(r"^(\d+(?:\.\d+)?)s_(" + "|".join(sorted(LABELS, key=len, reverse=True)) + r")_(\d{1,3})pct$")


@dataclass(frozen=True)
class NameInfo:
    """From a frame name like '0012.0s_right_95pct'."""

    time_s: float
    label: str
    conf: int  # classifier confidence, percent


def parse_name(stem: str) -> NameInfo | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return NameInfo(time_s=float(m.group(1)), label=m.group(2), conf=int(m.group(3)))


@dataclass(frozen=True)
class CarProbe:
    box: tuple[int, int, int, int]  # solid outline, source px, x2/y2 exclusive
    crop: tuple[int, int, int, int]
    w: int
    h: int
    lowest_y: int  # lowest solid row
    lowest_x: float
    lowest_side: float  # (lowest_x - box centre) / w; < 0 = left of centre
    tie: bool  # two equally low points far apart (e.g. both tyres in a pure side view)
    bottom: np.ndarray  # lowest solid row per box column (-1 = empty)
    contacts: tuple[tuple[float, float], ...]  # local lowest points of the bottom outline (x, y)
    cut: tuple[str, ...]  # photo edges the car touches: left/right/top/bottom

    @property
    def aspect(self) -> float:
        return self.w / self.h


def _local_maxima(b: np.ndarray, w: int, h: int) -> list[tuple[float, float]]:
    """Columns where the bottom outline is locally lowest (tyres, bumper corners)."""
    valid = b >= 0
    if valid.sum() < 3:
        return []
    x = np.arange(len(b))
    prof = np.interp(x, x[valid], b[valid]).astype(np.float64)
    k = max(3, int(round(0.01 * w)) | 1)
    prof = np.convolve(np.pad(prof, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    half = max(2, int(round(0.06 * w)))
    sep = 0.12 * w
    prom = 0.02 * h
    peaks = []
    for i in range(len(prof)):
        lo, hi = max(0, i - half), min(len(prof), i + half + 1)
        if prof[i] < prof[lo:hi].max() - 1e-9:
            continue
        wide_lo, wide_hi = max(0, i - 2 * half), min(len(prof), i + 2 * half + 1)
        if prof[i] - min(prof[wide_lo:i + 1].min(), prof[i:wide_hi].min()) < prom and not (
                i - 2 * half <= 0 or i + 2 * half >= len(prof) - 1):
            continue
        peaks.append((i, prof[i]))
    peaks.sort(key=lambda p: -p[1])
    kept: list[tuple[int, float]] = []
    for i, v in peaks:
        if all(abs(i - j) >= sep for j, _ in kept):
            kept.append((i, v))
    return sorted((float(i), float(v)) for i, v in kept)


def probe_car(car: CarMask) -> CarProbe:
    x1, y1, x2, y2 = car.box
    w, h = x2 - x1, y2 - y1
    bottom = car.bottom
    if bottom is None or len(bottom) != w:
        bottom = bottom_profile(car.alpha >= 0.5, car.box)
    lowest_y = int(bottom.max())
    near = np.flatnonzero(bottom >= lowest_y - 3)
    tie = len(near) > 1 and (near.max() - near.min()) > 0.3 * w and bool(
        np.any(np.diff(near) > 0.3 * w))
    lowest_x = float(x1 + (np.mean(near) if tie else np.median(near)))
    contacts = tuple((x1 + cx, cy) for cx, cy in _local_maxima(bottom, w, h))
    cut = tuple(side for side in ("left", "right", "top", "bottom") if car.cut.get(side))
    return CarProbe(box=car.box, crop=car.crop, w=w, h=h, lowest_y=lowest_y, lowest_x=lowest_x,
                    lowest_side=(lowest_x - (x1 + x2) / 2) / w, tie=tie, bottom=bottom,
                    contacts=contacts, cut=cut)
