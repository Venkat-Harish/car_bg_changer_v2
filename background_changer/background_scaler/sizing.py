"""How small the background should be for the photos at hand.

On a full-size scene a photo's car gets the scale  s = stage height of its view / its height in the
photo  (bg_changer.placement.raw_scale). s above 1 would enlarge the car. Shrinking the background to
k = 1 / s of full size gives the car exactly its own resolution there, so

    k = min(1, 1 / s)

k = 1 (a 4K photo, say) means the full-size background is fine.
"""

from __future__ import annotations

import math
from typing import Iterable

from bg_changer.placement import raw_scale, skip_reason
from bg_changer.probe import CarProbe
from bg_changer.scenes.base import Scene
from bg_changer.yaw import YawEstimate

MODES = ("folder", "photo", "off")
DEFAULT_MODE = "folder"
DEFAULT_MIN_WIDTH = 640  # px: a background is never shrunk below this width


def needed_scale(scene: Scene, probe: CarProbe, yaw: YawEstimate, cut_policy: str = "anchor") -> float | None:
    """How small (a share of full size, at most 1) `scene`'s background may be for this photo's car to
    fill the stage without being enlarged. None when the photo is not placed (cut off on two sides, or
    cut off with cut_policy 'skip'). `scene` is the full-size scene."""
    if skip_reason(probe, cut_policy):
        return None
    return min(1.0, 1.0 / raw_scale(probe, scene.table.at(yaw.deg), scene.px_per_m))


def folder_scale(needs: Iterable[float | None]) -> float:
    """One size for a whole folder: the smallest any of its placed photos needs (1 if none needs less)."""
    return min((k for k in needs if k is not None), default=1.0)


def background_size(full_wh: tuple[int, int], k: float, min_width: int = DEFAULT_MIN_WIDTH) -> tuple[int, int]:
    """(width, height) of the background shrunk to about k of full size, same aspect ratio. The width is
    rounded up, so the car never needs enlarging; it stays at least min_width and at most full size."""
    w_full, h_full = full_wh
    if k >= 1.0:
        return w_full, h_full
    w = math.ceil(w_full * k - 1e-9)
    w = min(w_full, max(w, min(int(min_width), w_full), 1))
    return w, max(1, round(h_full * w / w_full))
