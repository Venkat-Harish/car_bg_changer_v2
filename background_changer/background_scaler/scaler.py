"""Per run: which background size for each photo (or the whole folder), and the shrunk scenes."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from bg_changer.probe import CarProbe
from bg_changer.scenes.base import Scene
from bg_changer.segment import CarMask
from bg_changer.yaw import YawEstimate

from .rescale import scale_scene
from .sizing import DEFAULT_MIN_WIDTH, DEFAULT_MODE, MODES, background_size, folder_scale, needed_scale


class BackgroundScaler:
    """mode "folder": one size for all photos (call folder_scale with every photo's needed_scale);
    "photo": a size per photo (photo_scale); "off": always full size. Then apply() gives the scene and
    picture to place the car on."""

    def __init__(self, mode: str = DEFAULT_MODE, min_width: int = DEFAULT_MIN_WIDTH):
        if mode not in MODES:
            raise ValueError(f"background_scale must be one of: {', '.join(MODES)}")
        if isinstance(min_width, bool) or not isinstance(min_width, int) or min_width < 1:
            raise ValueError("the minimum background width must be a whole number of pixels, 1 or more")
        self.mode, self.min_width = mode, min_width
        self._last: dict[int, tuple[Scene, tuple[int, int], Scene, np.ndarray]] = {}

    def photo_scale(self, scene: Scene, probe: CarProbe, yaw: YawEstimate, cut_policy: str = "anchor") -> float:
        """The size this photo's car needs (1 = full size; also for "off" and for photos not placed)."""
        if self.mode == "off":
            return 1.0
        k = needed_scale(scene, probe, yaw, cut_policy)
        return 1.0 if k is None else k

    def folder_scale(self, needs: Iterable[float | None]) -> float:
        """One size for a folder, from each photo's needed_scale (None = not placed)."""
        return 1.0 if self.mode == "off" else folder_scale(needs)

    def size(self, scene: Scene, k: float) -> tuple[int, int]:
        return background_size(scene.size_wh, k, self.min_width)

    def apply(self, scene: Scene, plate: np.ndarray, k: float) -> tuple[Scene, np.ndarray]:
        """The scene and picture shrunk to about k of full size (the last one per scene is kept)."""
        size = self.size(scene, k)
        last = self._last.get(id(scene))
        if last is None or last[0] is not scene or last[1] != size:
            last = (scene, size, *scale_scene(scene, plate, size))
            self._last[id(scene)] = last
        return last[2], last[3]


def crop_to_car(image: np.ndarray, car: CarMask) -> tuple[np.ndarray, CarMask]:
    """Only the part of the photo the car occupies, with its mask moved to match. Placing it gives the
    same picture as placing the whole photo; in "folder" mode it keeps the waiting photos small."""
    x1, y1 = min(car.crop[0], car.box[0]), min(car.crop[1], car.box[1])
    x2, y2 = max(car.crop[2], car.box[2]), max(car.crop[3], car.box[3])

    def shift(b):
        return (b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1)

    bottom = None if car.bottom is None else np.where(car.bottom >= 0, car.bottom - y1, -1).astype(np.int32)
    mask = CarMask(alpha=car.alpha[y1:y2, x1:x2].copy(), box=shift(car.box), crop=shift(car.crop),
                   cut=dict(car.cut), bottom=bottom, isolation=car.isolation)
    return image[y1:y2, x1:x2].copy(), mask
