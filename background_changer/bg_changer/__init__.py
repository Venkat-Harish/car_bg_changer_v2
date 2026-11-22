"""Cut the car out of a photo and place it on a background the way a real car would stand there.
CPU only, ONNX Runtime (via rembg), no torch. The car's own pixels are only ever scaled down.

    import cv2
    from bg_changer import BackgroundChanger, CarShape

    changer = BackgroundChanger(car=CarShape(4.684, 1.834, 1.667, 2.829))   # metres
    image = cv2.imread("0012.0s_right_95pct.jpg")                           # BGR
    background = cv2.imread("backgrounds/background_1.webp")
    result, plan = changer.process(image, background, "background_1", stem="0012.0s_right_95pct")
    cv2.imwrite("car_studio.webp", result)

Each background has a scene module (bg_changer/scenes) that knows where its turntable or road spot is
and how its floor recedes. The car's view angle comes from the label in the file name
(`<time>s_<view>_<conf>pct`), its proportions from its dimensions.
"""

from __future__ import annotations

import numpy as np

from .carmodel import ASPECT_BIAS, CAPTURE_DISTANCE_M, CAPTURE_ELEVATION_DEG, CarShape, CarTable, ViewRow
from .cars import CarSpecError, load_cars, parse_cars, select_car
from .compose import DEFAULT_SHADOW, cutout_bgra, paste, place_car, scale_car
from .placement import CUT_POLICIES, Plan, plan_on_anchor
from .probe import CarProbe, NameInfo, parse_name, probe_car
from .scenes import SCENES, Scene, load_scene
from .segment import DEFAULT_EDGE_SHRINK, DEFAULT_MODEL, DEFAULT_THREADS, CarMask, CarSegmenter, NoCarFound, clean_alpha
from .yaw import BIN_DEG, YawEstimate, estimate_yaw

__all__ = [
    "ASPECT_BIAS", "BIN_DEG", "CAPTURE_DISTANCE_M", "CAPTURE_ELEVATION_DEG", "CUT_POLICIES",
    "DEFAULT_EDGE_SHRINK", "DEFAULT_MODEL", "DEFAULT_SHADOW", "DEFAULT_THREADS",
    "BackgroundChanger", "CarMask", "CarProbe", "CarSegmenter", "CarShape", "CarSpecError", "CarTable",
    "NameInfo", "NoCarFound", "Plan", "SCENES", "Scene", "ViewRow", "YawEstimate",
    "clean_alpha", "cutout_bgra", "estimate_yaw", "load_cars", "load_scene", "parse_cars", "parse_name",
    "paste", "place_car", "plan_on_anchor", "probe_car", "scale_car", "select_car",
]


class BackgroundChanger:
    """Segmentation model + car table; scenes are built per background on first use."""

    def __init__(self, model: str = DEFAULT_MODEL, edge_shrink: int = DEFAULT_EDGE_SHRINK,
                 threads: int = DEFAULT_THREADS, car: CarShape | None = None, shadow: float = DEFAULT_SHADOW,
                 cut_policy: str = "anchor"):
        self.segmenter = CarSegmenter(model, edge_shrink, threads)
        self.table = CarTable(car or CarShape())
        self.shadow = shadow
        self.cut_policy = cut_policy
        self._scenes: dict[tuple[str, tuple[int, ...]], Scene] = {}

    def scene(self, name: str, background_bgr: np.ndarray, overrides: dict | None = None) -> Scene:
        key = (name, background_bgr.shape, repr(sorted((overrides or {}).items())))
        if key not in self._scenes:
            self._scenes[key] = load_scene(name, background_bgr, self.table, overrides)
        return self._scenes[key]

    def cut_out(self, image_bgr: np.ndarray) -> CarMask:
        return self.segmenter.cut_out(image_bgr)

    def place(self, image_bgr: np.ndarray, car: CarMask, background_bgr: np.ndarray, scene: str = "background_1",
              stem: str | None = None, yaw_deg: float | None = None, overrides: dict | None = None):
        """Return (image or None if skipped, Plan or None). `stem` is the photo's file name without
        extension, used for its view label; `yaw_deg` overrides it; `overrides` are [scene.<name>] settings."""
        probe = probe_car(car)
        yaw = estimate_yaw(parse_name(stem) if stem else None, probe, self.table, yaw_deg)
        return place_car(image_bgr, car, background_bgr, self.scene(scene, background_bgr, overrides), probe, yaw,
                         self.cut_policy, self.shadow)

    def process(self, image_bgr: np.ndarray, background_bgr: np.ndarray, scene: str = "background_1",
                stem: str | None = None, yaw_deg: float | None = None, overrides: dict | None = None):
        """Cut out and place in one call. Raises NoCarFound if nothing is found."""
        return self.place(image_bgr, self.cut_out(image_bgr), background_bgr, scene, stem, yaw_deg, overrides)
