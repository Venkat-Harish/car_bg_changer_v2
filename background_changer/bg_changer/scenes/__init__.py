"""One module per background. To add a background with its own logic, measure it, copy
background_2.py to <file name>.py, fill in the numbers and register it in SCENES below. Any other
background uses the generic scene, configured with [scene.<name>] in config.toml.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

from ..carmodel import CarTable
from . import background_1, background_2, background_3, generic
from .base import Scene, SceneSpec, with_overrides

SCENES: dict[str, SceneSpec] = {
    "background_1": background_1.SPEC,
    "background_2": background_2.SPEC,
    "background_3": background_3.SPEC,
}
GENERIC_MARGIN = 120  # px kept free around every view of the car on a generic scene
MODULE_MARGIN = 40  # on a calibrated scene (the EQB keeps >= 150); only tall or long cars get near it


def scene_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _fits(scene: Scene, margin: float) -> bool:
    w, h = scene.size_wh
    ax, ay = scene.spec.anchor
    p = scene.px_per_m
    bottom = min(h - 30, scene.spec.max_lowest_row) if scene.spec.max_lowest_row is not None else h - 30
    for yaw in range(0, 360, 5):
        r = scene.table.at(yaw)
        if min(ax + p * r.x0, w - (ax + p * r.x1), ay + p * r.top) < margin or ay + p * r.low > bottom:
            return False
    return True


def _fit(scene: Scene, margin: float) -> Scene:
    """Shrink the car until every view keeps `margin` px to the edges and its bottom on the stage."""
    from dataclasses import replace
    h, w = scene.size_wh[1], scene.size_wh[0]
    for _ in range(60):
        if _fits(scene, margin):
            break
        scene = Scene(replace(scene.spec, size=scene.spec.size * 0.97), scene.table, (h, w))
    return scene


def load_scene(name: str, plate: np.ndarray, table: CarTable, overrides: dict | None = None) -> Scene:
    """The background's own module if it has one, else the generic scene, with config overrides.
    Either way the car is made smaller (with a warning) if some view would not fit the picture."""
    key = scene_key(name)
    h, w = plate.shape[:2]
    if key in SCENES:
        scene = Scene(with_overrides(SCENES[key], overrides), table, plate.shape)
        fitted = _fit(scene, MODULE_MARGIN)
    else:
        scene = Scene(generic.make_spec(name, w, h, overrides), table, plate.shape)
        fitted = _fit(scene, GENERIC_MARGIN)
    if fitted.spec.size < scene.spec.size:
        warnings.warn(f"{name}: {table.shape.name} made {100 * (1 - fitted.spec.size / scene.spec.size):.0f}% smaller "
                      "so every view stays inside the picture", stacklevel=2)
    return fitted


__all__ = ["SCENES", "Scene", "SceneSpec", "load_scene", "scene_key"]
