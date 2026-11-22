"""The same scene (a background's measured stage) for its picture shrunk to a smaller size.

Everything a scene module measured in pixels on the full-size picture is scaled with the picture:
stage centre, stage size (side_px, so px per metre), the floor camera, the floor mask, the turntable
rim, the lowest-row limit and the reflection blur. Sizes in metres (shadow widths and softness) need
nothing: they follow px per metre. The few fixed pixel sizes in the code (mask feather, guard shift,
debug marks) follow SceneSpec.px_scale.

Point coordinates use pixel centres, as cv2.resize does:  x_small = (x + 0.5) * k - 0.5.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from bg_changer.grounding import HomographyFloor, LevelFloor, PlateFloor
from bg_changer.scenes.base import Scene, SceneSpec


def _point(v: float, k: float) -> float:
    return (v + 0.5) * k - 0.5


def _scaled_floor(floor: PlateFloor, kx: float, ky: float, anchor: tuple[float, float]) -> PlateFloor:
    if isinstance(floor, LevelFloor):  # row = horizon + k / (z0 + s)
        return LevelFloor(_point(floor.horizon, ky), floor.k * ky, floor.z0)
    if isinstance(floor, HomographyFloor):  # floor metres -> image px, then px -> smaller px
        to_small = np.array([[kx, 0.0, 0.5 * kx - 0.5], [0.0, ky, 0.5 * ky - 0.5], [0.0, 0.0, 1.0]])
        return HomographyFloor(to_small @ floor.h, anchor)
    raise TypeError(f"background_scaler cannot scale a {type(floor).__name__} floor; add it to _scaled_floor")


def _scaled_mask(mask_fn, full_wh: tuple[int, int]):
    """The scene's floor mask, drawn at full size and shrunk like the picture (soft at its edges)."""
    w_full, h_full = full_wh

    def floor_mask(width: int, height: int) -> np.ndarray:
        full = np.ascontiguousarray(mask_fn(w_full, h_full), dtype=np.float32)
        return cv2.resize(full, (width, height), interpolation=cv2.INTER_AREA)

    return floor_mask


def scale_spec(spec: SceneSpec, size: tuple[int, int]) -> SceneSpec:
    """`spec` for its picture shrunk to `size` (width, height)."""
    w_full, h_full = spec.plate_size
    w, h = int(size[0]), int(size[1])
    kx, ky = w / w_full, h / h_full
    anchor = (_point(spec.anchor[0], kx), _point(spec.anchor[1], ky))
    changes = dict(plate_size=(w, h), anchor=anchor, side_px=spec.side_px * kx,
                   floor_mask=_scaled_mask(spec.floor_mask, spec.plate_size), px_scale=spec.px_scale * kx)
    if spec.plate_floor is not None:
        changes["plate_floor"] = _scaled_floor(spec.plate_floor, kx, ky, anchor)
    if spec.horizon_y is not None:
        changes["horizon_y"] = _point(spec.horizon_y, ky)
    if spec.reflection is not None:
        changes["reflection"] = replace(spec.reflection, blur_px=tuple(b * kx for b in spec.reflection.blur_px))
    if spec.ring is not None:
        cx, cy, a, b = spec.ring
        changes["ring"] = (_point(cx, kx), _point(cy, ky), a * kx, b * ky)
    if spec.max_lowest_row is not None:
        changes["max_lowest_row"] = _point(spec.max_lowest_row, ky)
    return replace(spec, **changes)


def scale_scene(scene: Scene, plate: np.ndarray, size: tuple[int, int]) -> tuple[Scene, np.ndarray]:
    """(scene, picture) shrunk to `size` (width, height); the picture with INTER_AREA. At full size
    both come back unchanged. `scene` is the full-size scene from bg_changer.load_scene."""
    size = (int(size[0]), int(size[1]))
    if size == tuple(scene.size_wh):
        return scene, plate
    if size[0] > scene.size_wh[0] or size[1] > scene.size_wh[1] or min(size) < 1:
        raise ValueError(f"background_scaler only shrinks: {scene.size_wh} -> {size}")
    small = cv2.resize(plate, size, interpolation=cv2.INTER_AREA)
    return Scene(scale_spec(scene.spec, size), scene.table, small.shape), small
