"""Put a cut-out car on a new background.

The car's pixels go through exactly one operation: a uniform downscale with INTER_AREA (plain area
averaging: no sharpening, no ringing, no colour change). Where the car's alpha is 1, the output pixel IS
that downscaled pixel. Only the thin soft edge is blended with the background. Where and how big the car
is comes from the background's scene module (bg_changer/scenes); shadows and reflections are drawn on
the background before the car goes on top, so they never touch the car.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .segment import CarMask

if TYPE_CHECKING:  # avoid an import cycle: scenes import paste() from here
    from .placement import Plan
    from .probe import CarProbe
    from .scenes.base import Scene
    from .yaw import YawEstimate

DEFAULT_SHADOW = 1.0  # gain on each scene's calibrated shadow


def paste(canvas: np.ndarray, car: np.ndarray, alpha: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-blend `car` onto `canvas` at (x0, y0), in place, clipped to the canvas.
    Where alpha == 1 the car pixel is copied as is (no float round trip)."""
    H, W = canvas.shape[:2]
    h, w = car.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 >= cx2 or cy1 >= cy2:
        return
    part = car[cy1 - y0 : cy2 - y0, cx1 - x0 : cx2 - x0]
    a = alpha[cy1 - y0 : cy2 - y0, cx1 - x0 : cx2 - x0]
    region = canvas[cy1:cy2, cx1:cx2]
    a3 = a[..., None]
    blended = np.rint(region.astype(np.float32) * (1 - a3) + part.astype(np.float32) * a3).astype(np.uint8)
    opaque = a >= 1.0
    blended[opaque] = part[opaque]
    canvas[cy1:cy2, cx1:cx2] = blended


def scale_car(image_bgr: np.ndarray, car: CarMask, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The car's crop and alpha, downscaled to `size` (w, h) with INTER_AREA. Alpha that averages to
    1 (within float error) stays exactly 1, so those pixels are copied unchanged."""
    x1, y1, x2, y2 = car.crop
    pixels = image_bgr[y1:y2, x1:x2]
    alpha = car.alpha[y1:y2, x1:x2]
    if tuple(size) != (x2 - x1, y2 - y1):
        pixels = cv2.resize(pixels, tuple(size), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, tuple(size), interpolation=cv2.INTER_AREA)
    alpha = alpha.copy()
    alpha[alpha > 1 - 1e-4] = 1.0
    return pixels, alpha


def place_car(image_bgr: np.ndarray, car: CarMask, background_bgr: np.ndarray, scene: "Scene",
              probe: "CarProbe", yaw: "YawEstimate", cut_policy: str = "anchor",
              shadow: float = DEFAULT_SHADOW) -> tuple[np.ndarray | None, "Plan | None"]:
    """Return (new image the size of the background, where the car went), or (None, None) when the
    frame is skipped (cut-off car with cut_policy 'skip')."""
    plan = scene.plan(car, probe, yaw, cut_policy)
    if plan is None:
        return None, None
    pixels, alpha = scale_car(image_bgr, car, plan.size)
    return scene.compose(background_bgr, pixels, alpha, plan, probe, shadow), plan


def cutout_bgra(image_bgr: np.ndarray, car: CarMask) -> np.ndarray:
    """The car alone at full source resolution, with its alpha as a 4th channel (for review or reuse)."""
    x1, y1, x2, y2 = car.crop
    a = np.rint(car.alpha[y1:y2, x1:x2] * 255).astype(np.uint8)
    return np.dstack([image_bgr[y1:y2, x1:x2], a])
