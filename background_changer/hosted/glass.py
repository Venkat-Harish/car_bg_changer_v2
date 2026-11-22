"""Glass: SAM 3 "car window" mask + a deterministic dark tint. The ONLY step that changes car pixels.

User decision (Nov 2026): all glass is tinted, windscreen included; the untouched frame is always saved next to the
output (<out>/originals/). Outside the window mask the frame stays bit-identical.

Contract (stub; Agent B implements):
- window_mask(client, image_bgr, car_mask, max_side=2048) -> bool HxW (glass of THIS car only; inside the eroded
  car mask).
- tint_glass(image_bgr, mask, strength=0.85, keep=0.15) -> new BGR uint8 image; pixels outside `mask` are
  bit-identical to the input; deterministic (same input -> same output, no randomness).
"""

from __future__ import annotations

import numpy as np

from .fal import FalClient


def window_mask(client: FalClient, image: np.ndarray, car_mask: np.ndarray, max_side: int = 2048) -> np.ndarray:
    raise NotImplementedError


def tint_glass(image: np.ndarray, mask: np.ndarray, strength: float = 0.85, keep: float = 0.15) -> np.ndarray:
    raise NotImplementedError


__all__ = ["tint_glass", "window_mask"]
