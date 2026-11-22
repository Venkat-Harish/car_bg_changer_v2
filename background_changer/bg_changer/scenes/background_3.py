"""background_3: outdoor race track (asphalt, red/white barriers).

Measured from the picture (read-only calibration, docs/findings.md):
- Horizon row 397 (+-4), roll 0; camera 1.24 m above the asphalt, from nine 0.80 m precast jersey
  barriers (sd 0.07, cross-checked with the tipper truck at 3.1 m); f ~ 1400 px, pitch 5.83 deg down.
- Ground homography (X right, D forward, 1) -> image: [[1400, 955.03, 120.96], [0, 394.96, 1795.06],
  [0, 0.994825, 0.126]].
- Anchor (960, 685) on open asphalt, ~6 m from the camera; the car's far end stays in front of the
  barriers (their bases are at rows 457-472). side_px 1232: the EQB at ~212 px/m, which matches its size
  next to the barriers, and V's eye level (397.8) matches the plate horizon.
- Hazy daylight, matte asphalt, no hard cast shadows; contact shadows are warm-neutral (barrier
  contact shadows multiply the asphalt by (0.70, 0.67, 0.64) RGB). The plate is ~17% darker in the
  outer columns, so the car stays centred.
"""

from __future__ import annotations

import numpy as np

from ..grounding import HomographyFloor, ShadowStyle
from .base import SceneSpec

GROUND_H = np.array([[1400.0, 955.03, 120.96], [0.0, 394.96, 1795.06], [0.0, 0.994825, 0.12600]])
ANCHOR = (960.0, 685.0)


def floor_mask(width: int, height: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    return np.broadcast_to(yy >= 475, (height, width))


SPEC = SceneSpec(
    name="background_3",
    plate_size=(1920, 1080),
    anchor=ANCHOR,
    side_px=1232.0,
    floor_mask=floor_mask,
    plate_floor=HomographyFloor(GROUND_H, ANCHOR),
    shadow=ShadowStyle(tyre=0.80, tyre_sigma_m=0.04, core=0.50, core_inset_m=0.10, core_sigma_m=0.20,
                       skirt=0.22, skirt_grow_m=0.35, skirt_sigma_m=0.30, contact=0.75, gap=0.55, band=0.30,
                       tint_gamma_bgr=(1.114, 1.0, 0.889)),  # grey 0.67 -> (0.70, 0.67, 0.64) RGB
    notes="hazy daylight; warm-neutral contact shadow",
)
