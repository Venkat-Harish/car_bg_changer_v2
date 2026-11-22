"""background_2: grey studio with curved walls and a matte concrete floor.

Measured from the picture (read-only calibration, docs/findings.md):
- Pinhole camera f = 1213 px (from the square floor-tile repeat), principal point (959.9, 512.5),
  pitched 1.62 deg up, yawed 1.71 deg, roll 0; horizon row 546.8 (three independent measures).
- Camera height assumed 1.0 m (absolute scale is not visible; with a side-matched stage it does not
  change the picture). Room 7.9 x 9.1 m, back-wall panels 1.6 m.
- Anchor (960, 780): the end views' shadows stay inside the frame; side_px 1243 (~214 px/m for the EQB).
- Matte floor (no mirrored features), large soft overhead light: soft shadow, nudged 0.10 m toward the
  camera. No reflection.
"""

from __future__ import annotations

import numpy as np

from ..grounding import HomographyFloor, ShadowStyle
from .base import SceneSpec

_K = np.array([[1213.0, 0.0, 959.9], [0.0, 1213.0, 512.5], [0.0, 0.0, 1.0]])
_R = np.array([[0.999555, 0.0, 0.029818], [-0.000843, -0.999600, 0.028276], [-0.029806, 0.028288, 0.999155]])
_CAM_H = 1.0
# Floor (X right, Z forward, 1) in metres -> image: K [r1, r3, -h r2]
FLOOR_H = _K @ np.column_stack([_R[:, 0], _R[:, 2], -_CAM_H * _R[:, 1]])

# Wall/floor junction (plinth bottom) measured along the image width.
_JUNCTION = np.array([
    (0, 781.9), (350, 699.3),
    (352, 699.8), (384, 694.3), (416, 689.8), (448, 686.3), (480, 683.7), (512, 681.9), (544, 680.5),
    (576, 679.7), (601, 678.9),
    (1324, 681.3), (1344, 682.0), (1376, 683.0), (1408, 684.2), (1440, 686.2), (1472, 689.1), (1504, 692.8),
    (1536, 697.3), (1568, 703.0), (1575, 703.7),
    (1919, 797.2),
])


# Public names for car_fit_to_bg.rooms (the 3D room of this background).
K, R, CAM_H, JUNCTION = _K, _R, _CAM_H, _JUNCTION


def floor_mask(width: int, height: int) -> np.ndarray:
    junction = np.interp(np.arange(width), _JUNCTION[:, 0], _JUNCTION[:, 1])
    yy = np.arange(height, dtype=np.float32)[:, None]
    return yy > junction[None, :] + 1.0


ANCHOR = (960.0, 780.0)

SPEC = SceneSpec(
    name="background_2",
    plate_size=(1920, 1080),
    anchor=ANCHOR,
    side_px=1243.0,
    floor_mask=floor_mask,
    plate_floor=HomographyFloor(FLOOR_H, ANCHOR),
    shadow=ShadowStyle(tyre=0.85, tyre_sigma_m=0.035, core=0.55, core_inset_m=0.10, core_sigma_m=0.20,
                       skirt=0.25, skirt_grow_m=0.30, skirt_sigma_m=0.28, contact=0.80, gap=0.65, band=0.35,
                       shift_m=0.05),
    notes="matte concrete; soft overhead strips",
)
