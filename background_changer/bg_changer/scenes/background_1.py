"""background_1: white studio with a glossy turntable.

Measured from the picture (read-only calibration, docs/findings.md):
- Level camera (pitch 0, roll 0), f = 1702.5 px, principal point (959.6, 502.5), horizon row 502.5
  (vanishing point of the side-wall floor LED strips).
- The TRUE disc centre is (958.8, 764.0), where the 8 radial seams meet. The rim ellipse's centre
  (958.7, 833.7) is 1.4 m nearer the camera; the old pipeline stood cars even lower (row 907).
- Rim ellipse: centre (958.73, 833.71), semi-axes 880.32 x 152.12; near rim row 985.8.
- With a 6.0 m disc: camera 1.00 m high, 6.53 m from the centre; floor row = 502.5 + 1707.6/(6.532 + s).
- side_px 1348: a side view through V spans ~73% of the 1761 px rim, like a real car on a turntable
  (the plate's own camera shows the EQB's side at 232 px/m at the disc centre).
- Glossy disc and floor (mirrored LED streaks), soft symmetric light, no key light: neutral soft shadow.
"""

from __future__ import annotations

import numpy as np

from ..grounding import LevelFloor, ReflectionStyle, ShadowStyle
from .base import SceneSpec

# The level plate camera (public: car_fit_to_bg.rooms rebuilds this background as a 3D room).
FOCAL = 1702.5
PRINCIPAL_POINT = (959.6, 502.5)


def floor_mask(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return (yy >= 640) & (yy >= -0.196465 * xx + 693.601) & (yy >= 0.203676 * xx + 304.434)


SPEC = SceneSpec(
    name="background_1",
    plate_size=(1920, 1080),
    anchor=(958.8, 764.0),
    side_px=1348.0,
    floor_mask=floor_mask,
    plate_floor=LevelFloor(horizon=502.5, k=1707.6, z0=6.532),
    shadow=ShadowStyle(tyre=0.85, tyre_sigma_m=0.035, core=0.55, core_inset_m=0.10, core_sigma_m=0.20,
                       skirt=0.25, skirt_grow_m=0.30, skirt_sigma_m=0.25, contact=0.80, gap=0.65, band=0.35),
    reflection=ReflectionStyle(opacity=0.18, fade_m=0.6, compress=0.90, blur_px=(3.0, 10.0, 25.0),
                               env_linear=0.60, outside_factor=0.6),
    ring=(958.73, 833.71, 880.32, 152.12),
    max_lowest_row=946.0,  # keep bumpers 40 px above the near rim (985.8)
    notes="turntable centre = seam crossing; car side view ~73% of the disc",
)
