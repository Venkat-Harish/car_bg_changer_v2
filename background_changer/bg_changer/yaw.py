"""Which way the car faces in a photo (yaw), from the view label in the file name.

The view classifier's 8 labels are used at their bin centres. The true angle can be off by up to
~24 degrees (accepted); `sigma` tells the placement how uncertain it is, so a wrong label near an
end view does not throw the car sideways. This is the one place to plug in a better estimate later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .carmodel import ASPECT_BIAS, CarTable
from .probe import CarProbe, NameInfo

BIN_DEG = {"front": 0.0, "front-right": 45.0, "right": 90.0, "back-right": 135.0,
           "back": 180.0, "back-left": 225.0, "left": 270.0, "front-left": 315.0}
LABEL_SIGMA = 15.0  # degrees
LOW_CONF = 60  # percent; below this the label is treated as less certain
LOW_CONF_SIGMA = 25.0
SHAPE_SIGMA = 30.0


@dataclass(frozen=True)
class YawEstimate:
    deg: float
    sigma: float
    source: str  # label | shape | override
    flags: tuple[str, ...] = ()


def yaw_from_label(info: NameInfo | None) -> YawEstimate | None:
    if info is None:
        return None
    low = info.conf < LOW_CONF
    return YawEstimate(BIN_DEG[info.label], LOW_CONF_SIGMA if low else LABEL_SIGMA, "label",
                       ("low-conf-label",) if low else ())


def yaw_from_shape(probe: CarProbe, table: CarTable) -> YawEstimate:
    """No label: the bin whose predicted width/height best matches the cut-out (left/right and
    front-left/front-right look alike, so the side may be wrong)."""
    if probe.cut:
        return YawEstimate(90.0, SHAPE_SIGMA, "shape", ("no-view-label",))
    target = math.log(probe.aspect)
    best = min(BIN_DEG.values(), key=lambda y: abs(math.log(table.at(y).aspect / ASPECT_BIAS) - target))
    return YawEstimate(best, SHAPE_SIGMA, "shape", ("no-view-label",))


def estimate_yaw(info: NameInfo | None, probe: CarProbe, table: CarTable,
                 override: float | None = None) -> YawEstimate:
    if override is not None:
        return YawEstimate(float(override) % 360.0, 3.0, "override")
    return yaw_from_label(info) or yaw_from_shape(probe, table)
