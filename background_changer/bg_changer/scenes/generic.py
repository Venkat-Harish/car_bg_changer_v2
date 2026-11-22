"""A background without its own module: placement from a few numbers in config.toml.

    [scene.my_showroom]
    anchor = [960, 760]   # where the car's footprint centre stands (px); default (0.5 W, 0.72 H)
    side_fill = 0.66      # a side view's length as a share of the width; default 0.66
    horizon_y = 520       # optional: the floor's horizon row, for realistic depth in front of the car
    floor = "matte"       # or "glossy" for a faint floor reflection

For the best result, measure the background like background_1-3 and give it its own module.
"""

from __future__ import annotations

from ..grounding import ReflectionStyle, ShadowStyle
from .base import SceneSpec

DEFAULT_ANCHOR = (0.5, 0.72)  # shares of width / height
DEFAULT_SIDE_FILL = 0.66
GENERIC_SHADOW = ShadowStyle()  # the studio defaults (background_1/2 values)
GLOSSY = ReflectionStyle(opacity=0.18 * 0.8)


def make_spec(name: str, width: int, height: int, settings: dict | None) -> SceneSpec:
    settings = settings or {}
    if "anchor" in settings:
        ax, ay = (float(v) for v in settings["anchor"])
    else:
        ax, ay = DEFAULT_ANCHOR[0] * width, DEFAULT_ANCHOR[1] * height
    horizon = settings.get("horizon_y")
    glossy = settings.get("floor", "matte") == "glossy" and settings.get("reflection", True) is not False
    top = float(horizon) + 10 if horizon is not None else 0.0

    def floor_mask(w: int, h: int):
        import numpy as np
        yy = np.arange(h, dtype=np.float32)[:, None]
        return np.broadcast_to(yy >= top, (h, w))

    return SceneSpec(
        name=name,
        plate_size=(width, height),
        anchor=(ax, ay),
        side_px=float(settings.get("side_fill", DEFAULT_SIDE_FILL)) * width,
        floor_mask=floor_mask,
        shadow=GENERIC_SHADOW,
        horizon_y=float(horizon) if horizon is not None else None,
        reflection=GLOSSY if glossy else None,
        size=float(settings.get("size", 1.0)),
        shadow_gain=float(settings.get("shadow", 1.0)),
        notes="generic scene (no module)",
    )
