"""Name a plate's paint colour, and how bright that paint looks next to white in the same light."""

from __future__ import annotations

import cv2
import numpy as np

# How bright each paint looks compared with white paint in the same light (grey-level ratio).
# Only light paints (factor >= MIN_FACTOR) are used to judge the light: dividing by a small,
# uncertain factor (dark green, blue, red) would magnify errors, so those plates rely on their
# white characters or on the photo instead. The darker values are listed for reference only.
# FIRST ESTIMATES from standard plate colours: check them on real photos of each plate type
# (photograph a white and a coloured plate side by side, divide their brightness) and edit here.
PAINT_FACTORS: dict[str, float] = {
    "white": 1.00,
    "yellow": 0.75,
    "orange": 0.60,
    "red": 0.30,
    "green": 0.35,
    "blue": 0.25,
    "black": 0.05,
}
MIN_FACTOR = 0.5

# Saturation (0-255, OpenCV HSV) below which a colour counts as neutral (white, grey, black).
# Real white plates measured 4-12; plate yellows are 200+.
NEUTRAL_SATURATION = 60


def name_colour(bgr) -> str:
    """'neutral', 'red', 'orange', 'yellow', 'green', 'teal', 'blue' or 'purple' for one BGR colour."""
    h, s, _ = cv2.cvtColor(np.uint8(bgr).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0].tolist()
    if s < NEUTRAL_SATURATION:
        return "neutral"
    # OpenCV hue runs 0-179 (degrees / 2).
    if h < 8 or h >= 165:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "teal"
    if h < 130:
        return "blue"
    return "purple"
