"""Pick how bright an overlay should be on a number plate, for any plate colour.

The camera only sees light x paint colour. To learn the light we need something on the plate
whose true colour we know. Readings are tried in this order; the first that works wins:

1. Something white on the plate: the background (most plates) or the characters (blue, green,
   red or black plates with white letters). How bright it looks is the light.
2. A light coloured paint (yellow taxi plates, yellow letters): how bright it looks, divided by
   how bright that paint is next to white (colours.PAINT_FACTORS, e.g. yellow 0.75).
3. The photo: how bright the brightest parts of the lower two-thirds of the image are (the car
   area, not the sky). Rougher; used when the plate can't be read.

Sanity check: a plate far brighter than everything else in the photo (a reflective plate lit by
a flash at night, or glare) is not trusted, and the photo reading is used instead.
Video: optionally smooth the main plate's reading over the last few frames (no flicker).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .colours import MIN_FACTOR, PAINT_FACTORS, name_colour

# Plate area kept clear of the plate's border line and rim (share of plate height), for a fitted
# outline and for a plain detector box (which also holds some bumper).
INSET_FITTED = 0.16
INSET_BOX = 0.22
# A side of the plate counts as its background when this share of its pixels is one connected
# piece (the background wraps round everything; characters are separate islands).
BACKGROUND_SHARE = 0.6
MIN_CONTRAST = 25  # grey levels between background and characters, below = unreadable plate
# Share of the light paint's pixels that must survive shaving 1 px off its edges. Below it, the
# paint (thin letters on a small plate) has no pure pixels: every one is blended with the dark
# part, so its brightness and colour can't be read. Measured: readable 45-83 %, tiny letters
# on a 25 px plate 8-16 %.
MIN_CORE_SHARE = 0.30
WORK_HEIGHT = 64  # plates taller than this (px) are measured on a shrunk copy


@dataclass(frozen=True)
class BrightnessReading:
    """How bright a white object would look on this plate, and where that answer came from."""

    level: float  # 0-255 at the plate's centre
    plane: tuple[float, float, float]  # the level across the plate: a + b*x + c*y (image pixels)
    source: str  # e.g. "white background", "white characters", "yellow background (/0.75)", "photo (...)"
    background: str  # plate background colour: "white", "yellow", "blue", ... or "unknown"
    characters: str  # character colour

    @property
    def percent(self) -> float:
        """The level as a share of full brightness, 0-100."""
        return min(100.0, 100.0 * self.level / 255)

    def level_at(self, x, y):
        a, b, c = self.plane
        return a + b * x + c * y


class BrightnessDetector:
    """Measure the light on plates. One instance per run (it remembers frames when smoothing).

    smoothing:  video only. Median of the main plate's reading over this many recent frames;
                0 = off (use 0 for unrelated photos).
    glow_ratio: a plate more than this many times brighter than the photo reading is not trusted.
    """

    def __init__(self, smoothing: int = 0, glow_ratio: float = 2.0, photo_percentile: float = 95.0):
        if smoothing < 0:
            raise ValueError("smoothing must be 0 or more")
        self.smoothing = smoothing
        self.glow_ratio = glow_ratio
        self.photo_percentile = photo_percentile
        self._history: deque[float] = deque(maxlen=max(1, smoothing))

    def reset(self) -> None:
        """Forget earlier frames (start of a new video)."""
        self._history.clear()

    # --- the photo-wide fallback -------------------------------------------------------
    def photo_level(self, image_bgr: np.ndarray) -> float:
        """Brightness of the brightest parts of the lower two-thirds of the photo (0-255)."""
        lower = image_bgr[image_bgr.shape[0] // 3:]
        step = max(1, max(lower.shape[:2]) // 480)  # every n-th pixel is plenty for a percentile
        gray = cv2.cvtColor(np.ascontiguousarray(lower[::step, ::step]), cv2.COLOR_BGR2GRAY)
        return max(1.0, float(np.percentile(gray, self.photo_percentile)))

    @staticmethod
    def _photo_reading(photo: float, why: str, background="unknown", characters="unknown") -> BrightnessReading:
        return BrightnessReading(photo, (photo, 0.0, 0.0), f"photo ({why})", background, characters)

    # --- one plate ---------------------------------------------------------------------
    def measure(self, image_bgr: np.ndarray, quad, fitted: bool = True, photo: float | None = None) -> BrightnessReading:
        """Reading for one plate. `quad`: its 4 corners (TL, TR, BR, BL) in image pixels.

        fitted: True if `quad` is the plate's own outline, False for a plain detector box.
        """
        if photo is None:
            photo = self.photo_level(image_bgr)
        q = np.asarray(quad, np.float32).reshape(4, 2)
        img_h, img_w = image_bgr.shape[:2]
        x0, y0 = max(0, math.floor(q[:, 0].min())), max(0, math.floor(q[:, 1].min()))
        x1, y1 = min(img_w, math.ceil(q[:, 0].max()) + 1), min(img_h, math.ceil(q[:, 1].max()) + 1)
        if x1 - x0 < 8 or y1 - y0 < 4:
            return self._photo_reading(photo, "plate too small")
        roi = image_bgr[y0:y1, x0:x1]
        height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        # A big plate is shrunk to about WORK_HEIGHT px tall: plenty to tell background from
        # characters, and several times faster. `scale` maps work pixels back to image pixels.
        scale = 1.0
        if height > WORK_HEIGHT:
            scale = WORK_HEIGHT / height
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            height = WORK_HEIGHT

        # The plate area, kept clear of its border line and rim.
        area = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillConvexPoly(area, np.round((q - np.float32([x0, y0])) * scale).astype(np.int32), 1)
        inset = (INSET_FITTED if fitted else INSET_BOX) * height
        if inset > 0:  # keep pixels at least `inset` px inside the outline (fast, any inset)
            area = (cv2.distanceTransform(area, cv2.DIST_L2, 3) > inset).astype(np.uint8)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        vals = gray[area > 0]
        if vals.size < 60:
            return self._photo_reading(photo, "plate too small")

        # Split into the light and the dark side (background vs characters, in some order).
        thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light = (gray > thr) & (area > 0)
        dark = (gray <= thr) & (area > 0)
        if light.sum() < 20 or dark.sum() < 20:
            return self._photo_reading(photo, "plate unreadable: one colour only")
        if float(np.median(gray[light])) - float(np.median(gray[dark])) < MIN_CONTRAST:
            return self._photo_reading(photo, "plate unreadable: too little contrast")
        light_core, dark_core = _core(light), _core(dark)  # drop the blurred pixels where they meet
        light_name = _paint_name(np.median(roi[light_core], axis=0), light=True)
        dark_name = _paint_name(np.median(roi[dark_core], axis=0), light=False)
        side = _background_side(light, dark)
        if side == "light":
            background, characters, part = light_name, dark_name, "background"
        elif side == "dark":
            background, characters, part = dark_name, light_name, "characters"
        else:
            background, characters, part = "unknown", "unknown", "paint"
        shaved = cv2.erode(light.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if shaved.sum() < MIN_CORE_SHARE * light.sum():
            return self._photo_reading(photo, f"plate too small to read its paint: {light_name} {part}",
                                       background, characters)

        # The ladder: white -> light coloured paint -> photo.
        if light_name == "white":
            factor, source = 1.0, f"white {part}"
        elif PAINT_FACTORS.get(light_name, 0.0) >= MIN_FACTOR:
            factor = PAINT_FACTORS[light_name]
            source = f"{light_name} {part} (/{factor:g})"
        else:
            return self._photo_reading(photo, f"no white or light paint on the plate: {light_name}/{dark_name}",
                                       background, characters)

        plane = _fit_plane(gray, light_core, x0, y0, scale)
        if plane is None:
            return self._photo_reading(photo, "plate unreadable", background, characters)
        plane = tuple(v / factor for v in plane)
        cx, cy = q.mean(0)
        level = float(np.clip(plane[0] + plane[1] * cx + plane[2] * cy, 1, 255))
        if level > self.glow_ratio * photo:
            return self._photo_reading(photo, f"plate much brighter than the photo, glare or flash? ({source})",
                                       background, characters)
        return BrightnessReading(level, plane, source, background, characters)

    # --- all plates in one image ----------------------------------------------------------
    def measure_all(self, image_bgr: np.ndarray, quads, fitted=None) -> list[BrightnessReading]:
        """Readings for every plate in one image (measured on the original, before any painting)."""
        quads = [np.asarray(q, np.float32).reshape(4, 2) for q in quads]
        if not quads:
            return []
        fitted = list(fitted) if fitted is not None else [True] * len(quads)
        photo = self.photo_level(image_bgr)
        readings = [self.measure(image_bgr, q, f, photo) for q, f in zip(quads, fitted)]
        if self.smoothing > 0:
            main = max(range(len(quads)), key=lambda i: cv2.contourArea(quads[i]))  # the main car's plate
            r = readings[main]
            self._history.append(r.level)
            smoothed = statistics.median(self._history)
            if abs(smoothed - r.level) > 0.5:
                scale = smoothed / r.level
                readings[main] = dataclasses.replace(
                    r, level=smoothed, plane=tuple(v * scale for v in r.plane),
                    source=f"{r.source}, smoothed over {len(self._history)} frames",
                )
        return readings


def _core(mask: np.ndarray) -> np.ndarray:
    """Mask without its outer pixel (blurred edges between characters and background); unchanged if that empties it."""
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return eroded if eroded.sum() >= 20 else mask


def _paint_name(bgr, light: bool) -> str:
    """Colour name; a neutral light side is white paint (in whatever light), a neutral dark side black."""
    name = name_colour(np.clip(bgr, 0, 255))
    if name != "neutral":
        return name
    if light:
        return "white"
    return "black" if max(bgr) < 110 else "grey"


def _background_side(light: np.ndarray, dark: np.ndarray) -> str | None:
    shares = {}
    for side, part in (("light", light), ("dark", dark)):
        _, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        shares[side] = areas.max() / areas.sum() if len(areas) else 0.0
    side = max(shares, key=shares.get)
    return side if shares[side] >= BACKGROUND_SHARE and min(shares.values()) < BACKGROUND_SHARE else None


def _fit_plane(gray: np.ndarray, mask: np.ndarray, x0: int, y0: int, scale: float = 1.0
               ) -> tuple[float, float, float] | None:
    """Brightness of the masked pixels as a plane a + b*x + c*y in image pixels; one refit drops glare.

    `gray`/`mask` may be a shrunk copy of the image area starting at (x0, y0); `scale` is that shrink.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    if len(xs) > 4000:
        pick = np.linspace(0, len(xs) - 1, 4000).astype(int)
        xs, ys = xs[pick], ys[pick]
    vals = gray[ys, xs].astype(np.float64)
    a = np.column_stack([np.ones(len(xs)), x0 + (xs + 0.5) / scale - 0.5, y0 + (ys + 0.5) / scale - 0.5])
    coef = np.linalg.lstsq(a, vals, rcond=None)[0]
    resid = vals - a @ coef
    keep = np.abs(resid) <= 2.5 * max(resid.std(), 1.0)
    if keep.sum() >= 20:
        coef = np.linalg.lstsq(a[keep], vals[keep], rcond=None)[0]
    return float(coef[0]), float(coef[1]), float(coef[2])
