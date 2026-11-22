"""Mask detected plates. Each mask style is one small function, looked up by name in MASKERS.

A masker has the signature

    masker(image, quad, **opts) -> None

It paints `image` (BGR, modified in place) inside `quad`: the plate's 4 corners as a float32
array (top-left, top-right, bottom-right, bottom-left), already padded. Each masker takes the
options it needs by keyword and ignores the rest.

Built in:
    "solid"  fill the plate shape with one colour          (option: color, RGB)
    "image"  warp an image onto the plate's 4 corners       (options: overlay, BGR or BGRA array;
             lighting, 0-100 %, default 100: match it to the original plate's brightness)

Plates whose corners couldn't be found (plate.fit == "box") get "solid" by default: an image
stretched over a straight box on a tilted plate, a grille or a badge looks wrong. (When corner
fitting is switched off on purpose, every plate gets the chosen style on its box.)

To add a style, write one function with the signature above and register it:

    MASKERS["blur"] = my_blur
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .corners import grow_quad, quad_size
from .detector import Plate

Masker = Callable[..., None]

DEFAULT_PAD = 0.0
DEFAULT_COLOR_RGB = (0, 0, 0)
_SHIFT = 4  # sub-pixel bits for OpenCV polygon drawing (1/16 px)
DEFAULT_LIGHTING = 100.0  # % of the original plate's lighting applied to the overlay
LIGHT_GAIN_MIN = 0.15  # never darken the overlay below 15 % of its own brightness


def _fixed(quad: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(quad, np.float64) * (1 << _SHIFT)).astype(np.int32)


def solid_fill(image: np.ndarray, quad: np.ndarray, *, color=DEFAULT_COLOR_RGB, **_) -> None:
    """Fill the plate shape with one colour. `color` is RGB (converted to OpenCV's BGR here)."""
    r, g, b = color
    cv2.fillConvexPoly(image, _fixed(quad), (b, g, r), lineType=cv2.LINE_AA, shift=_SHIFT)


def _plate_light(roi: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Brightness (0-255) of the original plate's white face across `roi`, or None if unclear.

    Samples the inner part of the plate (clear of its border and rim), keeps the light pixels
    (the face, not the characters), and fits a plane, so a plate that is half in shade gives a
    gradient. One refit drops outliers such as glare.
    """
    _, height = quad_size(quad)
    inner = grow_quad(quad, [-0.2 * height] * 4)
    region = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, _fixed(inner), 255, shift=_SHIFT)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    inside = gray[region > 0]
    if inside.size < 50:
        return None
    threshold, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.nonzero((region > 0) & (gray > threshold))
    if len(xs) < 0.2 * inside.size:  # too little light face to judge
        return None
    if len(xs) > 4000:
        pick = np.linspace(0, len(xs) - 1, 4000).astype(int)
        xs, ys = xs[pick], ys[pick]
    vals = gray[ys, xs].astype(np.float64)
    a = np.column_stack([np.ones(len(xs)), xs, ys]).astype(np.float64)
    coef = np.linalg.lstsq(a, vals, rcond=None)[0]
    resid = vals - a @ coef
    keep = np.abs(resid) <= 2.5 * max(resid.std(), 1.0)
    if keep.sum() >= 20:
        coef = np.linalg.lstsq(a[keep], vals[keep], rcond=None)[0]
    gy, gx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    return (coef[0] + coef[1] * gx + coef[2] * gy).astype(np.float32)


def image_overlay(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    overlay: np.ndarray,
    lighting: float = DEFAULT_LIGHTING,
    light_plane: tuple[float, float, float] | None = None,
    **_,
) -> None:
    """Warp `overlay` (BGR, or BGRA with transparency) so its corners land on the plate's corners.

    lighting (0-100 %): how far to match the overlay to the original plate's brightness.
    0 = the overlay as it is; 100 = its white looks like the plate's white did in this photo
    (sun, shade, a gradient across the plate); 50 = halfway.
    light_plane: (a, b, c) from plate_brightness_detector: a white object here would look
    a + b*x + c*y bright (image pixels). When given, it sets the brightness and `lighting` is not used.
    """
    h, w = image.shape[:2]
    q = np.asarray(quad, np.float32)
    x0, y0 = max(0, math.floor(q[:, 0].min())), max(0, math.floor(q[:, 1].min()))
    x1, y1 = min(w, math.ceil(q[:, 0].max()) + 1), min(h, math.ceil(q[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    dst = q - np.float32([x0, y0])
    # Shrink a big overlay to about the plate's size first (warpPerspective has no area filter).
    edge = lambda i, j: float(np.linalg.norm(q[j] - q[i]))  # noqa: E731
    tw = max(2, round(max(edge(0, 1), edge(3, 2))))
    th = max(2, round(max(edge(0, 3), edge(1, 2))))
    if overlay.shape[1] > 1.5 * tw or overlay.shape[0] > 1.5 * th:
        overlay = cv2.resize(overlay, (tw, th), interpolation=cv2.INTER_AREA)
    oh, ow = overlay.shape[:2]
    # Map the overlay's outer pixel edges (not its pixel centres) onto the plate's corners.
    src = np.float32([[-0.5, -0.5], [ow - 0.5, -0.5], [ow - 0.5, oh - 0.5], [-0.5, oh - 0.5]])
    m = cv2.getPerspectiveTransform(src, dst)
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(overlay, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    alpha = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillConvexPoly(alpha, _fixed(dst), 255, lineType=cv2.LINE_AA, shift=_SHIFT)
    alpha = alpha.astype(np.float32) / 255
    if warped.ndim == 3 and warped.shape[2] == 4:  # the PNG's own transparency
        alpha *= warped[:, :, 3].astype(np.float32) / 255
        warped = warped[:, :, :3]
    warped = warped.astype(np.float32)
    strength = min(max(float(lighting), 0.0), 100.0) / 100
    if light_plane is not None:  # automatic: plate_brightness_detector already measured the light
        a, b, c = light_plane
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        warped *= np.clip((a + b * gx + c * gy) / 255, LIGHT_GAIN_MIN, 1.0)[:, :, None]
    elif strength > 0:
        light = _plate_light(roi, dst)  # read before painting: roi still holds the original plate
        if light is not None:
            # Full match: overlay white (255) becomes the plate's white; darker colours scale with it.
            full = np.clip(light / 255, LIGHT_GAIN_MIN, 1.0)
            warped *= (1 - strength * (1 - full))[:, :, None]
    a = alpha[:, :, None]
    roi[:] = np.clip(roi.astype(np.float32) * (1 - a) + warped * a + 0.5, 0, 255).astype(np.uint8)


MASKERS: dict[str, Masker] = {"solid": solid_fill, "image": image_overlay}


def pad_quad(quad: np.ndarray, pad: float) -> np.ndarray:
    """Grow the shape by `pad` x its own height on every side (0 = unchanged)."""
    if pad <= 0:
        return np.asarray(quad, np.float32)
    _, height = quad_size(quad)
    return grow_quad(quad, [pad * height] * 4)


def mask_plates(
    image_bgr: np.ndarray,
    plates: list[Plate],
    style: str = "solid",
    pad: float = DEFAULT_PAD,
    box_style: str | None = "solid",
    **opts,
) -> np.ndarray:
    """Return a copy of the image with every plate masked. The input is not changed.

    `box_style` is used for plates without corners (corner fitting failed); None = use `style`
    for them too (e.g. when corner fitting was switched off on purpose).
    """
    for name in (style, box_style):
        if name is not None and name not in MASKERS:
            raise ValueError(f"unknown mask style {name!r}; available: {', '.join(MASKERS)}")
    out = image_bgr.copy()
    for plate in plates:
        name = style if (plate.corners is not None or box_style is None) else box_style
        light_plane = getattr(plate.brightness, "plane", None)  # set by automatic brightness
        MASKERS[name](out, pad_quad(plate.quad, pad), light_plane=light_plane, **opts)
    return out
