"""Find number plates in car images and mask them. CPU only, ONNX Runtime, no torch.

    import cv2
    from plate_masker import PlateMasker

    masker = PlateMasker()                      # loads the model once
    image = cv2.imread("car.jpg")
    masked, plates = masker.process(image)      # plates: list[Plate] with .corners and .fit
    cv2.imwrite("car_masked.jpg", masked)

    # Cover each plate with an image instead of black, warped onto the plate's corners:
    masker = PlateMasker(overlay=cv2.imread("sample_plate.png", cv2.IMREAD_UNCHANGED))
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np

from .corners import find_plate_corners, fit_plate, grow_quad, order_corners, quad_size
from .detector import DEFAULT_CONF, DEFAULT_MODEL, DEFAULT_THREADS, Plate, PlateDetector
from .masking import (
    DEFAULT_COLOR_RGB,
    DEFAULT_LIGHTING,
    DEFAULT_PAD,
    MASKERS,
    image_overlay,
    mask_plates,
    pad_quad,
    solid_fill,
)

__all__ = [
    "DEFAULT_COLOR_RGB",
    "DEFAULT_CONF",
    "DEFAULT_LIGHTING",
    "DEFAULT_MODEL",
    "DEFAULT_PAD",
    "DEFAULT_THREADS",
    "MASKERS",
    "Plate",
    "PlateDetector",
    "PlateMasker",
    "find_plate_corners",
    "fit_plate",
    "grow_quad",
    "image_overlay",
    "mask_plates",
    "order_corners",
    "pad_quad",
    "quad_size",
    "solid_fill",
]


class PlateMasker:
    """Detector + corner fitting + mask style in one object.

    overlay:   image (BGR or BGRA array) to warp onto each plate; None = solid fill.
    lighting:  0-100 %, how far to match the overlay to the original plate's brightness
               (0 = overlay as it is, 100 = looks lit like the original plate).
    brightness_detector: e.g. plate_brightness_detector.BrightnessDetector(). When given, every
               plate's brightness is picked by it (any plate colour) and `lighting` is not used.
    color:     RGB fill colour for solid fill, and for plates whose corners can't be found.
    tight_fit: find each plate's own 4 corners; False = mask the detector's straight box.
    pad:       extra margin beyond the plate's edge, as a share of the plate's height.
    style:     a name from MASKERS, for custom styles; normally leave None.
    """

    def __init__(
        self,
        model: str | os.PathLike[str] = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        pad: float = DEFAULT_PAD,
        threads: int = DEFAULT_THREADS,
        *,
        tight_fit: bool = True,
        overlay: np.ndarray | None = None,
        lighting: float = DEFAULT_LIGHTING,
        brightness_detector=None,
        color: tuple[int, int, int] = DEFAULT_COLOR_RGB,
        style: str | None = None,
        **mask_opts,
    ):
        self.style = style or ("image" if overlay is not None else "solid")
        if self.style not in MASKERS:
            raise ValueError(f"unknown mask style {self.style!r}; available: {', '.join(MASKERS)}")
        if self.style == "image" and overlay is None:
            raise ValueError("mask style 'image' needs an overlay image")
        if not 0 <= lighting <= 100:
            raise ValueError(f"lighting must be 0-100 (%), got {lighting}")
        self.detector = PlateDetector(model, conf, threads)
        self.brightness_detector = brightness_detector
        self.pad = pad
        self.tight_fit = tight_fit
        self.mask_opts = {"color": color, "overlay": overlay, "lighting": lighting, **mask_opts}

    def process(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[Plate]]:
        """Return (masked copy of the image, detected plates). No plates -> an unchanged copy."""
        plates = self.detector.detect(image_bgr)
        if self.tight_fit:
            plates = [fit_plate(image_bgr, p) for p in plates]
        if self.brightness_detector is not None and plates:
            # Measured on the original image, before anything is painted.
            readings = self.brightness_detector.measure_all(
                image_bgr, [p.quad for p in plates], [p.corners is not None for p in plates]
            )
            plates = [dataclasses.replace(p, brightness=r) for p, r in zip(plates, readings)]
        # A failed corner fit gets a solid box; with fitting off, the chosen style goes on the box.
        box_style = "solid" if self.tight_fit else None
        masked = mask_plates(image_bgr, plates, self.style, self.pad, box_style, **self.mask_opts)
        return masked, plates
