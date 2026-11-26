"""Privacy pass on what leaves the laptop: people are blurred in every image sent to fal.ai.

DPDP (India, 2023): send the processor only the personal data the purpose needs. Bystanders (and a driver seen
through the glass) are not needed to cut a car out or measure a camera, so every person box YOLOX-s finds is
pixelated in the *upload copy*. The photo on disk, the cut-out, the outputs and every car pixel stay as they were;
only the bytes fal receives change. Number plates are covered earlier, by number_plate_detection (the plate gate in
change_background.py refuses to upload otherwise).

Detector: car_isolation's YOLOX-s (COCO class "person"), already on disk for car_isolation; it runs at 640 px, so a
person only a few pixels tall in a 4K frame can be missed. Boxes are cached per image (sha1 of its pixels), so the
five calls that send the same photo run the detector once.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import cv2
import numpy as np

MIN_SCORE = 0.3  # person detections at or above this score are blurred
PAD = 0.15  # each box grows by this share of its size on every side (hair, hands, a loose box edge)
BLOCKS = 12  # a blurred box keeps at most this many pixel blocks on its long side (no face is readable at that)


def blur_boxes(image: np.ndarray, boxes: list[tuple[float, float, float, float]], pad: float = PAD,
               blocks: int = BLOCKS) -> np.ndarray:
    """A copy of image with each (x0, y0, x1, y1) box pixelated (block average, then a soft blur on top).
    Returns image itself when boxes is empty."""
    if not boxes:
        return image
    out = image.copy()
    h, w = image.shape[:2]
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        xa, ya = max(0, int(x0 - pad * bw)), max(0, int(y0 - pad * bh))
        xb, yb = min(w, int(np.ceil(x1 + pad * bw))), min(h, int(np.ceil(y1 + pad * bh)))
        if xb - xa < 2 or yb - ya < 2:
            continue
        roi = out[ya:yb, xa:xb]
        s = blocks / max(roi.shape[:2])
        small = cv2.resize(roi, (max(1, round(roi.shape[1] * s)), max(1, round(roi.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        k = max(3, (min(roi.shape[:2]) // blocks) | 1)
        out[ya:yb, xa:xb] = cv2.GaussianBlur(big, (k, k), 0)
    return out


class PeopleRedactor:
    """Callable(image BGR) -> image with every detected person pixelated. Only 3-channel images are touched
    (masks pass through). `detect` maps image -> list of boxes; by default YOLOX-s, loaded on first use."""

    def __init__(self, threads: int = 4, min_score: float = MIN_SCORE,
                 detect: Callable[[np.ndarray], list] | None = None):
        self.threads = threads
        self.min_score = min_score
        self._detect = detect
        self._boxes: dict[bytes, list] = {}
        self.images = 0  # distinct images checked
        self.people = 0  # person boxes blurred over those images

    def _yolox(self) -> Callable[[np.ndarray], list]:
        from car_isolation.detect import VehicleDetector
        from car_isolation.models import yolox_path

        det = VehicleDetector(yolox_path(), self.threads)

        def people(image: np.ndarray) -> list:
            return [d.box for d in det.detect(image, self.min_score) if d.kind == "person"]

        return people

    def boxes(self, image: np.ndarray) -> list:
        key = hashlib.sha1(np.ascontiguousarray(image).view(np.uint8).data).digest()
        if key not in self._boxes:
            if self._detect is None:
                self._detect = self._yolox()
            found = list(self._detect(image))
            self._boxes[key] = found
            self.images += 1
            self.people += len(found)
        return self._boxes[key]

    def __call__(self, image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return image
        return blur_boxes(image, self.boxes(image))


__all__ = ["BLOCKS", "MIN_SCORE", "PAD", "PeopleRedactor", "blur_boxes"]
