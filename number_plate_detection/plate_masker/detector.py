"""Number plate detector: a thin wrapper over open-image-models (YOLOv9, ONNX, CPU)."""

from __future__ import annotations

import os
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 384 px input: the low-compute default. See docs/findings.md for the other sizes.
DEFAULT_MODEL = "yolo-v9-t-384-license-plate-end2end"
# The model's own default. Lower it to catch more plates (a missed plate is a privacy leak).
DEFAULT_CONF = 0.25
# CPU threads for ONNX Runtime. 4 was fastest and steadiest on an i5-1240P (4 P + 8 E cores);
# "all cores" (0) was ~4x slower whenever other apps were busy. See docs/findings.md.
DEFAULT_THREADS = 4


@dataclass(frozen=True)
class Plate:
    """One detected plate.

    x1..y2 is the detector's box in original-image pixels (x2/y2 exclusive). `corners` is the
    plate's own outline (top-left, top-right, bottom-right, bottom-left), filled in by
    corners.fit_plate(); None means it couldn't be found and the box is used instead.
    `outline_from` says how: "corners" (traced from the plate's edge) or "letters" (worked out
    from its row of characters, rougher). `brightness` is a plate_brightness_detector
    BrightnessReading when automatic brightness is on.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    corners: tuple[tuple[float, float], ...] | None = None
    brightness: Any = None
    outline_from: str = ""

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def fit(self) -> str:
        """'corners' (outline traced from the plate's edge), 'letters' (from its characters) or 'box'."""
        if self.corners is None:
            return "box"
        return self.outline_from or "corners"

    @property
    def quad(self) -> np.ndarray:
        """The shape to mask: the fitted corners, or the box's corners. float32, shape (4, 2)."""
        if self.corners is not None:
            return np.float32(self.corners)
        x2, y2 = self.x2 - 1, self.y2 - 1
        return np.float32([[self.x1, self.y1], [x2, self.y1], [x2, y2], [self.x1, y2]])


class PlateDetector:
    """Load the model once, then call detect() on as many images as you like.

    `model` is a registered open-image-models name (weights download once to
    ~/.cache/open-image-models) or a path to a local YOLOv9 plate .onnx file.
    `threads` caps ONNX Runtime's CPU threads; 0 lets it use every core.
    """

    def __init__(
        self,
        model: str | os.PathLike[str] = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        threads: int = DEFAULT_THREADS,
    ):
        import onnxruntime as ort
        from open_image_models import create_detector

        local = str(model).lower().endswith(".onnx")
        if local and not Path(model).exists():
            raise FileNotFoundError(f"model file not found: {model}")
        self.model = str(model)
        self.conf = conf
        sess_options = None
        if threads:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1
        try:
            self._detector = create_detector(
                Path(model) if local else str(model),
                conf_thresh=conf,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
                **({"backend": "yolo_v9", "class_labels": ["License Plate"]} if local else {}),
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(_download_help(str(model), exc)) from None

    def detect(self, image_bgr: np.ndarray) -> list[Plate]:
        """Find plates in a BGR image (as from cv2.imread). Highest confidence first."""
        h, w = image_bgr.shape[:2]
        plates = []
        for det in self._detector.predict(image_bgr):
            b = det.bounding_box
            x1, x2 = max(0, min(w, int(b.x1))), max(0, min(w, int(b.x2)))
            y1, y2 = max(0, min(h, int(b.y1))), max(0, min(h, int(b.y2)))
            if x2 > x1 and y2 > y1:
                plates.append(Plate(x1, y1, x2, y2, float(det.confidence)))
        return sorted(plates, key=lambda p: p.confidence, reverse=True)


def _download_help(model: str, exc: Exception) -> str:
    """Explain a failed first-time model download and how to get the file by hand."""
    try:
        from open_image_models.detection.core.hub import DETECTION_MODELS

        url = DETECTION_MODELS[model].url
    except Exception:
        url = None
    lines = [f"could not download model '{model}': {getattr(exc, 'reason', exc)}"]
    if url:
        name = url.rsplit("/", 1)[-1]
        lines += [
            "Download it by hand (e.g. in a browser) from:",
            f"  {url}",
            f"save it as models\\{name} in the number_plate_detection folder, and set in config.toml:",
            f'  model = "models/{name}"',
        ]
    return "\n".join(lines)
