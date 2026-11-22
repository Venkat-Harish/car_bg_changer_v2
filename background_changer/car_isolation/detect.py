"""Vehicle boxes with YOLOX-s (ONNX Runtime, CPU).

The official YOLOX ONNX export takes a 640x640 BGR image, letterboxed at the top-left with grey 114,
not normalised, and returns raw grid predictions (cx, cy, w, h, objectness, 80 COCO class scores) that
are decoded here, as in YOLOX's own demo/ONNXRuntime/onnx_inference.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

INPUT = 640
STRIDES = (8, 16, 32)
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign "
        "parking-meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports-ball kite baseball-bat baseball-glove skateboard surfboard "
        "tennis-racket bottle wine-glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot-dog pizza donut cake chair couch potted-plant bed dining-table toilet tv laptop mouse remote keyboard "
        "cell-phone microwave oven toaster sink refrigerator book clock vase scissors teddy-bear hair-drier "
        "toothbrush").split()
NMS_IOU = 0.5


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in photo pixels
    score: float
    kind: str

    @property
    def vehicle(self) -> bool:
        return self.kind in VEHICLES.values()


def letterbox(image_bgr: np.ndarray, size: int = INPUT) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded, r


def decode(raw: np.ndarray, size: int = INPUT) -> np.ndarray:
    """(N, 85) raw grid outputs -> the same with cx, cy, w, h in input pixels."""
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        union = area(boxes[i]) + area(boxes[rest]) - inter
        order = rest[inter / np.maximum(union, 1e-6) <= iou]
    return keep


def vehicles_from_raw(raw: np.ndarray, r: float, frame_wh: tuple[int, int], min_score: float,
                      vehicles_only: bool = True) -> list[Detection]:
    """Decoded, NMS-ed detections in photo pixels: vehicles only, or every COCO class."""
    pred = decode(raw)
    cls = np.argmax(pred[:, 5:], axis=1)
    score = pred[:, 4] * pred[np.arange(len(pred)), 5 + cls]
    ok = score >= min_score
    if vehicles_only:
        ok &= np.isin(cls, list(VEHICLES))
    if not ok.any():
        return []
    pred, cls, score = pred[ok], cls[ok], score[ok]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / r
    W, H = frame_wh
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    # Vehicles: one NMS over all vehicle classes (a car is sometimes also scored as a truck); others per class.
    group = np.where(np.isin(cls, list(VEHICLES)), -1, cls)
    keep = []
    for g in np.unique(group):
        idx = np.nonzero(group == g)[0]
        keep += [int(idx[i]) for i in nms(boxes[idx], score[idx])]
    keep.sort(key=lambda i: -score[i])
    return [Detection(tuple(float(v) for v in boxes[i]), float(score[i]), COCO[int(cls[i])]) for i in keep]


class VehicleDetector:
    def __init__(self, path, threads: int = 4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def detect(self, image_bgr: np.ndarray, min_score: float) -> list[Detection]:
        """Every COCO object (vehicles first by score); `Detection.vehicle` tells them apart."""
        padded, r = letterbox(image_bgr)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        raw = self.session.run(None, {self.input: blob})[0][0]
        return vehicles_from_raw(raw, r, (image_bgr.shape[1], image_bgr.shape[0]), min_score, vehicles_only=False)
