"""One mask per vehicle box with SAM ViT-B (ONNX Runtime, CPU).

Uses the encoder/decoder pair rembg publishes, with rembg's own pre-processing (an RGB float image
scaled to fit 1024x684 at the top-left, not normalised; box prompts as point labels 2 and 3 plus a
padding point). Unlike rembg's `predict`, the heavy image encoder runs once per image and the light
decoder once per box.
"""

from __future__ import annotations

import cv2
import numpy as np

IN_W, IN_H = 1024, 684  # rembg's SAM export input (width, height)


class BoxSegmenter:
    def __init__(self, encoder_path, decoder_path, threads: int = 4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        cpu = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(encoder_path), sess_options=opts, providers=cpu)
        self.decoder = ort.InferenceSession(str(decoder_path), sess_options=opts, providers=cpu)
        self._embedding = None
        self._scale = 1.0
        self._size = (0, 0)

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Encode once; `mask(box)` then works on this image."""
        h, w = image_bgr.shape[:2]
        s = min(IN_W / w, IN_H / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        canvas = np.zeros((IN_H, IN_W, 3), np.float32)
        canvas[:nh, :nw] = cv2.cvtColor(cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB)
        name = self.encoder.get_inputs()[0].name
        self._embedding = self.encoder.run(None, {name: canvas})[0]
        self._scale, self._size = s, (w, h)

    def mask(self, box) -> tuple[np.ndarray, float]:
        """(bool HxW mask of the object in `box` = x1, y1, x2, y2 in image px, SAM's own quality score)."""
        if self._embedding is None:
            raise RuntimeError("set_image first")
        s = self._scale
        x1, y1, x2, y2 = box
        coords = np.array([[[x1 * s, y1 * s], [x2 * s, y2 * s], [0.0, 0.0]]], np.float32)
        labels = np.array([[2, 3, -1]], np.float32)
        masks, iou, _ = self.decoder.run(None, {
            "image_embeddings": self._embedding,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([IN_H, IN_W], np.float32),
        })
        k = int(np.argmax(iou[0]))
        logits = masks[0, k].astype(np.float32)
        w, h = self._size
        back = np.array([[1 / s, 0, 0], [0, 1 / s, 0]], np.float64)  # input space -> image px
        full = cv2.warpAffine(logits, back, (w, h), flags=cv2.INTER_LINEAR, borderValue=-20.0)
        return full > 0.0, float(iou[0, k])
