"""Hosted cut-out: BiRefNet on fal for the edge, SAM 3 (text prompt "car") to keep only this car.

Contract:
- birefnet_alpha(client, image_bgr, max_side=2048) -> float32 HxW in 0..1 at the photo's size (INTER_LINEAR up).
- sam3_instances(client, image_bgr, prompt, max_side=2048) -> list[Instance] (bool HxW masks at the photo's size).
- HostedIsolator(client).isolate(image_bgr, raw) -> car_isolation.IsolationResult, same contract as
  car_isolation/isolate.py CarIsolator.isolate, so bg_changer.segment.CarSegmenter.finish(image, raw, isolator)
  works unchanged. Raises nothing for "no car found": returns status "no-vehicle" / "unsure" / "no-object" with
  remove=None, glass_ok=None. fal failures (hosted.fal.FalError, BudgetExceeded) and a malformed response
  (FalError) propagate: the caller falls back to the local path.

Only masks are produced here; the photo's pixels are never read back from fal (returned colours are ignored).

fal schema facts relied on (read 2026-11-23):

fal-ai/sam-3/image  (https://fal.ai/models/fal-ai/sam-3/image/llms.txt, .../api)
  in : image_url (str, required), prompt (str, text prompt, default "wheel"), point_prompts, box_prompts,
       apply_mask (bool, default true: "apply the mask on the image"), sync_mode, output_format png|jpeg|webp,
       return_multiple_masks (bool, default false), max_masks (int 1..32, default 3), include_scores (bool),
       include_boxes (bool).
  out: masks: list[Image{url, content_type, file_name, file_size, width, height}] (required), one PNG per
       instance; image (preview, ignored); scores: list[float] (with include_scores); boxes: list[[cx, cy, w, h]]
       normalised 0..1 (with include_boxes); metadata: list[MaskMetadata{index, score, box (normalised cxcywh)}].
  The docs do not say how a mask PNG is encoded, so `mask_from_image` accepts grey, BGR or BGRA (alpha used unless
  fully opaque) and thresholds at half range. Masks come at the size that was sent (<= max_side); they are resized to
  the photo's size. Boxes are recomputed from the full-size mask (returned boxes ignored); empty masks dropped.

fal-ai/birefnet/v2  (https://fal.ai/models/fal-ai/birefnet/v2/llms.txt)
  in : image_url (required), model: "General Use (Light)" (default, BiRefNet) | "General Use (Light 2K)"
       (BiRefNet_lite-2K) | "General Use (Heavy)" (BiRefNet_lite) | "Matting" | "Portrait" |
       "General Use (Dynamic)"; operating_resolution "1024x1024" (default) | "2048x2048" | "2304x2304" (Dynamic
       only); output_mask (bool); refine_foreground (bool, default true); mask_only (bool: `image` is the
       segmentation mask, refinement skipped); output_format png|webp|gif; sync_mode.
  out: image: ImageFile{url, content_type, file_name, width, height} (the mask when mask_only), mask_image
       (optional, with output_mask).
  We request mask_only=true and read the mask's grey level (or alpha, if that is what varies); never its colours.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from bg_changer.segment import _bbox, _disk, fill_holes, select_main_car
from car_isolation.detect import Detection
from car_isolation.isolate import (EDGE, MIN_SHARE, SAME_CAR_IOU, IsolationResult, agrees, crop_box, free_edges,
                                   maybe_part, overlaps, pick_target, plan_removal)

from .fal import FalClient, FalError

SAM3_ENDPOINT = "fal-ai/sam-3/image"
BIREFNET_ENDPOINT = "fal-ai/birefnet/v2"
BIREFNET_MODEL = "General Use (Light 2K)"
BIREFNET_RESOLUTION = "2048x2048"
SAM3_MAX_MASKS = 16  # instances asked for per prompt (fal allows 1..32)


@dataclass
class Instance:
    mask: np.ndarray  # bool HxW at the photo's size
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 px at the photo's size
    label: str = ""


# ------------------------------------------------------------------ parsing

def _level(img: np.ndarray) -> np.ndarray:
    """float32 HxW 0..1 from a returned mask image: grey as is; BGRA -> its alpha unless that is fully opaque;
    otherwise (opaque or 3-channel) the channel max of a grey mask."""
    if img.ndim == 3:
        top = 65535 if img.dtype == np.uint16 else 255
        if img.shape[2] == 4 and int(img[..., 3].min()) < top:
            img = img[..., 3]
        else:
            img = img[..., :3].max(axis=2)
    full = 65535.0 if img.dtype == np.uint16 else (1.0 if img.dtype.kind == "f" else 255.0)
    return np.clip(img.astype(np.float32) / full, 0.0, 1.0)


def _to_size(level: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if level.shape != (h, w):
        level = cv2.resize(level, (w, h), interpolation=cv2.INTER_LINEAR)
    return level


def mask_from_image(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """bool HxW at size hw from a returned mask image of any size/encoding."""
    return _to_size(_level(img), hw) >= 0.5


def _url(entry) -> str | None:
    if isinstance(entry, dict):
        return entry.get("url")
    return entry if isinstance(entry, str) else None


def parse_sam3(client, result: dict, hw: tuple[int, int], label: str = "") -> list[Instance]:
    """Instances from a sam-3/image response (schema in the module docstring). Empty masks are dropped."""
    if not isinstance(result, dict) or not isinstance(result.get("masks"), list):
        raise FalError(f"{SAM3_ENDPOINT}: response has no 'masks' list")
    scores = result.get("scores") or []
    meta = {}
    for j, m in enumerate(result.get("metadata") or []):
        if isinstance(m, dict) and m:
            meta[int(m.get("index", j))] = m
    out = []
    for i, entry in enumerate(result["masks"]):
        url = _url(entry)
        if not url:
            raise FalError(f"{SAM3_ENDPOINT}: mask {i} has no url")
        mask = mask_from_image(client.fetch_image(url), hw)
        m = meta.get(i, {})
        score = m.get("score", scores[i] if i < len(scores) else None)
        score = 1.0 if score is None else float(score)
        if not mask.any():
            continue  # no pixels: nothing to keep or remove
        out.append(Instance(mask, score, tuple(float(v) for v in _bbox(mask)), label))
    return out


# ------------------------------------------------------------------ endpoints

def birefnet_alpha(client: FalClient, image: np.ndarray, max_side: int = 2048, model: str = BIREFNET_MODEL,
                   resolution: str = BIREFNET_RESOLUTION) -> np.ndarray:
    """BiRefNet's soft mask for the photo: float32 HxW 0..1 at the photo's size (INTER_LINEAR from the sent size)."""
    payload = {"model": model, "operating_resolution": resolution, "mask_only": True, "output_format": "png"}
    out = client.run(BIREFNET_ENDPOINT, payload, images={"image_url": image}, max_side=max_side)
    url = _url(out.get("image")) if isinstance(out, dict) else None
    if not url and isinstance(out, dict):
        url = _url(out.get("mask_image"))
    if not url:
        raise FalError(f"{BIREFNET_ENDPOINT}: response has no image url")
    return _to_size(_level(client.fetch_image(url)), image.shape[:2]).astype(np.float32)


def sam3_instances(client: FalClient, image: np.ndarray, prompt: str, max_side: int = 2048) -> list[Instance]:
    """Every instance SAM 3 finds for the text prompt, masks at the photo's size, in the API's order."""
    payload = {"prompt": prompt, "apply_mask": False, "return_multiple_masks": True, "max_masks": SAM3_MAX_MASKS,
               "include_scores": True, "include_boxes": True, "output_format": "png"}
    out = client.run(SAM3_ENDPOINT, payload, images={"image_url": image}, max_side=max_side)
    return parse_sam3(client, out, image.shape[:2], prompt)


# ------------------------------------------------------------------ isolation

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def pick_instance(blob: np.ndarray, cars: list[Instance]) -> int:
    """Index of the car instance that is BiRefNet's blob (agrees(), best IoU with the blob; ties -> first), or -1."""
    best, best_iou = -1, 0.0
    for i, c in enumerate(cars):
        if agrees(c.mask, blob):
            iou = _iou(c.mask, blob)
            if iou > best_iou:
                best, best_iou = i, iou
    return best


class HostedIsolator:
    """SAM 3 instead of YOLOX + SAM ViT-B: the target is the "car" instance that matches BiRefNet's main blob;
    other cars and `other_prompts` instances (people) touching it are removed. Same result as CarIsolator."""

    def __init__(self, client: FalClient, prompt: str = "car", other_prompts: tuple[str, ...] = ("person",),
                 remove_objects: bool = True, max_side: int = 2048):
        self.client = client
        self.prompt = prompt
        self.other_prompts = other_prompts
        self.remove_objects = remove_objects
        self.max_side = max_side

    def isolate(self, image: np.ndarray, raw: np.ndarray) -> IsolationResult:
        t0 = time.perf_counter()
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / 255.0
        H, W = raw.shape
        longer = max(H, W)
        binary = (raw >= 0.5).astype(np.uint8)
        if not binary.any():
            return IsolationResult(status="no-object", seconds=time.perf_counter() - t0)
        blob = select_main_car(fill_holes(binary)).astype(bool)

        cars = sam3_instances(self.client, image, self.prompt, self.max_side)
        ti = pick_instance(blob, cars)
        if ti < 0:
            # A car box holds the blob but no mask matches it -> unsure; no car at all -> no-vehicle.
            dets = [Detection(c.box, c.score, "car") for c in cars]
            pi, share = pick_target(blob, dets)
            status = "unsure" if pi >= 0 and share >= MIN_SHARE else "no-vehicle"
            return IsolationResult(status=status, target=dets[pi] if pi >= 0 else None,
                                   seconds=time.perf_counter() - t0)
        t_inst = cars[ti]
        target = Detection(t_inst.box, t_inst.score, "car")
        cx1, cy1, cx2, cy2 = crop = crop_box(_bbox(blob), t_inst.box, (W, H))
        win = (slice(cy1, cy2), slice(cx1, cx2))
        t_c = t_inst.mask[win]
        blob_c = blob[win]

        other_dets, other_masks = [], []
        for i, c in enumerate(cars):
            if i == ti or not overlaps(c.box, crop) or maybe_part(c.box, t_inst.box):
                continue
            if _iou(c.mask, t_inst.mask) >= SAME_CAR_IOU:
                continue  # the same car twice
            other_dets.append(Detection(c.box, c.score, "car"))
            other_masks.append(c.mask[win])

        # People etc.: removed (grown like plan_removal's vehicles, never T's eroded core).
        r_edge = max(1, round(EDGE * longer))
        t_core = cv2.erode(t_c.astype(np.uint8), _disk(r_edge)).astype(bool)
        obj_remove = np.zeros_like(t_c)
        obj_boxes = np.zeros_like(t_c)
        objects = 0
        for p in self.other_prompts:
            for inst in sam3_instances(self.client, image, p, self.max_side):
                if not overlaps(inst.box, crop):
                    continue
                grown = cv2.dilate(inst.mask[win].astype(np.uint8), _disk(r_edge)).astype(bool) & ~t_core
                if not grown.any():
                    continue  # wholly inside the car (a driver behind the windscreen): leave it
                if (grown & blob_c).any():
                    objects += 1
                obj_remove |= grown
                x1, y1, x2, y2 = (int(round(v)) for v in inst.box)
                obj_boxes[max(0, y1 - cy1):max(0, y2 - cy1), max(0, x1 - cx1):max(0, x2 - cx1)] = True
                other_dets.append(Detection(inst.box, inst.score, p))

        edges_c = free_edges((H, W), crop, t_c)
        rem_c, glass_c, vehicles, extra = plan_removal(blob_c & ~obj_remove, t_c, other_masks, longer,
                                                       self.remove_objects, obj_boxes, edges_c)
        rem_c |= obj_remove
        remove = np.zeros((H, W), bool)
        glass = np.zeros((H, W), bool)
        target_full = np.zeros((H, W), bool)
        remove[win] = rem_c
        glass[win] = glass_c
        target_full[win] = t_c
        return IsolationResult(remove=remove, glass_ok=glass, vehicles=vehicles, objects=objects + extra,
                               seconds=time.perf_counter() - t0, target=target, others=other_dets, crop=crop,
                               target_mask=target_full)


__all__ = ["BIREFNET_ENDPOINT", "BIREFNET_MODEL", "BIREFNET_RESOLUTION", "SAM3_ENDPOINT", "HostedIsolator",
           "Instance", "birefnet_alpha", "mask_from_image", "parse_sam3", "pick_instance", "sam3_instances"]
