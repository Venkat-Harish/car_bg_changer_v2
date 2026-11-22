"""Which pixels of the background-removal mask are *this* car.

BiRefNet (bg_changer.segment) marks "the salient object(s)": a car parked behind ours, a person or a
gate that touches its outline in the photo joins the same blob. This step asks an instance model:

1. YOLOX boxes every vehicle. The target is the box that best matches BiRefNet's main blob
   (share of the blob inside the box x how full the box is).
2. SAM masks the target (T) and every other vehicle near it (O_i), on a crop around the car.
3. `remove` = the other vehicles (grown a little), never the inside of T; plus, when remove_objects is
   on, any piece of the blob that sticks out of T by more than OBJECT_REACH (a person, a gate, a sign).
4. `glass_ok` = T grown a little: bg_changer.segment fills only holes lying on it (windows), so gaps
   such as mirror-to-pillar stay open.

Only the mask changes: the car's pixels are copied as before. If no vehicle is found, or SAM's mask
disagrees with BiRefNet's blob, nothing is changed and the photo is flagged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from bg_changer.segment import _bbox, _disk, fill_holes, select_main_car

from .detect import Detection

DEFAULT_MIN_SCORE = 0.35
EDGE = 0.003  # tolerance band around SAM's outline, share of the photo's longer side (~12 px at 4K)
OBJECT_REACH = 0.012  # a piece sticking out of T further than this is not the car (~46 px at 4K)
MIN_OBJECT = 0.002  # ... and only if it is at least this share of the blob's area
OBJECT_IN_BOX = 0.5  # ... and at least this share of it lies in another detected object's box
EDGE_PX = 3  # photo-border band (px) for "enters the photo from an edge"
CROP_PAD = 0.08  # SAM runs on the car's box plus this share of its size on every side
MIN_SHARE = 0.5  # the target box must hold at least this share of the blob
MIN_AGREE = 0.6  # at least this share of SAM's T must lie on BiRefNet's blob ...
MIN_COVER = 0.4  # ... and T must cover at least this share of the blob (which may hold a merged neighbour)
SAME_CAR_IOU = 0.7  # another box whose mask is this close to T is the same car (a double detection)
SAME_BOX_IOU = 0.5  # another object's box this close to the target's box is not evidence (likely the car itself)
PART_INSIDE = 0.8  # a box at least this much inside the target car's box may be a part of it (YOLOX calls a
                   # door mirror "car"): never removed, never evidence. A car behind ours sticks out of its box.
PART_SMALL = 0.03  # ... so may a box under this share of the target box's area that is at least
PART_TOUCH = 0.3   # this much inside it (a far-side mirror sticking out sideways in a 3/4 view)


@dataclass
class IsolationResult:
    remove: np.ndarray | None = None  # bool HxW: set to background before the mask is cleaned
    glass_ok: np.ndarray | None = None  # bool HxW: holes may be filled only here
    status: str = "ok"  # ok | no-vehicle | unsure | no-object
    vehicles: int = 0  # other vehicles removed
    objects: int = 0  # other pieces removed (remove_objects)
    seconds: float = 0.0
    target: Detection | None = None
    others: list[Detection] = field(default_factory=list)
    crop: tuple[int, int, int, int] | None = None
    target_mask: np.ndarray | None = None  # bool HxW (for the debug overlay)

    def summary(self) -> dict:
        return {"status": self.status, "vehicles": self.vehicles, "objects": self.objects,
                "seconds": round(self.seconds, 2), "text": self.text()}

    def text(self) -> str:
        """Short description for the console and results.csv."""
        if self.status != "ok":
            return self.status
        parts = []
        if self.vehicles:
            parts.append(f"{self.vehicles} vehicle(s)")
        if self.objects:
            parts.append(f"{self.objects} object(s)")
        return ("removed " + " + ".join(parts)) if parts else "nothing to remove"


def box_share(blob: np.ndarray, box, blob_area: int) -> tuple[float, float]:
    """(share of the blob inside the box, share of the box filled by the blob)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    inside = int(blob[max(0, y1):y2, max(0, x1):x2].sum())
    area = max(1, (x2 - x1) * (y2 - y1))
    return inside / max(1, blob_area), inside / area


def pick_target(blob: np.ndarray, dets: list[Detection]) -> tuple[int, float]:
    """Index of the detection that is the blob's car (-1 = none), and its blob share."""
    area = int(blob.sum())
    best, best_score, best_share = -1, 0.0, 0.0
    for i, d in enumerate(dets):
        share, fill = box_share(blob, d.box, area)
        if share * fill > best_score:
            best, best_score, best_share = i, share * fill, share
    if best < 0 or best_share < MIN_SHARE:
        return -1, best_share
    return best, best_share


def crop_box(blob_box, target_box, wh) -> tuple[int, int, int, int]:
    W, H = wh
    x1 = min(blob_box[0], target_box[0])
    y1 = min(blob_box[1], target_box[1])
    x2 = max(blob_box[2], target_box[2])
    y2 = max(blob_box[3], target_box[3])
    pad = CROP_PAD * max(x2 - x1, y2 - y1)
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(W, int(np.ceil(x2 + pad))), min(H, int(np.ceil(y2 + pad))))


def agrees(target: np.ndarray, blob: np.ndarray) -> bool:
    """SAM's target mask and BiRefNet's blob describe the same car (else the step changes nothing)."""
    both = int((target & blob).sum())
    return both >= MIN_AGREE * max(1, int(target.sum())) and both >= MIN_COVER * max(1, int(blob.sum()))


def box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside_share(box, outer) -> float:
    """Share of `box`'s area that lies inside `outer`."""
    iw = max(0.0, min(box[2], outer[2]) - max(box[0], outer[0]))
    ih = max(0.0, min(box[3], outer[3]) - max(box[1], outer[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return iw * ih / area if area > 0 else 0.0


def maybe_part(box, target_box) -> bool:
    """The box may be a part of the target car (a mirror YOLOX calls "car"): leave it alone."""
    inside = inside_share(box, target_box)
    if inside >= PART_INSIDE:
        return True
    area = (box[2] - box[0]) * (box[3] - box[1])
    t_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    return inside >= PART_TOUCH and t_area > 0 and area < PART_SMALL * t_area


def free_edges(frame_hw, crop, target_c: np.ndarray) -> np.ndarray:
    """Bool mask (crop coords): the photo's border band on the sides where the target car is not cut off."""
    H, W = frame_hw
    x1, y1, x2, y2 = crop
    out = np.zeros_like(target_c, dtype=bool)
    e = EDGE_PX
    sides = {  # side: (lies in the crop, the band in crop coords, the car touches it)
        "left": (x1 == 0, (slice(None), slice(0, e))),
        "right": (x2 == W, (slice(None), slice(x2 - x1 - e, None))),
        "top": (y1 == 0, (slice(0, e), slice(None))),
        "bottom": (y2 == H, (slice(y2 - y1 - e, None), slice(None))),
    }
    for inside, band in sides.values():
        if inside and not target_c[band].any():
            out[band] = True
    return out


def overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def plan_removal(blob: np.ndarray, target: np.ndarray, others: list[np.ndarray], longer: int,
                 remove_objects: bool, object_boxes: np.ndarray | None = None,
                 free_edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Pure mask logic (all bool arrays of one size): (remove, glass_ok, vehicles removed, objects removed).

    A piece of the blob sticking out of the target car is removed only with evidence that it is something
    else, never on shape alone (SAM sometimes leaves out a spoiler or an antenna):
    - at least OBJECT_IN_BOX of it lies in `object_boxes` (boxes the detector found for other objects), or
    - it reaches `free_edges` (the photo's border where the car itself is not cut off: a finger, a passer-by).
    """
    r_edge = max(1, round(EDGE * longer))
    r_far = max(r_edge + 2, round(OBJECT_REACH * longer))
    t8 = target.astype(np.uint8)
    t_core = cv2.erode(t8, _disk(r_edge)).astype(bool)
    t_grow = cv2.dilate(t8, _disk(r_edge)).astype(bool)
    t_far = cv2.dilate(t8, _disk(r_far)).astype(bool)

    remove = np.zeros_like(blob, dtype=bool)
    vehicles = 0
    for o in others:
        o_grow = cv2.dilate(o.astype(np.uint8), _disk(r_edge)).astype(bool) & ~t_core
        if (o_grow & blob).any():
            vehicles += 1
        remove |= o_grow

    objects = 0
    if remove_objects:
        outside = (blob & ~t_grow & ~remove).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
        if n > 1:
            far = np.bincount(labels[~t_far & outside.astype(bool)], minlength=n) > 0
            big = stats[:, cv2.CC_STAT_AREA] >= MIN_OBJECT * max(1, int(blob.sum()))
            evidence = np.zeros(n, bool)
            if object_boxes is not None:
                in_box = np.bincount(labels[object_boxes], minlength=n)
                evidence |= in_box >= OBJECT_IN_BOX * stats[:, cv2.CC_STAT_AREA]
            if free_edges is not None:
                evidence |= np.bincount(labels[free_edges], minlength=n) > 0
            drop = far & big & evidence
            drop[0] = False
            objects = int(drop.sum())
            if objects:
                remove |= drop[labels]
    return remove, t_grow, vehicles, objects


class CarIsolator:
    """Loads YOLOX-s and SAM ViT-B once. `isolate(image_bgr, raw_alpha)` -> IsolationResult."""

    def __init__(self, min_score: float = DEFAULT_MIN_SCORE, remove_objects: bool = True, threads: int = 4,
                 detector=None, segmenter=None):
        if detector is None or segmenter is None:
            from .detect import VehicleDetector
            from .instance import BoxSegmenter
            from .models import sam_paths, yolox_path

            detector = detector or VehicleDetector(yolox_path(), threads)
            segmenter = segmenter or BoxSegmenter(*sam_paths(), threads)
        self.detector = detector
        self.segmenter = segmenter
        self.min_score = min_score
        self.remove_objects = remove_objects

    def isolate(self, image_bgr: np.ndarray, raw_alpha: np.ndarray) -> IsolationResult:
        t0 = time.perf_counter()
        H, W = raw_alpha.shape
        longer = max(H, W)
        binary = (raw_alpha >= 0.5).astype(np.uint8)
        if not binary.any():
            return IsolationResult(status="no-object", seconds=time.perf_counter() - t0)
        blob = select_main_car(fill_holes(binary)).astype(bool)
        blob_box = _bbox(blob)

        found = self.detector.detect(image_bgr, self.min_score)
        dets = [d for d in found if d.vehicle]
        ti, _ = pick_target(blob, dets)
        if ti < 0:
            return IsolationResult(status="no-vehicle", seconds=time.perf_counter() - t0)
        target = dets[ti]
        cx1, cy1, cx2, cy2 = crop = crop_box(blob_box, target.box, (W, H))

        def local(box):
            return (box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1)

        self.segmenter.set_image(image_bgr[cy1:cy2, cx1:cx2])
        t_mask, _ = self.segmenter.mask(local(target.box))
        blob_c = blob[cy1:cy2, cx1:cx2]
        if not agrees(t_mask, blob_c):
            return IsolationResult(status="unsure", target=target, crop=crop, seconds=time.perf_counter() - t0)

        other_dets, other_masks = [], []
        for i, d in enumerate(dets):
            if i == ti or not overlaps(d.box, crop) or maybe_part(d.box, target.box):
                continue
            m, _ = self.segmenter.mask(local(d.box))
            inter = int((m & t_mask).sum())
            union = int((m | t_mask).sum())
            if union and inter / union >= SAME_CAR_IOU:
                continue  # the same car, detected twice
            other_dets.append(d)
            other_masks.append(m)

        # Evidence that a piece sticking out of the car is something else (see plan_removal).
        boxes_c = np.zeros_like(t_mask)
        for d in found:
            if (d is target or box_iou(d.box, target.box) > SAME_BOX_IOU or not overlaps(d.box, crop)
                    or maybe_part(d.box, target.box)):
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in local(d.box))
            boxes_c[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        edges_c = free_edges((H, W), crop, t_mask)
        rem_c, glass_c, vehicles, objects = plan_removal(blob_c, t_mask, other_masks, longer, self.remove_objects,
                                                         boxes_c, edges_c)
        remove = np.zeros((H, W), bool)
        glass = np.zeros((H, W), bool)
        target_full = np.zeros((H, W), bool)
        remove[cy1:cy2, cx1:cx2] = rem_c
        glass[cy1:cy2, cx1:cx2] = glass_c
        target_full[cy1:cy2, cx1:cx2] = t_mask
        return IsolationResult(remove=remove, glass_ok=glass, vehicles=vehicles, objects=objects,
                               seconds=time.perf_counter() - t0, target=target, others=other_dets, crop=crop,
                               target_mask=target_full)


def overlay(image_bgr: np.ndarray, result: IsolationResult) -> np.ndarray:
    """Debug picture: target car outline (green), removed areas (red), other vehicle boxes (orange)."""
    out = image_bgr.copy()
    if result.remove is not None:
        red = out.copy()
        red[result.remove] = (0, 0, 255)
        out = cv2.addWeighted(out, 0.55, red, 0.45, 0)
    th = max(2, round(0.002 * max(out.shape[:2])))
    if result.target_mask is not None:
        contours, _ = cv2.findContours(result.target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 220, 0), th)
    for d in result.others:
        cv2.rectangle(out, tuple(int(v) for v in d.box[:2]), tuple(int(v) for v in d.box[2:]), (0, 140, 255), th)
    cv2.putText(out, result.text(), (10, 10 + 12 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * th, (255, 255, 255), th)
    return out
