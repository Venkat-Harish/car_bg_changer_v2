"""car_isolation: model-free tests with a fake detector and a fake SAM."""

import cv2
import numpy as np
import pytest

import change_background as cb
from bg_changer import clean_alpha
from bg_changer.segment import fill_holes
from car_isolation import CarIsolator, Detection, plan_removal
from car_isolation.detect import decode, letterbox, nms, vehicles_from_raw

H, W = 600, 1000
CAR = (200, 250, 700, 500)  # x1, y1, x2, y2 of the target car's body
NEIGHBOUR = (560, 120, 900, 300)  # a car parked behind, overlapping the target's top-right in the photo


def rect(box, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def soft(mask):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, image, min_score):
        return [d for d in self.dets if d.score >= min_score]


class FakeSam:
    """Returns pre-made masks (in full-image coords) for boxes, cropped like the real one."""

    def __init__(self, by_box):
        self.by_box = by_box  # {box: bool mask HxW}
        self.offset = (0, 0)
        self.shape = (H, W)

    def set_image(self, crop):
        self.shape = crop.shape[:2]

    def mask(self, box):
        # find the full-image box that this crop-local box came from
        for full, m in self.by_box.items():
            dx, dy = full[0] - box[0], full[1] - box[1]
            if abs((full[2] - box[2]) - dx) < 1e-6 and abs((full[3] - box[3]) - dy) < 1e-6:
                h, w = self.shape
                return m[int(dy):int(dy) + h, int(dx):int(dx) + w].copy(), 0.95
        raise AssertionError(f"unexpected box {box}")


def scene_masks():
    target = rect(CAR)
    neighbour = rect(NEIGHBOUR) & ~target  # partly hidden behind our car
    return target, neighbour


def isolator(dets, masks, remove_objects=True):
    return CarIsolator(0.3, remove_objects, detector=FakeDetector(dets), segmenter=FakeSam(masks))


# ---------------------------------------------------------------- clean_alpha hooks

def test_clean_alpha_without_isolation_is_unchanged():
    raw = soft(rect(CAR) | rect((100, 100, 140, 600)))
    a = clean_alpha(raw, edge_shrink=1)
    b = clean_alpha(raw, 1, None, None)
    assert np.array_equal(a.alpha, b.alpha) and a.box == b.box and a.cut == b.cut


def test_fill_holes_keeps_non_glass_gaps_open():
    outline = rect((100, 100, 500, 400))
    window = (150, 130, 250, 200)
    mirror_gap = (420, 150, 470, 190)
    mask = (outline & ~rect(window) & ~rect(mirror_gap)).astype(np.uint8)
    car = rect((100, 100, 500, 400)) & ~rect(mirror_gap)  # SAM: the gap is not car
    assert fill_holes(mask).astype(bool)[mirror_gap[1] + 5, mirror_gap[0] + 5]  # today: filled
    out = fill_holes(mask, car).astype(bool)
    assert out[window[1] + 5, window[0] + 5]  # window: filled (original glass kept)
    assert not out[mirror_gap[1] + 5, mirror_gap[0] + 5]  # gap: open


def test_remove_mask_drops_a_touching_neighbour():
    target, neighbour = scene_masks()
    raw = soft(target | neighbour)
    before = clean_alpha(raw, 1)
    assert before.alpha[150, 800] > 0.9  # today the neighbour is part of the cut-out
    remove, glass, n_veh, n_obj = plan_removal(target | neighbour, target, [neighbour], max(H, W), True)
    after = clean_alpha(raw, 1, remove, glass)
    assert n_veh == 1
    assert after.alpha[150, 800] == 0.0 and after.alpha[200, 600] == 0.0
    inner = cv2.erode(target.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert np.all(after.alpha[inner] == 1.0)  # the car itself is untouched
    assert after.box[1] >= CAR[1] - 3


def test_plan_removal_drops_objects_only_with_evidence():
    target = rect(CAR)
    person = rect((300, 120, 330, 252))  # stands behind the roof, sticks out ~130 px
    bump = rect((400, 247, 430, 250))  # 3 px of edge disagreement: stays
    blob = target | person | bump
    person_box = rect((290, 100, 340, 260))
    remove, _, n_veh, n_obj = plan_removal(blob, target, [], max(H, W), True, person_box)
    assert n_veh == 0 and n_obj == 1
    assert remove[150, 315] and not remove[248, 415]
    remove2, _, _, n_obj2 = plan_removal(blob, target, [], max(H, W), False, person_box)
    assert n_obj2 == 0 and not remove2.any()


def test_plan_removal_keeps_a_car_part_sam_missed():
    """A spoiler sticking out of SAM's car, with no other object's box on it: kept (never cut on shape alone)."""
    target = rect(CAR)
    spoiler = rect((600, 200, 690, 252))
    remove, _, _, n_obj = plan_removal(target | spoiler, target, [], max(H, W), True, np.zeros((H, W), bool),
                                       np.zeros((H, W), bool))
    assert n_obj == 0 and not remove.any()


def test_plan_removal_drops_a_finger_entering_from_a_free_edge():
    target = rect(CAR)
    finger = rect((100, 480, 260, H))  # enters from the bottom edge; the car is not cut off there
    edges = np.zeros((H, W), bool)
    edges[-3:, :] = True
    remove, _, _, n_obj = plan_removal(target | finger, target, [], max(H, W), True, None, edges)
    assert n_obj == 1 and remove[590, 150] and not remove[400, 450]


# ---------------------------------------------------------------- the isolator, end to end (fake models)

def test_isolator_picks_the_blob_car_and_removes_the_other():
    target, neighbour = scene_masks()
    dets = [Detection(tuple(map(float, NEIGHBOUR)), 0.8, "car"), Detection(tuple(map(float, CAR)), 0.9, "car")]
    masks = {tuple(map(float, CAR)): target, tuple(map(float, NEIGHBOUR)): neighbour | rect(NEIGHBOUR) & target}
    raw = soft(target | neighbour)
    image = np.zeros((H, W, 3), np.uint8)
    res = isolator(dets, masks).isolate(image, raw)
    assert res.status == "ok" and res.vehicles == 1
    assert res.target.box == tuple(map(float, CAR))
    car = clean_alpha(raw, 1, res.remove, res.glass_ok)
    assert car.alpha[150, 800] == 0.0
    assert car.alpha[400, 450] == 1.0


def test_a_mirror_detected_as_a_car_is_kept():
    """YOLOX labels a door mirror "car" (EQB front views). Its box lies inside the target car's box, so it is
    neither removed as another vehicle nor used as evidence against a piece sticking out of SAM's car."""
    body = rect((250, 250, 650, 500))  # SAM's car, without the mirror
    mirror = rect((210, 270, 252, 300))  # sticks out of SAM's car by ~40 px
    car_box = (205.0, 245.0, 655.0, 505.0)  # YOLOX's car box includes the mirror
    mirror_box = (208.0, 266.0, 254.0, 304.0)
    dets = [Detection(car_box, 0.9, "car"), Detection(mirror_box, 0.5, "car")]
    raw = soft(body | mirror)
    res = isolator(dets, {car_box: body, mirror_box: mirror}).isolate(np.zeros((H, W, 3), np.uint8), raw)
    assert res.vehicles == 0 and res.objects == 0
    assert not res.remove[285, 225]
    assert clean_alpha(raw, 1, res.remove, res.glass_ok).alpha[285, 230] == 1.0


def test_maybe_part_rules():
    from car_isolation.isolate import maybe_part

    car = (290.0, 160.0, 1070.0, 760.0)
    assert maybe_part((1032.0, 258.0, 1110.0, 328.0), car)  # far-side mirror, half outside, ~1% of the car box
    assert not maybe_part((1110.0, 250.0, 1435.0, 362.0), car)  # a car parked beside: outside the box
    baleno = (105.0, 333.0, 382.0, 552.0)
    assert not maybe_part((155.0, 314.0, 225.0, 361.0), baleno)  # sample_vid 0000: car behind the roof (5.4%)
    assert not maybe_part((291.0, 311.0, 370.0, 358.0), baleno)  # ... and the other one (6.1%)


def test_isolator_without_a_vehicle_changes_nothing():
    target, _ = scene_masks()
    raw = soft(target)
    res = isolator([], {}).isolate(np.zeros((H, W, 3), np.uint8), raw)
    assert res.status == "no-vehicle" and res.remove is None and res.glass_ok is None


def test_isolator_unsure_when_sam_disagrees():
    target, _ = scene_masks()
    raw = soft(target)
    box = tuple(map(float, CAR))
    wrong = rect((0, 0, 150, 150))  # SAM found something else
    res = isolator([Detection(box, 0.9, "car")], {box: wrong}).isolate(np.zeros((H, W, 3), np.uint8), raw)
    assert res.status == "unsure" and res.remove is None


def test_cut_out_with_isolator_records_summary():
    from bg_changer.segment import CarSegmenter

    target, neighbour = scene_masks()
    raw = soft(target | neighbour)
    seg = CarSegmenter.__new__(CarSegmenter)  # no model: raw alpha is injected
    seg.edge_shrink, seg.last_isolation = 1, None
    seg.raw_alpha = lambda image: raw
    dets = [Detection(tuple(map(float, CAR)), 0.9, "car"), Detection(tuple(map(float, NEIGHBOUR)), 0.8, "car")]
    masks = {tuple(map(float, CAR)): target, tuple(map(float, NEIGHBOUR)): neighbour}
    image = np.zeros((H, W, 3), np.uint8)
    car = seg.cut_out(image, isolator(dets, masks))
    assert car.isolation["status"] == "ok" and car.isolation["vehicles"] == 1
    assert "removed 1 vehicle" in car.isolation["text"]
    plain = seg.cut_out(image)
    assert plain.isolation is None
    assert np.array_equal(plain.alpha, clean_alpha(raw, 1).alpha)


def test_two_pass_cached_mask_gives_the_same_cut_out():
    """The first pass caches rembg's 8-bit mask; finishing from it equals cutting out in one go."""
    from bg_changer.segment import CarSegmenter

    target, neighbour = scene_masks()
    q = np.round(soft(target | neighbour) * 255).astype(np.uint8)
    seg = CarSegmenter.__new__(CarSegmenter)
    seg.edge_shrink, seg.last_isolation = 1, None
    seg.raw_alpha = lambda image: np.asarray(q, dtype=np.float32) / 255.0  # as rembg's mask is read
    image = np.zeros((H, W, 3), np.uint8)
    assert np.array_equal(seg.finish(image, q).alpha, seg.cut_out(image).alpha)


# ---------------------------------------------------------------- YOLOX post-processing

def test_letterbox_and_decode_round_trip():
    img = np.zeros((832, 464, 3), np.uint8)
    padded, r = letterbox(img)
    assert padded.shape == (640, 640, 3) and r == pytest.approx(640 / 832)
    n = sum((640 // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), np.float32)
    # one confident "car" at stride-32 cell (5, 7): cx = (5 + 0.5) * 32, cy = (7 + 0.5) * 32, w = h = 64
    idx = 80 * 80 + 40 * 40 + 7 * 20 + 5
    raw[idx, :4] = [0.5, 0.5, np.log(2), np.log(2)]
    raw[idx, 4] = 0.9
    raw[idx, 5 + 2] = 0.9
    out = decode(raw)
    assert out[idx, :4] == pytest.approx([176, 240, 64, 64])
    dets = vehicles_from_raw(raw, r, (464, 832), 0.3)
    assert len(dets) == 1 and dets[0].kind == "car" and dets[0].vehicle
    assert dets[0].box == pytest.approx(((176 - 32) / r, (240 - 32) / r, (176 + 32) / r, (240 + 32) / r))


def test_nms_and_non_vehicle_filter():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]
    n = sum((640 // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), np.float32)
    raw[0, 4], raw[0, 5 + 0] = 0.9, 0.9  # a person: not a vehicle
    assert vehicles_from_raw(raw, 1.0, (640, 640), 0.3) == []
    everything = vehicles_from_raw(raw, 1.0, (640, 640), 0.3, vehicles_only=False)
    assert [d.kind for d in everything] == ["person"] and not everything[0].vehicle


# ---------------------------------------------------------------- config

def test_config_isolation_block(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[car_isolation]\nenabled = true\nmin_score = 0.5\n", encoding="utf-8")
    assert cb.load_config(p)["car_isolation"] == {"enabled": True, "min_score": 0.5}
    for bad in ('enabled = "yes"', "min_score = 2", "speed = 1"):
        p.write_text(f"[car_isolation]\n{bad}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            cb.load_config(p)


def test_default_config_has_isolation_off():
    assert cb.load_config(cb.DEFAULT_CONFIG)["car_isolation"].get("enabled") is False
