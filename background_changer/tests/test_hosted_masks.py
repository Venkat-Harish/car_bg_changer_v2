"""hosted.masks: SAM 3 / BiRefNet response parsing and HostedIsolator, with FakeFal (no network)."""

import cv2
import numpy as np
import pytest

from bg_changer.segment import CarSegmenter, clean_alpha
from hosted.fal import FalError
from hosted.masks import (BIREFNET_ENDPOINT, SAM3_ENDPOINT, HostedIsolator, birefnet_alpha, mask_from_image,
                          sam3_instances)
from hosted_fakes import FakeFal

H, W = 600, 1000
CAR = (200, 250, 700, 500)  # target car body
NEIGHBOUR = (560, 120, 900, 300)  # a car parked behind, overlapping the target's top-right
PERSON = (700, 200, 760, 520)  # someone standing against the car's right end


def rect(box, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def soft(mask):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)


def noise(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def half(mask, bgra=False):
    """A mask as fal would return it for an image sent at half size (grey PNG, or colours + alpha)."""
    m = cv2.resize(mask.astype(np.uint8) * 255, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    if not bgra:
        return m
    out = np.random.default_rng(1).integers(0, 256, (H // 2, W // 2, 4), dtype=np.uint8)  # junk colours
    out[..., 3] = m
    return out


def sam_fake(by_prompt, fake=None, bgra=False):
    """FakeFal whose sam-3 handler returns `by_prompt[prompt]` = list of (mask, score) as mask URLs + metadata."""
    fake = fake or FakeFal()

    def handler(payload, images):
        insts = by_prompt.get(payload["prompt"], [])
        masks, meta = [], []
        for i, (m, score) in enumerate(insts):
            url = fake.add_file(f"mem://{payload['prompt']}_{i}.png", half(m, bgra))
            masks.append({"url": url, "content_type": "image/png", "width": W // 2, "height": H // 2})
            meta.append({"index": i, "score": score, "box": [0.5, 0.5, 0.1, 0.1]})
        return {"masks": masks, "metadata": meta, "scores": [s for _, s in insts]}

    fake.handlers[SAM3_ENDPOINT] = handler
    return fake


def scene():
    target = rect(CAR)
    neighbour = rect(NEIGHBOUR) & ~target
    return target, neighbour


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("bgra", [False, True])
def test_sam3_masks_are_parsed_at_full_size(bgra):
    target, neighbour = scene()
    fake = sam_fake({"car": [(target, 0.9), (neighbour, 0.7), (np.zeros((H, W), bool), 0.2)]}, bgra=bgra)
    image = noise()
    insts = sam3_instances(fake, image, "car", max_side=500)
    assert len(insts) == 2  # the empty mask is dropped
    assert insts[0].mask.shape == (H, W) and insts[0].mask.dtype == bool
    assert np.array_equal(insts[0].mask, target)
    assert insts[0].box == (200.0, 250.0, 700.0, 500.0)
    assert insts[0].score == pytest.approx(0.9) and insts[1].score == pytest.approx(0.7)
    assert insts[0].label == "car"
    endpoint, payload, images = fake.calls[0]
    assert endpoint == SAM3_ENDPOINT and payload["prompt"] == "car" and payload["apply_mask"] is False
    assert payload["return_multiple_masks"] is True and images["image_url"] is image


def test_sam3_bad_response_raises():
    fake = FakeFal({SAM3_ENDPOINT: lambda p, i: {"image": {"url": "x"}}})
    with pytest.raises(FalError):
        sam3_instances(fake, noise(), "car")


def test_mask_from_image_reads_colour_only_when_alpha_is_opaque():
    grey = np.zeros((10, 20, 4), np.uint8)
    grey[..., 3] = 255
    grey[:, :10, :3] = 255
    assert mask_from_image(grey, (10, 20))[:, :10].all() and not mask_from_image(grey, (10, 20))[:, 10:].any()


def test_birefnet_alpha_full_size_from_mask_only():
    target, _ = scene()
    small = cv2.resize(soft(target), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    m8 = np.round(small * 255).astype(np.uint8)
    fake = FakeFal({BIREFNET_ENDPOINT: lambda p, i: {"image": {"url": "mem://alpha.png", "width": W // 2}}})
    fake.add_file("mem://alpha.png", m8)
    alpha = birefnet_alpha(fake, noise(), max_side=500)
    assert alpha.shape == (H, W) and alpha.dtype == np.float32
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0
    expect = cv2.resize(m8.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_LINEAR)
    assert np.allclose(alpha, expect, atol=1e-6)
    payload = fake.calls[0][1]
    assert payload["mask_only"] is True and "model" in payload


def test_birefnet_alpha_ignores_colours_of_an_rgba_cutout():
    target, _ = scene()
    rgba = np.random.default_rng(3).integers(0, 256, (H, W, 4), dtype=np.uint8)
    rgba[..., 3] = target.astype(np.uint8) * 255
    fake = FakeFal({BIREFNET_ENDPOINT: lambda p, i: {"image": {"url": "mem://cut.png"}}})
    fake.add_file("mem://cut.png", rgba)
    assert np.array_equal(birefnet_alpha(fake, noise()) >= 0.5, target)


def test_birefnet_empty_mask_is_a_fal_error():
    """fal's "Light 2K" returned all-black masks live; that must fall back to the local model, not skip the photo."""
    import pytest

    from hosted.fal import FalError

    fake = FakeFal({BIREFNET_ENDPOINT: lambda p, i: {"image": {"url": "mem://empty.png"}}})
    fake.add_file("mem://empty.png", np.zeros((H // 2, W // 2), np.uint8))
    with pytest.raises(FalError):
        birefnet_alpha(fake, noise())


# ---------------------------------------------------------------- isolation

def test_isolator_picks_the_blob_car_and_removes_the_neighbour():
    target, neighbour = scene()
    raw = soft(target | neighbour)
    # the neighbour comes first and scores higher: the pick must follow the blob, not the order or the score
    fake = sam_fake({"car": [(neighbour, 0.95), (target, 0.8)]})
    res = HostedIsolator(fake, other_prompts=()).isolate(noise(), raw)
    assert res.status == "ok" and res.vehicles == 1
    assert np.array_equal(res.target_mask, target)
    assert res.remove[150:240, 600:880].all()  # neighbour gone
    core = cv2.erode(target.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert not (res.remove & core).any()  # never inside the target
    assert res.glass_ok[target].all() and res.glass_ok.sum() > target.sum()
    assert res.target.box == (200.0, 250.0, 700.0, 500.0)


def test_duplicate_and_part_instances_are_not_removed():
    target, _ = scene()
    dup = rect((205, 250, 700, 500))
    mirror = rect((640, 230, 690, 270))  # a mirror SAM 3 calls "car": box mostly inside the target's
    raw = soft(target | mirror)
    fake = sam_fake({"car": [(target, 0.9), (dup, 0.8), (mirror, 0.4)]})
    res = HostedIsolator(fake, other_prompts=()).isolate(noise(), raw)
    assert res.status == "ok" and res.vehicles == 0 and not res.remove.any() and res.others == []


def test_a_person_touching_the_car_is_removed():
    target, _ = scene()
    person = rect(PERSON) & ~target
    raw = soft(target | rect(PERSON))
    fake = sam_fake({"car": [(target, 0.9)], "person": [(person, 0.9)]})
    res = HostedIsolator(fake).isolate(noise(), raw)
    assert res.status == "ok" and res.objects == 1 and res.vehicles == 0
    assert res.remove[230:245, 705:755].all()
    assert [d.kind for d in res.others] == ["person"]
    car = clean_alpha(raw, 1, res.remove, res.glass_ok)
    assert car.alpha[230:245, 705:755].max() == 0.0 and car.alpha[300:450, 250:650].min() == 1.0


def test_disagreement_is_unsure_and_changes_nothing():
    target, neighbour = scene()
    raw = soft(target | neighbour)
    part = rect((200, 400, 330, 500))  # SAM 3 found only a wheel-arch-sized piece of the car
    fake = sam_fake({"car": [(part, 0.9)]})
    res = HostedIsolator(fake).isolate(noise(), raw)
    assert res.status in ("unsure", "no-vehicle") and res.remove is None and res.glass_ok is None
    fake = sam_fake({"car": [(rect((150, 220, 750, 520)) & ~rect((200, 250, 450, 500)), 0.9)]})
    res = HostedIsolator(fake).isolate(noise(), soft(target))
    assert res.status == "unsure" and res.remove is None and res.glass_ok is None


def test_no_car_is_no_vehicle_and_empty_raw_is_no_object():
    target, _ = scene()
    res = HostedIsolator(sam_fake({})).isolate(noise(), soft(target))
    assert res.status == "no-vehicle" and res.remove is None
    res = HostedIsolator(sam_fake({})).isolate(noise(), np.zeros((H, W), np.float32))
    assert res.status == "no-object"


def test_isolation_is_deterministic():
    target, neighbour = scene()
    raw = soft(target | neighbour | rect(PERSON))
    by = {"car": [(target, 0.9), (neighbour, 0.8)], "person": [(rect(PERSON) & ~target, 0.9)]}
    a = HostedIsolator(sam_fake(by)).isolate(noise(), raw)
    b = HostedIsolator(sam_fake(by)).isolate(noise(), raw)
    assert np.array_equal(a.remove, b.remove) and np.array_equal(a.glass_ok, b.glass_ok)


def test_finish_with_hosted_isolator_changes_only_the_mask():
    target, neighbour = scene()
    raw = soft(target | neighbour)
    image = noise(5)
    before = image.copy()
    seg = CarSegmenter.__new__(CarSegmenter)
    seg.edge_shrink, seg.last_isolation = 1, None
    seg.raw_alpha = lambda img: raw
    fake = sam_fake({"car": [(target, 0.9), (neighbour, 0.8)]})
    car = seg.finish(image, (np.round(raw * 255)).astype(np.uint8), HostedIsolator(fake))
    assert np.array_equal(image, before)  # car pixels never modified
    assert car.isolation["status"] == "ok" and car.isolation["vehicles"] == 1
    assert car.alpha[150:240, 600:880].max() == 0.0  # neighbour dropped from the mask
    assert car.alpha[300:450, 250:650].min() == 1.0  # the car itself kept
    assert seg.last_isolation.target_mask is not None
    plain = clean_alpha(raw, 1)
    assert plain.alpha[150:240, 600:880].max() > 0  # without isolation the neighbour would stay
