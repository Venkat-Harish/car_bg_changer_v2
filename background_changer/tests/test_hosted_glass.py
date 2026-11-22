"""hosted.glass: SAM 3 window mask (FakeFal, no network) + the deterministic dark tint.

Run from the background_changer folder:  .venv\\Scripts\\python -m pytest tests\\test_hosted_glass.py -q
"""

import cv2
import numpy as np
import pytest

from bg_changer import load_scene, place_car, probe_car
from bg_changer.yaw import YawEstimate
from hosted import glass
from hosted.glass import glass_pct, save_original, tint_glass, window_mask
from hosted_fakes import FakeFal

H, W = 600, 1000
CAR = (100, 150, 900, 500)  # x1, y1, x2, y2


def noise(h=H, w=W, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def rect(x1, y1, x2, y2, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def sam_fake(per_prompt, scale=1.0):
    """FakeFal whose SAM 3 handler returns, per prompt, uint8 masks (optionally downscaled like a sent image)."""
    fake = FakeFal()

    def handler(payload, images):
        assert "image_url" in images and payload["apply_mask"] is False
        out = []
        for i, m in enumerate(per_prompt.get(payload["prompt"], [])):
            u8 = m.astype(np.uint8) * 255
            if scale != 1.0:
                u8 = cv2.resize(u8, (round(m.shape[1] * scale), round(m.shape[0] * scale)),
                                interpolation=cv2.INTER_NEAREST)
            out.append({"url": fake.add_file(f"mem://{payload['prompt']}/{i}.png", u8), "content_type": "image/png"})
        return {"masks": out}

    fake.handlers[glass.SAM3_ENDPOINT] = handler
    return fake


# ---------------------------------------------------------------------------------------------------- tint
def test_tint_changes_only_inside_the_mask_and_darkens_it():
    image = noise()
    before = image.copy()
    mask = rect(300, 200, 700, 320)
    out = tint_glass(image, mask)
    assert np.array_equal(image, before), "input was modified"
    assert out.dtype == np.uint8 and out.shape == image.shape
    assert np.array_equal(out[~mask], image[~mask]), "pixels outside the mask changed"
    inner = rect(310, 210, 690, 310)
    assert out[inner].mean() < image[inner].mean() - 60
    assert out[inner].std() < image[inner].std() * 0.5  # mostly the smooth gradient, a faint trace of the original
    # lighter at the top than at the bottom
    assert out[212:220, 310:690].mean() > out[300:308, 310:690].mean() + 10


def test_tint_edge_is_feathered_not_a_hard_step():
    image = np.full((H, W, 3), 200, np.uint8)
    mask = rect(300, 200, 700, 320)
    out = tint_glass(image, mask).astype(int)
    row = out[260, 295:320, 1]
    assert row[0] == 200  # outside
    assert 200 > row[5] > row[-1]  # first mask column only partly tinted, deeper inside darker


def test_tint_is_deterministic_and_image_independent_in_level():
    mask = rect(300, 200, 700, 320)
    a = tint_glass(noise(seed=1), mask)
    b = tint_glass(noise(seed=1), mask)
    assert np.array_equal(a, b)
    # two different frames get (nearly) the same glass look: the level comes from constants, not the image
    c = tint_glass(noise(seed=2), mask)
    inner = rect(310, 210, 690, 310)
    assert abs(float(a[inner].mean()) - float(c[inner].mean())) < 2


def test_empty_mask_changes_nothing():
    image = noise()
    out = tint_glass(image, np.zeros((H, W), bool))
    assert np.array_equal(out, image) and out is not image


# ---------------------------------------------------------------------------------------------------- mask
def test_window_mask_unions_prompts_and_stays_inside_the_eroded_car():
    image = noise()
    car = rect(*CAR)
    side = rect(50, 200, 400, 300)  # spills out of the car on the left, into the background
    screen = rect(500, 160, 800, 260)  # touches the car's top edge
    fake = sam_fake({"car window": [side], "windshield": [screen]}, scale=0.5)
    got = window_mask(fake, image, car.astype(np.float32), max_side=500)
    assert [c[1]["prompt"] for c in fake.calls] == list(glass.PROMPTS)
    assert got.dtype == bool and got.shape == (H, W)
    r = max(1, round(glass.ERODE_FRAC * W))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(car.astype(np.uint8), kernel, borderValue=0).astype(bool)
    assert not (got & ~inner).any(), "glass reaches the car edge or the background"
    assert got[250, 200] and got[210, 650], "a prompt's instance is missing"
    assert not got[250, 80]


def test_window_mask_drops_tiny_components():
    car = rect(*CAR)
    big = rect(300, 200, 600, 300)
    tiny = rect(700, 400, 710, 410)  # 100 px < 0.05% of 280000
    fake = sam_fake({"car window": [big | tiny]})
    got = window_mask(fake, noise(), car)
    assert got[250, 450] and not got[405, 705]


def test_empty_response_gives_no_glass_and_no_change():
    image = noise()
    fake = sam_fake({})
    got = window_mask(fake, image, rect(*CAR))
    assert got.shape == (H, W) and not got.any()
    assert np.array_equal(tint_glass(image, got), image)
    fake.handlers[glass.SAM3_ENDPOINT] = lambda p, i: {}
    assert not window_mask(fake, image, rect(*CAR)).any()


def test_rgba_mask_files_are_read():
    car = rect(*CAR)
    m = rect(300, 200, 600, 300)
    fake = FakeFal()
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = m * 255
    fake.add_file("mem://a.png", rgba)
    fake.handlers[glass.SAM3_ENDPOINT] = lambda p, i: {"masks": [{"url": "mem://a.png"}]}
    assert window_mask(fake, noise(), car)[250, 450]


def test_glass_pct():
    car = rect(*CAR)
    assert glass_pct(rect(100, 150, 500, 500), car) == pytest.approx(50.0)
    assert glass_pct(rect(0, 0, 10, 10), np.zeros((H, W), bool)) == 0.0


# ---------------------------------------------------------------------------------------------------- originals
def test_save_original_is_byte_identical(tmp_path):
    src_dir = tmp_path / "photos"
    src_dir.mkdir()
    src = src_dir / "frame_0001.jpg"
    data = bytes(range(256)) * 50
    src.write_bytes(data)
    dst = save_original(src, tmp_path / "out" / "originals")
    assert dst == tmp_path / "out" / "originals" / "frame_0001.jpg"
    assert dst.read_bytes() == data
    assert src.read_bytes() == data


def test_save_original_refuses_the_source_folder(tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        save_original(src, tmp_path / "sub" / "..")
    assert src.read_bytes() == b"x"


# ---------------------------------------------------------------------------------------------------- end to end
def test_tinted_frame_then_flat_placement_changes_only_the_windows(side_car, grey_plate, table):
    """Tint, then the real clean_alpha mask + flat place_car: opaque output pixels are exactly INTER_AREA of the
    tinted crop, and differ from the untinted frame only where the downscaled window mask is set."""
    image, car = side_car
    window = np.zeros(image.shape[:2], bool)
    window[800:1100, 1200:2000] = True  # inside the body (700..1550 x 400..3400)
    window &= car.alpha >= 1.0
    tinted = tint_glass(image, window)
    assert np.array_equal(tinted[~window], image[~window])
    scene = load_scene("background_1", grey_plate, table)
    out, plan = place_car(tinted, car, grey_plate, scene, probe_car(car), YawEstimate(90.0, 15.0, "label"), shadow=1.5)
    assert plan.scale < 1
    cx1, cy1, cx2, cy2 = car.crop
    expected = cv2.resize(tinted[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    untinted = cv2.resize(image[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(car.alpha[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    win = cv2.resize(window[cy1:cy2, cx1:cx2].astype(np.float32), plan.size, interpolation=cv2.INTER_AREA) > 0
    opaque = alpha > 1 - 1e-4
    x0, y0 = plan.top_left
    nw, nh = plan.size
    got = out[y0:y0 + nh, x0:x0 + nw]
    assert np.array_equal(got[opaque], expected[opaque]), "car pixels changed beyond the downscale"
    changed = np.any(got != untinted, axis=2) & opaque
    assert changed.any(), "the tint did not reach the output"
    assert not (changed & ~win).any(), "pixels outside the window mask changed"
