"""Model-free tests: the cut-out keeps only the main car, and the car's pixels are only downscaled.

Run from the background_changer folder:  .venv\\Scripts\\python -m pytest tests
"""

import cv2
import numpy as np
import pytest

from bg_changer import SCENES, clean_alpha, load_scene, place_car, probe_car
from bg_changer.yaw import YawEstimate

H_SRC, W_SRC = 1000, 2000
BODY = (200, 300, 1800, 800)  # x1, y1, x2, y2 of the fake car
WINDOW = (600, 400, 900, 500)  # a window the "model" was unsure about
PARKED = (20, 20, 120, 80)  # a smaller, off-centre object (another parked car)


def fake_photo():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (H_SRC, W_SRC, 3), dtype=np.uint8)  # noise = every pixel distinct
    raw = np.zeros((H_SRC, W_SRC), np.float32)
    x1, y1, x2, y2 = BODY
    raw[y1:y2, x1:x2] = 1.0
    x1, y1, x2, y2 = PARKED
    raw[y1:y2, x1:x2] = 1.0
    raw = cv2.GaussianBlur(raw, (0, 0), 2)  # soft edges, like a real model
    x1, y1, x2, y2 = WINDOW
    raw[y1:y2, x1:x2] = 0.2
    return image, raw


def test_clean_alpha_keeps_main_car_only_and_fills_windows():
    _, raw = fake_photo()
    car = clean_alpha(raw, edge_shrink=1)
    x1, y1, x2, y2 = PARKED
    assert car.alpha[y1:y2, x1:x2].max() == 0, "parked car was not removed"
    x1, y1, x2, y2 = WINDOW
    assert car.alpha[y1:y2, x1:x2].min() == 1.0, "window is not opaque"
    assert all(abs(a - b) <= 2 for a, b in zip(car.box, BODY))
    assert car.cut_sides == []
    assert car.bottom is not None and len(car.bottom) == car.box[2] - car.box[0]


def test_post_behind_roof_is_cut_whole_but_mirror_and_pillars_stay():
    _, raw = fake_photo()
    raw[100:300, 1000:1016] = 1.0  # 16 px lamp post standing on the roof (body top is y=300)
    raw[340:400, 1800:1900] = 1.0  # a mirror sticking out sideways from the body's right side...
    raw[355:370, 1790:1800] = 1.0  # ...joined by a thin (15 px tall) stalk
    raw[320:420, 400:1100] = 0.1  # wide see-through window...
    raw[320:420, 700:716] = 1.0  # ...split by a 16 px pillar (as thin as the post)
    raw[790:815, 1700:1716] = 1.0  # a tyre bottom poking 15 px below the body: as narrow as the post, but short
    car = clean_alpha(raw, edge_shrink=1)
    assert car.alpha[100:296, 1000:1016].max() == 0, "post above the roof was kept"
    assert car.alpha[345:395, 1805:1895].min() == 1.0, "mirror was cut"
    assert car.alpha[330:410, 420:1080].min() == 1.0, "window/pillar area is not solid"
    assert car.alpha[800:808, 1704:1712].min() > 0.9, "short edge detail was cut"


@pytest.mark.parametrize("scene_name", sorted(SCENES))
def test_car_pixels_are_only_downscaled_on_every_scene(scene_name, side_car, grey_plate, table):
    """With shadows, contact lines and reflections on, the opaque car pixels are exactly one
    INTER_AREA downscale of the photo: nothing is drawn over the car."""
    image, car = side_car
    scene = load_scene(scene_name, grey_plate, table)
    probe = probe_car(car)
    out, plan = place_car(image, car, grey_plate, scene, probe, YawEstimate(90.0, 15.0, "label"), shadow=1.5)
    assert plan.scale < 1
    cx1, cy1, cx2, cy2 = car.crop
    expected = cv2.resize(image[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(car.alpha[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    opaque = alpha > 1 - 1e-4
    assert opaque.mean() > 0.5
    x0, y0 = plan.top_left
    nw, nh = plan.size
    assert x0 >= 0 and y0 >= 0 and x0 + nw <= 1920 and y0 + nh <= 1080
    got = out[y0:y0 + nh, x0:x0 + nw]
    assert np.array_equal(got[opaque], expected[opaque]), "car pixels changed beyond the downscale"
    # ...and the floor just under the car, outside the car itself, got darker (the shadow is wired up).
    car_mask = np.zeros(out.shape[:2], bool)
    car_mask[y0:y0 + nh, x0:x0 + nw] = alpha > 0
    x1, _, x2, y2 = plan.box_out
    under = np.zeros_like(car_mask)
    under[y2:y2 + 20, x1:x2] = True
    under &= ~car_mask
    assert under.any()
    assert out[under].mean() < grey_plate[under].mean() - 10


def test_cut_frames_skip_returns_nothing(side_car, grey_plate, table):
    image, car = side_car
    car.cut["right"] = True
    scene = load_scene("background_1", grey_plate, table)
    out, plan = place_car(image, car, grey_plate, scene, probe_car(car), YawEstimate(90.0, 15.0, "label"),
                          cut_policy="skip")
    assert out is None and plan is None
