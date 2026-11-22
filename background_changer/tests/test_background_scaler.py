"""background_scaler: a background shrunk to the photos' resolution keeps its stage, the car fills it,
and the car is never enlarged."""

import cv2
import numpy as np
import pytest

from background_scaler import (BackgroundScaler, background_size, crop_to_car, folder_scale, needed_scale,
                               scale_scene)
from bg_changer import SCENES, clean_alpha, load_scene, place_car, probe_car
from bg_changer.yaw import YawEstimate

SIDE = YawEstimate(90.0, 15.0, "label")


def small_photo(body=(100, 175, 850, 390), tyres=((225, 425), (725, 425)), r=57, seed=1):
    """A 960x540 photo of a side view whose car (250 px tall by default: body plus tyres) is too small
    for any full-size stage (about 500 px for a side view on background_1)."""
    raw = np.zeros((540, 960), np.float32)
    x1, y1, x2, y2 = body
    raw[y1:y2, x1:x2] = 1.0
    for cx, cy in tyres:  # tyre bottoms at cy, overlapping the body
        cv2.circle(raw, (cx, cy - r), r, 1.0, -1)
    image = np.random.default_rng(seed).integers(0, 256, (540, 960, 3), dtype=np.uint8)
    return image, clean_alpha(cv2.GaussianBlur(raw, (0, 0), 1.5), edge_shrink=1)


def test_background_size_is_rounded_up_and_kept_in_range():
    assert background_size((1920, 1080), 1.0) == (1920, 1080)
    assert background_size((1920, 1080), 1.7) == (1920, 1080)  # never enlarged
    assert background_size((1920, 1080), 0.5) == (960, 540)
    assert background_size((1920, 1080), 0.34) == (653, 367)  # 652.8 rounded up: the car never needs enlarging
    assert background_size((1920, 1080), 0.1) == (640, 360)  # never below the minimum width
    assert background_size((1920, 1080), 0.1, min_width=100) == (192, 108)
    assert background_size((600, 400), 0.5, min_width=640) == (600, 400)


def test_folder_size_is_set_by_the_smallest_car():
    assert folder_scale([0.8, None, 0.34, 1.0]) == 0.34
    assert folder_scale([None, None]) == 1.0
    assert folder_scale([]) == 1.0


def test_scaler_settings_are_checked():
    with pytest.raises(ValueError):
        BackgroundScaler("big")
    with pytest.raises(ValueError):
        BackgroundScaler("photo", 0)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_shrunk_scene_is_the_same_stage_k_times_smaller(name, grey_plate, table):
    full = load_scene(name, grey_plate, table)
    small, plate = scale_scene(full, grey_plate, (960, 540))
    k = 0.5

    def point(v):  # pixel centres, as cv2.resize maps them
        return (v + 0.5) * k - 0.5

    assert plate.shape == (540, 960, 3) and small.size_wh == (960, 540)
    assert small.px_per_m == pytest.approx(k * full.px_per_m)
    (ax, ay), (sx, sy) = full.spec.anchor, small.spec.anchor
    assert (sx, sy) == pytest.approx((point(ax), point(ay)))
    for s in (-1.0, 0.0, 1.0, 4.0):  # the floor recedes the same way
        assert float(small.plate_floor.row(s)) == pytest.approx(point(float(full.plate_floor.row(s))), abs=1e-6)
    for yaw in range(0, 360, 30):  # every view stands on the same spot, k times smaller
        r = table.at(yaw)
        assert sy + small.px_per_m * r.low == pytest.approx(point(ay + full.px_per_m * r.low), abs=1e-6)
    assert small.mask[int(round(sy)), int(round(sx))] > 0.99, "anchor is not on the floor"
    assert small.mask.shape == (540, 960)
    if full.spec.max_lowest_row is not None:
        assert small.spec.max_lowest_row == pytest.approx(point(full.spec.max_lowest_row))
    if full.spec.reflection is not None:
        assert small.spec.reflection.blur_px == pytest.approx(tuple(k * b for b in full.spec.reflection.blur_px))
    if full.spec.ring is not None:
        assert small.inside[int(round(sy)), int(round(sx))] > 0.99, "stage centre is off the turntable"
    assert small.spec.px_scale == pytest.approx(k)


def test_a_small_photo_gets_a_smaller_background_and_keeps_every_car_pixel(grey_plate, table):
    image, car = small_photo()
    probe = probe_car(car)
    scene = load_scene("background_1", grey_plate, table)
    _, full_plan = place_car(image, car, grey_plate, scene, probe, SIDE)
    assert "scale-capped" in full_plan.flags  # at full size this car would be too small for the stage

    scaler = BackgroundScaler("photo")
    k = scaler.photo_scale(scene, probe, SIDE)
    assert 0.3 < k < 0.7
    small, plate = scaler.apply(scene, grey_plate, k)
    out, plan = place_car(image, car, plate, small, probe, SIDE, shadow=1.5)
    assert out.shape[:2] == (small.size_wh[1], small.size_wh[0])
    assert plan.scale == 1.0 and "scale-capped" not in plan.flags
    _, y1, _, y2 = plan.box_out
    assert y2 - y1 == pytest.approx(plan.target_h, abs=1.5), "the car does not fill the smaller stage"
    # Pasted pixel for pixel: the opaque car pixels are the photo's own, untouched.
    cx1, cy1, cx2, cy2 = car.crop
    opaque = car.alpha[cy1:cy2, cx1:cx2] >= 1.0
    x0, y0 = plan.top_left
    got = out[y0:y0 + cy2 - cy1, x0:x0 + cx2 - cx1]
    assert got.shape[:2] == opaque.shape and opaque.mean() > 0.5
    assert np.array_equal(got[opaque], image[cy1:cy2, cx1:cx2][opaque])


def test_folder_mode_sizes_the_background_for_the_smallest_car(grey_plate, table):
    big = small_photo()  # car 250 px tall
    tiny = small_photo(body=(200, 250, 700, 380), tyres=((300, 420), (600, 420)), r=40, seed=2)  # 170 px
    scene = load_scene("background_1", grey_plate, table)
    photos = [(img, car, probe_car(car)) for img, car in (big, tiny)]
    scaler = BackgroundScaler("folder")
    k = scaler.folder_scale(needed_scale(scene, p, SIDE) for _, _, p in photos)
    assert k == pytest.approx(needed_scale(scene, photos[1][2], SIDE))
    small, plate = scaler.apply(scene, grey_plate, k)
    plans = [place_car(img, car, plate, small, p, SIDE)[1] for img, car, p in photos]
    assert plans[1].scale == 1.0 and 0.5 < plans[0].scale < 1.0
    for plan in plans:  # both fill the stage, neither is enlarged
        assert "scale-capped" not in plan.flags
        assert plan.box_out[3] - plan.box_out[1] == pytest.approx(plan.target_h, abs=1.5)


def test_photos_with_enough_pixels_keep_the_full_size_background(side_car, grey_plate, table):
    _, car = side_car  # 4K: the car is far bigger than any stage needs
    scene = load_scene("background_1", grey_plate, table)
    scaler = BackgroundScaler("folder")
    k = scaler.folder_scale([needed_scale(scene, probe_car(car), SIDE)])
    assert k == 1.0
    small, plate = scaler.apply(scene, grey_plate, k)
    assert small is scene and plate is grey_plate
    assert BackgroundScaler("off").photo_scale(scene, probe_car(small_photo()[1]), SIDE) == 1.0


@pytest.mark.parametrize("name", ["background_1", "background_3"])
def test_the_shrunk_result_looks_like_the_full_size_result_made_smaller(name, side_car, grey_plate, table):
    """Shadows, contact lines and the reflection shrink with the background (nothing left at full-size
    pixel sizes). The car itself is compared elsewhere: here it is masked out."""
    image, car = side_car
    probe = probe_car(car)
    scene = load_scene(name, grey_plate, table)
    full, _ = place_car(image, car, grey_plate, scene, probe, SIDE)
    small_scene, small_plate = scale_scene(scene, grey_plate, (960, 540))
    small, plan = place_car(image, car, small_plate, small_scene, probe, SIDE)
    expected = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)
    cx1, cy1, cx2, cy2 = car.crop
    alpha = cv2.resize(car.alpha[cy1:cy2, cx1:cx2], plan.size, interpolation=cv2.INTER_AREA)
    on_car = np.zeros((540, 960), np.uint8)
    x0, y0 = plan.top_left
    on_car[y0:y0 + plan.size[1], x0:x0 + plan.size[0]] = alpha > 0
    floor = ~cv2.dilate(on_car, np.ones((5, 5), np.uint8)).astype(bool)
    diff = np.abs(small.astype(np.int16) - expected.astype(np.int16))[floor]
    shaded = (expected.astype(np.int16) < small_plate.astype(np.int16) - 3)[floor]
    assert shaded.any(), "no shadow to compare"
    assert diff.mean() < 0.5 and np.percentile(diff, 99.9) <= 12


def test_crop_to_car_places_the_same_picture(side_car, grey_plate, table):
    image, car = side_car
    scene = load_scene("background_1", grey_plate, table)
    whole, plan = place_car(image, car, grey_plate, scene, probe_car(car), SIDE)
    image_c, car_c = crop_to_car(image, car)
    part, plan_c = place_car(image_c, car_c, grey_plate, scene, probe_car(car_c), SIDE)
    assert image_c.shape[0] < image.shape[0] and image_c.shape[1] < image.shape[1]
    assert plan.box_out == plan_c.box_out and np.array_equal(whole, part)
