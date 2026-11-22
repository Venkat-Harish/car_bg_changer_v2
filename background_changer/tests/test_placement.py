"""Placement: height from the car table, ground pin, footprint centre on the anchor, cut frames."""

import numpy as np
import pytest

from bg_changer import clean_alpha, load_scene, probe_car
from bg_changer.carmodel import ASPECT_BIAS
from bg_changer.placement import XMap, plan_on_anchor
from bg_changer.yaw import YawEstimate

from conftest import side_car_alpha


@pytest.fixture()
def scene1(grey_plate, table):
    return load_scene("background_1", grey_plate, table)


def test_height_and_ground_pin(side_car, scene1, table):
    _, car = side_car
    probe = probe_car(car)
    yaw = YawEstimate(90.0, 15.0, "label")
    plan = scene1.plan(car, probe, yaw)
    x1, y1, x2, y2 = plan.box_out
    row = table.at(90.0)
    assert plan.scale < 1
    assert y2 - y1 == pytest.approx(plan.px_per_m * row.h, abs=1.5)
    assert y2 == pytest.approx(scene1.spec.anchor[1] + plan.px_per_m * row.low, abs=1.0)
    # Footprint centre on the anchor: the box centre sits at the (smoothed) predicted offset.
    expected_cx = scene1.spec.anchor[0] + plan.px_per_m * table.cx_smoothed(90.0, 15.0)
    assert (x1 + x2) / 2 == pytest.approx(expected_cx, abs=2.0)
    assert plan.margin >= 150


def test_end_views_are_taller_and_lower_than_side_views(side_car, scene1):
    _, car = side_car
    probe = probe_car(car)
    side = scene1.plan(car, probe, YawEstimate(90.0, 15.0, "label"))
    rear = scene1.plan(car, probe, YawEstimate(180.0, 15.0, "label"))
    assert rear.target_h > side.target_h * 1.2
    assert rear.box_out[3] > side.box_out[3] + 40  # the near end comes toward the camera


def test_never_enlarged(scene1, table):
    raw = side_car_alpha(h=540, w=960, body=(100, 175, 850, 425), tyres=((225, 425), (725, 425)), r=57)
    car = clean_alpha(raw, edge_shrink=1)
    plan = scene1.plan(car, probe_car(car), YawEstimate(90.0, 15.0, "label"))
    assert plan.scale == 1.0 and "scale-capped" in plan.flags


def test_cut_right_puts_the_uncut_edge_where_the_whole_car_would_end(side_car, scene1, table):
    _, car = side_car
    car.cut["right"] = True
    probe = probe_car(car)
    plan = scene1.plan(car, probe, YawEstimate(90.0, 15.0, "label"))
    row = table.at(90.0)
    pred_x0 = scene1.spec.anchor[0] + plan.px_per_m * row.x0
    expected = pred_x0 + 0.5 * (1 - 1 / ASPECT_BIAS) * plan.px_per_m * row.w
    assert plan.box_out[0] == pytest.approx(expected, abs=2.0)
    assert "cut-right" in plan.flags
    centred = scene1.plan(car, probe, YawEstimate(90.0, 15.0, "label"), cut_policy="centre")
    assert centred.box_out[0] != plan.box_out[0]
    assert scene1.plan(car, probe, YawEstimate(90.0, 15.0, "label"), cut_policy="skip") is None


def test_cut_frame_map_keeps_the_pasted_scale_past_the_tyre(scene1, table):
    """Cut on the right, near tyre on the left: beyond the tyre the floor drawing must follow the
    pasted car's scale (1/ASPECT_BIAS), not the short edge-to-tyre stretch."""
    raw = side_car_alpha(tyres=((900, 1700), (2900, 1640)))
    raw[700:1550, 3400:] = 1.0
    car = clean_alpha(raw, edge_shrink=1)
    probe = probe_car(car)
    plan = scene1.plan(car, probe, YawEstimate(90.0, 15.0, "label"))
    far = plan.xmap.src[-1] + 300
    slope = (plan.xmap.fwd(far) - plan.xmap.fwd(far - 100)) / 100
    assert slope == pytest.approx(1 / ASPECT_BIAS, rel=0.01)


def test_bottom_cut_pins_the_roof(side_car, scene1, table):
    _, car = side_car
    full = scene1.plan(car, probe_car(car), YawEstimate(90.0, 15.0, "label"))
    raw = side_car_alpha()
    raw[1600:, :] = 0  # the photo stops above the tyre bottoms
    cut = clean_alpha(raw, edge_shrink=1)
    cut.cut["bottom"] = True
    plan = scene1.plan(cut, probe_car(cut), YawEstimate(90.0, 15.0, "label"))
    row = table.at(90.0)
    assert plan.box_out[1] == pytest.approx(scene1.spec.anchor[1] + plan.px_per_m * row.top, abs=2)
    assert plan.box_out[3] < full.box_out[3]  # the missing bottom is not pulled down to the floor
    assert plan.hw_ratio is None


def test_guard_flags_a_car_that_cannot_fit(scene1, table):
    raw = side_car_alpha()
    car = clean_alpha(raw, edge_shrink=1)
    huge = plan_on_anchor(car, probe_car(car), YawEstimate(180.0, 15.0, "label"), table, scene1.spec.anchor,
                          scene1.px_per_m * 3, (1920, 1080), max_lowest_row=946)
    assert "cropped" in huge.flags


def test_xmap_round_trip_and_extrapolation():
    m = XMap([100, 500, 900], [120, 480, 950])
    x = np.array([0.0, 100, 300, 700, 900, 1200])
    assert np.allclose(m.inv(m.fwd(x)), x)
    assert m.fwd(1000) == pytest.approx(950 + 100 * (950 - 480) / 400)
