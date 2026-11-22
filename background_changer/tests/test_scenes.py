"""Every background's scene keeps every view of the car inside the picture and on its floor."""

import warnings

import numpy as np
import pytest

from bg_changer import SCENES, load_scene

H, W = 1080, 1920


@pytest.mark.parametrize("name", sorted(SCENES))
def test_all_views_fit_and_stand_on_the_floor(name, grey_plate, table):
    scene = load_scene(name, grey_plate, table)
    ax, ay = scene.spec.anchor
    p = scene.px_per_m
    for yaw in range(0, 360, 15):
        r = table.at(yaw)
        x0, x1, top, low = ax + p * r.x0, ax + p * r.x1, ay + p * r.top, ay + p * r.low
        assert min(x0, W - x1, top) >= 150, f"{name} yaw {yaw}: too close to the edge"
        assert low <= H - 120, f"{name} yaw {yaw}: too low"
        if scene.spec.max_lowest_row is not None:
            assert low <= scene.spec.max_lowest_row + 1, f"{name} yaw {yaw}: past the scene's limit"
    assert scene.mask[int(ay), int(ax)] > 0.99, "anchor is not on the floor"
    assert float(scene.plate_floor.row(0.0)) == pytest.approx(ay, abs=1.5), "plate floor misses the anchor"


def test_turntable_side_view_fills_about_three_quarters_of_the_disc(grey_plate, table):
    scene = load_scene("background_1", grey_plate, table)
    side_px = scene.px_per_m * table.side_length()
    assert 0.70 <= side_px / 1761 <= 0.80


def test_overrides_and_generic_scene(grey_plate, table):
    small = load_scene("background_1", grey_plate, table, {"size": 0.9, "reflection": False})
    base = load_scene("background_1", grey_plate, table)
    assert small.px_per_m == pytest.approx(0.9 * base.px_per_m)
    assert small.spec.reflection is None
    with pytest.warns(UserWarning, match="smaller"):  # too big for the turntable: shrunk back to fit
        big = load_scene("background_1", grey_plate, table, {"size": 1.3})
    assert base.px_per_m < big.px_per_m < 1.3 * base.px_per_m
    g = load_scene("My Showroom", grey_plate, table, {"anchor": [960, 760], "side_fill": 0.6, "horizon_y": 520})
    assert g.spec.anchor == (960.0, 760.0)
    assert g.px_per_m * table.side_length() == pytest.approx(0.6 * W, rel=0.01)
    assert float(g.plate_floor.row(0.0)) == pytest.approx(760, abs=1.0)


def test_generic_scene_shrinks_an_oversized_car_with_a_warning(grey_plate, table):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = load_scene("tiny_room", grey_plate, table, {"side_fill": 1.0})
    assert g.spec.size < 1.0
    assert any("smaller" in str(w.message) for w in caught)


def test_a_tall_van_is_shrunk_to_fit_with_a_warning(grey_plate):
    from bg_changer import CarShape, CarTable
    van = CarTable(CarShape(length=5.93, width=2.02, height=2.68, wheelbase=3.665, name="Sprinter"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene = load_scene("background_1", grey_plate, van)
    assert scene.spec.size < 1.0 and any("smaller" in str(w.message) for w in caught)
    ax, ay = scene.spec.anchor
    for yaw in range(0, 360, 15):
        r = van.at(yaw)
        assert ay + scene.px_per_m * r.top >= 40 and ay + scene.px_per_m * r.low <= 946 + 1


def test_anchor_override_re_pins_the_calibrated_floor(grey_plate, table):
    for name, anchor in (("background_1", [958, 800]), ("background_2", [960, 820]), ("background_3", [960, 720])):
        scene = load_scene(name, grey_plate, table, {"anchor": anchor})
        assert float(scene.plate_floor.row(0.0)) == pytest.approx(anchor[1], abs=1.5)


def test_generic_reflection_switch(grey_plate, table):
    on = load_scene("showroom", grey_plate, table, {"floor": "glossy"})
    off = load_scene("showroom", grey_plate, table, {"floor": "glossy", "reflection": False})
    assert on.spec.reflection is not None and off.spec.reflection is None


def test_wrong_plate_size_is_refused(table):
    with pytest.raises(ValueError):
        load_scene("background_1", np.zeros((720, 1280, 3), np.uint8), table)
