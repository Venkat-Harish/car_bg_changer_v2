"""Shadows and reflections only touch the floor, follow the real car, and stay in range."""

import numpy as np
import pytest

from bg_changer import clean_alpha, load_scene, probe_car, scale_car
from bg_changer.grounding import (GAP_MAX_M, PinnedFloorMap, ShadowStyle, _silhouette_layers, add_reflection,
                                  linear_to_srgb, outline_floor, shadow_multiplier, snap_contacts, srgb_to_linear)
from bg_changer.yaw import YawEstimate

from conftest import side_car_alpha


def _setup(car, scene_name, plate, table, yaw=90.0):
    image = np.random.default_rng(0).integers(0, 256, car.alpha.shape + (3,), dtype=np.uint8)
    scene = load_scene(scene_name, plate, table)
    probe = probe_car(car)
    plan = scene.plan(car, probe, YawEstimate(yaw, 15.0, "label"))
    pixels, alpha = scale_car(image, car, plan.size)
    x1, y1, x2, y2 = plan.box_out
    roi = (max(0, x1 - 200), (y1 + y2) // 2, min(1920, x2 + 200), 1080)
    floor = outline_floor(alpha, plan.top_left, plan.px_per_m)
    return scene, probe, plan, pixels, alpha, roi, floor


@pytest.fixture()
def setup(side_car, grey_plate, table):
    return _setup(side_car[1], "background_1", grey_plate, table)


def test_multiplier_in_range_and_one_outside_mask(setup, table):
    scene, probe, plan, pixels, alpha, roi, floor = setup
    rx1, ry1, rx2, ry2 = roi
    mask = np.ones((ry2 - ry1, rx2 - rx1), np.float32)
    mask[:, : (rx2 - rx1) // 2] = 0.0  # no floor on the left half
    fm = PinnedFloorMap(table, plan, scene.plate_floor)
    m = shadow_multiplier(plan, table, scene.spec.shadow, fm, snap_contacts(plan, probe, floor), mask, roi, floor)
    assert m.min() >= 0 and m.max() <= 1
    assert np.all(m[:, : (rx2 - rx1) // 2] == 1.0)
    assert m[:, (rx2 - rx1) // 2:].min() < 0.5  # a real shadow on the right half


def test_tint_leaves_unshadowed_floor_exactly_alone(setup, table):
    scene, probe, plan, pixels, alpha, roi, floor = setup
    rx1, ry1, rx2, ry2 = roi
    fm = PinnedFloorMap(table, plan, scene.plate_floor)
    style = ShadowStyle(tint_gamma_bgr=(1.114, 1.0, 0.889))
    mask = np.ones((ry2 - ry1, rx2 - rx1), np.float32)
    m = shadow_multiplier(plan, table, style, fm, snap_contacts(plan, probe, floor), mask, roi, floor, 0.0)
    assert np.all(m == 1.0)


def test_pinned_map_is_continuous_and_monotonic(setup, table):
    scene, probe, plan, *_ = setup
    fm = PinnedFloorMap(table, plan, scene.plate_floor)
    rows = np.arange(fm.v_att - 20, fm.v_att + 120, 0.5)
    src = fm.source_rows(rows)
    assert np.all(np.diff(src) > 0)
    assert abs(fm.source_rows(np.array([fm.v_att + 0.5]))[0] - (fm.v_att + 0.5)) < 1.5  # no jump at the car
    assert np.allclose(fm.output_rows(src), rows, atol=0.5)


def test_side_view_contacts_snap_onto_the_tyres(setup):
    scene, probe, plan, pixels, alpha, roi, floor = setup
    contacts = snap_contacts(plan, probe, floor)
    assert sum(c.snapped for c in contacts) == 2  # both near tyres matched
    for c in contacts:
        if c.snapped:
            assert abs(c.v - plan.box_out[3]) <= 3


def test_floor_line_follows_a_raised_tyre(grey_plate, table):
    """One tyre higher in the picture (the angle label is off): the floor line must still pass under
    it, so no dark slab is drawn below it."""
    raw = side_car_alpha(tyres=((900, 1640), (2900, 1720)))  # rear tyre 80 px higher than the front one
    car = clean_alpha(raw, edge_shrink=1)
    scene, probe, plan, pixels, alpha, roi, floor = _setup(car, "background_2", grey_plate, table)
    rear_u = plan.to_out(900, 0)[0]
    front_u = plan.to_out(2900, 0)[0]
    i_r = int(np.argmin(abs(floor.u - rear_u)))
    i_f = int(np.argmin(abs(floor.u - front_u)))
    assert floor.height[i_r] <= 3 and floor.height[i_f] <= 3, "floor line misses a tyre"
    contact, gap, band = _silhouette_layers(floor, roi, plan.px_per_m)
    rx1, ry1 = roi[0], roi[1]
    col = int(rear_u - rx1)
    below_tyre = gap[int(floor.b[i_r]) - ry1 + 5: int(floor.b[i_r]) - ry1 + 60, col]
    assert below_tyre.max() < 0.2, "a dark slab under the raised tyre"


def test_gap_is_capped_and_mirrors_are_not_underbody(grey_plate, table):
    raw = side_car_alpha()
    raw[1000:1080, 3400:3560] = 1.0  # a mirror sticking out past the body, ~1.2 m above the floor
    car = clean_alpha(raw, edge_shrink=1)
    scene, probe, plan, pixels, alpha, roi, floor = _setup(car, "background_1", grey_plate, table)
    assert not floor.body[-5:].any(), "mirror columns treated as underbody"
    assert np.all(floor.height[floor.body] <= GAP_MAX_M * plan.px_per_m + 400)
    contact, gap, band = _silhouette_layers(floor, roi, plan.px_per_m)
    mirror_cols = (floor.u[~floor.body] - roi[0]).astype(int)
    mirror_cols = mirror_cols[(mirror_cols >= 0) & (mirror_cols < gap.shape[1])]
    outer = mirror_cols[len(mirror_cols) // 2:]  # away from the blur spill of the body next to it
    assert gap[:, outer].max() < 0.05 and contact[:, outer].max() < 0.05


def test_no_snapping_onto_bumper_ends(grey_plate, table):
    """A local lowest point at the very end of the outline is a bumper corner, not a tyre."""
    raw = side_car_alpha()
    raw[1550:1600, 400:440] = 1.0  # a low spoiler tip at the rear end
    car = clean_alpha(raw, edge_shrink=1)
    scene, probe, plan, pixels, alpha, roi, floor = _setup(car, "background_1", grey_plate, table)
    end_u = plan.box_out[0] + 0.03 * (plan.box_out[2] - plan.box_out[0])
    assert all(abs(c.u - end_u) > 20 for c in snap_contacts(plan, probe, floor) if c.snapped)


def test_reflection_only_below_the_car(setup, grey_plate):
    scene, probe, plan, pixels, alpha, roi, floor = setup
    rx1, ry1, rx2, ry2 = roi
    lin = srgb_to_linear(grey_plate[ry1:ry2, rx1:rx2])
    before = lin.copy()
    mask = np.ones(lin.shape[:2], np.float32)
    add_reflection(lin, roi, srgb_to_linear(pixels), alpha, plan, floor, scene.spec.reflection, mask, None, 0.28)
    changed = np.any(lin != before, axis=2)
    rows = np.flatnonzero(changed.any(axis=1)) + ry1
    assert rows.size and rows.min() >= plan.box_out[3] - 30  # only around/below the car's bottom
    assert np.array_equal(linear_to_srgb(srgb_to_linear(grey_plate)), grey_plate)
