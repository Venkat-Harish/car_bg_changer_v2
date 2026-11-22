"""The car proxy seen through the capture camera V reproduces the measured reference values."""

import numpy as np
import pytest

from bg_changer import CarShape, CarTable

P, AX, AY = 231.0, 957.5, 764.0  # reference scale and anchor of the probe's lookup table


@pytest.mark.parametrize("yaw, roof, low", [(0, 347.9, 923.9), (90, 338.7, 835.5), (180, 264.2, 900.8),
                                            (45, 338.6, 928.1), (135, 281.6, 928.1)])
def test_rows_match_reference_table(table, yaw, roof, low):
    r = table.at(yaw)
    assert AY + P * r.top == pytest.approx(roof, abs=1.0)
    assert AY + P * r.low == pytest.approx(low, abs=1.0)


def test_heights_and_ratios(table):
    h0, h90, h180 = table.at(0).h, table.at(90).h, table.at(180).h
    assert (h0, h90, h180) == pytest.approx((2.494, 2.151, 2.756), abs=0.005)
    assert 1.08 <= h0 / h90 <= 1.20  # front views taller than side views...
    assert 1.20 <= h180 / h90 <= 1.35  # ...and rear views taller still (upright tailgate nearer the camera)
    assert table.side_length() == pytest.approx(5.786, abs=0.01)


def test_mirror_symmetry(table):
    for yaw in (10, 37.5, 80, 123, 170):
        a, b = table.at(yaw), table.at(360 - yaw)
        assert a.h == pytest.approx(b.h, abs=1e-6)
        assert a.low == pytest.approx(b.low, abs=1e-6)
        assert a.cx == pytest.approx(-b.cx, abs=1e-6)


def test_floor_mapping_is_consistent(table):
    hom = table.floor_homography(30.0, (AX, AY), P)
    q = hom @ np.array([0.0, 0.0, 1.0])
    assert (q[0] / q[2], q[1] / q[2]) == pytest.approx((AX, AY), abs=0.01)
    s = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    rows = table.floor_row(s, AY, P)
    assert np.allclose(table.floor_depth(rows, AY, P), s, atol=1e-9)
    assert np.all(np.diff(rows) < 0)  # farther = higher in the picture


def test_other_cars_stretch_with_their_dimensions():
    long = CarTable(CarShape(length=5.15, width=1.95, height=1.80, wheelbase=3.10, name="van"))
    base = CarTable()
    ratio = long.side_length() / base.side_length()
    assert ratio == pytest.approx(5.15 / 4.684, rel=0.03)
    assert long.at(0).h > base.at(0).h


def test_contacts_visibility(table):
    assert list(table.at(90).contacts_visible) == [False, True, False, True]  # right side: right wheels
    assert list(table.at(0).contacts_visible) == [True, True, False, False]  # front: front wheels
