"""Measuring the cut-out and reading the view label."""

import numpy as np

from bg_changer import clean_alpha, parse_name, probe_car
from bg_changer.yaw import BIN_DEG, estimate_yaw

from conftest import side_car_alpha


def test_parse_name():
    info = parse_name("0012.0s_right_95pct")
    assert (info.time_s, info.label, info.conf) == (12.0, "right", 95)
    assert parse_name("0006.0s_front-right_93pct").label == "front-right"
    assert parse_name("0032.0s_left_47pct").conf == 47
    assert parse_name("car_photo") is None
    assert parse_name("0012.0s_upside_95pct") is None


def test_probe_side_view_two_tyres_tie():
    car = clean_alpha(side_car_alpha(), edge_shrink=1)
    p = probe_car(car)
    assert p.tie, "two equally low tyres far apart"
    assert abs(p.lowest_y - 1699) <= 3
    xs = [x for x, _ in p.contacts]
    assert any(abs(x - 900) < 60 for x in xs) and any(abs(x - 2900) < 60 for x in xs), xs
    assert p.cut == ()


def test_probe_one_tyre_lower_and_cut():
    raw = side_car_alpha(tyres=((900, 1650), (2900, 1720)))
    raw[700:1500, 3400:] = 1.0  # body runs off the right edge
    car = clean_alpha(raw, edge_shrink=1)
    p = probe_car(car)
    assert not p.tie
    assert p.lowest_side > 0  # the lower tyre is right of centre
    assert p.cut == ("right",)


def test_yaw_from_label_and_fallback(table):
    car = clean_alpha(side_car_alpha(), edge_shrink=1)
    p = probe_car(car)
    y = estimate_yaw(parse_name("0020.0s_back_91pct"), p, table)
    assert (y.deg, y.source) == (180.0, "label")
    low = estimate_yaw(parse_name("0032.0s_left_47pct"), p, table)
    assert "low-conf-label" in low.flags and low.sigma > y.sigma
    none = estimate_yaw(None, p, table)
    assert none.source == "shape" and "no-view-label" in none.flags and none.deg in BIN_DEG.values()
    assert none.deg in (90.0, 270.0)  # a long low silhouette reads as a side view
    assert estimate_yaw(None, p, table, override=400).deg == 40.0
