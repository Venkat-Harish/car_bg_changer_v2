"""cars.json: a JSON array of cars, dimensions in centimetres."""

import json
from pathlib import Path

import pytest

from bg_changer import CarSpecError, load_cars, parse_cars, select_car

ROOT = Path(__file__).resolve().parents[1]
EQB = {"carModel": "Mercedes-Benz EQB 2024", "length": 468.4, "width": 183.4, "height": 166.7, "WheelBase": 282.9}


def test_parse_converts_cm_to_m_and_selects_ignoring_case():
    cars = parse_cars([EQB, {**EQB, "carModel": "Small Hatch", "length": 400.0, "WheelBase": 250.0}])
    eqb = select_car(cars, "  mercedes-benz eqb 2024 ")
    assert (eqb.length, eqb.width, eqb.height, eqb.wheelbase) == pytest.approx((4.684, 1.834, 1.667, 2.829))
    assert select_car(cars, "SMALL HATCH").length == pytest.approx(4.0)


def test_unknown_model_lists_available():
    cars = parse_cars([EQB])
    with pytest.raises(CarSpecError) as exc:
        select_car(cars, "Tesla Model Y")
    assert "Mercedes-Benz EQB 2024" in str(exc.value)


@pytest.mark.parametrize("bad, words", [
    ({k: v for k, v in EQB.items() if k != "WheelBase"}, "WheelBase"),
    ({**EQB, "height": -1}, "positive"),
    ({**EQB, "length": 4684, "width": 1834, "height": 1667, "WheelBase": 2829}, "millimetres"),
    ({**EQB, "length": 4.684, "width": 1.834, "height": 1.667, "WheelBase": 2.829}, "metres"),
    ({**EQB, "WheelBase": 500}, "shorter"),
    ({**EQB, "carModel": ""}, "carModel"),
])
def test_bad_entries_are_rejected(bad, words):
    with pytest.raises(CarSpecError) as exc:
        parse_cars([bad])
    assert words in str(exc.value)


def test_other_capitalisation_of_keys_is_accepted():
    entry = {"carModel": "x", "Length": 450, "WIDTH": 180, "height": 150, "wheelBase": 270}
    assert parse_cars([entry])["x"].wheelbase == pytest.approx(2.7)


def test_not_an_array_or_duplicate():
    with pytest.raises(CarSpecError):
        parse_cars(EQB)
    with pytest.raises(CarSpecError):
        parse_cars([EQB, EQB])


def test_shipped_cars_json(tmp_path):
    cars = load_cars(ROOT / "cars.json")
    assert select_car(cars, "Mercedes-Benz EQB 2024").length == pytest.approx(4.684)
    bad = tmp_path / "cars.json"
    bad.write_text("[{", encoding="utf-8")
    with pytest.raises(CarSpecError):
        load_cars(bad)
    utf16 = tmp_path / "cars16.json"
    utf16.write_text(json.dumps([EQB]), encoding="utf-16")  # e.g. saved by PowerShell 5.1 '>'
    with pytest.raises(CarSpecError):
        load_cars(utf16)
    assert json.loads((ROOT / "cars.json").read_text(encoding="utf-8"))[0]["carModel"]
