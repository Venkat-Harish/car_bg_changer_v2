"""Car dimensions from cars.json.

cars.json is a JSON array with one object per car, dimensions in centimetres:

    [
      {"carModel": "Mercedes-Benz EQB 2024", "length": 468.4, "width": 183.4,
       "height": 166.7, "WheelBase": 282.9}
    ]

Optional "body": "suv" | "hatch" | "sedan" picks the proxy's roof line for the 3D fit (room mode); without
it the body is guessed from the dimensions. Width is the body width without mirrors. Every car fills a background's stage the same way (a side
view spans the same pixels); the dimensions set the car's proportions and, through the capture
camera's fixed distance, how strongly its near end comes forward in front and rear views.
"""

from __future__ import annotations

import json
from pathlib import Path

from .carmodel import CarShape

DIMENSION_KEYS = ("length", "width", "height", "WheelBase")
BODY_TYPES = ("suv", "hatch", "sedan")


class CarSpecError(ValueError):
    pass


def _number(entry: dict, key: str, where: str) -> float:
    # Accept the documented key, but tolerate a different capitalisation (e.g. "wheelBase").
    matches = [k for k in entry if k.lower() == key.lower()]
    if not matches:
        raise CarSpecError(f"{where}: missing '{key}' (cm)")
    value = entry[matches[0]]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise CarSpecError(f"{where}: '{key}' must be a positive number of centimetres, got {value!r}")
    return float(value)


def parse_cars(data, source: str = "cars.json") -> dict[str, CarShape]:
    """JSON value (a list of car objects) -> {lower-case carModel: CarShape in metres}."""
    if not isinstance(data, list) or not data:
        raise CarSpecError(f"{source} must be a JSON array with at least one car object")
    cars: dict[str, CarShape] = {}
    for i, entry in enumerate(data, 1):
        where = f"{source}, car {i}"
        if not isinstance(entry, dict):
            raise CarSpecError(f"{where}: each car must be a JSON object")
        name = entry.get("carModel")
        if not isinstance(name, str) or not name.strip():
            raise CarSpecError(f"{where}: 'carModel' must be a non-empty text")
        name = name.strip()
        where = f"{source}, '{name}'"
        length, width, height, wheelbase = (_number(entry, k, where) for k in DIMENSION_KEYS)
        if max(length, width, height) > 2000:
            raise CarSpecError(f"{where}: dimensions look like millimetres; cars.json uses centimetres")
        if max(length, width, height) < 20:
            raise CarSpecError(f"{where}: dimensions look like metres; cars.json uses centimetres")
        if wheelbase >= length:
            raise CarSpecError(f"{where}: 'WheelBase' ({wheelbase} cm) must be shorter than 'length' ({length} cm)")
        body = entry.get("body")
        if body is not None and (not isinstance(body, str) or body.strip().lower() not in BODY_TYPES):
            raise CarSpecError(f"{where}: 'body' must be one of {', '.join(BODY_TYPES)}, got {body!r}")
        key = name.lower()
        if key in cars:
            raise CarSpecError(f"{source}: '{name}' is listed twice")
        cars[key] = CarShape(length=length / 100, width=width / 100, height=height / 100,
                             wheelbase=wheelbase / 100, name=name, body=body.strip().lower() if body else None)
    return cars


def load_cars(path: Path | str) -> dict[str, CarShape]:
    path = Path(path)
    if not path.is_file():
        raise CarSpecError(f"car file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CarSpecError(f"cannot read {path}: {exc}. Save it as UTF-8 JSON.") from None
    return parse_cars(data, str(path))


def select_car(cars: dict[str, CarShape], model: str) -> CarShape:
    """Pick a car by carModel, ignoring case and surrounding spaces."""
    key = model.strip().lower()
    if key not in cars:
        names = ", ".join(c.name for c in cars.values())
        raise CarSpecError(f"car '{model}' is not in the car file. Available: {names}")
    return cars[key]
