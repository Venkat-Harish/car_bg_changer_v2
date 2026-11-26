"""Config and background-selection tests (no model needed).

Run from the background_changer folder:  .venv\\Scripts\\python -m pytest tests
"""

from pathlib import Path

import pytest

import change_background as cb
from bg_changer import load_cars, select_car


def make_backgrounds(folder: Path, *names: str) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")  # selection only looks at file names
    (folder / "notes.txt").write_text("not an image")
    return cb.list_backgrounds(folder)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_select_by_name_list_and_all(tmp_path):
    available = make_backgrounds(tmp_path / "bg", "background_1.webp", "background_2.webp", "Studio.JPG")
    assert set(available) == {"background_1", "background_2", "studio"}
    assert cb.select_backgrounds(["background_2"], available) == [available["background_2"]]
    assert cb.select_backgrounds(["STUDIO"], available) == [available["studio"]]  # any case
    assert cb.select_backgrounds(["background_1.webp"], available) == [available["background_1"]]  # with extension
    assert cb.select_backgrounds(["studio", "background_1"], available) == [available["studio"], available["background_1"]]
    assert cb.select_backgrounds(["all"], available) == list(available.values())


def test_unknown_background_lists_available(tmp_path):
    available = make_backgrounds(tmp_path / "bg", "background_1.webp", "background_2.webp")
    with pytest.raises(SystemExit) as exc:
        cb.select_backgrounds(["background_9"], available)
    msg = str(exc.value)
    assert "background_9" in msg and "background_1" in msg and "background_2" in msg


def test_config_paths_car_and_scene_overrides(tmp_path):
    path = write_config(tmp_path, """
images_folder = "photos"
backgrounds_folder = "bgs"
background = ["background_1", "background_3"]
output_folder = "out"
car = "Mercedes-Benz EQB 2024"
cars_file = "my_cars.json"
cut_frames = "skip"
debug_mode = true

[scene.Background_1]
size = 1.1
reflection = false

[scene.my_showroom]
anchor = [960, 760]
side_fill = 0.6
horizon_y = 520
floor = "glossy"
""")
    cfg = cb.load_config(path)
    assert cfg["images_folder"] == tmp_path / "photos"
    assert cfg["cars_file"] == tmp_path / "my_cars.json"
    assert cfg["background"] == ["background_1", "background_3"]
    assert (cfg["car"], cfg["cut_frames"], cfg["debug_mode"]) == ("Mercedes-Benz EQB 2024", "skip", True)
    assert cfg["scene"]["background_1"] == {"size": 1.1, "reflection": False}
    assert cfg["scene"]["my_showroom"]["anchor"] == [960, 760]


def test_scene_block_names_are_normalised_like_file_names(tmp_path):
    cfg = cb.load_config(write_config(tmp_path, '[scene."My-Showroom 2"]\nside_fill = 0.6\n'))
    assert "my_showroom_2" in cfg["scene"]


@pytest.mark.parametrize("old", ['input = "x"', 'output = "x"', '[[backgrounds]]\npath = "x"',
                                 '[layout.default]\nground = 0.8', 'debug_overlay = true'])
def test_old_keys_say_what_replaced_them(tmp_path, old):
    with pytest.raises(SystemExit) as exc:
        cb.load_config(write_config(tmp_path, old + "\n"))
    assert "replaced" in str(exc.value)


@pytest.mark.parametrize("bad", ['cut_frames = "maybe"', '[scene.x]\nanchor = [1]', '[scene.x]\nsize = 9',
                                 '[scene.x]\nfloor = "wood"', '[scene.x]\ncolour = 1', 'debug_mode = "yes"',
                                 'background_scale = "big"', 'background_scale = 0.5', 'background_min_width = 10',
                                 'background_min_width = 640.5', 'background_min_width = true'])
def test_bad_values_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        cb.load_config(write_config(tmp_path, bad + "\n"))


def test_background_scale_settings(tmp_path):
    cfg = cb.load_config(write_config(tmp_path, 'background_scale = "photo"\nbackground_min_width = 800\n'))
    assert (cfg["background_scale"], cfg["background_min_width"]) == ("photo", 800)


def test_shipped_config_is_valid():
    cfg = cb.load_config(cb.DEFAULT_CONFIG)
    available = cb.list_backgrounds(cfg["backgrounds_folder"])
    assert {"background_1", "background_2", "background_3"} <= set(available)
    assert cb.select_backgrounds(cfg["background"], available)
    assert select_car(load_cars(cfg.get("cars_file", cb.DEFAULT_CARS)), cfg["car"]).length > 0
    assert cfg.get("background_scale", "folder") in ("folder", "photo", "off")


def test_shipped_cars_include_the_baleno():
    cars = load_cars(cb.DEFAULT_CARS)
    baleno = select_car(cars, "maruti suzuki baleno 2022")
    assert (baleno.length, baleno.width, baleno.height, baleno.wheelbase) == pytest.approx((3.99, 1.745, 1.50, 2.52))


def test_room_mode_settings(tmp_path):
    cfg = cb.load_config(write_config(tmp_path, """
background_view = "room"
max_output_size = 1280
keep_background_sharp = false
cut_frames = "frame"

[fit]
focal = 366.5
on_fail = "skip"
"""))
    assert (cfg["background_view"], cfg["max_output_size"], cfg["keep_background_sharp"]) == ("room", 1280, False)
    assert cfg["cut_frames"] == "frame" and cfg["fit"] == {"focal": 366.5, "on_fail": "skip"}
    assert cb.load_config(write_config(tmp_path, 'max_output_size = 0\n[fit]\nfocal = "auto"\n'))["fit"] == {"focal": "auto"}
    assert cb.load_config(write_config(tmp_path, 'background_view = "flat"\n'))["fit"] == {}  # [fit] is optional


@pytest.mark.parametrize("bad", ['background_view = "3d"', 'background_view = 1', 'max_output_size = 10',
                                 'max_output_size = 1920.0', 'max_output_size = true', 'keep_background_sharp = 1',
                                 'fit = 3', '[fit]\nfocal = -5', '[fit]\nfocal = "long"', '[fit]\nfocal = true',
                                 '[fit]\non_fail = "retry"', '[fit]\nmax_seconds = 5'])
def test_bad_room_values_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        cb.load_config(write_config(tmp_path, bad + "\n"))


def test_shipped_config_uses_room_mode():
    cfg = cb.load_config(cb.DEFAULT_CONFIG)
    assert cfg["background_view"] == "room" and cfg["cut_frames"] == "frame"
    assert cfg["max_output_size"] == cb.DEFAULT_MAX_OUTPUT and cfg["keep_background_sharp"] is True
    assert cfg["fit"] == {"focal": "auto", "on_fail": "place"}


def test_hosted_block_defaults_and_values(tmp_path):
    cfg = cb.load_config(write_config(tmp_path, 'background_view = "room"\n'))
    assert cfg["hosted"] == cb.HOSTED_DEFAULTS and cfg["hosted"]["enabled"] is False  # off unless asked for
    cfg = cb.load_config(write_config(tmp_path, """
[hosted]
enabled = true
glass_tint = false
tint_keep = 0.3
max_cost_per_car = 0
max_side = 0
blur_people = false
keep_cache_days = 0
"""))
    h = cfg["hosted"]
    assert (h["enabled"], h["glass_tint"], h["tint_keep"], h["max_cost_per_car"], h["max_side"]) == (True, False, 0.3, 0, 0)
    assert h["cutout"] is True and h["timeout_s"] == 120  # defaults filled in
    assert h["blur_people"] is False and h["keep_cache_days"] == 0 and h["require_masked_plates"] is True


@pytest.mark.parametrize("bad", ['hosted = 1', '[hosted]\nenabled = "yes"', '[hosted]\nvggt = 1',
                                 '[hosted]\ntint_strength = 1.5', '[hosted]\ntint_keep = -0.1',
                                 '[hosted]\nmax_cost_per_car = -1', '[hosted]\ntimeout_s = 0', '[hosted]\nmax_side = 100',
                                 '[hosted]\nmax_side = 1024.0', '[hosted]\napi_key = "x"',
                                 '[hosted]\nblur_people = 1', '[hosted]\nrequire_masked_plates = "no"',
                                 '[hosted]\nkeep_cache_days = -1', '[hosted]\nkeep_cache_days = true'])
def test_bad_hosted_values_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        cb.load_config(write_config(tmp_path, bad + "\n"))


def test_shipped_config_turns_hosted_on_and_the_baseline_off():
    assert cb.load_config(cb.DEFAULT_CONFIG)["hosted"] == {**cb.HOSTED_DEFAULTS, "enabled": True}
    assert cb.load_config(Path(__file__).parent / "baseline_flat.toml")["hosted"]["enabled"] is False
