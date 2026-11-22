"""change_background.py in room mode, without the segmentation model: a synthetic photo (noise with a car
proxy's silhouette as its cut-out) goes through stages 2-3 onto the three measured rooms, a generic room
(a user background with horizon_y) and a background with no room (placed flat with the fitted view angle).

Run from the background_changer folder:  .venv\\Scripts\\python -m pytest tests
"""

import csv
import io
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import car_fit_to_bg.batch as batch
import change_background as cb
from background_scaler import BackgroundScaler
from bg_changer import CarMask, CarTable, load_scene
from bg_changer.carmodel import CarShape
from bg_changer.scenes import generic
from car_fit_to_bg.level import downscale
from car_fit_to_bg.lines import LineCamera
from car_fit_to_bg.pose import body_points, car_to_camera, render_proxy

ROOT = Path(__file__).resolve().parents[1]
SHAPE = CarShape(4.2, 1.8, 1.5, 2.6, "Test hatch", body="hatch")
TRUE_YAW, F = 75.0, 600.0
NAME = "0010.0s_right_90pct.png"  # label "right" (90 deg): 15 deg from the true view, so a label fallback shows
MEASURED = ("background_1", "background_2", "background_3")
MAX_OUT = 640  # below the photo's 960 px: the output is downscaled, so INTER_AREA really runs
# results.csv columns before room mode existed (flat mode must keep writing exactly these)
FLAT_COLUMNS = ["file", "background", "scene", "out_w", "out_h", "bg_scale", "view_label", "view_conf", "yaw",
                "yaw_source", "scale", "px_per_m", "anchor_x", "anchor_y", "src_h", "target_h", "target_low_y",
                "car_x1", "car_y1", "car_x2", "car_y2", "out_x1", "out_y1", "out_x2", "out_y2", "hw_ratio", "margin",
                "cut_off", "flags", "segment_s", "note"]


def synthetic_photo(size=(960, 640), yaw=TRUE_YAW, D=5.0, h=1.5, aim=(470.0, 420.0)):
    W, H = size
    c = ((W - 1) / 2, (H - 1) / 2)
    R, t = car_to_camera(yaw, D, math.degrees(math.asin(h / D)), aim, 0.0, F, c)
    alpha = render_proxy(body_points(SHAPE, SHAPE.body) @ R.T + t, F, c, (W, H)).astype(np.float32) / 255.0
    frame = np.random.default_rng(3).integers(0, 256, (H, W, 3), dtype=np.uint8)
    return frame, alpha


def make_photo(folder: Path, name: str, frame: np.ndarray, alpha: np.ndarray, cut=(), measure=True) -> cb.Photo:
    path = folder / name
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)  # PNG: re-read exactly in stage 3
    ys, xs = np.nonzero(alpha >= 0.5)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    ys, xs = np.nonzero(alpha > 0)
    crop = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    car = CarMask(alpha=alpha, box=box, crop=crop, cut={s: s in cut for s in ("left", "right", "top", "bottom")})
    photo = cb.prepare_photo(path, frame, car, CarTable(SHAPE), 0.0)
    photo.frame_size = (frame.shape[1], frame.shape[0])
    photo.measured = (cb.measure_photo(path, frame, alpha) if measure else
                      cb.Measured(path=path, stem=path.stem, size=photo.frame_size, cam=LineCamera(), exif_f=None,
                                  time_s=None))
    return photo


def make_run(out_dir: Path, overrides: dict, quality: int = 101) -> cb.Run:
    table = CarTable(SHAPE)
    loaded = []
    for name in MEASURED:
        plate = cv2.imread(str(ROOT / "backgrounds" / f"{name}.webp"), cv2.IMREAD_COLOR)
        loaded.append((name, plate, load_scene(name, plate, table)))
    rng = np.random.default_rng(5)
    for name in ("my_hall", "my_lot"):  # user backgrounds: a grey wall over a speckled floor
        plate = np.full((720, 1280, 3), 170, np.uint8)
        plate[330:] = rng.integers(90, 140, (390, 1280, 3), dtype=np.uint8)
        loaded.append((name, plate, load_scene(name, plate, table, overrides.get(name))))
    for name, _, _ in loaded:
        (out_dir / name / "debug").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    return cb.Run(loaded, BackgroundScaler("folder", 640), "frame", 1.0, out_dir, quality, True)


def room_settings(overrides: dict, cut_policy: str = "frame", on_fail: str = "place") -> cb.RoomRun:
    return cb.RoomRun(SHAPE, F, on_fail, MAX_OUT, 1.0, cut_policy, None, overrides)


@pytest.fixture(scope="module")
def room_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("room_mode")
    overrides = {"my_hall": {"horizon_y": 330.0}}
    frame, alpha = synthetic_photo()
    photo = make_photo(tmp / "photos", NAME, frame, alpha)
    run = make_run(tmp / "out", overrides)
    captured = []
    real = cb.place_in_room

    def spy(fit, name, *args, **kwargs):
        placed = real(fit, name, *args, **kwargs)
        captured.append((fit, name, placed))
        return placed

    rows: list[dict] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cb, "place_in_room", spy)
        cb.place_room_photos([photo], run, room_settings(overrides), rows)
    return run, rows, captured, frame


def test_every_background_gets_the_photo(room_run):
    run, rows, _, _ = room_run
    by_bg = {r.get("background", ""): r for r in rows}
    assert not [r["note"] for r in rows if "error" in r.get("note", "")]
    assert set(by_bg) == {*MEASURED, "my_hall", "my_lot"}
    for name in (*MEASURED, "my_hall"):
        r = by_bg[name]
        assert r["view_mode"] == "room" and (r["frame_w"], r["frame_h"]) == (960, 640)
        assert r["yaw_source"] == "fit" and r["fit_iou"] >= 0.9  # the 3D fit itself, not the label's angle
        assert abs((r["yaw"] - TRUE_YAW + 180) % 360 - 180) <= 6.0
        assert r["focal_source"] == "config" and r["focal_px"] == F
        assert r["out_scale"] < 1 and max(r["out_w"], r["out_h"]) <= MAX_OUT + 4
        assert r["out_w"] * 640 == pytest.approx(r["out_h"] * 960, abs=2 * 960)  # the photo's own framing
        assert (run.out_dir / name / f"{Path(NAME).stem}.webp").is_file()
        steps = run.out_dir / "debug" / Path(NAME).stem / name  # debug_mode: a picture per room step
        for pic in ("12_room", "13_room_surfaces", "14_shadows", "15_background_ready", "16_placed"):
            assert (steps / f"{pic}.jpg").is_file(), pic
    assert "generic-room" in by_bg["my_hall"]["flags"].split(";")
    assert "generic-room" not in by_bg["background_1"]["flags"].split(";")
    lot = by_bg["my_lot"]  # no room: the flat placement, with the fitted view angle
    assert lot["view_mode"] == "flat (no room)" and lot["yaw_source"] == "fit"
    assert "no-room" in lot["flags"].split(";") and lot["yaw"] == by_bg["background_1"]["yaw"]
    assert lot["fit_iou"] == by_bg["background_1"]["fit_iou"] and "set horizon_y" in lot["note"]
    assert (run.out_dir / "my_lot" / f"{Path(NAME).stem}.webp").is_file()
    photo_dbg = run.out_dir / "debug" / Path(NAME).stem
    for pic in ("06_camera_lines", "07_angle_search", "08_outline_fit", "09_outline_compare", "10_level", "11_tyres",
                "my_lot/12_flat_placement"):
        assert (photo_dbg / f"{pic}.jpg").is_file(), pic


def test_rows_fit_the_results_csv_columns(room_run):
    _, rows, _, _ = room_run
    writer = csv.DictWriter(io.StringIO(), fieldnames=cb.ROOM_CSV_FIELDS, restval="")
    writer.writerows(rows)  # raises on a key that is not a column
    assert cb.CSV_FIELDS == FLAT_COLUMNS  # flat mode's results.csv is unchanged
    assert cb.ROOM_CSV_FIELDS[:len(FLAT_COLUMNS)] == FLAT_COLUMNS


def test_written_images_carry_the_photos_own_pixels(room_run):
    """Every opaque car pixel in each written image is INTER_AREA of the photo's own pixels (the photo is not
    rolled, so the car is not rotated)."""
    run, _, captured, frame = room_run
    assert {name for _, name, _ in captured} == {*MEASURED, "my_hall"}
    for fit, name, placed in captured:
        assert fit.theta == 0.0 and placed.s_out < 1
        x1, y1 = fit.car.origin
        h0, w0 = fit.car.alpha.shape
        assert np.array_equal(fit.car.pixels, frame[y1:y1 + h0, x1:x1 + w0])  # the photo's pixels, untouched
        written = cv2.imread(str(run.out_dir / name / f"{Path(NAME).stem}.webp"), cv2.IMREAD_COLOR)
        assert np.array_equal(written, placed.image)  # lossless WEBP in this test
        pixels, a_s, corner = downscale(fit.car, placed.s_out)
        x0, y0 = placed.window
        px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
        H, W = written.shape[:2]
        h, w = a_s.shape
        ya, xa, yb, xb = max(0, py), max(0, px), min(H, py + h), min(W, px + w)
        opaque = a_s[ya - py:yb - py, xa - px:xb - px] >= 1.0
        assert opaque.mean() > 0.3
        want = cv2.resize(frame[y1:y1 + h0, x1:x1 + w0], (w, h), interpolation=cv2.INTER_AREA)
        assert np.array_equal(written[ya:yb, xa:xb][opaque], want[ya - py:yb - py, xa - px:xb - px][opaque])


@pytest.mark.parametrize("policy", ["skip", "frame"])
def test_cut_off_cars_are_skipped_without_a_fit(tmp_path, monkeypatch, policy):
    def no_fit(*args, **kwargs):
        raise AssertionError("a skipped photo must not be fitted")

    monkeypatch.setattr(cb, "fit_measured", no_fit)
    frame, alpha = synthetic_photo()
    run = make_run(tmp_path / "out", {})
    folder = tmp_path / "photos"
    photos = [make_photo(folder, "0000.0s_front_90pct.png", frame, alpha, cut=("bottom",), measure=False),
              make_photo(folder, "0002.0s_front_90pct.png", frame, alpha, cut=("left", "top"), measure=False)]
    if policy == "skip":
        photos.append(make_photo(folder, "0004.0s_front_90pct.png", frame, alpha, cut=("left",), measure=False))
    rows: list[dict] = []
    cb.place_room_photos(photos, run, room_settings({}, cut_policy=policy), rows)
    notes = {r["file"]: r["note"] for r in rows}
    assert len(rows) == len(photos) and all(n.startswith("skipped: ") for n in notes.values())
    assert "bottom" in notes["0000.0s_front_90pct.png"]
    assert "2 sides" in notes["0002.0s_front_90pct.png"]
    if policy == "skip":
        assert "cut_frames = skip" in notes["0004.0s_front_90pct.png"]
    assert not [p for name, _, _ in run.loaded for p in (run.out_dir / name).glob("*.webp")]


def test_no_room_rows_carry_the_fit_flags(tmp_path):
    """A failed fit placed on a background without a room is still flagged, with the label's angle."""
    frame, alpha = synthetic_photo()
    run = make_run(tmp_path / "out", {})
    photo = make_photo(tmp_path / "photos", NAME, frame, alpha, measure=False)
    summary = {"yaw_deg": 90.0, "yaw_source": "label-fallback", "fit_iou": 0.61, "fit_margin": "", "yaw_change": 0.0,
               "flags": ["weak-lines", "fit-failed"]}
    stem, plate, scene = next(x for x in run.loaded if x[0] == "my_lot")
    rows: list[dict] = []
    cb.place_flat_fallback([(photo, summary)], stem, plate, scene, run, rows, "hint")
    (row,) = rows
    assert row["view_mode"] == "flat (no room)" and row["yaw"] == 90.0 and row["yaw_source"] == "label-fallback"
    assert row["flags"].split(";")[:3] == ["no-room", "weak-lines", "fit-failed"] and row["fit_iou"] == 0.61
    assert row["out_w"] and row["note"] == "hint"
    assert (run.out_dir / "my_lot" / f"{Path(NAME).stem}.webp").is_file()


def test_line_errors_leave_the_photo_fitted_without_lines(tmp_path, monkeypatch):
    frame, alpha = synthetic_photo()
    path = tmp_path / NAME

    def broken(*args, **kwargs):
        raise cv2.error("no segments")

    monkeypatch.setattr(cb, "measure", broken)
    m = cb.measure_photo(path, frame, alpha)  # stage 1
    assert m.cam.source == "none" and m.cam.roll is None and m.flags == ["lines-error"]
    m.focal, m.focal_source = F, "config"
    seen = {}

    def fake_fit(frame, alpha, cut, shape, focal, focal_source, cam, **kwargs):
        seen["cam"] = cam
        return SimpleNamespace(flags=list(cam.flags))

    monkeypatch.setattr(batch, "line_camera", broken)  # stage 3 fails the same way
    monkeypatch.setattr(batch, "fit_photo", fake_fit)
    fit = batch.fit_measured(m, frame, alpha, set(), SHAPE, [m], None)
    assert seen["cam"].source == "none" and fit.flags.count("lines-error") == 1


def test_no_room_hint_says_what_is_missing():
    spec = generic.make_spec("my_lot", 1280, 720, None)
    assert "set horizon_y" in cb.no_room_hint(spec)
    low = generic.make_spec("my_lot", 1280, 720, {"horizon_y": 515.0, "anchor": [640, 520]})
    assert "must be more than" in cb.no_room_hint(low) and "520" in cb.no_room_hint(low)


def test_rows_are_sorted_by_photo_then_background():
    images = [Path("b.jpg"), Path("a.jpg")]
    loaded = [("background_2", None, None), ("background_1", None, None)]
    rows = [{"file": "a.jpg", "background": "background_1"}, {"file": "a.jpg", "note": "skipped"},
            {"file": "b.jpg", "background": "background_1"}, {"file": "b.jpg", "background": "background_2"}]
    cb.file_order(rows, images, loaded)
    assert [(r["file"], r.get("background", "")) for r in rows] == [
        ("b.jpg", "background_2"), ("b.jpg", "background_1"), ("a.jpg", ""), ("a.jpg", "background_1")]
