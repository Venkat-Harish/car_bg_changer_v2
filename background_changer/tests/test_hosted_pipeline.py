"""hosted.pipeline.HostedRun stage 1 with a fake fal (no network, no model): cut-out, SAM 3 isolation, glass tint,
originals, fallbacks."""

import cv2
import numpy as np
import pytest

import hosted.pipeline as hp
from hosted.fal import FalError
from hosted_fakes import FakeFal

H, W = 400, 800
SETTINGS = {"enabled": True, "cutout": True, "glass_tint": True, "camera": False, "vggt": False,
            "tint_strength": 0.85, "tint_keep": 0.15, "max_cost_per_car": 0, "timeout_s": 10, "max_side": 0}


def scene():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    car = np.zeros((H, W), np.uint8)
    cv2.rectangle(car, (150, 150), (650, 330), 255, -1)
    win = np.zeros((H, W), np.uint8)
    cv2.rectangle(win, (300, 170), (500, 220), 255, -1)
    return img, car, win


def fake(car, win, birefnet_fails=False):
    f = FakeFal()
    f.add_file("mem://alpha", car)
    f.add_file("mem://car", car)
    f.add_file("mem://win", win)

    def birefnet(payload, images):
        if birefnet_fails:
            raise FalError("down")
        return {"image": {"url": "mem://alpha"}}

    def sam3(payload, images):
        p = payload["prompt"]
        if p == "car":
            return {"masks": [{"url": "mem://car"}], "scores": [0.95]}
        if p in ("car window", "windshield"):
            return {"masks": [{"url": "mem://win"}]}
        return {"masks": []}

    f.handlers = {"fal-ai/birefnet/v2": birefnet, "fal-ai/sam-3/image": sam3}
    return f


def run(tmp_path, client, **over):
    return hp.HostedRun({**SETTINGS, **over}, tmp_path / "out", tmp_path / "cache", tmp_path / ".env",
                        ("birefnet-general-lite", 1, 1), 1, client=client)


def test_stage1_cuts_out_tints_only_the_glass_and_keeps_the_original(tmp_path):
    img, car, win = scene()
    src = tmp_path / "in" / "0001.png"
    src.parent.mkdir()
    cv2.imencode(".png", img)[1].tofile(str(src))
    hr = run(tmp_path, fake(car, win))
    mask, raw = hr.cut_out(src, img)
    assert mask.alpha[240, 400] == 1.0 and mask.alpha[50, 50] == 0.0
    tinted, frame_path, wm = hr.glass(src, img, mask.alpha)
    assert wm.any() and not (wm & (car == 0)).any()
    changed = np.any(tinted != img, axis=2)
    assert changed.any() and not (changed & ~wm).any()  # car pixels change only inside the window mask
    assert np.array_equal(cv2.imdecode(np.fromfile(str(frame_path), np.uint8), cv2.IMREAD_COLOR), tinted)
    assert (tmp_path / "out" / "originals" / "0001.png").read_bytes() == src.read_bytes()
    cols = hr.cols("0001")
    assert cols["hosted"] == "fal" and cols["sam3_status"] == "ok" and 0 < cols["glass_pct"] < 50


def test_glass_off_changes_nothing(tmp_path):
    img, car, win = scene()
    hr = run(tmp_path, fake(car, win), glass_tint=False)
    mask, _ = hr.cut_out(tmp_path / "0001.png", img)
    out, frame_path, wm = hr.glass(tmp_path / "0001.png", img, mask.alpha)
    assert out is img and frame_path is None and wm is None


def test_fal_down_falls_back_to_the_local_model(tmp_path, monkeypatch):
    img, car, win = scene()

    class Local:
        def __init__(self, *a):
            pass

        def raw_alpha(self, image):
            return car.astype(np.float32) / 255

    monkeypatch.setattr(hp, "CarSegmenter", type("CS", (hp.CarSegmenter,), {"__init__": Local.__init__,
                                                                             "raw_alpha": Local.raw_alpha}))
    hr = run(tmp_path, fake(car, win, birefnet_fails=True))
    mask, _ = hr.cut_out(tmp_path / "0001.png", img)
    assert mask.alpha[240, 400] == 1.0
    assert hr.cols("0001")["hosted"] == "local" and "hosted-cutout-failed" in hr.flags("0001")


def test_missing_key_stops_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(SystemExit, match="FAL_KEY"):
        hp.HostedRun(SETTINGS, tmp_path / "out", tmp_path / "cache", tmp_path / "none.env", ("m", 1, 1), 1)


def test_hosted_results_csv_has_the_isolation_columns():
    """[hosted] isolates with SAM 3 while [car_isolation] is off: its rows carry isolation/isolate_s, so the
    header must too (a live run lost its results.csv to csv.DictWriter's 'fields not in fieldnames')."""
    import change_background as cb

    fields = cb.results_fields("room", isolate=False, hosted=True)
    assert set(cb.ISOLATION_CSV_FIELDS) <= set(fields) and set(cb.HOSTED_CSV_FIELDS) <= set(fields)
    assert cb.results_fields("flat", isolate=False, hosted=False) == cb.CSV_FIELDS  # flat CSV unchanged
    assert cb.results_fields("room", isolate=False, hosted=False) == cb.ROOM_CSV_FIELDS
