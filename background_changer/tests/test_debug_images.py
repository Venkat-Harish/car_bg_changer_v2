"""debug_mode: the step pictures, and the hooks that feed them changing nothing.

Run from the background_changer folder:  .venv\\Scripts\\python -m pytest tests
"""

from types import SimpleNamespace

import cv2
import numpy as np

from bg_changer.segment import clean_alpha
from car_fit_to_bg.lines import line_camera
from debug_images import DebugWriter, cutout_pictures, glass_pictures, hosted_camera_picture


def raw_mask():
    raw = np.zeros((240, 320), np.float32)
    cv2.rectangle(raw, (60, 80), (260, 190), 1.0, -1)  # the car
    cv2.rectangle(raw, (120, 100), (160, 130), 0.0, -1)  # a window the model saw through
    cv2.rectangle(raw, (290, 20), (305, 60), 1.0, -1)  # something else in the photo
    return cv2.GaussianBlur(raw, (0, 0), 1.5)


def test_trace_changes_nothing_and_the_cutout_pictures_are_written(tmp_path):
    raw = raw_mask()
    plain = clean_alpha(raw, 2)
    trace = {}
    traced = clean_alpha(raw, 2, trace=trace)
    assert np.array_equal(plain.alpha, traced.alpha) and plain.box == traced.box and plain.cut == traced.cut
    assert {"binary", "posts", "car", "solid", "core"} <= set(trace)
    assert trace["solid"][115, 140] and not trace["binary"][115, 140]  # the window hole was filled
    image = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    dw = DebugWriter(tmp_path / "debug")
    dw.write_readme()
    cutout_pictures(dw, "p", image, raw, trace, traced, "test-model")
    for name in ("01_photo", "02_cutout_model", "03_cutout_cleanup", "05_cutout_result"):
        pic = cv2.imread(str(tmp_path / "debug" / "p" / f"{name}.jpg"))
        assert pic is not None and pic.shape[1] >= 320, name
    assert not (tmp_path / "debug" / "p" / "04_cutout_isolation.jpg").exists()  # car_isolation was off
    assert "01_photo" in (tmp_path / "debug" / "README.txt").read_text(encoding="utf-8")


def test_no_car_still_gets_the_photo_and_the_model_mask(tmp_path):
    image = np.zeros((120, 160, 3), np.uint8)
    cutout_pictures(DebugWriter(tmp_path), "p", image, np.zeros((120, 160), np.float32), {}, None, "test-model")
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == ["01_photo.jpg", "02_cutout_model.jpg"]


def test_glass_pictures_only_read_and_are_written(tmp_path):
    rng = np.random.default_rng(1)
    before = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    car = np.zeros((240, 320), np.float32)
    car[80:190, 60:260] = 1.0
    window = np.zeros((240, 320), bool)
    window[95:130, 110:200] = True
    after = before.copy()
    after[window] = 20
    copies = [x.copy() for x in (before, after, window, car)]
    glass_pictures(DebugWriter(tmp_path), "p", before, after, window, car, 14.3)
    for a, b in zip((before, after, window, car), copies):
        assert np.array_equal(a, b)
    for name in ("05a_glass_mask", "05b_glass_tint"):
        assert cv2.imread(str(tmp_path / "p" / f"{name}.jpg")) is not None, name
    glass_pictures(DebugWriter(tmp_path / "none"), "p", before, before, None, car)  # glass mask failed: nothing
    assert not (tmp_path / "none").exists()


def test_hosted_camera_picture_with_and_without_a_hosted_camera(tmp_path):
    from hosted.camera import HostedCamera

    frame = np.random.default_rng(2).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    alpha = np.zeros((240, 320), np.float32)
    alpha[100:200, 60:260] = 1.0
    fit = SimpleNamespace(f=300.0, focal_source="hosted", pose=SimpleNamespace(h=1.4),
                          cam=SimpleNamespace(roll=0.5, pitch=3.0, source="hosted", flags=["camera-hosted"]))
    ext = HostedCamera(f_px=300.0, pitch=3.0, roll=0.5, height_m=1.4, rel_yaw_deg=90.0, conf=0.8)
    keep = frame.copy()
    hosted_camera_picture(DebugWriter(tmp_path), "a", frame, alpha, ext, fit, {"moge_f": 310.0, "vggt_ok": "yes"})
    hosted_camera_picture(DebugWriter(tmp_path), "b", frame, alpha, None, fit)
    assert np.array_equal(frame, keep)
    for stem in "ab":
        assert cv2.imread(str(tmp_path / stem / "05c_hosted_camera.jpg")) is not None


def test_line_detail_changes_nothing():
    img = np.full((480, 640, 3), 90, np.uint8)
    for x in range(40, 640, 70):  # walls' vertical edges, slightly converging
        cv2.line(img, (x, 20), (x + (x - 320) // 25, 300), (230, 230, 230), 2)
    for y in range(330, 480, 30):  # floor lines toward a vanishing point on the right
        cv2.line(img, (0, y), (640, y - (y - 300) // 3), (20, 20, 20), 2)
    car = np.zeros((480, 640), np.float32)
    car[360:440, 250:420] = 1.0
    detail = {}
    a, b = line_camera(img, car), line_camera(img, car, detail=detail)
    assert a == b
    assert len(detail["segments"]) == len(detail["labels"]) > 0
