"""[hosted] camera in room mode: HostedRun.camera (MoGe-3 / VGGT stubbed) and stages 2-3 using Measured.external.
No network, no model."""

from types import SimpleNamespace

import numpy as np

import change_background as cb
import hosted.camera as hcam
import hosted.pipeline as hp
from hosted.camera import FrameCam, HostedCamera
from hosted_fakes import FakeFal
from test_room_mode import NAME, make_photo, make_run, room_settings, synthetic_photo

SETTINGS = {**cb.HOSTED_DEFAULTS, "enabled": True, "max_cost_per_car": 0}


def test_hosted_camera_fills_external_and_a_yaw_prior(tmp_path, monkeypatch):
    frame, alpha = synthetic_photo()
    names = ["0001.0s_front_90pct.png", "0002.0s_front-right_90pct.png", "0003.0s_right_90pct.png",
             "0004.0s_back-right_90pct.png"]
    photos = [make_photo(tmp_path / "photos", n, frame, alpha, measure=False) for n in names]
    for p in photos:
        p.hosted = {}
    monkeypatch.setattr(hcam, "moge_frame", lambda c, img, m, max_side=1024: FrameCam(
        f_px=600.0, pitch=9.0, roll=0.5, height_m=1.45, conf=0.9, source="moge"))
    rel = [0.0, 45.0, 90.0, 135.0]
    seen = {}

    def vggt(c, imgs, masks, max_side=518):
        seen["n"], seen["sizes"] = len(imgs), [i.shape for i in imgs]
        return [FrameCam(f_px=610.0 * i.shape[1] / 960, rel_yaw_deg=r, conf=0.9, source="vggt")
                for i, r in zip(imgs, rel)]

    monkeypatch.setattr(hcam, "vggt_car", vggt)
    hr = hp.HostedRun(SETTINGS, tmp_path / "out", tmp_path / "c", tmp_path / ".env", ("m", 1, 1), 1, client=FakeFal())
    prior = hr.camera(photos, cb.read_image, cb.full_frame_alpha)
    assert seen["n"] == 4 and max(seen["sizes"][0][:2]) <= 1036  # small copies go to VGGT
    for p in photos:
        ext = p.measured.external
        assert isinstance(ext, HostedCamera) and ext.height_m == 1.45 and abs(ext.f_px - 610.0) < 1.0
        assert p.hosted["moge_h"] == 1.45 and p.hosted["vggt_ok"] == "yes"
    # labels front/front-right/right/back-right = 0/45/90/135 deg: the prior peaks on each photo's own bin
    for p, lab in zip(photos, ["front", "front-right", "right", "back-right"]):
        probs = prior[p.path.stem]
        assert max(probs, key=probs.get) == lab
    items = [p.measured for p in photos]
    hr.apply_focals(items, None)
    assert all(m.focal_source == "hosted" and abs(m.focal - 610.0) < 1.0 for m in items)
    hr.apply_focals(items, 500.0)  # a [fit] focal from the config is never overridden
    assert all(m.focal_source == "hosted" for m in items)


def test_room_mode_uses_the_hosted_camera(tmp_path):
    frame, alpha = synthetic_photo()
    photo = make_photo(tmp_path / "photos", NAME, frame, alpha, measure=False)  # no lines: hosted is primary
    photo.hosted = {"hosted": "fal", "sam3_status": "ok", "glass_pct": 0.0, "camera_source": "", "moge_f": "",
                    "moge_h": "", "vggt_ok": "", "cost_usd": 0.0}
    run = make_run(tmp_path / "out", {})
    rr = room_settings({})

    def camera(photos, read_frame, frame_alpha):
        for p in photos:
            p.measured.external = HostedCamera(f_px=None, pitch=8.0, roll=0.0, height_m=1.5, conf=0.9,
                                               flags=["camera-moge"])
        return {}

    rr.hosted = SimpleNamespace(camera=camera, apply_focals=lambda items, f: None)
    rows = []
    cb.place_room_photos([photo], run, rr, rows)
    placed = [r for r in rows if r.get("out_w") and "roll_source" in r]  # room rows (my_lot is flat)
    assert placed, rows
    for r in placed:
        assert r["roll_source"] == "hosted" and r.get("camera_source") == "hosted"
        assert "camera-hosted" in r["flags"] and r["hosted"] == "fal"
        assert set(r) <= set(cb.ROOM_CSV_FIELDS + cb.HOSTED_CSV_FIELDS)
    assert np.isfinite(float(placed[0]["cam_height_m"]))
