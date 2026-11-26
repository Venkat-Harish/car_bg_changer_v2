"""hosted.camera: MoGe floor fit and VGGT relative yaw on synthetic point clouds / poses built with car_fit_to_bg's own
camera model (pose.car_to_camera, cam_angles), combine(), the yaw helpers, and the Measured.external hook in
batch.fit_measured / pose.fit_pose. FakeFal only: nothing goes to fal."""

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bg_changer.carmodel import CarShape
from car_fit_to_bg import batch, pose
from car_fit_to_bg.lines import LineCamera
from car_fit_to_bg.pose import body_points, cam_angles, car_to_camera, render_proxy
from hosted.camera import (MOGE_ENDPOINT, VGGT_ENDPOINT, FrameCam, HostedCamera, anchor_yaws, combine, moge_frame,
                           read_ply_points, vggt_car, yaw_probs)
from hosted_fakes import FakeFal

W, H = 1920, 1080
F = 1500.0
C = ((W - 1) / 2, (H - 1) / 2)
SHAPE = CarShape(4.2, 1.8, 1.5, 2.6, "Test hatch", body="hatch")


def _camera(yaw=60.0, D=6.0, h=1.45, roll=2.0, aim=(1000.0, 640.0), f=F):
    delta = math.degrees(math.asin(h / D))
    return car_to_camera(yaw, D, delta, aim, roll, f, C)


def _ply(points: np.ndarray) -> bytes:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\n"
            "property float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n").encode()
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.zeros(len(points), dt)
    v["x"], v["y"], v["z"] = points[:, 0], points[:, 1], points[:, 2]
    v["r"] = 200
    return head + v.tobytes()


def _scene_points(R, t, car_mask, seed=0):
    """Per-pixel (every 6 px) camera-frame points: the floor (car frame z = 0), a back wall, and the car region
    lifted off the floor (distractor). OpenCV axes."""
    rng = np.random.default_rng(seed)
    n = R[:, 2]  # floor normal (car z) in camera coordinates
    p0 = t  # the car's footprint centre lies on the floor
    us, vs = np.meshgrid(np.arange(0, W, 6), np.arange(0, H, 6))
    rays = np.stack([(us - C[0]) / F, (vs - C[1]) / F, np.ones_like(us, float)], -1).reshape(-1, 3)
    pix = np.stack([us.ravel(), vs.ravel()], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n @ p0) / (rays @ n)
    wall_n = R[:, 1]  # a wall 9 m behind the car along car +Y (left) ... any vertical plane will do
    wall_p = t + 9.0 * R[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = (wall_n @ wall_p) / (rays @ wall_n)
    s = np.where((s > 0) & ((sw <= 0) | (s < sw)), s, np.where(sw > 0, sw, np.nan))
    P = rays * s[:, None]
    inside = car_mask[pix[:, 1], pix[:, 0]] > 0
    P[inside] *= 0.85  # the car stands in front of the floor/wall behind it
    P = P[np.isfinite(P).all(1) & (P[:, 2] < 60)]
    return P + rng.normal(0, 0.003, P.shape)


def _car_mask(R, t):
    return render_proxy(body_points(SHAPE, SHAPE.body) @ R.T + t, F, C, (W, H))


def _moge_fake(points, opengl=True):
    fake = FakeFal()
    pts = points * np.array([1.0, -1.0, -1.0]) if opengl else points
    fake.add_file("mem://pc.ply", _ply(pts))
    fov_x = math.degrees(2 * math.atan((W / 2) / F))
    fov_y = math.degrees(2 * math.atan((H / 2) / F))
    fake.handlers[MOGE_ENDPOINT] = lambda payload, images: {
        "fov_x": fov_x, "fov_y": fov_y, "intrinsics": [[F / W, 0, 0.5], [0, F / H, 0.5], [0, 0, 1]],
        "point_cloud": {"url": "mem://pc.ply"}, "depth_map": {"url": "mem://d.png"}}
    return fake


@pytest.mark.parametrize("opengl", [True, False])
def test_moge_floor_recovers_pitch_roll_and_height(opengl):
    R, t = _camera()
    mask = _car_mask(R, t)
    fake = _moge_fake(_scene_points(R, t, mask), opengl)
    image = np.full((H, W, 3), 90, np.uint8)
    cam = moge_frame(fake, image, mask)
    pitch, roll = cam_angles(R)
    h = float((-R.T @ t)[2])
    assert cam is not None and cam.source == "moge"
    assert cam.f_px == pytest.approx(F, rel=1e-6)
    assert abs(cam.pitch - pitch) < 0.5 and abs(cam.roll - roll) < 0.5
    assert abs(cam.height_m - h) < 0.02
    assert cam.conf > 0.3
    assert fake.calls[0][0] == MOGE_ENDPOINT and fake.calls[0][1]["export_ply"] is True


def test_moge_signs_follow_the_project_conventions():
    """Looking down more gives a bigger pitch; the floor fit's roll has the sign of pose's roll."""
    for roll in (-4.0, 3.0):
        R, t = _camera(h=2.2, D=5.0, roll=roll)
        mask = _car_mask(R, t)
        cam = moge_frame(_moge_fake(_scene_points(R, t, mask)), np.zeros((H, W, 3), np.uint8), mask)
        p, r = cam_angles(R)
        assert p > 15 and cam.pitch == pytest.approx(p, abs=0.5)
        assert np.sign(cam.roll) == np.sign(r) and cam.roll == pytest.approx(r, abs=0.5)


def test_moge_is_deterministic_and_none_without_a_floor():
    R, t = _camera()
    mask = _car_mask(R, t)
    pts = _scene_points(R, t, mask)
    a = moge_frame(_moge_fake(pts), np.zeros((H, W, 3), np.uint8), mask)
    b = moge_frame(_moge_fake(pts), np.zeros((H, W, 3), np.uint8), mask)
    assert (a.pitch, a.roll, a.height_m, a.conf) == (b.pitch, b.roll, b.height_m, b.conf)
    # only a wall facing the camera: no plane passes the "up" test -> focal only, no floor
    wall = np.random.default_rng(0).uniform(-3, 3, (5000, 3)) * np.array([1, 1, 0]) + np.array([0, 0, 8.0])
    cam = moge_frame(_moge_fake(wall), np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.uint8))
    assert cam is not None and cam.f_px == pytest.approx(F) and cam.pitch is None and cam.height_m is None


def test_read_ply_ascii():
    data = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n" \
           b"end_header\n1 2 3\n4 5 6\n"
    assert np.array_equal(read_ply_points(data), [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------- VGGT
YAWS = [30.0, 75.0, 140.0, 200.0, 300.0, 340.0]


def _vggt_fake(yaws, aims, rolls, f=F, size=(W, H)):
    Ws, Hs = size
    c = ((Ws - 1) / 2, (Hs - 1) / 2)
    Rs, ts, masks = [], [], []
    for yaw, aim, roll in zip(yaws, aims, rolls):
        R, t = car_to_camera(yaw, 6.0, math.degrees(math.asin(1.4 / 6.0)), aim, roll, f, c)
        Rs.append(R)
        ts.append(t)
        q = R @ np.array([0.0, 0.0, 0.7]) + t  # a point above the footprint centre
        m = np.zeros((Hs, Ws), np.uint8)
        cv2.circle(m, (int(round(c[0] + f * q[0] / q[2])), int(round(c[1] + f * q[1] / q[2]))), 60, 255, -1)
        masks.append(m)
    R0, t0 = Rs[0], ts[0]
    ext = []
    for R, t in zip(Rs, ts):
        Rw = R @ R0.T
        ext.append(np.c_[Rw, 0.4 * (t - Rw @ t0)].tolist())  # world = frame 0's camera, arbitrary scale
    fs = f * 518 / max(Ws, Hs)
    K = [[fs, 0, 259.0], [0, fs, 259.0], [0, 0, 1]]
    fake = FakeFal({VGGT_ENDPOINT: lambda payload, images: {
        "num_frames": len(yaws), "depth_range": [0.1, 20], "extrinsics": ext, "intrinsics": [K] * len(yaws)}})
    return fake, masks


def test_vggt_relative_yaw_and_focal():
    aims = [(960, 560), (700, 600), (1300, 500), (960, 700), (500, 540), (1400, 650)]  # off-centre cars too
    fake, masks = _vggt_fake(YAWS, aims, [0.0] * len(YAWS))
    images = [np.full((H, W, 3), 200, np.uint8) for _ in YAWS]
    cams = vggt_car(fake, images, masks)
    for cam, yaw in zip(cams, YAWS):
        assert cam.source == "vggt" and cam.f_px == pytest.approx(F, rel=1e-6)
        want = (yaw - YAWS[0] + 180) % 360 - 180
        assert abs((cam.rel_yaw_deg - want + 180) % 360 - 180) < 0.5
        assert cam.conf > 0.7
    endpoint, payload, sent = fake.calls[0]
    assert endpoint == VGGT_ENDPOINT and len(sent["image_urls"]) == len(YAWS)
    assert payload["export_point_cloud"] is False and payload["export_prediction_data"] is True  # fal: 422 if all off
    car = masks[1] > 0
    assert (sent["image_urls"][1][car] == 128).all()  # the car is greyed out before sending
    assert (sent["image_urls"][1][~cv2.dilate(masks[1], np.ones((101, 101), np.uint8)).astype(bool)] == 200).all()


def test_vggt_small_rolls_still_give_the_yaw():
    rolls = [1.5, -1.0, 0.5, -1.5, 1.0, 0.0]
    fake, masks = _vggt_fake(YAWS, [(960, 560)] * len(YAWS), rolls)
    cams = vggt_car(fake, [np.zeros((H, W, 3), np.uint8)] * len(YAWS), masks)
    for cam, yaw in zip(cams, YAWS):
        want = (yaw - YAWS[0] + 180) % 360 - 180
        assert abs((cam.rel_yaw_deg - want + 180) % 360 - 180) < 1.5


def test_vggt_bad_answer_gives_none():
    fake = FakeFal({VGGT_ENDPOINT: lambda payload, images: {"extrinsics": [], "intrinsics": []}})
    img = np.zeros((100, 200, 3), np.uint8)
    assert vggt_car(fake, [img, img], [np.zeros((100, 200), np.uint8)] * 2) == [None, None]


# ---------------------------------------------------------------- combine
def test_combine_takes_focal_from_vggt_and_angles_from_moge():
    moge = [FrameCam(f_px=1400.0, pitch=10.0, roll=1.0, height_m=1.5, conf=0.9, source="moge"),
            FrameCam(f_px=1600.0, pitch=8.0, roll=-1.0, height_m=1.4, conf=0.8, source="moge"),
            None,
            FrameCam(f_px=900.0, pitch=5.0, roll=0.0, height_m=1.2, conf=0.7, source="moge")]
    vggt = [FrameCam(f_px=1490.0, rel_yaw_deg=0.0, conf=0.8, source="vggt"),
            FrameCam(f_px=1510.0, rel_yaw_deg=40.0, conf=0.8, source="vggt"),
            FrameCam(f_px=1500.0, rel_yaw_deg=80.0, conf=0.8, source="vggt"),
            None]
    sizes = [(W, H), (W, H), (W, H), (1280, 720)]
    out = combine(moge, vggt, sizes)
    assert out[0].f_px == pytest.approx(1500.0) and out[1].f_px == pytest.approx(1500.0)
    want = math.degrees(math.atan(math.tan(math.radians(10.0)) * 1500 / 1400))
    assert out[0].pitch == pytest.approx(want) and out[0].roll == 1.0 and out[0].height_m == 1.5
    assert out[1].rel_yaw_deg == 40.0 and out[0].source == "hosted"
    assert "camera-moge" in out[0].flags and "camera-vggt" in out[0].flags and "focal-vggt" in out[0].flags
    assert out[2].source == "vggt" and out[2].pitch is None and "moge-missing" in out[2].flags
    assert out[3].f_px == 900.0 and out[3].pitch == 5.0 and "vggt-missing" in out[3].flags
    assert "focal-moge" in out[3].flags
    assert combine([None], None, [(W, H)]) == [None]
    only_moge = combine(moge[:1], None, [(W, H)])[0]
    assert only_moge.f_px == 1400.0 and only_moge.pitch == 10.0 and "vggt-missing" in only_moge.flags


# ---------------------------------------------------------------- yaw helpers
def test_yaw_probs_peak_on_the_nearest_bin():
    p = yaw_probs(95.0)
    assert set(p) == set(pose.BIN_DEG) and sum(p.values()) == pytest.approx(1.0)
    assert max(p, key=p.get) == "right" and p["right"] > 0.9
    q = yaw_probs(337.0, sigma=15)
    assert max(q, key=q.get) == "front-left" and q["front"] > q["left"]


def test_anchor_yaws_is_a_circular_median():
    rel = [0.0, 40.0, 90.0, 180.0, None, 270.0]
    fitted = [350.0, 32.0, 81.0, 170.0, 10.0, 100.0]  # one outlier (270 -> 100); offset about -9
    off = anchor_yaws(rel, fitted)
    assert abs((off - 351.0 + 180) % 360 - 180) <= 2.0
    assert anchor_yaws([None, 5.0], [3.0, None]) is None


# ---------------------------------------------------------------- hooks in car_fit_to_bg
def _measured(external, focal=600.0):
    return batch.Measured(path=Path("photo.png"), stem="photo", size=(960, 640), cam=LineCamera(), exif_f=None,
                          time_s=None, focal=focal, focal_source="config", external=external)


def _run_fit(monkeypatch, m, lines: LineCamera, h_prior=(1.6, 0.25)):
    seen = {}

    def fake_fit(frame, alpha, cut, shape, focal, focal_source, cam, **kwargs):
        seen.update(cam=cam, **kwargs)
        return SimpleNamespace(flags=list(cam.flags))

    monkeypatch.setattr(batch, "line_camera", lambda *a, **k: lines)
    monkeypatch.setattr(batch, "fit_photo", fake_fit)
    fit = batch.fit_measured(m, np.zeros((640, 960, 3), np.uint8), np.zeros((640, 960), np.float32), set(), SHAPE,
                             [m], None, h_prior=h_prior)
    return fit, seen


STRONG = LineCamera(roll=0.5, pitch=10.0, f=600.0, conf=0.9, conf_f=0.9, source="lines")
WEAK = LineCamera(roll=None, pitch=None, f=600.0, conf=0.1, source="lines", flags=["no-lines"])


def test_no_external_changes_nothing(monkeypatch):
    fit, seen = _run_fit(monkeypatch, _measured(None), STRONG)
    assert seen["cam"] is STRONG and seen["h_prior"] == (1.6, 0.25) and "strong_h_prior" not in seen


def test_external_camera_is_used_with_weak_lines(monkeypatch):
    ext = HostedCamera(f_px=600.0, pitch=12.0, roll=1.0, height_m=1.4, flags=["camera-moge", "camera-vggt"])
    fit, seen = _run_fit(monkeypatch, _measured(ext), WEAK)
    cam = seen["cam"]
    assert cam.source == "hosted" and cam.strong and cam.conf == 0.8 and cam.f == 600.0
    assert cam.pitch == pytest.approx(12.0) and cam.roll == 1.0
    assert cam.flags[0] == "camera-hosted" and "camera-hosted" in fit.flags
    assert seen["h_prior"] == (1.4, 0.15) and seen["strong_h_prior"] == (1.4, 0.15)


def test_external_pitch_moves_to_the_chosen_focal(monkeypatch):
    ext = HostedCamera(f_px=500.0, pitch=12.0, roll=1.0)
    _, seen = _run_fit(monkeypatch, _measured(ext, focal=600.0), WEAK)
    want = math.degrees(math.atan(math.tan(math.radians(12.0)) * 600 / 500))
    assert seen["cam"].pitch == pytest.approx(want) and seen["h_prior"] == (1.6, 0.25)
    assert "strong_h_prior" not in seen


def test_agreeing_strong_lines_give_way_to_the_hosted_camera(monkeypatch):
    ext = HostedCamera(f_px=600.0, pitch=12.0, roll=-2.0, height_m=1.5)
    _, seen = _run_fit(monkeypatch, _measured(ext), STRONG)
    assert seen["cam"].source == "hosted" and seen["cam"].roll == -2.0


def test_conflicting_strong_lines_win(monkeypatch):
    ext = HostedCamera(f_px=600.0, pitch=18.0, roll=0.0, height_m=1.5)
    fit, seen = _run_fit(monkeypatch, _measured(ext), STRONG)
    cam = seen["cam"]
    assert cam.source == "lines" and cam.pitch == 10.0 and "camera-lines-fallback" in fit.flags
    assert seen["h_prior"] == (1.6, 0.25) and "strong_h_prior" not in seen


def test_fit_pose_uses_a_strong_height_prior_only_when_given(monkeypatch):
    c = (479.5, 319.5)
    R, t = car_to_camera(60.0, 5.0, math.degrees(math.asin(1.5 / 5.0)), (470.0, 420.0), 0.0, 600.0, c)
    alpha = render_proxy(body_points(SHAPE, SHAPE.body) @ R.T + t, 600.0, c, (960, 640)).astype(np.float32) / 255
    seen = []

    class Stop(Exception):
        pass

    def fake_local(work, pts, shape, yaw0, **kw):
        if "hlim" in kw or "hprior" in kw:  # the first refine: the priors every candidate is judged by
            seen.append(kw)
            raise Stop
        return pose.Pose(yaw=yaw0, roll=0.0, D=5.0, h=1.5, delta=17.0, aim=(100.0, 100.0), iou=0.5, e=-0.5)

    monkeypatch.setattr(pose, "fit_local", fake_local)
    for extra in ({}, {"strong_h_prior": (1.3, 0.15)}):
        with pytest.raises(Stop):
            pose.fit_pose(alpha, set(), SHAPE, 600.0, "hatch", line_roll=0.0, line_pitch=10.0, strong_lines=True,
                          h_prior=(1.8, 0.25), **extra)
    assert "hprior" not in seen[0] and seen[1]["hprior"] == (1.3, 0.15)
