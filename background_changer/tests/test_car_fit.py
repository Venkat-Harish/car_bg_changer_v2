"""car_fit_to_bg: the car's pixels only get one rigid rotation and one INTER_AREA downscale; the pose fit
recovers a known pose; degenerate inputs fail cleanly; focal pooling never mixes lenses."""

import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from bg_changer.carmodel import CarShape
from bg_changer.compose import paste
from bg_changer.scenes import SCENES
from car_fit_to_bg import FitError, fit_photo, place_in_room
from car_fit_to_bg.level import downscale, level_car
from car_fit_to_bg.lines import LineCamera, choose_focal, estimate_camera, pool_focals
from car_fit_to_bg.place import stage_scale
from car_fit_to_bg.pose import body_points, cam_angles, car_to_camera, render_proxy
from car_fit_to_bg.rooms import build_room

ROOT = Path(__file__).resolve().parents[1]
SHAPE = CarShape(4.2, 1.8, 1.5, 2.6, "Test hatch", body="hatch")


def _scene(yaw=60.0, D=5.0, h=1.5, roll=0.0, size=(960, 640), f=600.0, aim=(470.0, 420.0)):
    """A noise photo with the proxy's silhouette as the car's alpha, from a known camera."""
    W, H = size
    c = ((W - 1) / 2, (H - 1) / 2)
    delta = math.degrees(math.asin(h / D))
    R, t = car_to_camera(yaw, D, delta, aim, roll, f, c)
    pts = body_points(SHAPE, SHAPE.body)
    m = render_proxy(pts @ R.T + t, f, c, (W, H))
    alpha = m.astype(np.float32) / 255.0
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    pitch, r = cam_angles(R)
    cam = LineCamera(roll=r, pitch=pitch, f=f, conf=0.95, conf_f=0.95)
    return frame, alpha, cam, dict(yaw=yaw, D=D, h=h, f=f)


@pytest.fixture(scope="module")
def fitted():
    frame, alpha, cam, truth = _scene()
    fit = fit_photo(frame, alpha, set(), SHAPE, truth["f"], "test", cam, probs=None, label=None, stem="synthetic")
    return frame, alpha, fit, truth


def test_pose_fit_recovers_a_known_pose(fitted):
    _, _, fit, truth = fitted
    assert abs((fit.pose.yaw - truth["yaw"] + 180) % 360 - 180) <= 3.0
    assert fit.pose.D == pytest.approx(truth["D"], rel=0.05)
    assert fit.pose.iou > 0.95


@pytest.mark.parametrize("name", sorted(SCENES))
def test_room_path_car_pixels_are_one_rotation_and_one_inter_area(fitted, name):
    """With shadow and reflection on, every opaque output pixel is INTER_AREA(level(source)) exactly."""
    frame, alpha, fit, _ = fitted
    plate = cv2.imread(str(ROOT / "backgrounds" / f"{name}.webp"), cv2.IMREAD_COLOR)
    placed = place_in_room(fit, name, plate, SCENES[name], shadow_gain=1.5, max_output_size=640)
    assert placed.image is not None and placed.s_out < 1
    pixels, a_s, corner = downscale(fit.car, placed.s_out)
    x0, y0 = placed.window
    px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
    H, W = placed.image.shape[:2]
    h, w = a_s.shape
    ya, xa, yb, xb = max(0, py), max(0, px), min(H, py + h), min(W, px + w)
    opaque = a_s[ya - py:yb - py, xa - px:xb - px] >= 1.0
    assert opaque.mean() > 0.3
    got = placed.image[ya:yb, xa:xb][opaque]
    want = cv2.resize(fit.car.pixels, (w, h), interpolation=cv2.INTER_AREA)[ya - py:yb - py, xa - px:xb - px][opaque]
    assert np.array_equal(got, want)


def test_level_below_half_a_degree_is_a_plain_crop():
    frame, alpha, _, _ = _scene()
    car = level_car(frame, alpha, 0.3, ((frame.shape[1] - 1) / 2, (frame.shape[0] - 1) / 2))
    x0, y0 = car.origin
    h, w = car.alpha.shape
    assert car.theta == 0.0
    assert np.array_equal(car.pixels, frame[y0:y0 + h, x0:x0 + w])


def _two_tone_car(H=400, W=600):
    """A striped two-tone car with a soft edge, the contrasting old background right next to it."""
    alpha = np.zeros((H, W), np.float32)
    cv2.ellipse(alpha, (300, 220), (180, 90), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    alpha[alpha < 1 / 255] = 0.0  # as segment.clean_alpha leaves it
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = (20, 220, 40)  # old background, painted wherever alpha is 0
    stripes = (np.arange(W) // 12) % 2 == 0
    car = alpha > 0
    frame[car & stripes[None, :]] = (90, 60, 200)
    frame[car & ~stripes[None, :]] = (130, 100, 160)
    return frame, alpha


def test_rotation_never_creates_new_colours_or_dark_fringes():
    """After the rotation every visible car pixel stays inside the car's own colour range (no old background,
    no Lanczos ringing): premultiplication and the clamp over alpha > 0 are both needed for this."""
    frame, alpha = _two_tone_car()
    car = level_car(frame, alpha, 3.0, ((frame.shape[1] - 1) / 2, (frame.shape[0] - 1) / 2))
    m = car.alpha > 0.02
    assert m.sum() > 1000
    lo, hi = np.array([90, 60, 160]), np.array([130, 100, 200])
    px = car.pixels[m].astype(int)
    assert (px >= lo - 1).all() and (px <= hi + 1).all()
    # ...and each one is a blend of the two car colours only (the old background never enters): on the line
    # from A = (90, 60, 200) to B = (130, 100, 160) the three channel offsets move together.
    t = px - np.array([90, 60, 200])
    assert np.abs(t[:, 0] - t[:, 1]).max() <= 2 and np.abs(t[:, 0] + t[:, 2]).max() <= 2


def test_soft_edge_after_rotation_and_downscale_is_a_clean_blend():
    """On a plain canvas every pasted soft-edge pixel is canvas*(1-a) + car*a (+-1): nothing outside the car
    (black canvas fill, old background) leaks into the outline through the downscale."""
    frame, alpha = _two_tone_car()
    frame[alpha > 0] = (90, 60, 200)  # one colour, so the expected blend is exact
    car = level_car(frame, alpha, 3.0, ((frame.shape[1] - 1) / 2, (frame.shape[0] - 1) / 2))
    pixels, a, _ = downscale(car, 0.3)
    canvas = np.full((pixels.shape[0], pixels.shape[1], 3), 200, np.uint8)
    paste(canvas, pixels, a, 0, 0)
    edge = (a > 0.02) & (a < 1.0)
    assert edge.sum() > 50
    want = 200 * (1 - a[edge, None]) + np.array([90, 60, 200]) * a[edge, None]
    assert np.abs(canvas[edge].astype(float) - want).max() <= 1.5


def test_rolled_photo_is_levelled_once_and_tracked_everywhere():
    """A photo with 3 degrees of roll: the car is rotated (only) once, and the angle every later step uses is
    the angle applied to the pixels; opaque output pixels equal INTER_AREA(level(source)) recomputed here."""
    frame, alpha, cam, truth = _scene(roll=3.0)
    fit = fit_photo(frame, alpha, set(), SHAPE, truth["f"], "test", cam, stem="rolled")
    assert abs(fit.theta) >= 0.5 and fit.theta == fit.car.theta
    ref = level_car(frame, alpha, fit.theta, fit.c)
    assert np.array_equal(ref.pixels, fit.car.pixels) and ref.origin == fit.car.origin
    plate = cv2.imread(str(ROOT / "backgrounds" / "background_1.webp"), cv2.IMREAD_COLOR)
    placed = place_in_room(fit, "background_1", plate, SCENES["background_1"], max_output_size=640)
    pixels, a_s, corner = downscale(ref, placed.s_out)
    x0, y0 = placed.window
    px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
    H, W = placed.image.shape[:2]
    h, w = a_s.shape
    ya, xa, yb, xb = max(0, py), max(0, px), min(H, py + h), min(W, px + w)
    opaque = a_s[ya - py:yb - py, xa - px:xb - px] >= 1.0
    want = cv2.resize(ref.pixels, (w, h), interpolation=cv2.INTER_AREA)[ya - py:yb - py, xa - px:xb - px][opaque]
    assert np.array_equal(placed.image[ya:yb, xa:xb][opaque], want)


def test_a_level_angle_below_half_a_degree_is_zero_everywhere():
    frame, alpha, cam, truth = _scene(roll=0.3)
    fit = fit_photo(frame, alpha, set(), SHAPE, truth["f"], "test", cam, stem="nearly level")
    assert fit.theta == 0.0 and fit.car.theta == 0.0
    R, t = fit.pose.rt(fit.c)
    assert np.allclose(fit.R_lev, R)


def test_cut_edge_stays_on_the_output_border():
    """A car running off the right edge of a rolled photo, on every room: the output's last column is car
    wherever the column next to it is car (no strip of room between the slanted cut and the border), and every
    opaque car pixel left after the crop at the cut edge is INTER_AREA(level(source)) exactly."""
    frame, alpha, cam, truth = _scene(roll=2.0, aim=(820.0, 430.0))
    sm = alpha >= 0.5
    assert sm[:, -3:].any() and not sm[:, :3].any()
    fit = fit_photo(frame, alpha, {"right"}, SHAPE, truth["f"], "test", cam, stem="cut")
    assert fit.cut_edges["right"]["crop_pct"] <= 2.0 + 1e-6
    ref = level_car(frame, alpha, fit.theta, fit.c)
    assert np.array_equal(ref.pixels, fit.car.pixels)
    for name in sorted(SCENES):
        plate = cv2.imread(str(ROOT / "backgrounds" / f"{name}.webp"), cv2.IMREAD_COLOR)
        placed = place_in_room(fit, name, plate, SCENES[name], max_output_size=640)
        assert placed.s_out < 1
        pixels, a_s, corner = downscale(fit.car, placed.s_out)
        x0, y0 = placed.window
        Ho, Wo = placed.image.shape[:2]
        a_out = np.zeros((Ho, Wo), np.float32)
        px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
        h, w = a_s.shape
        ya, xa, yb, xb = max(0, py), max(0, px), min(Ho, py + h), min(Wo, px + w)
        assert xb == Wo and px + w > Wo  # the car really is cropped at the output's right border
        a_out[ya:yb, xa:xb] = a_s[ya - py:yb - py, xa - px:xb - px]
        rows = a_out[:, -3] >= 0.99
        assert rows.sum() > 10, name
        assert (a_out[rows, -1] >= 0.5).mean() > 0.97, name
        opaque = a_s[ya - py:yb - py, xa - px:xb - px] >= 1.0
        want = cv2.resize(ref.pixels, (w, h), interpolation=cv2.INTER_AREA)[ya - py:yb - py, xa - px:xb - px][opaque]
        assert np.array_equal(placed.image[ya:yb, xa:xb][opaque], want), name


def test_fit_and_render_are_deterministic():
    frame, alpha, cam, truth = _scene(yaw=120.0, roll=1.5)
    runs = []
    for _ in range(2):
        fit = fit_photo(frame, alpha, set(), SHAPE, truth["f"], "test", cam, stem="repeat")
        plate = cv2.imread(str(ROOT / "backgrounds" / "background_3.webp"), cv2.IMREAD_COLOR)
        placed = place_in_room(fit, "background_3", plate, SCENES["background_3"], max_output_size=480)
        runs.append((fit.pose.yaw, fit.pose.D, fit.theta, placed.image))
    assert runs[0][:3] == runs[1][:3]
    assert np.array_equal(runs[0][3], runs[1][3])


def test_scene_size_override_scales_the_room_the_other_way():
    room = build_room("background_1")
    spec = SCENES["background_1"]
    S = room.floor_point(spec.anchor)
    a = stage_scale(spec, 4.868, room, S)
    b = stage_scale(replace(spec, size=0.8), 4.868, room, S)
    assert b == pytest.approx(0.8 * a)


@pytest.mark.parametrize("alpha", [np.zeros((200, 300), np.float32), np.ones((200, 300), np.float32)])
def test_degenerate_masks_fail_cleanly(alpha):
    frame = np.zeros((200, 300, 3), np.uint8)
    cam = LineCamera()
    with pytest.raises(FitError):
        fit_photo(frame, alpha, set(), SHAPE, 300.0, "test", cam)


def test_uniform_image_has_no_line_camera():
    assert estimate_camera(np.full((240, 320, 3), 128, np.uint8)) is None


def test_two_lenses_at_one_image_size_are_not_pooled():
    cams = [LineCamera(f=366.0 * (1 + 0.01 * i), conf_f=0.9, sd_logf=0.02) for i in range(4)]
    cams += [LineCamera(f=700.0 * (1 + 0.01 * i), conf_f=0.9, sd_logf=0.02) for i in range(3)]
    size = (464, 832)
    pooled = pool_focals(cams, [size] * len(cams))
    assert pooled[size] == pytest.approx(370.0, rel=0.03)
    f, src, flags = choose_focal(cams[-1], size, pooled, None, None)
    assert f == pytest.approx(cams[-1].f) and "lens-change" in flags


def test_synthetic_vanishing_points_recover_roll_pitch_and_focal():
    """Edges of boxes (verticals + two horizontal directions) drawn from a camera with known tilt and focal."""
    W, H, f = 800, 600, 640.0
    roll, pitch = 3.0, 12.0
    r, p = math.radians(roll), math.radians(pitch)
    down = np.array([math.sin(r) * math.cos(p), math.cos(r) * math.cos(p), math.sin(p)])
    e1 = np.array([1.0, 0, 0]) - down[0] * down
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)
    a = math.radians(30)
    hx, hy = math.cos(a) * e1 + math.sin(a) * e2, -math.sin(a) * e1 + math.cos(a) * e2
    img = np.full((H, W, 3), 200, np.uint8)
    rng = np.random.default_rng(3)
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    for _ in range(140):
        P0 = np.array([rng.uniform(-8, 8), rng.uniform(-4, 4), rng.uniform(6, 20)])
        d = [-down, hx, hy][rng.integers(0, 3)]
        P1 = P0 + d * rng.uniform(1.5, 4.0)
        q0, q1 = c + f * P0[:2] / P0[2], c + f * P1[:2] / P1[2]
        cv2.line(img, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), (20, 20, 20), 2, cv2.LINE_AA)
    est = estimate_camera(img, None, None, n_boot=5)
    assert est is not None
    assert est["roll"] == pytest.approx(roll, abs=0.7)
    assert est["pitch"] == pytest.approx(pitch, abs=0.7)
    assert est["f"] == pytest.approx(f, rel=0.06)


def test_paste_keeps_opaque_pixels():
    canvas = np.zeros((10, 10, 3), np.uint8)
    car = np.full((4, 4, 3), 200, np.uint8)
    a = np.ones((4, 4), np.float32)
    paste(canvas, car, a, 3, 3)
    assert (canvas[3:7, 3:7] == 200).all()
