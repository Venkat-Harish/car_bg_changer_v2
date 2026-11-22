"""room3d: a background rebuilt as a 3D room re-renders correctly from its own camera and from new ones."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from car_fit_to_bg.rooms import BUILDERS, build_room
from room3d import FLOOR, Room, kmat, linear_to_srgb, look_at, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name, tol", [("background_1", 0), ("background_2", 1)])
def test_room_re_renders_its_own_plate_exactly(name, tol):
    room = build_room(name)
    plate = cv2.imread(str(ROOT / "backgrounds" / f"{name}.webp"), cv2.IMREAD_COLOR)
    out = render(plate, room, room.K, room.R, room.C, room.size)
    img = linear_to_srgb(out.lin)
    assert out.hole.mean() == 0.0
    assert np.abs(img.astype(int) - plate.astype(int)).max() <= tol  # background_2's calibrated R is not quite orthonormal


def test_every_room_contains_its_plate_camera_and_rejects_a_wrong_plate():
    for name in BUILDERS:
        room = build_room(name)
        assert room.contains(room.C, 0.3)
        with pytest.raises(ValueError):
            render(np.zeros((540, 960, 3), np.uint8), room, room.K, room.R, room.C, (64, 36))


def test_camera_above_the_ceiling_is_outside_and_clamped_back_in():
    room = build_room("background_2")
    C = np.array([0.0, -3.0, room.ceiling + 0.5])
    assert not room.contains(C, 0.3)
    Cc = room.clamp_inside(C, 0.3)
    assert room.contains(Cc, 0.3) and Cc[2] <= room.ceiling - 0.3 + 1e-9


# ---------------------------------------------------------------- a synthetic box seen from a new camera
BOX = dict(x0=-4.0, x1=4.0, y0=-12.0, y1=6.0, top=3.0)


def _texture(u, v):
    """A smooth, unambiguous texture of surface coordinates (u, v) in metres."""
    r = 0.5 + 0.25 * np.sin(1.3 * u) + 0.2 * np.cos(0.9 * v)
    g = 0.5 + 0.25 * np.sin(0.7 * v + 0.5) * np.cos(0.4 * u)
    b = 0.5 + 0.2 * np.sin(0.5 * (u + v))
    return np.clip(np.stack([b, g, r], -1), 0, 1)


def _analytic(K, R, C, size):
    """Ray-cast the textured box: floor z=0, ceiling z=top, four walls. Returns (BGR uint8, hit points)."""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    d = (R.T @ np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.stack([
            np.where(d[:, 2] < 0, -C[2] / d[:, 2], np.inf),
            np.where(d[:, 2] > 0, (BOX["top"] - C[2]) / d[:, 2], np.inf),
            np.where(d[:, 0] < 0, (BOX["x0"] - C[0]) / d[:, 0], np.inf),
            np.where(d[:, 0] > 0, (BOX["x1"] - C[0]) / d[:, 0], np.inf),
            np.where(d[:, 1] < 0, (BOX["y0"] - C[1]) / d[:, 1], np.inf),
            np.where(d[:, 1] > 0, (BOX["y1"] - C[1]) / d[:, 1], np.inf)], 1)
    ts[ts <= 0] = np.inf
    k = ts.argmin(1)
    t = ts[np.arange(len(k)), k]
    P = C + t[:, None] * d
    su = np.where(k <= 1, P[:, 0], np.where(k <= 3, P[:, 1], P[:, 0]))
    sv = np.where(k <= 1, P[:, 1], P[:, 2] + 10.0 * k)
    col = _texture(su, sv)
    img = np.clip(np.rint(col * 255), 0, 255).astype(np.uint8).reshape(h, w, 3)
    return img, P.reshape(h, w, 3)


def test_a_new_viewpoint_matches_an_analytic_render_where_the_plate_saw_it():
    size_a = (640, 400)
    KA = kmat(500.0, 319.5, 199.5)
    CA = np.array([0.0, -10.0, 1.2])
    RA = look_at(CA, (0.0, 0.0, 0.4))
    plate, _ = _analytic(KA, RA, CA, size_a)
    poly = [(BOX["x0"], BOX["y0"]), (BOX["x1"], BOX["y0"]), (BOX["x1"], BOX["y1"]), (BOX["x0"], BOX["y1"])]
    room = Room("box", KA, RA, CA, size_a, poly, BOX["top"], BOX["top"], texture_box=(-1.0, -6.0, 2.0, 1.0))
    # A Baleno-like camera: closer, higher, looking down 25 degrees and rolled a little.
    CB = np.array([1.0, -5.0, 2.0])
    RB = look_at(CB, (0.0, 0.0, 0.0))
    a = math.radians(3.0)
    RB = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]) @ RB
    KB = kmat(420.0, 239.5, 159.5)
    out = render(plate, room, KB, RB, CB, (480, 320))
    want, P = _analytic(KB, RB, CB, (480, 320))
    got = linear_to_srgb(out.lin)
    # Pixels whose surface point the plate saw must match the analytic render.
    u, v, z = room.to_plate(P[..., 0], P[..., 1], P[..., 2])
    seen = room.inside(u, v, z, margin=3.0)
    assert seen.mean() > 0.4
    err = np.abs(got.astype(int) - want.astype(int)).mean(-1)
    inner = cv2.erode(seen.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert float(np.median(err[inner])) <= 2.0
    assert float(np.percentile(err[inner], 95)) <= 8.0
    # The renderer's hole mask is (nearly) the analytically unseen set.
    agree = (out.hole == ~room.inside(u, v, z, margin=-0.5)).mean()
    assert agree > 0.98
    assert (out.kinds == FLOOR).any()
