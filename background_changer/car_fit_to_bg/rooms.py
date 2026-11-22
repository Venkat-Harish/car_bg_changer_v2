"""The calibrated backgrounds as 3D rooms (room3d.Room), built from their scene modules' measured numbers.

Each room's world origin is the scene's measured stage point (the turntable centre or road spot) on the
floor; x right, y away from the plate camera, z up, in the plate's own units:
  background_1  white box studio: 6.0 m disc units, level camera 1.003 high; the box comes from its LED
                strips (half-maximum centre lines: back/floor row 632.15, back/ceiling row 34.09, corner
                columns 350.43 and 1573.60);
  background_2  grey studio with rounded corners: metres with the camera 1.0 high (its calibration); the
                walls are the measured wall/floor junction back-projected onto the floor and extruded;
                wall top 3.232 (the top shadow line), ceiling 3.474 (the dark band's top);
  background_3  race track: metres (0.80 m jersey barriers); ground + a backdrop standing on a smooth
                line through the nearest barrier bases (trees, hills and sky are painted on it).
A background without a builder here has no room: it is placed on the flat path (flag no-room).
"""

from __future__ import annotations

import numpy as np

from bg_changer.grounding import LevelFloor, ShadowStyle
from bg_changer.scenes import background_1, background_2, background_3, scene_key
from bg_changer.scenes.base import GENERIC_HFOV_FOCAL, SceneSpec
from room3d import Room, look_at

# background_1's level camera (docs/findings.md); its floor is SPEC.plate_floor.
BG1_F, BG1_PP = background_1.FOCAL, background_1.PRINCIPAL_POINT
BG1_LED = dict(back_floor_row=632.15, back_ceiling_row=34.09, left_col=350.43, right_col=1573.60)
BG2_WALL_TOP, BG2_CEILING = 3.232, 3.474
# Default backdrop line of background_3: the near envelope of the barrier bases (running max of the
# per-column rows over +-300 px, Gaussian sigma 40 px, Douglas-Peucker 0.5 px). No depth jumps, so the
# sky and trees painted on it stay continuous when the camera rises.
BG3_BACKDROP = [(0, 463.5), (552, 465.9), (744, 470.4), (816, 471.3), (1256, 471.2), (1476, 469.6), (1666, 466.7),
                (1746, 464.0), (1918, 455.3), (1919, 455.3)]
BG3_DEVIGNETTE = (0.18, 960.0)  # plate brightness ~ 1 - 0.18 (d / 960)^2 (asphalt 0.81 of the centre at the edges)


def room_background_1() -> Room:
    pf = background_1.SPEC.plate_floor
    assert isinstance(pf, LevelFloor)
    f, (cx, cy), k = BG1_F, BG1_PP, pf.k
    h = k / f  # camera height in disc units
    ax, ay = background_1.SPEC.anchor
    za = k / (ay - cy)  # stage depth from the camera
    xa = (ax - cx) * za / f
    C = np.array([-xa, -za, h])
    R = look_at(C, C + np.array([0.0, 1.0, 0.0]))
    zb = k / (BG1_LED["back_floor_row"] - cy)
    y_back = zb - za
    x_left = (BG1_LED["left_col"] - cx) * zb / f - xa
    x_right = (BG1_LED["right_col"] - cx) * zb / f - xa
    ceil = h + (cy - BG1_LED["back_ceiling_row"]) * zb / f
    y_front = C[1] - 30.0  # a closing wall behind the plate camera (never shown)
    poly = [(x_left, y_front), (x_right, y_front), (x_right, y_back), (x_left, y_back)]
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return Room("background_1", K, R, C, background_1.SPEC.plate_size, poly, ceil, ceil,
                texture_box=(-1.6, -3.5, 3.2, 0.4),  # terrazzo between the disc's near rim and the bottom row
                notes="white box from the LED strips; turntable is floor texture")


def room_background_2() -> Room:
    K, Rt, h = background_2.K, background_2.R, background_2.CAM_H
    FH = K @ np.column_stack([Rt[:, 0], Rt[:, 2], -h * Rt[:, 1]])
    a = np.linalg.solve(FH, [*background_2.SPEC.anchor, 1.0])
    xa, za = a[:2] / a[2]
    J = background_2.JUNCTION
    q = np.linalg.solve(FH, np.c_[J, np.ones(len(J))].T)
    xy = (q[:2] / q[2]).T - (xa, za)
    C = np.array([-xa, -za, h])
    R = Rt[:, [0, 2, 1]]  # world (x, y, z) = the calibration's (X, Z, Y)
    y_front = C[1] - 30.0
    poly = np.vstack([[xy[0, 0], y_front], xy, [xy[-1, 0], y_front]])
    return Room("background_2", K, R, C, background_2.SPEC.plate_size, poly, BG2_CEILING, BG2_CEILING,
                texture_box=(-1.5, -2.6, 3.0, 1.4),  # open concrete in front of the stage
                notes="curved walls from the junction polyline")


def room_background_3() -> Room:
    G = background_3.GROUND_H
    K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    cols = np.linalg.solve(K, G)
    cols = cols / np.linalg.norm(cols[:, 0])
    r1, r2, t = cols.T
    r_up = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r_up])  # world (x right, y forward, z up) -> camera
    h = -float(t @ r_up)
    a = np.linalg.solve(G, [*background_3.SPEC.anchor, 1.0])
    xa, ya = a[:2] / a[2]
    C = np.array([-xa, -ya, h])
    b = np.asarray(BG3_BACKDROP, float)
    q = np.linalg.solve(G, np.c_[b, np.ones(len(b))].T)
    xy = (q[:2] / q[2]).T - (xa, ya)
    y_behind = C[1] - 400.0
    poly = np.vstack([[xy[0, 0] - 400.0, y_behind], [xy[0, 0] - 400.0, xy[0, 1]], xy,
                      [xy[-1, 0] + 400.0, xy[-1, 1]], [xy[-1, 0] + 400.0, y_behind]])
    return Room("background_3", K, R, C, background_3.SPEC.plate_size, poly, 1.0e4, None,
                texture_box=(0.3, -1.0, 2.0, 1.5),  # plain asphalt right of the stage (no paint)
                devignette=BG3_DEVIGNETTE, notes="asphalt ground + backdrop at the barrier bases")


# Shadow styles for the room path, retuned on the photo-framed composites (the flat path's styles were
# "invisible at a glance" under a car seen from above). Contact/gap/band = each scene's outline layers.
_ROOM_STYLE = dict(tyre=0.90, tyre_sigma_m=0.025, core=0.70, core_inset_m=0.05, core_sigma_m=0.15, skirt=0.30,
                   skirt_grow_m=0.30, skirt_sigma_m=0.30, contact=0.80, gap=0.65, band=0.35)
ROOM_SHADOW = {
    "background_1": ShadowStyle(**_ROOM_STYLE),
    "background_2": ShadowStyle(**{**_ROOM_STYLE, "skirt_sigma_m": 0.33, "shift_m": 0.05}),
    "background_3": ShadowStyle(**{**_ROOM_STYLE, "tyre": 0.85, "tyre_sigma_m": 0.03, "core": 0.62, "skirt": 0.26,
                                   "skirt_grow_m": 0.35, "skirt_sigma_m": 0.35, "contact": 0.75, "gap": 0.55,
                                   "band": 0.30, "tint_gamma_bgr": (1.114, 1.0, 0.889)}),
}


DEFAULT_ROOM_SHADOW = ShadowStyle(**_ROOM_STYLE)
GENERIC_BACKDROP_ROWS = 10.0  # the generic scene's floor ends this many px below the horizon (its floor mask)


def generic_room(spec: SceneSpec, plate_shape) -> Room | None:
    """A background without its own module but with its horizon row ([scene.<name>] horizon_y): a level camera
    with the generic focal (about 60 degrees across), the floor up to just below the horizon, and everything
    above it painted on a far backdrop standing there. A rough model (flagged generic-room); None without a
    horizon below the anchor."""
    if spec.horizon_y is None:
        return None
    h, w = plate_shape[:2]
    horizon = float(spec.horizon_y)
    ax, ay = spec.anchor
    if ay <= horizon + GENERIC_BACKDROP_ROWS:
        return None
    f = GENERIC_HFOV_FOCAL * w
    cx = (w - 1) / 2
    K = np.array([[f, 0.0, cx], [0.0, f, horizon], [0.0, 0.0, 1.0]])  # level: the horizon row is the principal row
    cam_h = 1.0  # units: camera heights
    za = cam_h * f / (ay - horizon)  # the stage's depth
    xa = (ax - cx) * za / f
    C = np.array([-xa, -za, cam_h])
    R = look_at(C, C + np.array([0.0, 1.0, 0.0]))
    y_back = cam_h * f / GENERIC_BACKDROP_ROWS - za
    poly = [(-1e4, C[1] - 50.0), (1e4, C[1] - 50.0), (1e4, y_back), (-1e4, y_back)]
    tex = (-0.2 * za, -0.3 * za, 0.4 * za, 0.15 * za)  # floor between the stage and the camera
    return Room(spec.name, K, R, C, (w, h), poly, 1.0e4, None, texture_box=tex,
                notes="generic room: level camera, floor to the horizon, backdrop beyond")


BUILDERS = {"background_1": room_background_1, "background_2": room_background_2, "background_3": room_background_3}
_CACHE: dict = {}


def build_room(background_name: str) -> Room | None:
    """The background's own room (cached), or None when it has no room module."""
    key = scene_key(background_name)
    if key not in BUILDERS:
        return None
    if key not in _CACHE:
        _CACHE[key] = BUILDERS[key]()
    return _CACHE[key]


def room_for(background_name: str, spec: SceneSpec, plate_shape) -> tuple[Room | None, bool]:
    """(room, generic?): the background's own room, else a generic room from its horizon row, else None."""
    room = build_room(background_name)
    if room is not None:
        return room, False
    key = ("generic", scene_key(background_name), tuple(spec.anchor), spec.horizon_y, tuple(plate_shape[:2]))
    if key not in _CACHE:
        _CACHE[key] = generic_room(spec, plate_shape)
    return _CACHE[key], _CACHE[key] is not None


def stage_point(room: Room, anchor_px) -> np.ndarray:
    """The floor point (room units, relative to the room's origin) under a plate pixel, e.g. a
    [scene.<name>] anchor override; (0, 0) for the measured anchor."""
    return room.floor_point(anchor_px)


__all__ = ["BUILDERS", "DEFAULT_ROOM_SHADOW", "ROOM_SHADOW", "build_room", "generic_room", "room_background_1",
           "room_background_2", "room_background_3", "room_for", "stage_point"]
