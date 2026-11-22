"""room3d: a background picture rebuilt as a 3D room and re-rendered from any pinhole camera.

    from room3d import Room, render
    out = render(plate, room, K, R, C, (w, h))   # out.lin: linear-light BGR float32

The background side only: nothing here knows about cars (see car_fit_to_bg for that).
"""

from .render import FILL_DEFAULT, Render, render, texels_per_px
from .room import (CEIL, FLOOR, NONE, WALL, Room, draw_room, kmat, linear_to_srgb, look_at, plane_h, project,
                   srgb_to_linear)

__all__ = ["CEIL", "FILL_DEFAULT", "FLOOR", "NONE", "WALL", "Render", "Room", "draw_room", "kmat",
           "linear_to_srgb", "look_at", "plane_h", "project", "render", "srgb_to_linear", "texels_per_px"]
