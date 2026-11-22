"""Hosted camera: MoGe-3 per frame (focal + floor plane -> camera height, pitch, roll) and VGGT per car (all frames
solved together: shared focal, relative yaw between shots).

Conventions (car_fit_to_bg/CLAUDE.md): pitch > 0 looks down, roll > 0 = horizon rises to the right, degrees;
focal in px at the photo's FULL size; principal point = frame centre; height in metres above the floor.

Contract (stub; Agent C implements):
- moge_frame(client, image_bgr, car_mask, max_side=1024) -> FrameCam | None
- vggt_car(client, images, car_masks, max_side=518) -> list[FrameCam | None]  (one per input, same order)
- combine(moge, vggt, sizes) -> list[HostedCamera | None]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fal import FalClient

MOGE_ENDPOINT = "microsoft/moge-3"
VGGT_ENDPOINT = "fal-ai/vggt-1b"


@dataclass
class FrameCam:
    f_px: float | None = None  # full-size px
    pitch: float | None = None
    roll: float | None = None
    height_m: float | None = None
    rel_yaw_deg: float | None = None  # VGGT: camera yaw relative to the first frame, degrees
    conf: float = 0.0  # 0..1
    source: str = ""  # moge | vggt


@dataclass
class HostedCamera:
    """What the fit consumes for one photo (car_fit_to_bg.batch.Measured.external)."""
    f_px: float | None = None
    pitch: float | None = None
    roll: float | None = None
    height_m: float | None = None
    rel_yaw_deg: float | None = None
    conf: float = 0.0
    source: str = "hosted"  # hosted | moge | vggt
    flags: list[str] = field(default_factory=list)


def moge_frame(client: FalClient, image: np.ndarray, car_mask: np.ndarray, max_side: int = 1024) -> FrameCam | None:
    raise NotImplementedError


def vggt_car(client: FalClient, images: list[np.ndarray], car_masks: list[np.ndarray],
             max_side: int = 518) -> list[FrameCam | None]:
    raise NotImplementedError


def combine(moge: list[FrameCam | None], vggt: list[FrameCam | None] | None,
            sizes: list[tuple[int, int]]) -> list[HostedCamera | None]:
    raise NotImplementedError


__all__ = ["MOGE_ENDPOINT", "VGGT_ENDPOINT", "FrameCam", "HostedCamera", "combine", "moge_frame", "vggt_car"]
