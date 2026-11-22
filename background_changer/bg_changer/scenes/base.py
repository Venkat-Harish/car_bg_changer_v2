"""What every background ("scene") module provides, and the shared compose step.

A scene module only has to describe its background, with numbers measured once from the picture:
  - anchor: the floor point where the car's footprint centre stands (turntable centre, road spot);
  - side_px: how long a pure side view of the car is through the capture camera, in px;
    this sets the stage size, so every car fills the stage the same way;
  - plate_floor: the background camera's floor (how fast the floor recedes in front of the car);
  - floor_mask: where shadows may fall;
  - shadow / reflection styles and any limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import cv2
import numpy as np

from ..carmodel import CarTable
from ..compose import paste
from ..grounding import (CompressionFloor, HomographyFloor, LevelFloor, PinnedFloorMap, PlateFloor, ReflectionStyle,
                         ShadowStyle, add_reflection, linear_to_srgb, outline_floor, shadow_multiplier, snap_contacts,
                         srgb_to_linear)
from ..placement import MAX_GUARD_SHIFT, Plan, plan_on_anchor
from ..probe import CarProbe
from ..segment import CarMask
from ..yaw import YawEstimate

GENERIC_HFOV_FOCAL = 0.866  # focal length / width for an unknown plate (~60 deg horizontal FOV)
GENERIC_METRIC_RATIO = 0.89  # side-matched px/m is ~0.89 x the plate's metric px/m (measured on all 3 plates)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    plate_size: tuple[int, int]  # (width, height) the numbers were measured at
    anchor: tuple[float, float]
    side_px: float
    floor_mask: Callable[[int, int], np.ndarray]  # (width, height) -> bool mask of the floor
    shadow: ShadowStyle
    plate_floor: PlateFloor | None = None  # None: from horizon_y, else constant compression
    horizon_y: float | None = None
    reflection: ReflectionStyle | None = None
    ring: tuple[float, float, float, float] | None = None  # glossy disc ellipse (cx, cy, a, b)
    max_lowest_row: float | None = None
    size: float = 1.0  # multiplies side_px
    shadow_gain: float = 1.0  # multiplies the run's shadow gain
    notes: str = ""
    px_scale: float = 1.0  # this picture's px per px of the picture it was measured on (background_scaler)


class Scene:
    def __init__(self, spec: SceneSpec, table: CarTable, plate_shape: tuple[int, ...]):
        h, w = plate_shape[:2]
        if (w, h) != tuple(spec.plate_size):
            raise ValueError(
                f"{spec.name}: the background is {w}x{h} but its scene was measured at "
                f"{spec.plate_size[0]}x{spec.plate_size[1]}; resize the picture or give it a new name "
                "(it then uses the generic scene)")
        self.spec, self.table = spec, table
        self.size_wh = (w, h)
        self.px_per_m = spec.side_px * spec.size / table.side_length()
        ax, ay = spec.anchor
        if spec.plate_floor is not None:
            self.plate_floor = spec.plate_floor
        elif spec.horizon_y is not None and spec.horizon_y < ay:
            z0 = GENERIC_HFOV_FOCAL * w / (self.px_per_m / GENERIC_METRIC_RATIO)
            self.plate_floor = LevelFloor.from_anchor(ay, spec.horizon_y, z0)
        else:
            self.plate_floor = CompressionFloor(0.6, table, ay, self.px_per_m)
        ps = spec.px_scale
        mask = spec.floor_mask(w, h).astype(np.float32)
        self.mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.5, 1.5 * ps))
        self.inside = None
        if spec.ring is not None:
            cx, cy, a, b = spec.ring
            yy, xx = np.mgrid[0:h, 0:w]
            inside = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0).astype(np.float32)
            self.inside = cv2.GaussianBlur(inside, (0, 0), sigmaX=max(1.0, 4 * ps))

    @property
    def name(self) -> str:
        return self.spec.name

    def plan(self, car: CarMask, probe: CarProbe, yaw: YawEstimate, cut_policy: str = "anchor") -> Plan | None:
        return plan_on_anchor(car, probe, yaw, self.table, self.spec.anchor, self.px_per_m, self.size_wh,
                              cut_policy, self.spec.max_lowest_row, MAX_GUARD_SHIFT * self.spec.px_scale)

    def compose(self, plate: np.ndarray, pixels: np.ndarray, alpha: np.ndarray, plan: Plan, probe: CarProbe,
                shadow_gain: float = 1.0) -> np.ndarray:
        """Background effects (reflection, shadow), then the car on top, unchanged."""
        out = plate.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = plan.box_out
        pad = int(round(0.9 * plan.px_per_m))
        roi = (max(0, x1 - pad), max(0, min(h - 1, (y1 + y2) // 2)), min(w, x2 + pad), h)
        rx1, ry1, rx2, ry2 = roi
        if rx2 <= rx1 or ry2 <= ry1:
            paste(out, pixels, alpha, *plan.top_left)
            return out
        floor = outline_floor(alpha, plan.top_left, plan.px_per_m)
        contacts = snap_contacts(plan, probe, floor)
        plan.floor, plan.contacts = floor, contacts  # kept for the debug overlay
        self._flag_contacts(plan, contacts)
        lin = srgb_to_linear(out[ry1:ry2, rx1:rx2])
        mask_roi = self.mask[ry1:ry2, rx1:rx2]
        if self.spec.reflection is not None:
            inside = self.inside[ry1:ry2, rx1:rx2] if self.inside is not None else None
            add_reflection(lin, roi, srgb_to_linear(pixels), alpha, plan, floor, self.spec.reflection,
                           mask_roi, inside, self.table.elevation)
        gain = shadow_gain * self.spec.shadow_gain
        if gain > 0:
            floor_map = PinnedFloorMap(self.table, plan, self.plate_floor)
            lin *= shadow_multiplier(plan, self.table, self.spec.shadow, floor_map, contacts, mask_roi, roi,
                                     floor, gain)
        out[ry1:ry2, rx1:rx2] = linear_to_srgb(lin)
        paste(out, pixels, alpha, *plan.top_left)
        return out

    @staticmethod
    def _flag_contacts(plan: Plan, contacts) -> None:
        """In side and 3/4 views, say when a near-side tyre could not be matched on the car."""
        s = np.sin(np.radians(plan.yaw.deg))
        if abs(s) < 0.5:
            return
        near = [c for i, c in enumerate(contacts) if c.visible and ((s > 0) == (i % 2 == 1))]
        matched = sum(c.snapped for c in near)
        flag = "contact-unsnapped" if near and not matched else ("contact-partial" if matched < len(near) else "")
        if flag and flag not in plan.flags:
            plan.flags.append(flag)

    def debug_overlay(self, image: np.ndarray, plan: Plan, probe: CarProbe) -> np.ndarray:
        """A copy with the anchor, the proxy's predicted box and footprint, and the tyre contacts."""
        out = image.copy()
        p, (ax, ay), row = plan.px_per_m, plan.anchor, plan.view
        ps = self.spec.px_scale  # marks and text shrink with a shrunk background (background_scaler)
        cv2.drawMarker(out, (int(round(ax)), int(round(ay))), (0, 0, 255), cv2.MARKER_CROSS,
                       max(12, round(40 * ps)), max(1, round(2 * ps)))
        box = (ax + p * row.x0, ay + p * row.top, ax + p * row.x1, ay + p * row.low)
        cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 128, 0), 1)
        h_floor = self.table.floor_homography(plan.yaw.deg, plan.anchor, p)
        fp = self.table.shape.footprint(n_arc=16)
        q = h_floor @ np.c_[fp, np.ones(len(fp))].T
        floor_map = PinnedFloorMap(self.table, plan, self.plate_floor)
        poly = np.stack([plan.xmap.fwd(q[0] / q[2]), floor_map.output_rows(q[1] / q[2])], axis=1)
        cv2.polylines(out, [np.round(poly).astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
        floor = plan.floor
        if floor is not None:
            line = np.stack([floor.u, floor.line], axis=1)
            cv2.polylines(out, [np.round(line).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
        for c in (plan.contacts or snap_contacts(plan, probe, floor)):
            colour = (0, 255, 0) if c.snapped else ((0, 0, 255) if c.visible else (128, 128, 128))
            cv2.circle(out, (int(round(c.u)), int(round(c.v))), max(3, round(6 * ps)), colour,
                       -1 if c.snapped else 1, cv2.LINE_AA)
        text = f"{self.name}  yaw {plan.yaw.deg:.0f} ({plan.yaw.source})  scale {plan.scale:.3f}  {' '.join(plan.flags)}"
        font = max(0.35, 0.8 * ps)
        cv2.putText(out, text, (max(6, round(20 * ps)), max(14, round(40 * ps))), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (0, 0, 255), 2 if font >= 0.6 else 1, cv2.LINE_AA)
        return out


def with_overrides(spec: SceneSpec, overrides: dict | None) -> SceneSpec:
    """Apply [scene.<name>] settings from config.toml (already validated)."""
    if not overrides:
        return spec
    changes = {}
    if "anchor" in overrides:
        anchor = tuple(float(v) for v in overrides["anchor"])
        changes["anchor"] = anchor
        pf = spec.plate_floor  # re-pin the calibrated floor at the new spot
        if isinstance(pf, HomographyFloor):
            changes["plate_floor"] = HomographyFloor(pf.h, anchor)
        elif isinstance(pf, LevelFloor) and anchor[1] > pf.horizon + 1:
            changes["plate_floor"] = LevelFloor(pf.horizon, pf.k, pf.k / (anchor[1] - pf.horizon))
    if "size" in overrides:
        changes["size"] = float(overrides["size"])
    if "side_fill" in overrides:
        changes["side_px"] = float(overrides["side_fill"]) * spec.plate_size[0]
    if "shadow" in overrides:
        changes["shadow_gain"] = float(overrides["shadow"])
    if "reflection" in overrides and not overrides["reflection"]:
        changes["reflection"] = None
    if "horizon_y" in overrides:
        changes["horizon_y"] = float(overrides["horizon_y"])
    return replace(spec, **changes)
