"""Where the car goes on a background: scale from height, ground pin, footprint centre on the anchor.

    s  = min(1, px_per_m * h_rel(yaw) / H)                  height only (width is cut in some frames)
    y  : lowest solid row      -> anchor_y + px_per_m * low_rel(yaw)
    x  : box centre            -> anchor_x + px_per_m * cx(yaw, sigma)      (uncut)
         uncut edge            -> the proxy's predicted edge, corrected for ASPECT_BIAS   (cut frames)

The car is only scaled down (never up) and moved by whole pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .carmodel import ASPECT_BIAS, CarTable, ViewRow
from .probe import CarProbe
from .segment import CarMask
from .yaw import YawEstimate

CUT_POLICIES = ("anchor", "centre", "skip")
HW_RANGE = (0.90, 1.18)  # expected (w_rel / W) / (h_rel / H) for an uncut frame: 1.05 +- 0.04
MAX_GUARD_SHIFT = 100  # px on the picture the scene was measured on
CAP_TOL = 0.01  # "scale-capped" only when the car would need more than 1% enlargement


class XMap:
    """Piecewise-linear map of image columns (linear beyond the ends), with its inverse."""

    def __init__(self, src, dst):
        order = np.argsort(src)
        self.src = np.asarray(src, dtype=np.float64)[order]
        self.dst = np.asarray(dst, dtype=np.float64)[order]
        if len(self.src) == 1:  # a single knot: unit slope
            self.src = np.array([self.src[0], self.src[0] + 1.0])
            self.dst = np.array([self.dst[0], self.dst[0] + 1.0])

    @staticmethod
    def _apply(x, xs, ys):
        x = np.asarray(x, dtype=np.float64)
        y = np.interp(x, xs, ys)
        lo, hi = x < xs[0], x > xs[-1]
        y = np.where(lo, ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]), y)
        return np.where(hi, ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), y)

    def fwd(self, x):
        return self._apply(x, self.src, self.dst)

    def inv(self, x):
        return self._apply(x, self.dst, self.src)


@dataclass
class Plan:
    scale: float
    top_left: tuple[int, int]  # where the scaled crop goes
    size: tuple[int, int]  # scaled crop size (w, h)
    sx: float
    sy: float
    box_out: tuple[int, int, int, int]  # the solid outline on the background
    yaw: YawEstimate
    view: ViewRow
    anchor: tuple[float, float]
    px_per_m: float
    target_h: float
    target_low: float
    rho: float  # pasted width / V's predicted width
    xmap: XMap  # V's columns -> background columns, so floor effects follow the pasted car
    hw_ratio: float | None
    margin: int  # px to the left, right and top edges (the bottom is checked separately)
    cut: tuple[str, ...]
    flags: list[str] = field(default_factory=list)
    floor: object = None  # set by Scene.compose (OutlineFloor), for the debug overlay
    contacts: list = field(default_factory=list)

    def to_out(self, x, y):
        """Source-photo coordinates -> background coordinates."""
        cx1, cy1 = self.crop_origin
        return (self.top_left[0] + (np.asarray(x) - cx1) * self.sx,
                self.top_left[1] + (np.asarray(y) - cy1) * self.sy)

    crop_origin: tuple[int, int] = (0, 0)


def _fit_shift(lo: float, hi: float, lo_lim: float, hi_lim: float) -> float:
    """Shift that brings [lo, hi] inside [lo_lim, hi_lim]; if it cannot fit, keep its middle centred."""
    if hi - lo > hi_lim - lo_lim:
        return (lo_lim + hi_lim) / 2 - (lo + hi) / 2
    if lo < lo_lim:
        return lo_lim - lo
    if hi > hi_lim:
        return hi_lim - hi
    return 0.0


def skip_reason(probe: CarProbe, cut_policy: str) -> str:
    """Why a photo is not placed ('' if it is)."""
    sides = [c for c in probe.cut if c in ("left", "right")]
    vert = [c for c in probe.cut if c in ("top", "bottom")]
    if sides and vert:
        return (f"not placed: the photo cuts the car off on two sides ({','.join(probe.cut)}), "
                "so neither its width nor its height can be trusted")
    if probe.cut and cut_policy == "skip":
        return f"skipped: car cut off ({','.join(probe.cut)}) and cut_frames = 'skip'"
    return ""


def raw_scale(probe: CarProbe, row: ViewRow, px_per_m: float) -> float:
    """Photo px -> background px that gives the car its stage size. Above 1 when the photo's car is
    smaller than the stage needs (it is then capped at 1: the car is never enlarged)."""
    if any(c in ("top", "bottom") for c in probe.cut):  # height unreliable: from width instead
        return px_per_m * row.w / (ASPECT_BIAS * probe.w)
    return px_per_m * row.h / probe.h


def plan_on_anchor(car: CarMask, probe: CarProbe, yaw: YawEstimate, table: CarTable,
                   anchor: tuple[float, float], px_per_m: float, canvas_wh: tuple[int, int],
                   cut_policy: str = "anchor", max_lowest_row: float | None = None,
                   max_shift: float = MAX_GUARD_SHIFT) -> Plan | None:
    """None when the frame should be skipped (cut_policy 'skip', or cut on both axes)."""
    if cut_policy not in CUT_POLICIES:
        raise ValueError(f"cut_policy must be one of {', '.join(CUT_POLICIES)}")
    if skip_reason(probe, cut_policy):
        return None
    cut = probe.cut
    side_cut = [c for c in cut if c in ("left", "right")]
    vert_cut = [c for c in cut if c in ("top", "bottom")]
    row = table.at(yaw.deg)
    ax, ay = anchor
    flags: list[str] = list(yaw.flags) + [f"cut-{c}" for c in cut]

    target_h = px_per_m * row.h
    s_raw = raw_scale(probe, row, px_per_m)
    if s_raw > 1.0 + CAP_TOL:
        flags.append("scale-capped")
    s = min(1.0, s_raw)

    cx1, cy1, cx2, cy2 = car.crop
    cw, ch = cx2 - cx1, cy2 - cy1
    new_w, new_h = max(1, round(cw * s)), max(1, round(ch * s))
    sx, sy = new_w / cw, new_h / ch
    bx1, by1, bx2, by2 = probe.box

    # Ground pin: the bottom edge of the lowest solid row lands on the proxy's lowest row. When the
    # photo cuts the car's bottom off, its lowest row is the photo edge, so pin the intact roof instead.
    target_low = ay + px_per_m * row.low
    if "bottom" in vert_cut and "top" not in vert_cut and cut_policy == "anchor":
        y0 = round(ay + px_per_m * row.top - (by1 - cy1) * sy)
    else:
        y0 = round(target_low - (by2 - cy1) * sy)

    # Horizontal: footprint centre on the anchor.
    pred_w = px_per_m * row.w
    bias_shift = 0.5 * (1 - 1 / ASPECT_BIAS) * pred_w  # pasted car is ~5% narrower than V predicts
    if "right" in side_cut and "left" not in side_cut and cut_policy == "anchor":
        x_left = ax + px_per_m * row.x0 + bias_shift
        x0 = round(x_left - (bx1 - cx1) * sx)
    elif "left" in side_cut and "right" not in side_cut and cut_policy == "anchor":
        x_right = ax + px_per_m * row.x1 - bias_shift
        x0 = round(x_right - (bx2 - cx1) * sx)
    else:
        x_c = ax + px_per_m * table.cx_smoothed(yaw.deg, yaw.sigma)
        x0 = round(x_c - ((bx1 + bx2) / 2 - cx1) * sx)

    W, H = canvas_wh
    box_out = [x0 + (bx1 - cx1) * sx, y0 + (by1 - cy1) * sy, x0 + (bx2 - cx1) * sx, y0 + (by2 - cy1) * sy]
    # Guard: keep the car's outline on the canvas and above the scene's limit row, moving it at most
    # max_shift px; whatever still sticks out is flagged as cropped.
    bottom_limit = min(H, max_lowest_row) if max_lowest_row is not None else H
    dx = _fit_shift(box_out[0], box_out[2], 0, W)
    dy = _fit_shift(box_out[1], box_out[3], 0, bottom_limit)
    if dx or dy:
        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))
        x0, y0 = round(x0 + dx), round(y0 + dy)
        box_out = [box_out[0] + dx, box_out[1] + dy, box_out[2] + dx, box_out[3] + dy]
        flags.append("near-edge")
    if box_out[0] < -0.5 or box_out[2] > W + 0.5 or box_out[1] < -0.5 or box_out[3] > bottom_limit + 0.5:
        flags.append("cropped")
    box_i = tuple(int(round(v)) for v in box_out)
    margin = int(min(box_i[0], W - box_i[2], box_i[1]))

    # Map V's drawing of the car onto the pasted car, so shadows and reflections follow the real car
    # even when the label's angle is off: its ends, and its lowest point (the near tyre or bumper).
    pred_x0, pred_x1 = ax + px_per_m * row.x0, ax + px_per_m * row.x1
    pred_low = ax + px_per_m * row.low_x
    pasted_w = probe.w * sx
    low_out = x0 + (probe.lowest_x - cx1) * sx
    one_side_cut = side_cut and cut_policy == "anchor" and len(side_cut) == 1
    if one_side_cut:
        rho, hw = 1 / ASPECT_BIAS, None
        edge = (pred_x0, box_out[0]) if "right" in side_cut else (pred_x1, box_out[2])
        toward_cut = 1.0 if "right" in side_cut else -1.0
        knots = [edge]
        lowest_at_cut = abs(low_out - edge[1]) < 0.02 * pasted_w or \
            (("right" in side_cut and low_out > box_out[2] - 0.02 * pasted_w) or
             ("left" in side_cut and low_out < box_out[0] + 0.02 * pasted_w))
        if not probe.tie and not lowest_at_cut:
            knots.append((pred_low, low_out))
            # Past the lowest point, keep the pasted car's own scale: the short edge-to-tyre stretch is
            # mostly the end face, whose slope would carry the angle error across the whole car.
            knots.append((pred_low + toward_cut * 1000.0, low_out + toward_cut * 1000.0 * rho))
        else:
            knots.append((edge[0] + toward_cut * 1000.0, edge[1] + toward_cut * 1000.0 * rho))
        if pasted_w > pred_w * 1.02:  # the visible part alone is wider than the whole predicted car
            flags.append("hw-mismatch")
    elif len(side_cut) == 2:  # cut at both ends: only the scale is known, around the lowest point
        rho, hw = 1 / ASPECT_BIAS, None
        centre = (pred_low, low_out) if not probe.tie else (ax + px_per_m * row.cx, (box_out[0] + box_out[2]) / 2)
        knots = [centre, (centre[0] + 1000.0, centre[1] + 1000.0 * rho)]
    else:
        rho = pasted_w / pred_w if pred_w > 0 else 1.0
        knots = [(pred_x0, box_out[0]), (pred_x1, box_out[2])]
        inside = pred_x0 + 0.1 * pred_w < pred_low < pred_x1 - 0.1 * pred_w and \
            box_out[0] + 0.1 * pasted_w < low_out < box_out[2] - 0.1 * pasted_w
        if not probe.tie and inside:
            knots.insert(1, (pred_low, low_out))
        hw = None if (side_cut or vert_cut) else (row.w / probe.w) / (row.h / probe.h)
        if hw is not None and not HW_RANGE[0] <= hw <= HW_RANGE[1]:
            flags.append("hw-mismatch")
    xs, ys = np.array([k[0] for k in knots]), np.array([k[1] for k in knots])
    order = np.argsort(xs)
    slopes = np.diff(ys[order]) / np.maximum(np.diff(xs[order]), 1e-6)
    if np.any(slopes < 0.5 * rho) or np.any(slopes > 2.0 * rho):  # implausible: drop the lowest-point knot
        if one_side_cut:
            toward_cut = 1.0 if "right" in side_cut else -1.0
            knots = [knots[0], (knots[0][0] + toward_cut * 1000.0, knots[0][1] + toward_cut * 1000.0 * rho)]
        elif len(knots) > 2:
            knots = [knots[0], knots[-1]]
    xmap = XMap([k[0] for k in knots], [k[1] for k in knots])
    return Plan(scale=s, top_left=(int(x0), int(y0)), size=(new_w, new_h), sx=sx, sy=sy, box_out=box_i,
                yaw=yaw, view=row, anchor=(ax, ay), px_per_m=px_per_m, target_h=target_h, target_low=target_low,
                rho=rho, xmap=xmap, hw_ratio=hw, margin=margin, cut=cut, flags=flags, crop_origin=(cx1, cy1))
