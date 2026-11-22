"""Glass: SAM 3 "car window" mask + a deterministic dark tint. The ONLY step that changes car pixels.

User decision (Nov 2026): all glass is tinted, windscreen included, with a faint trace of the original kept; the
untouched frame is always saved next to the output (<out>/originals/, `save_original`). Outside the window mask the
frame stays bit-identical, and downstream the tinted frame is treated as "the photo" (so the bit-exact car-pixel
rules still hold from there on). Approach (a) of docs/glass-cutout-camera-options.md section 4.

    mask = window_mask(client, image_bgr, car_alpha)       # bool HxW, glass of THIS car, inside the eroded car mask
    tinted = tint_glass(image_bgr, mask)                    # new uint8 BGR; outside `mask` bit-identical
    pct = glass_pct(mask, car_alpha)                        # results.csv
    save_original(photo_path, out_dir / "originals")        # byte-for-byte copy of the input file

SAM 3 on fal (schema read 2026-11-23 from https://fal.ai/models/fal-ai/sam-3/image/llms.txt):
- endpoint `fal-ai/sam-3/image`, ~$0.005 per call (hosted.fal.COST).
- input: `image_url` (required), `prompt` (text, default "wheel"), `point_prompts`, `box_prompts`, `apply_mask`
  (default true: mask drawn on the image), `sync_mode` (data URIs), `output_format` png|jpeg|webp (default png),
  `return_multiple_masks` (default false), `max_masks` (1-32, default 3), `include_scores`, `include_boxes`.
- output: `masks` (required, list of Image objects `{url, content_type, file_name, file_size, width, height}`, one
  per instance), optional `image` (preview), `metadata`, `scores` (list of float), `boxes` (normalised [cx, cy, w, h]).
We send apply_mask=false + return_multiple_masks=true and read every `masks[i].url`; the files may come back at the
sent (downscaled) size, as grey, RGB or RGBA PNGs, so `_decode_mask` copes with each and resizes to the frame.
Two prompts per frame ("car window", "windshield"), instances unioned: ~$0.01 per frame.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from .fal import FalClient

SAM3_ENDPOINT = "fal-ai/sam-3/image"
PROMPTS = ("car window", "windshield")
MAX_MASKS = 8  # instances per prompt (fal allows 1-32)

ERODE_FRAC = 0.004  # car mask eroded by this share of the frame's long side: the tint never touches the body edge
MIN_COMPONENT_FRAC = 0.0005  # glass components smaller than this share of the car area are dropped (0.05%)

# Tint look (constants, not measured from the image: every frame of a car gets the same glass).
TINT_TOP = 58.0  # grey level at the top of the glass bounding box (slightly lighter)
TINT_BOTTOM = 10.0  # grey level at the bottom (near black)
TINT_BGR = (1.10, 1.0, 0.90)  # neutral blue-grey: per-channel factors on the grey level (B, G, R)
FEATHER_FRAC = 0.002  # feather width (inside the mask only) as a share of the frame's long side
MIN_FEATHER_PX = 2.0


def _as_bool(mask: np.ndarray) -> np.ndarray:
    """bool HxW from a bool mask, a float alpha (0-1, > 0.5) or a uint8 mask (> 127)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == bool:
        return m
    if np.issubdtype(m.dtype, np.floating):
        return m > 0.5
    return m > (np.iinfo(m.dtype).max // 2)


def _decode_mask(img: np.ndarray | None, h: int, w: int) -> np.ndarray:
    """A SAM 3 mask file (grey / BGR / BGRA, any bit depth, any size) -> bool HxW at the frame size."""
    if img is None or img.size == 0:
        return np.zeros((h, w), bool)
    if img.ndim == 3:
        if img.shape[2] == 4 and img[..., 3].min() < img[..., 3].max():
            img = img[..., 3]  # transparency carries the mask
        else:
            img = img[..., :3].max(axis=2)
    img = img.astype(np.float32) / float(np.iinfo(img.dtype).max if img.dtype.kind in "ui" else 1.0)
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img > 0.5


def _instance_urls(result: dict) -> list[str]:
    """URLs of every per-instance mask in a SAM 3 response (`masks: [{url, ...}]`); [] when there are none."""
    urls = []
    for m in (result or {}).get("masks") or []:
        url = m.get("url") if isinstance(m, dict) else m if isinstance(m, str) else None
        if url:
            urls.append(url)
    return urls


def window_mask(client: FalClient, image: np.ndarray, car_mask: np.ndarray, max_side: int = 2048) -> np.ndarray:
    """bool HxW: the glass of THIS car. SAM 3 prompts PROMPTS (instances unioned), resized to full size, kept only
    inside the car mask eroded by ERODE_FRAC of the long side, components < MIN_COMPONENT_FRAC of the car area
    dropped. An empty response gives an all-False mask. FalError / BudgetExceeded propagate to the caller."""
    h, w = image.shape[:2]
    car = _as_bool(car_mask)
    if car.shape != (h, w):
        raise ValueError(f"car mask {car.shape} does not match the image {(h, w)}")
    glass = np.zeros((h, w), bool)
    if not car.any():
        return glass
    payload = {"apply_mask": False, "return_multiple_masks": True, "max_masks": MAX_MASKS, "output_format": "png"}
    for prompt in PROMPTS:
        result = client.run(SAM3_ENDPOINT, {**payload, "prompt": prompt}, images={"image_url": image},
                            max_side=max_side)
        for url in _instance_urls(result):
            glass |= _decode_mask(client.fetch_image(url), h, w)
    if not glass.any():
        return glass
    r = max(1, round(ERODE_FRAC * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(car.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    glass &= inner
    min_area = MIN_COMPONENT_FRAC * int(car.sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(glass.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def tint_glass(image: np.ndarray, mask: np.ndarray, strength: float = 0.85, keep: float = 0.15) -> np.ndarray:
    """New uint8 BGR image: inside `mask` the glass becomes keep*orig + (1-keep)*gradient (a fixed dark blue-grey
    vertical gradient over the mask's bounding box, TINT_TOP -> TINT_BOTTOM), blended in with weight
    strength * feather, where feather ramps 0->1 over FEATHER_FRAC of the long side measured INSIDE the mask
    (distance transform). Pixels outside `mask` are bit-identical; deterministic; the input is not modified."""
    if not (0.0 <= strength <= 1.0 and 0.0 <= keep <= 1.0):
        raise ValueError("strength and keep must be in 0-1")
    out = image.copy()
    m = _as_bool(mask)
    if m.shape != image.shape[:2]:
        raise ValueError(f"mask {m.shape} does not match the image {image.shape[:2]}")
    if not m.any():
        return out
    ys, xs = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max())
    t = (ys - y0).astype(np.float32) / max(1, y1 - y0)
    level = TINT_TOP + (TINT_BOTTOM - TINT_TOP) * t  # per masked pixel
    grad = level[:, None] * np.asarray(TINT_BGR, np.float32)[None, :]
    orig = image[ys, xs].astype(np.float32)
    tinted = keep * orig + (1.0 - keep) * grad
    feather = max(MIN_FEATHER_PX, FEATHER_FRAC * max(image.shape[:2]))
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)[ys, xs]  # >= 1 on every mask pixel
    wgt = (strength * np.clip(dist / feather, 0.0, 1.0)).astype(np.float32)[:, None]
    out[ys, xs] = np.clip(np.rint(orig + wgt * (tinted - orig)), 0, 255).astype(np.uint8)
    return out


def glass_pct(mask: np.ndarray, car_mask: np.ndarray) -> float:
    """Share of the car's area that is glass, in % (0.0 for an empty car mask)."""
    car = _as_bool(car_mask)
    n = int(car.sum())
    return 0.0 if n == 0 else 100.0 * int((_as_bool(mask) & car).sum()) / n


def save_original(src_path: Path, dst_dir: Path) -> Path:
    """Copy the untouched input file byte-for-byte (shutil.copy2) into dst_dir; returns the copy's path. Refuses
    (ValueError) when dst_dir resolves to the source's own folder."""
    src = Path(src_path)
    dst_dir = Path(dst_dir)
    if dst_dir.resolve() == src.resolve().parent:
        raise ValueError(f"originals folder {dst_dir} is the photo's own folder; refusing to copy over it")
    dst_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(src, dst_dir / src.name))


__all__ = ["glass_pct", "save_original", "tint_glass", "window_mask"]
