"""Classify which side of a car each sampled video frame shows.

Grabs one frame every N seconds, runs the view_classifier model on CPU, and
saves each frame as <time>s_<side>_<confidence>pct.jpg plus a results.csv
in <output>/<video name>/.

With pick_sides on (or --pick-sides), it also keeps the best frame for each of
the 8 sides in <output>/<video name>_sides/ (same file names, results.csv with
picked_for and picked_prob columns): the input for an 8-photo run downstream.

Settings come from config.toml; command-line options override them.

Usage (PowerShell, from the frame_view_type folder):
  .venv\\Scripts\\python classify_video.py
  .venv\\Scripts\\python classify_video.py other_car.mp4 --every 3
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "view_classifier"
sys.path.insert(0, str(VENDOR))

from view_classifier.labels import POSE_CLASSES, sincos_to_deg  # noqa: E402
from view_classifier.model import ConvNeXtViewClassifier  # noqa: E402
from view_classifier.transforms_cfv import eval_tensorize, square_reflect_crop  # noqa: E402

DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_CKPT = VENDOR / "models" / "joint_v2_best.pt"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_EVERY = 2.0
PATH_KEYS = ("input", "output", "checkpoint")
CONFIG_KEYS = (*PATH_KEYS, "every_seconds", "pick_sides", "pick_min_prob")
DEFAULT_PICK_MIN_PROB = 0.3
SIDES_SUFFIX = "_sides"

# Model label -> name used in output files.
SIDE_NAMES = {
    "front": "front",
    "front_right": "front-right",
    "right": "right",
    "rear_right": "back-right",
    "rear": "back",
    "rear_left": "back-left",
    "left": "left",
    "front_left": "front-left",
}


def load_config(path: Path, required: bool) -> dict:
    """Read config.toml. Relative paths in it are resolved against the config's folder."""
    if not path.exists():
        if required:
            raise SystemExit(f"config file not found: {path}")
        return {}
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(
            f"cannot read {path}: {exc}\n"
            "Hint: put Windows paths in single quotes, e.g. input = 'D:\\videos\\car.mp4'"
        ) from None
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit(
            f"unknown setting(s) in {path}: {', '.join(unknown)}\n"
            f"Allowed settings: {', '.join(CONFIG_KEYS)}"
        )
    for key in PATH_KEYS:
        if key in cfg:
            if not isinstance(cfg[key], str):
                raise SystemExit(f"'{key}' in {path} must be a text path in quotes")
            cfg[key] = path.parent / cfg[key]
    every = cfg.get("every_seconds")
    if every is not None and (isinstance(every, bool) or not isinstance(every, (int, float))):
        raise SystemExit(f"'every_seconds' in {path} must be a number, e.g. every_seconds = 2")
    if "pick_sides" in cfg and not isinstance(cfg["pick_sides"], bool):
        raise SystemExit(f"'pick_sides' in {path} must be true or false")
    pmin = cfg.get("pick_min_prob")
    if pmin is not None and (isinstance(pmin, bool) or not isinstance(pmin, (int, float)) or not 0 <= pmin < 1):
        raise SystemExit(f"'pick_min_prob' in {path} must be a number from 0 to below 1, e.g. pick_min_prob = 0.3")
    return cfg


def load_model(ckpt_path: Path) -> ConvNeXtViewClassifier:
    # A Git LFS pointer file is ~134 bytes; the real checkpoint is ~198 MB.
    if not ckpt_path.exists() or ckpt_path.stat().st_size < 1_000_000:
        raise SystemExit(
            f"checkpoint missing or still a Git LFS pointer: {ckpt_path}\n"
            "Run: git -C vendor/view_classifier lfs pull"
        )
    model = ConvNeXtViewClassifier(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def sample_frames(video: Path, every: float):
    """Yield (time_s, RGB image) for one frame every `every` seconds."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise SystemExit(f"video reports invalid fps ({fps}): {video}")
    try:
        # grab() every frame and decode only the sampled ones; more reliable than seeking.
        idx = 0
        k = 0
        while cap.grab():
            if idx >= round(k * every * fps):
                ok, bgr = cap.retrieve()
                if ok:
                    yield idx / fps, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                while round(k * every * fps) <= idx:
                    k += 1
            idx += 1
    finally:
        cap.release()


@torch.inference_mode()
def classify(model: ConvNeXtViewClassifier, tfm, image: Image.Image) -> dict:
    x = tfm(square_reflect_crop(image, bbox=None)).unsqueeze(0)
    logits, sincos = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    i = int(probs.argmax())
    return {
        "side": SIDE_NAMES[POSE_CLASSES[i]],
        "angle": sincos_to_deg(float(sincos[0, 0]), float(sincos[0, 1])),
        "conf": float(probs[i]),
        "probs": {SIDE_NAMES[name]: float(probs[j]) for j, name in enumerate(POSE_CLASSES)},
    }


def pick_sides(rows: list[dict], min_prob: float) -> tuple[dict[str, dict], list[str]]:
    """Best frame for each side: greedy over (prob_<side>, frame) pairs, highest first, each frame used once.

    Returns {side: row} and the sides no frame reaches min_prob for (not shown in the video). Those are never
    filled with a repeated or weak frame.
    """
    sides = list(SIDE_NAMES.values())
    pairs = sorted(
        ((r[f"prob_{side}"], -r["time_s"], side, i) for i, r in enumerate(rows) for side in sides),
        reverse=True,
    )
    picked: dict[str, dict] = {}
    used: set[int] = set()
    for prob, _, side, i in pairs:
        if prob < min_prob:
            break
        if side in picked or i in used:
            continue
        picked[side] = rows[i]
        used.add(i)
    return picked, [s for s in sides if s not in picked]


def write_sides(out_dir: Path, rows: list[dict], min_prob: float) -> Path:
    """Copy the picked frames (under their own names) and their rows to <out_dir>_sides/."""
    sides_dir = out_dir.with_name(out_dir.name + SIDES_SUFFIX)
    sides_dir.mkdir(exist_ok=True)
    for old in [*sides_dir.glob("*.jpg"), sides_dir / "results.csv", sides_dir / "missing_sides.txt"]:
        old.unlink(missing_ok=True)  # generated folder: a rerun must not leave stale picks behind
    picked, missing = pick_sides(rows, min_prob)
    chosen = sorted(
        ({**row, "picked_for": side, "picked_prob": row[f"prob_{side}"]} for side, row in picked.items()),
        key=lambda r: r["time_s"],
    )
    for row in chosen:
        shutil.copy2(out_dir / row["file"], sides_dir / row["file"])
    if chosen:
        with open(sides_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
            writer.writeheader()
            writer.writerows(chosen)
    if missing:
        (sides_dir / "missing_sides.txt").write_text(
            f"Sides no sampled frame shows (best prob_<side> below pick_min_prob {min_prob:g}):\n"
            + "\n".join(missing) + "\n",
            encoding="utf-8",
        )
    print(f"\npicked {len(chosen)} of 8 sides -> {sides_dir}")
    for side in SIDE_NAMES.values():
        row = picked.get(side)
        if row:
            print(f"  {side:<11} {row['file']}  (prob_{side} {row[f'prob_{side}']:.0%})")
        else:
            print(f"  {side:<11} MISSING: no frame with prob_{side} >= {min_prob:.0%}")
    return sides_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label car video frames by the side of the car shown. "
        "Settings come from config.toml; options given here override it."
    )
    parser.add_argument("video", type=Path, nargs="?", help="input video file (overrides 'input')")
    parser.add_argument("--every", type=float, help="seconds between sampled frames (overrides 'every_seconds')")
    parser.add_argument("--out", type=Path, help="output base folder (overrides 'output')")
    parser.add_argument("--ckpt", type=Path, help="model checkpoint (overrides 'checkpoint')")
    parser.add_argument(
        "--pick-sides", action="store_true", default=None,
        help="also keep the best frame per side in <output>/<video name>_sides/ (turns 'pick_sides' on)",
    )
    parser.add_argument("--config", type=Path, help=f"config file (default: {DEFAULT_CONFIG.name} next to this script)")
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG, required=args.config is not None)
    video = args.video or cfg.get("input")
    every = args.every if args.every is not None else cfg.get("every_seconds", DEFAULT_EVERY)
    out_base = args.out or cfg.get("output", DEFAULT_OUTPUT)
    ckpt = args.ckpt or cfg.get("checkpoint", DEFAULT_CKPT)
    want_sides = args.pick_sides if args.pick_sides is not None else cfg.get("pick_sides", False)
    min_prob = cfg.get("pick_min_prob", DEFAULT_PICK_MIN_PROB)

    if video is None:
        parser.error("no input video: set 'input' in config.toml or pass a video path")
    if not video.exists():
        raise SystemExit(f"input video not found: {video}")
    if every <= 0:
        raise SystemExit("every_seconds must be greater than 0")

    out_dir = out_base / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"input:  {video.resolve()}\noutput: {out_dir.resolve()}\nevery:  {every:g} s", flush=True)

    t_load = time.perf_counter()
    model = load_model(ckpt)
    tfm = eval_tensorize()
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s (CPU, {torch.get_num_threads()} threads)", flush=True)

    rows = []
    t_start = time.perf_counter()
    for t, image in sample_frames(video, every):
        result = classify(model, tfm, image)
        name = f"{t:06.1f}s_{result['side']}_{round(result['conf'] * 100)}pct.jpg"
        image.save(out_dir / name, quality=95)
        print(
            f"t={t:7.1f}s  {result['side']:<11} {result['angle']:5.0f} deg  {result['conf']:4.0%}  -> {name}",
            flush=True,
        )
        rows.append(
            {
                "file": name,
                "time_s": round(t, 2),
                "side": result["side"],
                "angle_deg": round(result["angle"], 1),
                "confidence": round(result["conf"], 3),
                **{f"prob_{side}": round(p, 3) for side, p in result["probs"].items()},
            }
        )
    elapsed = time.perf_counter() - t_start

    if not rows:
        raise SystemExit("no frames were read from the video")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["side"] for r in rows)
    print(f"\n{len(rows)} frames in {elapsed:.1f}s ({elapsed / len(rows):.2f}s per frame incl. decoding)")
    print("  " + ", ".join(f"{side}: {counts[side]}" for side in SIDE_NAMES.values() if counts[side]))
    print(f"saved to {out_dir}")
    if want_sides:
        write_sides(out_dir, rows, min_prob)


if __name__ == "__main__":
    main()
