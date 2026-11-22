"""Find number plates in car images and mask them (CPU only).

Takes one image or a folder of images, detects plates with a small YOLOv9 ONNX model, finds
each plate's own 4 corners, and covers the plate exactly: with a sample image warped onto the
corners (overlay = true) or a solid colour. Saves the masked copies plus a detections.csv in
<output>/<input name>/. Images without a plate are saved unchanged.

Settings come from config.toml; command-line options override them.

Usage (PowerShell, from the number_plate_detection folder):
  .venv\\Scripts\\python mask_plates.py
  .venv\\Scripts\\python mask_plates.py D:\\photos\\car.jpg --no-overlay --conf 0.15
"""

from __future__ import annotations

import argparse
import csv
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np

from plate_brightness_detector import BrightnessDetector
from plate_masker import (
    DEFAULT_COLOR_RGB,
    DEFAULT_CONF,
    DEFAULT_LIGHTING,
    DEFAULT_MODEL,
    DEFAULT_PAD,
    DEFAULT_THREADS,
    PlateMasker,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_OVERLAY_IMAGE = ROOT / "assets" / "sample_plate.png"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_KEYS = ("input", "output", "overlay_image")
BOOL_KEYS = ("tight_fit", "overlay", "auto_brightness")
CONFIG_KEYS = (*PATH_KEYS, *BOOL_KEYS, "lighting", "brightness_smoothing", "model", "confidence", "padding",
               "threads", "color")


def load_config(path: Path, required: bool) -> dict:
    """Read config.toml. Relative paths in it are resolved against the config's folder."""
    if not path.exists():
        if required:
            raise SystemExit(f"config file not found: {path}")
        return {}
    try:
        # utf-8-sig: tolerate the BOM that some Windows editors and PowerShell add.
        cfg = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        raise SystemExit(f"cannot read {path}: save it as UTF-8 text") from None
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(
            f"cannot read {path}: {exc}\n"
            "Hint: put Windows paths in single quotes, e.g. input = 'D:\\photos\\car.jpg'"
        ) from None
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        hint = "\n'style' was replaced by 'overlay = true/false'" if "style" in unknown else ""
        raise SystemExit(
            f"unknown setting(s) in {path}: {', '.join(unknown)}\n"
            f"Allowed settings: {', '.join(CONFIG_KEYS)}{hint}"
        )
    for key in (*PATH_KEYS, "model"):
        if key in cfg and not isinstance(cfg[key], str):
            raise SystemExit(f"'{key}' in {path} must be text in quotes")
    for key in BOOL_KEYS:
        if key in cfg and not isinstance(cfg[key], bool):
            raise SystemExit(f"'{key}' in {path} must be true or false (no quotes)")
    for key in PATH_KEYS:
        if key in cfg:
            cfg[key] = path.parent / cfg[key]
    # A local .onnx model path is relative to the config folder too; a model name is left as is.
    if cfg.get("model", "").lower().endswith(".onnx"):
        cfg["model"] = str(path.parent / cfg["model"])
    for key in ("confidence", "padding", "lighting"):
        value = cfg.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise SystemExit(f"'{key}' in {path} must be a number")
    threads = cfg.get("threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 0):
        raise SystemExit(f"'threads' in {path} must be a whole number, 0 or more (0 = all cores)")
    smoothing = cfg.get("brightness_smoothing")
    if smoothing is not None and (isinstance(smoothing, bool) or not isinstance(smoothing, int) or smoothing < 0):
        raise SystemExit(f"'brightness_smoothing' in {path} must be a whole number, 0 or more (0 = off)")
    if "color" in cfg:
        cfg["color"] = check_color(cfg["color"], f"'color' in {path}")
    return cfg


def is_color(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in value)
    )


def check_color(value, where: str) -> tuple[int, int, int]:
    if not is_color(value):
        raise SystemExit(f"{where} must be three numbers 0-255 (red, green, blue), e.g. [0, 0, 0]")
    return tuple(value)


def parse_color(text: str) -> tuple[int, int, int]:
    try:
        parts = [int(p) for p in text.split(",")]
    except ValueError:
        parts = []
    if not is_color(parts):
        raise argparse.ArgumentTypeError("use three numbers 0-255 for red,green,blue, e.g. 0,0,0")
    return tuple(parts)


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"input not found: {source}")


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    # imdecode instead of imread: imread cannot open non-ASCII paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def load_overlay(path: Path) -> np.ndarray:
    """Overlay image as BGR, or BGRA if it has transparency."""
    if not path.is_file():
        raise SystemExit(f"overlay image not found: {path}\nSet 'overlay_image' in config.toml, or overlay = false")
    image = read_image(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"cannot read overlay image: {path}")
    if image.dtype != np.uint8:  # e.g. 16-bit PNG
        image = cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise SystemExit(f"cannot encode image as {ext}: {path}")
    buf.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find number plates in car images and mask them. Defaults come from config.toml."
    )
    parser.add_argument("input", type=Path, nargs="?", default=None, help="image or folder (overrides 'input')")
    parser.add_argument("--out", type=Path, default=None, help="base output folder (overrides 'output')")
    parser.add_argument("--model", default=None, help="model name or .onnx path (overrides 'model')")
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold 0-1 (overrides 'confidence')")
    parser.add_argument("--pad", type=float, default=None, help="extra margin beyond the plate (overrides 'padding')")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads, 0 = all cores (overrides 'threads')")
    parser.add_argument("--tight-fit", action=argparse.BooleanOptionalAction, default=None,
                        help="follow each plate's own corners (overrides 'tight_fit')")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=None,
                        help="cover plates with the overlay image instead of a colour (overrides 'overlay')")
    parser.add_argument("--overlay-image", type=Path, default=None, help="overlay image file (overrides 'overlay_image')")
    parser.add_argument("--lighting", type=float, default=None,
                        help="0-100: match the overlay to the plate's lighting (overrides 'lighting')")
    parser.add_argument("--auto-brightness", action=argparse.BooleanOptionalAction, default=None,
                        help="pick each plate's brightness with plate_brightness_detector (overrides 'auto_brightness')")
    parser.add_argument("--brightness-smoothing", type=int, default=None,
                        help="video: smooth over this many frames, 0 = off (overrides 'brightness_smoothing')")
    parser.add_argument("--color", type=parse_color, default=None, help="fill colour R,G,B (overrides 'color')")
    parser.add_argument("--config", type=Path, default=None, help=f"config file (default: {DEFAULT_CONFIG.name})")
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG, required=args.config is not None)
    pick = lambda arg, key, default: arg if arg is not None else cfg.get(key, default)  # noqa: E731
    source = pick(args.input, "input", None)
    if source is None:
        parser.error("no input given: set 'input' in config.toml or pass an image/folder")
    model = pick(args.model, "model", DEFAULT_MODEL)
    conf = pick(args.conf, "confidence", DEFAULT_CONF)
    pad = pick(args.pad, "padding", DEFAULT_PAD)
    threads = pick(args.threads, "threads", DEFAULT_THREADS)
    tight_fit = pick(args.tight_fit, "tight_fit", True)
    use_overlay = pick(args.overlay, "overlay", False)
    overlay_path = pick(args.overlay_image, "overlay_image", DEFAULT_OVERLAY_IMAGE)
    lighting = pick(args.lighting, "lighting", DEFAULT_LIGHTING)
    auto_brightness = pick(args.auto_brightness, "auto_brightness", False)
    smoothing = pick(args.brightness_smoothing, "brightness_smoothing", 0)
    color = pick(args.color, "color", DEFAULT_COLOR_RGB)
    if not 0 < conf < 1:
        raise SystemExit("confidence must be between 0 and 1")
    if pad < 0:
        raise SystemExit("padding must be 0 or more")
    if threads < 0:
        raise SystemExit("threads must be 0 or more (0 = all cores)")
    if not 0 <= lighting <= 100:
        raise SystemExit("lighting must be between 0 and 100 (%)")
    if smoothing < 0:
        raise SystemExit("brightness_smoothing must be 0 or more (0 = off)")
    overlay = load_overlay(overlay_path) if use_overlay else None
    # Automatic brightness only matters for the overlay (a solid fill keeps its exact colour).
    brightness = BrightnessDetector(smoothing=smoothing) if (use_overlay and auto_brightness) else None

    images = list_images(source)
    if not images:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_EXTS))}) in {source}")
    name = source.stem if source.is_file() else source.resolve().name
    out_dir = pick(args.out, "output", DEFAULT_OUTPUT) / name
    if out_dir.resolve() == images[0].parent.resolve():
        raise SystemExit(f"output folder is the input folder; refusing to overwrite the originals: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    try:
        masker = PlateMasker(
            model, conf, pad, threads, tight_fit=tight_fit, overlay=overlay, lighting=lighting,
            brightness_detector=brightness, color=color,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"model {model} loaded in {time.perf_counter() - t_load:.1f}s (CPU, {threads or 'all'} threads)")
    if brightness is not None:
        light_how = "brightness auto (plate_brightness_detector"
        light_how += f", smoothed over {smoothing} frames)" if smoothing else ")"
    else:
        light_how = f"lighting {lighting:g}%"
    if not use_overlay:
        how = "solid fill"
    elif tight_fit:
        how = f"overlay {overlay_path.name}, {light_how} (solid fill where a plate's corners aren't found)"
    else:
        how = f"overlay {overlay_path.name}, {light_how}"
    print(f"mask: {how}, {'tight fit to plate corners' if tight_fit else 'detector box'}, padding {pad:g}", flush=True)

    rows = []
    t_total = 0.0
    t_start = time.perf_counter()
    n_done = n_with_plates = 0
    for path in images:
        image = read_image(path)
        if image is None:
            print(f"skip (cannot read image): {path.name}", flush=True)
            continue
        t0 = time.perf_counter()
        masked, plates = masker.process(image)
        t_image = time.perf_counter() - t0
        t_total += t_image
        write_image(out_dir / path.name, masked)
        n_done += 1
        n_with_plates += bool(plates)

        def describe(p):
            text = f"{p.confidence:.0%} {p.fit}"
            if p.brightness is not None and p.corners is not None:  # box plates get the solid fill
                text += f" light {p.brightness.percent:.0f}% ({p.brightness.source})"
            return text

        found = ", ".join(describe(p) for p in plates)
        print(f"{path.name:<36} {len(plates)} plate(s) {found:<26} {t_image * 1000:4.0f} ms", flush=True)
        empty = {"plate": "", "fit": "", "x1": "", "y1": "", "x2": "", "y2": "", "confidence": "", "corners": "",
                 "brightness_pct": "", "brightness_source": "", "plate_colours": ""}
        if not plates:
            rows.append({"file": path.name, "plates": 0, **empty})
        for i, p in enumerate(plates, 1):
            corners = " ".join(f"{x:.0f},{y:.0f}" for x, y in p.corners) if p.corners else ""
            b = p.brightness
            rows.append(
                {"file": path.name, "plates": len(plates), "plate": i, "fit": p.fit, "x1": p.x1, "y1": p.y1,
                 "x2": p.x2, "y2": p.y2, "confidence": round(p.confidence, 3), "corners": corners,
                 "brightness_pct": round(b.percent, 1) if b else "", "brightness_source": b.source if b else "",
                 "plate_colours": f"{b.background}/{b.characters}" if b else ""}
            )
    elapsed = time.perf_counter() - t_start

    if not n_done:
        raise SystemExit("no images could be read")
    # utf-8-sig so Excel shows non-ASCII file names correctly.
    with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{n_done} image(s), {n_with_plates} with a plate, in {elapsed:.1f}s "
        f"(find + fit + mask {t_total / n_done * 1000:.0f} ms per image, the rest is reading/writing files)"
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
