"""Video -> plate-masked -> new background -> video.

Stages (each skipped if its output is already complete, so a run can be resumed):
  1. extract frames        (OpenCV)
  2. mask number plates    (number_plate_detection/mask_plates.py)
  3. change the background (background_changer/change_background.py, room mode)
  4. assemble the video    (frames in order; frames the changer left out repeat the previous result)

Settings come from config.toml. Run with the background_changer venv:
  ..\\background_changer\\.venv\\Scripts\\python make_video.py
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PLATES = REPO / "number_plate_detection"
BGCHANGER = REPO / "background_changer"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config() -> dict:
    path = ROOT / "config.toml"
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("video", "car", "background"):
        if key not in cfg:
            raise SystemExit(f"'{key}' missing in {path}")
    cfg["video"] = (ROOT / cfg["video"]).resolve()
    cfg["output"] = (ROOT / cfg.get("output", "output")).resolve()
    cfg.setdefault("fps", 5)
    cfg.setdefault("output_fps", "source")
    cfg.setdefault("max_frames", 0)
    cfg.setdefault("max_seconds", 0)
    cfg.setdefault("brightness_smoothing", 5)
    return cfg


def images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def extract(video: Path, out: Path, fps: float, max_frames: int, max_seconds: float = 0) -> tuple[int, float]:
    """Write sampled frames as 00001.jpg ...; return (count, source fps)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 if fps <= 0 or fps >= src_fps else src_fps / fps
    out.mkdir(parents=True, exist_ok=True)
    n, idx, next_take = 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (max_seconds and idx / src_fps >= max_seconds):
            break
        if idx >= next_take - 1e-6:
            n += 1
            cv2.imwrite(str(out / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            next_take += step
            if max_frames and n >= max_frames:
                break
        idx += 1
    cap.release()
    return n, src_fps


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(str(c) for c in cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"stage failed: {Path(cmd[1]).name}")


def py(module: Path) -> str:
    exe = module / ".venv" / "Scripts" / "python.exe"
    return str(exe if exe.exists() else sys.executable)


def fit_to_canvas(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox a finished composite onto the video canvas (a plain resize, no other change)."""
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas


def main() -> None:
    cfg = load_config()
    video: Path = cfg["video"]
    work = cfg["output"] / video.stem
    frames_dir = work / "frames"
    masked_root = work / "masked"
    bg_root = work / "background"
    out_video = work / f"{video.stem}.mp4"
    marker = work / "frames.done"

    # 1. extract
    frames = images(frames_dir)
    if frames and marker.exists():
        n_marked, src_fps = marker.read_text().split()
        src_fps = float(src_fps)
        print(f"[1/4] frames: {len(frames)} already extracted")
    else:
        for old in frames:
            old.unlink()
        n, src_fps = extract(video, frames_dir, cfg["fps"], cfg["max_frames"], cfg["max_seconds"])
        work.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{n} {src_fps}")
        frames = images(frames_dir)
        print(f"[1/4] extracted {n} frames (source {src_fps:.2f} fps)")
    if not frames:
        raise SystemExit("no frames extracted")

    # 2. mask plates -> masked/frames/
    masked_dir = masked_root / frames_dir.name
    if len(images(masked_dir)) == len(frames):
        print("[2/4] plates: already masked")
    else:
        run([py(PLATES), str(PLATES / "mask_plates.py"), str(frames_dir), "--out", str(masked_root),
             "--overlay", "--auto-brightness", "--brightness-smoothing", str(cfg["brightness_smoothing"])])

    # 3. background -> background/frames/<background>/
    result_dir = bg_root / masked_dir.name / cfg["background"]
    if {p.stem for p in frames} <= {p.stem for p in images(result_dir)}:
        print("[3/4] background: already changed")
    else:
        run([py(BGCHANGER), str(BGCHANGER / "change_background.py"), str(masked_dir),
             "--out", str(bg_root), "--car", cfg["car"], "--background", cfg["background"],
             "--background-view", "room"])

    # 4. assemble
    results = {p.stem: p for p in images(result_dir)}
    ordered: list[Path | None] = []
    last, filled = None, 0
    for f in frames:
        img = results.get(f.stem)
        if img is not None:
            last = img
        elif last is not None:
            filled += 1
        ordered.append(last)
    first = next((p for p in ordered if p is not None), None)
    if first is None:
        raise SystemExit("the background changer produced no frames; see its results.csv")
    filled += sum(p is None for p in ordered)
    ordered = [p or first for p in ordered]  # leading gaps take the first result

    ref = cv2.imread(str(first))
    h, w = ref.shape[:2]
    w, h = w - w % 2, h - h % 2
    if cfg["output_fps"] == "source":
        out_fps = src_fps if cfg["fps"] <= 0 else min(float(cfg["fps"]), src_fps)
    else:
        out_fps = float(cfg["output_fps"])
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    for p in ordered:
        writer.write(fit_to_canvas(cv2.imread(str(p)), w, h))
    writer.release()
    print(f"[4/4] wrote {out_video}  ({len(ordered)} frames, {out_fps:g} fps, {w}x{h}, "
          f"{filled} gap(s) filled from a neighbouring frame)")


if __name__ == "__main__":
    main()
