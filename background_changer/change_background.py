"""Cut the car out of each photo and place it on the chosen background(s) like a real car (CPU only).

Everything is set in config.toml: the images folder, the backgrounds, and which car it is (its
dimensions come from cars.json). Then run, from the background_changer folder:

    .venv\\Scripts\\python change_background.py

Room mode (the default, background_view = "room"): every photo goes through car_fit_to_bg. Its camera
(roll, pitch, lens) is measured from the photo's own background lines and the car's pose from its
outline; the background is then re-rendered as a 3D room (room3d) from that same camera, so the car
stands on the floor at any camera angle. The output keeps the photo's own framing; a car cut off by the
photo edge keeps its cut edge on the output border. Runs in three stages: cut out every photo and
measure its lines, pool the lens over the folder, then fit and place each photo.

Flat mode (background_view = "flat"): today's placement through each background's scene code
(bg_changer/scenes): the car is sized and moved onto the stage from its dimensions and the side of the car
the photo shows (the view label in the file name, e.g. 0012.0s_right_95pct.jpg). When the photos are too
small for a full-size background, background_scaler shrinks the background (config: background_scale).

Results: <output_folder>/<images folder name>/<background name>/<photo>.webp, transparent car
cut-outs in .../cutouts/, and a results.csv with the placement and any flags.

The car's own pixels: at most one rigid rotation that levels the photo (room mode), then one downscale;
no colour change, retouching or generated content. Shadows, reflections and the room go on the background.
Command-line options override config.toml for one run (see --help).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

VENV_HINT = (
    "Run it with this folder's own Python, from the background_changer folder:\n"
    "    .venv\\Scripts\\python change_background.py\n"
    "First time only (creates .venv): see 'Setup' in README.md."
)

try:
    import argparse
    import csv
    import gc
    import time
    import tomllib
    import warnings
    from pathlib import Path

    import cv2
    import numpy as np

    from background_scaler import DEFAULT_MIN_WIDTH, BackgroundScaler, crop_to_car, needed_scale
    from background_scaler import DEFAULT_MODE as DEFAULT_SCALE_MODE
    from background_scaler import MODES as SCALE_MODES
    from bg_changer import (CUT_POLICIES, DEFAULT_EDGE_SHRINK, DEFAULT_MODEL, DEFAULT_SHADOW, DEFAULT_THREADS,
                            CarMask, CarProbe, CarSegmenter, CarSpecError, CarTable, NameInfo, NoCarFound, Scene,
                            YawEstimate, cutout_bgra, estimate_yaw, load_cars, load_scene, parse_name, place_car,
                            probe_car, select_car)
    from bg_changer.placement import skip_reason
    from bg_changer.scenes import SCENES, generic, scene_key
    from bg_changer.scenes.base import with_overrides
    from bg_changer.yaw import BIN_DEG
    from car_fit_to_bg import FitError, place_in_room
    from car_fit_to_bg.batch import (Measured, choose_focals, find_view_results, fit_measured, folder_height_prior,
                                     measure, read_view_results)
    from car_fit_to_bg.batch import skip_reason as room_skip_reason
    from car_fit_to_bg.lines import LineCamera
    from car_fit_to_bg.rooms import GENERIC_BACKDROP_ROWS, room_for
    from debug_images import (DebugWriter, cutout_pictures, fit_pictures, flat_picture, glass_pictures,
                              hosted_camera_picture, room_pictures)
except ImportError as exc:  # usually: started with the system Python instead of .venv
    sys.exit(f"Missing package: {exc.name or exc}.\n{VENV_HINT}")

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_BACKGROUNDS = ROOT / "backgrounds"
DEFAULT_CARS = ROOT / "cars.json"
DEFAULT_QUALITY = 90
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_KEYS = ("images_folder", "backgrounds_folder", "output_folder", "cars_file")
CONFIG_KEYS = (*PATH_KEYS, "background", "car", "cut_frames", "debug_mode", "model", "threads", "shadow",
               "edge_shrink", "quality", "save_cutouts", "background_scale", "background_min_width", "scene",
               "background_view", "max_output_size", "keep_background_sharp", "fit", "car_isolation", "hosted")
ISOLATION_KEYS = ("enabled", "min_score", "remove_objects")
ISOLATION_CSV_FIELDS = ["isolation", "isolate_s"]  # appended only when car_isolation or [hosted] is on
# [hosted]: Idea B, hosted open models on fal.ai (hosted/, docs/glass-cutout-camera-options.md set-up B)
HOSTED_BOOL_KEYS = ("enabled", "cutout", "glass_tint", "camera", "vggt", "require_masked_plates", "blur_people")
HOSTED_KEYS = (*HOSTED_BOOL_KEYS, "tint_strength", "tint_keep", "max_cost_per_car", "timeout_s", "max_side",
               "keep_cache_days")
HOSTED_DEFAULTS = {"enabled": False, "cutout": True, "glass_tint": True, "camera": True, "vggt": True,
                   "require_masked_plates": True, "blur_people": True,
                   "tint_strength": 0.85, "tint_keep": 0.15, "max_cost_per_car": 3.0, "timeout_s": 120,
                   "max_side": 2048, "keep_cache_days": 30}
HOSTED_CSV_FIELDS = ["hosted", "sam3_status", "glass_pct", "camera_source", "moge_f", "moge_h", "vggt_ok",
                     "cost_usd"]  # appended only when [hosted] is on
MIN_WIDTH_FLOOR = 64  # smallest allowed background_min_width
VIEW_MODES = ("room", "flat")
ROOM_CUT_POLICIES = ("frame", "skip")
FIT_KEYS = ("focal", "on_fail")
ON_FAIL = ("place", "skip")
DEFAULT_MAX_OUTPUT = 1920
YAW_CHANGE_LISTED = 24.0  # deg: a fitted view angle this far from the label is listed (flag yaw-vs-label)
QUIET_FLAGS = {"flip-ambiguous", "fit-slow"}  # kept in results.csv, left out of the console line
RENAMED = {
    "input": "'images_folder'",
    "output": "'output_folder'",
    "backgrounds": "'backgrounds_folder' (the folder) and 'background' (which one)",
    "layout": "per-background scene code (bg_changer/scenes); fine-tune with [scene.<name>] "
              "(anchor, size, side_fill, shadow, reflection)",
    "debug_overlay": "'debug_mode' (true = a debug picture for every step, in <output>\\debug\\<photo>\\)",
}
SCENE_KEYS = ("anchor", "size", "side_fill", "shadow", "reflection", "horizon_y", "floor")
CSV_FIELDS = ["file", "background", "scene", "out_w", "out_h", "bg_scale", "view_label", "view_conf", "yaw",
              "yaw_source", "scale", "px_per_m",
              "anchor_x", "anchor_y", "src_h", "target_h", "target_low_y", "car_x1", "car_y1", "car_x2", "car_y2",
              "out_x1", "out_y1", "out_x2", "out_y2", "hw_ratio", "margin", "cut_off", "flags", "segment_s", "note"]
ROOM_CSV_FIELDS = [*CSV_FIELDS,
              "view_mode", "frame_w", "frame_h", "crop_x1", "crop_y1", "out_scale", "fit_iou", "fit_margin",
              "yaw_change", "roll", "roll_source", "car_roll", "pitch", "focal_px", "focal_source", "body_source",
              "cam_dist_m", "cam_height_m", "h_source", "elev_deg", "level_deg", "cut_crop_px", "cut_crop_pct",
              "tyres_found", "lock_mode", "contact_residual_px", "invented_pct", "invented_near_pct", "floor_texels",
              "fit_s", "render_s"]


def is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def load_config(path: Path, required: bool = True) -> dict:
    """Read and check config.toml. Relative paths are resolved against the config's folder.
    'scene' comes back as {lower-case background name: settings}."""
    if not path.exists():
        if required:
            raise SystemExit(f"config file not found: {path}")
        return {"scene": {}, "fit": {}, "car_isolation": {}, "hosted": dict(HOSTED_DEFAULTS)}
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(
            f"cannot read {path}: {exc}\n"
            "Hint: put Windows paths in single quotes, e.g. images_folder = 'D:\\photos\\car_123'"
        ) from None
    for old, new in RENAMED.items():
        if old in cfg:
            raise SystemExit(f"'{old}' in {path} has been replaced: use {new} instead (see config.toml comments)")
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit(f"unknown setting(s) in {path}: {', '.join(unknown)}\nAllowed settings: {', '.join(CONFIG_KEYS)}")
    for key in (*PATH_KEYS, "model", "car", "cut_frames", "background_scale", "background_view"):
        if key in cfg and not isinstance(cfg[key], str):
            raise SystemExit(f"'{key}' in {path} must be text in quotes")
    for key in PATH_KEYS:
        if key in cfg:
            cfg[key] = path.parent / cfg[key]
    for key in ("shadow", "edge_shrink", "quality", "threads"):
        if key in cfg and not is_number(cfg[key]):
            raise SystemExit(f"'{key}' in {path} must be a number")
    for key in ("save_cutouts", "debug_mode"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise SystemExit(f"'{key}' in {path} must be true or false")
    all_cut = tuple(dict.fromkeys((*ROOM_CUT_POLICIES, *CUT_POLICIES)))
    if "cut_frames" in cfg and cfg["cut_frames"] not in all_cut:
        raise SystemExit(f"'cut_frames' in {path} must be one of: {', '.join(all_cut)}")
    if "background_view" in cfg and cfg["background_view"] not in VIEW_MODES:
        raise SystemExit(f"'background_view' in {path} must be one of: {', '.join(VIEW_MODES)}")
    if "max_output_size" in cfg:
        v = cfg["max_output_size"]
        if isinstance(v, bool) or not isinstance(v, int) or (v != 0 and v < MIN_WIDTH_FLOOR):
            raise SystemExit(f"'max_output_size' in {path} must be a whole number of pixels, {MIN_WIDTH_FLOOR} or more "
                             "(0 = no limit)")
    if "keep_background_sharp" in cfg and not isinstance(cfg["keep_background_sharp"], bool):
        raise SystemExit(f"'keep_background_sharp' in {path} must be true or false")
    cfg["fit"] = parse_fit(cfg.get("fit", {}), path)
    cfg["car_isolation"] = parse_isolation(cfg.get("car_isolation", {}), path)
    cfg["hosted"] = parse_hosted(cfg.get("hosted", {}), path)
    if "background_scale" in cfg and cfg["background_scale"] not in SCALE_MODES:
        raise SystemExit(f"'background_scale' in {path} must be one of: {', '.join(SCALE_MODES)}")
    if "background_min_width" in cfg:
        v = cfg["background_min_width"]
        if isinstance(v, bool) or not isinstance(v, int) or v < MIN_WIDTH_FLOOR:
            raise SystemExit(f"'background_min_width' in {path} must be a whole number of pixels, "
                             f"{MIN_WIDTH_FLOOR} or more")
    if "background" in cfg:
        cfg["background"] = check_background_choice(cfg["background"], f"'background' in {path}")
    cfg["scene"] = parse_scene_overrides(cfg.get("scene", {}), path)
    return cfg


def parse_fit(table, path: Path) -> dict:
    """[fit] focal = "auto" | px;  on_fail = "place" | "skip"."""
    if not isinstance(table, dict):
        raise SystemExit(f"'fit' in {path} must be a [fit] block")
    unknown = sorted(set(table) - set(FIT_KEYS))
    if unknown:
        raise SystemExit(f"[fit] in {path}: unknown setting(s) {', '.join(unknown)}; allowed: {', '.join(FIT_KEYS)}")
    out = dict(table)
    if "focal" in out:
        f = out["focal"]
        if not (f == "auto" or (is_number(f) and f > 0)):
            raise SystemExit(f"[fit] in {path}: 'focal' must be \"auto\" or the lens's focal length in pixels at the "
                             "photo's size")
    if "on_fail" in out and out["on_fail"] not in ON_FAIL:
        raise SystemExit(f"[fit] in {path}: 'on_fail' must be one of: {', '.join(ON_FAIL)}")
    return out


def parse_isolation(table, path: Path) -> dict:
    """[car_isolation] enabled = bool; min_score = 0..1; remove_objects = bool."""
    where = f"[car_isolation] in {path}"
    if not isinstance(table, dict):
        raise SystemExit(f"'car_isolation' in {path} must be a [car_isolation] block")
    unknown = sorted(set(table) - set(ISOLATION_KEYS))
    if unknown:
        raise SystemExit(f"{where}: unknown setting(s) {', '.join(unknown)}; allowed: {', '.join(ISOLATION_KEYS)}")
    for key in ("enabled", "remove_objects"):
        if key in table and not isinstance(table[key], bool):
            raise SystemExit(f"{where}: '{key}' must be true or false")
    if "min_score" in table and not (is_number(table["min_score"]) and 0 < table["min_score"] < 1):
        raise SystemExit(f"{where}: 'min_score' must be a number between 0 and 1")
    return dict(table)


def parse_hosted(table, path: Path) -> dict:
    """[hosted] enabled/cutout/glass_tint/camera/vggt/require_masked_plates/blur_people = bool; tint_strength,
    tint_keep = 0..1; max_cost_per_car >= 0 (0 = no limit); timeout_s > 0; max_side = 0 or >= 256 px;
    keep_cache_days >= 0 (0 = keep forever). Returns the full settings
    (defaults filled in)."""
    where = f"[hosted] in {path}"
    if not isinstance(table, dict):
        raise SystemExit(f"'hosted' in {path} must be a [hosted] block")
    unknown = sorted(set(table) - set(HOSTED_KEYS))
    if unknown:
        raise SystemExit(f"{where}: unknown setting(s) {', '.join(unknown)}; allowed: {', '.join(HOSTED_KEYS)}")
    for key in HOSTED_BOOL_KEYS:
        if key in table and not isinstance(table[key], bool):
            raise SystemExit(f"{where}: '{key}' must be true or false")
    for key in ("tint_strength", "tint_keep"):
        if key in table and not (is_number(table[key]) and 0 <= table[key] <= 1):
            raise SystemExit(f"{where}: '{key}' must be a number from 0 to 1")
    if "max_cost_per_car" in table and not (is_number(table["max_cost_per_car"]) and table["max_cost_per_car"] >= 0):
        raise SystemExit(f"{where}: 'max_cost_per_car' must be a number of US dollars, 0 or more (0 = no limit)")
    if "timeout_s" in table and not (is_number(table["timeout_s"]) and table["timeout_s"] > 0):
        raise SystemExit(f"{where}: 'timeout_s' must be a number of seconds above 0")
    if "max_side" in table:
        v = table["max_side"]
        if isinstance(v, bool) or not isinstance(v, int) or (v != 0 and v < 256):
            raise SystemExit(f"{where}: 'max_side' must be a whole number of pixels, 256 or more (0 = full size)")
    if "keep_cache_days" in table and not (is_number(table["keep_cache_days"]) and table["keep_cache_days"] >= 0):
        raise SystemExit(f"{where}: 'keep_cache_days' must be a number of days, 0 or more (0 = keep forever)")
    return {**HOSTED_DEFAULTS, **table}


def plates_masked(input_dir: Path, hosted_cfg: dict) -> bool:
    """[hosted] plate gate: photos may go to fal.ai only once number_plate_detection has covered their plates,
    which leaves a detections.csv in the folder it writes (or when require_masked_plates = false)."""
    if not hosted_cfg.get("require_masked_plates", True):
        return True
    return (input_dir / "detections.csv").is_file()


def check_background_choice(value, where: str) -> list[str]:
    """"name" | "all" | ["name", ...]  ->  list of names."""
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise SystemExit(f'{where} must be a name in quotes, a list like ["background_1", "background_3"], or "all"')
    return [n.strip() for n in names]


def parse_scene_overrides(table, path: Path) -> dict[str, dict]:
    if not isinstance(table, dict):
        raise SystemExit(f"'scene' in {path} must be [scene.<name>] blocks")
    out = {}
    for name, entry in table.items():
        where = f"[scene.{name}] in {path}"
        if not isinstance(entry, dict):
            raise SystemExit(f"{where} must be a block of settings")
        unknown = sorted(set(entry) - set(SCENE_KEYS))
        if unknown:
            raise SystemExit(f"{where}: unknown setting(s) {', '.join(unknown)}; allowed: {', '.join(SCENE_KEYS)}")
        if "anchor" in entry:
            a = entry["anchor"]
            if not (isinstance(a, list) and len(a) == 2 and all(is_number(v) for v in a)):
                raise SystemExit(f"{where}: 'anchor' must be [x, y] in pixels")
        for key, lo, hi in (("size", 0.2, 3.0), ("side_fill", 0.1, 1.0), ("shadow", 0.0, 3.0)):
            if key in entry and not (is_number(entry[key]) and lo <= entry[key] <= hi):
                raise SystemExit(f"{where}: '{key}' must be a number from {lo} to {hi}")
        if "horizon_y" in entry and not is_number(entry["horizon_y"]):
            raise SystemExit(f"{where}: 'horizon_y' must be a number (pixel row)")
        if "reflection" in entry and not isinstance(entry["reflection"], bool):
            raise SystemExit(f"{where}: 'reflection' must be true or false")
        if "floor" in entry and entry["floor"] not in ("matte", "glossy"):
            raise SystemExit(f"{where}: 'floor' must be \"matte\" or \"glossy\"")
        out[scene_key(name)] = dict(entry)
    return out


def list_backgrounds(folder: Path) -> dict[str, Path]:
    """Background images in `folder`, by lower-case file name without extension."""
    if not folder.is_dir():
        raise SystemExit(f"backgrounds folder not found: {folder}")
    found: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            key = p.stem.lower()
            if key in found:
                raise SystemExit(f"two backgrounds share the name '{p.stem}' ({found[key].name}, {p.name}); rename one")
            found[key] = p
    if not found:
        raise SystemExit(f"no background images ({', '.join(sorted(IMAGE_EXTS))}) in {folder}")
    return found


def select_backgrounds(choice: list[str], available: dict[str, Path]) -> list[Path]:
    """Names (with or without extension, any case) or "all" -> background files, in the given order."""
    if any(c.lower() == "all" for c in choice):
        return list(available.values())
    picked, missing = [], []
    for name in choice:
        key = name.lower()
        if Path(key).suffix in IMAGE_EXTS:
            key = Path(key).stem
        if key not in available:
            missing.append(name)
        elif available[key] not in picked:
            picked.append(available[key])
    if missing:
        names = ", ".join(p.stem for p in available.values())
        raise SystemExit(f"background(s) not found: {', '.join(missing)}\nAvailable: {names} (or \"all\")")
    return picked


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"images folder not found: {source}")


def read_image(path: Path) -> np.ndarray | None:
    # imdecode instead of imread: imread cannot open non-ASCII paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, image: np.ndarray, quality: int = DEFAULT_QUALITY) -> None:
    """quality 1-100 for lossy WEBP/JPEG; above 100 means lossless WEBP."""
    ext = path.suffix.lower()
    if ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, min(quality, 100)]
    else:
        params = []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise SystemExit(f"cannot encode image as {ext}: {path}")
    buf.tofile(str(path))


SCALE_TEXT = {
    "folder": "one size for the whole folder, set by its smallest car",
    "photo": "a size per photo, set by its own car",
    "off": "always full size",
}


@dataclass
class Photo:
    """A cut-out photo, ready to place. `image` and `car` hold only the car's part of the photo."""

    path: Path
    image: np.ndarray
    car: CarMask
    box: tuple[int, int, int, int]  # where the car is in the whole photo (for results.csv)
    info: NameInfo | None
    probe: CarProbe
    yaw: YawEstimate
    t_seg: float
    frame_size: tuple[int, int] = (0, 0)  # room mode: the whole photo's (width, height)
    measured: Measured | None = None  # room mode: its line camera (stage 1)
    frame_path: Path | None = None  # [hosted] glass tint: the tinted frame stage 3 re-reads instead of `path`
    hosted: dict | None = None  # [hosted]: HOSTED_CSV_FIELDS values


@dataclass
class Run:
    """What every photo is placed with."""

    loaded: list[tuple[str, np.ndarray, Scene]]  # (background name, full-size picture, its scene)
    scaler: BackgroundScaler
    cut_policy: str
    shadow: float
    out_dir: Path
    quality: int
    debug: bool


def prepare_photo(path: Path, image: np.ndarray, car: CarMask, table: CarTable, t_seg: float) -> Photo:
    box = car.box
    image, car = crop_to_car(image, car)
    info = parse_name(path.stem)
    probe = probe_car(car)
    return Photo(path, image, car, box, info, probe, estimate_yaw(info, probe, table), t_seg)


def results_fields(view: str, isolate: bool, hosted: bool) -> list[str]:
    """results.csv columns. [hosted] isolates too (SAM 3, with [car_isolation] itself off), so its rows carry the
    isolation columns as well; without them csv.DictWriter raises at the very end and the CSV is lost."""
    return ((ROOM_CSV_FIELDS if view == "room" else CSV_FIELDS)
            + (ISOLATION_CSV_FIELDS if isolate or hosted else [])
            + (HOSTED_CSV_FIELDS if hosted else []))


def isolation_cols(photo: Photo) -> dict:
    """results.csv columns of the optional car_isolation step (empty when it is off)."""
    iso = photo.car.isolation
    return {"isolation": iso["text"], "isolate_s": iso["seconds"]} if iso else {}


def precompute_raw(segmenter: CarSegmenter, images: list[Path]) -> dict:
    """car_isolation's first pass: the cut-out model's raw alpha for every photo, kept as 8 bits where that is
    exact (rembg's mask is 8-bit), with its seconds. {path: (alpha, seconds)}"""
    out = {}
    print(f"first pass: cutting out {len(images)} photo(s) before the isolation models load", flush=True)
    for i, path in enumerate(images, 1):
        try:
            image = read_image(path)
        except OSError:
            image = None
        if image is None:
            continue
        t0 = time.perf_counter()
        raw = segmenter.raw_alpha(image)
        q = np.round(raw * 255).astype(np.uint8)
        if np.abs(q.astype(np.float32) / 255.0 - raw).max() < 1e-6:
            raw = q
        out[path] = (raw, time.perf_counter() - t0)
        print(f"  [{i}/{len(images)}] {path.name:<40} {out[path][1]:5.1f}s", flush=True)
    return out


def debug_step(draw, *args) -> None:
    """Write debug pictures (debug_mode); one that fails is reported and skipped, it never stops the run."""
    try:
        draw(*args)
    except Exception as exc:  # noqa: BLE001 - debug pictures only
        print(f"warning: debug pictures ({draw.__name__}) failed: {type(exc).__name__}: {exc}", flush=True)


def iso_text(photo: Photo) -> str:
    iso = photo.car.isolation
    text = f"  [isolation: {iso['text']}, {iso['seconds']:.1f}s]" if iso else ""
    if photo.hosted:
        h = photo.hosted
        text += f"  [hosted: cut-out {h['hosted']}, SAM 3 {h['sam3_status'] or '-'}, glass {h['glass_pct']}%]"
    return text


def describe(photo: Photo, cut_policy: str) -> str:
    info, yaw, probe = photo.info, photo.yaw, photo.probe
    text = f"{info.label} -> {yaw.deg:.0f} deg" if info else f"no label -> {yaw.deg:.0f} deg (from shape)"
    if probe.cut:
        text += f"  CUT OFF: {','.join(probe.cut)}"
    reason = skip_reason(probe, cut_policy)
    return text + (f"  ({reason.split(':')[0]})" if reason else "")


def place_photo(photo: Photo, run: Run, scales: dict[str, float]) -> tuple[list[dict], list[int]]:
    """Place one photo on every background, each shrunk to about scales[name] of full size, and write
    the images. Returns its results.csv rows and, per background it went on, that background's size
    in % of full size."""
    rows, pcts = [], []
    info, probe, yaw = photo.info, photo.probe, photo.yaw
    cut = ",".join(probe.cut)
    for stem, plate, scene in run.loaded:
        scene_k, plate_k = run.scaler.apply(scene, plate, scales[stem])
        out, plan = place_car(photo.image, photo.car, plate_k, scene_k, probe, yaw, run.cut_policy, run.shadow)
        base = {"file": photo.path.name, "background": stem, "scene": scene.name,
                "view_label": info.label if info else "", "view_conf": info.conf if info else "",
                "yaw": round(yaw.deg, 1), "yaw_source": yaw.source, "car_x1": photo.box[0], "car_y1": photo.box[1],
                "car_x2": photo.box[2], "car_y2": photo.box[3], "cut_off": cut, "segment_s": round(photo.t_seg, 2),
                **isolation_cols(photo), **(photo.hosted or {})}
        if plan is None:
            rows.append({**base, "note": skip_reason(probe, run.cut_policy) or "not placed"})
            continue
        w, h = scene_k.size_wh
        pcts.append(round(100 * w / scene.size_wh[0]))
        write_image(run.out_dir / stem / f"{photo.path.stem}.webp", out, run.quality)
        if run.debug:
            write_image(run.out_dir / stem / "debug" / f"{photo.path.stem}.jpg", scene_k.debug_overlay(out, plan, probe), 85)
        rows.append({**base, "out_w": w, "out_h": h, "bg_scale": round(w / scene.size_wh[0], 4),
                     "scale": round(plan.scale, 4), "px_per_m": round(plan.px_per_m, 1),
                     "anchor_x": round(plan.anchor[0], 1), "anchor_y": round(plan.anchor[1], 1), "src_h": probe.h,
                     "target_h": round(plan.target_h, 1), "target_low_y": round(plan.target_low, 1),
                     "out_x1": plan.box_out[0], "out_y1": plan.box_out[1], "out_x2": plan.box_out[2],
                     "out_y2": plan.box_out[3], "hw_ratio": "" if plan.hw_ratio is None else round(plan.hw_ratio, 3),
                     "margin": plan.margin, "flags": ";".join(plan.flags),
                     "note": f"car runs off the photo ({cut}); part of it is missing" if cut else ""})
    return rows, pcts


def shrink_text(pcts: list[int]) -> str:
    if not pcts or min(pcts) == 100:
        return ""
    lo, hi = min(pcts), max(pcts)
    return f"  background at {lo}%" if lo == hi else f"  backgrounds at {lo}-{hi}%"


def folder_scales(photos: list[Photo], run: Run) -> dict[str, float]:
    """'folder' mode: one background size per background for all the photos, set by the smallest car."""
    scales = {}
    print()
    for stem, _, scene in run.loaded:
        needs = [needed_scale(scene, p.probe, p.yaw, run.cut_policy) for p in photos]
        k = run.scaler.folder_scale(needs)
        scales[stem] = k
        w, h = run.scaler.size(scene, k)
        if (w, h) == scene.size_wh:
            print(f"{stem}: full size, {w}x{h}")
            continue
        by = photos[needs.index(k)].path.name
        line = f"{stem}: shrunk to {w}x{h} ({100 * w / scene.size_wh[0]:.0f}%), so the smallest car ({by}) fills the stage"
        if w > round(k * scene.size_wh[0]) + 1:
            line = (f"{stem}: shrunk to {w}x{h}, the background_min_width; photos with a smaller car "
                    f"({by} needs {100 * k:.0f}%) are placed smaller than the stage")
        print(line)
    return scales


# ---------------------------------------------------------------- room mode (car_fit_to_bg + room3d)

@dataclass
class RoomRun:
    """Room-mode settings."""

    shape: object  # the car's CarShape
    focal: float | None  # [fit] focal in px, or None = from the background lines
    on_fail: str
    max_output_size: int
    min_texels: float  # 1.0 = the background is never magnified at the stage; 0 = only max_output_size limits
    cut_policy: str  # frame | skip
    view_rows: dict | None  # frame_view_type's classifier results for these photos
    overrides: dict  # [scene.<name>] settings
    hosted: object = None  # hosted.pipeline.HostedRun when [hosted] camera is on (between stages 1 and 2)


def room_spec(stem: str, plate: np.ndarray, overrides: dict):
    """The background's scene spec with the config's [scene] overrides, but without the flat path's auto-fit
    shrink: in room mode the car never has to fit inside the background picture's frame."""
    key = scene_key(stem)
    h, w = plate.shape[:2]
    if key in SCENES:
        return with_overrides(SCENES[key], overrides.get(key))
    return generic.make_spec(stem, w, h, overrides.get(key))


def full_frame_alpha(photo: Photo) -> tuple[np.ndarray, tuple[int, int]]:
    """The car's alpha on the whole photo, rebuilt from the kept crop, and the crop's top-left."""
    W, H = photo.frame_size
    ox, oy = photo.box[0] - photo.car.box[0], photo.box[1] - photo.car.box[1]
    alpha = np.zeros((H, W), np.float32)
    h, w = photo.car.alpha.shape
    alpha[oy:oy + h, ox:ox + w] = photo.car.alpha
    return alpha, (ox, oy)


def base_row(photo: Photo) -> dict:
    info = photo.info
    return {"file": photo.path.name, "view_label": info.label if info else "", "view_conf": info.conf if info else "",
            "car_x1": photo.box[0], "car_y1": photo.box[1], "car_x2": photo.box[2], "car_y2": photo.box[3],
            "cut_off": ",".join(photo.probe.cut), "segment_s": round(photo.t_seg, 2), "view_mode": "room",
            **isolation_cols(photo), **(photo.hosted or {})}


def fit_summary(photo: Photo, fit) -> dict:
    """What every results.csv row of a photo needs from its fit, whatever the background: kept per photo instead
    of the whole fit (which holds the car's full-resolution pixels)."""
    p = fit.pose
    flags = list(fit.flags)
    yaw_change = ""
    if photo.info is not None and photo.info.label in BIN_DEG:
        yaw_change = round(abs((p.yaw - BIN_DEG[photo.info.label] + 180) % 360 - 180), 1)
        if yaw_change > YAW_CHANGE_LISTED:
            flags.append("yaw-vs-label")
    return {"yaw_deg": float(p.yaw), "yaw_source": p.yaw_source, "fit_iou": round(p.iou, 3),
            "fit_margin": round(p.margin, 1) if p.margin < 90 else "", "yaw_change": yaw_change, "flags": flags}


def room_row(photo: Photo, stem: str, scene_name: str, fit, summary: dict, placed, crop_xy, h_source: str) -> dict:
    p, mt = fit.pose, placed.metrics
    flags = list(dict.fromkeys(summary["flags"] + placed.flags))
    roll_source = (fit.cam.source if fit.cam.source in ("neighbours", "hosted")
                   else ("lines" if fit.cam.strong else "silhouette"))
    crop = max((v["crop_px"] for v in fit.cut_edges.values()), default=0.0)
    crop_pct = max((v["crop_pct"] for v in fit.cut_edges.values()), default=0.0)
    cut = ",".join(photo.probe.cut)
    measured = bool(fit.lock.get("res_after") or fit.lock.get("res_before"))  # no tyre found: nothing to miss by
    return {**base_row(photo), "background": stem, "scene": scene_name, "out_w": mt["out_w"], "out_h": mt["out_h"],
            "yaw": round(p.yaw, 1), "yaw_source": summary["yaw_source"], "flags": ";".join(flags),
            "frame_w": fit.frame_size[0], "frame_h": fit.frame_size[1], "crop_x1": crop_xy[0], "crop_y1": crop_xy[1],
            "out_scale": mt["out_scale"], "fit_iou": summary["fit_iou"], "fit_margin": summary["fit_margin"],
            "yaw_change": summary["yaw_change"],
            "roll": round(p.roll, 2), "roll_source": roll_source, "car_roll": round(fit.car_roll, 2),
            "pitch": round(p.pitch, 2), "focal_px": round(fit.f, 1), "focal_source": fit.focal_source,
            "body_source": fit.body_source, "cam_dist_m": round(p.D, 2), "cam_height_m": round(p.h, 2),
            "h_source": h_source, "elev_deg": round(p.delta, 1), "level_deg": round(fit.theta, 2),
            "cut_crop_px": round(crop, 1) if cut else "", "cut_crop_pct": round(crop_pct, 2) if cut else "",
            "tyres_found": sum(b.status == "found" for b in fit.bottoms), "lock_mode": fit.lock["mode"],
            "contact_residual_px": mt["contact_residual_px"] if measured else "", "invented_pct": mt["invented_pct"],
            "invented_near_pct": mt["invented_near_pct"], "floor_texels": mt["floor_texels"],
            "fit_s": round(fit.seconds, 1), "render_s": mt["render_s"],
            "note": f"car runs off the photo ({cut}); framed like the photo" if cut else "",
            **({"camera_source": roll_source} if photo.hosted is not None else {})}


def measure_photo(path: Path, image: np.ndarray, alpha: np.ndarray) -> Measured:
    """Stage 1 of room mode: the photo's line camera at a free focal (no lines on error: fitted without them)."""
    try:
        return measure(path, image, alpha)
    except Exception as exc:  # noqa: BLE001 - one photo's line trouble must not stop the run
        print(f"warning: {path.name}: background lines not measured ({type(exc).__name__}: {exc})", flush=True)
        h, w = alpha.shape
        return Measured(path=path, stem=path.stem, size=(w, h), cam=LineCamera(source="none", flags=["lines-error"]),
                        exif_f=None, time_s=None, flags=["lines-error"])


def describe_room(photo: Photo, cut_policy: str) -> str:
    info, cam = photo.info, photo.measured.cam
    text = info.label if info else "no view label"
    text += (f", lines: roll {cam.roll:+.1f} pitch {cam.pitch:.0f} deg" if cam.strong and cam.pitch is not None
             else ", few background lines")
    if photo.probe.cut:
        text += f"  CUT OFF: {','.join(photo.probe.cut)}"
        reason = room_skip_reason(photo.probe.cut) or ("cut_frames = skip" if cut_policy == "skip" else "")
        text += f" ({reason.split(':')[0]})" if reason else " (framed like the photo)"
    return text


FOCAL_TEXT = {"hosted": "the hosted camera models (VGGT / MoGe-3)", "config": "[fit] focal", "pooled": "the folder's background lines", "lines": "its own lines",
              "exif": "EXIF", "lines-weak": "uncertain lines", "prior": "a typical phone lens (no lines)"}


def place_room_photos(photos: list[Photo], run: Run, rr: RoomRun, rows: list[dict]) -> None:
    """Stages 2 and 3 of room mode: pool the lens over the folder, then fit each photo and place it on every
    background, appending to rows as it goes. Backgrounds without a room (no module, no usable horizon_y) get
    the flat placement, with the fitted view angle."""
    if not photos:
        return
    items = [p.measured for p in photos]
    if rr.hosted is not None:  # [hosted] camera: MoGe-3 per frame + VGGT per folder -> Measured.external
        yaw_prior = rr.hosted.camera(photos, read_image, full_frame_alpha)
        if yaw_prior:  # VGGT's angles between shots, anchored on the labels: the view prior for the fit
            rows_vp = dict(rr.view_rows or {})
            for stem, probs in yaw_prior.items():
                rows_vp[stem] = {**rows_vp.get(stem, {"angle": float("nan"), "time": None}), "probs": probs}
            rr.view_rows = rows_vp
            print(f"view prior: VGGT's angles between shots, anchored on the view labels ({len(yaw_prior)} photos)")
    choose_focals(items, rr.focal)
    if rr.hosted is not None:
        rr.hosted.apply_focals(items, rr.focal)
    specs = {stem: room_spec(stem, plate, rr.overrides) for stem, plate, _ in run.loaded}
    rooms = {stem: room_for(stem, specs[stem], plate.shape) for stem, plate, _ in run.loaded}
    flat_jobs: dict[str, list] = {stem: [] for stem, _, _ in run.loaded}
    # Photos with strong background lines first: their camera heights help photos with weak lines.
    order = sorted(range(len(photos)), key=lambda i: (0 if photos[i].measured.cam.strong else 1, i))
    heights = []  # what folder_height_prior reads from each fit (not the fits themselves: they hold the car's pixels)
    lenses: dict[tuple[int, str], int] = {}
    for m in items:
        lenses[(round(m.focal), m.focal_source)] = lenses.get((round(m.focal), m.focal_source), 0) + 1
    groups = sorted(lenses.items(), key=lambda kv: -kv[1])
    print("\nlens: " + ", ".join(f"{f} px from {FOCAL_TEXT.get(src, src)} ({k} photo{'s' * (k > 1)})"
                                 for (f, src), k in groups[:4])
          + (f", {len(groups) - 4} other focal lengths" if len(groups) > 4 else ""))
    print("fitting each car in 3D and placing it (photos with clear background lines first):", flush=True)
    for n, i in enumerate(order, 1):
        photo = photos[i]
        m = photo.measured
        cut = {side for side, v in photo.car.cut.items() if v}
        reason = room_skip_reason(cut)
        if not reason and cut and rr.cut_policy == "skip":
            reason = "car runs off the photo (cut_frames = skip)"
        head = f"[{n}/{len(photos)}] {photo.path.name:<40}"
        if reason:
            rows.append({**base_row(photo), "note": f"skipped: {reason}"})
            print(f"{head} skipped ({reason})", flush=True)
            continue
        try:
            frame = read_image(photo.frame_path or photo.path)
        except OSError:  # moved or deleted since it was cut out
            frame = None
        if frame is None or (frame.shape[1], frame.shape[0]) != photo.frame_size:
            rows.append({**base_row(photo), "note": "skipped: cannot re-read the photo"})
            print(f"{head} skipped (cannot re-read the photo)", flush=True)
            continue
        alpha, crop_xy = full_frame_alpha(photo)
        hp = folder_height_prior(heights)
        try:
            fit = fit_measured(m, frame, alpha, cut, rr.shape, items, rr.view_rows, video=True, h_prior=hp)
        except FitError as exc:
            rows.append({**base_row(photo), "note": f"skipped: {exc}"})
            print(f"{head} skipped ({exc})", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001 - one photo's numeric trouble must not stop the run
            rows.append({**base_row(photo), "note": f"fit error: {type(exc).__name__}: {exc}"})
            print(f"{head} fit error ({type(exc).__name__}: {exc})", flush=True)
            continue
        heights.append(SimpleNamespace(pose=SimpleNamespace(h=fit.pose.h, failed=fit.pose.failed, iou=fit.pose.iou),
                                       cam=SimpleNamespace(strong=fit.cam.strong)))
        summary = fit_summary(photo, fit)
        # Strong lines: the height is fitted freely; weak lines: pulled toward the folder's (or a default) height.
        h_source = "fit" if fit.cam.strong else ("folder" if hp else "default")
        if fit.pose.failed and rr.on_fail == "skip":
            rows.append({**base_row(photo), "note": "skipped: the 3D fit failed ([fit] on_fail = skip)",
                         "fit_iou": summary["fit_iou"], "flags": ";".join(summary["flags"])})
            print(f"{head} skipped (fit failed)", flush=True)
            continue
        dw = DebugWriter(run.out_dir / "debug") if run.debug else None
        if dw is not None:
            info = photo.info
            if rr.hosted is not None:
                debug_step(hosted_camera_picture, dw, photo.path.stem, frame, alpha, m.external, fit, photo.hosted)
            debug_step(fit_pictures, dw, photo.path.stem, frame, alpha, fit, info.label if info else None,
                       BIN_DEG.get(info.label) if info else None)
        del frame, alpha
        for stem, plate, scene in run.loaded:
            room, is_generic = rooms[stem]
            if room is None:
                flat_jobs[stem].append((photo, summary))
                continue
            try:
                placed = place_in_room(fit, scene_key(stem), plate, specs[stem], run.shadow, rr.max_output_size,
                                       run.debug, rr.min_texels, room=room, generic=is_generic)
            except Exception as exc:  # noqa: BLE001
                rows.append({**base_row(photo), "background": stem, "note": f"place error: {type(exc).__name__}: {exc}"})
                print(f"{head} {stem}: place error ({type(exc).__name__}: {exc})", flush=True)
                continue
            if "camera-outside-room" in placed.flags and rr.on_fail == "skip":
                rows.append({**base_row(photo), "background": stem, "flags": "camera-outside-room",
                             "note": "skipped: the photo's camera would be outside this room ([fit] on_fail = skip)"})
                continue
            write_image(run.out_dir / stem / f"{photo.path.stem}.webp", placed.image, run.quality)
            if dw is not None:
                debug_step(room_pictures, dw, photo.path.stem, stem, placed)
            rows.append(room_row(photo, stem, scene.name, fit, summary, placed, crop_xy, h_source))
        flags = [fl for fl in summary["flags"] if fl not in QUIET_FLAGS]
        print(f"{head} {fit.pose.yaw:5.0f} deg (fit {fit.pose.iou:.2f}), camera {fit.pose.D:.1f} m away "
              f"{fit.pose.delta:.0f} deg down, levelled {fit.theta:+.1f} deg, {fit.seconds:.0f}s"
              + (f"  [{', '.join(flags)}]" if flags else ""), flush=True)
        del fit
    for stem, plate, scene in run.loaded:
        if flat_jobs[stem]:
            place_flat_fallback(flat_jobs[stem], stem, plate, scene, run, rows, no_room_hint(specs[stem]))


def place_flat_fallback(jobs, stem: str, plate: np.ndarray, scene: Scene, run: Run, rows: list[dict],
                        hint: str) -> None:
    """A background without a room in room mode: today's flat placement, with the fitted view angle.
    jobs: (photo, fit_summary) pairs. Rows are appended as each image is written."""
    yaws = [YawEstimate(summary["yaw_deg"] % 360.0, 10.0, "fit") for _, summary in jobs]

    def row(photo, summary, yaw, **extra):
        flags = list(dict.fromkeys(["no-room", *summary["flags"], *extra.pop("plan_flags", [])]))
        return {**base_row(photo), "background": stem, "scene": scene.name, "yaw": round(yaw.deg, 1),
                "yaw_source": summary["yaw_source"], "fit_iou": summary["fit_iou"], "fit_margin": summary["fit_margin"],
                "yaw_change": summary["yaw_change"], "view_mode": "flat (no room)", "flags": ";".join(flags), **extra}

    try:
        if run.scaler.mode == "folder":
            k = run.scaler.folder_scale([needed_scale(scene, ph.probe, y, "anchor") for (ph, _), y in zip(jobs, yaws)])
            scales = [k] * len(jobs)
        else:
            scales = [run.scaler.photo_scale(scene, ph.probe, y, "anchor") for (ph, _), y in zip(jobs, yaws)]
    except Exception as exc:  # noqa: BLE001
        for (photo, summary), yaw in zip(jobs, yaws):
            rows.append(row(photo, summary, yaw, note=f"place error: {type(exc).__name__}: {exc}"))
        return
    for (photo, summary), yaw, k in zip(jobs, yaws, scales):
        try:
            scene_k, plate_k = run.scaler.apply(scene, plate, k)
            out, plan = place_car(photo.image, photo.car, plate_k, scene_k, photo.probe, yaw, "anchor", run.shadow)
            if plan is None:
                rows.append(row(photo, summary, yaw, note="not placed"))
                continue
            w, h = scene_k.size_wh
            write_image(run.out_dir / stem / f"{photo.path.stem}.webp", out, run.quality)
            if run.debug:
                debug_step(flat_picture, DebugWriter(run.out_dir / "debug"), photo.path.stem, stem,
                           scene_k.debug_overlay(out, plan, photo.probe), plan)
        except Exception as exc:  # noqa: BLE001
            rows.append(row(photo, summary, yaw, note=f"place error: {type(exc).__name__}: {exc}"))
            print(f"{photo.path.name} {stem}: place error ({type(exc).__name__}: {exc})", flush=True)
            continue
        rows.append(row(photo, summary, yaw, plan_flags=plan.flags, out_w=w, out_h=h,
                        bg_scale=round(w / scene.size_wh[0], 4), scale=round(plan.scale, 4),
                        px_per_m=round(plan.px_per_m, 1), note=hint))


def no_room_hint(spec) -> str:
    """Why a background has no room in room mode, and what would give it one."""
    if spec.horizon_y is None:
        return "no 3D room for this background: set horizon_y in [scene.<name>] to get one"
    return (f"no 3D room for this background: its horizon_y ({spec.horizon_y:g}) must be more than "
            f"{GENERIC_BACKDROP_ROWS:g} px above the stage row ({spec.anchor[1]:g})")


def room_info(stem: str, plate: np.ndarray, overrides: dict) -> tuple[bool, str]:
    """(has a room?, how the background is used in room mode)."""
    spec = room_spec(stem, plate, overrides)
    room, is_generic = room_for(stem, spec, plate.shape)
    if room is None:
        return False, "no 3D room: placed flat with the fitted view angle; " + no_room_hint(spec).split(": ", 1)[1]
    return True, ("generic 3D room from its horizon_y: a rough model" if is_generic else "measured 3D room")


def file_order(rows: list[dict], images: list[Path], loaded) -> None:
    """Sort results.csv rows by photo, then by background (in place)."""
    rank = {p.name: i for i, p in enumerate(images)}
    bg_rank = {stem: j for j, (stem, _, _) in enumerate(loaded)}
    rows.sort(key=lambda r: (rank.get(r.get("file", ""), len(rank)), bg_rank.get(r.get("background", ""), -1)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut the car out and place it on background(s) like a real car. All settings come from "
        "config.toml; these options override it for one run."
    )
    parser.add_argument("images", type=Path, nargs="?", default=None, help="images folder or one image (overrides 'images_folder')")
    parser.add_argument("--background", action="append", default=None, metavar="NAME",
                        help='background name, or "all"; repeat for several (overrides \'background\')')
    parser.add_argument("--backgrounds-folder", type=Path, default=None, help="overrides 'backgrounds_folder'")
    parser.add_argument("--out", type=Path, default=None, help="overrides 'output_folder'")
    parser.add_argument("--car", default=None, help="carModel from the car file (overrides 'car')")
    parser.add_argument("--cut-frames", choices=tuple(dict.fromkeys((*ROOM_CUT_POLICIES, *CUT_POLICIES))), default=None,
                        help="overrides 'cut_frames'")
    parser.add_argument("--background-view", choices=VIEW_MODES, default=None,
                        help="room = the background re-rendered as a 3D room from each photo's camera; flat = the "
                             "scene-code placement (overrides 'background_view')")
    parser.add_argument("--max-output-size", type=int, default=None, metavar="PX",
                        help="room mode: cap on the output's long side, 0 = none (overrides 'max_output_size')")
    parser.add_argument("--model", default=None, help="overrides 'model'")
    parser.add_argument("--threads", type=int, default=None, help="overrides 'threads'")
    parser.add_argument("--shadow", type=float, default=None, help="shadow strength, 1 = as calibrated (overrides 'shadow')")
    parser.add_argument("--quality", type=int, default=None, help="WEBP quality 1-100 (overrides 'quality')")
    parser.add_argument("--background-scale", choices=SCALE_MODES, default=None,
                        help="shrink the background to the photos' resolution: per folder, per photo, or off "
                             "(overrides 'background_scale')")
    parser.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=None,
                        help="remove other vehicles/objects stuck to the car's outline (overrides [car_isolation] enabled)")
    parser.add_argument("--hosted", action=argparse.BooleanOptionalAction, default=None,
                        help="hosted models on fal.ai: cut-out, glass tint, camera (overrides [hosted] enabled)")
    parser.add_argument("--debug", action="store_true",
                        help="also write a debug picture for every step, in <output>\\debug\\<photo>\\ "
                             "(turns on 'debug_mode' for this run)")
    parser.add_argument("--config", type=Path, default=None, help=f"config file (default: {DEFAULT_CONFIG.name})")
    args = parser.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG, required=args.config is not None)
    source = args.images or cfg.get("images_folder")
    if source is None:
        parser.error("no images given: set 'images_folder' in config.toml or pass a folder/image")
    choice = check_background_choice(args.background, "--background") if args.background else cfg.get("background")
    if not choice:
        parser.error("no background chosen: set 'background' in config.toml (a name, a list, or \"all\")")
    model = args.model or cfg.get("model", DEFAULT_MODEL)
    threads = args.threads if args.threads is not None else cfg.get("threads", DEFAULT_THREADS)
    shadow = args.shadow if args.shadow is not None else cfg.get("shadow", DEFAULT_SHADOW)
    edge_shrink = cfg.get("edge_shrink", DEFAULT_EDGE_SHRINK)
    quality = args.quality if args.quality is not None else cfg.get("quality", DEFAULT_QUALITY)
    save_cutouts = cfg.get("save_cutouts", True)
    view = args.background_view or cfg.get("background_view", "room")
    cut_setting = args.cut_frames or cfg.get("cut_frames")
    if view == "room":  # cut-off cars are framed like the photo, or left out
        cut_policy = "skip" if cut_setting == "skip" else "frame"
    else:  # "frame" is room mode's name for keeping the car where it stands
        cut_policy = "anchor" if cut_setting in (None, "frame") else cut_setting
    max_output = args.max_output_size if args.max_output_size is not None else cfg.get("max_output_size",
                                                                                       DEFAULT_MAX_OUTPUT)
    if not isinstance(max_output, int) or (max_output != 0 and max_output < MIN_WIDTH_FLOOR):
        raise SystemExit(f"max_output_size must be a whole number of pixels, {MIN_WIDTH_FLOOR} or more (0 = no limit)")
    keep_sharp = cfg.get("keep_background_sharp", True)
    fit_cfg = cfg.get("fit", {})
    iso_cfg = cfg.get("car_isolation", {})
    isolate = args.isolate if args.isolate is not None else iso_cfg.get("enabled", False)
    hosted_cfg = cfg.get("hosted", HOSTED_DEFAULTS)
    hosted = args.hosted if args.hosted is not None else hosted_cfg.get("enabled", False)
    if hosted and source is not None:
        src_dir = (source if source.is_dir() else source.parent).resolve()
        if not plates_masked(src_dir, hosted_cfg):
            print(f"[hosted] is OFF for this run: {src_dir} has no detections.csv, so its number plates were not "
                  "covered by number_plate_detection (mask_plates.py) and the photos must not be sent to fal.ai. "
                  "Run mask_plates.py first and use its output folder, or set [hosted] require_masked_plates = "
                  "false if these photos hold no plates. Continuing with the laptop-only path.")
            hosted = False
    if hosted and isolate:
        print("note: [hosted] is on, so SAM 3 keeps only this car; [car_isolation] is not used in this run")
        isolate = False
    debug = args.debug or cfg.get("debug_mode", False)
    scaler = BackgroundScaler(args.background_scale or cfg.get("background_scale", DEFAULT_SCALE_MODE),
                              cfg.get("background_min_width", DEFAULT_MIN_WIDTH))
    if not 0 <= shadow <= 1.5:
        raise SystemExit("shadow must be between 0 (none) and 1.5 (1 = as calibrated)")
    if not (isinstance(edge_shrink, int) and edge_shrink >= 0):
        raise SystemExit("edge_shrink must be a whole number, 0 or more")
    if not (isinstance(quality, int) and 1 <= quality <= 100):
        raise SystemExit("quality must be a whole number from 1 to 100")
    if not (isinstance(threads, int) and threads >= 0):
        raise SystemExit("threads must be a whole number, 0 or more (0 = automatic)")

    car_name = args.car or cfg.get("car")
    cars_file = cfg.get("cars_file", DEFAULT_CARS)
    try:
        cars = load_cars(cars_file)
        if car_name is None:
            raise CarSpecError(f"no car chosen: set car = \"<carModel>\" in config.toml. "
                               f"In {cars_file.name}: {', '.join(c.name for c in cars.values())}")
        shape = select_car(cars, car_name)
    except CarSpecError as exc:
        raise SystemExit(str(exc)) from None
    table = CarTable(shape)

    bg_folder = args.backgrounds_folder or cfg.get("backgrounds_folder", DEFAULT_BACKGROUNDS)
    available = list_backgrounds(bg_folder)
    known = {scene_key(p.stem) for p in available.values()}
    for key in cfg["scene"]:
        if key not in known:
            print(f"warning: [scene.{key}] in config.toml matches no background in {bg_folder} "
                  f"(available: {', '.join(p.stem for p in available.values())}); it is ignored")
    bg_paths = select_backgrounds(choice, available)
    loaded = []
    for path in bg_paths:
        bg = read_image(path)
        if bg is None:
            raise SystemExit(f"cannot read background: {path}")
        overrides = cfg["scene"].get(scene_key(path.stem))
        has_room = view == "room" and room_info(path.stem, bg, cfg["scene"])[0]
        try:
            with warnings.catch_warnings():
                if has_room:  # "made smaller to fit the picture": only the flat placement does that
                    warnings.simplefilter("ignore")
                scene = load_scene(path.stem, bg, table, overrides)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        loaded.append((path.stem, bg, scene))

    images = list_images(source)
    if not images:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_EXTS))}) in {source}")
    name = source.stem if source.is_file() else source.resolve().name
    out_dir = (args.out or cfg.get("output_folder", DEFAULT_OUTPUT)) / name
    input_dir = images[0].parent.resolve()
    targets = [out_dir / stem for stem, _, _ in loaded] + ([out_dir / "cutouts"] if save_cutouts else [])
    if debug:
        targets.append(out_dir / "debug")  # debug_images: one folder per photo, a picture per step
        if view == "flat":  # the flat placement's overlays stay where they always were
            targets += [out_dir / stem / "debug" for stem, _, _ in loaded]
    if any(t.resolve() == input_dir for t in targets):
        raise SystemExit(f"an output folder is the images folder; refusing to overwrite the originals: {input_dir}")

    try:
        import onnxruntime  # noqa: F401
        import rembg  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing package: {exc.name or exc}.\n{VENV_HINT}")

    print(f"{len(images)} image(s) from {source}")
    print(f"car: {shape.name} ({100 * shape.length:.1f} x {100 * shape.width:.1f} x {100 * shape.height:.1f} cm, "
          f"wheelbase {100 * shape.wheelbase:.1f} cm)")
    rr = None
    keep_crops = True  # room mode needs each photo's car crop only for a background without a room
    if view == "room":
        vr_path = find_view_results(source if source.is_dir() else source.parent)
        focal = fit_cfg.get("focal", "auto")
        rr = RoomRun(shape, None if focal == "auto" else float(focal), fit_cfg.get("on_fail", "place"), max_output,
                     1.0 if keep_sharp else 0.0, cut_policy, read_view_results(vr_path) if vr_path else None,
                     cfg["scene"])
        print("view: room (each background re-rendered as a 3D room from the photo's own camera)")
        flat_bgs = []
        for stem, plate, _ in loaded:
            has, text = room_info(stem, plate, cfg["scene"])
            if not has:
                flat_bgs.append(stem)
            print(f"background: {stem} ({text})")
        keep_crops = bool(flat_bgs)
        print("output: each photo's own framing" + (f", at most {max_output} px on the long side" if max_output else "")
              + ("; the background is never magnified at the car" if keep_sharp else ""))
        if flat_bgs:
            print(f"background size for {', '.join(flat_bgs)}: {SCALE_TEXT[scaler.mode]} "
                  f"(background_scale = \"{scaler.mode}\")")
        if cut_setting in ("anchor", "centre"):
            print(f"note: cut_frames = \"{cut_setting}\" is for background_view = \"flat\"; room mode frames "
                  "cut-off cars like the photo (\"frame\")")
        if rr.focal:
            print(f"lens: focal length {rr.focal:g} px from [fit] focal")
        print("view hints: " + (f"frame_view_type results ({vr_path})" if vr_path else "the file names' view labels"))
    else:
        for stem, _, scene in loaded:
            kind = "own scene" if scene.spec.notes and "generic" not in scene.spec.notes else "generic scene"
            print(f"background: {stem} ({kind}, {scene.px_per_m:.0f} px/m at the stage centre at full size)")
        print(f"background size: {SCALE_TEXT[scaler.mode]} (background_scale = \"{scaler.mode}\""
              + (f", at least {scaler.min_width} px wide)" if scaler.mode != "off" else ")"))
    t_load = time.perf_counter()
    hr = None
    if hosted:
        from hosted.pipeline import HostedRun
        hr = HostedRun(hosted_cfg, out_dir, (args.out or cfg.get("output_folder", DEFAULT_OUTPUT)) / "_hosted_cache",
                       ROOT / ".env", (model, edge_shrink, threads), edge_shrink)
        if rr is not None and hosted_cfg["camera"]:
            rr.hosted = hr
        parts = [t for t, on in (("BiRefNet cut-out", hosted_cfg["cutout"]), ("SAM 3 keeps only this car", True),
                                 ("glass tint (all windows; untouched photos kept in originals)", hosted_cfg["glass_tint"]),
                                 ("MoGe-3 camera", hosted_cfg["camera"] and view == "room"),
                                 ("VGGT joint solve", hosted_cfg["camera"] and hosted_cfg["vggt"] and view == "room"))
                 if on]
        limit = hosted_cfg["max_cost_per_car"]
        print("hosted on fal.ai: " + ", ".join(parts) + (f"; at most ${limit:g} for this folder" if limit else ""))
    try:
        segmenter = hr.segmenter if hr is not None else CarSegmenter(model, edge_shrink, threads)
    except ValueError as exc:  # rembg: unknown model name
        raise SystemExit(f"cannot load model '{model}': {exc}\n"
                         "Use 'birefnet-general-lite' (default) or 'birefnet-general'.") from None
    if hr is None:
        print(f"model {model} loaded in {time.perf_counter() - t_load:.1f}s (CPU, threads={threads or 'auto'})",
              flush=True)
    isolator, raws = None, {}
    if isolate:  # two passes, so BiRefNet and SAM are never in memory together (16 GB laptop)
        raws = precompute_raw(segmenter, images)
        segmenter.release()
        gc.collect()
        from car_isolation import DEFAULT_MIN_SCORE, CarIsolator
        t_load = time.perf_counter()
        isolator = CarIsolator(iso_cfg.get("min_score", DEFAULT_MIN_SCORE), iso_cfg.get("remove_objects", True),
                               threads or 4)
        print(f"car isolation on: YOLOX-s + SAM ViT-B loaded in {time.perf_counter() - t_load:.1f}s (other vehicles "
              "and objects touching the car are removed from its mask)", flush=True)
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
    dw = None
    if debug:
        dw = DebugWriter(out_dir / "debug")
        dw.write_readme()
        print(f"debug_mode on: a picture for every step in {dw.root}\\<photo>\\ (what each one shows: README.txt there)")
    print(flush=True)

    run = Run(loaded, scaler, cut_policy, shadow, out_dir, quality, debug)
    rows = []
    waiting: list[Photo] = []  # "folder" mode and room mode: cut out, waiting for the rest of the folder
    t_start = time.perf_counter()
    t_seg_total = 0.0
    n_done = 0
    try:
        for i, path in enumerate(images, 1):
            try:
                image = read_image(path)
            except OSError:
                image = None
            if image is None:
                print(f"[{i}/{len(images)}] skip (cannot read image): {path.name}", flush=True)
                rows.append({"file": path.name, "note": "skipped: cannot read the image"})
                continue
            t0 = time.perf_counter()
            trace, raw_dbg = ({}, None) if dw is not None else (None, None)  # debug_mode: each step's mask
            try:
                if hr is not None:
                    car, raw = hr.cut_out(path, image, trace)
                    raw_dbg = raw if dw is not None else None
                    del raw
                elif isolator is None:
                    if dw is None:
                        car = segmenter.cut_out(image)
                    else:  # the same as cut_out, keeping the model's raw mask for the pictures
                        raw_dbg = segmenter.raw_alpha(image)
                        car = segmenter.finish(image, raw_dbg, None, trace)
                elif path not in raws:
                    raise NoCarFound("the cut-out model failed on this photo in the first pass")
                else:
                    raw, t_raw = raws.pop(path)
                    t0 -= t_raw  # segment_s = BiRefNet (first pass) + isolation
                    raw_dbg = raw if dw is not None else None
                    car = segmenter.finish(image, raw, isolator, trace)
            except NoCarFound as exc:
                print(f"[{i}/{len(images)}] skip ({exc}): {path.name}", flush=True)
                rows.append({"file": path.name, "note": f"skipped: {exc}"})
                if dw is not None and raw_dbg is not None:  # the photo and the model's mask still help here
                    debug_step(cutout_pictures, dw, path.stem, image, raw_dbg, {}, None, model)
                continue
            frame_path = None
            if hr is not None:  # glass tint: the only change to the car's pixels, inside the window mask only
                untinted = image
                image, frame_path, window = hr.glass(path, image, car.alpha)
                if dw is not None:
                    debug_step(glass_pictures, dw, path.stem, untinted, image, window, car.alpha,
                               hr.cols(path.stem).get("glass_pct"))
                del untinted, window
            t_seg = time.perf_counter() - t0
            t_seg_total += t_seg
            n_done += 1
            if save_cutouts:
                write_image(out_dir / "cutouts" / f"{path.stem}.webp", cutout_bgra(image, car), quality=101)
            if dw is not None:
                iso = segmenter.last_isolation if (isolator is not None or hr is not None) else None
                iso_img = None
                if iso is not None:
                    from car_isolation import overlay
                    iso_img = overlay(image, iso)
                debug_step(cutout_pictures, dw, path.stem, image, raw_dbg, trace, car, model, iso_img,
                           iso.text() if iso is not None else "")
                del raw_dbg, trace
            segmenter.last_isolation = None

            measured = measure_photo(path, image, car.alpha) if view == "room" else None
            if measured is not None and hr is not None:
                measured.flags += hr.flags(path.stem)
            photo = prepare_photo(path, image, car, table, t_seg)
            if hr is not None:
                photo.frame_path, photo.hosted = frame_path, hr.cols(path.stem)
            if measured is not None:  # room mode: placed once every photo's lines are measured
                photo.frame_size, photo.measured = (image.shape[1], image.shape[0]), measured
                if not keep_crops:  # stage 3 re-reads the photo; the crop is only for flat placements
                    photo.image = photo.image[:0, :0]
            del image, car  # the photo keeps only the car's part
            if view == "room":
                waiting.append(photo)
                print(f"[{i}/{len(images)}] {path.name:<40} {t_seg:5.1f}s  {describe_room(photo, cut_policy)}"
                      + iso_text(photo), flush=True)
                continue
            line = f"[{i}/{len(images)}] {path.name:<40} {t_seg:5.1f}s  {describe(photo, cut_policy)}" + iso_text(photo)
            if scaler.mode == "folder":  # placed once every photo is cut out and the size is known
                waiting.append(photo)
                print(line, flush=True)
                continue
            scales = {stem: scaler.photo_scale(scene, photo.probe, photo.yaw, cut_policy) for stem, _, scene in loaded}
            photo_rows, pcts = place_photo(photo, run, scales)
            rows += photo_rows
            print(line + shrink_text(pcts), flush=True)
        if view == "room":
            del segmenter, isolator  # the models' memory is not needed for fitting and rendering
            gc.collect()
            place_room_photos(waiting, run, rr, rows)
        elif waiting:
            scales = folder_scales(waiting, run)
            for photo in waiting:
                rows += place_photo(photo, run, scales)[0]
    finally:  # also when the run stops part-way (an error, Ctrl+C): what was placed is listed
        if view == "room":  # and every photo that was cut out but never placed
            listed = {r.get("file") for r in rows}
            rows += [{**base_row(p), "note": "not placed: the run stopped before this photo was placed"}
                     for p in waiting if p.path.name not in listed]
        if rows:
            if view == "room":
                file_order(rows, images, loaded)
            with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results_fields(view, isolate, hr is not None), restval="")
                writer.writeheader()
                writer.writerows(rows)
    elapsed = time.perf_counter() - t_start

    if not n_done:
        raise SystemExit("no images could be processed")

    n_cut = len({r["file"] for r in rows if r.get("cut_off")})
    print(
        f"\n{n_done} image(s) x {len(loaded)} background(s) in {elapsed:.1f}s "
        f"(cutting out the car: {t_seg_total / n_done:.1f}s per image)"
    )
    if n_cut:
        print(f"{n_cut} image(s) show a car cut off by the photo edge; see 'cut_off' in results.csv")
    if view == "room":
        placed = {r["file"] for r in rows if r.get("out_w") not in (None, "")}
        skipped = {r["file"] for r in rows if str(r.get("note", "")).startswith(("skipped", "fit error", "place error"))}
        flagged = {r["file"] for r in rows if r.get("flags")}
        print(f"{len(placed)} photo(s) placed" + (f", {len(skipped - placed)} not placed (see 'note')"
                                                  if skipped - placed else "")
              + (f"; {len(flagged)} flagged for a look (see 'flags' in results.csv)" if flagged else ""))
    if hr is not None:
        sp = hr.client.spend
        billed = ", ".join(f"{k} ${v:.2f}" for k, v in sp.by_endpoint.items()) or "nothing billed"
        print(f"hosted: {sp.calls} fal call(s), {sp.cached} from the cache, about ${sp.usd:.2f} ({billed})")
        red = getattr(hr.client, "redact", None)
        if red is not None and hasattr(red, "people"):
            print(f"hosted privacy: {red.people} person box(es) blurred over {red.images} image(s) before upload")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
