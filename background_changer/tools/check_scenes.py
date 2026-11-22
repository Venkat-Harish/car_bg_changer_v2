"""Check the scene placement on saved cut-outs, without the segmentation model (seconds, not minutes).

Reads the transparent cut-outs a normal run saved (output/<images>/cutouts/*.webp) and the cut-off
flags from that run's results.csv, places every car on every background, prints the numbers and
checks them against the acceptance limits. Optionally writes contact sheets and debug overlays.

Usage (from the background_changer folder):
  .venv\\Scripts\\python tools\\check_scenes.py
  .venv\\Scripts\\python tools\\check_scenes.py --sheets --debug
  .venv\\Scripts\\python tools\\check_scenes.py output\\sample_vid --car "Maruti Suzuki Baleno 2022" --background-scale folder --sheets
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from background_scaler import DEFAULT_MIN_WIDTH, MODES, BackgroundScaler, needed_scale  # noqa: E402
from bg_changer import CarMask, CarTable, estimate_yaw, load_cars, load_scene, parse_name, place_car, probe_car, select_car  # noqa: E402
from bg_changer.segment import bottom_profile, fill_holes  # noqa: E402

DEFAULT_RUN = ROOT / "output" / "Final_zoomed_car"
BACKGROUNDS = ROOT / "backgrounds"
EQB = "Mercedes-Benz EQB 2024"
# The EQB walkaround's reference frames on background_1 (full size): side, front and rear heights.
EQB_EXPECT = {"0012.0s_right": 499, "0000.0s_front": 579, "0020.0s_back": 642}


def read(path: Path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flag)


def car_from_cutout(bgra: np.ndarray, cut_sides: set[str]) -> tuple[np.ndarray, CarMask]:
    """Rebuild the CarMask of a saved cut-out (coordinates relative to the cut-out)."""
    alpha = bgra[..., 3].astype(np.float32) / 255.0
    solid = fill_holes((alpha >= 0.5).astype(np.uint8))
    ys, xs = np.nonzero(solid)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    h, w = alpha.shape
    cut = {s: (s in cut_sides) for s in ("left", "right", "top", "bottom")}
    car = CarMask(alpha=alpha, box=box, crop=(0, 0, w, h), cut=cut, bottom=bottom_profile(solid, box))
    return np.ascontiguousarray(bgra[..., :3]), car


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN, help="a run's output folder (with cutouts/ and results.csv)")
    ap.add_argument("--car", default=EQB)
    ap.add_argument("--background-scale", choices=MODES, default="off",
                    help="shrink the backgrounds to the photos' resolution, as background_scaler does in a run")
    ap.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH, help="smallest background width (px)")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "check")
    ap.add_argument("--sheets", action="store_true", help="write a contact sheet per background, sorted by yaw")
    ap.add_argument("--debug", action="store_true", help="write per-frame debug overlays")
    ap.add_argument("--full", action="store_true", help="also write every composite at full size")
    args = ap.parse_args()

    cut_by_file: dict[str, set[str]] = {}
    with open(args.run / "results.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sides = {s for s in (r.get("cut_off") or "").replace(";", ",").split(",") if s}
            sides |= {f.removeprefix("cut-") for f in (r.get("flags") or "").split(";") if f.startswith("cut-")}
            cut_by_file.setdefault(Path(r["file"]).stem, set()).update(sides)

    table = CarTable(select_car(load_cars(ROOT / "cars.json"), args.car))
    plates = {p.stem: read(p) for p in sorted(BACKGROUNDS.glob("*.webp"))}
    scenes = {name: load_scene(name, plate, table) for name, plate in plates.items()}
    scaler = BackgroundScaler(args.background_scale, args.min_width)
    cutouts = sorted((args.run / "cutouts").glob("*.webp"))
    if not cutouts:
        raise SystemExit(f"no cut-outs in {args.run / 'cutouts'}")

    photos = []
    for path in cutouts:
        image, car = car_from_cutout(read(path, cv2.IMREAD_UNCHANGED), cut_by_file.get(path.stem, set()))
        probe = probe_car(car)
        photos.append((path.stem, image, car, probe, estimate_yaw(parse_name(path.stem), probe, table)))
    folder_k = {name: scaler.folder_scale(needed_scale(scene, p[3], p[4]) for p in photos) for name, scene in scenes.items()}

    rows, tiles = [], {name: [] for name in scenes}
    for stem, image, car, probe, yaw in photos:
        for name, scene in scenes.items():
            k = folder_k[name] if scaler.mode == "folder" else scaler.photo_scale(scene, probe, yaw)
            scene_k, plate_k = scaler.apply(scene, plates[name], k)
            out, plan = place_car(image, car, plate_k, scene_k, probe, yaw)
            if plan is None:
                continue
            x1, y1, x2, y2 = plan.box_out
            rows.append(dict(file=stem, bg=name, yaw=yaw.deg, scale=plan.scale, height=y2 - y1, width=x2 - x1,
                             target_h=plan.target_h, low=y2, margin=plan.margin, flags=";".join(plan.flags),
                             size=scene_k.size_wh, kx=scene_k.size_wh[0] / scene.size_wh[0],
                             limit=scene_k.spec.max_lowest_row, cut=probe.cut))
            if args.sheets or args.debug or args.full:
                folder = args.out / name
                folder.mkdir(parents=True, exist_ok=True)
                if args.full:
                    cv2.imencode(".webp", out, [cv2.IMWRITE_WEBP_QUALITY, 90])[1].tofile(str(folder / f"{stem}.webp"))
                if args.debug:
                    dbg = scene_k.debug_overlay(out, plan, probe)
                    cv2.imencode(".jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(folder / f"{stem}_debug.jpg"))
                small = cv2.resize(out, (480, 270), interpolation=cv2.INTER_AREA)
                cv2.putText(small, f"{stem[:6]} {yaw.deg:.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                tiles[name].append((yaw.deg, small))

    print(f"car: {table.shape.name}   background_scale: {scaler.mode}")
    print(f"{'file':<28} {'bg':<13} {'size':>9} {'yaw':>5} {'scale':>6} {'height':>6} {'target':>6} {'width':>6} {'low':>5} "
          f"{'margin':>6}  flags")
    for r in rows:
        size = f"{r['size'][0]}x{r['size'][1]}"
        print(f"{r['file']:<28} {r['bg']:<13} {size:>9} {r['yaw']:5.0f} {r['scale']:6.3f} {r['height']:6d} "
              f"{r['target_h']:6.1f} {r['width']:6d} {r['low']:5d} {r['margin']:6d}  {r['flags']}")

    # Acceptance checks (label yaws).
    fails = []
    if any(r["scale"] > 1.0 for r in rows):
        fails.append("a scale is above 1")
    small_margin = [r for r in rows if r["margin"] < 150 * r["kx"]]
    if small_margin:
        fails.append(f"{len(small_margin)} placements have less than 150 px margin (at full size)")
    # Every car that was not capped stands at its stage height (scale from width when cut top/bottom).
    off_stage = [r for r in rows if "scale-capped" not in r["flags"] and not {"top", "bottom"} & set(r["cut"])
                 and abs(r["height"] - r["target_h"]) > max(2.0, 0.01 * r["target_h"])]
    if off_stage:
        fails.append(f"{len(off_stage)} cars are not at their stage height: "
                     + ", ".join(f"{r['file'][:6]}/{r['bg']}" for r in off_stage[:6]))
    if scaler.mode == "photo":
        resampled = [r for r in rows if "scale-capped" not in r["flags"] and r["scale"] != 1.0]
        if resampled:
            fails.append(f"{len(resampled)} cars were resampled in 'photo' mode (expected pixel for pixel)")
    past_limit = [r for r in rows if r["limit"] is not None and r["low"] > r["limit"] + 0.5]
    if past_limit:
        fails.append(f"{len(past_limit)} cars stand past their scene's lowest-row limit")
    capped = sorted({r["file"] for r in rows if "scale-capped" in r["flags"]})
    if capped:
        print(f"\n{len(capped)} photo(s) hold a car smaller than the stage needs (placed smaller): {', '.join(capped)}")
    sizes = sorted({(r["bg"], r["size"]) for r in rows})
    print("background sizes: " + ", ".join(f"{bg} {w}x{h}" for bg, (w, h) in sizes[:9])
          + (f" ... ({len(sizes)} in all)" if len(sizes) > 9 else ""))

    if table.shape.name == EQB:
        bg1 = {r["file"].rsplit("_", 1)[0]: r for r in rows if r["bg"] == "background_1"}  # e.g. 0020.0s_back
        for key, h in EQB_EXPECT.items():
            if key in bg1 and not abs(bg1[key]["height"] - h * bg1[key]["kx"]) <= 0.03 * h * bg1[key]["kx"]:
                fails.append(f"background_1 {key} height {bg1[key]['height']} not within 3% of {h * bg1[key]['kx']:.0f}")
        if all(k in bg1 for k in EQB_EXPECT):
            side = bg1["0012.0s_right"]
            rs, fs = bg1["0020.0s_back"]["height"] / side["height"], bg1["0000.0s_front"]["height"] / side["height"]
            print(f"background_1 height ratios: rear/side {rs:.3f} (1.20-1.32), front/side {fs:.3f} (1.08-1.20)")
            if not 1.20 <= rs <= 1.32:
                fails.append(f"rear/side ratio {rs:.3f}")
            if not 1.08 <= fs <= 1.20:
                fails.append(f"front/side ratio {fs:.3f}")
            share = side["width"] / (1761 * side["kx"])
            print(f"background_1 side view 0012 = {100 * share:.1f}% of the rim (70-77%)")
            if not 0.70 <= share <= 0.77:
                fails.append(f"side view share {share:.3f}")

    if args.sheets:
        for name, items in tiles.items():
            if not items:
                continue
            items.sort(key=lambda t: t[0])
            imgs = [t[1] for t in items]
            while len(imgs) % 4:
                imgs.append(np.zeros_like(imgs[0]))
            sheet = np.vstack([np.hstack(imgs[i:i + 4]) for i in range(0, len(imgs), 4)])
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(args.out / f"sheet_{name}.jpg"))
        print(f"sheets written to {args.out}")

    print("\nACCEPTANCE: " + ("PASS" if not fails else "FAIL\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
