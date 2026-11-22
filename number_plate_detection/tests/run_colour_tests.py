"""Test every plate colour in turn on the colour test set (assets/test_photos).

For each case (one plate colour) and each of its photos, the masker runs with automatic
brightness on and off, and the result is compared with the truth in expected.csv:

  PASS needs  - the plate found by the detector
              - its 4 corners found (not a plain box), overlapping the true outline >= 90 %
              - the overlay's white within +-5 points of the true brightness
                (cases where the plate has no white or light paint must use the photo reading)

Results go to output/colour_tests/<case>/ (masked photos, compare.jpg, results.csv) and a
summary to output/colour_tests/summary.csv. Folders in assets/test_photos without answers
(e.g. your own real photos) are run too, with numbers only.

Usage (PowerShell, from the number_plate_detection folder):
  .venv\\Scripts\\python tests\\run_colour_tests.py                    every case, one after the other
  .venv\\Scripts\\python tests\\run_colour_tests.py --pause            wait for Enter after each case
  .venv\\Scripts\\python tests\\run_colour_tests.py --case yellow_black --case blue_white
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mask_plates  # noqa: E402  (config loading and image helpers)
from plate_brightness_detector import BrightnessDetector  # noqa: E402
from plate_masker import PlateMasker, mask_plates as paint_plates  # noqa: E402

PHOTOS_DIR = ROOT / "assets" / "test_photos"
OUT_DIR = ROOT / "output" / "colour_tests"
PASS_OVERLAP = 0.90
PASS_POINTS = 5.0


def parse_quad(text: str) -> np.ndarray | None:
    return np.float32([[float(v) for v in pair.split(",")] for pair in text.split()]) if text else None


def overlap(a: np.ndarray, b: np.ndarray, shape) -> float:
    """Intersection over union of two 4-corner shapes."""
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(ma, np.round(a).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(b).astype(np.int32), 1)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def board_white(image: np.ndarray, quad: np.ndarray) -> float:
    """How bright the overlay's white came out (%): 75th percentile inside the middle of the plate."""
    c = quad.mean(0)
    middle = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(middle, np.round(c + (quad - c) * 0.7).astype(np.int32), 1)
    return float(np.percentile(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[middle > 0], 75)) / 2.55


def crop(image: np.ndarray, quad: np.ndarray, width: int, label: str) -> np.ndarray:
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int)
    m = int(0.45 * max(x1 - x0, y1 - y0))
    cut = image[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    tile = cv2.resize(cut, (width, max(1, int(width * cut.shape[0] / cut.shape[1]))), interpolation=cv2.INTER_AREA)
    tile = cv2.copyMakeBorder(tile, 30, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def stack(rows: list[list[np.ndarray]]) -> np.ndarray:
    lines = []
    for tiles in rows:
        h = max(t.shape[0] for t in tiles)
        lines.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]))
    w = max(line.shape[1] for line in lines)
    return np.vstack([cv2.copyMakeBorder(line, 0, 0, 0, w - line.shape[1], cv2.BORDER_CONSTANT) for line in lines])


def load_cases(photos: Path) -> list[tuple[str, list[dict]]]:
    """[(case, rows)] in the test order: expected.csv's order, then folders without answers."""
    cases: dict[str, list[dict]] = {}
    expected = photos / "expected.csv"
    if expected.exists():
        with open(expected, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cases.setdefault(row["case"], []).append(row)
    for folder in sorted(p for p in photos.iterdir() if p.is_dir() and p.name not in cases):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in mask_plates.IMAGE_EXTS)
        if files:  # e.g. your own real photos: no right answer known, numbers only
            cases[folder.name] = [{"case": folder.name, "view": f.stem, "file": f"{folder.name}/{f.name}",
                                   "plate": "your own photos (no answers: numbers only)", "expected_reading": "",
                                   "true_pct": "", "outline": ""} for f in files]
    return list(cases.items())


def judge(plate, iou, auto_pct, row) -> tuple[str, str]:
    if not row["outline"]:
        return "-", "no answer known"
    if plate is None:
        return "FAIL", "plate not detected"
    if plate.fit == "box":
        return "FAIL", "corners not found (gets a plain box)"
    if iou < PASS_OVERLAP:
        return "FAIL", f"outline off ({iou:.0%} overlap)"
    source = plate.brightness.source if plate.brightness is not None else ""
    if row["expected_reading"] == "photo":
        return ("PASS", "used the photo reading") if source.startswith("photo") else ("FAIL", "expected the photo reading")
    diff = auto_pct - float(row["true_pct"])
    if abs(diff) <= PASS_POINTS:
        return "PASS", f"brightness {diff:+.0f} points"
    return "FAIL", f"brightness off by {diff:+.0f} points"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test each plate colour in turn on assets/test_photos.")
    parser.add_argument("--case", action="append", default=None, help="only this case (repeat for more)")
    parser.add_argument("--pause", action="store_true", help="wait for Enter after each case")
    parser.add_argument("--photos", type=Path, default=PHOTOS_DIR, help="the test set (default: assets/test_photos)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="results folder (default: output/colour_tests)")
    parser.add_argument("--config", type=Path, default=None, help="config file (default: config.toml)")
    args = parser.parse_args()

    if not args.photos.is_dir():
        raise SystemExit(f"no test set at {args.photos}\nMake it first: .venv\\Scripts\\python tests\\make_test_photos.py")
    cases = load_cases(args.photos)
    if args.case:
        known = {name for name, _ in cases}
        unknown = [c for c in args.case if c not in known]
        if unknown:
            raise SystemExit(f"unknown case(s): {', '.join(unknown)}\nAvailable: {', '.join(n for n, _ in cases)}")
        cases = [(name, rows) for name, rows in cases if name in args.case]
    if not cases:
        raise SystemExit(f"no test photos in {args.photos}")

    # Same detector, threads and overlay image as mask_plates.py uses.
    cfg = mask_plates.load_config(args.config or mask_plates.DEFAULT_CONFIG, required=args.config is not None)
    overlay_path = cfg.get("overlay_image", mask_plates.DEFAULT_OVERLAY_IMAGE)
    overlay = mask_plates.load_overlay(overlay_path)
    lighting = cfg.get("lighting", mask_plates.DEFAULT_LIGHTING)
    color = cfg.get("color", mask_plates.DEFAULT_COLOR_RGB)
    masker = PlateMasker(cfg.get("model", mask_plates.DEFAULT_MODEL), cfg.get("confidence", mask_plates.DEFAULT_CONF),
                         0.0, cfg.get("threads", mask_plates.DEFAULT_THREADS), tight_fit=True, overlay=overlay,
                         brightness_detector=BrightnessDetector(), color=color)
    print(f"overlay {overlay_path.name}; auto brightness ON vs OFF (lighting {lighting:g}%); "
          f"pass = corners found, outline >= {PASS_OVERLAP:.0%}, brightness within +-{PASS_POINTS:g} points\n")

    summary = []
    t_start = time.perf_counter()
    for n, (case, rows) in enumerate(cases, 1):
        print(f"=== {n}/{len(cases)}  {case}: {rows[0]['plate']}")
        if rows[0]["expected_reading"]:
            print(f"    expected reading: {rows[0]['expected_reading']}")
        print(f"    {'view':<13} {'found':<6} {'fit':<8} {'outline':>7}  {'truth':>5}  {'auto ON':>7}  "
              f"{'auto OFF':>8}  {'result':<5} note / where auto read the light")
        case_dir = args.out / case
        case_dir.mkdir(parents=True, exist_ok=True)
        results, sheet = [], []
        for row in rows:
            image = mask_plates.read_image(args.photos / row["file"])
            if image is None:
                print(f"    {row['view']:<13} cannot read {row['file']}")
                continue
            truth = parse_quad(row["outline"])
            masked_on, plates = masker.process(image)
            # The plate that matches the true outline (or the most confident one).
            plate, iou = None, 0.0
            for p in plates:
                o = overlap(p.quad, truth, image.shape[:2]) if truth is not None else p.confidence
                if plate is None or o > iou:
                    plate, iou = p, o
            if truth is None:
                iou = 0.0
            # Same plates, but the fixed 'lighting' setting instead of the brightness detector.
            masked_off = paint_plates(image, [dataclasses.replace(p, brightness=None) for p in plates], "image", 0.0,
                                      "solid", color=color, overlay=overlay, lighting=lighting)
            shape = truth if truth is not None else (plate.quad if plate is not None else None)
            on = board_white(masked_on, shape) if shape is not None and plate is not None else None
            off = board_white(masked_off, shape) if shape is not None and plate is not None else None
            verdict, note = judge(plate, iou, on if on is not None else 0.0, row)
            source = plate.brightness.source if plate is not None and plate.brightness is not None else ""
            fmt = lambda v: "-" if v is None else f"{v:.0f}%"  # noqa: E731
            print(f"    {row['view']:<13} {('yes ' + format(plate.confidence, '.0%')) if plate else 'no':<6} "
                  f"{plate.fit if plate else '-':<8} {(format(iou, '.0%') if truth is not None and plate else '-'):>7}  "
                  f"{(row['true_pct'] + '%') if row['true_pct'] else '-':>5}  {fmt(on):>7}  {fmt(off):>8}  "
                  f"{verdict:<5} {note}" + (f"; {source}" if source else ""))
            mask_plates.write_image(case_dir / Path(row["file"]).name, masked_on)
            if shape is not None:
                sheet.append([crop(image, shape, 380, f"{row['view']}: test photo"),
                              crop(masked_off, shape, 380, f"auto OFF  white {fmt(off)}"),
                              crop(masked_on, shape, 380, f"auto ON  white {fmt(on)}  {verdict}")])
            results.append({"case": case, "view": row["view"], "detected": plate is not None,
                            "confidence": round(plate.confidence, 3) if plate else "", "fit": plate.fit if plate else "",
                            "outline_overlap": round(iou, 3) if truth is not None and plate else "",
                            "true_pct": row["true_pct"], "auto_on_pct": round(on, 1) if on is not None else "",
                            "auto_off_pct": round(off, 1) if off is not None else "", "source": source,
                            "result": verdict, "note": note})
        if sheet:
            cv2.imencode(".jpg", stack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(case_dir / "compare.jpg"))
        if results:
            with open(case_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
        judged = [r for r in results if r["result"] != "-"]
        passed = sum(r["result"] == "PASS" for r in judged)
        verdict = "-" if not judged else ("PASS" if passed == len(judged) else "FAIL")
        mean = lambda key: (sum(float(r[key]) for r in results if r[key] != "") /  # noqa: E731
                            max(1, sum(r[key] != "" for r in results))) if any(r[key] != "" for r in results) else None
        summary.append({"case": case, "result": verdict, "views_passed": f"{passed}/{len(judged)}" if judged else "-",
                        "true_pct": round(mean("true_pct"), 1) if mean("true_pct") is not None else "",
                        "auto_on_pct": round(mean("auto_on_pct"), 1) if mean("auto_on_pct") is not None else "",
                        "auto_off_pct": round(mean("auto_off_pct"), 1) if mean("auto_off_pct") is not None else "",
                        "plate": rows[0]["plate"]})
        print(f"    -> {case}: {verdict} ({summary[-1]['views_passed']} views)   see {case_dir / 'compare.jpg'}\n")
        if args.pause and n < len(cases):
            input("Press Enter for the next case... ")

    print(f"=== summary ({time.perf_counter() - t_start:.0f} s)")
    print(f"    {'case':<14} {'result':<6} {'views':<6} {'truth':>6} {'auto ON':>8} {'auto OFF':>9}")
    for s in summary:
        cell = lambda v: f"{v:.0f}%" if v != "" else "-"  # noqa: E731
        print(f"    {s['case']:<14} {s['result']:<6} {s['views_passed']:<6} {cell(s['true_pct']):>6} "
              f"{cell(s['auto_on_pct']):>8} {cell(s['auto_off_pct']):>9}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nresults in {args.out}")


if __name__ == "__main__":
    main()
