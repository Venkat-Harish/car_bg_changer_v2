"""Review tool for the room path (car_fit_to_bg + room3d): run the real three-stage flow on folders of
photos and write composites, debug overlays, contact sheets and a metrics table.

    .venv\\Scripts\\python tools\\check_room.py                       # both test videos, folder mode
    .venv\\Scripts\\python tools\\check_room.py --per-photo           # each photo alone (no pooling)
    .venv\\Scripts\\python tools\\check_room.py --frames 0012 0020    # a few frames only

Output: output/check_room/<mode>/<folder>/
  <background>/<photo>.webp        the room composite
  <background>/debug/<photo>.jpg   footprint, stage, horizon, tyres, invented areas tinted
  fit/<photo>.jpg                   the photo with the mask, the fitted proxy and its contacts
  sheets/<photo>.jpg                photo + fit | flat baseline | room background_1..3
  results.csv                       per photo x background: fit, camera, grounding, quality metrics
Car masks are cached in output/check_room/_masks/<folder>/ (16-bit PNG alpha) after the first run.
Exit code 1 when a self-check fails: the fitted proxy drawn through the room camera no longer matches the
pasted car (chain IoU more than 0.02 below the fit), or an opaque car pixel differs from INTER_AREA(level).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bg_changer import load_cars, select_car  # noqa: E402
from bg_changer.scenes import SCENES, scene_key  # noqa: E402
from car_fit_to_bg import fit_overlay, place_in_room  # noqa: E402
from car_fit_to_bg.batch import (choose_focals, cut_sides, find_view_results, fit_measured, folder_height_prior,  # noqa: E402
                                 measure, read_view_results, skip_reason)
from car_fit_to_bg.level import downscale  # noqa: E402
from car_fit_to_bg.pose import body_points, iou, render_proxy  # noqa: E402

IMAGES = ROOT.parent / "number_plate_detection" / "output"
DEFAULT_SETS = {"sample_vid": "Maruti Suzuki Baleno 2022", "Final_zoomed_car": "Mercedes-Benz EQB 2024"}
# Reference yaws: Baleno from the diagnosis (two-tyre poses; '1T' frames one-tyre); EQB from the video-pooled
# walk-prior fit (not hand-measured).
REF_YAW = {
    "sample_vid": {"0000": 182, "0002": 197, "0004": 224, "0006": 276, "0012": 9, "0014": 11, "0016": 23, "0018": 42,
                   "0020": 84, "0022": 136, "0024": 157, "0026": 175},
    "Final_zoomed_car": {"0000": 0.8, "0002": 3.0, "0004": 14.7, "0006": 32.8, "0008": 52.6, "0010": 73.8, "0012": 92.6,
                         "0014": 120.3, "0016": 141.6, "0018": 162.6, "0020": 173.0, "0022": 192.7, "0024": 211.7,
                         "0026": 231.0, "0028": 255.8, "0030": 280.7, "0032": 301.9, "0034": 325.3, "0036": 348.8,
                         "0038": 1.7},
}
TWO_TYRE = {"sample_vid": {"0000", "0002", "0004", "0012", "0014", "0016", "0018", "0026"}}
FIELDS = ["file", "background", "out_w", "out_h", "out_scale", "yaw", "yaw_ref", "yaw_err", "yaw_source", "fit_iou",
          "fit_margin", "chain_iou", "roll", "car_roll", "level_deg", "pitch", "focal_px", "focal_source", "cam_dist_m",
          "cam_height_m", "elev_deg", "tyres_found", "lock_mode", "contact_residual_px", "invented_pct",
          "invented_near_pct", "floor_texels", "fit_s", "render_s", "flags", "note"]


def read(path: Path, flag=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, flag) if data.size else None


def write(path: Path, img, quality=90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_WEBP_QUALITY, quality] if ext == ".webp" else [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))


def get_alpha(path: Path, frame: np.ndarray, cache: Path, segmenter_box: list) -> np.ndarray:
    """The car's full-frame alpha: cached 16-bit PNG, else the real segmenter (BiRefNet via rembg)."""
    cp = cache / f"{path.stem}.png"
    if cp.is_file():
        a = read(cp, cv2.IMREAD_UNCHANGED)
        if a is not None and a.shape[:2] == frame.shape[:2]:
            return a.astype(np.float32) / 65535.0
    if not segmenter_box:
        from bg_changer import CarSegmenter
        segmenter_box.append(CarSegmenter("birefnet-general-lite", 2, 4))
    car = segmenter_box[0].cut_out(frame)
    cache.mkdir(parents=True, exist_ok=True)
    write_png16(cp, car.alpha)
    return car.alpha.astype(np.float32)


def write_png16(path: Path, alpha: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", np.clip(np.rint(alpha * 65535), 0, 65535).astype(np.uint16))
    if ok:
        buf.tofile(str(path))


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def sheet(tiles: list[tuple[np.ndarray | None, str]], height: int = 520) -> np.ndarray:
    out = []
    for img, cap in tiles:
        if img is None:
            img = np.full((height, height // 2, 3), 40, np.uint8)
        s = height / img.shape[0]
        t = cv2.resize(img, (max(1, int(round(img.shape[1] * s))), height), interpolation=cv2.INTER_AREA)
        bar = np.full((26, t.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, cap[:60], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, t]))
        out.append(np.full((height + 26, 6, 3), 255, np.uint8))
    return np.hstack(out[:-1])


def chain_check(fit, placed, name) -> float:
    """IoU of the fitted proxy drawn through the levelled (pre-lock) output camera with the pasted car."""
    s = placed.s_out
    x0, y0 = placed.window
    Wo, Ho = placed.size
    f = fit.f * s
    c = ((fit.c[0] + 0.5) * s - 0.5 - x0, (fit.c[1] + 0.5) * s - 0.5 - y0)
    pts = body_points(fit.shape, fit.pose.body)
    q = (pts - _c_car(fit)) @ fit.R_lev.T  # levelling keeps the camera centre
    pr = render_proxy(q, f, c, (Wo, Ho))
    pixels, a_s, corner = downscale(fit.car, s)
    a_out = np.zeros((Ho, Wo), np.float32)
    px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
    hh, ww = a_s.shape
    ya, xa, yb, xb = max(0, py), max(0, px), min(Ho, py + hh), min(Wo, px + ww)
    if yb > ya and xb > xa:
        a_out[ya:yb, xa:xb] = a_s[ya - py:yb - py, xa - px:xb - px]
    return iou(pr, (a_out >= 0.5).astype(np.uint8) * 255)


def _c_car(fit) -> np.ndarray:
    R, t = fit.pose.rt(fit.c)
    return -R.T @ t


def car_pixels_ok(fit, placed) -> bool:
    s = placed.s_out
    x0, y0 = placed.window
    pixels, a_s, corner = downscale(fit.car, s)
    px, py = int(round(corner[0] - x0)), int(round(corner[1] - y0))
    Ho, Wo = placed.image.shape[:2]
    hh, ww = a_s.shape
    ya, xa, yb, xb = max(0, py), max(0, px), min(Ho, py + hh), min(Wo, px + ww)
    if yb <= ya or xb <= xa:
        return True
    got = placed.image[ya:yb, xa:xb]
    exp = pixels[ya - py:yb - py, xa - px:xb - px]
    opaque = a_s[ya - py:yb - py, xa - px:xb - px] >= 1.0
    return bool(np.array_equal(got[opaque], exp[opaque]))


def run_folder(folder: str, car_name: str, args, backgrounds: dict) -> list[str]:
    images = IMAGES / folder
    mode = ("per_photo" if args.per_photo else "folder") + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "output" / "check_room" / mode / folder
    cache = ROOT / "output" / "check_room" / "_masks" / folder
    shape = select_car(load_cars(ROOT / "cars.json"), car_name)
    vr_path = find_view_results(images)
    view_rows = read_view_results(vr_path) if vr_path else None
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.frames:
        paths = [p for p in paths if any(p.stem.startswith(f) for f in args.frames)]
    seg: list = []
    rows, failures = [], []
    t_all = time.perf_counter()
    # Stage 1: segment (cached) and measure the free-focal line camera.
    items, alphas = [], {}
    for p in paths:
        try:
            frame = read(p)
            if frame is None:
                raise ValueError("cannot read the image")
            alpha = get_alpha(p, frame, cache, seg)
            m = measure(p, frame, alpha)
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the review
            rows.append(dict(file=p.name, note=f"skipped: {type(exc).__name__}: {exc}"))
            print(f"  {p.stem}: skipped ({exc})", flush=True)
            continue
        items.append(m)
        alphas[p.stem] = alpha
        print(f"  measured {p.stem}: f {m.cam.f or 0:.0f} (conf {m.cam.conf_f:.2f}) roll {m.cam.roll or 0:+.2f} "
              f"pitch {m.cam.pitch or 0:.2f} (conf {m.cam.conf:.2f})", flush=True)
    # Stage 2: focals.
    choose_focals(items, None, pool=not args.per_photo)
    # Stage 3: fit (strong lines first, so weak-line photos can borrow the folder's camera height), place.
    order = sorted(items, key=lambda m: 0 if (m.cam.roll is not None and m.cam.conf >= 0.5) else 1)
    fits = []
    for m in order:
        stem = m.stem
        alpha = alphas[stem]
        cut = cut_sides(alpha)
        reason = skip_reason(cut)
        if reason:
            rows.append(dict(file=m.path.name, note=f"skipped: {reason}"))
            print(f"  {stem}: skipped ({reason})", flush=True)
            continue
        frame = read(m.path)
        hp = folder_height_prior(fits) if not args.per_photo else None
        try:
            fit = fit_measured(m, frame, alpha, cut, shape, items, view_rows, video=not args.per_photo,
                               h_prior=hp, max_seconds=args.max_seconds)
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the review
            rows.append(dict(file=m.path.name, note=f"fit error: {type(exc).__name__}: {exc}"))
            failures.append(f"{stem}: fit error {exc!r}")
            continue
        fits.append(fit)
        write(out / "fit" / f"{stem}.jpg", fit_overlay(frame, fit, alpha), 88)
        key = stem[:4]
        ref = REF_YAW.get(folder, {}).get(key)
        err = abs(wrap(fit.pose.yaw - ref)) if ref is not None else None
        tiles = [(fit_overlay(frame, fit, alpha), f"{stem[:4]} fit yaw {fit.pose.yaw:.0f} iou {fit.pose.iou:.3f}")]
        base = ROOT / "output" / "_baseline_flat" / folder / "background_1" / f"{stem}.webp"
        tiles.append((read(base) if base.is_file() else None, "flat baseline (background_1)"))
        for name, plate in backgrounds.items():
            try:
                placed = place_in_room(fit, name, plate, SCENES[scene_key(name)], debug=True,
                                       max_output_size=args.max_output_size, min_floor_texels=args.min_texels)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(file=m.path.name, background=name, note=f"place error: {type(exc).__name__}: {exc}"))
                failures.append(f"{stem} {name}: place error {exc!r}")
                continue
            write(out / name / f"{stem}.webp", placed.image)
            write(out / name / "debug" / f"{stem}.jpg", placed.debug, 85)
            tiles.append((placed.image, f"room {name}"))
            ch = chain_check(fit, placed, name)
            if ch < fit.pose.iou - 0.02:  # (the fit's IoU is on a smoothed 480 px raster)
                failures.append(f"{stem} {name}: chain IoU {ch:.3f} vs fit {fit.pose.iou:.3f}")
            if not car_pixels_ok(fit, placed):
                failures.append(f"{stem} {name}: car pixels changed")
            mt = placed.metrics
            flags = list(dict.fromkeys(fit.flags + placed.flags))
            if ref is not None and err > 24:
                flags.append("yaw-vs-reference")
            rows.append(dict(file=m.path.name, background=name, out_w=mt["out_w"], out_h=mt["out_h"],
                             out_scale=mt["out_scale"], yaw=round(fit.pose.yaw, 1), yaw_ref=ref if ref is not None else "",
                             yaw_err=round(err, 1) if err is not None else "", yaw_source=fit.pose.yaw_source,
                             fit_iou=round(fit.pose.iou, 3), fit_margin=round(fit.pose.margin, 1), chain_iou=round(ch, 3),
                             roll=round(fit.pose.roll, 2), car_roll=round(fit.car_roll, 2), level_deg=round(fit.theta, 2),
                             pitch=round(fit.pose.pitch, 2), focal_px=round(fit.f, 1), focal_source=fit.focal_source,
                             cam_dist_m=round(fit.pose.D, 2), cam_height_m=round(fit.pose.h, 2),
                             elev_deg=round(fit.pose.delta, 1),
                             tyres_found=sum(b.status == "found" for b in fit.bottoms), lock_mode=fit.lock["mode"],
                             contact_residual_px=mt["contact_residual_px"], invented_pct=mt["invented_pct"],
                             invented_near_pct=mt["invented_near_pct"], floor_texels=mt["floor_texels"],
                             fit_s=round(fit.seconds, 1), render_s=mt["render_s"], flags=";".join(flags)))
        write(out / "sheets" / f"{stem}.jpg", sheet(tiles), 85)
        print(f"  {stem}: yaw {fit.pose.yaw:6.1f}" + (f" (ref {ref}, err {err:4.1f})" if ref is not None else "") +
              f" iou {fit.pose.iou:.3f} margin {fit.pose.margin:4.1f} level {fit.theta:+.1f} car_roll {fit.car_roll:+.1f} "
              f"tyres {sum(b.status == 'found' for b in fit.bottoms)} lock {fit.lock['mode']} "
              f"flags {fit.flags} fit {fit.seconds:.1f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    write_index(out, folder, rows)
    # Yaw summary.
    errs = [(r["file"][:4], r["yaw_err"]) for r in rows if r.get("background") == "background_1" and r.get("yaw_err") != ""]
    if errs:
        e = np.array([v for _, v in errs], float)
        flips = int((e > 150).sum())
        fine = e[e <= 150]
        two = TWO_TYRE.get(folder)
        msg = f"  yaw vs reference: n {len(e)}, flips {flips}, median {np.median(fine):.1f}, RMS {math.sqrt((fine ** 2).mean()):.1f}, max {fine.max():.1f}"
        if two:
            e2 = np.array([v for k, v in errs if k in two], float)
            msg += f"; two-tyre frames: max {e2.max():.1f}"
        print(msg)
    print(f"  {folder}: {len(fits)} placed in {time.perf_counter() - t_all:.0f}s -> {out}")
    return failures


def write_index(out: Path, folder: str, rows: list[dict]) -> None:
    """index.html next to the sheets: one row per photo (sheet image, fit and flags), for review in a browser."""
    import html
    by_file: dict[str, list[dict]] = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
             f"<title>{html.escape(folder)} room review</title><style>body{{font:14px system-ui,sans-serif;margin:12px;"
             f"background:#fff;color:#111}}img{{width:100%;height:auto;border:1px solid #ccc}}h3{{margin:18px 0 4px}}"
             f"p{{margin:2px 0;color:#444;font-size:12.5px}}</style><h2>{html.escape(folder)}: photo + fit | flat baseline | "
             f"room background_1..3</h2>"]
    for name, rs in by_file.items():
        stem = Path(name).stem
        first = rs[0]
        parts.append(f"<h3>{html.escape(stem)}</h3>")
        if first.get("note"):
            parts.append(f"<p>{html.escape(first['note'])}</p>")
            continue
        parts.append(f"<p>yaw {first.get('yaw')} (reference {first.get('yaw_ref') or '-'}; {first.get('yaw_source')}), "
                     f"fit {first.get('fit_iou')}, camera {first.get('cam_dist_m')} m away, {first.get('elev_deg')} deg down, "
                     f"levelled {first.get('level_deg')} deg, tyres {first.get('tyres_found')}</p>")
        flags = sorted({fl for r in rs for fl in (r.get("flags") or "").split(";") if fl})
        if flags:
            parts.append(f"<p>flags: {html.escape(', '.join(flags))}</p>")
        parts.append(f"<a href='sheets/{html.escape(stem)}.jpg'><img loading=lazy src='sheets/{html.escape(stem)}.jpg'></a>")
    (out / "index.html").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="*", default=list(DEFAULT_SETS), help="folders under number_plate_detection/output")
    ap.add_argument("--per-photo", action="store_true", help="each photo alone: no focal pooling, no video neighbours")
    ap.add_argument("--frames", nargs="*", help="only frames whose names start with these")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--max-output-size", type=int, default=1920)
    ap.add_argument("--min-texels", type=float, default=1.0,
                    help="plate texels per output px kept at the stage (0 = only --max-output-size limits the size)")
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. a size variant)")
    args = ap.parse_args()
    backgrounds = {n: read(ROOT / "backgrounds" / f"{n}.webp") for n in SCENES}
    failures = []
    for folder in args.folders:
        print(f"== {folder}", flush=True)
        failures += run_folder(folder, DEFAULT_SETS.get(folder, "Mercedes-Benz EQB 2024"), args, backgrounds)
    if failures:
        print("SELF-CHECK FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all self-checks passed")


if __name__ == "__main__":
    main()
