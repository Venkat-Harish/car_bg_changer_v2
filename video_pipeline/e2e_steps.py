"""One end-to-end test run on a walk-around video, with a picture for every step.

  1. frame_view_type    sample a frame every `every_seconds`, label its side, keep the best frame per side
                        (classify_video.py --pick-sides -> frame_view_type/output/<video>_sides/)
  2. plates             number_plate_detection/mask_plates.py on those frames
  3. background          background_changer/change_background.py, room mode, every background, --debug
                        (debug pictures 01-16, plus 05a-05c for the hosted steps)
  4. report             <output>/e2e_<video>/index.html: per side, every step's picture left to right

Unlike make_video.py this keeps the view labels (the images folder is <video>_sides, whose results.csv
background_changer reads as the view prior) and makes no mp4: a handful of stills is not a video.

Settings come from e2e.toml. Run with the background_changer venv, from this folder:
  ..\\background_changer\\.venv\\Scripts\\python e2e_steps.py
  ..\\background_changer\\.venv\\Scripts\\python e2e_steps.py --report-only   (rebuild index.html only)
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
VIEW = REPO / "frame_view_type"
PLATES = REPO / "number_plate_detection"
BGCHANGER = REPO / "background_changer"
SIDES = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
KEYS = {"video", "every_seconds", "car", "backgrounds", "hosted", "output"}
TIMES_FILE = "step_times.csv"


def load_config(path: Path) -> dict:
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from None
    unknown = sorted(set(cfg) - KEYS)
    if unknown:
        raise SystemExit(f"unknown setting(s) in {path}: {', '.join(unknown)}; allowed: {', '.join(sorted(KEYS))}")
    for key in ("video", "car"):
        if not isinstance(cfg.get(key), str):
            raise SystemExit(f"'{key}' in {path} must be text")
    every = cfg.get("every_seconds", 0.5)
    if isinstance(every, bool) or not isinstance(every, (int, float)) or every <= 0:
        raise SystemExit(f"'every_seconds' in {path} must be a number above 0")
    bgs = cfg.get("backgrounds", ["all"])
    if not isinstance(bgs, list) or not bgs or not all(isinstance(b, str) for b in bgs):
        raise SystemExit(f"'backgrounds' in {path} must be a list of names, e.g. [\"all\"]")
    hosted = cfg.get("hosted", "config")
    if hosted not in (True, False, "config"):
        raise SystemExit(f"'hosted' in {path} must be true, false or \"config\"")
    return {"video": (path.parent / cfg["video"]).resolve(), "every": float(every), "car": cfg["car"],
            "backgrounds": bgs, "hosted": hosted, "output": (path.parent / cfg.get("output", "output")).resolve()}


def py(tool: Path) -> str:
    exe = tool / ".venv" / "Scripts" / "python.exe"
    if not exe.exists():
        raise SystemExit(f"no venv for {tool.name}: expected {exe}")
    return str(exe)


def run(cmd: list, cwd: Path, log) -> float:
    """Run one step, echoing its output to the console and run.log; return the seconds it took."""
    line = "> " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n{line}", flush=True)
    log.write(f"\n{line}\n")
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=env) as p:
        for out in p.stdout:
            sys.stdout.write(out)
            log.write(out)
        log.flush()
        code = p.wait()
    if code != 0:
        raise SystemExit(f"step failed (exit {code}): {Path(cmd[1]).name}")
    return time.perf_counter() - t0


# ---------------------------------------------------------------- reading the tools' outputs
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_image(path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def plate_picture(masked: Path, det_rows: list[dict], out: Path) -> str:
    """The masked frame with each detected plate's box (orange) and fitted corners (green). Report only."""
    img = read_image(masked)
    if img is None:
        return "masked frame missing"
    t = max(2, round(max(img.shape[:2]) / 400))
    found = [r for r in det_rows if r.get("x1")]
    for r in found:
        x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), t)
        pts = [tuple(int(float(v)) for v in c.split(",")) for c in r.get("corners", "").split()]
        if len(pts) == 4:
            cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 210, 0), t)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(out))
    if not found:
        return "no plate found: frame unchanged"
    return "; ".join(f"plate {r.get('plate')}: {r.get('fit') or 'box'} fit, conf {r.get('confidence')}" for r in found)


STEP_TEXT = {
    "01_photo": "Photo as read (plate already masked; glass already tinted when hosted)",
    "02_cutout_model": "Cut-out model's raw mask",
    "03_cutout_cleanup": "Mask clean-up",
    "04_cutout_isolation": "SAM 3 keeps only this car",
    "05_cutout_result": "Final cut-out",
    "05a_glass_mask": "SAM 3 glass mask",
    "05b_glass_tint": "Glass tint before / after",
    "05c_hosted_camera": "Hosted camera (MoGe-3 + VGGT)",
    "06_camera_lines": "Camera from background lines",
    "07_angle_search": "View angle search",
    "08_outline_fit": "3D car outline fit",
    "09_outline_compare": "Outline comparison",
    "10_level": "Levelling (one rotation)",
    "11_tyres": "Tyre contacts",
    "12_room": "Room re-rendered from the photo's camera",
    "12_flat_placement": "Flat placement (no room)",
    "13_room_surfaces": "Room surfaces",
    "14_shadows": "Shadows and reflection",
    "15_background_ready": "Background ready",
    "16_placed": "Placed, with marks",
}


def sort_key(p: Path):
    m = re.match(r"(\d+)([a-z]?)_", p.stem)
    return (int(m.group(1)), m.group(2)) if m else (99, p.stem)


# ---------------------------------------------------------------- the report
CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#66665f;--line:#deddd6;--accent:#2f6fde;--warn:#b5560b;
--ok:#227a3c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#17181a;--card:#212226;--ink:#ececea;
--muted:#a3a39c;--line:#35363b;--accent:#7aa7ff;--warn:#f0a35e;--ok:#6fcf8a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:36px 0 8px}h3{font-size:16px;margin:18px 0 6px;color:var(--muted)}
.muted{color:var(--muted)}.warn{color:var(--warn)}.ok{color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}
.strip figure{flex:0 0 auto;width:92px;margin:0;opacity:.55}.strip figure.pick{opacity:1}
.strip figure.pick img{outline:3px solid var(--accent)}
.strip img{width:92px;border-radius:4px;display:block}.strip figcaption{font-size:11px;line-height:1.25}
.steps{display:flex;gap:10px;overflow-x:auto;padding:4px 0 10px;scroll-snap-type:x proximity}
.steps figure{flex:0 0 auto;width:300px;margin:0;scroll-snap-align:start}
.steps img{width:100%;border-radius:6px;border:1px solid var(--line);display:block;background:#000}
.steps figcaption{font-size:12.5px;margin-top:4px}.steps b{display:block;font-size:13px}
.n{display:inline-block;min-width:22px;padding:0 5px;border-radius:9px;background:var(--accent);color:#fff;
font-size:11px;text-align:center;margin-right:4px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);
padding:5px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.tablewrap{overflow-x:auto}nav a{margin-right:12px;color:var(--accent)}
"""


def rel(path: Path, base: Path) -> str:
    return html.escape(Path(os.path.relpath(path, base)).as_posix(), quote=True)


def fig(path: Path, base: Path, n: int, title: str, text: str = "") -> str:
    src = rel(path, base)
    return (f'<figure><a href="{src}" target="_blank"><img loading="lazy" src="{src}" alt="{html.escape(title)}"></a>'
            f'<figcaption><b><span class="n">{n}</span>{html.escape(title)}</b>{html.escape(text)}</figcaption></figure>')


def build_report(w: Path, stem: str, cfg: dict) -> Path:
    sides_name = f"{stem}_sides"
    sampled = read_csv(VIEW / "output" / stem / "results.csv")
    picked = read_csv(VIEW / "output" / sides_name / "results.csv")
    masked_dir = w / "masked" / sides_name
    dets = read_csv(masked_dir / "detections.csv")
    bg_dir = w / "background" / sides_name
    results = read_csv(bg_dir / "results.csv")
    times = read_csv(w / TIMES_FILE)
    picked_files = {r["file"] for r in picked}
    by_side = {r["picked_for"]: r for r in picked}
    missing = [s for s in SIDES if s not in by_side]
    bgs = sorted({r["background"] for r in results if r.get("background")})
    log = (w / "run.log").read_text(encoding="utf-8", errors="replace") if (w / "run.log").exists() else ""
    spend = re.findall(r"hosted: .*fal call.*", log)

    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f"<meta name=viewport content='width=device-width,initial-scale=1'><title>E2E Steps {html.escape(stem)}"
           f"</title><style>{CSS}</style></head><body><main>",
           f"<h1>End-to-end run: {html.escape(cfg['video'].name)}</h1>",
           f"<p class=muted>{html.escape(cfg['car'])} &middot; frames every {cfg['every']:g} s &middot; "
           f"backgrounds: {html.escape(', '.join(bgs) or '-')} &middot; hosted: {html.escape(str(cfg['hosted']))} "
           f"&middot; {time.strftime('%Y-%m-%d %H:%M')}</p>"]
    nav = " ".join(f'<a href="#{s}">{s}</a>' for s in SIDES if s in by_side)
    out.append(f"<nav>{nav} <a href='#summary'>summary</a></nav>")

    # Step 1 overview: every sampled frame, the picked ones outlined
    out.append("<h2>Step 1 &middot; frame_view_type: every sampled frame, best one per side outlined</h2>")
    strip = []
    for r in sampled:
        cls = " class=pick" if r["file"] in picked_files else ""
        src = rel(VIEW / "output" / stem / r["file"], w)
        strip.append(f'<figure{cls}><a href="{src}" target=_blank><img loading=lazy src="{src}" alt=""></a>'
                     f'<figcaption>{float(r["time_s"]):.1f}s {html.escape(r["side"])}<br>'
                     f'{float(r["confidence"]):.0%} &middot; {float(r["angle_deg"]):.0f}&deg;</figcaption></figure>')
    out.append(f"<div class='card'><div class=strip>{''.join(strip)}</div>")
    out.append(f"<p>{len(sampled)} frames sampled; <b>{len(picked)} of 8 sides</b> picked.")
    if missing:
        out.append(f" <span class=warn>Not in the video: {', '.join(missing)}</span> (no frame reaches the "
                   "minimum probability for them; they are left out, not faked).")
    out.append("</p></div>")

    # One section per side
    for side in SIDES:
        row = by_side.get(side)
        if row is None:
            continue
        name = row["file"]
        p = Path(name).stem
        out.append(f"<h2 id='{side}'>{side} &middot; <span class=muted>{html.escape(name)}</span></h2>")
        n = 1
        figs = [fig(VIEW / "output" / sides_name / name, w, n, "Video frame + side label",
                    f"label {row['side']} {float(row['confidence']):.0%}, angle {float(row['angle_deg']):.0f} deg, "
                    f"picked for {side} (p {float(row['picked_prob']):.0%})")]
        n += 1
        plate_pic = w / "plates" / f"{p}.jpg"
        text = plate_picture(masked_dir / name, [d for d in dets if d.get("file") == name], plate_pic)
        figs.append(fig(plate_pic, w, n, "Number plate masked", text + " (orange = detector box, green = corners)"))
        dbg = bg_dir / "debug" / p
        for pic in sorted(dbg.glob("*.jpg"), key=sort_key) if dbg.is_dir() else []:
            n += 1
            figs.append(fig(pic, w, n, STEP_TEXT.get(pic.stem, pic.stem), pic.name))
        out.append(f"<h3>Per photo: label, plate, cut-out, glass, camera, 3D fit</h3><div class=steps>{''.join(figs)}</div>")
        for bg in bgs:
            figs = []
            for pic in sorted((dbg / bg).glob("*.jpg"), key=sort_key) if (dbg / bg).is_dir() else []:
                n += 1
                figs.append(fig(pic, w, n, STEP_TEXT.get(pic.stem, pic.stem), f"{bg}/{pic.name}"))
            final = bg_dir / bg / f"{p}.webp"
            if final.exists():
                n += 1
                figs.append(fig(final, w, n, "Final result", f"{bg}/{final.name}"))
            r = next((x for x in results if x["file"] == name and x.get("background") == bg), None)
            note = ""
            if r is not None:
                flags = r.get("flags") or "none"
                note = (f"<p class=muted>flags: {html.escape(flags)}"
                        + (f" &middot; note: {html.escape(r['note'])}" if r.get("note") else "")
                        + (f" &middot; fit IoU {html.escape(r['fit_iou'])}" if r.get("fit_iou") else "") + "</p>")
            if figs or note:
                out.append(f"<h3>On {html.escape(bg)}</h3>{note}<div class=steps>{''.join(figs)}</div>")
        orig = bg_dir / "originals" / name
        if orig.exists():
            out.append(f"<p class=muted>Untouched photo kept at <a href='{rel(orig, w)}' target=_blank>"
                       f"originals/{html.escape(name)}</a>.</p>")

    # Summary
    out.append("<h2 id=summary>Summary</h2><div class=card>")
    if times:
        out.append("<table><tr><th>step</th><th>seconds</th></tr>"
                   + "".join(f"<tr><td>{html.escape(t['step'])}</td><td>{float(t['seconds']):.1f}</td></tr>"
                             for t in times) + "</table>")
    if spend:
        out.append(f"<p>{html.escape(spend[-1])}</p>")
    out.append("</div>")
    cols = [c for c in ("file", "background", "view_label", "yaw", "yaw_source", "fit_iou", "camera_source",
                        "focal_px", "focal_source", "cam_height_m", "glass_pct", "sam3_status", "cost_usd",
                        "flags", "note") if results and c in results[0]]
    if results:
        out.append("<h3>results.csv</h3><div class='card tablewrap'><table><tr>"
                   + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
                   + "".join("<tr>" + "".join(f"<td>{html.escape(r.get(c) or '')}</td>" for c in cols) + "</tr>"
                             for r in results) + "</table></div>")
    out.append("</main></body></html>")
    index = w / "index.html"
    index.write_text("".join(out), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end test run with a picture for every step (settings: e2e.toml)")
    ap.add_argument("--config", type=Path, default=ROOT / "e2e.toml")
    ap.add_argument("--report-only", action="store_true", help="only rebuild index.html from the last run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    video = cfg["video"]
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    stem = video.stem
    w = cfg["output"] / f"e2e_{stem}"
    if not args.report_only:
        cache = w / "background" / "_hosted_cache"  # fal answers: kept, so a rerun costs nothing for unchanged photos
        kept = cfg["output"] / "_e2e_hosted_cache"
        if cache.is_dir():
            if kept.exists():
                shutil.rmtree(kept)
            cache.rename(kept)
        if w.exists():  # a test run starts clean: stale results would mix into the report
            shutil.rmtree(w)
        w.mkdir(parents=True)
        if kept.is_dir():
            cache.parent.mkdir(parents=True)
            kept.rename(cache)
        frames = VIEW / "output" / stem
        if frames.exists():  # frame_view_type never deletes old frames (stale-frames gotcha)
            shutil.rmtree(frames)
        times = []
        with open(w / "run.log", "w", encoding="utf-8") as log:
            times.append(("1 frame_view_type: sample, label, pick sides", run(
                [py(VIEW), VIEW / "classify_video.py", video, "--every", f"{cfg['every']:g}", "--pick-sides"],
                VIEW, log)))
            sides = VIEW / "output" / f"{stem}_sides"
            if not any(sides.glob("*.jpg")):
                raise SystemExit(f"no frames picked in {sides}")
            times.append(("2 number_plate_detection: mask plates", run(
                [py(PLATES), PLATES / "mask_plates.py", sides, "--out", w / "masked"], PLATES, log)))
            bg_args = [a for b in cfg["backgrounds"] for a in ("--background", b)]
            hosted = [] if cfg["hosted"] == "config" else ["--hosted" if cfg["hosted"] else "--no-hosted"]
            times.append(("3 background_changer: cut-out, glass, camera, fit, rooms", run(
                [py(BGCHANGER), BGCHANGER / "change_background.py", w / "masked" / sides.name, "--out",
                 w / "background", "--car", cfg["car"], *bg_args, "--background-view", "room", "--debug", *hosted],
                BGCHANGER, log)))
        with open(w / TIMES_FILE, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["step", "seconds"])
            wr.writerows((s, round(t, 1)) for s, t in times)
    index = build_report(w, stem, cfg)
    print(f"\nreport: {index}")


if __name__ == "__main__":
    main()
