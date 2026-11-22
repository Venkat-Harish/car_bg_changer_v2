"""Make the colour test set: sample plates pasted onto real car photos, with the right answers.

1. Draw one flat plate image per standard colour into assets/test_plates. Existing files are
   kept, so your edits survive (--redraw draws them again). Any other image in that folder is
   tested too.
2. For 3 views of the car (front, front angled, rear angled) from frame_view_type, find the
   real plate's outline and how bright its white is: that is the true light at that spot.
   Paste every sample plate there, lit the same way, slightly softened to match the camera.
   Plus one hard case, night_flash.
3. Write assets/test_photos/<case>/<view>.jpg and assets/test_photos/expected.csv (the true
   outline, the true brightness and the expected kind of reading for every photo).

Usage (PowerShell, from the number_plate_detection folder):
  .venv\\Scripts\\python tests\\make_test_photos.py
  .venv\\Scripts\\python tests\\make_test_photos.py --redraw
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plate_brightness_detector.colours import MIN_FACTOR, PAINT_FACTORS, name_colour  # noqa: E402
from plate_masker import PlateDetector, fit_plate, grow_quad, quad_size  # noqa: E402
from plate_masker.masking import image_overlay  # noqa: E402

PLATES_DIR = ROOT / "assets" / "test_plates"
PHOTOS_DIR = ROOT / "assets" / "test_photos"
FRAMES_DIR = ROOT.parent / "frame_view_type" / "output" / "Final_zoomed_car"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# View name -> start of the frame's file name (its time in the video).
VIEWS = {"front": "0000.0s_", "front_angled": "0034.0s_", "rear_angled": "0024.0s_"}

PLATE_W, PLATE_H = 1040, 224  # 520 x 112 mm at 2 px per mm
PLATE_TEXT = "TEST 123"  # generic, not a real registration

# name: (background RGB, characters RGB, what it stands for). Also the test order.
STANDARD_PLATES = {
    "white_black": ((255, 255, 255), (20, 20, 20), "white, black letters (most of Europe, India private cars)"),
    "white_red": ((255, 255, 255), (200, 30, 40), "white, red letters (Belgium)"),
    "yellow_black": ((255, 200, 0), (20, 20, 20), "yellow, black letters (taxis and commercial, UK rear, Netherlands)"),
    "green_white": ((0, 120, 70), (255, 255, 255), "green, white letters (electric cars, e.g. India)"),
    "green_yellow": ((0, 120, 70), (255, 200, 0), "green, yellow letters (electric taxis or commercial, e.g. India)"),
    "green_black": ((0, 120, 70), (20, 20, 20), "green, black letters (no white or light paint: tests the photo fallback)"),
    "blue_white": ((20, 60, 170), (255, 255, 255), "blue, white letters (e.g. China)"),
    "red_white": ((190, 25, 35), (255, 255, 255), "red, white letters (e.g. temporary or diplomatic plates)"),
    "black_white": ((20, 20, 20), (255, 255, 255), "black, white letters (e.g. rental or vintage cars)"),
    "black_yellow": ((20, 20, 20), (255, 200, 0),
                     "black, yellow letters, no border line (self-drive rental cars, e.g. India): outline from the letters"),
}
# Plates drawn in another style, like the real plates they stand for. black_yellow copies the
# Indian rental plates in sample_vid: no border line, long condensed text filling the plate.
PLATE_STYLES = {
    "black_yellow": {"border": False, "text": "AB 12 CD 3456", "text_h": 0.62, "text_w": 0.92},
}
NIGHT_DARKEN = 0.25  # night_flash: the scene at 25 % brightness, the plate glowing at 100 %

FIELDS = ["case", "view", "file", "plate", "expected_reading", "true_level", "true_pct", "outline"]


def bgr(rgb) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def draw_plate(background, characters, border: bool = True, text: str | None = None,
               text_h: float = 0.0, text_w: float = 0.0) -> np.ndarray:
    """A flat plate: background, a border line in the character colour, bold text.

    Line and rim sizes follow the real plate in the test video: a border line about 5 % of the
    plate's height, with a rim of background colour about 4.5 % wide outside it.
    With `text`, the letters are drawn text_h of the plate's height tall and squeezed to at most
    text_w of its width (condensed, like real plate fonts).
    """
    img = np.full((PLATE_H, PLATE_W, 3), bgr(background), np.uint8)
    rim, line = 10, 12
    if border:
        cv2.rectangle(img, (rim + line // 2, rim + line // 2),
                      (PLATE_W - 1 - rim - line // 2, PLATE_H - 1 - rim - line // 2), bgr(characters), line, cv2.LINE_AA)
    if text:
        return _draw_condensed_text(img, text, characters, text_h, text_w)
    font, thick = cv2.FONT_HERSHEY_DUPLEX, 20
    scale = 6.0
    while scale > 1:
        (tw, th), _ = cv2.getTextSize(PLATE_TEXT, font, scale, thick)
        if tw <= 0.84 * PLATE_W and th <= 0.58 * PLATE_H:
            break
        scale -= 0.1
    cv2.putText(img, PLATE_TEXT, ((PLATE_W - tw) // 2, (PLATE_H + th) // 2), font, scale, bgr(characters), thick,
                cv2.LINE_AA)
    return img


def _draw_condensed_text(img: np.ndarray, text: str, colour, text_h: float, text_w: float) -> np.ndarray:
    """Draw `text` centred, its ink text_h of the plate's height tall, squeezed to <= text_w of its width."""
    font, thick = cv2.FONT_HERSHEY_DUPLEX, 22
    (tw, th), base = cv2.getTextSize(text, font, 4.0, thick)
    ink = np.zeros((th + base + 4 * thick, tw + 4 * thick), np.uint8)
    cv2.putText(ink, text, (2 * thick, th + 2 * thick), font, 4.0, 255, thick, cv2.LINE_AA)
    ys, xs = np.nonzero(ink)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height = int(text_h * PLATE_H)
    width = min(int(ink.shape[1] * height / ink.shape[0]), int(text_w * PLATE_W))
    ink = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, None] / 255
    y0, x0 = (PLATE_H - height) // 2, (PLATE_W - width) // 2
    roi = img[y0:y0 + height, x0:x0 + width].astype(np.float32)
    img[y0:y0 + height, x0:x0 + width] = (roi * (1 - ink) + np.float32(bgr(colour)) * ink + 0.5).astype(np.uint8)
    return img


def expected_reading(colour_a_bgr, colour_b_bgr) -> str:
    """What plate_brightness_detector should read the light from, judging by the plate's two colours."""
    luma = lambda c: 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]  # noqa: E731
    lighter = max(colour_a_bgr, colour_b_bgr, key=luma)
    name = name_colour(lighter)
    if name == "neutral":
        return "white"
    return "light paint" if PAINT_FACTORS.get(name, 0.0) >= MIN_FACTOR else "photo"


def two_colours(img_bgr: np.ndarray):
    """The two main colours of a plate image (k-means), as BGR tuples."""
    data = img_bgr.reshape(-1, 3).astype(np.float32)
    data = data[np.linspace(0, len(data) - 1, min(len(data), 20000)).astype(int)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return [tuple(float(v) for v in c) for c in centres]


def read_image(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(path))


def prepare_plates(plates_dir: Path, redraw: bool) -> list[tuple[str, np.ndarray, str, str]]:
    """Draw missing standard plates; return (case, image, description, expected reading) for every plate image."""
    plates_dir.mkdir(parents=True, exist_ok=True)
    for name, (background, characters, _) in STANDARD_PLATES.items():
        path = plates_dir / f"{name}.png"
        if redraw or not path.exists():
            cv2.imencode(".png", draw_plate(background, characters, **PLATE_STYLES.get(name, {})))[1].tofile(str(path))
            print(f"drew {path.relative_to(ROOT)}")
    cases = []
    files = {p.stem: p for p in plates_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    order = [n for n in STANDARD_PLATES if n in files] + sorted(n for n in files if n not in STANDARD_PLATES)
    for name in order:
        img = read_image(files[name])
        if img is None:
            print(f"skip (cannot read): {files[name].name}")
            continue
        if name in STANDARD_PLATES:
            background, characters, about = STANDARD_PLATES[name]
            reading = expected_reading(bgr(background), bgr(characters))
        else:
            about = "your own plate image"
            reading = expected_reading(*two_colours(img))
        cases.append((name, img, about, reading))
    return cases


def plate_white(image: np.ndarray, quad: np.ndarray) -> float:
    """How bright the real (white) plate's light parts are: the true light at that spot (0-255)."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    _, height = quad_size(quad)
    inner = cv2.distanceTransform(mask, cv2.DIST_L2, 3) > 0.16 * height  # clear of border and rim
    vals = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[inner]
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.percentile(vals[vals > thr], 80))


def paste(frame: np.ndarray, quad: np.ndarray, plate: np.ndarray, level: float) -> np.ndarray:
    """Paste `plate` onto `quad`, lit to `level`, softened a little to look like the camera's plate."""
    out = frame.copy()
    image_overlay(out, quad, overlay=plate, light_plane=(level, 0.0, 0.0))
    _, height = quad_size(quad)
    edge = grow_quad(quad, [2.0] * 4)
    x0, y0 = np.maximum(np.floor(edge.min(0)).astype(int), 0)
    x1, y1 = np.ceil(edge.max(0)).astype(int) + 1
    roi = out[y0:y1, x0:x1]
    soft = cv2.GaussianBlur(roi, (0, 0), max(0.6, 0.008 * height))
    inside = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, np.round(edge - np.float32([x0, y0])).astype(np.int32), 1)
    roi[inside > 0] = soft[inside > 0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the colour test set in assets/test_photos.")
    parser.add_argument("--plates", type=Path, default=PLATES_DIR, help="folder of sample plate images")
    parser.add_argument("--out", type=Path, default=PHOTOS_DIR, help="where the test photos go")
    parser.add_argument("--frames", type=Path, default=FRAMES_DIR, help="folder with the car frames")
    parser.add_argument("--redraw", action="store_true", help="draw the standard plates again (overwrites them)")
    args = parser.parse_args()

    cases = prepare_plates(args.plates, args.redraw)
    if not cases:
        raise SystemExit(f"no plate images in {args.plates}")

    detector = PlateDetector()
    views = {}
    for view, prefix in VIEWS.items():
        found = sorted(args.frames.glob(f"{prefix}*.jpg")) if args.frames.is_dir() else []
        if not found:
            raise SystemExit(f"no frame starting with {prefix} in {args.frames}\n"
                             "Run frame_view_type's classify_video.py first, or pass --frames")
        frame = read_image(found[0])
        plates = [fit_plate(frame, p) for p in detector.detect(frame)]
        plates = [p for p in plates if p.corners is not None]
        if not plates:
            raise SystemExit(f"could not find the plate's outline in {found[0].name}")
        main_plate = max(plates, key=lambda p: p.width * p.height)  # the car in front of the camera
        quad = main_plate.quad
        views[view] = (frame, quad, plate_white(frame, quad))
        print(f"view {view:<12} from {found[0].name}: plate white {views[view][2]:.0f} ({views[view][2] / 2.55:.0f} %)")

    # Remove folders of cases made last time that no longer exist (never touches other folders).
    expected_csv = args.out / "expected.csv"
    names = {c[0] for c in cases} | {"night_flash"}
    if expected_csv.exists():
        with open(expected_csv, encoding="utf-8-sig") as f:
            old = {r["case"] for r in csv.DictReader(f)}
        for gone in sorted(old - names):
            if (args.out / gone).is_dir():
                shutil.rmtree(args.out / gone)
                print(f"removed old case {gone}")

    rows = []

    def add(case, view, image, quad, about, reading, level):
        file = f"{case}/{view}.jpg"
        write_jpeg(args.out / file, image)
        rows.append({
            "case": case, "view": view, "file": file, "plate": about, "expected_reading": reading,
            "true_level": "" if level is None else round(level, 1),
            "true_pct": "" if level is None else round(100 * level / 255, 1),
            "outline": " ".join(f"{x:.1f},{y:.1f}" for x, y in quad),
        })

    for case, plate, about, reading in cases:
        for view, (frame, quad, level) in views.items():
            add(case, view, paste(frame, quad, plate, level), quad, about, reading, level)
        print(f"made {case:<14} ({reading}): {about}")

    white = next((c[1] for c in cases if c[0] == "white_black"), draw_plate((255, 255, 255), (20, 20, 20)))
    for view, (frame, quad, _) in views.items():
        dark = (frame.astype(np.float32) * NIGHT_DARKEN).astype(np.uint8)
        add("night_flash", view, paste(dark, quad, white, 255.0), quad,
            "night: dark scene, white plate glowing in a flash (expect the photo reading)", "photo", None)
    print("made night_flash    (photo): dark scene, plate glowing in a flash")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(expected_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} test photos in {args.out} ({len(cases) + 1} cases x {len(views)} views), answers in {expected_csv.name}")


if __name__ == "__main__":
    main()
