"""Debug pictures for every step of change_background.py (debug_mode = true in config.toml, or --debug).

Each photo gets its own folder, <output_folder>\\<images folder name>\\debug\\<photo>\\, with one picture per
step, numbered in the order the steps run. Every picture has a caption strip on top: what the step did, its
numbers, and what each colour means.

  01_photo.jpg               the photo as read
  02_cutout_model.jpg        the cut-out model's raw mask: white = car, black = background, grey = unsure
  03_cutout_cleanup.jpg      how that mask became the car's outline: blobs dropped, posts cut off, objects
                             removed (car_isolation), window holes filled, the soft edge band
  04_cutout_isolation.jpg    (car_isolation on) the car SAM found, other vehicles and what was removed
  05_cutout_result.jpg       the final mask and the car alone on a checkerboard (what gets pasted)
 [hosted] on:
  05a_glass_mask.jpg         SAM 3's window mask on the photo: the only pixels the glass tint may change
  05b_glass_tint.jpg         the photo before and after the tint, and the changed pixels outlined
  05c_hosted_camera.jpg      (room mode) the MoGe-3 + VGGT camera: focal, pitch, roll, height, angle between shots,
                             its horizon, and which camera the 3D fit used
 room mode (background_view = "room"):
  06_camera_lines.jpg        the camera's tilt from the background's straight lines: segments by direction,
                             vanishing directions, the horizon from the lines and from the car's 3D fit
  07_angle_search.jpg        how the car's view angle (yaw) was found: outline match at every 10 degrees, the
                             candidates refined, the winner; the camera's distance, height and angle down
  08_outline_fit.jpg         the photo's car outline (red) and the fitted 3D car's outline (green) on the photo
  09_outline_compare.jpg     the same two outlines filled (both / only the photo / only the 3D car), and every
                             refined candidate angle the same way
  10_level.jpg               car alignment: the car's bottom line before and after its one rotation
  11_tyres.jpg               tyre bottoms on the levelled car: predicted, found, and after the contact lock
  <background>\\12_room.jpg             the background's 3D room re-rendered from the photo's own camera
  <background>\\13_room_surfaces.jpg    floor / walls / ceiling, and areas the background picture never showed
  <background>\\14_shadows.jpg          what the shadows and the floor reflection changed
  <background>\\15_background_ready.jpg the background just before the car is pasted
  <background>\\16_placed.jpg           the result with footprint, stage, horizon and tyre marks
  <background>\\12_flat_placement.jpg   (a background without a 3D room) the flat placement's overlay
 flat mode keeps its overlay where it always was: <output>\\<background>\\debug\\<photo>.jpg

Only pictures: nothing here changes a result. Pictures are at most 1600 px on the long side.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from car_fit_to_bg.ground import project_rc
from car_fit_to_bg.level import bottom_profile
from car_fit_to_bg.lines import line_camera
from car_fit_to_bg.pose import body_points, car_to_camera, render_proxy

MAX_LONG = 1600  # px: debug pictures are shrunk to this
MIN_WIDTH = 720  # px: the caption's width; narrower pictures are enlarged to it
QUALITY = 85
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (255, 255, 255), (200, 200, 200)
RED, GREEN, BLUE, YELLOW = (0, 0, 255), (0, 210, 0), (255, 120, 0), (0, 230, 255)
ORANGE, MAGENTA, CYAN, PURPLE = (0, 140, 255), (255, 0, 255), (255, 255, 0), (200, 80, 160)
STATUS_COLOUR = {"found": GREEN, "occluded": ORANGE, "cut": RED, "implausible": MAGENTA}
CAM_SOURCE = {"lines": "this photo's background lines",
              "neighbours": "neighbouring video frames (this photo's own lines were too weak)",
              "none": "no usable background lines: the car's outline alone sets the tilt"}
FOCAL_SOURCE = {"config": "[fit] focal in config.toml", "pooled": "the folder's background lines (pooled)",
                "lines": "this photo's lines", "exif": "the photo's EXIF", "lines-weak": "uncertain lines",
                "prior": "a typical phone lens (nothing measured)"}


# ---------------------------------------------------------------- drawing helpers
def thick(shape) -> int:
    """Line width that stays visible after the picture is shrunk to MAX_LONG."""
    return max(1, round(max(shape[:2]) / 700))


def shrink(img: np.ndarray, max_long: int = MAX_LONG) -> np.ndarray:
    h, w = img.shape[:2]
    k = max_long / max(h, w)
    if k >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _wrap(text: str, width: int, scale: float) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(trial, FONT, scale, 1)[0][0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def captioned(img: np.ndarray, title: str, lines=()) -> np.ndarray:
    """The picture (shrunk) under a dark strip with the title and text lines; a line given as (text, colour)
    is written in that colour (the colour it has in the picture)."""
    img = bgr(shrink(img))
    h0, w0 = img.shape[:2]
    if w0 < MIN_WIDTH:  # a small photo (e.g. a phone video frame): enlarged to the caption's width
        k = min(MIN_WIDTH / w0, MAX_LONG / max(h0, w0))
        if k > 1.05:
            img = cv2.resize(img, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_LINEAR)
    w = max(img.shape[1], MIN_WIDTH)
    if img.shape[1] < w:
        img = cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=(24, 24, 24))
    rows = [(title, WHITE, 0.7, 2)]
    for line in lines:
        text, col = (line, GREY) if isinstance(line, str) else line
        rows += [(part, col, 0.5, 1) for part in _wrap(text, w - 20, 0.5)]
    hh = 14 + sum(30 if s > 0.6 else 22 for _, _, s, _ in rows)
    head = np.full((hh, w, 3), 32, np.uint8)
    y = 6
    for text, col, s, t in rows:
        y += 30 if s > 0.6 else 22
        cv2.putText(head, text, (10, y - 6), FONT, s, col, t, cv2.LINE_AA)
    return np.vstack([head, img])


def checker(h: int, w: int, n: int = 16) -> np.ndarray:
    yy, xx = np.indices((h, w))
    v = np.where(((yy // n) + (xx // n)) % 2 == 1, 205, 160).astype(np.uint8)
    return cv2.merge([v, v, v])


def on_checker(pixels: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha.astype(np.float32)[..., None]
    out = pixels.astype(np.float32) * a + checker(*alpha.shape).astype(np.float32) * (1 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def paint(img: np.ndarray, mask: np.ndarray, colour, a: float = 0.55) -> None:
    if mask is not None and mask.any():
        img[mask] = (img[mask] * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def outline(img: np.ndarray, mask: np.ndarray, colour, t: int) -> None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, t, cv2.LINE_AA)


def label(img: np.ndarray, text: str, xy, colour, scale: float | None = None) -> None:
    """Text on a dark box (readable on any background), kept inside the picture."""
    s = scale or max(0.5, 0.45 * max(img.shape[:2]) / 1000)
    t = max(1, round(s * 2))
    (tw, th), base = cv2.getTextSize(text, FONT, s, t)
    H, W = img.shape[:2]
    x = int(min(max(2, round(xy[0])), max(2, W - tw - 4)))
    y = int(min(max(th + 4, round(xy[1])), max(th + 4, H - base - 4)))
    x1, y1, x2, y2 = x - 2, y - th - 3, min(W, x + tw + 2), min(H, y + base + 2)
    img[y1:y2, x1:x2] = (img[y1:y2, x1:x2].astype(np.float32) * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, s, colour, t, cv2.LINE_AA)


def side_by_side(*imgs: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    parts = []
    for i, im in enumerate(imgs):
        im = bgr(im)
        if im.shape[0] < h:
            im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        parts.append(im)
        if i < len(imgs) - 1:
            parts.append(np.full((h, gap, 3), 24, np.uint8))
    return np.hstack(parts)


def grid(imgs: list[np.ndarray], cols: int, gap: int = 8) -> np.ndarray:
    rows = [side_by_side(*imgs[i:i + cols], gap=gap) for i in range(0, len(imgs), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, gap, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(24, 24, 24)) for r in rows]
    return np.vstack(rows)


def box_of(mask: np.ndarray, margin: float = 0.06):
    """Bounding box (x1, y1, x2, y2) of a mask, grown by `margin` of its size, inside the picture."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    mx, my = int(margin * (x2 - x1)) + 4, int(margin * (y2 - y1)) + 4
    return max(0, x1 - mx), max(0, y1 - my), min(mask.shape[1], x2 + mx), min(mask.shape[0], y2 + my)


def line_through(img: np.ndarray, l, colour, t: int) -> None:
    """Draw the image line a x + b y + c = 0."""
    a, b, c = (float(v) for v in l)
    H, W = img.shape[:2]
    if abs(b) > 1e-9:
        p = [(x, -(a * x + c) / b) for x in (-W, 2 * W)]
    elif abs(a) > 1e-9:
        p = [(-c / a, y) for y in (-H, 2 * H)]
    else:
        return
    if all(abs(v) < 1e7 for q in p for v in q):
        cv2.line(img, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), colour, t, cv2.LINE_AA)


def thirds_line(mask: np.ndarray):
    """The car's bottom line as levelling measures it (level.thirds_tilt): the lowest outline point in the
    left third to the lowest in the right third. ((x, y), (x, y), tilt deg + rising right) or None."""
    if mask.sum() < 50:
        return None
    u, v = bottom_profile(mask)
    o = np.argsort(u)
    u, v = u[o], v[o]
    n = len(u) // 3
    if n < 2:
        return None
    i, j = int(np.argmax(v[:n])), len(u) - n + int(np.argmax(v[-n:]))
    return (u[i], v[i]), (u[j], v[j]), math.degrees(math.atan2(v[i] - v[j], u[j] - u[i]))


def draw_bottom_line(img: np.ndarray, line, t: int) -> None:
    if line is None:
        return
    (x1, y1), (x2, y2), _ = line
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), YELLOW, t, cv2.LINE_AA)
    for x in range(int(x1), int(x2), max(4, 6 * t)):  # dashed level reference from the left point
        cv2.line(img, (x, int(y1)), (min(int(x2), x + 3 * t), int(y1)), WHITE, max(1, t // 2), cv2.LINE_AA)


# ---------------------------------------------------------------- the writer
class DebugWriter:
    """Writes the step pictures under `root` (= <output>\\<images>\\debug)."""

    def __init__(self, root: Path):
        self.root = root

    def save(self, stem: str, name: str, img: np.ndarray, title: str, lines=(), bg: str | None = None) -> Path:
        folder = self.root / stem / bg if bg else self.root / stem
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.jpg"
        ok, buf = cv2.imencode(".jpg", captioned(img, title, lines), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if ok:
            buf.tofile(str(path))  # tofile: non-ASCII paths on Windows
        return path

    def write_readme(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "README.txt").write_text(__doc__, encoding="utf-8")


# ---------------------------------------------------------------- steps 1-5: the cut-out
def cutout_pictures(dw: DebugWriter, stem: str, image: np.ndarray, raw: np.ndarray, trace: dict, car,
                    model: str, isolation_overlay: np.ndarray | None = None, isolation_text: str = "") -> None:
    H, W = image.shape[:2]
    t = thick(image.shape)
    dw.save(stem, "01_photo", image, "1. The photo", [f"{W} x {H} px"])

    raw = raw.astype(np.float32) / 255.0 if raw.dtype == np.uint8 else raw
    soft = float(((raw > 0.05) & (raw < 0.95)).mean())
    dw.save(stem, "02_cutout_model", (np.clip(raw, 0, 1) * 255).astype(np.uint8),
            f"2. Cut-out model's raw mask ({model})",
            ["White = car, black = background, grey = unsure (soft edges, glass, shadows).",
             f"{100 * (raw >= 0.5).mean():.1f}% of the photo is above the 0.5 threshold; "
             f"{100 * soft:.1f}% is soft (between 0.05 and 0.95)."]
            + ([] if car is not None else ["NO CAR FOUND: the photo was skipped after this step."]))
    if car is None or "core" not in trace:
        return

    binary, posts, solid, core = trace["binary"], trace["posts"], trace["solid"], trace["core"]
    remove = trace.get("remove")
    over = (image.astype(np.float32) * 0.45).astype(np.uint8)
    kept = solid & binary
    dropped = binary & ~solid & ~posts
    removed = (remove & (raw >= 0.5)) if remove is not None else None
    filled = solid & ~binary
    band = solid & ~core
    paint(over, kept, (150, 150, 150), 0.45)
    paint(over, dropped, ORANGE)
    paint(over, posts, RED, 0.7)
    paint(over, removed, MAGENTA, 0.6)
    paint(over, filled, BLUE, 0.6)
    paint(over, band, YELLOW, 0.5)
    outline(over, solid, GREEN, t)
    area = max(1, int(solid.sum()))
    pct = lambda m: 0.0 if m is None else 100.0 * float(m.sum()) / area  # noqa: E731
    lines = [("grey = the car the model found, kept", GREY),
             (f"orange = other blobs dropped (not the biggest, most central one): {pct(dropped):.1f}% of the car's area",
              ORANGE),
             (f"red = thin posts/poles behind the car, cut off: {pct(posts):.1f}%", RED)]
    if remove is not None:
        lines.append((f"magenta = removed by car_isolation (other vehicles, objects): {pct(removed):.1f}%", MAGENTA))
    lines += [(f"blue = holes inside the outline filled (windows keep their glass): {pct(filled):.1f}%", BLUE),
              ("yellow = soft edge band (blended with the new background; inside it the car is copied pixel for pixel)",
               YELLOW),
              ("green = the car's final outline", GREEN)]
    dw.save(stem, "03_cutout_cleanup", over, "3. Mask clean-up: from the model's mask to the car's outline", lines)

    if isolation_overlay is not None:
        dw.save(stem, "04_cutout_isolation", isolation_overlay, "4. Car isolation (YOLOX + SAM)",
                [isolation_text, ("green = the car SAM found", GREEN), ("red = removed from the mask", RED),
                 ("orange boxes = other vehicles", ORANGE)])

    x1, y1, x2, y2 = box_of(car.alpha > 0, 0.04)
    a = car.alpha[y1:y2, x1:x2]
    cut = ", ".join(car.cut_sides) or "none"
    dw.save(stem, "05_cutout_result", side_by_side((a * 255).astype(np.uint8), on_checker(image[y1:y2, x1:x2], a)),
            "5. The cut-out: final mask (left) and the car alone (right)",
            [f"car box {car.box[2] - car.box[0]} x {car.box[3] - car.box[1]} px at ({car.box[0]}, {car.box[1]}); "
             f"touches the photo edge: {cut}",
             "This is what gets pasted: the car's own pixels, nothing re-coloured."])


# ---------------------------------------------------------------- [hosted]: glass tint (05a, 05b), camera (05c)
def glass_pictures(dw: DebugWriter, stem: str, before: np.ndarray, after: np.ndarray, window: np.ndarray | None,
                   car_alpha: np.ndarray, glass_pct=None) -> None:
    """05a: SAM 3's window mask on the photo; 05b: the photo before and after the tint, cropped to the car."""
    if window is None:
        return
    t = thick(before.shape)
    car = car_alpha >= 0.5
    over = (before.astype(np.float32) * 0.6).astype(np.uint8)
    paint(over, window, CYAN, 0.6)
    outline(over, window, CYAN, t)
    outline(over, car, GREEN, t)
    pct = glass_pct if glass_pct not in (None, "") else (100.0 * window.sum() / max(1, car.sum()))
    dw.save(stem, "05a_glass_mask", over, "5a. Glass found by SAM 3 (\"car window\" + \"windshield\")",
            [f"glass = {float(pct):.1f}% of the car's area; kept only inside the car mask, a little in from its edge",
             ("cyan = the window mask: the only pixels the tint may change", CYAN),
             ("green = the car's outline", GREEN)]
            + (["no glass found: the photo is used untouched"] if not window.any() else []))
    if not window.any():
        return
    changed = np.any(before != after, axis=2)
    x1, y1, x2, y2 = box_of(car, 0.04)
    b, a = before[y1:y2, x1:x2].copy(), after[y1:y2, x1:x2].copy()
    marked = a.copy()
    outline(marked, changed[y1:y2, x1:x2], MAGENTA, t)
    outside = int((changed & ~window).sum())
    dw.save(stem, "05b_glass_tint", side_by_side(b, a, marked),
            "5b. Glass tint: the photo (left), tinted (middle), changed pixels outlined (right)",
            [f"{int(changed.sum())} pixels changed, {outside} of them outside the window mask (must be 0)",
             "A fixed dark blue-grey gradient, the same on every photo of the car: nothing measured or generated.",
             "From here on the tinted frame is the photo; the untouched file is in originals\\.",
             ("magenta = outline of the changed pixels", MAGENTA)])


def hosted_camera_picture(dw: DebugWriter, stem: str, frame: np.ndarray, alpha: np.ndarray, ext, fit,
                          cols: dict | None = None) -> None:
    """05c: the hosted camera (MoGe-3 per photo + VGGT over the folder) and which camera the fit used."""
    H, W = alpha.shape
    t = thick(frame.shape)
    c = ((W - 1) / 2, (H - 1) / 2)
    img = (frame.astype(np.float32) * 0.55).astype(np.uint8)
    paint(img, alpha >= 0.5, (90, 90, 90), 0.6)

    def horizon(roll, pitch, f, colour):
        if roll is None or pitch is None or not f:
            return False
        r, p = math.radians(roll), math.radians(pitch)
        y_h = c[1] - f * math.tan(p)
        line_through(img, (math.tan(r), 1.0, -(y_h + c[0] * math.tan(r))), colour, t)
        return True

    cols = cols or {}
    lines = []
    if ext is None:
        lines.append("no hosted camera for this photo (MoGe-3 / VGGT failed or off): the background lines alone")
    else:
        f = ext.f_px or fit.f
        drew = horizon(ext.roll, ext.pitch, f, MAGENTA)
        fmt = lambda v, s: "-" if v is None else s.format(v)  # noqa: E731
        lines += [f"hosted: focal {fmt(ext.f_px, '{:.0f}')} px, pitch {fmt(ext.pitch, '{:+.2f}')} deg, roll "
                  f"{fmt(ext.roll, '{:+.2f}')} deg, camera height {fmt(ext.height_m, '{:.2f}')} m (MoGe-3 floor), "
                  f"angle from the first shot {fmt(ext.rel_yaw_deg, '{:.0f}')} deg (VGGT), confidence {ext.conf:.2f}",
                  f"MoGe-3 alone: focal {cols.get('moge_f') or '-'} px, height {cols.get('moge_h') or '-'} m; "
                  f"VGGT solve: {cols.get('vggt_ok') or '-'}"]
        if drew:
            lines.append(("magenta = horizon from the hosted camera", MAGENTA))
    cam = fit.cam
    horizon(cam.roll, cam.pitch, fit.f, CYAN)
    used = {"hosted": "the hosted camera", "lines": "this photo's own background lines (they disagreed with the "
            "hosted one by more than 5 deg: camera-lines-fallback)" if "camera-lines-fallback" in cam.flags
            else "this photo's own background lines", "neighbours": "neighbouring frames",
            "none": "no camera: the car's outline alone"}.get(cam.source, cam.source)
    lines += [f"the 3D fit used: {used}; focal {fit.f:.0f} px from {FOCAL_SOURCE.get(fit.focal_source, fit.focal_source)}",
              f"fitted camera height {fit.pose.h:.2f} m",
              ("cyan = horizon of the camera the fit used (on top of magenta when the hosted camera won)", CYAN)]
    dw.save(stem, "05c_hosted_camera", img, "5c. Camera from the hosted models (MoGe-3 + VGGT)", lines)


# ---------------------------------------------------------------- steps 6-11: camera, 3D fit, level, tyres
def _camera_lines(frame: np.ndarray, alpha: np.ndarray, fit) -> tuple[np.ndarray, list]:
    H, W = alpha.shape
    t = thick(frame.shape)
    detail: dict = {}
    try:
        line_camera(frame, alpha, f_full=fit.f, detail=detail)
    except Exception:  # noqa: BLE001 - a picture only
        detail = {}
    img = (frame.astype(np.float32) * 0.5).astype(np.uint8)
    paint(img, alpha >= 0.5, (90, 90, 90), 0.7)
    segs = detail.get("segments", np.zeros((0, 4)))
    labs = detail.get("labels", np.full(len(segs), -1))
    cols = {0: GREEN, 1: RED, 2: BLUE, -1: (140, 140, 140)}
    for s, k in sorted(zip(segs, labs), key=lambda q: q[1] >= 0):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), cols[int(k)], t if k >= 0 else max(1, t // 2),
                 cv2.LINE_AA)
    c = fit.c
    for name, p, col in zip(("vertical", "horizontal 1", "horizontal 2"), detail.get("vps", []), (GREEN, RED, BLUE)):
        if p is None:
            continue
        if -0.25 * W < p[0] < 1.25 * W and -0.25 * H < p[1] < 1.25 * H:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), col, cv2.MARKER_TILTED_CROSS, 12 * t, t)
            label(img, f"VP {name}", (p[0] + 8 * t, p[1]), col)
        else:  # far away: an arrow from the centre toward it
            d = np.asarray(p, float) - np.asarray(c, float)
            d /= max(np.linalg.norm(d), 1e-9)
            q = (c[0] + d[0] * 0.3 * min(W, H), c[1] + d[1] * 0.3 * min(W, H))
            cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(q[0]), int(q[1])), col, t, cv2.LINE_AA, tipLength=0.08)
            label(img, f"to VP {name}", (q[0] + 6 * t, q[1]), col)
    cam = fit.cam
    K = np.array([[fit.f, 0.0, c[0]], [0.0, fit.f, c[1]], [0.0, 0.0, 1.0]])
    if cam.roll is not None and cam.pitch is not None:
        r, p = math.radians(cam.roll), math.radians(cam.pitch)
        y_h = c[1] - fit.f * math.tan(p)
        line_through(img, (math.tan(r), 1.0, -(y_h + c[0] * math.tan(r))), CYAN, t)
    R, _ = fit.pose.rt(fit.c)
    line_through(img, np.linalg.inv(K).T @ (R @ np.array([0.0, 0.0, 1.0])), YELLOW, t)  # the car floor's horizon
    n = [int((labs == k).sum()) for k in (0, 1, 2, -1)]
    hfov = math.degrees(2 * math.atan(0.5 * W / fit.f))
    lines = []
    if cam.roll is not None:
        lines.append(f"camera roll {cam.roll:+.2f} deg, pitch {cam.pitch:+.2f} deg (+ = looking down), confidence "
                     f"{cam.conf:.2f} ({'strong: trusted' if cam.strong else 'weak: only a soft hint'}); "
                     f"from {CAM_SOURCE.get(cam.source, cam.source)}")
    else:
        lines.append(f"no camera tilt from lines: {CAM_SOURCE.get(cam.source, cam.source)}")
    lines += [f"lens: focal length {fit.f:.0f} px = {hfov:.0f} deg across, from "
              f"{FOCAL_SOURCE.get(fit.focal_source, fit.focal_source)}",
              f"segments: {n[0]} vertical, {n[1]} + {n[2]} horizontal (two directions), {n[3]} unused; "
              "lines on the car (grey) are ignored",
              ("green = vertical lines (walls, poles, door frames)", GREEN),
              ("red / blue = the two horizontal directions (kerbs, wall bases, LED strips)", RED),
              ("cyan = horizon from the lines (the camera's tilt against gravity)", CYAN),
              ("yellow = horizon of the car's floor from the 3D fit (differs from cyan when the car stands on a slope)",
               YELLOW)]
    return img, lines


def _proxy_on_work(fit, yaw, D, delta, aim_w, roll) -> np.ndarray:
    w = fit.pose.work
    R, t = car_to_camera(yaw, D, delta, aim_w, roll, w.f, w.c)
    return render_proxy(body_points(fit.shape, fit.pose.body) @ R.T + t, w.f, w.c,
                        (w.mask.shape[1], w.mask.shape[0])) > 127


def _compare(photo: np.ndarray, proxy: np.ndarray) -> tuple[np.ndarray, float]:
    img = np.full(photo.shape + (3,), 30, np.uint8)
    img[photo & proxy] = (200, 200, 200)
    img[photo & ~proxy] = RED
    img[proxy & ~photo] = GREEN
    union = int((photo | proxy).sum())
    return img, (int((photo & proxy).sum()) / union if union else 0.0)


def _angle_chart(fit, label_yaw: float | None) -> np.ndarray:
    pose = fit.pose
    Wc, Hc, L, Rm, T, B = 1200, 520, 70, 30, 30, 60
    img = np.full((Hc, Wc, 3), 245, np.uint8)
    sw = pose.sweep
    ious = [i for _, i, _ in sw] + [i for _, i, _ in pose.alternatives] + [pose.iou]
    lo, hi = max(0.0, min(ious) - 0.05) if ious else 0.0, 1.0
    x_of = lambda yaw: L + (yaw % 360) / 360.0 * (Wc - L - Rm)  # noqa: E731
    y_of = lambda v: T + (hi - v) / max(hi - lo, 1e-6) * (Hc - T - B)  # noqa: E731
    for v in np.linspace(lo, hi, 5):
        cv2.line(img, (L, int(y_of(v))), (Wc - Rm, int(y_of(v))), (220, 220, 220), 1)
        cv2.putText(img, f"{v:.2f}", (8, int(y_of(v)) + 4), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    names = {0: "front", 45: "front-right", 90: "right", 135: "back-right", 180: "back", 225: "back-left",
             270: "left", 315: "front-left"}
    for deg, name in names.items():
        x = int(x_of(deg))
        cv2.line(img, (x, T), (x, Hc - B), (225, 225, 225), 1)
        cv2.putText(img, f"{deg}", (x - 10, Hc - B + 18), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, name, (x - 22, Hc - B + 36), FONT, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "yaw (deg) ->", (Wc - 150, Hc - 6), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "outline match (IoU)", (L + 4, T - 10), FONT, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    if label_yaw is not None:
        x = int(x_of(label_yaw))
        for y in range(T, Hc - B, 12):
            cv2.line(img, (x, y), (x, y + 6), ORANGE, 2)
    if sw:
        sc = np.array([s for _, _, s in sw])
        s_lo, s_hi = float(sc.min()), float(sc.max())
        pts_s = [(int(x_of(y)), int(y_of(hi - (s - s_lo) / max(s_hi - s_lo, 1e-9) * (hi - lo)))) for y, _, s in sw]
        pts_i = [(int(x_of(y)), int(y_of(i))) for y, i, _ in sw]
        cv2.polylines(img, [np.array(pts_s, np.int32)], False, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.polylines(img, [np.array(pts_i, np.int32)], False, (200, 90, 0), 2, cv2.LINE_AA)
        for p in pts_i:
            cv2.circle(img, p, 3, (200, 90, 0), -1, cv2.LINE_AA)
    best = int(np.argmin([s for _, _, s in pose.alternatives])) if pose.alternatives else -1
    for k, (yaw, iou_k, _) in enumerate(pose.alternatives):
        p = (int(x_of(yaw)), int(y_of(iou_k)))
        cv2.circle(img, p, 8, PURPLE, 2, cv2.LINE_AA)
        cv2.putText(img, f"{yaw:.0f}", (p[0] + 9, p[1] - 9), FONT, 0.45, PURPLE, 1, cv2.LINE_AA)
    p = (int(x_of(pose.yaw)), int(y_of(pose.iou)))
    cv2.circle(img, p, 11, (0, 170, 0), -1, cv2.LINE_AA)
    cv2.putText(img, f"chosen {pose.yaw:.0f}", (p[0] + 14, p[1] + 20), FONT, 0.55, (0, 140, 0), 2, cv2.LINE_AA)
    return img, best


def fit_pictures(dw: DebugWriter, stem: str, frame: np.ndarray, alpha: np.ndarray, fit, view_label: str | None = None,
                 label_yaw: float | None = None) -> None:
    H, W = alpha.shape
    t = thick(frame.shape)
    pose = fit.pose

    img, lines = _camera_lines(frame, alpha, fit)
    dw.save(stem, "06_camera_lines", img, "6. Camera tilt and lens from the background's straight lines", lines)

    chart, _ = _angle_chart(fit, label_yaw)
    flags = ", ".join(fit.flags) or "none"
    lines = [f"chosen yaw {pose.yaw:.1f} deg (source: {pose.yaw_source}), outline match IoU {pose.iou:.3f}"
             + (f", {pose.margin:.1f} nats better than the car turned around" if pose.margin < 90 else ""),
             f"camera {pose.D:.2f} m from the car's centre, {pose.h:.2f} m above its floor, looking {pose.delta:.1f} deg"
             f" down at it; pitch {pose.pitch:+.1f}, roll {pose.roll:+.1f} deg (relative to the car's floor); "
             f"car body: {pose.body}",
             ("blue = the 3D car's outline match at every 10 deg (camera height fixed, coarse raster)", (200, 90, 0)),
             ("grey = the score with the priors added (view label, background lines, tail lamps), scaled so higher "
              "is better: this is what picks the candidates", (170, 170, 170)),
             ("purple circles = the candidates refined in full (finer raster, all six pose values free)", PURPLE),
             ("green dot = the final angle", GREEN)]
    if label_yaw is not None:
        lines.append((f"orange dashed = the view label in the file name ({view_label}, {label_yaw:.0f} deg)", ORANGE))
    lines.append(f"flags: {flags}")
    dw.save(stem, "07_angle_search", chart, "7. Finding the car's view angle (yaw) and the camera's position", lines)

    over = frame.copy()
    R, tv = pose.rt(fit.c)
    pts = body_points(fit.shape, pose.body)
    prox = render_proxy(pts @ R.T + tv, fit.f, fit.c, (W, H)) > 127
    outline(over, alpha >= 0.5, RED, t)
    outline(over, prox, GREEN, t)
    q = fit.shape.contacts() @ R.T + tv
    for p in q:
        if p[2] > 0.05:
            cv2.circle(over, (int(round(fit.c[0] + fit.f * p[0] / p[2])), int(round(fit.c[1] + fit.f * p[1] / p[2]))),
                       4 * t, YELLOW, t, cv2.LINE_AA)
    cv2.drawMarker(over, (int(round(pose.aim[0])), int(round(pose.aim[1]))), MAGENTA, cv2.MARKER_CROSS, 16 * t, t)
    dw.save(stem, "08_outline_fit", over, "8. The fitted 3D car on the photo",
            [f"IoU {pose.iou:.3f} between the two outlines" + ("  (FIT FAILED: placed with the label's angle)"
                                                             if pose.failed else ""),
             ("red = the car's outline from the cut-out", RED), ("green = the outline of the fitted 3D car", GREEN),
             ("yellow circles = its four tyre contacts", YELLOW),
             ("magenta + = where the centre of its footprint appears", MAGENTA)])

    work = pose.work
    if work is not None:
        photo = work.mask > 127
        main, iou_main = _compare(photo, _proxy_on_work(fit, pose.yaw, pose.D, pose.delta, work.from_frame(pose.aim),
                                                        pose.roll))
        label(main, f"final: yaw {pose.yaw:.0f}, IoU {iou_main:.3f}", (10, 30), WHITE, 0.8)
        tiles = []
        best = int(np.argmin([s for _, _, s in pose.alternatives])) if pose.alternatives else -1
        for k, (cand, (yaw, iou_k, score)) in enumerate(zip(pose.candidates, pose.alternatives)):
            tile, _ = _compare(photo, _proxy_on_work(fit, cand.yaw, cand.D, cand.delta, cand.aim, cand.roll))
            tile = cv2.resize(tile, (max(1, tile.shape[1] * 300 // max(tile.shape[:2])),
                                     max(1, tile.shape[0] * 300 // max(tile.shape[:2]))), interpolation=cv2.INTER_AREA)
            label(tile, f"yaw {yaw:.0f}  IoU {iou_k:.3f}", (6, 20), WHITE, 0.5)
            label(tile, f"score {score:.2f}" + ("  best" if k == best else ""), (6, 40),
                  (0, 230, 0) if k == best else WHITE, 0.5)
            tiles.append(tile)
        pic = main if not tiles else np.vstack([
            cv2.copyMakeBorder(main, 0, 8, 0, max(0, grid(tiles, 4).shape[1] - main.shape[1]), cv2.BORDER_CONSTANT,
                               value=(24, 24, 24)),
            cv2.copyMakeBorder(grid(tiles, 4), 0, 0, 0, max(0, main.shape[1] - grid(tiles, 4).shape[1]),
                               cv2.BORDER_CONSTANT, value=(24, 24, 24))])
        dw.save(stem, "09_outline_compare", pic, "9. Outline from the cut-out vs outline of the 3D car",
                [("light grey = both agree", (200, 200, 200)), ("red = only in the cut-out (the 3D car is missing it)", RED),
                 ("green = only in the 3D car (the cut-out is missing it)", GREEN),
                 "Top: the final pose. Below: every candidate angle refined in full; the lowest score wins "
                 "(outline match plus the view-label, line and tail-lamp priors).",
                 "Residual blobs: " + (", ".join(f"{b['kind']} {100 * b['share']:.0f}%" for b in pose.blobs) or "none")])

    _level_picture(dw, stem, frame, alpha, fit, t)
    _tyre_picture(dw, stem, fit, t)


def _level_picture(dw: DebugWriter, stem: str, frame: np.ndarray, alpha: np.ndarray, fit, t: int) -> None:
    solid = alpha >= 0.5
    x1, y1, x2, y2 = box_of(solid, 0.06)
    before = frame[y1:y2, x1:x2].copy()
    outline(before, solid[y1:y2, x1:x2], RED, t)
    line_b = thirds_line(solid[y1:y2, x1:x2])
    draw_bottom_line(before, line_b, t)
    car = fit.car
    after = on_checker(car.pixels, car.alpha)
    sa = car.alpha >= 0.5
    outline(after, sa, GREEN, t)
    line_a = thirds_line(sa)
    draw_bottom_line(after, line_a, t)
    k = before.shape[0] / max(1, after.shape[0])
    after = cv2.resize(after, (max(1, round(after.shape[1] * k)), before.shape[0]), interpolation=cv2.INTER_AREA)
    cam_roll = "" if fit.cam.roll is None else f"; the background lines' roll was {fit.cam.roll:+.2f} deg"
    if fit.theta == 0.0:
        what = ("No rotation: the car's pixels are untouched (" +
                ("level angle below 0.5 deg" if fit.cam.strong else "the background lines were too weak to trust")
                + ")")
    else:
        what = f"The car's pixels were rotated once by {fit.theta:+.2f} deg about the photo centre"
    lines = [what,
             f"full level angle {fit.theta_full:+.2f} deg, including the car's own roll on the road {fit.car_roll:+.2f} "
             f"deg{cam_roll}" + ("; reduced so a slanted cut edge loses at most 2% of the car" if "partial-level" in
                                 fit.flags else ""),
             (f"yellow = the car's bottom line (lowest point of the left and right thirds): "
              f"{'n/a' if line_b is None else f'{line_b[2]:+.2f}'} deg before, "
              f"{'n/a' if line_a is None else f'{line_a[2]:+.2f}'} deg after", YELLOW),
             ("white dashes = level, for comparison (a 3/4 view's bottom line is not level even on flat ground)", WHITE),
             ("left: the photo (car outline red); right: the levelled car (outline green)", GREY)]
    dw.save(stem, "10_level", side_by_side(before, after), "10. Car alignment: levelling the photo", lines)


def _tyre_picture(dw: DebugWriter, stem: str, fit, t: int) -> None:
    car = fit.car
    h, w = car.alpha.shape
    pad = int(0.12 * max(h, w)) + 10
    canvas = checker(h + 2 * pad, w + 2 * pad)
    canvas[pad:pad + h, pad:pad + w] = on_checker(car.pixels, car.alpha)
    ox, oy = car.origin[0] - pad, car.origin[1] - pad
    K = np.array([[fit.f, 0.0, fit.c[0]], [0.0, fit.f, fit.c[1]], [0.0, 0.0, 1.0]])
    ul, vl, zl = project_rc(fit.shape.contacts(), fit.R_lock, fit.C_lock, K)
    for i, tb in enumerate(fit.bottoms):
        col = STATUS_COLOUR.get(tb.status, (150, 150, 150))
        c = (int(round(tb.u_pred - ox)), int(round(tb.v_pred - oy)))
        cv2.circle(canvas, c, max(4, int(round(tb.r_px)) or 6), col, t, cv2.LINE_AA)
        label(canvas, f"{tb.name} {tb.status}", (c[0] + 6, c[1] - 6 - max(4, int(tb.r_px))), col)
        if tb.status == "found":
            cv2.drawMarker(canvas, (int(round(tb.u - ox)), int(round(tb.v - oy))), MAGENTA, cv2.MARKER_TILTED_CROSS,
                           10 * t, t)
        if tb.status == "found" and np.isfinite(ul[i]) and zl[i] > 0:
            cv2.circle(canvas, (int(round(ul[i] - ox)), int(round(vl[i] - oy))), 2 * t + 1, CYAN, -1, cv2.LINE_AA)
    lock = fit.lock
    res_b = ", ".join(f"{r:+.1f}" for r in lock.get("res_before", [])) or "-"
    res_a = ", ".join(f"{r:+.1f}" for r in lock.get("res_after", [])) or "-"
    found = sum(b.status == "found" for b in fit.bottoms)
    dw.save(stem, "11_tyres", canvas, "11. Tyre bottoms and the contact lock (on the levelled car)",
            [f"{found} tyre bottom(s) found; contact lock: {lock.get('mode', 'none')} "
             f"(camera height {100 * lock.get('dh', 0.0):+.1f}%, pitch {lock.get('dp', 0.0):+.2f} deg"
             + (", at its limit" if lock.get("at_bound") else "") + ")",
             f"predicted contact minus found bottom, px: before the lock {res_b}; after {res_a}",
             ("circles = where the 3D car predicts each tyre's contact (green found, orange hidden by the body, "
              "red at a cut edge, magenta implausible, grey not visible / far side)", GREEN),
             ("magenta x = the tyre bottom found on the car's outline", MAGENTA),
             ("cyan dot = the contact after the lock (the floor is brought to the tyres; the car's pixels never move)",
              CYAN)])


# ---------------------------------------------------------------- steps 12-16: per background
def room_pictures(dw: DebugWriter, stem: str, bg: str, placed) -> None:
    st = placed.steps
    m = placed.metrics
    if "room" in st:
        dw.save(stem, "12_room", st["room"], f"12. {bg}: its 3D room re-rendered from the photo's camera",
                [f"output {m.get('out_w')} x {m.get('out_h')} px (scale {m.get('out_scale')}); "
                 f"{m.get('floor_texels')} background texels per output pixel on the floor at the car",
                 ("magenta = the room's edges (floor outline, wall corners, wall tops)", MAGENTA),
                 ("red + = the stage point where the car's footprint centre stands", RED)], bg=bg)
    if "surfaces" in st:
        dw.save(stem, "13_room_surfaces", st["surfaces"], f"13. {bg}: what each pixel of the room shows",
                [("green = floor", (60, 180, 60)), ("blue = walls", (200, 140, 40)), ("purple = ceiling", (170, 60, 170)),
                 (f"red hatching = never visible in the background picture, filled in ({m.get('invented_pct')}% of "
                  f"the picture, {m.get('invented_near_pct')}% near the car)", RED)], bg=bg)
    if "shadows" in st:
        dw.save(stem, "14_shadows", st["shadows"], f"14. {bg}: shadows and reflection drawn on the floor",
                ["grey = unchanged; darker = shadow (contact, tyre patches, soft skirt); orange = floor reflection "
                 "(change exaggerated 3x)", ("green = where the car will be pasted", GREEN),
                 f"contact residual {m.get('contact_residual_px')} px; reflection scale "
                 f"{m.get('reflection_scale') or 'none'}"], bg=bg)
    if "background_ready" in st:
        dw.save(stem, "15_background_ready", st["background_ready"],
                f"15. {bg}: the background just before the car is pasted", [], bg=bg)
    if placed.debug is not None:
        dw.save(stem, "16_placed", placed.debug, f"16. {bg}: the result, with its placement marks",
                [("orange = the car's footprint on the floor", (0, 200, 255)), ("red + = the stage point", RED),
                 ("cyan line = the room's horizon", CYAN),
                 ("circles = tyre contacts (green found, grey hidden); magenta x = found tyre bottoms", GREEN),
                 ("blue tint = areas invented by the room fill", BLUE),
                 "flags: " + (", ".join(placed.flags) or "none")], bg=bg)


def flat_picture(dw: DebugWriter, stem: str, bg: str, overlay: np.ndarray, plan) -> None:
    dw.save(stem, "12_flat_placement", overlay, f"12. {bg}: flat placement (this background has no 3D room)",
            [f"scale {plan.scale:.3f}, {plan.px_per_m:.0f} px per metre at the stage",
             "stage, footprint and tyre contacts as drawn by the scene code",
             "flags: " + (", ".join(plan.flags) or "none")], bg=bg)


__all__ = ["DebugWriter", "cutout_pictures", "fit_pictures", "flat_picture", "glass_pictures",
           "hosted_camera_picture", "room_pictures"]
