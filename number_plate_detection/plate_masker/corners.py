"""Find a detected plate's 4 corners, so masks follow the plate's own outline.

The detector only gives a straight rectangle. On angled views the plate is a tilted 4-sided
shape, so the rectangle is 2-3x bigger than the plate. This refines it with plain OpenCV
inside the detector's box (about 1-3 ms per plate, no extra model):

1. Threshold the area around the box (Otsu) and take the bright blob under the box centre:
   the plate's white face.
2. Simplify its outline to 4 corners (top-left, top-right, bottom-right, bottom-left).
3. Grow each edge out to the plate's outer trim. Outside the white face, plates here have a
   thin dark border line and then a light rim; each edge is measured on its own.
4. Sanity-check the shape. If anything looks wrong the plate keeps its rectangle.

Coloured plates (dark background, light letters) are found by their background colour instead
of brightness. Last resort, when the plate's own edge can't be traced at all (a dark plate on a
dark car, or a plate so small the background round its letters is a pixel or two): the outline
is worked out from the row of characters (fit "letters").
"""

from __future__ import annotations

import dataclasses
import statistics

import cv2
import numpy as np

from .detector import Plate

# Margin around the detector box used for thresholding (share of box size).
THRESH_MARGIN = 0.15
# Outer trim, as a share of plate height: typical value (used when an edge can't be measured)
# and the most we ever add. Measured on our frames: 0.13-0.15 on most edges, 0.09-0.23 overall.
TRIM_TYPICAL = 0.14
TRIM_MAX = 0.22
# Most a plate's top/bottom edges may lean from horizontal (and its sides from vertical), in
# degrees. Our most tilted real plate leans ~25 deg; a sideways shape is a false hit.
MAX_TILT = 40.0
# Share of one side's pixels that must sit in its biggest connected piece for it to count as the
# plate's background (see background_side).
BACKGROUND_SHARE = 0.6
# A plate's background must lie within the detector's box, give or take this share of the box's
# size. A blob reaching further has merged with something round the plate (e.g. a white bumper
# through a faint border line). Real plates' outer edges reach at most 2 % / 6 % past the box.
FACE_BOX_MARGIN = 0.10
# Coloured plates: the most the border line (character colour) and the rim (background colour)
# outside the found background may add, as a share of plate height; and the rim width assumed
# when its end can't be seen (a black plate in a black holder).
LINE_MAX, RIM_MAX, RIM_TYPICAL = 0.12, 0.12, 0.06
# Outline from the characters: margin above and below the row of characters, as a share of the
# character height (measured: 0.27-0.4), and at each end, as a share of the row's length (about
# 3-5 % on plates whose letters fill the plate; a share of the row, not of the letter height,
# because an angled view squeezes the row but not the letters' height). And how far past the
# detector's box the result may reach left to right (the box is tight that way).
LETTER_MARGIN_V, LETTER_MARGIN_H = 0.35, 0.045
LETTER_BOX_MARGIN_X = 0.05


def order_corners(points) -> np.ndarray:
    """4 points -> float32 array ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # clockwise on screen
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def quad_size(quad: np.ndarray) -> tuple[float, float]:
    """(width, height) of a TL, TR, BR, BL quad: means of opposite edges."""
    q = np.asarray(quad, np.float64)
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(width), float(height)


def _outward_normals(q: np.ndarray) -> list[np.ndarray]:
    """Unit normal of each edge (0 top, 1 right, 2 bottom, 3 left), pointing away from the centre."""
    c = q.mean(0)
    normals = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([t[1], -t[0]])
        normals.append(n if np.dot((a + b) / 2 - c, n) >= 0 else -n)
    return normals


def grow_quad(quad: np.ndarray, offsets) -> np.ndarray:
    """Move each edge (top, right, bottom, left) outward by its offset in pixels; re-intersect corners."""
    q = np.asarray(quad, np.float64)
    normals = _outward_normals(q)
    lines = [(q[i] + normals[i] * offsets[i], q[(i + 1) % 4] - q[i]) for i in range(4)]
    out = []
    for j in range(4):  # corner j joins edge j-1 (ending here) and edge j (starting here)
        (p1, d1), (p2, d2) = lines[j - 1], lines[j]
        a = np.array([d1, -d2]).T
        if abs(np.linalg.det(a)) < 1e-9:
            out.append(q[j] + normals[j - 1] * offsets[j - 1] + normals[j] * offsets[j])
        else:
            s = np.linalg.solve(a, p2 - p1)[0]
            out.append(p1 + d1 * s)
    return np.float32(out)


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest-pixel values at (x, y) points; NaN outside the image."""
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < gray.shape[1]) & (ys >= 0) & (ys < gray.shape[0])
    if ok.all():
        return gray[ys, xs].astype(np.float64)
    vals = np.full(len(pts), np.nan)
    vals[ok] = gray[ys[ok], xs[ok]]
    return vals


def _edge_trim(gray: np.ndarray, a: np.ndarray, b: np.ndarray, n: np.ndarray, h: float) -> float | None:
    """How far (px) the plate's outer trim lies outside the white-face edge a-b. None if unmeasurable."""
    n_in, n_out = max(2, int(0.05 * h)), max(6, int(0.35 * h))
    steps = np.arange(-n_in, n_out)  # negative = inside the white face
    starts = a + np.outer(np.linspace(0.2, 0.8, 7), b - a)  # 7 profiles spread along the edge
    pts = starts[:, None, :] + steps[None, :, None] * n[None, None, :]
    profiles = _sample(gray, pts.reshape(-1, 2)).reshape(len(starts), -1)
    window = max(3, int(0.12 * h))
    offsets = []
    for row in profiles:  # plain lists from here: much faster than numpy on ~40 values
        if np.isnan(row).any():
            continue
        vals = row.tolist()
        inside = sorted(vals[:n_in])
        white = inside[len(inside) // 2]
        if white < 100:
            continue
        out = vals[n_in:]  # out[d] = brightness d px outside the edge
        t = next((i for i, v in enumerate(out) if v < 0.6 * white), None)  # where the white face ends
        if t is None or t > 0.15 * h:
            continue
        seg = out[t:t + window]
        j = t + seg.index(min(seg))  # darkest point: the border line (or the background)
        seg = out[j:j + window]
        k = j + seg.index(max(seg))  # brightest point after it: the light rim, if any
        if out[k] - out[j] >= 40 and out[k] >= 0.5 * white:
            mid = (out[k] + out[j]) / 2
            e = k
            while e + 1 < len(out) and out[e + 1] > mid and e - k < 0.1 * h:
                e += 1
            offsets.append(e + 1)  # just past the rim
        else:
            offsets.append(t)  # no rim: the plate ends where the white face ends
    return float(statistics.median(offsets)) if offsets else None


def _inside_box(q: np.ndarray, plate: Plate, margin: float) -> bool:
    """All corners within the detector's box, grown by `margin` x its width/height on every side."""
    mx, my = margin * plate.width, margin * plate.height
    return bool(
        (q[:, 0] >= plate.x1 - mx).all() and (q[:, 0] <= plate.x2 + mx).all()
        and (q[:, 1] >= plate.y1 - my).all() and (q[:, 1] <= plate.y2 + my).all()
    )


def _colour_edge_trim(lab: np.ndarray, a: np.ndarray, b: np.ndarray, n: np.ndarray, h: float,
                      face_lab: np.ndarray, ink_lab: np.ndarray, from_ring: bool = False) -> float | None:
    """Coloured plates: how far (px) the plate goes on outside the found edge a-b.

    Looks at how different each pixel is from the background colour `face_lab`, going outward.
    Seen from the background, a plate ends in one of two ways:
      background | different (car) .................... the edge is the plate's edge (0)
      background | different (border line) | background again (rim) | different (car)
                                                       ... the plate ends after the rim
    from_ring: the edge is the outside of a light border line round a dark background (found by
    the line's colour), so only the rim is left: background colour | different (car).
    A thin, blurred border line never matches the character colour exactly, but it still shows
    up as a "different" bump, which is all this needs. A rim whose end can't be seen (a black
    plate in a black holder) is taken as RIM_TYPICAL x plate height.
    """
    weight = np.float32([0.5, 1.0, 1.0])
    limit = max(10.0, 0.5 * float(np.sqrt((((ink_lab - face_lab) * weight) ** 2).sum())))
    steps = np.arange(0, max(6, int(0.3 * h)))
    starts = a + np.outer(np.linspace(0.2, 0.8, 7), b - a)
    pts = (starts[:, None, :] + steps[None, :, None] * n[None, None, :]).reshape(-1, 2)
    xs, ys = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < lab.shape[1]) & (ys >= 0) & (ys < lab.shape[0])
    vals = np.full((len(pts), 3), np.nan, np.float32)
    vals[ok] = lab[ys[ok], xs[ok]]
    diff = np.sqrt((((vals - face_lab) * weight) ** 2).sum(axis=1)).reshape(len(starts), -1)
    line_max, rim_max, start_max = LINE_MAX * h, RIM_MAX * h, 0.05 * h
    offsets = []
    for row in diff:
        if np.isnan(row).any():
            continue
        near = (row < limit).tolist()  # True = looks like the background colour
        end, d = len(near), 0
        if from_ring:
            while d < end and not near[d] and d < 2:  # the line's own blurred edge pixels
                d += 1
        else:
            while d < end and near[d] and d < start_max:  # the found edge may sit a pixel or two inside
                d += 1
            line_start = d
            while d < end and not near[d]:
                d += 1
            if d - line_start > line_max or d >= end:  # a long "different" run: the car, not a line
                offsets.append(line_start)
                continue
        rim_start = d
        while d < end and near[d] and d - rim_start < rim_max:
            d += 1
        if d - rim_start >= rim_max:  # no visible end to the rim: assume a typical one
            d = rim_start + int(round(RIM_TYPICAL * h))
        offsets.append(d)  # just past the rim
    return float(statistics.median(offsets)) if offsets else None


def _plausible(q: np.ndarray, plate: Plate, max_area: float) -> bool:
    edges = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    if min(edges) < 0.15 * min(plate.width, plate.height):
        return False
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return False
    for i in range(4):  # edges 0, 2 = top, bottom (near horizontal); 1, 3 = sides (near vertical)
        dx, dy = q[(i + 1) % 4] - q[i]
        angle = abs(np.degrees(np.arctan2(dy, dx)))  # 0..180
        lean = min(angle, 180 - angle) if i % 2 == 0 else abs(90 - angle)
        if lean > MAX_TILT:
            return False
    width, height = quad_size(q)
    if not 1.0 <= width / height <= 8.0:
        return False
    area = cv2.contourArea(q.astype(np.float32)) / (plate.width * plate.height)
    if not 0.2 <= area <= max_area:
        return False
    span_x = (q[:, 0].max() - q[:, 0].min()) / plate.width
    span_y = (q[:, 1].max() - q[:, 1].min()) / plate.height
    return span_x >= 0.6 and span_y >= 0.6


def _mask_quad(mask: np.ndarray, opening: int, centre, box_area: float) -> np.ndarray | None:
    """Corners of the blob in `mask` under `centre` that is plate-sized, in the mask's pixels."""
    if opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fits = [
        c for c in contours
        if 0.3 * box_area <= cv2.contourArea(c) <= 1.25 * box_area and cv2.pointPolygonTest(c, centre, False) >= 0
    ]
    if not fits:
        return None
    hull = cv2.convexHull(max(fits, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def _face_quad(gray: np.ndarray, blur: int, opening: int, centre, box_area: float) -> np.ndarray | None:
    """Corners of the bright blob under `centre` (a light plate's face), in `gray`'s pixels."""
    g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur else gray
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _mask_quad(mask, opening, centre, box_area)


def background_side(gray: np.ndarray, face: np.ndarray) -> str | None:
    """'light' or 'dark': which side of the plate inside `face` is its background, or None if unclear.

    Splits the pixels into light and dark (Otsu). The background is one connected area wrapped
    round everything; the characters are separate islands. Measured: the background's biggest
    connected piece holds 0.91-0.98 of its pixels, the characters' only 0.15-0.18.
    """
    q = grow_quad(face, [-2] * 4)
    x0, y0 = np.maximum(np.floor(q.min(0)).astype(int), 0)
    x1, y1 = np.ceil(q.max(0)).astype(int) + 1
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    inside = np.zeros(roi.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.round(q - np.float32([x0, y0])).astype(np.int32), 1)
    vals = roi[inside > 0]
    if vals.size < 50:
        return None
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shares = {}
    for side, part in (("light", roi > thr), ("dark", roi <= thr)):
        _, _, stats, _ = cv2.connectedComponentsWithStats((part & (inside > 0)).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        shares[side] = areas.max() / areas.sum() if len(areas) else 0.0
    side = max(shares, key=shares.get)
    return side if shares[side] >= BACKGROUND_SHARE and min(shares.values()) < BACKGROUND_SHARE else None


def _colour_mask(lab: np.ndarray, face_lab: np.ndarray, ink_lab: np.ndarray) -> np.ndarray:
    """255 where the (blurred, float32 Lab) crop is close to the background colour `face_lab`."""
    weight = np.float32([0.5, 1.0, 1.0])  # brightness counts half: shade changes it, colour stays
    dist2 = (((lab - face_lab) * weight) ** 2).sum(axis=2)
    limit = max(10.0, 0.5 * float(np.sqrt((((ink_lab - face_lab) * weight) ** 2).sum())))
    return (dist2 < limit * limit).astype(np.uint8) * 255


def _two_colours(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two main colours (Lab), darker first: for a plate, its background and its characters."""
    pix = bgr.reshape(-1, 3)
    if len(pix) < 20:
        return None
    lab = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    data = lab[np.linspace(0, len(lab) - 1, min(len(lab), 3000)).astype(int)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dark, light = sorted(centres, key=lambda c: c[0])
    return dark, light


def _character_row(mask: np.ndarray, plate: Plate, x0: int, y0: int):
    """The plate's row of characters in `mask` (a crop starting at x0, y0): (score, [(xs, ys)]) or None.

    Characters are separate shapes of about the same height whose centres lie on one line, at
    most MAX_TILT from horizontal. The biggest such group (at least 3) wins.
    """
    w, h = plate.width, plate.height
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cands = []
    for i in range(1, n):
        cx, cy = centres[i][0] + x0, centres[i][1] + y0
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if not (plate.x1 <= cx <= plate.x2 and plate.y1 <= cy <= plate.y2):
            continue
        if bh < max(4, 0.2 * h) or bh > 0.95 * h or bw > 0.4 * w or area < 8:
            continue
        if not 0.12 <= area / (bw * bh) <= 0.9:
            continue
        cands.append((i, cx, cy, float(bh)))
    if len(cands) < 3:
        return None
    best = None
    max_slope = np.tan(np.radians(MAX_TILT))
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            _, ax, ay, ah = cands[a]
            _, bx, by, bh = cands[b]
            if max(ah, bh) > 1.6 * min(ah, bh) or abs(bx - ax) < 1 or abs(by - ay) > max_slope * abs(bx - ax):
                continue
            d = float(np.hypot(bx - ax, by - ay))
            ux, uy = (bx - ax) / d, (by - ay) / d
            href = (ah + bh) / 2
            members = [c for c in cands
                       if abs((c[2] - ay) * ux - (c[1] - ax) * uy) <= 0.3 * href and 0.6 * href <= c[3] <= 1.6 * href]
            if len(members) < 3:
                continue
            us = [(c[1] - ax) * ux + (c[2] - ay) * uy for c in members]
            score = (len(members), max(us) - min(us))
            if best is None or score > best[0]:
                best = (score, members, (ux, uy))
    if best is None:
        return None
    _, members, (ux, uy) = best
    # Characters carry similar amounts of ink (even a "1" has ~80 % of a typical letter's); a
    # thin sliver with far less is the plate's edge or a highlight, not a character.
    ink = np.float32([stats[c[0], cv2.CC_STAT_AREA] for c in members])
    members = [c for c, n in zip(members, ink) if n >= 0.3 * float(np.median(ink))]
    if len(members) < 3:
        return None
    us = [c[1] * ux + c[2] * uy for c in members]
    span = max(us) - min(us)
    if span < 0.35 * w:  # the characters must run across most of the plate
        return None
    pixels = []
    for c in members:
        ys, xs = np.nonzero(labels == c[0])
        pixels.append((xs.astype(np.float32) + x0, ys.astype(np.float32) + y0))
    return (len(members), span), pixels


def _letters_down(pixels, t: np.ndarray) -> np.ndarray:
    """Unit vector pointing down the characters (the plate's sides), from their long axes.

    Uses characters clearly taller than wide (long axis >= 1.3x short axis); falls back to square
    to the text direction `t`. Kept within 35 degrees of square to `t`.
    """
    square = np.float32([-t[1], t[0]])  # perpendicular to t, pointing down the image
    axes = []
    for xs, ys in pixels:
        if len(xs) < 6:
            continue
        cov = np.cov(np.vstack([xs, ys]))
        vals, vecs = np.linalg.eigh(cov)
        if vals[0] <= 0 or vals[1] < 1.69 * vals[0]:  # sqrt ratio < 1.3: no clear long axis
            continue
        axis = vecs[:, 1]
        axis = axis if np.dot(axis, square) >= 0 else -axis
        if np.dot(axis, square) < np.cos(np.radians(35)):  # e.g. letters merged side by side: long axis sideways
            continue
        axes.append(axis)
    if len(axes) < 2:
        return square
    down = np.mean(axes, axis=0)
    down /= np.linalg.norm(down)
    if np.degrees(np.arccos(np.clip(np.dot(down, square), -1, 1))) > 35:
        return square
    return down.astype(np.float32)


def _letters_quad(image_bgr: np.ndarray, plate: Plate) -> np.ndarray | None:
    """The plate's outline worked out from its row of characters (image pixels), or None.

    Used when the plate's own edge can't be traced. Finds the characters (light or dark), fits
    lines along their tops and bottoms (the plate's tilt and perspective), takes the plate's
    sides along the letters' upright strokes, and adds margins (LETTER_MARGIN_V x the letter
    height above and below, LETTER_MARGIN_H x the row's length at each end), kept within the
    detector's box left to right.
    """
    w, h = plate.width, plate.height
    if w < 16 or h < 8:
        return None
    img_h, img_w = image_bgr.shape[:2]
    x0, y0 = max(0, plate.x1 - int(0.08 * w)), max(0, plate.y1 - int(0.08 * h))
    x1, y1 = min(img_w, plate.x2 + int(0.08 * w)), min(img_h, plate.y2 + int(0.08 * h))
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    inside = gray[plate.y1 - y0:plate.y2 - y0, plate.x1 - x0:plate.x2 - x0]
    thr, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = None
    for part in (gray > thr, gray <= thr):  # light characters, then dark characters
        found = _character_row(part.astype(np.uint8), plate, x0, y0)
        if found is not None and (best is None or found[0] > best[0]):
            best = found
    if best is None:
        return None
    pixels = best[1]

    # Text direction t from the characters' centres.
    centres = np.float32([[xs.mean(), ys.mean()] for xs, ys in pixels])
    vx, vy, _, _ = cv2.fitLine(centres.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    if vx < 0:
        vx, vy = -vx, -vy
    t = np.float32([vx, vy])
    # "Down" direction d of the plate, from the characters' long axes (plate fonts are tall and
    # narrow). Seen at an angle, a plate's sides stay near upright in the photo instead of being
    # square to its slanted row of letters, so d is not simply perpendicular to t.
    d = _letters_down(pixels, t)
    origin = centres.mean(0)
    to_ab = np.linalg.inv(np.column_stack([t, d]))  # image offset -> (a along t, b along d)
    tops, bots, mids, los, his = [], [], [], [], []
    for xs, ys in pixels:
        a, b = to_ab @ np.vstack([xs - origin[0], ys - origin[1]])
        mids.append(float(np.median(a)))
        tops.append(float(np.percentile(b, 2)))
        bots.append(float(np.percentile(b, 98)))
        los.append(float(np.percentile(a, 1)))
        his.append(float(np.percentile(a, 99)))
    top = np.polyfit(mids, tops, 1)  # lines along the characters' tops and bottoms: their
    bot = np.polyfit(mids, bots, 1)  # different slopes carry the plate's perspective
    a_left, a_right = min(los), max(his)
    c_left = np.polyval(bot, a_left) - np.polyval(top, a_left)
    c_right = np.polyval(bot, a_right) - np.polyval(top, a_right)
    if min(c_left, c_right) < 3:
        return None
    # Each end's margin follows the letter height there: the end nearer the camera looks bigger.
    c_mean = (c_left + c_right) / 2
    m_left = LETTER_MARGIN_H * (a_right - a_left) * c_left / c_mean
    m_right = LETTER_MARGIN_H * (a_right - a_left) * c_right / c_mean
    al, ar = a_left - m_left, a_right + m_right
    ab = [(al, np.polyval(top, al) - LETTER_MARGIN_V * c_left), (ar, np.polyval(top, ar) - LETTER_MARGIN_V * c_right),
          (ar, np.polyval(bot, ar) + LETTER_MARGIN_V * c_right), (al, np.polyval(bot, al) + LETTER_MARGIN_V * c_left)]
    quad = np.float32([origin + a * t + b * d for a, b in ab])
    vx, vy = float(t[0]), float(t[1])

    # The detector's box is tight left to right: pull an end back inside it, but never into the text.
    lo_x, hi_x = plate.x1 - LETTER_BOX_MARGIN_X * w, plate.x2 + LETTER_BOX_MARGIN_X * w
    along = np.float32([vx, vy])
    over_left = lo_x - min(quad[0, 0], quad[3, 0])
    if over_left > 0 and vx > 0.3:
        shift = min(over_left / vx, m_left)
        quad[0] += shift * along
        quad[3] += shift * along
    over_right = max(quad[1, 0], quad[2, 0]) - hi_x
    if over_right > 0 and vx > 0.3:
        shift = min(over_right / vx, m_right)
        quad[1] -= shift * along
        quad[2] -= shift * along
    if not _plausible(quad, plate, max_area=1.6) or not _inside_box(quad, plate, 0.25):
        return None
    return quad


def find_plate_corners(image_bgr: np.ndarray, plate: Plate) -> np.ndarray | None:
    """The plate's outer corners (TL, TR, BR, BL) in image pixels, or None if they can't be found."""
    found = find_plate_outline(image_bgr, plate)
    return None if found is None else found[0]


def find_plate_outline(image_bgr: np.ndarray, plate: Plate) -> tuple[np.ndarray, str] | None:
    """(corners, how) or None. how: "corners" (the plate's own edge) or "letters" (from its characters)."""
    quad = _edge_outline(image_bgr, plate)
    if quad is not None:
        return quad, "corners"
    quad = _letters_quad(image_bgr, plate)
    return None if quad is None else (quad, "letters")


def _edge_outline(image_bgr: np.ndarray, plate: Plate) -> np.ndarray | None:
    """The plate's outer corners traced from its own edge (background, border line, rim), or None."""
    w, h = plate.width, plate.height
    if w < 12 or h < 6:
        return None
    img_h, img_w = image_bgr.shape[:2]
    # Big crop for measuring the trim, small crop (inside it) for thresholding.
    m = int(0.5 * max(w, h))
    bx1, by1 = max(0, plate.x1 - m), max(0, plate.y1 - m)
    bx2, by2 = min(img_w, plate.x2 + m), min(img_h, plate.y2 + m)
    gray = cv2.cvtColor(image_bgr[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
    sx1, sy1 = max(0, int(plate.x1 - THRESH_MARGIN * w) - bx1), max(0, int(plate.y1 - THRESH_MARGIN * h) - by1)
    sx2 = min(gray.shape[1], int(plate.x2 + THRESH_MARGIN * w) - bx1)
    sy2 = min(gray.shape[0], int(plate.y2 + THRESH_MARGIN * h) - by1)
    centre = ((plate.x1 + plate.x2) / 2 - bx1 - sx1, (plate.y1 + plate.y2) / 2 - by1 - sy1)
    offset = np.float32([bx1, by1])
    # (blur, opening) passes, first plausible shape wins. Opening cuts thin bridges to a white
    # bumper, but on small plates it also erases the 1-2 px white margin round the characters.
    k = max(3, int(min(w, h) * 0.04)) | 1
    passes = [(5, k), (5, 2 * k + 1), (3, 0)] if h >= 50 else [(3, 0), (3, 3), (5, 0)]

    def first_plausible(find, side):
        for blur, opening in passes:
            quad = find(blur, opening)
            if quad is None:
                continue
            quad = quad + np.float32([sx1, sy1])  # in big-crop pixels
            if (_plausible(quad + offset, plate, max_area=1.1) and _inside_box(quad + offset, plate, FACE_BOX_MARGIN)
                    and background_side(gray, quad) == side):
                return quad
        return None

    # 1. Light background, dark characters (white, yellow, silver plates): the bright blob.
    face = first_plausible(lambda blur, opening: _face_quad(gray[sy1:sy2, sx1:sx2], blur, opening, centre, w * h),
                           "light")
    # 2. Dark or coloured background, light characters (blue, green, red, black plates): the blob
    #    close to the background colour. Never fall back to the bright blob here: on these plates
    #    it is the lettering or a light border line, which gives a crooked outline.
    plate_colours = None  # (background, characters) in Lab, when found by colour
    from_ring = False  # found by its light border line rather than its background
    if face is None:
        bgr = image_bgr[by1:by2, bx1:bx2]
        cy1, cy2 = int(plate.y1 + 0.25 * h) - by1, int(plate.y2 - 0.25 * h) - by1
        cx1, cx2 = int(plate.x1 + 0.15 * w) - bx1, int(plate.x2 - 0.15 * w) - bx1
        colours = _two_colours(bgr[max(0, cy1):cy2, max(0, cx1):cx2])
        if colours is not None:
            darker, lighter = colours
            small = bgr[sy1:sy2, sx1:sx2]
            labs: dict[int, np.ndarray] = {}  # blurred Lab crop, made once per blur size
            masks: dict[tuple[int, str], np.ndarray] = {}  # colour mask, once per blur and colour
            pair = {"darker": (darker, lighter), "lighter": (lighter, darker)}

            def colour_mask(blur, which):
                if blur not in labs:
                    labs[blur] = cv2.cvtColor(cv2.GaussianBlur(small, (blur, blur), 0) if blur else small,
                                              cv2.COLOR_BGR2LAB).astype(np.float32)
                if (blur, which) not in masks:
                    masks[(blur, which)] = _colour_mask(labs[blur], *pair[which])
                return masks[(blur, which)]

            # (blob colour, background side it must give, is the blob the background?)
            # 1. The darker colour is the background: white letters on blue/green/red/black.
            # 2. The lighter colour is the background, yet darker than the car round it
            #    (green with black letters), which the bright-blob pass misses.
            # 3. The lighter colour is a border line round a dark background whose own edge is lost
            #    in a dark holder (a black plate): the line's outline is the plate, less the rim.
            for which, side, blob_is_background in (("darker", "dark", True), ("lighter", "light", True),
                                                    ("lighter", "dark", False)):
                face = first_plausible(
                    lambda blur, opening: _mask_quad(colour_mask(blur, which), opening, centre, w * h), side)
                if face is not None:
                    blob_lab, other_lab = pair[which]
                    plate_colours = (blob_lab, other_lab) if blob_is_background else (other_lab, blob_lab)
                    from_ring = not blob_is_background
                    break
    if face is None:
        return None

    _, face_h = quad_size(face)
    normals = _outward_normals(face.astype(np.float64))
    trims = []
    if plate_colours is None:
        # Light plates: white face -> dark border line -> light rim, measured along each edge.
        for i in range(4):
            t = _edge_trim(gray, face[i], face[(i + 1) % 4], normals[i], face_h)
            trims.append(float(np.clip(TRIM_TYPICAL * face_h if t is None else t, 0, TRIM_MAX * face_h)))
    else:
        # Coloured plates: the found blob may be the whole plate already (its outer rim) or only the
        # background inside a border line, so measure what still belongs to the plate outside it.
        lab = cv2.cvtColor(cv2.GaussianBlur(image_bgr[by1:by2, bx1:bx2], (3, 3), 0), cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)
        for i in range(4):
            t = _colour_edge_trim(lab, face[i], face[(i + 1) % 4], normals[i], face_h, *plate_colours, from_ring)
            trims.append(float(np.clip(0.0 if t is None else t, 0, TRIM_MAX * face_h)))
    outer = grow_quad(face, trims) + offset
    if not np.isfinite(outer).all() or not _plausible(outer, plate, max_area=1.6):
        return None
    return outer


def fit_plate(image_bgr: np.ndarray, plate: Plate) -> Plate:
    """Return the plate with `corners` (and `outline_from`) filled in, or unchanged if they can't be found."""
    found = find_plate_outline(image_bgr, plate)
    if found is None:
        return plate
    corners, how = found
    return dataclasses.replace(plate, corners=tuple((round(float(x), 1), round(float(y), 1)) for x, y in corners),
                               outline_from=how)
