"""Cut the main car out of a photo.

A background-removal model (rembg, ONNX Runtime, CPU) gives a soft alpha for the salient object(s).
`clean_alpha` then turns that into an alpha for the *main car only*:

1. threshold at 0.5 and keep the biggest, most central blob (drops parked cars in the background),
   plus small detached pieces that sit wholly inside its box. A thin post standing behind the car
   (lamp post, sign pole) that touches its outline in the photo is cut off whole (`drop_posts`);
2. fill holes inside the outline, so windows keep the original glass instead of turning see-through;
3. force alpha = 1 everywhere inside the outline except a thin edge band ("core"). Inside the core the
   car is copied pixel for pixel; only the edge band is blended with the new background;
4. optionally pull the soft edge in by `edge_shrink` px to drop a fringe of the old background.

With the optional `car_isolation` step (config `[car_isolation] enabled = true`), two masks from an
instance model refine this: `remove` (other vehicles and objects stuck to the car in the photo) is
cleared from the model's alpha first, and `glass_ok` limits step 2 to holes that lie on the target car
(windows), so gaps such as mirror-to-pillar stay open. Both default to None: unchanged behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_MODEL = "birefnet-general-lite"
DEFAULT_THREADS = 4  # fastest on a 4 performance + 8 efficiency core laptop CPU; 0 = ONNX Runtime decides
DEFAULT_EDGE_SHRINK = 2  # pixels at source resolution
CORE_BAND = 0.003  # width of the soft edge band, as a share of the image's longer side
EDGE_KEEP = 0.004  # soft alpha kept this far outside the solid outline, same unit
POST_WIDTH = 0.012  # posts narrower than this (same unit; ~46 px at 4K) that stick out of the car are cut off
BORDER_TOL = 3  # a car pixel this close (px) to the photo edge counts as "cut off" there


class NoCarFound(RuntimeError):
    pass


@dataclass
class CarMask:
    """Result of cutting the car out. All coordinates are source-image pixels, x2/y2 exclusive."""

    alpha: np.ndarray  # float32 HxW, 0..1
    box: tuple[int, int, int, int]  # solid outline (alpha >= 0.5, holes filled): used for placement
    crop: tuple[int, int, int, int]  # every pixel with alpha > 0: what gets copied
    cut: dict[str, bool] = field(default_factory=dict)  # left/right/top/bottom: car touches that photo edge
    bottom: np.ndarray | None = None  # per box column: lowest row of the solid outline (-1 = none)
    isolation: dict | None = None  # car_isolation's summary for results.csv (None = the step did not run)

    @property
    def cut_sides(self) -> list[str]:
        return [side for side, is_cut in self.cut.items() if is_cut]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _disk(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def select_main_car(binary: np.ndarray) -> np.ndarray:
    """Keep the biggest, most central blob, plus small blobs lying wholly inside its box."""
    h, w = binary.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        raise NoCarFound("the model found no object in the image")
    half_diag = np.hypot(w / 2, h / 2)
    best, best_score = 0, -1.0
    for i in range(1, n):
        cx, cy = centroids[i]
        off_centre = np.hypot(cx - w / 2, cy - h / 2) / half_diag  # 0 = centre, 1 = corner
        score = stats[i, cv2.CC_STAT_AREA] * (1.0 - 0.5 * off_centre)
        if score > best_score:
            best, best_score = i, score
    bx, by, bw, bh = stats[best, :4]
    keep = [best]
    for i in range(1, n):
        x, y, cw, ch = stats[i, :4]
        if i != best and x >= bx and y >= by and x + cw <= bx + bw and y + ch <= by + bh:
            keep.append(i)
    return np.isin(labels, keep).astype(np.uint8)


def drop_posts(mask: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(main car without posts, the removed posts): thin posts standing behind the car that touch
    its outline in the photo.

    An opening with a horizontal line `width` px long removes everything narrower than `width`
    side to side: posts and poles. Mirrors, wipers and roof rails stick out sideways, so they are
    wide in that direction and stay. Every removed piece that reaches more than `width` px out of
    the remaining car is a post and is dropped whole, base included; shorter pieces are small edge
    details of the car itself (rounded tyre bottoms, corners) and are put back.
    """
    none = np.zeros_like(mask)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1)))
    if not opened.any():
        return select_main_car(mask), none
    main = select_main_car(opened)
    thin = mask & (1 - main)
    n, labels = cv2.connectedComponents(thin, connectivity=8)
    if n <= 1:
        return main, none
    far = thin.astype(bool) & ~cv2.dilate(main, _disk(width)).astype(bool)
    posts = np.isin(labels, np.unique(labels[far])) & (labels > 0)
    keep = (labels > 0) & ~posts
    return select_main_car(main | keep.astype(np.uint8)), posts.astype(np.uint8)


GLASS_SHARE = 0.6  # with `glass_ok`: a hole is filled only if at least this share of it lies on the target car


def fill_holes(mask: np.ndarray, glass_ok: np.ndarray | None = None) -> np.ndarray:
    """Fill the holes inside the outline. With `glass_ok` (bool HxW), only holes lying mostly (GLASS_SHARE)
    on it are filled (windows); the others (mirror-to-pillar gaps, open spokes) stay open."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    if glass_ok is None:
        return filled
    holes = filled & (1 - mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    if n <= 1:
        return filled
    on_car = np.bincount(labels[glass_ok], minlength=n)
    keep_open = on_car < GLASS_SHARE * stats[:, cv2.CC_STAT_AREA]
    keep_open[0] = False
    filled[keep_open[labels]] = 0
    return filled


def clean_alpha(raw_alpha: np.ndarray, edge_shrink: int = DEFAULT_EDGE_SHRINK, remove: np.ndarray | None = None,
                glass_ok: np.ndarray | None = None, trace: dict | None = None) -> CarMask:
    """Turn the model's soft alpha (float 0..1, HxW) into the main car's alpha. See module docstring.
    `remove` / `glass_ok` (bool HxW, optional) come from car_isolation. `trace`: a dict that, when given,
    receives each step's mask for the debug pictures (binary, posts, car, solid, core); nothing else changes."""
    h, w = raw_alpha.shape
    longer = max(h, w)
    if remove is not None:
        raw_alpha = np.where(remove, np.float32(0), raw_alpha.astype(np.float32))
    binary = (raw_alpha >= 0.5).astype(np.uint8)
    if not binary.any():
        raise NoCarFound("the model found no object in the image")
    # Fill windows first, so thin pillars between see-through windows never look like posts.
    car, posts = drop_posts(fill_holes(binary, glass_ok), max(3, round(POST_WIDTH * longer)))
    solid = fill_holes(car, glass_ok)
    if trace is not None:
        trace.update(binary=binary.astype(bool), posts=posts.astype(bool), car=car.astype(bool),
                     solid=solid.astype(bool), remove=remove)

    # Soft alpha only near the main car's outline (removes faint halos of other objects),
    # and none on a removed post or its anti-aliased rim.
    near = cv2.dilate(solid, _disk(max(1, round(EDGE_KEEP * longer))))
    near &= 1 - cv2.dilate(posts, _disk(2))
    alpha = raw_alpha.astype(np.float32) * near
    # Inside the outline (minus the edge band) the car is fully opaque: original pixels, nothing blended.
    core = cv2.erode(solid, _disk(max(1, round(CORE_BAND * longer))))
    if trace is not None:
        trace["core"] = core.astype(bool)
    alpha = np.maximum(alpha, core.astype(np.float32))
    if edge_shrink > 0:
        alpha = cv2.erode(alpha, _disk(edge_shrink))
        alpha[core.astype(bool)] = 1.0
    alpha[alpha < 1 / 255] = 0.0

    t = BORDER_TOL
    cut = {
        "left": bool(solid[:, :t].any()),
        "right": bool(solid[:, w - t :].any()),
        "top": bool(solid[:t, :].any()),
        "bottom": bool(solid[h - t :, :].any()),
    }
    box = _bbox(solid)
    return CarMask(alpha=alpha, box=box, crop=_bbox(alpha > 0), cut=cut, bottom=bottom_profile(solid, box))


def bottom_profile(solid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Lowest solid row in each column of `box` (absolute rows; -1 where a column is empty)."""
    x1, y1, x2, y2 = box
    cols = solid[y1:y2, x1:x2].astype(bool)
    has = cols.any(axis=0)
    last = (y2 - y1 - 1) - np.argmax(cols[::-1, :], axis=0)
    return np.where(has, last + y1, -1).astype(np.int32)


class CarSegmenter:
    """Loads the background-removal model once. `cut_out(image_bgr)` -> CarMask."""

    def __init__(self, model: str = DEFAULT_MODEL, edge_shrink: int = DEFAULT_EDGE_SHRINK, threads: int = DEFAULT_THREADS):
        # Imported here so compose-only use (and the tests) need no model packages.
        import onnxruntime as ort
        from rembg import new_session

        self.model = model
        self.edge_shrink = edge_shrink
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = new_session(model, providers=["CPUExecutionProvider"], sess_opts=opts)
        self.last_isolation = None

    def raw_alpha(self, image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        from rembg import remove

        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mask = remove(rgb, session=self.session, only_mask=True)
        alpha = np.asarray(mask, dtype=np.float32) / 255.0
        if alpha.shape != image_bgr.shape[:2]:
            alpha = cv2.resize(alpha, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return alpha

    def release(self) -> None:
        """Free the model's memory (raw_alpha no longer works; finish() still does)."""
        self.session = None

    def cut_out(self, image_bgr: np.ndarray, isolator=None) -> CarMask:
        """`isolator`: an optional car_isolation.CarIsolator; None = as before."""
        return self.finish(image_bgr, self.raw_alpha(image_bgr), isolator)

    def finish(self, image_bgr: np.ndarray, raw: np.ndarray, isolator=None, trace: dict | None = None) -> CarMask:
        """The car's mask from a raw alpha (float 0..1, or uint8 0..255 as cached by a two-pass run).
        `trace`: see clean_alpha (debug pictures only)."""
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / 255.0
        if isolator is None:
            return clean_alpha(raw, self.edge_shrink, trace=trace)
        found = isolator.isolate(image_bgr, raw)
        self.last_isolation = found  # kept for a debug overlay
        try:
            car = clean_alpha(raw, self.edge_shrink, found.remove, found.glass_ok, trace=trace)
        except NoCarFound:  # everything was removed: keep today's cut-out rather than lose the photo
            car = clean_alpha(raw, self.edge_shrink, trace=trace)
            found.status, found.remove, found.glass_ok = "unsure", None, None
        car.isolation = found.summary()
        return car
