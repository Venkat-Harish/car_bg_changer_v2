# plate_masker: YOLOv9 plate detector, 4-corner plate outline fit, and mask styles

## Purpose
The library behind `../mask_plates.py`. It finds plate boxes with an ONNX detector, turns each box into the plate's own quad with plain OpenCV, and paints an overlay or a solid fill inside the quad. It has no CLI and does no file I/O.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Public API; `PlateMasker` = detector + fit + brightness + mask in one object | `PlateMasker.__init__(model, conf, pad, threads, *, tight_fit, overlay, lighting, brightness_detector, color, style, **mask_opts)`, `PlateMasker.process(image_bgr) -> (masked, plates)`; re-exports everything in `__all__` |
| `detector.py` | Thin wrapper over open-image-models (CPU ONNX Runtime) | `Plate` (frozen dataclass), `PlateDetector.detect()`, `DEFAULT_MODEL` (`yolo-v9-t-384-license-plate-end2end`), `DEFAULT_CONF` 0.25, `DEFAULT_THREADS` 4, `_download_help()` |
| `corners.py` | Outline fitting (README: about 4-8 ms for light plates, about 20 ms for coloured ones) | `fit_plate()`, `find_plate_outline()`, `find_plate_corners()`; internal `_edge_outline`, `_letters_quad`, `_plausible`, `_inside_box`; helpers `grow_quad`, `order_corners`, `quad_size`, `background_side`; tuning constants at the top |
| `masking.py` | Mask styles and the paint loop | `MASKERS`, `solid_fill`, `image_overlay`, `_plate_light`, `mask_plates()`, `pad_quad()`, `DEFAULT_PAD` 0.0, `DEFAULT_COLOR_RGB` (0,0,0), `DEFAULT_LIGHTING` 100.0, `LIGHT_GAIN_MIN` 0.15, `_SHIFT` 4 |

`Plate` fields: `x1, y1, x2, y2` (original pixels, x2/y2 exclusive), `confidence`, `corners` (TL TR BR BL tuple, or None), `brightness` (a `BrightnessReading`, or None), `outline_from` ("corners" | "letters").
Properties: `.fit` ("corners" | "letters" | "box"), `.quad` (float32 (4,2): the corners, or the box's inclusive corners `x2-1, y2-1`), `.width`, `.height`.

## How it works
`PlateMasker.__init__` picks the style. It uses `style` if given, otherwise `"image"` if an overlay was passed, otherwise `"solid"`. It validates the style against `MASKERS` and `lighting` against 0-100, builds the `PlateDetector`, and stores `mask_opts = {color, overlay, lighting, **extra}`.

`PlateMasker.process(image)`:
1. `detector.detect`: open-image-models `predict`, with `conf_thresh` applied inside. Boxes are clamped to the image, empty boxes are dropped, and the rest are sorted by confidence, highest first.
2. If `tight_fit`, `fit_plate` runs on each plate. It fills in `corners` (rounded to 0.1 px) and `outline_from`, or returns the plate unchanged, which gives fit "box".
3. If a `brightness_detector` is set, `measure_all(image, [p.quad], [corners found?])` runs on the **original** image, and each result goes into `Plate.brightness`.
4. `mask_plates(image, plates, style, pad, box_style, **mask_opts)` copies the image, then paints each plate. Plates with corners get `style`. Plates without corners get `box_style`: `"solid"` when `tight_fit` is on, or `None` (meaning the same style) when it is off. Each call is `MASKERS[name](out, pad_quad(p.quad, pad), light_plane=p.brightness.plane or None, **opts)`.

`corners.find_plate_outline(image, plate)` tries three things in order:
1. **`_edge_outline`** runs only if the box is at least 12x6 px. It uses two crops: a big one (box ± 0.5·max(w,h)) for the trim profiles, and a small one (box ± `THRESH_MARGIN` 0.15) for thresholding. It tries several (blur, opening) passes: `(5,k),(5,2k+1),(3,0)` if h ≥ 50, else `(3,0),(3,3),(5,0)`, where `k = odd(max(3, 0.04·min(w,h)))`. A candidate is accepted only if it passes all three checks:
   - `_plausible(max_area=1.1)`
   - `_inside_box(FACE_BOX_MARGIN 0.10)`
   - `background_side(...) == expected side`

   The candidates come from these passes:
   - a. **Light face:** the Otsu-bright blob under the box centre, with an area of 0.3-1.25x the box. Expected side: "light".
   - b. **Coloured plates** (only if a. failed): `_two_colours` runs k-means (Lab) on the box interior, trimmed by 25% at the top and bottom and 15% at the sides. It then tries colour masks in this order:
     - darker colour = background ("dark")
     - lighter colour = background ("light")
     - lighter colour = a border ring around a dark background (`from_ring`)

   It then pushes each edge outward:
   - Light plates use `_edge_trim`: 7 profiles across the edge (white face → dark line → light rim), median, `TRIM_TYPICAL` 0.14·h if an edge can't be measured, capped at `TRIM_MAX` 0.22·h.
   - Coloured plates use `_colour_edge_trim` (`LINE_MAX`/`RIM_MAX` 0.12, `RIM_TYPICAL` 0.06).

   Then `grow_quad` and a final check, `_plausible(max_area=1.6)`.
2. **`_letters_quad`** runs only if the box is at least 16x8 px. It applies Otsu inside the box, then `_character_row` tries light characters, then dark ones. A row needs at least 3 components with similar heights, on a line tilted at most `MAX_TILT` 40°, each with ink ≥ 30% of the median, spanning ≥ 0.35·w. The quad is built from:
   - lines fitted along the characters' tops and bottoms (tilt and perspective)
   - sides from the letters' stroke axes (`_letters_down`, ≤35° off perpendicular)
   - margins: `LETTER_MARGIN_V` 0.35·letter height, `LETTER_MARGIN_H` 0.045·row length, scaled per end for perspective

   The ends are pulled back inside box ± `LETTER_BOX_MARGIN_X` 0.05. Final checks: `_plausible(1.6)` and `_inside_box(0.25)`.
3. Otherwise it returns None, and the plate keeps its box.

`_plausible` requires all of the following:
- min edge ≥ 0.15·min(box w,h)
- convex
- every edge within 40° of horizontal or vertical
- aspect ratio 1.0-8.0
- area 0.2-max_area × box area
- spans ≥ 60% of the box in both x and y

`masking.image_overlay(image, quad, *, overlay, lighting, light_plane)`:
1. Works on the quad's bounding-box ROI. If the overlay is larger than 1.5x the plate's edge lengths, it is first shrunk with `INTER_AREA`, because `warpPerspective` has no area filter.
2. Maps the overlay's outer pixel edges (−0.5) onto the corners with `warpPerspective` (`INTER_LINEAR`, `BORDER_REPLICATE`).
3. Builds alpha as an anti-aliased `fillConvexPoly` (`_SHIFT` = 1/16 px) times the PNG's own alpha.
4. Sets brightness:
   - with `light_plane` (auto mode): gain = clip(plane/255, 0.15, 1).
   - else if `lighting` > 0: `_plate_light` fits a plane to the Otsu-light pixels inside the quad shrunk by 0.2·h. It needs ≥ 50 px, ≥ 20% of them light, and refits once to drop outliers. Gain = 1 − s·(1 − clip(light/255, 0.15, 1)). If the plane is None, there is no darkening.
5. Float blend, +0.5 rounding.

## Invariants & gotchas
- **Corner order TL, TR, BR, BL everywhere.** This covers `order_corners`, `grow_quad` (edges 0 top, 1 right, 2 bottom, 3 left), `pad_quad`, the warp source points, `plate_brightness_detector` and `detections.csv`. Breaking it twists the warp and grows the wrong edges.
- `Plate` is frozen: change it with `dataclasses.replace`. `corners is None` ⇔ fit "box".
- Every masker must paint in place and accept `**_`. `mask_plates` passes `light_plane`, `color`, `overlay`, `lighting` and any extra `mask_opts` to every style.
- `mask_plates` never modifies its input. Brightness must be measured on the unpainted image: `process` does that, and `_plate_light` reads the ROI before blending.
- The overlay is only ever darkened (gain ≤ 1, floor 0.15). Auto `light_plane` overrides `lighting` completely.
- When the corner fit fails, the plate gets a **solid box even with overlay on**. This is deliberate: an image stretched over a straight box on a tilted plate, grille or badge looks wrong. With `tight_fit=False`, every plate gets the chosen style on its box.
- The thresholds were measured on two real videos and the synthetic set (see [../docs/findings.md](../docs/findings.md)):
  - 4K `Final_zoomed_car`, white plates
  - 464x832 `sample_vid`, black/yellow plates of about 80x25 px with 11 px letters

  After changing any of them, rerun `../tests/run_colour_tests.py`. findings.md says the letters fix left the 4K output pixel-identical; aim to keep that.
- The pass lists are split at h ≥ 50 because morphological opening cuts a plate free of a white bumper but also erases the 1-2 px white margin around small plates' letters.
- `background_side` (0.6 share rule) is duplicated as `_background_side` in `../plate_brightness_detector/detector.py`. The two packages deliberately don't import each other, so change both together.
- `PlateDetector` imports `onnxruntime`/`open_image_models` lazily inside `__init__`. `corners.py` and `masking.py` need only cv2 and numpy; `tests/make_test_photos.py` uses `image_overlay` on its own.
- `threads` > 0 sets ORT `intra_op_num_threads=threads, inter_op_num_threads=1`. 0 passes no session options, so ORT uses all cores.
- A model string ending in `.onnx` is a local file (backend `yolo_v9`, labels `["License Plate"]`, `FileNotFoundError` if missing). Anything else must be a registered open-image-models name: the 256/384/416/512/640 `yolo-v9-t-*` models or `yolo-v9-s-608-license-plate-end2end`. A download `URLError` is re-raised as `RuntimeError` with manual steps.
- "letters" outlines assume the letters fill most of the plate. On US-style plates the cover will be undersized, though the letters are still covered. Rounded plate corners are fitted sharp, so the cover can reach 1-2 px onto the holder.

## How to extend
- **New mask style (e.g. blur):** in `masking.py`, write `def blur(image, quad, *, ksize=31, **_) -> None` that paints `image` in place inside `quad`; build an anti-aliased mask with `_fixed(quad)` + `cv2.fillConvexPoly(..., lineType=cv2.LINE_AA, shift=_SHIFT)` (without `LINE_AA` it is hard-edged, as in `_plate_light`). Privacy: scale the kernel with the quad's size (a fixed 31 px won't hide 4K letters) and blur only pixels inside the quad, so nothing outside it leaks in or gets altered. Register it with `MASKERS["blur"] = blur`. Use it as `PlateMasker(style="blur", ksize=...)`: extra kwargs reach the style through `**mask_opts`. Box plates still get "solid" while `tight_fit` is on; change the `box_style` line in `process` if the new style should cover boxes too. Expose it in the CLI through a config key ([../CLAUDE.md](../CLAUDE.md)).
- **New outline method:** write a function returning a float32 TL,TR,BR,BL quad or None, gated by `_plausible` + `_inside_box`. Call it from `find_plate_outline` with a new `how` string. `Plate.fit` then reports that string in the console and CSV. `run_colour_tests.judge` fails only "box".
- **New detector backend:** keep the `PlateDetector.detect` contract: `list[Plate]` in original-image pixels, clamped, highest confidence first.
- **Tuning:** constants live at the top of `corners.py`/`masking.py`; each comment records what it was measured on. After a change, rerun the colour tests and check `sample_vid`.

## Tests
No unit tests. [`../tests/run_colour_tests.py`](../tests/CLAUDE.md) runs `PlateMasker` end to end and checks four things:
- the plate is detected
- fit is not "box"
- outline IoU ≥ 0.90 against `expected.csv`
- brightness within ±5 points

`../tests/make_test_photos.py` builds the truth with `PlateDetector`, `fit_plate`, `grow_quad`, `quad_size` and `image_overlay`.
