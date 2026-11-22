# plate_brightness_detector: how bright the overlay's white should be on a plate, for any plate colour

## Purpose
Takes an image and a plate quad, and returns a `BrightnessReading`: a level, plus a plane `a + b·x + c·y` in image pixels. `plate_masker.image_overlay` uses that plane as `light_plane`. The package never imports plate_masker. The user-facing explanation is [README.md](README.md).

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Re-exports | `BrightnessDetector`, `BrightnessReading`, `PAINT_FACTORS`, `MIN_FACTOR`, `name_colour` |
| `colours.py` | Paint factor table and hue naming | `PAINT_FACTORS` (white 1.0, yellow 0.75, orange 0.60, red 0.30, green 0.35, blue 0.25, black 0.05), `MIN_FACTOR` 0.5, `NEUTRAL_SATURATION` 60, `name_colour(bgr)` |
| `detector.py` | The reading ladder and video smoothing | `BrightnessDetector(smoothing=0, glow_ratio=2.0, photo_percentile=95.0)` with `.measure()`, `.measure_all()`, `.photo_level()`, `.reset()`; `BrightnessReading(level, plane, source, background, characters)` with `.percent`, `.level_at()`; constants `INSET_FITTED` 0.16, `INSET_BOX` 0.22, `BACKGROUND_SHARE` 0.6, `MIN_CONTRAST` 25, `MIN_CORE_SHARE` 0.30, `WORK_HEIGHT` 64 |
| `README.md` | Idea, steps, results table, tuning, limits | — |

## How it works
`measure(image, quad, fitted=True, photo=None)`:
1. **Photo level.** `photo_level` takes the 95th-percentile grey of the lower 2/3 of the image, subsampled to about 480 px; minimum 1.
2. **Crop and shrink.** The ROI is the quad's bounding box; anything smaller than 8x4 falls back to the photo reading. A plate taller than 64 px is shrunk with `INTER_AREA`; plane coordinates are mapped back to image pixels.
3. **Inset.** The quad is inset with `distanceTransform`, by 0.16·h if `fitted`, else 0.22·h. Fewer than 60 px left → photo.
4. **Split light from dark.** Otsu splits the plate. Each side needs at least 20 px and a median gap of at least 25 grey levels, else photo.
5. **Name the colours.** Each side's 1 px-eroded core is named by `name_colour` (OpenCV hue 0-179). A neutral light side is "white"; a neutral dark side is "black" (<110) or "grey".
6. **Find the background.** The background is the side whose largest connected piece holds ≥0.6 of its pixels while the other side's holds <0.6. The source part is then "background" (light side) or "characters" (dark side); if unclear, background and characters are "unknown" and the part is "paint".
7. **Thin-paint check.** If less than 30% of the light side survives a 3x3 erosion → photo. Thin letters on small plates have no pure pixels.
8. **Ladder.**
   - Light side white → factor 1.
   - Light side is a paint with `PAINT_FACTORS` ≥ `MIN_FACTOR` (yellow 0.75 or orange 0.60) → divide by that factor.
   - Otherwise → photo.
9. **Fit the plane.** A plane is fitted to the light core (≤4000 samples, one 2.5σ refit) and divided by the factor. The level at the quad centre is clipped to 1..255. If that level is more than `glow_ratio` (2) times the photo level → photo (glare or flash).

Photo readings use plane `(photo, 0, 0)` and source `"photo (<why>)"`.

`measure_all(image, quads, fitted)` computes `photo_level` once per image. With `smoothing > 0`, it then replaces only the largest-area quad (the main plate) with the median of the last `smoothing` levels; the plane is scaled to match and `", smoothed over N frames"` is appended to the source.

Callers and consumers:
- `plate_masker.PlateMasker.process` stores the reading in `Plate.brightness`.
- `masking.mask_plates` passes `.plane` on to `image_overlay`, which uses gain = clip(plane/255, 0.15, 1).
- `../mask_plates.py` constructs the detector with `smoothing=brightness_smoothing`, only when `overlay` and `auto_brightness` are both on. It writes three CSV columns:
  - `brightness_pct` = `.percent`
  - `brightness_source` = `.source`
  - `plate_colours` = `background/characters`
- `../tests/make_test_photos.py` uses `PAINT_FACTORS`, `MIN_FACTOR` and `name_colour` to label expected readings.

## Invariants & gotchas
- The README's rules match the code: 2x glow, 30% thin-paint, yellow 0.75, only factors ≥0.5, saturation 60, lower two-thirds, largest plate smoothed. It leaves out three things: orange (0.60) also counts as a light paint; there are extra photo fallbacks (too small, one colour only, contrast <25, plane fit failed); and `photo_percentile` is 95. Its result tables and speed figures are measurements and have not been re-run.
- `PAINT_FACTORS` are first estimates from simulations; only ≥`MIN_FACTOR` are ever divided by. Small yellow letters flip between "yellow" and "orange" (hue boundary 20), which swings the result between ÷0.75 and ÷0.6. `MIN_CORE_SHARE` exists because of this (`sample_vid` plates keep 8-16% after erosion; readable plates keep 45-83%).
- `.percent` caps at 100, but the plane can exceed 255 after dividing by the factor. `image_overlay` clamps the gain at 1.
- Smoothing state lives in one instance for one sequence. History advances only on frames that have plates. `reset()` exists, but `mask_plates.py` never calls it. The main plate is the largest by `contourArea`, not the most confident.
- Measure the original, unpainted image.
- Quads must be TL, TR, BR, BL. Height is the mean of the left and right edges.
- `_background_side` deliberately duplicates `plate_masker.corners.background_side` (same 0.6 rule) so the two packages don't import each other.
- A dirty white plate reads the same as a white plate in shade, so its overlay comes out slightly dark. The pixels alone can't tell these apart.

## How to extend
- **New paint colour or factor:** edit `PAINT_FACTORS` (the source string prints the factor). For a new hue name, add a bin in `name_colour`. `tests/make_test_photos.expected_reading` uses the same table, so regenerate the test set (`make_test_photos.py`), then run `run_colour_tests.py`. Add a sample plate in `../assets/test_plates/` that covers the new colour.
- **New fallback or sanity check:** add it in `measure()` before the ladder, returning `self._photo_reading(photo, "<why>", background, characters)`. `run_colour_tests.judge` accepts any source that starts with "photo".
- **New user-facing knob:** add a constructor argument here, pass it from `BrightnessDetector(...)` in `../mask_plates.py main()`, and add a config key ([../CLAUDE.md](../CLAUDE.md)).

## Tests
[`../tests/run_colour_tests.py`](../tests/CLAUDE.md) runs auto brightness ON vs OFF. Each result must land within ±5 points of `true_pct`; cases expected to use the photo reading must have a source starting with "photo". There are no unit tests.
