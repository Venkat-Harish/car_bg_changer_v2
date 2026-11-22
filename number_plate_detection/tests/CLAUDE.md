# number_plate_detection/tests: synthetic plate-colour test set and its runner (scripts, not pytest)

## Purpose
The real videos only show white plates and black/yellow plates. `make_test_photos.py` pastes sample plates of every colour onto real frames, where the true outline and lighting are known. `run_colour_tests.py` masks those photos and judges the outline and brightness.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `make_test_photos.py` | Builds `../assets/test_photos/<case>/<view>.jpg` and `expected.csv` | `main()` with `--plates`, `--out`, `--frames`, `--redraw`; `draw_plate`, `prepare_plates`, `expected_reading`, `plate_white`, `paste`; `STANDARD_PLATES`, `PLATE_STYLES`, `VIEWS`, `FIELDS`, `NIGHT_DARKEN` 0.25, `PLATE_W, PLATE_H` 1040x224 |
| `run_colour_tests.py` | Masks every test photo with auto brightness ON and OFF, then judges each one | `main()` with `--case` (repeatable), `--pause`, `--photos`, `--out`, `--config`; `load_cases`, `judge`, `overlap` (IoU), `board_white`, `crop`/`stack` (compare sheet); `PASS_OVERLAP` 0.90, `PASS_POINTS` 5.0 |

## How it works
`make_test_photos.py`:
1. Draws any missing `STANDARD_PLATES` (10 entries: name → background RGB, character RGB, description) as PNGs in `../assets/test_plates/`. `--redraw` overwrites them. Any other image in that folder becomes a case named after its stem, with its expected reading worked out from 2 k-means colours.
2. For each view in `VIEWS` (`front` = `0000.0s_*`, `front_angled` = `0034.0s_*`, `rear_angled` = `0024.0s_*`), takes the first matching JPG in `../../frame_view_type/output/Final_zoomed_car/`. It runs `PlateDetector()` with code defaults (not `config.toml`), then `fit_plate`, and keeps the largest plate with corners. The truth level is `plate_white`: the 80th percentile of the Otsu-light pixels at least 0.16·h inside the plate.
3. Pastes each sample plate with `image_overlay(light_plane=(level,0,0))` plus a slight Gaussian soften. `night_flash` is the frame at 25% brightness with `white_black` pasted at full brightness.
4. Writes `<case>/<view>.jpg` (JPEG q92) and `expected.csv` (utf-8-sig) with columns `case, view, file, plate, expected_reading, true_level, true_pct, outline`. `expected_reading` is white, light paint or photo; `outline` is TL TR BR BL "x,y". It deletes only case folders listed in the old `expected.csv` that no longer exist.

`run_colour_tests.py`:
- **Masker setup.** Reads `model`, `confidence`, `threads`, `overlay_image`, `lighting` and `color` through `mask_plates.load_config`. It forces `tight_fit=True`, `pad=0`, overlay on and `BrightnessDetector()` with no smoothing, so `overlay`, `auto_brightness`, `padding`, `tight_fit` and `brightness_smoothing` in the config are ignored.
- **Each photo.** Runs `masker.process` (auto ON), then calls `plate_masker.mask_plates` again on the same plates with `brightness=None` (auto OFF, using the config's `lighting`). It judges the plate with the highest IoU against the truth outline.
- **PASS needs all of these:**
  - the plate is detected
  - fit is not "box" ("letters" is fine)
  - IoU ≥ 0.90
  - the auto-ON board white (75th percentile of the quad shrunk to 70%) is within 5 points of `true_pct`

  If `expected_reading == "photo"`, the source must start with "photo" instead, and brightness is not checked.
- **Output.** Writes to `../output/colour_tests/<case>/`: the masked photos, `compare.jpg` (test photo | auto OFF | auto ON) and `results.csv`. A `summary.csv` covering all cases goes in `../output/colour_tests/`.
- **Folders without answers.** Folders under `test_photos` that have no `expected.csv` rows (your own real photos) are run too, with numbers only (result "-").

## Invariants & gotchas
- These are not pytest tests. Nothing asserts, and the exit code is 0 even when cases FAIL. Read the console or `summary.csv`.
- Both scripts need the detector model, which downloads on first use. `make_test_photos.py` also needs frame_view_type's `Final_zoomed_car` frames. Both the video and `output/` are gitignored. Because `assets/test_photos` is tracked, the runner still works on a fresh clone. To regenerate, run `classify_video.py Final_zoomed_car.mp4` with `every_seconds` set to a divisor of 24 and 34 (e.g. 2).
- `VIEWS` depends on frame_view_type's `<t:06.1f>s_` name prefix.
- Results depend on `../config.toml`: the overlay image (`RC.png`), model and confidence.
- `run_colour_tests.py` imports `mask_plates` as a module and uses `load_config`, `load_overlay`, `read_image`, `write_image`, `IMAGE_EXTS` and `DEFAULT_*`. Keep those names, including the re-imported `DEFAULT_*` constants.
- The truth is partly self-referential:
  - The truth level is measured with the same 0.16·h inset as `INSET_FITTED`.
  - Plates are pasted with `image_overlay` itself.
  - `expected_reading` uses `PAINT_FACTORS`, `MIN_FACTOR` and `name_colour` from the module under test, so retuning them also changes the expected labels after regeneration.

  The synthetic plates are flat-lit (no gradient).
- Both scripts add `..` to `sys.path` and build paths from their own location, so they run from any cwd.
- README says "all 11 cases pass on all 3 views" (33 photos). That has not been re-verified here.

## How to extend
- **New plate colour case:** add an entry to `STANDARD_PLATES`. For an unusual layout, also add `PLATE_STYLES[name] = {"border": ..., "text": ..., "text_h": ..., "text_w": ...}`. Run `make_test_photos.py` (it only draws missing PNGs), then `run_colour_tests.py --case <name>`. Add a row to `../assets/test_plates/README.md`. The quick alternative is to drop a plate image into `../assets/test_plates/` and regenerate.
- **New view:** add `name: "<time prefix>"` to `VIEWS`. The prefix must match a frame that exists in `--frames`.
- **Real photos:** put a folder of photos under `../assets/test_photos/`. They are run with numbers only.
- **New pass criterion:** edit `judge()` and add a `PASS_*` constant. Add any new field to the `results` dict. `summary.csv` averages only `true_pct`, `auto_on_pct` and `auto_off_pct`, so a new field needs its own summary entry.

## Tests
These scripts are the only tests for [plate_masker](../plate_masker/CLAUDE.md) and [plate_brightness_detector](../plate_brightness_detector/CLAUDE.md). Nothing tests the scripts themselves.
