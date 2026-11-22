# number_plate_detection: find number plates in car images and cover exactly the plate (CPU, ONNX, no torch)

## Purpose
Second tool in the chain. It detects plates with a YOLOv9 ONNX model, fits each plate's own 4 corners, and covers them
with a warped overlay image lit to match, or with a solid colour. It writes masked copies under the same file names, plus `detections.csv`.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `mask_plates.py` | CLI entry: config + CLI merge, folder loop, image I/O, `detections.csv` | `main()`, `load_config()`, `list_images()`, `read_image()`/`write_image()` (imdecode/imencode), `load_overlay()`; `CONFIG_KEYS`, `PATH_KEYS`, `BOOL_KEYS`, `IMAGE_EXTS`, `DEFAULT_OVERLAY_IMAGE` |
| `config.toml` | The 13 allowed settings, each with a comment | `input`, `output`, `overlay`, `overlay_image`, `auto_brightness`, `brightness_smoothing`, `lighting`, `color`, `tight_fit`, `padding`, `model`, `confidence`, `threads` |
| `requirements.txt` | `open-image-models[onnx]==0.6.0`, `opencv-python-headless`, `numpy` (unpinned) | — |
| `README.md` | User docs: settings, CLI table, test set, speed (partly stale, see gotchas) | — |
| `assets/RC.png` | Overlay set in `config.toml` (`overlay_image`): brand logo on a white board | — |
| `assets/sample_plate.png` | Code fallback overlay (`DEFAULT_OVERLAY_IMAGE`) when `overlay_image` is unset; neutral placeholder | — |
| `assets/test_plates/` | 10 flat 1040x224 sample plates + README; input to `tests/make_test_photos.py` | — |
| `assets/test_photos/` | Generated colour test set, tracked in git: `<case>/{front,front_angled,rear_angled}.jpg` for 11 cases (10 colours + `night_flash`) + `expected.csv` | columns `case, view, file, plate, expected_reading, true_level, true_pct, outline` |
| `docs/findings.md` | Research notes and measured results. Some early sections are superseded by later dated sections. | — |
| [`plate_masker/`](plate_masker/CLAUDE.md) | Detector, 4-corner fit, mask styles | `PlateMasker`, `Plate`, `MASKERS` |
| [`plate_brightness_detector/`](plate_brightness_detector/CLAUDE.md) | Overlay brightness for any plate colour | `BrightnessDetector` |
| [`tests/`](tests/CLAUDE.md) | Script harness for the colour test set (not pytest) | `make_test_photos.py`, `run_colour_tests.py` |

## How it works
1. `load_config` reads `config.toml` (or `--config`, which must then exist) as `utf-8-sig`. It rejects unknown keys (and hints if it sees `style`), type-checks each value, and resolves `PATH_KEYS` and any `model` ending in `.onnx` against the config's folder. CLI flags win through `pick()`. Missing keys fall back to `plate_masker` constants.
2. `list_images(input)`: a single file, or a non-recursive scan of a folder for `IMAGE_EXTS` (`.jpg .jpeg .png .bmp .webp`), sorted by name.
3. `out_dir = <output>/<file stem or folder name>`. The script refuses to run if this equals the input folder.
4. `PlateMasker(model, conf, pad, threads, tight_fit=, overlay=, lighting=, brightness_detector=, color=)`. `BrightnessDetector(smoothing)` is built only when `overlay` and `auto_brightness` are both on.
5. For each image: `masker.process()`, then the result is written under the **same file name** (JPEG quality 95, other formats by extension), plus one CSV row per plate.
6. `detections.csv` is written as `utf-8-sig`. Columns: `file, plates, plate, fit, x1, y1, x2, y2, confidence, corners, brightness_pct, brightness_source, plate_colours`. `fit` is corners, letters or box. `x2`/`y2` are exclusive. `corners` holds `"x,y x,y x,y x,y"` in TL TR BR BL order. An image with no plate gets one row with `plates=0`.

Cross-project contract:
- The default input `../frame_view_type/output/sample_vid` comes from [frame_view_type](../frame_view_type/CLAUDE.md). The output goes to `output/sample_vid`, keeping the **folder name and file names**. background_changer's default `images_folder` is `../number_plate_detection/output/sample_vid`. It parses the view label from each `<t>s_<side>_<conf>pct` name and, in room mode, reads `frame_view_type/output/<same folder name>/results.csv` ([../background_changer/CLAUDE.md](../background_changer/CLAUDE.md)).
- [video_pipeline](../video_pipeline/CLAUDE.md) runs `mask_plates.py <frames> --out <masked> --overlay --auto-brightness --brightness-smoothing N`. Every other setting comes from this `config.toml`.
- `tests/run_colour_tests.py` imports this script as a module.

## Invariants & gotchas
- Needs Python 3.11 because the code uses `tomllib`. README says "3.10+", which is wrong. README also says the default input is `Final_zoomed_car`; `config.toml` says `sample_vid`.
- The code fallbacks differ from the values in `config.toml`: `overlay=False`, `auto_brightness=False`, `overlay_image=assets/sample_plate.png`. Deleting one of these keys changes behaviour. The other fallbacks match `config.toml`: lighting 100, color [0,0,0], tight_fit true, padding 0.0, model `yolo-v9-t-384-license-plate-end2end`, confidence 0.25, threads 4, smoothing 0.
- Never rename outputs or the output folder. Downstream tools key on both.
- Outputs are re-encoded (JPEG q95), never byte copies, even when an image has no plate. An unreadable input is skipped: it gets no output file and no CSV row. That breaks video_pipeline's count-based skip check.
- Keep `np.fromfile`+`cv2.imdecode` / `cv2.imencode`+`tofile`. `cv2.imread` fails on non-ASCII Windows paths.
- `color` is RGB in config and CLI. It is converted to BGR only inside `masking.solid_fill`.
- `confidence` must be strictly between 0 and 1. A missed plate is a privacy leak, and lowering confidence costs nothing (0.15 caught background plates that 0.25 missed). `threads=0` uses all cores, which was about 4x slower on the i5-1240P.
- `brightness_smoothing` is one `BrightnessDetector` for the whole run, fed in sorted file-name order. It only makes sense for frames of one video whose names sort by time (frame_view_type's zero-padded `0012.0s_`, video_pipeline's `00001.jpg`). Keep it at 0 for unrelated photos.
- The brightness columns stay empty unless `overlay` and `auto_brightness` are both on.
- The CSV header comes from `rows[0]`. The no-plate `empty` dict and the per-plate dict must have the same keys in the same order; otherwise `DictWriter` raises when the first image has no plate.
- The first run downloads the model to `%USERPROFILE%\.cache\open-image-models`. On a `URLError`, the error message gives a manual-download recipe (`models\<file>.onnx` + `model = "models/<file>"`).
- Reruns overwrite same-name files and never delete stale ones.

## How to extend
- **New config key:** add it to `CONFIG_KEYS` (and to `PATH_KEYS`/`BOOL_KEYS` if it is a path or bool), add validation in `load_config`, an argparse flag and a `pick()` line in `main()`, and pass it into `PlateMasker`/`BrightnessDetector`. Then add a commented entry to `config.toml` and update the README settings and CLI tables. If the colour tests should honour it, also update the `PlateMasker(...)` call in `tests/run_colour_tests.py`.
- **New `detections.csv` column:** add it to both `empty` and the per-plate dict in `main()`.
- **New mask style:** add it to `MASKERS` (see [plate_masker/CLAUDE.md](plate_masker/CLAUDE.md)). Exposing it needs a config key. If you name that key `style`, remove the "'style' was replaced" hint in `load_config`. Decide its precedence over `overlay` explicitly: `main()` derives overlay loading, the `BrightnessDetector` and the console "mask:" text from `use_overlay` alone, `PlateMasker` picks `style or ("image" if overlay else "solid")`, and video_pipeline always passes `--overlay --auto-brightness`. `tests/run_colour_tests.py` hard-codes the image style and judges overlay brightness, so it does not test other styles.
- **New input format:** add it to `IMAGE_EXTS` (`run_colour_tests.py` reuses it).

## Tests
`tests/make_test_photos.py` and `tests/run_colour_tests.py` are scripts that need the model, not pytest. See [tests/CLAUDE.md](tests/CLAUDE.md). Nothing unit-tests `mask_plates.py` itself.
