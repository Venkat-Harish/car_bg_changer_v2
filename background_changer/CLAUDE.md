# background_changer: cut the car out of each photo and place it on studio background(s) like a real car

## Purpose
CLI tool (CPU only, ONNX Runtime via rembg, no torch). `change_background.py` reads `config.toml`, cuts the car out
of every photo (BiRefNet), and either re-renders each background as a 3D room from the photo's own camera (room mode,
the default) or scales the car onto a fixed stage (flat mode). Repo-wide rules: [../CLAUDE.md](../CLAUDE.md).

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `change_background.py` | Entry point (1056 lines): config loading and validation, CLI, the per-photo loop, room stages 2-3, flat placement, results.csv | `main()`, `load_config`, `parse_fit` / `parse_isolation` / `parse_scene_overrides`, `precompute_raw`, `place_photo`, `folder_scales`, `place_room_photos`, `place_flat_fallback`, `room_row`, `debug_step`; `Photo`, `Run`, `RoomRun`; `CONFIG_KEYS`, `PATH_KEYS`, `FIT_KEYS`, `ISOLATION_KEYS`, `SCENE_KEYS`, `RENAMED`, `CSV_FIELDS`, `ROOM_CSV_FIELDS`, `ISOLATION_CSV_FIELDS`, `VENV_HINT` |
| `debug_images.py` | `debug_mode` pictures, one per step, captioned; pictures only | `DebugWriter` (`save`, `write_readme`), `cutout_pictures` (01-05), `fit_pictures` (06-11), `room_pictures` (12-16), `flat_picture` (12_flat_placement); `MAX_LONG = 1600` |
| `config.toml` | Shipped settings, every key commented; the only user interface | top keys + `[car_isolation]`, `[fit]`, optional `[scene.<name>]` |
| `cars.json` | Car list: JSON array, cm (`carModel`, `length`, `width`, `height`, `WheelBase`, `body` suv/hatch/sedan) | EQB 2024, Baleno 2022; parsed by `bg_changer/cars.py` |
| `requirements.txt` | Pins (Python 3.11): rembg[cpu] 2.0.86, onnxruntime 1.31.1, opencv-python-headless 5.0.0.93, numpy 2.4.6, pillow 13.0.0, scipy 1.17.1, pytest 9.2.0 | `pooch` (car_isolation downloads) arrives via rembg, not pinned |
| `tools/check_flat.py` | Flat run vs the pinned baseline, pixel for pixel; exit 1 on any difference | `compare(base, new)` |
| `tools/check_room.py` | Room-path review: real 3-stage flow on folders under `../number_plate_detection/output`, contact sheets, `index.html`, self-checks | `run_folder`, `chain_check`, `car_pixels_ok`; `DEFAULT_SETS`, `REF_YAW` |
| `tools/check_scenes.py` | Flat placement on a run's saved `cutouts/` (no model), acceptance limits | `car_from_cutout`; `EQB_EXPECT` |

Data: `backgrounds/` (`background_1/2/3.webp`, 1920x1080; any image dropped here is selectable by stem), `docs/`
(`findings.md` research notes, `spyne-360-spin-findings.md` competitor study,
`glass-cutout-camera-options.md` model and cost options for cut-out, glass and camera; information only), `output/`
(gitignored, holds the pinned baseline `output/_baseline_flat/{Final_zoomed_car,sample_vid}`). README.md is the user doc.

Subpackages (own docs; internals not repeated here):
- [bg_changer/](bg_changer/CLAUDE.md): cut-out (`CarSegmenter`, `clean_alpha`), cars.json, file-name probe, yaw, flat placement, compositing, grounding
- [bg_changer/scenes/](bg_changer/scenes/CLAUDE.md): per-background scene specs (`SCENES`, `load_scene`, `scene_key`, generic stage)
- [car_fit_to_bg/](car_fit_to_bg/CLAUDE.md): line camera, focal pooling, 3D pose fit, levelling, `place_in_room`, `rooms.room_for`
- [room3d/](room3d/CLAUDE.md): 3D room model and re-render from a new camera
- [car_isolation/](car_isolation/CLAUDE.md): optional YOLOX-s + SAM pass that keeps only the target car in the mask
- [background_scaler/](background_scaler/CLAUDE.md): shrink background + scene for small photos (flat placement only)
- [tests/](tests/CLAUDE.md): model-free pytest suite

## How it works
Run from `background_changer\` (PowerShell): `.venv\Scripts\python change_background.py [images] [flags]`. Flags
override config for one run: `--background NAME` (repeatable, or `all`), `--backgrounds-folder`, `--out`, `--car`,
`--cut-frames frame|skip|anchor|centre`, `--background-view room|flat`, `--max-output-size PX`, `--model`,
`--threads`, `--shadow`, `--quality`, `--background-scale folder|photo|off`, `--isolate/--no-isolate`, `--debug`
(on only), `--config FILE`. No flag for `cars_file`, `edge_shrink`, `save_cutouts`, `keep_background_sharp`,
`background_min_width`, `[fit]`, `[scene]`, `min_score`, `remove_objects`.

`main()` order:
1. `load_config`: rejects unknown keys and `RENAMED` old keys (message names the replacement); type checks; `PATH_KEYS`
   resolved against the config file's folder; `[scene.<name>]` keys normalised with `scene_key`. Default
   `config.toml` may be missing; a `--config` file may not. Range checks that must also cover CLI values (shadow
   0-1.5, quality 1-100 int, threads >=0, edge_shrink >=0, max_output_size 0 or >=64) run in `main()` after the merge.
2. Car: `load_cars` + `select_car` -> `CarTable`. Backgrounds: `list_backgrounds` (stem, lower-case) ->
   `select_backgrounds` -> `read_image` + `load_scene` each (flat auto-shrink warning silenced for room backgrounds).
3. Output guard: `out_dir = <output_folder>/<images folder name>`; refuses if any target dir (each background,
   `cutouts`, `debug`, flat `<bg>/debug`) resolves to the images folder. Then `onnxruntime`/`rembg` import check.
4. Room mode: `RoomRun` (focal, on_fail, max_output, `min_texels` 1.0/0.0 from `keep_background_sharp`, cut policy,
   `frame_view_type/output/<images folder name>/results.csv` via `find_view_results`, scene overrides).
5. `CarSegmenter(model, edge_shrink, threads)`. If isolation: `precompute_raw` (raw alpha of every photo, uint8 when
   exact) -> `segmenter.release()` + `gc.collect()` -> lazy `from car_isolation import CarIsolator`.
6. Per photo: `read_image` -> `segmenter.cut_out` (or `raw_alpha` + `finish(..., trace)` under debug, or
   `finish(image, raws.pop(path), isolator, trace)` with isolation) -> cutout (`quality=101` = lossless WEBP) ->
   `debug_step(cutout_pictures, ...)` -> room: `measure_photo` (line camera; errors become `lines-error`, never
   fatal) + `prepare_photo` (`crop_to_car`, `parse_name`, `probe_car`, `estimate_yaw`) -> `waiting`.
   Flat: `prepare_photo`, then `place_photo` now (`photo`/`off` scale) or after the loop (`folder`: `folder_scales`).
7. Room stages 2-3 (`place_room_photos`, after `del segmenter, isolator`): `choose_focals` over all photos; strong-line
   photos first (their heights feed `folder_height_prior`); per photo skip check, re-read the photo from disk,
   `fit_measured(..., video=True)` (+ `fit_pictures`), then per background `place_in_room` (+ `room_pictures`), or
   queue for `place_flat_fallback` when `room_for` gives no room; those are placed after all fits.
8. `finally` (also on error/Ctrl+C): room mode adds a "not placed" row per waiting photo, sorts rows (`file_order`),
   writes `results.csv` with `ROOM_CSV_FIELDS` or `CSV_FIELDS`, plus `ISOLATION_CSV_FIELDS` when isolation is on.

Outputs: `<out>/<bg>/<photo>.webp`, `cutouts/<photo>.webp`, `results.csv`; debug: `debug/README.txt`,
`debug/<photo>/01..11`, `debug/<photo>/<bg>/12..16`; flat mode keeps its overlay at `<bg>/debug/<photo>.jpg`.

Tools (from `background_changer\`):
- `.venv\Scripts\python tools\check_flat.py output\_baseline_flat output\_flat_check`: every baseline image exactly
  (cut-outs: alpha, colour where alpha > 0), results.csv on the baseline's columns minus `segment_s`; extras ignored.
- `.venv\Scripts\python tools\check_room.py [folders] [--per-photo] [--frames 0012 0020] [--max-seconds 120]
  [--max-output-size 1920] [--min-texels 1.0] [--tag X]`: default folders `sample_vid` (Baleno), `Final_zoomed_car`
  (EQB); writes `output/check_room/{folder,per_photo}[_<tag>]/<images folder>/`; masks cached as 16-bit PNG in
  `output/check_room/_masks/<folder>/` (BiRefNet loads on a cache miss). Exit 1 if chain IoU < fit IoU - 0.02 or an
  opaque car pixel differs from `INTER_AREA(level)`. Uses built-in `SCENES` only, no config overrides.
- `.venv\Scripts\python tools\check_scenes.py [run_dir] [--car NAME] [--background-scale off] [--min-width]
  [--out output/check] [--sheets] [--debug] [--full]`: flat placement only; needs a run with `save_cutouts = true`
  (default `output/Final_zoomed_car`, car EQB, scale `off`); exit 1 on an acceptance failure.

Downstream: `video_pipeline/make_video.py` runs this script with `<images> --out --car --background
--background-view room` and reads `<out>/<images folder name>/<background>/<stem>.webp`.

## Invariants & gotchas
- **Car pixels**: at most one rigid rotation (room levelling) + one `INTER_AREA` downscale (never enlarged) + crop at
  a cut edge; car pasted last; damage stays visible. This file never edits `photo.image`; ask the user before
  anything else touches car pixels. Checked by tests and `check_room.py`.
- **Flat mode is bit-identical** to `output/_baseline_flat`. Re-check per folder with `--config
  tests\baseline_flat.toml <images folder> --car "<carModel>" --out output\_flat_check`, then `check_flat.py`.
  Without `--out` the run overwrites the baseline (that toml's `output_folder`). CLI paths are relative to the cwd,
  config paths to the config's folder. The flat overlay (`<bg>/debug/<photo>.jpg`) is part of the baseline and is
  written straight from `place_photo`, not through `debug_step`.
- **`car_isolation` off is bit-identical**: isolator `None` -> plain `clean_alpha`; package never imported; no
  isolation columns.
- Read images with `np.fromfile` + `cv2.imdecode`, write with `imencode` + `tofile`: `cv2.imread` fails on non-ASCII
  Windows paths.
- **Memory (16 GB laptop)**: one run was seen to commit ~17 GB. Never run two at once, nor pytest beside one. BiRefNet
  and SAM are never loaded together (two-pass); `precompute_raw` holds every raw mask of the folder in RAM. Room mode
  keeps per-photo summaries, not fits (they hold full-res car pixels), and empties `photo.image` when every background
  has a room (`keep_crops`). Stage 3 re-reads photos from disk: a photo moved mid-run is skipped.
- **Room-less background in room mode** (no builder in `car_fit_to_bg/rooms.py` and no `horizon_y` more than
  `GENERIC_BACKDROP_ROWS` (10 px) above the anchor row): flat placement with the fitted yaw (`YawEstimate(.., 10.0,
  "fit")`), cut policy forced to `anchor`, `background_scale` applies, flag `no-room`, `view_mode = "flat (no room)"`.
- **`[fit] on_fail`**: `place` = a failed pose is placed and flagged `fit-failed`; `skip` = the photo gets a "skipped:
  the 3D fit failed" row, and a background whose `placed.flags` has `camera-outside-room` is skipped. `FitError` always
  skips; other exceptions give `fit error` / `place error` rows and the run continues.
- **`cut_frames`**: room = `skip` or `frame` (`anchor`/`centre` accepted, treated as `frame` with a note); bottom cuts and
  cuts on 2+ sides are always skipped (`car_fit_to_bg.batch.skip_reason`). Flat = unset/`frame` -> `anchor`;
  `anchor`/`centre`/`skip` pass through to `bg_changer` placement.
- `results.csv` is written in `finally` with `csv.DictWriter` (default `extrasaction="raise"`): a row key missing from
  the field lists raises there and the CSV is lost. Flat `folder`-mode photos still waiting at Ctrl+C are not listed.
- `yaw-vs-label` is added here (`YAW_CHANGE_LISTED = 24` deg), not in car_fit_to_bg; `QUIET_FLAGS` (`flip-ambiguous`,
  `fit-slow`) stay in the CSV, off the console.
- Import guard: every top-level import sits in `try`; `ImportError` exits with `VENV_HINT` (started with the system
  Python instead of `.venv\Scripts\python`).
- `threads = 0` means ORT auto for BiRefNet but 4 for the isolation models (`threads or 4`).
- Tests pin shipped values: `test_config.py` (room, `cut_frames = "frame"`, 1920, keep sharp, `[fit]` auto/place,
  backgrounds 1-3 present, Baleno in cars.json) and `test_car_isolation.py` (isolation off). Changing `config.toml`
  defaults means updating them.
- `[scene.<name>]` matching no background only warns. The view prior is keyed by images folder *name*.

## How to extend
- **Config option**: add the key to `CONFIG_KEYS` (or `FIT_KEYS` / `ISOLATION_KEYS` / `SCENE_KEYS`; also `PATH_KEYS` for
  a path, the "must be text" tuple for a string, the `("shadow", "edge_shrink", "quality", "threads")` "must be a
  number" tuple for a number); validate type *and range* in `load_config` or `parse_*` (that is what
  `tests/test_config.py` can reach; nothing tests `main()`); for a CLI override add an argparse flag with
  `default=None`, merge `args.x if args.x is not None else cfg.get(...)` and repeat the range check in `main()` (CLI
  values bypass `load_config`); use it in `main()`. A new `Run` / `RoomRun` field goes **last with a default**:
  `tests/test_room_mode.py` builds both positionally. Then: commented block in `config.toml`; README settings table
  (and overrides table); `tests/test_config.py` (valid + bad values). Renaming a key: add the old name to `RENAMED`.
  Default must leave flat output bit-identical.
- **results.csv column**: add it to `CSV_FIELDS` (both modes), `ROOM_CSV_FIELDS` (room only) or `ISOLATION_CSV_FIELDS`
  first; then fill it in `place_photo` (flat), `room_row` / `base_row` / `place_flat_fallback.row` (room) or
  `isolation_cols`. New columns don't break `check_flat.py`; changed values in old columns do. Prefer
  `ROOM_CSV_FIELDS`: `test_room_mode.py::test_rows_fit_the_results_csv_columns` pins `CSV_FIELDS` to `FLAT_COLUMNS`
  (flat CSV unchanged) and writes room rows against `ROOM_CSV_FIELDS`. Update README's results.csv tables.
- **Debug picture**: `dw.save(stem, "NN_name", img, title, lines, bg=...)` in the matching `debug_images.py` function,
  numbered by run order (01-05 cut-out, 06-11 fit, `<bg>/12-16` per background). Update the module docstring (it is
  written verbatim as `debug/README.txt`) and README's table. A new function: add to `__all__`, import it in the
  guarded block, call it only through `debug_step(fn, ...)` (logs and continues; never stops a run). Pictures only read:
  get inner data through an optional dict defaulting to `None` (`clean_alpha(trace=)`, `line_camera(detail=)`,
  `placed.steps`) and test that passing it changes nothing (`test_debug_images.py`).
- **Car**: an entry in `cars.json` (cm, `body`); no code change.
- **Background**: drop the image in `backgrounds/`; `[scene.<name>] anchor/side_fill/horizon_y` gives a generic stage
  and generic room. A measured one: see [bg_changer/scenes/](bg_changer/scenes/CLAUDE.md) and
  [car_fit_to_bg/](car_fit_to_bg/CLAUDE.md) (`rooms.BUILDERS`).

## Tests
`.venv\Scripts\python -m pytest tests` from `background_changer\` (`python -m` puts this folder on `sys.path`; no model
needed). Covering this folder: `tests/test_config.py` (load_config, background selection, shipped config/cars),
`tests/test_room_mode.py` (`place_room_photos`, flat fallback, skips, `lines-error`, CSV fields, row order),
`tests/test_car_isolation.py` (`[car_isolation]` block, two-pass cache), `tests/test_debug_images.py` (`DebugWriter`,
`cutout_pictures`). Flat path: `tools/check_flat.py`. Details: [tests/CLAUDE.md](tests/CLAUDE.md).
