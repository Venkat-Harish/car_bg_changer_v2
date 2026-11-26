# car_demo_img

Turns a walk-around video of a car, or a folder of car photos, into studio shots of that car. Four independent,
CPU-only Python CLI tools chain together; each has its own folder and `config.toml`, and all but video_pipeline
have their own `.venv` and README. No root README: the engineer index is [docs/README.md](docs/README.md); user-facing docs are
[background_changer](background_changer/README.md), [number_plate_detection](number_plate_detection/README.md) and
[frame_view_type](frame_view_type/README.md). Windows 11 + PowerShell.

## Stack and commands
- Python 3.11 (every tool uses `tomllib`), one venv per tool: `<tool>\.venv`. video_pipeline has none: it runs on
  background_changer's venv and launches each tool with that tool's own venv.
- CPU only: ONNX Runtime with `CPUExecutionProvider` everywhere; frame_view_type's torch loads with
  `map_location="cpu"`. Models download on first use (rembg → `%USERPROFILE%\.rembg\models\`, open-image-models →
  `%USERPROFILE%\.cache\open-image-models\`); the view classifier comes from a gitignored git-LFS clone in
  `frame_view_type/vendor/view_classifier`.
- Pins: background_changer — rembg[cpu] 2.0.86, onnxruntime 1.31.1, opencv-python-headless 5.0.0.93, numpy 2.4.6,
  pillow 13.0.0, scipy 1.17.1, pytest 9.2.0. number_plate_detection — open-image-models[onnx] 0.6.0.
  frame_view_type — torch 2.9.1, torchvision 0.24.1. Others unpinned.

Run each tool from its own folder; settings live in its `config.toml`, CLI flags override them for one run,
`--config <file>` swaps the file:

```
frame_view_type>         .venv\Scripts\python classify_video.py
number_plate_detection>  .venv\Scripts\python mask_plates.py
background_changer>      .venv\Scripts\python change_background.py
video_pipeline>          ..\background_changer\.venv\Scripts\python make_video.py
video_pipeline>          ..\background_changer\.venv\Scripts\python e2e_steps.py   (test run, step report)
```

Tests and checks:
- background_changer (pytest, no model needed, ~160 tests, ~4 min): `.venv\Scripts\python -m pytest tests`, from
  `background_changer\`. Must be `python -m` from that folder (no pytest config; `pytest.exe` can't import packages).
- Flat-mode baseline, pixel for pixel (needs BiRefNet): per baseline folder run `.venv\Scripts\python change_background.py
  --config tests\baseline_flat.toml <images folder> --car "<carModel>" --out output\_flat_check`, then
  `.venv\Scripts\python tools\check_flat.py output\_baseline_flat output\_flat_check`. Without `--out` it overwrites the baseline.
- Room-mode review sheets and self-checks: `.venv\Scripts\python tools\check_room.py` (masks cached after first run).
- number_plate_detection (scripts, not pytest; need the detector): `.venv\Scripts\python tests\run_colour_tests.py`
  on the tracked `assets/test_photos`; exits 0 even on FAIL, so read `output/colour_tests/summary.csv`.
  `tests\make_test_photos.py` only regenerates that set (rewrites tracked files; needs frame_view_type's frames).
- frame_view_type, video_pipeline: no tests. Lint: none configured (no ruff/flake8/mypy config in the repo).

## Architecture
```
car video ─► frame_view_type ─► output/<video>/<t>s_<side>_<conf>pct.jpg + results.csv
                                       │                         ┊ room-mode yaw prior (same folder name)
photo folder ──────────────────────────┴─► number_plate_detection ─► same file names, plates covered
                                                                   │   + detections.csv
                                                                   ▼
  background_changer/change_background.py  (room mode = default)
    stage 1  every photo: cut out (BiRefNet via rembg) [+ car_isolation pass] + measure background lines
    stage 2  pool the focal length over the folder
    stage 3  per photo: fit camera + 3D car pose once ─► per background: re-render it as a 3D room
             from that camera (room3d), draw grounding, level + downscale the car, paste it last
    flat mode instead: scene-code placement (bg_changer/scenes) + background_scaler
                                                                   ▼
  output/<images folder>/<background>/<photo>.webp, cutouts/, debug/, results.csv

video_pipeline/make_video.py: video ─► frames 00001.jpg ─► mask_plates.py ─► change_background.py
  --background-view room ─► mp4  (does not run frame_view_type, so its frames carry no view label)
video_pipeline/e2e_steps.py: video ─► classify_video --pick-sides (one frame per side, <video>_sides) ─►
  mask_plates ─► change_background --debug (all backgrounds) ─► output/e2e_<video>/index.html, every step pictured
```

## Code map
- [background_changer/](background_changer/CLAUDE.md) — entry point `change_background.py`, config, debug pictures, `tools/` checks
  - [bg_changer/](background_changer/bg_changer/CLAUDE.md) — cut-out (rembg), cars.json, file-name probe, flat placement, compositing, grounding
  - [bg_changer/scenes/](background_changer/bg_changer/scenes/CLAUDE.md) — per-background scene specs measured from each picture; generic stage
  - [car_fit_to_bg/](background_changer/car_fit_to_bg/CLAUDE.md) — camera from background lines, 3D pose fit, levelling, room placement
  - [room3d/](background_changer/room3d/CLAUDE.md) — 3D room model and NumPy/OpenCV re-render from a new camera
  - [car_isolation/](background_changer/car_isolation/CLAUDE.md) — optional YOLOX-s + SAM pass that keeps only the target car in the mask
  - [hosted/](background_changer/hosted/CLAUDE.md) — `[hosted]` (on by default): fal.ai BiRefNet + SAM 3 cut-out, glass tint, MoGe-3 + VGGT camera; needs `FAL_KEY`. Privacy (DPDP): uploads only plate-masked folders (`detections.csv`), people pixelated in the uploaded copy, cache pruned after `keep_cache_days`
  - [background_scaler/](background_changer/background_scaler/CLAUDE.md) — shrinks background + scene for small photos (flat placement)
  - [tests/](background_changer/tests/CLAUDE.md) — model-free pytest suite, bit-exact car-pixel checks
- [number_plate_detection/](number_plate_detection/CLAUDE.md) — entry point `mask_plates.py`, config, assets
  - [plate_masker/](number_plate_detection/plate_masker/CLAUDE.md) — YOLOv9 ONNX detector, 4-corner plate fit, mask styles
  - [plate_brightness_detector/](number_plate_detection/plate_brightness_detector/CLAUDE.md) — overlay brightness from the plate's colours
  - [tests/](number_plate_detection/tests/CLAUDE.md) — synthetic colour test set and its runner
- [frame_view_type/](frame_view_type/CLAUDE.md) — `classify_video.py`: one frame every N s, 8-way side classifier
- [video_pipeline/](video_pipeline/CLAUDE.md) — `make_video.py`: video → masked → new background → mp4
- `docs/` — the current flow step by step (`pipeline-steps.md`: what each step does, solves, takes and gives, with
  pictures from the 24 Nov 2026 e2e run in `docs/images/`), `glossary.md`, and `ceo/` (the CEO PDF and its HTML
  source). Pictures must show covered plates only. Each tool also keeps research notes in its own `docs/`.

## Rules that always hold
1. **Car pixels are evidence.** The photos document vehicle condition, so dents and scratches must stay as shot.
   background_changer may only rotate the car once rigidly (room-mode levelling, `car_fit_to_bg/level.py`),
   downscale it once with `INTER_AREA` (never enlarge) and crop it at a cut edge. No relighting, colour matching,
   sharpening or generated content; shadows, reflections and rooms go on the background and the car is pasted
   last. Bit-exact tests enforce this. Ask the user before any change that would touch car pixels otherwise.
   **One approved exception (Nov 2026):** with `[hosted] glass_tint` on, every window (windscreen included) gets a
   fixed dark tint, only inside SAM 3's window mask; the untouched photo is copied to `<out>/originals/`, and the
   tinted frame then counts as the photo for every rule above ([hosted/](background_changer/hosted/CLAUDE.md)).
   video_pipeline may then resize a whole finished frame once (`fit_to_canvas`).
2. **Bit-identical paths.** Flat mode must reproduce the pinned baseline (`output/_baseline_flat`,
   `tests/baseline_flat.toml`, `tools/check_flat.py`). `[car_isolation] enabled = false` must give exactly the
   pre-isolation cut-out.
3. **Config-driven.** Every option is a commented `config.toml` key, optionally with a CLI override; no GUIs or
   launchers. background_changer rejects unknown keys, so a new key needs its allow-list entry and validation.
4. **File-name contract.** frame_view_type writes `<time>s_<side>_<conf>pct.jpg` (8 sides 45° apart, the car's own
   left/right); number_plate_detection keeps names; `bg_changer/probe.py` parses them. In room mode
   background_changer also reads `frame_view_type/output/<images folder name>/results.csv`. Renamed files lose
   the view prior silently (flag `no-view-label`).
5. **Units.** `cars.json` is a JSON array in cm (`carModel`, `length`, `width`, `height`, `WheelBase`, optional
   `body` suv|hatch|sedan); `bg_changer/cars.py` converts to metres.
6. **Images** are BGR uint8 NumPy arrays. Read with `np.fromfile` + `cv2.imdecode` (`cv2.imread` fails on
   non-ASCII Windows paths; video_pipeline and some tests still use `imread`).
7. **Outputs.** Tools refuse to write into their input folder. Results go to each tool's `output/` (gitignored, as
   are `*.pt`, `*.mp4`, `.venv/` and `frame_view_type/vendor/`).
8. **Memory (16 GB laptop).** One change_background run was observed to commit ~17 GB: never run two at once, nor
   pytest beside one. BiRefNet and SAM are never loaded together (isolation runs as a second pass).
9. **Flag, don't drop.** Unsure or failed fits are placed and flagged in `results.csv` (`[fit] on_fail = "place"`).

## Keeping docs current
- When you add, rename or delete a file, or change behaviour in a folder, update that folder's CLAUDE.md in the same change.
- A new folder with real logic gets its own CLAUDE.md (same headings) and a line in the code map.
- Edit this root file only when the architecture, commands or cross-cutting rules change. Keep it short.
- User-visible changes also update the README.
- Docs explain the why, the connections and the gotchas. Don't restate code line by line. Delete anything that's no longer true.
