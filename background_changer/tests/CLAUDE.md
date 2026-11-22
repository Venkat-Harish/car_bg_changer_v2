# tests: model-free pytest suite for background_changer

## Purpose
Pytest suite for every background_changer package plus `change_background.py`'s config and room-mode orchestration. No model runs (no BiRefNet/rembg, no YOLOX/SAM download): synthetic images and drawn masks stand in for photos and model output.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `conftest.py` | Shared helpers/fixtures | `side_car_alpha()` (3840x2160 side view: body box + two round tyres, blurred); fixtures `table` (session, default `CarTable()` = EQB 2024), `side_car` (noise image + `clean_alpha` mask), `grey_plate` (1920x1080, grey 180) |
| `baseline_flat.toml` | NOT a test: pinned config of the flat-mode baseline in `output\_baseline_flat`, compared by `tools/check_flat.py` | `background_view="flat"`, background_1-3, `cut_frames="anchor"`, `debug_mode`/`save_cutouts=true`, `quality=90`; paths relative to `tests/` |
| `test_background_scaler.py` | [background_scaler](../background_scaler/CLAUDE.md) | `background_size`, `folder_scale`, `needed_scale`, `scale_scene`, `BackgroundScaler`, `crop_to_car`; local `small_photo()` (960x540) |
| `test_car_fit.py` | [car_fit_to_bg](../car_fit_to_bg/CLAUDE.md): pose fit, levelling, room placement, line camera, focal pooling | `fit_photo`, `place_in_room`, `level_car`/`downscale`, `estimate_camera`, `pool_focals`/`choose_focal`, `stage_scale`, `compose.paste`; `_scene()` (proxy silhouette from a known camera), module fixture `fitted`, `_two_tone_car()` |
| `test_car_isolation.py` | [car_isolation](../car_isolation/CLAUDE.md) + its hooks in `segment.py` + `[car_isolation]` config | `CarIsolator`, `plan_removal`, `maybe_part`, `detect.letterbox/decode/nms/vehicles_from_raw`, `fill_holes`, `CarSegmenter.cut_out/finish`; `FakeDetector`, `FakeSam`, `isolator()`, `rect()`, `soft()` |
| `test_carmodel.py` | [bg_changer](../bg_changer/CLAUDE.md) `carmodel` | `CarTable.at/floor_homography/floor_row/side_length`, `CarShape`; reference scale/anchor `P, AX, AY = 231, 957.5, 764` |
| `test_cars.py` | `bg_changer.cars` + shipped `cars.json` | `parse_cars` (cm -> m, unit sanity), `select_car`, `load_cars`, `CarSpecError` |
| `test_compose.py` | `segment.clean_alpha` (main car only, posts, windows) + flat `compose.place_car` | `fake_photo()` (2000x1000); flat bit-exact test |
| `test_config.py` | `change_background` config + background selection; shipped `config.toml`/`cars.json` | `cb.load_config`, `list_backgrounds`, `select_backgrounds`; `make_backgrounds()`, `write_config()` |
| `test_debug_images.py` | [debug_images.py](../CLAUDE.md) + the `trace=`/`detail=` hooks feeding it | `DebugWriter`, `cutout_pictures`, `clean_alpha(trace=)`, `lines.line_camera(detail=)` |
| `test_grounding.py` | `bg_changer.grounding` (shadows, contacts, reflection) | `shadow_multiplier`, `PinnedFloorMap`, `snap_contacts`, `outline_floor`, `_silhouette_layers`, `add_reflection`; `_setup()`, fixture `setup` |
| `test_placement.py` | `bg_changer.placement` | `Scene.plan`, `plan_on_anchor`, `XMap`, cut policies; fixture `scene1` |
| `test_probe.py` | `bg_changer.probe` + `yaw` | `parse_name`, `probe_car`, `estimate_yaw` |
| `test_room3d.py` | [room3d](../room3d/CLAUDE.md) + `car_fit_to_bg.rooms` | `render`, `Room`, `kmat`, `look_at`, `build_room`, `BUILDERS`; `_analytic()` ray-casts a textured box |
| `test_room_mode.py` | `change_background` room mode, stages 2-3, + `car_fit_to_bg.batch.fit_measured` | `place_room_photos`, `place_flat_fallback`, `measure_photo`, `no_room_hint`, `file_order`, `CSV_FIELDS`/`ROOM_CSV_FIELDS`; `synthetic_photo()`, `make_photo()`, `make_run()`, `room_settings()`, module fixture `room_run` |
| `test_scenes.py` | [bg_changer/scenes](../bg_changer/scenes/CLAUDE.md) | `load_scene`, `SCENES`, `[scene]` overrides, generic scene, shrink-to-fit warnings |

## How it works
- **Synthetic photos.** Images are uniform noise (`rng.integers`, seeded) so every pixel is distinct and a bit-exact comparison means something. Masks are drawn (rects, circles, ellipses, `GaussianBlur`) and go through the real `clean_alpha`, as rembg's raw alpha would.
- **No segmenter.** `CarSegmenter.__init__` imports rembg/onnxruntime; tests skip it with `seg = CarSegmenter.__new__(CarSegmenter)`, set `seg.edge_shrink, seg.last_isolation`, and inject `seg.raw_alpha = lambda image: raw`.
- **3D ground truth.** `pose.car_to_camera` + `body_points` + `render_proxy` render a `CarShape` proxy at a known yaw/distance/height/focal; that silhouette is the car's alpha, and `LineCamera` comes from `cam_angles(R)`. `fit_photo` must recover the pose.
- **Room mode without BiRefNet** (`test_room_mode.py`): `make_photo` writes the noise frame as PNG (re-read exactly in stage 3), builds a `CarMask` straight from the synthetic alpha, then calls the real `cb.prepare_photo` and `cb.measure_photo` (stage 1 line camera) or a blank `cb.Measured` (`measure=False`). `make_run` loads the three real `backgrounds/*.webp` plus two synthetic user backgrounds: `my_hall` (`horizon_y = 330` -> generic room) and `my_lot` (no horizon_y -> no room -> flat fallback with the fitted yaw). `room_settings` = `cb.RoomRun(SHAPE, F=600, on_fail, MAX_OUT=640, 1.0, cut_policy, None, overrides)`, so focal comes from "config". Then `cb.place_room_photos([photo], run, rr, rows)` runs stages 2 (lens pooling) and 3 (fit + place + write).
- **Spies/monkeypatches go on the name where it is looked up:** `cb.place_in_room`, `cb.fit_measured`, `cb.measure` (imported by name into `change_background`), and `batch.line_camera`, `batch.fit_photo` (in `car_fit_to_bg.batch`). Patching `car_fit_to_bg.place_in_room` would do nothing.
- **Fake detector/SAM.** `CarIsolator(0.3, remove_objects, detector=FakeDetector(dets), segmenter=FakeSam(masks))`: passing both skips the model import. `FakeSam` is keyed by full-image float box tuples; it maps the crop-local box back by offset and returns the pre-made full-image mask cropped like the real `BoxSegmenter`. YOLOX post-processing is tested on hand-built `(8400, 85)` raw arrays.
- **Real assets read:** `backgrounds/background_{1,2,3}.webp` (test_car_fit, test_room3d, test_room_mode), `cars.json`, `config.toml`. Everything written goes to `tmp_path`.
- **Expensive work is module-scoped:** `fitted` (test_car_fit) and `room_run` (test_room_mode) fit once per module.

## Invariants & gotchas
- **Bit-exact car pixels** (`np.array_equal` over opaque pixels, alpha >= 1.0 or > 1 - 1e-4), with shadow/reflection on. They enforce the project rule: one rigid roll rotation + one INTER_AREA downscale + crop at a cut edge only. If one fails, fix the code; never loosen the test.
  - Flat (`test_compose::test_car_pixels_are_only_downscaled_on_every_scene`, every scene, `shadow=1.5`): output == `cv2.resize(image[crop], plan.size, INTER_AREA)`.
  - Scaler (`test_a_small_photo_gets_a_smaller_background...`): `plan.scale == 1.0`, pasted pixels == the photo's own.
  - Room (`test_car_fit` room-path/rolled/cut-edge tests, `test_room_mode::test_written_images_carry_the_photos_own_pixels`): output == `cv2.resize(level(source).pixels, (w, h), INTER_AREA)`, recomputed in the test; `downscale(car, s_out)` only supplies alpha and corner. Unrolled photo: `fit.car.pixels == frame[crop]`; level angle < 0.5 deg -> `theta == 0.0` exactly (plain crop).
  - Isolation off: `clean_alpha(raw, 1) == clean_alpha(raw, 1, None, None)`; `cut_out` without isolator == `clean_alpha(raw, 1)` with `isolation is None`. The debug `trace=`/`detail=` hooks also change nothing.
- Downscale must really run: `max_output_size=640` < the 960 px photo, and tests assert `s_out < 1` / `plan.scale < 1`. `quality=101` in `make_run` = lossless WEBP, so written files equal `placed.image`.
- `FLAT_COLUMNS` pins `cb.CSV_FIELDS` exactly (flat results.csv must not change); `ROOM_CSV_FIELDS` must start with it; `csv.DictWriter` raises on any room-row key not in `ROOM_CSV_FIELDS`.
- `baseline_flat.toml` is not collected by pytest. Workflow (from its comments): run `change_background.py --config tests\baseline_flat.toml <images> --car "<model>" --out output\_flat_check`, then `tools\check_flat.py output\_baseline_flat output\_flat_check` (exit 1 on any diff). `output/` is gitignored, so the baseline exists only on this machine. The toml must stay valid under `load_config` (renamed keys raise "replaced").
- Shipped-file tests pin defaults: `config.toml` must keep `background_view="room"`, `cut_frames="frame"`, `max_output_size == DEFAULT_MAX_OUTPUT` (1920), `keep_background_sharp=true`, `[fit]` exactly `{focal="auto", on_fail="place"}`, and an explicit `[car_isolation] enabled = false` (`.get("enabled") is False`). `cars.json` must hold EQB 2024 and Baleno 2022 (3.99/1.745/1.50/2.52 m). Changing a default means updating these tests on purpose.
- Tests parametrized over `sorted(SCENES)` (test_scenes, test_compose, test_background_scaler, test_car_fit) pick up a new scene automatically. test_car_fit also needs `backgrounds/<name>.webp` and a `BUILDERS` room, or `place_in_room` returns `image=None`. test_scenes wants every view >= 150 px from the edges with the EQB table.
- test_room3d compares only background_1 (exact) and background_2 (1 grey level; its R is not quite orthonormal). background_3 is skipped because its plate is de-vignetted (`BG3_DEVIGNETTE` in rooms.py).
- `test_fit_and_render_are_deterministic`: fit and render must be bitwise reproducible, so no unseeded randomness.
- Imports: there is no pytest.ini/pyproject/sys.path tweak. `from conftest import side_car_alpha` works because pytest's default prepend mode puts `tests/` (no `__init__.py`) on sys.path. The packages and `change_background` import only because `python -m pytest` puts the cwd on sys.path, so run from `background_changer/`. `change_background.py` turns an import-time ImportError into `SystemExit` with a venv hint, so under the wrong Python a test module errors at collection with that text.
- **Memory:** never run pytest while a `change_background` run is going (~17 GB each; MemoryError on the 16 GB laptop).

## How to extend
- New test: `tests/test_<module>.py`; import packages top-level (`from bg_changer import ...`) and helpers via `from conftest import side_car_alpha`. Reuse `side_car`, `grey_plate`, `table`. Use noise images. Any code that touches car pixels gets a bit-exact assertion in the style above.
- New config key: add it to `CONFIG_KEYS` and validate it in `load_config` (or `FIT_KEYS`/`parse_fit`, `ISOLATION_KEYS`/`parse_isolation`, `SCENE_KEYS`/`parse_scene_overrides`). In test_config.py, read a good value back (like `test_room_mode_settings`) and add bad values to `test_bad_values_are_rejected` / `test_bad_room_values_are_rejected`. A renamed key goes in `RENAMED` + `test_old_keys_say_what_replaced_them`; a `[car_isolation]` key goes in `test_config_isolation_block`. If it ships in config.toml, extend `test_shipped_config_*`.
- New background with its own scene/room: register it in `SCENES`, add its webp to `backgrounds/`, add a `BUILDERS` room; the parametrized tests then cover it. Also add it to `MEASURED` in test_room_mode, to test_room3d's re-render parametrize (with a tolerance) and to `test_anchor_override_re_pins_the_calibrated_floor`. A generic background is tested through `load_scene` overrides (test_scenes) or like `my_hall`/`my_lot` in `make_run`.
- New results.csv column: room-only -> append it to `ROOM_CSV_FIELDS` (never to `CSV_FIELDS`), fill it in `room_row`/`place_flat_fallback`, assert its value in `test_every_background_gets_the_photo`. Isolation-only -> `ISOLATION_CSV_FIELDS`. A new flat column breaks `FLAT_COLUMNS` and the flat baseline by design.
- Room-mode orchestration: reuse `synthetic_photo`/`make_photo`/`make_run`/`room_settings`, and monkeypatch `cb.<name>`. `make_run` and `room_settings` build `cb.Run` / `cb.RoomRun` positionally, so a new field in either goes last with a default.
- Range checks that live only in `change_background.main()` (shadow, quality, threads, ...) are untested: test_config only calls `load_config`. Put a new key's range check in `load_config` too so it can be tested here.
- car_isolation: build `Detection` lists for `FakeDetector` and a `{box: mask}` dict for `FakeSam` with the same float box tuples.

## Tests
From `F:\PROJECTS\car_demo_img\background_changer` (PowerShell):
- All: `.venv\Scripts\python -m pytest tests -q`
- One file: `.venv\Scripts\python -m pytest tests\test_room_mode.py -q`
- One test: `.venv\Scripts\python -m pytest "tests\test_compose.py::test_car_pixels_are_only_downscaled_on_every_scene[background_1]"` (or `-k <substring>`)
- Measured 2026-11-22: **158 passed, 1 warning in 222 s** (~3.7 min). Peak RSS seen was about 0.9 GB. The one warning is expected: `test_anchor_override_re_pins_the_calibrated_floor` triggers the "EQB 2024 made 17% smaller" shrink-to-fit UserWarning on background_1.
- `.venv\Scripts\pytest.exe tests` (plain console script) fails at conftest with `ModuleNotFoundError: No module named 'bg_changer'`. Always use `python -m pytest` from `background_changer/`.
