# car_fit_to_bg: fit each car photo in 3D, then re-render each background's room from that photo's camera

## Purpose
Room mode, the default (`background_view = "room"`). Per photo, independent of the background: the camera (roll, pitch, focal), the car's pose, the level angle, the tyre contacts. Per background: put that camera into the background's [room3d](../room3d/CLAUDE.md) room, render it, shade the floor, and paste the car last.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Per-photo, background-independent fit: pose, then level, then tyres, then contact lock | `fit_photo(...) -> PhotoFit`, `PhotoFit`, `FitError`, `fit_overlay`, `LOCK_IOU = 0.90` (the one in use) |
| `batch.py` | The three-stage flow over a folder (used by `change_background.py` and `tools/check_room.py`) | `measure`, `Measured`, `choose_focals`, `fit_measured`, `folder_height_prior`, `find_view_results`, `view_hints`, `cut_sides`, `skip_reason` |
| `lines.py` | The photo's camera from its own background lines (LSD, then vanishing points, then soft_l1 refit, then bootstrap), plus the focal choice for a run | `line_camera`, `estimate_camera`, `LineCamera` (`.strong` = roll known and conf >= 0.5), `pool_focals`, `choose_focal`, `exif_focal_px`, `prior_focal`, `video_time`, `neighbour_roll_pitch` |
| `pose.py` | Car pose from the silhouette: a convex-piece proxy rendered through a pinhole and IoU-fitted (Nelder-Mead) | `fit_pose -> PoseFit`, `refit_with_roll`, `body_points`, `render_proxy`, `car_to_camera`, `cam_angles`, `red_cue`, `read_view_results`, `probs_from_label`, `PIECES`/`TYRE_PIECE` |
| `level.py` | The only rotation allowed on the car's pixels, and its one INTER_AREA downscale | `level_car -> LevelledCar`, `downscale`, `roll_of`, `limit_for_cut`, `cut_crop`, `to_levelled`, `rz`, `MIN_DEG = 0.5`, `MAX_CUT_CROP = 0.02`, `CAR_ROLL_RANGE = (1, 8)` |
| `ground.py` | Tyre bottoms on the levelled outline, the contact lock, shadow layers and glossy reflection on the output floor | `find_tyre_bottoms -> [TyreBottom]`, `contact_lock`, `floor_points_of`, `shadow_layers`, `shade`, `pin_offsets_from`, `room_reflection`, `project_rc` |
| `place.py` | Per background: room camera, output scale and window, render, shadows, reflection, paste | `place_in_room(...) -> Placement`, `stage_scale`, `room_camera`, `output_window` |
| `rooms.py` | The calibrated backgrounds as `room3d.Room`s, a generic room built from `horizon_y`, and the room shadow styles | `room_for`, `build_room`, `BUILDERS`, `generic_room`, `GENERIC_BACKDROP_ROWS = 10`, `ROOM_SHADOW`, `DEFAULT_ROOM_SHADOW` |

## How it works
**Stages, as driven by `change_background.place_room_photos`** (see [../CLAUDE.md](../CLAUDE.md)):
1. **Measure** (per photo, while cutting out): `batch.measure(path, frame, alpha)` runs `line_camera` at a free focal on the full frame, with the car masked, at most 1280 px on the long side. It also reads the EXIF 35 mm focal and a video time from the stem prefix `<t>s_`. `change_background.measure_photo` wraps it; an exception there gives the photo the flag `lines-error`.
2. **Choose focals** (per run): `choose_focals(items, config_focal)` picks, in order:
   1. `[fit] focal`;
   2. the pooled focal per image size: at least 3 photos with `conf_f >= 0.5` in a ±8% cluster, taking their weighted median. A photo whose own confident focal is more than 20% off keeps it and gets `lens-change`;
   3. the photo's own lines (`conf_f >= 0.5`);
   4. EXIF;
   5. `lines-weak` (`conf_f >= 0.15`, flag `focal-weak`);
   6. a 70° field of view across the long side (`focal-prior`).
3. **Fit** (per photo, once): `fit_measured` does the following:
   - re-solves `line_camera` with f fixed at the chosen focal;
   - if `roll is None or conf < 0.3`, and the stem is a video still, borrows roll and pitch from neighbours (`conf * exp(-(dt/4 s)^2)`, within ±8 s, neighbours with conf >= 0.3). Flag `roll-from-neighbours`;
   - gets the view hints from `view_hints`;
   - calls `fit_photo`.

   Strong-line photos go first, so `folder_height_prior` has data for the weak ones: N(median h, 0.25 m) over at least 3 fits that are strong, not failed and have IoU >= 0.9.
4. **Place** (per background): `room_for(name, spec, plate.shape)` gives `(Room | None, generic)`. `None` means flat fallback in `change_background` (flag `no-room`). Then `place_in_room(fit, name, plate, spec, shadow_gain, max_output_size, debug, min_floor_texels, room=, generic=)` (change_background passes `scene_key(stem)`, `run.shadow`, `rr.min_texels`).

**Yaw prior** (`view_hints`): uses the classifier row from `frame_view_type/output/<images folder name>/results.csv` (`find_view_results` resolves it from the repo root, `parents[2]` of batch.py) when its probabilities are non-empty. Otherwise it uses the file-name label (`probs_from_label`: label gets p = conf clamped to [0.3, 0.95], each other bin gets (1-p)/7). Otherwise None, and `fit_measured` flags `no-view-label`. Pooling and neighbours are only measurement aids: `fit_photo` works on one photo alone (tests call it directly).

**`fit_photo`**:
1. `FitError` if the car has fewer than 400 solid pixels or covers more than 90% of the frame.
2. Body preset: `shape.body`, else `guess_body` (flag `body-guessed`).
3. `red_cue` (Lab a* > 25).
4. `pose.fit_pose`.
5. Level angle:
   - Strong lines: `theta_full = roll_of(R)`. For an uncut, non-failed view within 15° of an end or side, compare the levelled mask's bottom-thirds tilt with the proxy's. If the difference tau is in [1°, 8°], `refit_with_roll` and flag `car-on-slope`, as long as IoU drops at most 0.02; otherwise `possible-slope`.
   - Weak lines: `theta_full = 0` and flag `roll-from-silhouette`. The room carries the silhouette roll instead of the pixels.
6. `limit_for_cut` keeps a slanted cut edge's loss to <= 2% of the visible car width (flag `partial-level`). An angle under 0.5° is set to 0.
7. `R_lev = rz(theta) @ R`.
8. `level_car`, `cut_crop`, `find_tyre_bottoms`.
9. `contact_lock` only if IoU >= 0.90 and the fit did not fail.
10. `floor_points_of`, which gives `patch_xy`.

**`fit_pose`**:
- **Rasters:** car long side 320 px for the sweep, 480 px for refinement.
- **Strong-line priors:** pitch σ 3°, roll σ 3°, h in [0.5, 3.0].
- **Weak-line priors:** h prior is `h_prior` or (1.5, 0.25), h in [1.0, 2.2].
- **Steps:**
  1. Sweep 36 yaws at h 1.5 (roll fixed to the line roll when the lines are strong).
  2. Up to 5 candidates: top IoU modes, the best score, and their 180° partners.
  3. Refine each.
  4. Score each: `e/TAU - log prior + red penalty` (TAU 0.02 IoU per nat).
  5. Take the margin to the best flipped candidate (±45°).
  6. Polish the winner, try the alternative body preset, run the label rescue, refit around residual blobs.
- **Conventions:** car frame X forward, Y left, Z up, origin at the footprint centre. Yaw 0 means the front faces the camera, 90 the right side. Pitch > 0 looks down; roll > 0 means the horizon rises to the right. The principal point is always the frame centre `((W-1)/2, (H-1)/2)`.

**`place_in_room`**:
1. `S = room.floor_point(spec.anchor)`.
2. `sigma = stage_scale(...)`: room units per car metre, so that a side view spans `side_px * size` at the stage.
3. `room_camera` turns the locked camera about the vertical, so it sits on the plate camera's side of the stage.
4. If `room.contains` fails, `clamp_inside(margin 0.3, toward=S)` and flag `camera-outside-room`.
5. Output scale `s = min(1, max_output_size / long side, t1 / min_floor_texels)`, where `t1` = plate texels per px at the stage at s = 1 (`room3d.texels_per_px`). `soft-floor` if `t1/s < 1`.
6. `output_window`: the photo frame on the levelled grid. A cut border sits on the innermost point of the slanted cut edge; an uncut car grows the window and is never cropped.
7. `room3d.render`.
8. `level.downscale(fit.car, s)`.
9. Floor pixels mapped to car-floor metres; then `shadow_layers` and `shade` in linear light, on floor pixels only.
10. If `spec.reflection`: `room_reflection`. On background_1 the gloss is limited to the disc, `DISC_RADIUS = 3.0` room units.
11. `linear_to_srgb`, then `bg_changer.compose.paste` of the car, **last**.
12. Metrics and flags: `invented` if more than 35% of non-car pixels are holes, or more than 1% of the car box padded by 25%; `contact-mismatch` if lock residual × s > 1% of the output height.
13. With `debug`: `Placement.steps` (`room`, `surfaces`, `shadows`, `background_ready`) and `.debug`, consumed by `debug_images.room_pictures`.

**Callers outside the folder:**
- `change_background.py`: `place_in_room`, `FitError`, `batch.*`, `lines.LineCamera`, `rooms.room_for` and `GENERIC_BACKDROP_ROWS`.
- `debug_images.py`: `line_camera(detail=)`, `pose.body_points`, `car_to_camera`, `render_proxy`, `level.bottom_profile`, `ground.project_rc`, and many `PhotoFit` fields.
- `tools/check_room.py`: the same three stages over test folders, plus self-checks.
- Inputs from [bg_changer](../bg_changer/CLAUDE.md): `carmodel.CarShape`/`CarTable`, `grounding` (footprint, outline and contact layers, reflection), `compose.paste`, `probe` (`parse_name`, `LABELS`), `yaw.BIN_DEG`, `segment.fill_holes`, and the scene specs and plate cameras from [bg_changer/scenes](../bg_changer/scenes/CLAUDE.md).

## Invariants & gotchas
- **Hosted camera hook** (`[hosted]`, see [../hosted/CLAUDE.md](../hosted/CLAUDE.md)): `Measured.external` (a
  `hosted.camera.HostedCamera`, default None) is turned into a strong `LineCamera(source="hosted")` by
  `batch._with_external`, unless the photo's own strong lines disagree by > 5 deg (lines kept, flag
  `camera-lines-fallback`). Its height reaches `fit_pose` as `strong_h_prior=(h, 0.15)`, a separate argument so
  that the folder `h_prior` keeps being ignored on strong photos exactly as before. With `external` None nothing changes.
- **Car pixels (hard rule).** The car gets exactly one rigid rotation (`level_car`), one INTER_AREA downscale (`downscale`, never up: `s <= 1` in place, and `s >= 1` returns the pixels unchanged) and a crop at a cut edge.
  - Colour: `Lanczos4(a*c)/Lanczos4(a)`, set to 0 where the denominator is under 1/255, clamped to the 3×3 min/max of the source car pixels.
  - Alpha: bilinear, with values above 1-1e-4 set to 1.
  - Below 0.5° the canvas is a plain crop and bit-identical.
  - Every effect (shadow, reflection, room) is drawn on `lin` **before** `paste`. Never add anything after the paste, and never re-colour the car.
  - Tests check opaque pixels == `INTER_AREA(level(source))` bit-exactly. See [../tests/CLAUDE.md](../tests/CLAUDE.md).
- **The applied angle is the assumed angle.** `theta` (0 if under `MIN_DEG`) goes into both `LevelledCar.theta` and `R_lev`. Anything that rotates the car must also update `R_lev`, or the proxy, room and car disagree. `check_room` fails a photo when its chain IoU is more than 0.02 below the fit IoU.
- **Grid conventions:** photo px `p` goes to `c + Rot(theta)(p - c)` on the full-resolution levelled grid. Scaling uses pixel centres: `(p + 0.5) s - 0.5`. `car.origin` is on the levelled grid. Changing either breaks the pixel and chain checks.
- **The fit is background-independent and reused.** `place_in_room` must not mutate `fit`. One `PhotoFit` is placed on every background.
- **Memory.** `PhotoFit` holds the full-resolution `LevelledCar` and `pose.work` rasters. `change_background` keeps only `fit_summary` plus a `SimpleNamespace(pose=(h, failed, iou), cam=(strong))` per photo, and calls `del fit`. If `folder_height_prior` starts reading more fields, update that SimpleNamespace in `place_room_photos`. `tools/check_room.py` keeps whole fits (review tool only). Never run two `change_background` runs at once (~17 GB each).
- **Stage-3 inputs.** The frame is re-read from disk, and the full-frame alpha is rebuilt from the kept crop (`change_background.full_frame_alpha`). The fit needs the full frame: the principal point is the frame centre, and cut sides refer to the frame borders.
- **`spec` passed to `place_in_room`** includes the `[scene]` overrides but not the flat path's auto-fit shrink (`change_background.room_spec`). A smaller `size` gives a smaller sigma (the room scales the other way).
- **The plate must be the full-size background:** `room3d.render` raises `ValueError` on any other size, so background_scaler is never applied to room backgrounds.
- **Weak lines never rotate the pixels.** A neighbour-borrowed camera gets `conf = 0.5`, which equals `WEAK_LINES`, so it counts as **strong** and does level the car. A photo whose own lines have 0.3 <= conf < 0.5 borrows nothing and gets `roll-from-silhouette`.
- **0.90 is used twice:** `WEAK_IOU` (pose.py) and `LOCK_IOU` (`__init__.py`). Together they mean "weak-fit ⇔ tyres not locked". `ground.LOCK_IOU` is dead code; edit the one in `__init__`.
- **The proxy layout is coupled to carmodel.** `pose.py`'s `_br`/`XS`/`G` grid and `PIECES`/`TYRE_PIECE`/`body_points` row indices mirror `CarShape.points()` in `bg_changer/carmodel.py` (sections, greenhouse, rails/mirrors, 4 tyres × 96 points; tyres in FL, FR, RL, RR order like `contacts()`). Change both together.
- **Cuts.** `skip_reason` skips a car cut at the bottom or on two or more sides (no fit). With `cut_frames = skip`, `change_background` also skips one-side cuts. The tyre search stays more than 4 px from cut columns. The slope check runs only on uncut cars.
- **Determinism.** The RNGs are seeded (`lines`: `default_rng(0)`/`(1)`); `max_seconds` only adds `fit-slow`, and the search never depends on the clock (`test_fit_and_render_are_deterministic`).
- **`rooms._CACHE`** holds Room objects at module level (generic ones keyed by anchor, `horizon_y` and plate shape), and each Room caches its plate pyramids. Edit room numbers in a fresh process.
- **`GENERIC_BACKDROP_ROWS = 10`** duplicates `generic.make_spec`'s floor-mask top `horizon + 10`; keep the two in sync. `generic_room` returns None (so the background falls back to flat) when `anchor_y <= horizon_y + 10`.
- **Room units differ per room.** background_1 uses disc units (6.0 m disc), background_2 and background_3 metres, generic rooms camera heights. So `CAMERA_MARGIN = 0.3` is in different units. background_1 is special-cased by name (disc gloss); `ROOM_SHADOW` is keyed by `scene_key`.
- **Exceptions.** `line_camera` catches only ValueError, LinAlgError and FloatingPointError. `fit_measured` catches everything (`lines-error`). `change_background` turns other fit or place exceptions into a csv `note`.
- **Stale docstrings** (trust the code): pose.py says h is in [1.0, 2.2] and that line priors are used only in the final refit. In fact, with strong lines h is in [0.5, 3.0], and the priors apply to every candidate. The `TyreBottom.status` comment omits `far`.

Key results.csv flags (fit flags in `fit.flags`, per-background flags in `Placement.flags`):

| Flag | Where | Condition |
|---|---|---|
| `fit-failed` | pose | IoU < 0.75, or h outside (0.3, 3.0) m. The pose is reset to the label yaw (else the fitted one) at h 1.5; no lock, no 3D shadows (`with_3d=False`) |
| `weak-fit` | pose | 0.75 <= IoU < 0.90 |
| `yaw-unsure` | pose | margin < 2 nats. A refit from the label wins if its IoU is within 0.02 (`yaw_source = label`) |
| `flip-ambiguous` | pose | a candidate within 20° of yaw+180 has IoU >= the winner's - 0.01 |
| `neighbour-car`, `merged-object`, `mask-reflection`, `occluder` | pose | residual blob > 3% of the car's area (blobs > 8% are excluded and the fit is redone) |
| `camera-height-at-bound`, `body-ambiguous`, `fit-slow` | pose | h within 0.02 of the limit / the other preset's IoU within ±0.01 / took over 120 s |
| `roll-from-silhouette`, `car-on-slope`, `possible-slope`, `partial-level`, `body-guessed` | `__init__` | see `fit_photo` above; `possible-slope` is also set when \|pose roll - line roll\| > 3° |
| `no-lines`, `lines-error`, `roll-from-neighbours`, `no-view-label` | lines / batch | |
| `lens-change`, `focal-weak`, `focal-prior` | `lines.choose_focal` | |
| `no-room`, `generic-room`, `camera-outside-room`, `soft-floor`, `invented`, `contact-mismatch`, `reflection-skipped` | place | see `place_in_room` above |

`yaw-vs-label` (more than 24° from the label) is added by `change_background.fit_summary`, not here.

## How to extend
- **A new calibrated background with its own 3D room:**
  1. Write a scene module in `bg_changer/scenes/` (copy `background_2.py`) and register it in `SCENES` in `scenes/__init__.py`.
  2. Expose the plate camera's public numbers from that module, as background_1 does with `FOCAL`/`PRINCIPAL_POINT`, background_2 with `K, R, CAM_H, JUNCTION`, and background_3 with `GROUND_H`.
  3. Write `room_<name>()` in `rooms.py`. It returns `Room(name, K, R, C, SPEC.plate_size, poly, wall_top, ceiling, texture_box=..., devignette=...)` with the world origin at `SPEC.anchor`'s floor point, x right, y away from the camera, z up. Close the polygon behind the camera, as the existing rooms do with `C[1] - 30`.
  4. Add it to `BUILDERS`, and optionally a `ROOM_SHADOW` entry. Also add the module to rooms.py's `from bg_changer.scenes import ...` line, the builder to `__all__`, and the room to the module docstring.
  5. Put the plate in `backgrounds/<name>.webp` at exactly `plate_size` (1920x1080: the scene-parametrized tests build plates of that size). Tests and `check_room` read it as `.webp`.
  6. Update the tests: `test_room3d`'s re-render parametrize, `MEASURED` in `test_room_mode` and `test_scenes::test_anchor_override_re_pins_the_calibrated_floor`. `test_car_fit` picks the new scene up automatically through `SCENES`.
  7. Text that lists the backgrounds: the `config.toml` `[scene]` comment, README's background tables, `tools/check_room.py`'s docstring and sheet header.
- **A user background without code:** set `horizon_y` in `[scene.<name>]` in config.toml, at least 10 px above the anchor row. `generic_room` then builds a level camera (focal 0.866·w), a floor up to the horizon, and a backdrop beyond it. Nothing changes here.
- **A new flag:**
  1. A per-photo flag goes in `fit.flags` (`__init__`, `pose` or `batch`); a per-background flag goes in `Placement.flags` in `place.py`. Both reach `results.csv` through `change_background.room_row`.
  2. Document it in the README flag list.
  3. Add it to `QUIET_FLAGS` in change_background if it should stay off the console.
- **A new results.csv column:** put the value in `place.py`'s `metrics` dict (per background) or read it from the fit, then add it to `ROOM_CSV_FIELDS` and `room_row` in `change_background.py`. Optionally add it to `tools/check_room.FIELDS`.
- **A new body preset:** add it to `pose.ROOF` and `pose.BODIES`, and to `BODY_TYPES` in `bg_changer/cars.py`. Adjust `guess_body`/`body_ambiguous` if it should be guessed.
- **Where the tuning lives:**
  - fit thresholds: `pose.py` top constants;
  - level and cut: `level.py`;
  - lock bounds (±15% height, ±3° pitch) and patch size: `ground.py`;
  - output and flag thresholds: `place.py` (`MIN_FLOOR_TEXELS`, `CAMERA_MARGIN`, `CONTACT_MISMATCH`);
  - shadow looks: `rooms.ROOM_SHADOW`;
  - focal pooling: `lines.py` (`POOL_*`, `LENS_TOL`, `PRIOR_HFOV_LONG`).

  `max_output_size` (default 1920; 0 means no cap) and `keep_background_sharp` (default true; `RoomRun.min_texels` is 1.0 or 0.0) are parsed in `change_background.py`.

## Tests
- `tests/test_car_fit.py`: pose recovery, the bit-exact pixel rule on every room, levelling (under 0.5° means a plain crop; a rolled photo is tracked everywhere), cut edge on the border, determinism, `stage_scale` against `size`, degenerate masks, synthetic vanishing points, focal pooling and `lens-change`, `paste`.
- `tests/test_room_mode.py`: `change_background` stages 2-3 through `batch.fit_measured`, onto the three rooms plus a generic room and a no-room background; csv columns; written pixels.
- `tests/test_room3d.py`: `rooms.BUILDERS`/`build_room`.
- `tests/test_debug_images.py`: `line_camera(detail=)` changes nothing.
- Not pytest: `tools/check_room.py` (chain IoU and car-pixel self-checks on real folders; needs the segmenter or its cached masks).
