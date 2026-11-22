# bg_changer/scenes: one measured scene per background (stage, floor camera, shadow style), plus the generic scene

## Purpose
This folder turns a background picture into a `Scene`. The scene knows where the car's footprint centre stands (`anchor`), how big the stage is (`side_px`), how the floor recedes (`PlateFloor`), where shadows may fall (`floor_mask`), and the shadow/reflection styles. `Scene.plan` and `Scene.compose` are the flat path's placement and compositing. Room mode reuses the specs.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Registry and loader with the shrink-to-fit step | `SCENES`, `load_scene`, `scene_key`, `_fit`/`_fits`, `GENERIC_MARGIN`=120, `MODULE_MARGIN`=40 |
| `base.py` | Spec dataclass, the `Scene` class, config overrides | `SceneSpec`, `Scene` (`plan`, `compose`, `debug_overlay`, `px_per_m`, `plate_floor`, `mask`, `inside`), `with_overrides`, `GENERIC_HFOV_FOCAL`=0.866, `GENERIC_METRIC_RATIO`=0.89 |
| `generic.py` | Any background without its own module, configured by `[scene.<name>]` | `make_spec`, `DEFAULT_ANCHOR`=(0.5 W, 0.72 H), `DEFAULT_SIDE_FILL`=0.66, `GENERIC_SHADOW`=`ShadowStyle()`, `GLOSSY` (opacity 0.18*0.8) |
| `background_1.py` | White studio with a glossy turntable. Level camera f 1702.5. | `SPEC` (anchor (958.8, 764.0), side_px 1348, `LevelFloor(502.5, 1707.6, 6.532)`, reflection, `ring` rim ellipse, `max_lowest_row`=946), `FOCAL`, `PRINCIPAL_POINT`, `floor_mask` |
| `background_2.py` | Grey studio, matte concrete floor | `SPEC` (anchor (960, 780), side_px 1243, `HomographyFloor(FLOOR_H, ANCHOR)`, `shift_m`=0.05, no reflection), `K`, `R`, `CAM_H`, `JUNCTION`, `FLOOR_H`, `ANCHOR` |
| `background_3.py` | Race track, asphalt | `SPEC` (anchor (960, 685), side_px 1232, `HomographyFloor(GROUND_H, ANCHOR)`, warm `tint_gamma_bgr`), `GROUND_H`, `ANCHOR` |

## How it works
- **`load_scene(name, plate, table, overrides)`**:
  - `key = scene_key(name)`: lower-case, runs of non-alphanumerics become "_", then stripped. Config `[scene.<name>]` blocks and the room builders use the same normalisation.
  - If the key is in `SCENES`: `Scene(with_overrides(SPEC, overrides))`, then `_fit` with `MODULE_MARGIN`.
  - Otherwise: `Scene(generic.make_spec(name, w, h, overrides))`, then `_fit` with `GENERIC_MARGIN`.
  - `_fit` multiplies `spec.size` by 0.97, up to 60 times, until every yaw (in 5 deg steps) keeps `margin` px to the left, right and top, and `low <= min(h-30, max_lowest_row)`. When it shrinks, it warns "<name>: <car> made N% smaller ...".
- **`Scene.__init__`**:
  - Raises ValueError if the plate size != `spec.plate_size`.
  - `px_per_m = side_px * size / table.side_length()`.
  - `plate_floor`: the spec's own if set. Else, with `horizon_y` above the anchor, `LevelFloor.from_anchor`, whose z0 comes from `GENERIC_HFOV_FOCAL` and `GENERIC_METRIC_RATIO`. Else `CompressionFloor(0.6)`.
  - `mask` = `floor_mask(w, h)` blurred with sigma max(0.5, 1.5*px_scale). `inside` = the blurred `ring` ellipse (the glossy disc), only when `ring` is set.
- **`Scene.plan`** calls `placement.plan_on_anchor(..., spec.max_lowest_row, MAX_GUARD_SHIFT * px_scale)`.
- **`Scene.compose(plate, pixels, alpha, plan, probe, shadow_gain)`**, in order:
  1. The ROI is the car box ± 0.9 m, from the car's mid-height to the bottom of the image.
  2. `outline_floor`, then `snap_contacts`. Both are stored on `plan.floor/contacts`, and `_flag_contacts` adds `contact-unsnapped`/`contact-partial` in side and 3/4 views.
  3. Convert to linear light, then `add_reflection` if `spec.reflection` is set.
  4. `shadow_multiplier` with gain = run `shadow` * `spec.shadow_gain` (skipped at 0).
  5. Back to sRGB, then `paste` the car last.
- **`debug_overlay`** draws the anchor, V's box, the footprint through `PinnedFloorMap`, the floor line, the contacts and a text line. It is written by change_background (flat `<bg>/debug/`), by debug_images.flat_picture (room mode's flat fallback) and by tools/check_scenes.py.
- **`with_overrides`** (modules only) handles these keys:
  - `anchor`: also re-pins a `HomographyFloor`/`LevelFloor` at the new anchor;
  - `size`;
  - `side_fill`: sets `side_px = side_fill * plate width`;
  - `shadow`: sets `shadow_gain`;
  - `reflection`: only `false` does anything (it sets None);
  - `horizon_y`: stored, but unused while `plate_floor` is set;
  - `floor` is ignored for modules.
- **`generic.make_spec`** reads `anchor`, `side_fill`, `horizon_y`, `floor` ("glossy" and `reflection` not false gives `GLOSSY`), `size` and `shadow`. Its floor mask is the rows >= `horizon_y + 10`, or the whole picture when there is no horizon.
- **Validation.** All keys are validated before they get here, by `change_background.parse_scene_overrides` (`SCENE_KEYS`; ranges: size 0.2-3, side_fill 0.1-1, shadow 0-3).

**Room mode** ([car_fit_to_bg](../../car_fit_to_bg/CLAUDE.md), [room3d](../../room3d/CLAUDE.md)):
- change_background's `room_spec` builds the spec with `with_overrides` / `generic.make_spec`, but without `_fit`. In room mode the car never has to fit inside the picture's frame.
- `place_in_room` uses the spec's `anchor` (through `room.floor_point`), `side_px*size`, `shadow_gain` and `reflection`.
- Room shadow styles come from `rooms.ROOM_SHADOW[name]` (else `DEFAULT_ROOM_SHADOW`), not from `spec.shadow`.
- `rooms.BUILDERS` read each module's public constants and its unoverridden `SPEC`:
  - bg1: `FOCAL`, `PRINCIPAL_POINT`, `SPEC.plate_floor.k`, anchor;
  - bg2: `K`, `R`, `CAM_H`, `JUNCTION`, anchor;
  - bg3: `GROUND_H`, anchor.
- `rooms.generic_room` builds a room from `spec.horizon_y` + `GENERIC_HFOV_FOCAL` for any background without a builder.
- `load_scene` still runs for every background in room mode; its shrink warning is silenced when the background has a room.

[background_scaler](../../background_scaler/CLAUDE.md)'s `scale_spec` shrinks every pixel-valued `SceneSpec` field for small photos.

## Invariants & gotchas
- **Flat bit-identity.** Every number in background_1-3, `base.Scene.compose`/`debug_overlay` and the generic defaults feeds the pinned flat baseline (tests/baseline_flat.toml runs background_1-3). Changing any of them is a deliberate re-baseline.
- **Plate size.** The modules' numbers are measured at `plate_size` (1920, 1080). A resized background file raises ValueError, and change_background exits on it, even in room mode. Fixes: rename the file (it then uses the generic scene), or let background_scaler shrink it at run time (it only shrinks).
- **Required for every scene** (test_scenes, parametrized over `sorted(SCENES)`):
  - `plate_floor.row(0)` equals `anchor_y` within 1.5 px. That is why `with_overrides` re-pins the floor on an anchor change; any new calibrated floor must be pinned at the anchor too.
  - The anchor lies on `floor_mask` (mask > 0.99).
  - With the EQB, every view stays >= 150 px from the edges and has `low` <= H - 120 and <= `max_lowest_row` + 1.
- **Names must agree:** the `SCENES` key == `scene_key(image file stem)` == `SPEC.name`. `notes` must not contain "generic": change_background prints "generic scene" for any spec whose `notes` contain it.
- **Public names read by car_fit_to_bg/rooms.py.** Do not rename `FOCAL`, `PRINCIPAL_POINT`, `K`, `R`, `CAM_H`, `JUNCTION`, `GROUND_H`, `ANCHOR`, `SPEC`, `GENERIC_HFOV_FOCAL`, or `scene_key`. Changing a module's measured camera numbers also changes its 3D room.
- **The generic "+10" is coupled.** The generic floor mask starts 10 px below the horizon, matching `rooms.GENERIC_BACKDROP_ROWS = 10`, which no_room_hint also uses. Change both together.
- **`px_scale`.** Any fixed pixel size in `Scene` (blur sigma, guard shift, marker or text size) must be multiplied by `spec.px_scale`, or a shrunk background stops matching the full-size result downscaled. Any new pixel-valued `SceneSpec` field must be scaled in background_scaler/rescale.scale_spec.
- **Allowed `plate_floor` types** in a spec are `LevelFloor` and `HomographyFloor`. `rescale._scaled_floor` raises TypeError for anything else. `CompressionFloor` is only ever created inside `Scene`.
- **`side_px` is V's predicted side-view length.** The pasted car is about 5% narrower (`ASPECT_BIAS`). That is why background_1's docstring says "~73% of the 1761 px rim" while 1348/1761 = 77%.
- **Override limits.** `reflection = true` cannot add a reflection to background_2/3. `spec.shadow` is used by the flat path only.
- **Stale docstring.** background_2's docstring says the shadow is "nudged 0.10 m toward the camera", but `SPEC.shadow.shift_m` is 0.05 (rooms.py uses 0.05 too).

## How to extend
- **Add a measured background** (flat + room):
  1. Measure the picture: camera, stage point, floor. Record how in the module docstring and in docs/findings.md.
  2. Copy background_2.py (homography floor) or background_1.py (level floor + glossy disc) to `scenes/<key>.py`, where `<key> = scene_key(image stem)`. Fill in `SPEC = SceneSpec(name=<key>, plate_size=(W, H), anchor, side_px, floor_mask, plate_floor pinned at the anchor, shadow=ShadowStyle(...), [reflection, ring, max_lowest_row], notes)`. Make the camera constants public names.
  3. In `scenes/__init__.py`, import the module and add `"<key>": <module>.SPEC` to `SCENES`.
  4. Put the picture in `backgrounds/<key>.webp` at exactly `plate_size`. Use **.webp** (test_car_fit and tools/check_room.py load `<name>.webp`) and **1920x1080** (the tests parametrized over `SCENES` build 1920x1080 plates: `grey_plate`, test_scenes `H, W`); a native 4:3 photo raises ValueError in `Scene.__init__`.
  5. Room: in car_fit_to_bg/rooms.py write `room_<key>()` (a `Room` built from the module's constants), register it in `BUILDERS`, and optionally add `ROOM_SHADOW["<key>"]`.
     - Without a builder, room mode places the background flat (`no-room`), or in a generic room if the spec sets `horizon_y`.
     - test_car_fit fails without one: it parametrizes `place_in_room` over `SCENES` and needs a room.
  6. Run the tests parametrized over `sorted(SCENES)`. For the extra spots to update (test_room_mode, test_room3d, the anchor re-pin test), see [../../tests/CLAUDE.md](../../tests/CLAUDE.md).
- **Tune a background for a run:** use `[scene.<name>]` in config.toml, not code.
- **New `[scene]` key:**
  - add it to `SCENE_KEYS` and validate it in `change_background.parse_scene_overrides`;
  - apply it in `with_overrides` and in `generic.make_spec`;
  - if it is in pixels, scale it in `rescale.scale_spec`;
  - if room mode should honour it, use it in `place_in_room`;
  - update README's settings table and the config.toml comments.
- **New `SceneSpec` field:** its default must reproduce today's output (the flat baseline). If it is in pixels, scale it in `rescale.scale_spec` and assert it in test_background_scaler.

## Tests
- `test_scenes.py`: fit and margins, turntable fill, overrides, the generic scene, shrink warnings, anchor re-pin, wrong plate size.
- `test_compose.py`: bit-exact car pixels on every scene.
- `test_placement.py` (`scene1` = background_1), `test_grounding.py`.
- `test_background_scaler.py`: every scene shrunk.
- `test_car_fit.py`: `place_in_room` with `SCENES[name]`.
- `test_room_mode.py`: `load_scene`, `generic.make_spec`.
- `test_config.py`: `[scene]` parsing.
