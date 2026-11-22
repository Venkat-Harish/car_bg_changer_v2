# bg_changer: flat-path core: cut the car out, model it in 3D, place it on a scene, ground it, paste it

## Purpose
This folder holds the flat placement (`background_view = "flat"`) and the building blocks that room mode reuses: the BiRefNet cut-out (`segment`), a 3D proxy of the car seen through a fixed capture camera V (`carmodel`), placement on a scene's anchor (`placement`), shadows and reflection (`grounding`), and the only resample of car pixels in the flat path (`compose`). Per-background numbers live in [scenes/](scenes/CLAUDE.md).

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Re-exports the public API. `BackgroundChanger` is a convenience class (segmenter + `CarTable` + scene cache) used only in docs and the README. `change_background.py` calls the pieces directly instead. | `BackgroundChanger.cut_out/place/process/scene`, `__all__` |
| `segment.py` | rembg soft alpha (ONNX Runtime, CPU) -> `clean_alpha` -> mask of the main car only. Also holds the car_isolation hooks. | `CarSegmenter` (`cut_out`, `raw_alpha`, `finish`, `release`, `last_isolation`), `CarMask`, `clean_alpha`, `fill_holes`, `drop_posts`, `select_main_car`, `bottom_profile`, `NoCarFound`, `DEFAULT_MODEL`="birefnet-general-lite", `DEFAULT_THREADS`=4, `DEFAULT_EDGE_SHRINK`=2 |
| `probe.py` | Measures the cut-out: solid box, lowest point, `tie`, local lowest points ("contacts"), cut sides. Parses the frame name. | `probe_car` -> `CarProbe`, `parse_name` -> `NameInfo`, `LABELS`, `NAME_RE` |
| `yaw.py` | Gets the view angle from the label, the shape, or an override | `estimate_yaw` -> `YawEstimate(deg, sigma, source, flags)`, `BIN_DEG`, `LABEL_SIGMA`=15, `LOW_CONF`=60 (sigma 25 below it), `SHAPE_SIGMA`=30 |
| `carmodel.py` | An EQB-shaped convex proxy, stretched to any car's dimensions and projected through V at every yaw | `CarShape` (metres; `points`, `contacts`, `footprint`), `CarTable` (`at(yaw)` -> `ViewRow`, `project`, `side_length`, `cx_smoothed`, `floor_row/floor_depth/floor_homography`), `CAPTURE_DISTANCE_M`=4.70, `CAPTURE_ELEVATION_DEG`=16.08, `ASPECT_BIAS`=1.05 |
| `cars.py` | Reads cars.json (cm) into `CarShape` (m) | `load_cars`, `parse_cars`, `select_car`, `CarSpecError`, `DIMENSION_KEYS`, `BODY_TYPES` |
| `placement.py` | Scale from height, ground pin, footprint centre on the anchor, edge guard, and a map from V's columns to the pasted car's columns | `plan_on_anchor` -> `Plan`, `XMap`, `raw_scale`, `skip_reason`, `CUT_POLICIES`=("anchor","centre","skip"), `MAX_GUARD_SHIFT`=100, `HW_RANGE`=(0.90,1.18), `CAP_TOL`=0.01 |
| `grounding.py` | Draws shadows and reflection on the background, in linear light | `shadow_multiplier`, `add_reflection`, `outline_floor` -> `OutlineFloor`, `snap_contacts` -> `Contact`, `contact_offsets`, `PinnedFloorMap`, `PlateFloor` with subclasses `LevelFloor`, `HomographyFloor` and `CompressionFloor`, `ShadowStyle`, `ReflectionStyle`, `srgb_to_linear`/`linear_to_srgb`, public aliases `footprint_layer`/`silhouette_layers` |
| `compose.py` | The flat path's single car-pixel operation, plus the paste | `place_car`, `scale_car`, `paste`, `cutout_bgra`, `DEFAULT_SHADOW`=1.0 |

## How it works
Flat path, per photo, as `change_background.py` drives it (see [../CLAUDE.md](../CLAUDE.md)):
1. **Cut out.** `CarSegmenter.cut_out(img)` runs `raw_alpha` (rembg `only_mask`, resized to the photo), then `finish`, then `clean_alpha`. Inside `clean_alpha`:
   - threshold at 0.5, `fill_holes`, then `drop_posts`: an opening with a horizontal line `POST_WIDTH`=1.2% of the long side; `select_main_car` keeps the blob with the best area x (1 - 0.5 x off-centre);
   - `fill_holes` again, which gives `solid`;
   - alpha is set to 1 on `solid` eroded by `CORE_BAND` (0.3%), and soft alpha is kept only within `EDGE_KEEP` (0.4%) of `solid`;
   - then the `edge_shrink` erosion, and alpha below 1/255 becomes 0.

   Outputs: `cut[side]` is set where `solid` comes within `BORDER_TOL`=3 px of that photo edge. `box` is the bbox of `solid` (used for placement). `crop` is the bbox of alpha > 0 (what gets copied). `bottom` is the lowest solid row per column.
2. **Prepare** (`prepare_photo` in change_background): [background_scaler](../background_scaler/CLAUDE.md)`.crop_to_car`, then `parse_name(path.stem)`, `probe_car`, `estimate_yaw(info, probe, table)`.
3. **Scene**, once per background: `scenes.load_scene(name, plate, CarTable(shape), overrides)`. background_scaler may then shrink the scene and the plate together.
4. **Place.** `compose.place_car` calls:
   - `Scene.plan`, which is `plan_on_anchor`;
   - `scale_car`: one `cv2.INTER_AREA` resize of crop + alpha to `plan.size`, skipped when the size is unchanged;
   - `Scene.compose`, which draws reflection and shadow on the plate, then calls `paste`.

   It returns `(None, None)` when the plan is None (the frame is skipped).

**Geometry.** `CarTable` works in "anchor metres": image px = anchor + px_per_m * value. The scene sets px_per_m = `side_px * size / table.side_length()`. So a pure side view of any car spans `side_px`, and the car's dimensions only set its proportions and how far its near end comes forward. V (4.70 m away, 16.08 deg down) is the walkaround's viewpoint and a constant of the method. The background's own camera enters only through the scene's `PlateFloor`, via `PinnedFloorMap`: identity down to the car's lowest row, then the plate's foreshortening, which keeps shadows compact.

**Placement** (`plan_on_anchor`):
- **Scale:** `s = min(1, raw_scale)`. `raw_scale = px_per_m*row.h/probe.h`, or the width divided by ASPECT_BIAS when the car is cut at top or bottom.
- **y:** the lowest solid row lands on `anchor_y + px_per_m*row.low`. With a bottom cut and the "anchor" policy, the roof is pinned instead.
- **x:** the box centre goes to `anchor_x + px_per_m*cx_smoothed(yaw, sigma)`. With one side cut and "anchor", the uncut edge goes to V's edge, moved in by half the ASPECT_BIAS difference.
- **Guard:** moves the box onto the canvas and above `max_lowest_row`, by at most `max_shift`, and sets the flags `near-edge` / `cropped`.
- **XMap:** maps V's columns to output columns, with knots at the car's ends and its lowest point. Any knot slope outside 0.5-2x `rho` drops the lowest-point knot. This lets grounding follow the real car even when the label's angle is off.

**Yaw** (convention: 0 = front faces the camera, 90 = right side, 180 = rear, 270 = left):
- From a label: the bin centre, sigma 15; below 60% confidence, sigma 25 and the flag `low-conf-label`.
- No label: the best aspect match over the 8 bins (a cut car gets 90), sigma 30, flag `no-view-label`.
- Override: sigma 3.

**Used outside this folder:**
- [car_fit_to_bg](../car_fit_to_bg/CLAUDE.md):
  - `carmodel`: `CarShape`, `CarTable`, `REF_*`, `SECTION_STEP`, `_ref_half_width`, `_wheel_visible` (pose.py, ground.py);
  - `grounding`: `ShadowStyle`, `ReflectionStyle`, `Contact`, `add_reflection`, `contact_offsets`, `outline_floor`, `footprint_layer`, `silhouette_layers`, `LevelFloor` (ground.py, rooms.py);
  - `compose.paste` (place.py), `segment.fill_holes` (level.py), `probe.parse_name` (batch.py), `probe.LABELS` and `yaw.BIN_DEG` (pose.py).
- [car_isolation](../car_isolation/CLAUDE.md)/isolate.py imports `segment._bbox`, `_disk`, `fill_holes` and `select_main_car`.
- `debug_images.py` reads `clean_alpha`'s `trace` dict and `Plan`.
- tools/check_scenes.py rebuilds `CarMask` objects from saved cut-outs using `fill_holes` and `bottom_profile`.

## Invariants & gotchas
- **Car pixels.** In the flat path only `compose.scale_car` resamples the car: INTER_AREA, and never an enlargement, because `plan.scale` is capped at 1. It snaps alpha > 1-1e-4 to exactly 1. `paste` copies pixels where alpha >= 1 with no float round trip. Shadows and reflection are drawn on `out` before `paste`. tests/test_compose.py checks this bit-exactly on every scene with `shadow=1.5`. Never colour-correct, sharpen or draw over the car.
- **Flat output must stay bit-identical** to the pinned baseline (tests/baseline_flat.toml, tools/check_flat.py). The comparison covers:
  - the composites;
  - the `<bg>/debug/*.jpg` overlays (the baseline has `debug_mode = true`);
  - the cut-outs (`cutout_bgra`);
  - the results.csv values (flag names and their order, rounded numbers).

  So a change to any constant, any float operation order, a flag string or the overlay drawing breaks it. Re-baseline only on purpose.
- **Shared with room mode.** `grounding`, `carmodel` and `compose.paste` also drive room output, so a change here alters both paths.
- **Private names used elsewhere** (do not rename): `carmodel._wheel_visible`, `_ref_half_width`, `REF_*`, `SECTION_STEP`; `segment._bbox`, `_disk`, `fill_holes`, `select_main_car`, `bottom_profile`, `BORDER_TOL` (car_fit_to_bg/batch.py mirrors it); `grounding._silhouette_layers` (tests).
- **car_isolation off must be bit-identical.** `finish(..., isolator=None)` is plain `clean_alpha(raw, edge_shrink, trace=trace)`, and `fill_holes(glass_ok=None)` is a plain fill.
  - With an isolator, if everything gets removed (`NoCarFound`), `finish` falls back to the plain cut-out and sets `status="unsure"`.
  - `finish` also accepts a uint8 raw alpha (the cache from the two-pass run) and must equal a one-pass run.
  - `release()` drops the session: `raw_alpha` then fails, `finish` still works.
  - change_background resets `last_isolation = None` after each photo.
- **`trace` must change nothing.** debug_images.cutout_pictures reads its keys `binary`, `posts`, `solid`, `core` and `remove` (`car` is also set).
- `CarSegmenter.__init__` imports onnxruntime/rembg lazily, and the model-free tests depend on that. Keep those imports inside the class.
- In `clean_alpha`, holes are filled *before* `drop_posts`. Otherwise thin pillars between see-through windows look like posts.
- **Units.** cars.json is in cm; `CarShape` is in m and rejects lengths outside 1-20 m and heights outside 0.5-6 m. `parse_cars` rejects values > 2000 (mm) and < 20 (m). The `WheelBase` key is matched case-insensitively, the file is read as `utf-8-sig`, and duplicate names are an error. `body` (suv/hatch/sedan) only matters in room mode.
- **File-name contract.** `parse_name` takes the stem (no extension) and matches `^<num>s_<label>_<1-3 digits>pct$`. Labels are sorted longest-first, so "front-right" wins over "front". `BIN_DEG` keys must equal `probe.LABELS`, because car_fit_to_bg/pose.py uses both. A name that does not parse falls back to `yaw_from_shape`.
- `YawEstimate.source` is documented as label/shape/override, but room mode's flat fallback also builds `YawEstimate(fit_yaw, 10.0, "fit")`.
- **V and the proxy shape** set every flat placement, through `row.h`, `row.low` and `side_length`, which in turn sets px_per_m. Changing them moves every car and breaks the reference rows in test_carmodel. `ASPECT_BIAS` is also used in `yaw_from_shape` and in the cut-frame logic.
- **Skip rules.** A car cut on a side *and* on top/bottom is never placed; with `cut_policy` "skip", any cut skips the frame. background_scaler/sizing.py calls `raw_scale` and `skip_reason` too, so keep their meaning in sync.
- **Flags raised here:** yaw flags, `cut-<side>`, `scale-capped`, `near-edge`, `cropped`, `hw-mismatch`. scenes adds `contact-unsnapped`/`contact-partial`. `Scene.compose` mutates `Plan` (`floor`, `contacts`, `flags`) for the debug overlay.
- **The sRGB <-> linear round trip is exact** for untouched 8-bit values, and `shadow_multiplier` is exactly 1 outside shadows. That is why the ROI round trip leaves the rest of the background bit-identical (test_tint_leaves_unshadowed_floor_exactly_alone).
- **Shadow size is clipped.** `ShadowStyle` blurs are in metres (`tyre_sigma_m` 0.035, `core_sigma_m` 0.20, `skirt_sigma_m` 0.25); the contact/gap/band blurs in `_silhouette_layers` are pixel-based (`thick = max(2, 0.06 * px_per_m)`). The footprint raster only reaches 1.2 m past the car (`_footprint_layer`) and flat `Scene.compose` only shades the car box ± 0.9 m, so softer or wider shadows end in a straight cut unless both grow. Room mode uses the same footprint layer, scaled by `run.shadow * spec.shadow_gain`.
- **No background-scaling logic here.** Scenes only carry `SceneSpec.px_scale`; the shrinking belongs in background_scaler.

## How to extend
- **Better yaw estimate:** `yaw.estimate_yaw` is the single hook. Return a `YawEstimate` with an honest `sigma`: placement uses sigma through `cx_smoothed`. This changes flat output, so re-baseline.
- **New car:** add an entry to cars.json (cm, optional `body`). No code change.
- **New body type:** add it to `cars.BODY_TYPES`, plus a proxy preset in car_fit_to_bg (pose.py / `guess_body`).
- **New shadow layer:**
  - add a `ShadowStyle` field whose default leaves the multiplier at exactly 1 (so the flat baseline holds);
  - add the layer to the opacity loop in `shadow_multiplier`;
  - mirror it in car_fit_to_bg/ground.py if room mode should get it.
- **New floor model:**
  - subclass `PlateFloor` (`row`, `depth`);
  - re-pin it at a new anchor in `scenes/base.with_overrides`;
  - scale it in `background_scaler/rescale._scaled_floor`, which raises TypeError for unknown types.
- **New cut policy:**
  - add it to `CUT_POLICIES` and handle it in `plan_on_anchor` and `skip_reason`;
  - change_background validates `cut_frames` against `CUT_POLICIES` and maps room's "frame" to "anchor" in flat mode, so check that mapping;
  - update the README.
- **New flag:** append it in `plan_on_anchor` and document it in the README's flags list. It changes the flat results.csv only when it fires.

## Tests
- `test_carmodel.py`: proxy rows, floor maps, stretching, contact visibility.
- `test_cars.py`: cars.json parsing and the shipped file.
- `test_probe.py`: `parse_name`, `probe_car`, `estimate_yaw`.
- `test_placement.py`, `test_grounding.py`.
- `test_compose.py`: `clean_alpha`, posts, the bit-exact car pixels.
- `test_car_isolation.py`: the `clean_alpha`/`finish` hooks, two-pass equality.
- `test_debug_images.py`: `trace`.
- `test_config.py`: `load_cars`/`select_car` on the shipped files.
- Indirectly: `test_scenes.py`, `test_background_scaler.py`, `test_car_fit.py`.
- Flat bit-identity is checked manually with tools/check_flat.py (it needs a model run). See [../tests/CLAUDE.md](../tests/CLAUDE.md).
