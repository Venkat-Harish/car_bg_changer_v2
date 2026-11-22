# background_scaler: shrink a background and its scene measurements to the photos' resolution (flat path)

## Purpose
The car is never enlarged. So when a photo's car is smaller than the stage needs (for example 464x832 video frames), this package shrinks the background picture *and* every pixel measurement of its scene until the car fills the stage at its own resolution. It is used by the flat path, and in room mode only for backgrounds without a 3D room.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Re-exports and usage docstring | `__all__` |
| `sizing.py` | Decides how small | `needed_scale`, `folder_scale`, `background_size`, `MODES`=("folder","photo","off"), `DEFAULT_MODE`="folder", `DEFAULT_MIN_WIDTH`=640 |
| `rescale.py` | Builds the shrunk scene | `scale_scene`, `scale_spec`, `_scaled_floor`, `_scaled_mask` |
| `scaler.py` | Per-run mode, last-size cache, photo crop | `BackgroundScaler` (`photo_scale`, `folder_scale`, `size`, `apply`), `crop_to_car` |

## How it works
- **How small.** `needed_scale` gives `k = min(1, 1/raw_scale(probe, table.at(yaw), px_per_m))` on the *full-size* scene, or None when `skip_reason` says the photo is not placed. Per mode:
  - "folder": the minimum k over the placed photos (1 if none);
  - "photo": each photo's own k;
  - "off": always 1.
- **What size.** `background_size` returns the full size when k >= 1. Otherwise width = ceil(W*k), so the car never needs enlarging, clamped to [min(min_width, W), W]; height = round(H*w/W).
- **The shrunk scene.** `scale_scene(scene, plate, size)` returns the inputs unchanged at the same size, and raises ValueError for a bigger size. Otherwise it resizes the plate with INTER_AREA and builds `Scene(scale_spec(spec, size))`. `scale_spec` handles:
  - points, with pixel centres `(v+0.5)*k-0.5`: anchor, horizon_y, ring centre, max_lowest_row;
  - lengths: side_px by kx, ring axes, reflection `blur_px`;
  - floors: `LevelFloor` horizon and k; `HomographyFloor` pre-multiplied by the pixel map;
  - the floor mask: drawn at full size, then INTER_AREA;
  - `px_scale *= kx`;
  - sizes in metres need nothing, because they follow px_per_m.
- **Callers** in change_background ([../CLAUDE.md](../CLAUDE.md)):
  - `prepare_photo` runs `crop_to_car` on every photo, in both modes.
  - Flat "folder" mode cuts out every photo first, then calls `folder_scales`. "photo" and "off" place each photo right away. Room mode's `place_flat_fallback` uses `needed_scale(..., "anchor")`.
  - Config: `background_scale`, `background_min_width` (>= 64 there), CLI `--background-scale`. tools/check_scenes.py mirrors this flow.

## Invariants & gotchas
- **Scaling logic stays here.** [bg_changer](../bg_changer/CLAUDE.md) only exposes `SceneSpec.px_scale`, which [scenes](../bg_changer/scenes/CLAUDE.md) use for fixed pixel sizes: mask/ring feather, guard shift, debug marks.
- **Only shrinks.** At full size the same objects come back, so "off" and large photos match a run without the scaler. Any change to the sizing or rescale math changes flat output whenever a background is shrunk. The pinned baseline runs "folder" with min width 640.
- **Car pixels are untouched here.** The car still gets only `compose.scale_car`. Only the plate (background) is resampled.
- **Keep `scale_spec` complete.** Every pixel-valued `SceneSpec` field must be scaled there. A new `PlateFloor` type must be added to `_scaled_floor`, or it raises TypeError.
- **`apply` caching.** `apply` keeps only the last size per scene (`id(scene)` plus an identity check). In "photo" mode, a new size rebuilds the `Scene` (full-size mask drawing included).
- **`crop_to_car`** crops to the union of `crop` and `box` and shifts the mask. The placement must equal placing the whole photo.

## How to extend
- **New mode:** add it to `MODES` and branch on it in `BackgroundScaler.photo_scale`/`folder_scale`. Then handle it in change_background (`SCALE_TEXT`, the flat loop in `main`, `place_flat_fallback`), in tools/check_scenes.py and in the README.
- **New pixel-valued `SceneSpec` field:** scale it in `scale_spec` and assert it in `test_a_shrunk_scene_is_the_same_stage_k_times_smaller`.

## Tests
`test_background_scaler.py` (sizes, modes, shrunk scenes per `SCENES`, bit-exact car at scale 1, `crop_to_car`). `test_room_mode.py` builds `Run` with `BackgroundScaler("folder", 640)`.
