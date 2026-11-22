# room3d: a background picture rebuilt as a 3D room and re-rendered from any pinhole camera

## Purpose
This is the background side of room mode only: it knows nothing about cars. A `Room` is a floor, extruded walls and an optional ceiling, textured by projecting into the plate's calibrated camera. `render` ray-casts it from a new camera on the CPU with NumPy and OpenCV (no neural network).

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Re-exports only | everything listed below |
| `room.py` | The room model, camera helpers, colour conversion, hole fills in surface coordinates | `Room` (`floor_point`, `px_per_unit_at`, `contains`, `clamp_inside`, `plate_layers`, `wall_hits`, `wall_visibility`, `fill_wall`, `fill_plane`, `tile`), `look_at`, `kmat`, `plane_h`, `project`, `srgb_to_linear`, `linear_to_srgb`, `draw_room`, `FLOOR/WALL/CEIL/NONE = 0/1/2/3`, `FILL_MARGIN = 2.0` px, `FRONT_EPS = 0.05` |
| `render.py` | Per-pixel ray-cast, plate sampling, fills | `render(plate, room, K, R, C, (w, h), fill=None, bands=(0.45, 1.7), hp_sigma=10, sky_sigma=25, chunk_px=1<<20) -> Render`, `Render(lin, hole, kinds, floor_xy, texels, seconds)`, `texels_per_px`, `FILL_DEFAULT` |

## How it works
- **Conventions:**
  - World origin is the stage point on the floor; x right, y forward (away from the plate camera), z up, floor at z = 0.
  - A camera is `(K, R, C)`: K has pixel centres at integer coordinates; R maps world to camera, with rows right, down, forward; C is the camera centre in world units. A world point P lands at plate pixel `K R (P - C)`.
  - Units are the plate's own calibrated units. The caller scales car metres into them (`car_fit_to_bg.place.stage_scale`).
- **`Room(name, K, R, C, size, poly, wall_top, ceiling=None, texture_box=None, devignette=None, notes)`:**
  - `poly` is a closed floor-plan polygon, extruded from the floor up to `wall_top`. The part of it behind the plate camera is never shown.
  - `ceiling=None` means open sky: the walls act as a backdrop, and background_3 and the generic room set `wall_top = 1e4`.
  - `texture_box (x0, y0, w, h)` is a clean floor patch that supplies fine texture for floor fills.
  - `devignette (a, r)`: the plate's brightness falls off as `1 - a (d/r)^2`, and this is undone in sRGB before linearising.
  - The rooms themselves are built in [../car_fit_to_bg/rooms.py](../car_fit_to_bg/CLAUDE.md) from the measured numbers in the scene modules ([../bg_changer/scenes/](../bg_changer/scenes/CLAUDE.md)).
- **`render`, per output pixel:**
  1. Cast a ray from the output camera.
  2. Find the nearest hit with the floor, a wall or the ceiling. Hits behind the camera never count. Wall hits depend only on the ray's azimuth, so they are looked up in a 65536-entry azimuth table built with `wall_hits`, covering the full circle when the nadir or zenith is in view.
  3. Project the hit point into the plate camera.
  4. Sample the plate with mip-mapping: a Gaussian pyramid of the linear plate with 6 `pyrDown` levels, the level chosen by log2 of the footprint's longer axis (eroded 3×3), INTER_CUBIC within a level, trilinear between levels.
- **Points the plate never showed** (`hole`: outside the plate, closer than `FRONT_EPS`, or no hit at all) are refilled in their surface's own coordinates. That way seams, LED strips and plinths keep running straight instead of smearing.
  - **Floor, `"hybrid"`:** the lighting comes from a low-passed plate (`hp_sigma`) at the nearest shown point (`fill_plane`, clamp along the plate camera's forward axis), plus high-frequency detail tiled mirror-periodically from `texture_box`. Without a box it falls back to `"band"`.
  - **Walls, `"band"`:** same height, folded back as a triangle wave within `bands[1]` along the wall, then clamped to the nearest shown height. Points moved in height are sampled from the `sky_sigma`-blurred plate, so no streaks appear.
  - **Ceiling, `"smooth"`:** the clamped point, sampled from the blurred plate. This avoids repeated light bars.
  - **Last resort:** anything still outside the plate is clamped in plate image space.
- **Outputs:**
  - `lin`: linear-light BGR float32. The caller shades the floor, then converts with `linear_to_srgb`.
  - `kinds`.
  - `floor_xy`: the room floor point per floor pixel, NaN elsewhere. Used for shadows.
  - `hole`: feeds the caller's `invented` metric.
  - `texels`: plate texels per output pixel.
- **`texels_per_px(room, K, R, C, px)`** gives the floor magnification at one output pixel (default: where the stage appears). `place_in_room` derives its output scale from it, so the background is never magnified (`keep_background_sharp`).
- **Caller:** `car_fit_to_bg/place.py`, which uses `render`, `texels_per_px`, `plane_h`, `draw_room`, `srgb_to_linear`/`linear_to_srgb`, `FLOOR/WALL/CEIL` and `Room.floor_point/px_per_unit_at/contains/clamp_inside`. `car_fit_to_bg/rooms.py` uses `Room` and `look_at`.

## Invariants & gotchas
- **The plate must be the exact image the room was measured on.** `render` raises `ValueError` if the plate size differs from `room.size`. Room mode always passes the full-size background, and [background_scaler](../background_scaler/CLAUDE.md) never touches it.
- **Self-consistency is tested.** Rendering through the room's own `(K, R, C, size)` must reproduce the plate: exactly for background_1, within 1 grey level for background_2 (its calibrated R is not quite orthonormal), and with no holes. Changes to sampling, the colour LUT or rounding will show up there first. background_3 is not compared because of its devignette.
- **Everything stays in linear light** until the caller's final `linear_to_srgb`. `srgb_to_linear` is a uint8 LUT; do not feed it floats.
- **Caches live on the Room object.** `plate_layers` is keyed by a blake2b hash of `plate[::7, ::7]` plus shape and sigmas. It holds the linear plate, the pyramid, and the low, high and sky layers (several float32 copies of the plate), and is cleared once it holds more than 2 entries. `wall_visibility` is cached in `_vis`. Rooms themselves are cached in `car_fit_to_bg.rooms._CACHE`, so these caches persist for the whole run (built once per background). Mutating a Room's geometry after first use leaves stale `_vis`.
- **`wall_visibility`** samples heights up to `min(ceiling or wall_top, 200)` units: 1201 levels up to 30 units, then 120 more. Very tall rooms in small units lose wall-fill precision above that.
- **`clamp_inside`** clamps the height into `[margin, top - margin]`, then pulls the camera toward `toward` by a factor of 0.95 per step, for at most 60 steps. It can return a point that is still outside; the caller does not re-check.
- **`contains`** uses `min(wall_top, ceiling)` as the top and needs the camera at least `margin` from the floor, the walls and the top.
- **Chunking.** Rays are computed `chunk_px` pixels at a time, but the full-size output maps (`map_x/y`, `floor_xy`, masks) and the sampled float32 image are allocated whole.
- **Fills are not real data.** `hole` must stay a faithful "plate never saw this" mask. The `test_room3d` analytic test requires at least 98% agreement with the true unseen set.

## How to extend
- **A new room for a background:** do not touch room3d. Build a `Room` in `car_fit_to_bg/rooms.py` (see [its CLAUDE.md](../car_fit_to_bg/CLAUDE.md)).
  - Put the origin at the stage's floor point.
  - Close the polygon behind the plate camera.
  - Give it a `texture_box` on clean, evenly lit floor so the hybrid floor fill works.
  - Add `devignette` if the plate darkens toward its edges.
- **A new fill mode:** add a branch in `Room._fold` (it turns an out-of-range surface coordinate into one the plate showed), then wire it into `render`'s per-kind fill block and `FILL_DEFAULT` (keys `floor`, `wall`, `ceil`). Floor modes besides `"hybrid"` pass straight to `fill_plane`.
- **A new surface type** (e.g. a slanted plane): extend the `T = stack([t_f, t_w, t_c])` hit selection and the `FLOOR/WALL/CEIL/NONE` kinds in `room.py`. Give it a fill in its own coordinates, and update `draw_room` and the `_debug_steps` colours in `car_fit_to_bg/place.py`.
- **Sampling changes:** keep `_mip_sample` returning `(image, texels)`. Re-run the exact re-render test in `test_room3d.py`, and the bit-exact car-pixel tests, which render through every room.

## Tests
- `tests/test_room3d.py`:
  - exact re-render of background_1 and background_2 from their own cameras;
  - every room in `BUILDERS` contains its plate camera, and a wrong-size plate raises;
  - a camera above the ceiling counts as outside and is clamped back in;
  - a synthetic textured box seen from a new rolled, downward-looking camera matches an analytic ray-cast where the plate saw it (median error <= 2, 95th percentile <= 8 grey levels; hole mask agreement > 98%).
- Indirectly: `tests/test_car_fit.py` and `tests/test_room_mode.py`, which render through every room.
