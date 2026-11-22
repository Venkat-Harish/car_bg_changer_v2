# hosted: Idea B, hosted open models on fal.ai (cut-out, glass tint, camera)

## Purpose
Set-up B of [../docs/glass-cutout-camera-options.md](../docs/glass-cutout-camera-options.md): every step runs on an
open model hosted on fal.ai, and nothing is generated. BiRefNet on fal cuts the car out, SAM 3 ("car") keeps only
this car, SAM 3 ("car window", "windshield") finds the glass, a fixed dark tint replaces it, MoGe-3 measures each
photo's focal and floor (camera height, pitch, roll), and VGGT solves all photos of the car together (shared focal,
the angles between shots). This costs about $0.04 per photo. Only imported when `[hosted] enabled = true`.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `fal.py` | Queue REST client (`queue.fal.run`, plain `requests`), uploads (fal CDN, data-URI fallback), disk cache, cost guard | `FalClient.run/fetch_bytes/fetch_image`, `load_key`, `encode_image`, `COST`, `FalError`, `BudgetExceeded`, `Spend` |
| `masks.py` | BiRefNet v2 mask-only alpha; SAM 3 instances; `HostedIsolator`, with the same contract as `car_isolation.CarIsolator` | `birefnet_alpha`, `sam3_instances`, `parse_sam3`, `pick_instance`, `HostedIsolator.isolate`, `Instance` |
| `glass.py` | Window mask (two SAM 3 prompts, inside the eroded car mask); deterministic tint; originals | `window_mask`, `tint_glass`, `glass_pct`, `save_original`; constants `PROMPTS`, `TINT_*`, `ERODE_FRAC`, `FEATHER_FRAC` |
| `camera.py` | MoGe-3: PLY point cloud, RANSAC floor, height/pitch/roll. VGGT: per-frame focal and relative yaw. Combining them; yaw prior helpers | `moge_frame`, `vggt_car`, `combine`, `yaw_probs`, `anchor_yaws`, `fit_floor`, `read_ply_points`; `FrameCam`, `HostedCamera` |
| `pipeline.py` | Glue used by `change_background.py` | `HostedRun` (`cut_out`, `glass`, `camera`, `apply_focals`, `cols`, `flags`), `HostedSegmenter`, `HostedInfo` |

## How it works
- **Key:** `FAL_KEY` from the environment, else from `background_changer\.env` (gitignored). It is never printed,
  logged or cached; `FalClient.__repr__` hides it.
- **Cache:** each result is stored under `<output_folder>/_hosted_cache/`, keyed by sha1(endpoint + JSON payload +
  sha1 of each image's bytes). Images are encoded before hashing, so a rerun uploads nothing and costs $0. Result
  files (masks, PLY) are cached by URL, because fal's URLs expire.
- **Cost guard:** `COST` holds list-price estimates per call. MoGe passes `cost=` per megapixel and VGGT $0.001 per
  frame. Once `[hosted] max_cost_per_car` would be passed, `BudgetExceeded` is raised (a `FalError`), and every
  caller falls back to the local path.
- **Stage 1** (`HostedRun.cut_out`, then `glass`):
  1. Raw alpha comes from fal BiRefNet. If that fails, the local rembg model loads lazily and the photo is flagged
     `hosted-cutout-failed`.
  2. `CarSegmenter.finish(image, raw, HostedIsolator)` keeps only this car. If that fails: plain `clean_alpha`,
     flag `sam3-failed`.
  3. Window mask, then `save_original` to `<out>/originals/`, then `tint_glass`. The tinted frame is written
     losslessly to `<out>/_frames/<stem>.png`, and `Photo.frame_path` points to it, so stage 3 re-reads the tinted
     frame. If the mask fails: no tint, flag `glass-mask-failed`.
- **Camera** (`HostedRun.camera`, start of `place_room_photos`):
  - MoGe-3 runs per photo on the full frame.
  - VGGT gets one call over downscaled copies (≤1036 px). Its focal is scaled back to full size.
  - `combine` sets `Measured.external`.
  - The relative yaws are anchored on the view labels (circular median), then become `yaw_probs` (σ 20°). These
    replace the photo's `view_rows` probs.
  - `apply_focals` lets the hosted focal win over the lines after `choose_focals`, but never over a `[fit] focal`.
- **Fit hooks** (in `car_fit_to_bg/batch.py`, `_with_external`):
  - The external camera becomes a strong `LineCamera(source="hosted")`, flagged `camera-hosted`.
  - If the photo's own strong lines disagree by more than 5° in pitch or roll, the lines are kept, flagged
    `camera-lines-fallback`.
  - The MoGe height goes in as `strong_h_prior=(h, 0.15)`. This is a new `fit_pose` argument; `h_prior` stays the
    folder prior, unchanged.

## Invariants & gotchas
- **Glass is the only car-pixel change.** User decision, Nov 2026: tint every window, windscreen included. Pixels
  outside the window mask are bit-identical to the photo. After the tint, the tinted frame *is* the photo, so
  every downstream bit-exact rule holds against it (`tests/test_hosted_glass.py` checks this end to end). The
  untouched file is always in `originals/`.
- **Never stop a run because fal fails:** every hosted step catches `FalError` and falls back, with a flag.
- **Schemas were read from fal's docs (Nov 2026), not from live calls:**
  - the SAM 3 mask PNG encoding;
  - the axes, units and density of the MoGe-3 PLY;
  - VGGT's square-padding resize.
  The parsers accept the plausible variants. Check one real response of each before trusting heights.
- `tools/check_room.py` still runs the local path only.
- Memory: `camera()` keeps only ≤1036 px copies for VGGT, never the full frames.

## How to extend
- **New hosted endpoint:** add its price to `COST` (or pass `cost=`), call `client.run` with `images={field:
  ndarray}`, and read files through `fetch_*`. Tests use `tests/hosted_fakes.FakeFal`.
- **New `[hosted]` key:** `HOSTED_KEYS` / `HOSTED_DEFAULTS` / `parse_hosted` in `change_background.py`, then
  `config.toml`, then `tests/test_config.py`.

## Tests
`tests/test_hosted_fal.py` (fake HTTP session), `test_hosted_masks.py`, `test_hosted_glass.py`, `test_hosted_camera.py`,
`test_hosted_pipeline.py`, `test_hosted_room.py` (stages 2-3 with a hosted camera). None of them touches the network.
