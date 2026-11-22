# car_isolation: optional YOLOX-s + SAM ViT-B pass that keeps only the target car in the cut-out mask

## Purpose
BiRefNet marks "the salient object(s)", so a car parked behind ours, a person or a hand touching it joins the blob,
and `fill_holes` fills every gap. This package returns two masks (`remove`, `glass_ok`) that
`bg_changer/segment.py:clean_alpha` applies. Off by default (`[car_isolation] enabled`); it changes the mask only.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `__init__.py` | Public API re-exports | `CarIsolator`, `IsolationResult`, `DEFAULT_MIN_SCORE`, `Detection`, `VehicleDetector`, `overlay`, `plan_removal` |
| `models.py` | Model locations; download once on first use | `yolox_path()` (pooch, `YOLOX_URL` release 0.1.1rc0, `YOLOX_SHA256` pinned, ~35 MB), `sam_paths()` (rembg's `SamSession.download_models(sam_model="sam_vit_b_01ec64")`, rembg's checksums, ~360 MB), `downloaded()` (unused helper) |
| `detect.py` | YOLOX-s ONNX on CPU: letterbox, grid decode, NMS | `VehicleDetector.detect(image, min_score)` -> all 80 COCO classes; `Detection.vehicle` = car/motorcycle/bus/truck; `letterbox`, `decode`, `nms`, `vehicles_from_raw` |
| `instance.py` | SAM ViT-B encoder/decoder (rembg's ONNX pair) on CPU | `BoxSegmenter.set_image(img)` (encoder once), `.mask(box)` -> (bool mask, score) per box |
| `isolate.py` | Target choice, removal logic, debug overlay; all thresholds | `CarIsolator.isolate(image, raw_alpha)` -> `IsolationResult`; `plan_removal` (pure), `pick_target`, `agrees`, `maybe_part`, `free_edges`, `overlay` |

## How it works
Called only from `change_background.main()` when isolation is on: pass 1 `precompute_raw` (BiRefNet raw alpha for every
photo) -> `segmenter.release()` -> `CarIsolator(min_score, remove_objects, threads or 4)` -> per photo
`CarSegmenter.finish(image, raw, isolator, trace)` -> `isolator.isolate` -> `clean_alpha(raw, edge_shrink, remove, glass_ok)`.

`isolate()`:
1. `blob` = `select_main_car(fill_holes(raw >= 0.5))`, the same main-blob rule `clean_alpha` uses (empty -> `no-object`).
2. YOLOX on the full photo. `pick_target`: the vehicle box with the best (share of blob in box x box fill); share
   < `MIN_SHARE` 0.5 -> `no-vehicle`.
3. Crop = blob box U target box + `CROP_PAD` 8 %. SAM encodes the crop once; `mask(target)` = T. `agrees`: >= 60 % of T on
   the blob and T covers >= 40 % of the blob, else `unsure`.
4. Other vehicles in the crop (not `maybe_part`, not IoU >= 0.7 with T) get SAM masks. Evidence for other pieces: boxes
   of every other detection (any class; not target duplicates with box IoU > 0.5, not possible parts) and
   `free_edges` (3 px photo-border bands on crop sides that touch the border where T does not).
5. `plan_removal` (radii scale with the photo's longer side: `EDGE` 0.003, `OBJECT_REACH` 0.012): `remove` = other
   vehicles grown by the edge band, minus T's eroded core; with `remove_objects`, a blob piece outside T is removed only
   if it reaches past `OBJECT_REACH`, is >= `MIN_OBJECT` 0.2 % of the blob, AND has evidence (>= 50 % in an object box,
   or touches a free edge). `glass_ok` = T grown by the band.
6. In `segment.py`: `remove` zeroes the raw alpha before thresholding; holes are filled only if >= 60 % on `glass_ok`
   (`GLASS_SHARE`). If that leaves no car, `finish` falls back to plain `clean_alpha` and sets status `unsure`.

`CarMask.isolation = result.summary()` -> results.csv `isolation` (`text()`) and `isolate_s`; `segment_s` includes the
first pass. `segmenter.last_isolation` feeds `overlay()` -> debug `04_cutout_isolation.jpg`.

## Invariants & gotchas
- Mask only: never touches car pixels. Statuses `no-vehicle` / `unsure` / `no-object` leave `remove` / `glass_ok` as
  `None`, so that photo gets exactly the plain cut-out. The status shows in the `isolation` column, not in `flags`.
- Off = bit-identical: `change_background` imports this package lazily, only when enabled; `finish(isolator=None)` is plain
  `clean_alpha`.
- Never remove T's inside. Never remove a non-vehicle piece on shape alone: SAM sometimes drops a spoiler or antenna
  (`sample_vid` 0022). YOLOX calls door mirrors "car" (`Final_zoomed_car` 0002, 0036): `maybe_part` (`PART_INSIDE` 0.8,
  or `PART_SMALL` 0.03 with `PART_TOUCH` 0.3) keeps such boxes out of both removal and evidence. Re-check those frames
  before loosening any threshold. When unsure, change nothing.
- Memory: BiRefNet and SAM are never loaded together (16 GB laptop: together they page). Build `CarIsolator` only after
  `segmenter.release()`; `raw_alpha` no longer works after that, `finish` does. The pass-1 cache is uint8 only where
  exact (rembg's mask is 8-bit); `finish` takes uint8 or float.
- Preprocessing must match the exports: YOLOX = 640x640 BGR, letterboxed top-left with grey 114, not normalised; SAM =
  RGB float fitted top-left into 1024x684, not normalised, box as point labels 2/3 plus a -1 padding point. Get either
  wrong and the masks degrade silently. Boxes are photo px; SAM works on the crop (`local()` shifts them).
- `models.py` depends on rembg 2.0.86 internals (`BaseSession.rembg_home`, `SamSession.resolve_existing` /
  `download_models`). Files go under `<rembg home>/models/` (default `%USERPROFILE%\.rembg\models\`, YOLOX in `yolox/`);
  first use needs the network.

## How to extend
- Tune a threshold: the constants at the top of `isolate.py`. Check debug `03`/`04` pictures on the frames named above,
  then `tests/test_car_isolation.py`.
- Config option: `ISOLATION_KEYS` + `parse_isolation` in `change_background.py`, pass it to `CarIsolator` in `main()`,
  the `[car_isolation]` block in `config.toml`, README, `test_config_isolation_block`.
- Swap a model: anything with `detect(image, min_score) -> list[Detection]`, or with `set_image(img)` + `mask(box) ->
  (bool HxW, score)`, can be injected via `CarIsolator(detector=, segmenter=)` (the tests do this). Fetch its weights in
  `models.py` with a pinned hash.
- New evidence rule: add an optional bool-mask argument to `plan_removal` and keep it pure (no models).
- New status: `IsolationResult.status` + `text()`; it reaches the console and results.csv. Update README's text.

## Tests
`tests/test_car_isolation.py` (fake detector + fake SAM, no download): touching neighbour removed, car inside kept, window
filled but mirror gap open, evidence rule (object box, free edge, spoiler kept), mirror-as-car kept, `no-vehicle` /
`unsure` change nothing, summary on `CarMask`, two-pass uint8 cache equals one pass, YOLOX decode/NMS, config block,
shipped config off. `tests/test_debug_images.py` checks no `04_` picture when off.
