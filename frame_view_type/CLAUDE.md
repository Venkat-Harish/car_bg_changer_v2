# frame_view_type: sample a car video every N seconds and label each frame with the side of the car shown

## Purpose
First tool in the chain. It takes one frame every `every_seconds` and runs an 8-way side classifier on it: [veerupandey/view_classifier](https://github.com/veerupandey/view_classifier), ConvNeXt-Small, torch on CPU. Each frame is saved with its side and confidence in the file name, and every frame gets a row in `results.csv`.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `classify_video.py` | CLI entry | `main()`, `load_config()`, `load_model()`, `sample_frames()`, `classify()`; `SIDE_NAMES`, `DEFAULT_EVERY` 2.0, `DEFAULT_CKPT`, `CONFIG_KEYS` |
| `config.toml` | `input` (`sample_vid.mp4`), `output` (`output`), `every_seconds` (2), `checkpoint` (`vendor/view_classifier/models/joint_v2_best.pt`) | — |
| `requirements.txt` | `torch==2.9.1`, `torchvision==0.24.1` (match the vendor pins), plus unpinned `pillow`, `numpy`, `opencv-python-headless` | — |
| `README.md` | Setup (clone, pinned commit, `git lfs pull`), run, output, speed | — |
| `docs/` | Research notes from before the build (01-05). Index: [docs/README.md](docs/README.md). Not code docs. | — |
| `vendor/view_classifier/` | Gitignored third-party clone at commit `6514ec2b2bee695c45c2152d95f44ea1bf4ca7e0`. Checkpoint `models/joint_v2_best.pt` (~198 MB, git LFS). | Only what is listed under How it works is imported |
| `sample_vid.mp4`, `Final_zoomed_car.mp4` | Local test videos (gitignored `*.mp4`). `sample_vid` is 464x832 portrait; `Final_zoomed_car` is 4K, 38 s. | — |

## How it works
1. `load_config` reads the config and rejects unknown keys. `input`, `output` and `checkpoint` are resolved against the config folder, and `every_seconds` must be a number. CLI flags override config values: positional `video`, `--every`, `--out`, `--ckpt`, `--config`. CLI paths are relative to the cwd.
2. `load_model` rejects a checkpoint that is missing or under 1 MB (an LFS pointer) and tells you to run `git -C vendor/view_classifier lfs pull`. It then loads with `ConvNeXtViewClassifier(pretrained=False)` + `torch.load(map_location="cpu", weights_only=False)["model_state"]`.
3. `sample_frames` calls `grab()` on every frame and `retrieve()` only when `idx >= round(k·every·fps)`. A frame's time is `idx/fps`.
4. `classify` builds the model input with `square_reflect_crop(img, bbox=None)`: the whole frame, padded to a square by reflection, with no car crop. `eval_tensorize` then resizes to 224x224 with ImageNet normalisation. The model returns logits and a sin/cos head:
   - side = argmax mapped through `SIDE_NAMES`
   - conf = top softmax probability
   - angle = `sincos_to_deg`: 0° = front, clockwise, 90° = the car's right. It comes from a separate head, so it can disagree with the side.
5. Frames are written to `output/<video stem>/<t:06.1f>s_<side>_<round(conf·100)>pct.jpg` (PIL RGB, quality 95, full resolution), e.g. `0012.0s_front-right_93pct.jpg`.
6. `results.csv` is written as UTF-8 with columns `file, time_s` (2 dp), `side, angle_deg` (1 dp), `confidence` (3 dp), then `prob_front, prob_front-right, prob_right, prob_back-right, prob_back, prob_back-left, prob_left, prob_front-left` (3 dp, in `POSE_CLASSES` order).

Vendor names used: `view_classifier.labels.{POSE_CLASSES, sincos_to_deg}`, `view_classifier.model.ConvNeXtViewClassifier`, `view_classifier.transforms_cfv.{eval_tensorize, square_reflect_crop}`. `vendor/view_classifier` is inserted at `sys.path[0]`.

Downstream:
- [number_plate_detection](../number_plate_detection/CLAUDE.md) reads `../frame_view_type/output/sample_vid` and keeps the file names.
- background_changer ([../background_changer/CLAUDE.md](../background_changer/CLAUDE.md)) parses the names with `NAME_RE` in `bg_changer/probe.py`. In room mode it also reads `frame_view_type/output/<images folder name>/results.csv`, keyed by file stem, using the `prob_<label>`, `angle_deg` and `time_s` columns.
- video_pipeline does not run this tool.

## Invariants & gotchas
- **The names are a contract with background_changer.** The `SIDE_NAMES` values (`front, front-right, right, back-right, back, back-left, left, front-left`), the file-name pattern `<float>s_<label>_<int>pct` and the `prob_<label>` columns must match `LABELS`/`NAME_RE` in `bg_changer/probe.py` and `read_view_results` in `car_fit_to_bg/pose.py`. If they drift, the view prior is lost silently (flag `no-view-label`). Left and right are the car's own sides.
- The output folder is named after the video stem. Downstream tools keep that folder name, and background_changer finds `results.csv` by it.
- **Reruns leave stale frames.** A rerun overwrites files with the same name but never deletes old ones. After a run with a different `every_seconds` or checkpoint, stale JPGs stay in the folder and get processed downstream, while `results.csv` lists only the new run. Clear `output/<stem>/` first.
- Times in names have 0.1 s resolution, so `every_seconds` below 0.1 can produce clashing names.
- README says the default input is `Final_zoomed_car.mp4`, but `config.toml` uses `sample_vid.mp4`, so the README is stale. Its "Python 3.11+" is correct (`tomllib`).
- `config.toml` is parsed as bytes by `tomllib.load`, so a UTF-8 BOM (Notepad, PowerShell `Out-File`) is a parse error. number_plate_detection reads `utf-8-sig` instead.
- `torch.load(weights_only=False)` unpickles arbitrary objects, so only load trusted checkpoints. torch uses its default thread count; there is no setting for it.
- README speed on the i5-1240P: about 0.8 s per sampled 4K frame, plus about 19 s to decode the whole 38 s 4K video.
- Every frame gets one of the 8 sides, even interiors and close-ups. Low confidence is the only warning sign. There is no temporal smoothing.
- Never edit `vendor/`: it is a gitignored clone, recreated from the README steps.

## How to extend
- **New config key:** add it to `CONFIG_KEYS` (and to `PATH_KEYS` for a path), validate it in `load_config`, then add the argparse flag and the `args.x if args.x is not None else cfg.get(...)` merge in `main()`. Comment it in `config.toml` and update the README tables.
- **New `results.csv` column:** add it to the row dict in `main()`. background_changer's `read_view_results` ignores unknown columns.
- **Car crop or temporal smoothing:** change `classify()` or the main loop. `square_reflect_crop` accepts a `bbox`. Keep the file names and columns stable.
- **Different model:** it must still produce exactly the 8 `SIDE_NAMES` labels.

## Tests
None.
