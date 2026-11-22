# video_pipeline: video → sampled frames → plates masked → background changed (room mode) → mp4

## Purpose
Orchestrator only, with no README and no venv of its own. It runs on background_changer's venv (it needs just cv2, numpy and `tomllib`) and starts each tool as a subprocess with that tool's own `.venv` python.

## Files
| File | Role | Key types / entry points |
|---|---|---|
| `make_video.py` | Four-stage driver with per-stage resume | `main()`, `load_config()`, `extract()`, `run()`, `py()`, `fit_to_canvas()`, `images()` |
| `config.toml` | Settings | required: `video` (`../frame_view_type/sample_vid.mp4`), `car` (`Maruti Suzuki Baleno 2022`, a `carModel` in `background_changer/cars.json`), `background` (`background_1`). Optional (code default in brackets; values in `config.toml`): `fps` [5], `output_fps` ["source"], `max_seconds` [0; 10 in config], `max_frames` [0], `brightness_smoothing` [5], `output` ["output"] |

Run it with `..\background_changer\.venv\Scripts\python make_video.py`. `run.log` and `output/` are local and gitignored.

## How it works
The work dir `W` is `<output>/<video stem>/`.
1. **Extract** to `W/frames/00001.jpg…` (cv2, JPEG q95).
   - The sampling step is `src_fps/fps`. Every frame is taken if `fps ≤ 0` or `fps ≥ src_fps`. If the video reports fps 0, `src_fps` falls back to 30.
   - Extraction stops at `max_seconds` of video or after `max_frames` sampled frames.
   - Writes `W/frames.done` containing `"<n> <src_fps>"`.
   - **Skipped** only if frames exist **and** `frames.done` exists; `src_fps` is then read back from the marker. Otherwise the old frames are deleted and extraction runs again.
2. **Mask plates.** **Skipped** if `W/masked/frames/` holds as many images as `W/frames/`. Otherwise it runs:
   `number_plate_detection\.venv\Scripts\python.exe number_plate_detection\mask_plates.py W\frames --out W\masked --overlay --auto-brightness --brightness-smoothing <brightness_smoothing>`
   Every other plate setting comes from `number_plate_detection/config.toml`. Output goes to `W/masked/frames/` plus `detections.csv`.
3. **Change background.** **Skipped** if every frame stem has a result in `W/background/frames/<background>/`. Otherwise it runs:
   `background_changer\.venv\Scripts\python.exe background_changer\change_background.py W\masked\frames --out W\background --car <car> --background <background> --background-view room`
   Every other setting comes from `background_changer/config.toml`. Output: `W/background/frames/<background>/<stem>.webp`, plus `results.csv` and `cutouts/` in `W/background/frames/`.
4. **Assemble.** Always runs, and overwrites `W/<stem>.mp4`.
   - Each extracted frame, in order, uses its own result. If it has none, it reuses the previous result; leading gaps take the first result.
   - The canvas is the first result's size, rounded down to even. Every frame goes through `fit_to_canvas`, and the video is written with fourcc `mp4v`.
   - `output_fps = "source"` gives `min(fps, src_fps)`, or `src_fps` if `fps ≤ 0`, so the duration matches the source. A number is used as-is.

`run()` prints the full command line and exits with `stage failed: <script>` on a non-zero return code. `py(tool)` returns `tool/.venv/Scripts/python.exe` if it exists, and otherwise falls back silently to `sys.executable`.

## Invariants & gotchas
- **Frames named `00001.jpg` carry no view label.** Room mode then looks for `frame_view_type/output/frames/results.csv`, which doesn't exist, so every frame is flagged `no-view-label` and yaw comes from the fit alone. That is expected: frame_view_type is not part of this pipeline.
- **`fit_to_canvas` is the only image operation allowed after background_changer.** It is one uniform `INTER_AREA` resize of the whole finished frame, plus a black letterbox. Never add colour, sharpening or any other per-car edit here: car pixels are evidence ([../CLAUDE.md](../CLAUDE.md), rule 1; [../background_changer/CLAUDE.md](../background_changer/CLAUDE.md)). It also enlarges results smaller than the canvas, and an odd-sized first result is shrunk by up to 1 px.
- **Resume is per stage, not per frame, and the skip checks never record settings:**
  - `frames.done` ignores `fps`, `max_seconds`, `max_frames` and the video's content. After changing any of them, delete `W/`. Deleting only the marker re-extracts frames, but stage 2 compares counts only, so it can reuse stale masked frames.
  - Changing plate settings or `brightness_smoothing` needs `W/masked/` deleted.
  - Changing `car` needs `W/background/` deleted. A new `background` name gets a new folder automatically.
  - Stage 3's check needs a result for **every** frame, but background_changer skips frames it can't place. `run.log` shows 11 of 50 skipped for "car cut off at the bottom". So in practice stage 3 reruns in full on every invocation; that took 1046 s for 50 frames in `run.log`.
- `config.toml` is not validated. Unknown or misspelled keys are silently ignored and the default is used (a typo in `max_seconds` means the whole video). The file is read as `utf-8`, so a BOM fails to parse.
- `background` is a single name. change_background accepts repeated `--background`, but stages 3 and 4 assume one result folder.
- The venv layout `.venv\Scripts\python.exe` is Windows-only. If a tool's venv is missing, the tool runs under the current interpreter. background_changer's venv lacks `open_image_models`, so stage 2 then fails.
- Uses `cv2.imwrite`/`cv2.imread` with `str` paths, which break on non-ASCII Windows paths. The other tools use imdecode/imencode.
- Memory: change_background was observed committing about 17 GB. Never run make_video next to another change_background run or pytest.
- **Timing:** the `config.toml` comment says the room fit costs about 25-40 s per frame. The only recorded run (`run.log`, 464x832 `sample_vid`, 50 frames) took 1046 s for stage 3, about 21 s per frame (11.4 s cut-out + 10-13 s fit). Stage 2 took 3.2 s.

## How to extend
- **New config key:** add a `cfg.setdefault(...)` (or a required-key check) in `load_config`, use it in `main()`, and comment it in `config.toml`. To pass a tool option through, append the flag to that stage's command list in `main()`. Check the flag exists in [../number_plate_detection/mask_plates.py](../number_plate_detection/CLAUDE.md) or `../background_changer/change_background.py` first. Remember that a setting change does not invalidate that stage's output.
- **New stage:** give it its own subfolder under `W`, a skip check based on its outputs, and `run([py(TOOL_DIR), str(TOOL_DIR / "script.py"), ...])`.
- **Several backgrounds:** loop stages 3 and 4 over the names, giving each its own `result_dir` and mp4.

## Tests
None. The engineer write-up is [../docs/video_pipeline/README.md](../docs/video_pipeline/README.md).
