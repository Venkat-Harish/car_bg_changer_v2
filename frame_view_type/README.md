# Car side classifier for videos

Takes a video of a car, grabs one frame every N seconds (default 2), works out which side of the car each frame shows, and saves the frame with that side in its file name.

The 8 sides: `front`, `front-right`, `right`, `back-right`, `back`, `back-left`, `left`, `front-left`. "Left" and "right" are the **car's own** sides (left = driver side on a left-hand-drive car).

The model is [`veerupandey/view_classifier`](https://github.com/veerupandey/view_classifier) (ConvNeXt-Small, MIT code). It runs on CPU, so no GPU is needed. Research notes are in [`docs/`](docs/README.md).

## Setup (once)

Needs Python 3.11+, Git and Git LFS. In PowerShell, from this `frame_view_type` folder:

```powershell
git clone https://github.com/veerupandey/view_classifier vendor/view_classifier
git -C vendor/view_classifier checkout 6514ec2b2bee695c45c2152d95f44ea1bf4ca7e0
git -C vendor/view_classifier lfs pull
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Check that `vendor\view_classifier\models\joint_v2_best.pt` is about 198 MB. If it is a tiny file, `git lfs pull` did not run.

## Run

1. Edit [`config.toml`](config.toml): set the video, the output folder and the interval.
2. From this folder, run:

```powershell
.venv\Scripts\python classify_video.py
```

### Settings in `config.toml`

| Setting | Default | Meaning |
|---|---|---|
| `input` | `"Final_zoomed_car.mp4"` | Video to process |
| `output` | `"output"` | Base folder for results. Each video gets its own subfolder: `output\<video name>\`. |
| `every_seconds` | `2` | Take one frame every N seconds (`3` = every 3 s, `0.5` = twice a second) |
| `checkpoint` | `"vendor/view_classifier/models/joint_v2_best.pt"` | Model weights (normally leave as is) |
| `pick_sides` | `false` | Also keep the best frame for each of the 8 sides in `output\<video name>_sides\` (see below) |
| `pick_min_prob` | `0.3` | A side whose best frame scores below this counts as not in the video and is left out |

- Relative paths are relative to the `frame_view_type` folder.
- For a full Windows path, use single quotes so the backslashes are kept: `input = 'D:\videos\car.mp4'`. Forward slashes in double quotes also work: `"D:/videos/car.mp4"`.

### One-off overrides

Command-line options win over `config.toml`, which is handy for a quick run without editing the file:

```powershell
.venv\Scripts\python classify_video.py D:\videos\other_car.mp4 --every 3
```

| Option | Overrides |
|---|---|
| `<video>` | `input` |
| `--every N` | `every_seconds` |
| `--out <folder>` | `output` |
| `--ckpt <file>` | `checkpoint` |
| `--pick-sides` | `pick_sides` (turns it on) |
| `--config <file>` | Use a different config file |

## Output

```
output\Final_zoomed_car\
  0000.0s_front_92pct.jpg
  0006.0s_front-right_93pct.jpg
  0010.0s_right_95pct.jpg
  0016.0s_back-right_93pct.jpg
  0020.0s_back_91pct.jpg
  0024.0s_back-left_93pct.jpg
  0028.0s_left_94pct.jpg
  0034.0s_front-left_93pct.jpg
  ...
  results.csv
```

- File name = **time in the video** + **side** + **model confidence**. Files sort in video order.
- Images are the original full-size frames.
- `results.csv` has one row per frame: file, time, side, angle (0° = front, clockwise, 90° = car's right), confidence, and the probability of each of the 8 sides.
- Running again with the same video writes into the same folder, overwriting files with the same name.

### One frame per side (`pick_sides`)

With `pick_sides = true` (or `--pick-sides`) the run also fills `output\<video name>_sides\` with the best frame for each of the 8 sides: the frame with the highest probability for that side, each frame used once. The frames keep their own names (so the side in the name is still the model's own answer), and its `results.csv` has two extra columns, `picked_for` and `picked_prob`. A side no frame reaches `pick_min_prob` for is not in the video: it is listed in `missing_sides.txt` and on the console, never filled with a weaker or repeated frame. The folder is rewritten on every run.

Sample more often for more choice, e.g. `--every 0.5 --pick-sides`, then point number_plate_detection at `output\<video name>_sides`. On `sample_vid.mp4` this finds 6 sides: the walk-around never shows a straight left or right side (best probability 13%).

## Speed on CPU

Measured on this PC (Intel i5-1240P, no GPU) with `Final_zoomed_car.mp4` (38 s, 4K, 30 fps):

| Interval | Frames saved | Total time |
|---|---|---|
| every 2 s | 20 | 47 s |
| every 5 s | 8 | 35 s |

- About 19 s of that is reading through every frame of the 4K video, whatever the interval.
- Each sampled frame then takes about 0.8 s (pad the 4K frame, run the model) plus the JPEG save.
- Lower-resolution videos will be much faster.

## Known limits

- **No car detector crop.** The whole frame is used. Accuracy is best when the car fills most of the frame.
- **Each frame is judged on its own.** There is no smoothing across neighbouring frames, so an occasional frame can be wrong, sometimes by a lot (e.g. front-left called back-right).
- **Every frame gets one of the 8 sides,** even close-ups or interiors. A low confidence in the file name is the warning sign.
