# Background changer for car images

Takes a folder of car photos, cuts the car out of each one, and puts it on a studio background from the [`backgrounds`](backgrounds) folder **the way a real car would stand there**:
- on the turntable (background_1);
- on the floor of the grey studio (background_2);
- on the race track (background_3).

The result is a WEBP per photo per background.

**Why it looks real (room mode, the default).** Every photo is looked at on its own:
- **its camera** is measured from the photo's own background lines: how far the phone was tilted and rolled, and its lens;
- **the car's pose** in it is fitted with a 3D outline of the car (its real dimensions from [`cars.json`](cars.json)): the angle it is seen from, how far away and how high the camera was.

The background is then **re-rendered as a 3D room from that same camera**, so floor and car share one viewpoint and the car stands on the floor at any camera angle: close up, from high above, from a portrait phone video. The output keeps the photo's own framing (portrait stays portrait; the car stays where it was in the frame).

**The car itself is not changed.** Its pixels get at most one rigid rotation (to level a photo taken with a tilted phone) and one plain downscale, with no colour change, relighting, retouching, sharpening or generated content, so dents, scratches and smudges stay visible. Windows keep their original glass. The room, shadows and the floor reflection are drawn on the background, never on the car.

The older **flat mode** (`background_view = "flat"`) is still there: it scales and moves the car onto a fixed stage on the background picture as it is (see [Flat mode](#flat-mode)).

It runs on the CPU only: a background-removal model ([BiRefNet](https://github.com/ZhengPeng7/BiRefNet), MIT) runs on ONNX Runtime through [rembg](https://github.com/danielgatis/rembg) (MIT). No GPU and no torch. Research notes are in [`docs/findings.md`](docs/findings.md).

## How to run

1. **Make sure the car is in [`cars.json`](cars.json).** Each car is one entry, dimensions in **centimetres** (width without mirrors), plus its body type:

   ```json
   [
     {
       "carModel": "Mercedes-Benz EQB 2024",
       "length": 468.4,
       "width": 183.4,
       "height": 166.7,
       "WheelBase": 282.9,
       "body": "suv"
     },
     {
       "carModel": "Maruti Suzuki Baleno 2022",
       "length": 399.0,
       "width": 174.5,
       "height": 150.0,
       "WheelBase": 252.0,
       "body": "hatch"
     }
   ]
   ```

   Add a new car as another `{ … }` entry in the array. `body` is `"suv"`, `"hatch"` or `"sedan"`: the shape of the 3D outline the photos are fitted with. Without it the shape is guessed from the dimensions (flagged `body-guessed`). (The Baleno is the car in `sample_vid`: the 2022 Indian-market model, which is 1500 mm tall; the Australian-market one is 1470 mm.)

2. **Open [`config.toml`](config.toml) and set the top part:**

   ```toml
   images_folder      = "../number_plate_detection/output/Final_zoomed_car"   # your car photos
   car                = "Mercedes-Benz EQB 2024"                              # its carModel in cars.json
   backgrounds_folder = "backgrounds"                                         # where the backgrounds are
   background         = "background_1"                                        # which one to use
   output_folder      = "output"
   ```

   `background` can be:
   - one name: `"background_2"`;
   - several: `["background_1", "background_3"]`;
   - `"all"`.

   Names are file names without the extension.

3. **Save, then run** in PowerShell from this `background_changer` folder:

   ```powershell
   .venv\Scripts\python change_background.py
   ```

4. **The results are in** `output\<images folder name>\<background name>\`.

Use `.venv\Scripts\python`, not plain `python`. The plain one doesn't have the packages, and the script tells you so if you try.

The photos' file names should carry the view label from `frame_view_type`, e.g. `0012.0s_right_95pct.jpg`. In room mode the angle is fitted, and the label (with the classifier's own results, when `frame_view_type\output\<folder>\results.csv` exists) is only a hint for which way round the car is. Without either, the outline alone decides (flagged `no-view-label`).

### Setup (once, already done on this PC)

Needs Python 3.11+. From this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The first run downloads the model into `%USERPROFILE%\.rembg\models\`. That is about 220 MB for the default `birefnet-general-lite`, and about 970 MB for `birefnet-general`. After that it works offline.

## How the car is placed (room mode)

The run has three stages:
1. **Cut out every photo** and measure its background lines.
2. **Settle the lens** for the folder: photos of the same size with clear lines (at least 3, e.g. one video's frames) share one focal length; the others fall back to their own lines, the photo's EXIF, or a typical phone lens (flagged).
3. **Fit and place** each photo on every chosen background.

For each photo (code in [`car_fit_to_bg/`](car_fit_to_bg), the 3D rooms in [`room3d/`](room3d)):

1. **The camera.** Straight edges in the photo's background (buildings, kerbs, poles; the car itself is masked out) give the camera's roll and pitch. A photo with too few lines borrows them from neighbouring video frames, or from the car's outline (flagged).
2. **The car's pose.** A 3D outline of the car is turned through every angle and fitted to the cut-out: the view angle, the camera's distance and height. The view label, the classifier and the red tail lamps decide which way round the car is when the outline alone can't (a car looks almost the same from the front and the back). A fit that is unsure, or that disagrees with the label by more than 24°, is listed in `results.csv`.
3. **Level.** If the phone was rolled and the background lines show by how much, the car is rotated once, so its floor is level in the output; a car standing on a sloping road is levelled to its own floor. With too few lines the car is left as it is and the room is tilted to match it instead (flagged `roll-from-silhouette`).
4. **Tyres.** The tyre bottoms are found on the cut-out's outline, and the camera is nudged (never the car) so the 3D contacts sit exactly on them.
5. **The room.** The background is re-rendered from the photo's camera:

   | Background | Its 3D room |
   |---|---|
   | `background_1` (white studio, turntable) | A white box measured from its LED strips; the turntable is floor texture |
   | `background_2` (grey studio) | Curved walls along the measured wall/floor junction, and its ceiling |
   | `background_3` (race track) | The asphalt, and the barriers, trees and sky as a backdrop standing at the barriers' bases |

   The car's size on the stage is the same as in flat mode (and `[scene.<name>] size` still scales it). Parts of the output the background picture never showed (above its top edge, below its bottom edge, a ceiling) are filled smoothly from what is there: `invented_pct` in `results.csv`, flagged `invented` when there is a lot of it or any near the car.
6. **Grounding** is drawn on the floor in the room: the soft shadow of the car's footprint, a darker one under the body, contact patches at the found tyre bottoms and a contact line along the car's lower outline; on the glossy turntable, a faint reflection. Then the car is pasted on top.

**Output size.** Each output has the photo's own shape and framing, at the photo's size or smaller (a few pixels larger only when levelling a car near the photo's edge would otherwise cut part of it off):
- at most `max_output_size` (default 1920 px) on the long side;
- with `keep_background_sharp = true` (default), never so large that the background picture would be magnified where the car stands. 4K photos then come out about 900-1450 px wide, with the car about as detailed as in flat mode, and a close-up or a photo from high above can come out smaller than the photo, by a slightly different amount on each background. `false` gives outputs as large as the photo and `max_output_size` allow, with a slightly soft floor near the car (flagged `soft-floor`).
- A background without a 3D room is placed flat and keeps flat mode's sizes.

So most of `sample_vid`'s 464×832 frames keep that size (its two closest shots, 0006 and 0020, come out at 75-79% and 92-97%), and `Final_zoomed_car`'s 3840×2160 frames come out 907-1442 px wide.

**Cut-off photos:**
- A car that runs off one side of the photo is framed like the photo: its cut edge stays on the output's border, and the room fills the rest. `cut_frames = "skip"` leaves these photos out instead.
- A car cut off at the bottom, or on two sides, is not placed: there is nothing to stand it on. `results.csv` says so in `note`.

**Unsure fits are placed, flagged.** A photo whose 3D fit fails (an unusual car shape, a car mostly hidden) is still placed with its label's angle, flagged `fit-failed`. `[fit] on_fail = "skip"` leaves these out instead.

### Your own backgrounds

Drop the image into the `backgrounds` folder and put its name in `background`.

**Without scene code**, it gets a generic stage. Tell it where the car stands, or it uses the defaults (centre column, 72% down, side view 66% of the width):

```toml
[scene.my_showroom]
anchor = [960, 760]  # floor point (px) under the middle of the car
side_fill = 0.66     # a side view's length as a share of the picture width
horizon_y = 520      # the floor's horizon row
floor = "matte"      # or "glossy"
```

With `horizon_y`, room mode builds a rough 3D room from it (flagged `generic-room`): the floor up to the horizon, everything above it a far backdrop. Without it, the car is placed flat on that background, with its fitted angle (flagged `no-room`).

**For the best result**, measure the picture like the three built-in ones: copy `bg_changer/scenes/background_2.py` into a new module (anchor, stage size, camera floor model, floor mask, shadow style), register it in `bg_changer/scenes/__init__.py`, and add a room builder for it in `car_fit_to_bg/rooms.py`.

## Flat mode

`background_view = "flat"` places the car as it is, through each background's scene code in [`bg_changer/scenes/`](bg_changer/scenes). The output is the background's size (1920×1080), or smaller for small photos.

| Background | Stage (where the car's centre stands) | Size | Floor |
|---|---|---|---|
| `background_1` (white studio, turntable) | The turntable's true centre, where its seams meet (row 764; not the rim ellipse's centre, which is 1.4 m nearer) | A side view spans ~73% of the turntable | Glossy: faint reflection plus soft shadow |
| `background_2` (grey studio) | Floor centre, row 780 | A side view spans ~65% of the width | Matte concrete: soft shadow |
| `background_3` (race track) | Open asphalt, row 685, in front of the barriers | Matches the 0.8 m barriers at that depth | Matte asphalt: warm-neutral shadow |

For each photo:

1. **The car's angle** comes from the view label: front 0°, front-right 45°, right 90°, … The 8 labels are 45° apart, so the angle can be off by up to ~24°.
2. **A 3D outline of the car** is built from its dimensions and viewed at that angle through a fixed "capture camera": about 4.7 m away, looking down 16°, as the Mercedes walkaround was shot. This gives the car's height on the picture and where its lowest point sits.
3. **The photo is scaled down** so the car has that height (never enlarged), and moved so that its lowest point lands on the stage row and its middle stands on the stage centre.
4. **Grounding** is drawn on the background first (a contact line under the tyres, the shaded floor under the body, a soft footprint shadow, the turntable reflection). Then the car is pasted on top, unchanged.

Flat mode works well only for photos shot like that walkaround. A photo from much higher up, much closer or with a rolled phone (like `sample_vid`) keeps its own viewpoint on a floor seen from about 9°, so the car can look like it floats; room mode fixes exactly that.

**Cut-off photos:** `cut_frames = "anchor"` places the whole car where it would stand, so the missing part shows as a straight cut (`"frame"` means the same in flat mode); `"centre"` centres the visible part; `"skip"` leaves them out. A car cut off at the bottom is placed by its intact roof; one cut off on two sides is not placed.

**Big cars:** if a car's dimensions would make some view stick out of the picture or past the turntable's front edge (a tall van, say), it is made a little smaller on that background, with a warning.

### Small photos: `background_scaler`

The car is never enlarged. A frame from a low-resolution video holds a car smaller than a full-size background's stage needs. For example, `sample_vid` is a 464×832 portrait phone video with the car 210-474 px tall, while background_1's stage wants 580-650 px.

So in flat mode the [`background_scaler`](background_scaler) folder shrinks the **background** instead, until the car fills the stage at its own resolution. It shrinks the picture and every measurement of its scene together. How small is set by `background_scale` in `config.toml`:

| `background_scale` | Background size | Use it when |
|---|---|---|
| `"folder"` (default) | One size for all photos of the folder, set by the photo with the smallest car | You want every output the same size |
| `"photo"` | Each photo gets the size its own car needs, so the car is pasted pixel for pixel | You want the most detail in every photo |
| `"off"` | Always full size. A small car looks small on it (flagged `scale-capped`). | |

- Photos with enough pixels (4K frames, say) keep the full-size background in every mode.
- `background_min_width` (default 640 px) stops a background shrinking further.
- In room mode this only applies to a background without a 3D room.

## Keeping only the car: `car_isolation`

The cut-out model marks "the salient object", not "this car". Whatever touches the car in the photo can come along: a car parked behind it (seen past the mirror or above the roof), a person, a gate. It also fills every gap inside the outline, so the old background seen between the mirror and the door stays in. `car_isolation` (its own folder, off by default) fixes that:

1. **YOLOX-s** boxes every vehicle in the photo. The target is the box that best matches the cut-out's main blob.
2. **SAM ViT-B** draws the target car's mask and the masks of the other vehicles near it (on a crop around the car).
3. The other vehicles are removed from the cut-out's mask, never the inside of the target car. With `remove_objects`, any piece that sticks out of the target car by more than about 1.2% of the photo's long side is removed too.
4. Only holes that lie on the target car (windows) are filled; mirror-to-pillar gaps, spoiler gaps and open spokes stay see-through.

**What it never does:** change a car pixel. It only decides which pixels belong to the car. When no vehicle is found, or SAM's car does not match the cut-out, the photo keeps today's cut-out and `results.csv` says why (`no-vehicle`, `unsure`).

**Turning it on:** `[car_isolation] enabled = true` in `config.toml`, or `--isolate` for one run. With `debug_mode = true`, `debug\<photo>\04_cutout_isolation.jpg` shows SAM's car (green outline), what was removed (red) and the other vehicles (orange boxes).

**Evidence before removing anything but a vehicle.** SAM's car mask sometimes leaves out a car part (a spoiler, an antenna), so a piece sticking out of it is removed only if the detector also found another object there (a person, a scooter, any of the 80 COCO classes) or it enters from a photo edge where the car is not cut off (the cameraman's hand). A spoiler meets neither and stays.

**Memory and speed.** BiRefNet and SAM never sit in memory together: with isolation on, every photo is cut out first, BiRefNet is freed, then YOLOX and SAM load. (Loaded together on this 16 GB laptop, BiRefNet slowed from ~15 s to ~50 s per photo.) The step adds about 10-13 s per photo.

**Models:** YOLOX-s (Megvii, Apache-2.0, 35 MB) and SAM ViT-B (Meta, Apache-2.0, about 360 MB, the ONNX pair rembg publishes). Both download once into `%USERPROFILE%\.rembg\models\` and are checked against pinned hashes. No new Python packages.

## Settings in `config.toml`

| Setting | Default | Meaning |
|---|---|---|
| `images_folder` | `"../number_plate_detection/output/sample_vid"` | The car photos: a folder of `.jpg .jpeg .png .bmp .webp` images (not subfolders), or one image |
| `car` | `"Maruti Suzuki Baleno 2022"` | Which car is in the photos: its `carModel` in the car file |
| `backgrounds_folder` | `"backgrounds"` | Folder with the background images |
| `background` | `"background_1"` | Which background(s): a name, a list of names, or `"all"` |
| `output_folder` | `"output"` | Base folder for results. Each images folder gets its own subfolder. |
| `background_view` | `"room"` | `"room"`: the background re-rendered as a 3D room from each photo's camera. `"flat"`: the car scaled and moved onto the picture as it is. |
| `max_output_size` | `1920` | Room mode: cap on the output's long side (px); `0` = none |
| `keep_background_sharp` | `true` | Room mode: never magnify the background where the car stands (smaller outputs from 4K photos); `false` = as large as `max_output_size` allows |
| `cut_frames` | `"frame"` | Cars cut off by the photo edge: `"frame"` (room: keep the photo's framing; flat: same as `"anchor"`) or `"skip"`; flat mode also `"anchor"` (true position, the slice shows) and `"centre"` |
| `background_scale` | `"folder"` | Flat mode: shrink the background for small photos. `"folder"`, `"photo"` or `"off"` (see [Small photos](#small-photos-background_scaler)) |
| `background_min_width` | `640` | Flat mode: never shrink a background below this width (px) |
| `cars_file` | `"cars.json"` | The car list (JSON array, dimensions in cm, `body` type) |
| `shadow` | `1.0` | Shadow and contact strength, 1 = as calibrated per background, 0 = none (up to 1.5) |
| `debug_mode` | `false` | `true` = a debug picture for every step in `…\debug\<photo>\` (see [Debug pictures](#debug-pictures-debug_mode)); `false` = none |
| `model` | `"birefnet-general-lite"` | Background-removal model. `"birefnet-general"` is about 4× slower and looks the same at this output size (see [findings](docs/findings.md#cutting-the-car-out)). |
| `threads` | `4` | CPU threads for the model. 4 was fastest on this laptop; `0` = automatic. |
| `edge_shrink` | `2` | Pulls the cut-out's soft edge in by this many pixels (of the original photo), to drop a thin fringe of the old background. `0` = off. |
| `quality` | `90` | WEBP quality, 1-100 |
| `save_cutouts` | `true` | Also save each car alone, transparent, at full resolution (lossless WEBP) |
| `[scene.<name>]` | none | Optional per-background tweaks: `size`, `shadow`, `reflection`; for backgrounds without scene code also `anchor`, `side_fill`, `horizon_y`, `floor` |
| `[car_isolation] enabled` | `false` | Keep only the target car in the cut-out (see [Keeping only the car](#keeping-only-the-car-car_isolation)). `false` = runs exactly as before. |
| `[car_isolation] min_score` | `0.35` | Vehicle detector confidence (0-1); lower finds more half-hidden cars behind ours |
| `[car_isolation] remove_objects` | `true` | Also remove non-vehicle things sticking out of the car's outline (a person, a gate) |
| `[fit] focal` | `"auto"` | Room mode: the lens's focal length. `"auto"` = from the background lines (shared by photos of the same size), or a number in px at the photo's own size. |
| `[fit] on_fail` | `"place"` | Room mode: a photo whose 3D fit fails is still placed with its label's angle and flagged (`"place"`), or left out (`"skip"`; this also skips a background whose room is too small for the photo's camera) |

- Relative paths are relative to the config file's folder: the `background_changer` folder for the shipped `config.toml`.
- For a full Windows path, use single quotes so the backslashes are kept: `images_folder = 'D:\photos\car_123'`. Forward slashes in double quotes also work.

### One-off overrides

Command-line options win over `config.toml` for one run:

```powershell
.venv\Scripts\python change_background.py D:\photos\car_123 --background all --car "Mercedes-Benz EQB 2024" --debug
```

| Option | Overrides |
|---|---|
| `<folder or image>` | `images_folder` |
| `--background NAME` | `background`. Repeat it for several, or use `all`. |
| `--car NAME` | `car` |
| `--background-view room\|flat` | `background_view` |
| `--max-output-size PX` | `max_output_size` |
| `--cut-frames frame\|skip\|anchor\|centre` | `cut_frames` |
| `--background-scale folder\|photo\|off` | `background_scale` |
| `--backgrounds-folder <folder>` | `backgrounds_folder` |
| `--out <folder>` | `output_folder` |
| `--debug` | `debug_mode` (turns it on) |
| `--isolate` / `--no-isolate` | `[car_isolation] enabled` |
| `--model`, `--threads`, `--shadow`, `--quality` | The setting of the same name |
| `--config <file>` | Use a different config file |

### Checking without the model

Segmentation is the slow part. Two tools re-use saved cut-outs or masks:

```powershell
.venv\Scripts\python tools\check_room.py sample_vid Final_zoomed_car
.venv\Scripts\python tools\check_room.py sample_vid --per-photo
.venv\Scripts\python tools\check_scenes.py --sheets --debug
```

- `check_room.py` runs room mode on folders under `number_plate_detection\output` (masks cached in `output\check_room\_masks\`), writes contact sheets (photo with its fit, flat result, the three rooms) and an `index.html` to browse them, and checks that the car's pixels are exact and the fitted outline lands on the pasted car. `--per-photo` fits every photo on its own, with no help from the rest of the folder.
- `check_scenes.py` re-places flat-mode cut-outs on every background and checks the numbers.
- `check_flat.py` compares a flat run with the pinned baseline in `output\_baseline_flat` pixel for pixel (see `tests\baseline_flat.toml`).

## Output

```
output\Final_zoomed_car\
  background_1\0000.0s_front_92pct.webp       the car on background 1 (same name as the photo, .webp)
  background_2\...                            (one folder per chosen background)
  cutouts\0000.0s_front_92pct.webp            the car alone, transparent, full resolution
  debug\0000.0s_front_92pct\...               (debug_mode = true) a picture for every step, see below
  results.csv
```

### Debug pictures (`debug_mode`)

With `debug_mode = true` in `config.toml` (or `--debug` for one run), every photo gets a folder `debug\<photo>\` with one picture per step, numbered in the order the steps run. Each picture has a caption strip on top: what the step did, its numbers, and what each colour means. `debug\README.txt` lists them too. Pictures are at most 1600 px on the long side; expect a few seconds more per photo and about 1-3 MB per photo and background. Nothing in a result changes.

| Picture | Step |
|---|---|
| `01_photo` | The photo as read |
| `02_cutout_model` | The cut-out model's raw mask: white = car, black = background, grey = unsure |
| `03_cutout_cleanup` | How that mask became the car's outline: other blobs dropped (orange), posts cut off (red), objects removed by car_isolation (magenta), window holes filled (blue), the soft edge band (yellow), the final outline (green) |
| `04_cutout_isolation` | (car_isolation on) the car SAM found, other vehicles, what was removed |
| `05_cutout_result` | The final mask and the car alone on a checkerboard |
| `06_camera_lines` | Room mode: the camera's roll, pitch and lens from the background's straight lines: segments by direction, vanishing directions, the horizon from the lines (cyan) and from the car's 3D fit (yellow) |
| `07_angle_search` | How the car's view angle was found: the outline match at every 10°, the candidates refined, the winner, the file-name label; the camera's distance, height and angle down |
| `08_outline_fit` | The cut-out's outline (red) and the fitted 3D car's outline (green) on the photo, its tyre contacts and footprint centre |
| `09_outline_compare` | The same two outlines filled: grey = both, red = only the cut-out, green = only the 3D car; below it every candidate angle the same way, with its score |
| `10_level` | Car alignment: the car's bottom line before and after its one rotation, and the angle applied |
| `11_tyres` | Tyre bottoms on the levelled car: predicted, found, and after the contact lock |
| `<background>\12_room` | The background's 3D room re-rendered from the photo's own camera, with the room's edges and the stage point |
| `<background>\13_room_surfaces` | Floor, walls and ceiling, and (red hatching) what the background picture never showed |
| `<background>\14_shadows` | What the shadows and the floor reflection changed |
| `<background>\15_background_ready` | The background just before the car is pasted |
| `<background>\16_placed` | The result with footprint, stage, horizon and tyre marks |
| `<background>\12_flat_placement` | A background without a 3D room: the flat placement's overlay |

Flat mode (`background_view = "flat"`) writes pictures 01-05, and its placement overlays where they always were, in `<background>\debug\<photo>.jpg` (so `check_flat.py` still compares them with the pinned baseline).

`results.csv` has one row per photo per background, and a row with a `note` for every photo that was not placed. Columns in both modes:

| Column | Meaning |
|---|---|
| `out_w`, `out_h` | The output's size |
| `view_label`, `view_conf` | The view label in the file name and the classifier's confidence |
| `yaw`, `yaw_source` | The angle used: `fit`, `label` (room mode: unsure fit, the label fitted as well) or `label-fallback` (failed fit); flat mode: `label` or `shape` |
| `car_x1..car_y2` | Where the car is in the photo |
| `cut_off` | Which sides of the car run off the photo |
| `flags` | Anything worth checking (below) |
| `segment_s` | Seconds spent cutting the car out |
| `note` | Why a photo was not placed, or a reminder about cut-off cars |

Room mode adds:

| Column | Meaning |
|---|---|
| `view_mode` | `room`, or `flat (no room)` on a background without a 3D room |
| `frame_w`, `frame_h`, `crop_x1`, `crop_y1` | The photo's size and where the kept cut-out starts in it |
| `out_scale` | Photo pixels to output pixels (never above 1) |
| `fit_iou` | How well the fitted 3D outline covers the cut-out (1 = exactly); below 0.90 = `weak-fit`, below 0.75 = `fit-failed` |
| `fit_margin` | How much better the chosen angle fits than the car turned round (log-odds; below 2 = `yaw-unsure`) |
| `yaw_change` | Degrees between the fitted angle and the label's (over 24 = `yaw-vs-label`) |
| `roll`, `roll_source`, `pitch` | The camera's roll and pitch to the car's floor, and where the roll came from (`lines`, `neighbours` = nearby video frames, `silhouette`) |
| `car_roll`, `level_deg` | The car's own tilt on a sloping floor, and the rotation applied to the car's pixels |
| `focal_px`, `focal_source` | The lens: `config`, `pooled` (shared by the folder's photos of that size), `lines`, `exif`, `lines-weak`, `prior` |
| `body_source` | `cars.json`, or `guess` |
| `cam_dist_m`, `cam_height_m`, `h_source`, `elev_deg` | Camera distance to the car, its height (`fit`, or pulled toward the `folder`'s or a `default` height when the lines were weak) and how far it looked down |
| `cut_crop_px`, `cut_crop_pct` | How much of a cut-off car the levelled frame trims at its cut edge |
| `tyres_found`, `lock_mode`, `contact_residual_px` | Tyre bottoms found, how the camera was locked to them (`height+pitch`, `height`, `none`), and the remaining miss in output px (empty when no tyre was found) |
| `invented_pct`, `invented_near_pct` | Share of the output the background never showed, overall and around the car |
| `floor_texels` | Background pixels per output pixel at the car (below 1 = magnified, `soft-floor`) |
| `fit_s`, `render_s` | Seconds for the fit (once per photo) and for the background's render |

With `[car_isolation] enabled = true`, both modes add `isolation` (`removed 1 vehicle(s) + 1 object(s)`, `nothing to remove`, or why nothing was changed: `no-vehicle`, `unsure`) and `isolate_s` (its seconds, included in `segment_s`).

Flat mode adds `bg_scale`, `scale`, `px_per_m`, `anchor_x`, `anchor_y`, `src_h`, `target_h`, `target_low_y`, `out_x1..out_y2`, `hw_ratio` and `margin` (the stage scale and centre, where the car landed, and how its shape compares with the 3D outline at the label's angle).

`flags` in room mode, roughly in order of importance:
- `fit-failed`: the 3D outline doesn't match the cut-out, or the camera would be at an implausible height; placed with the label's angle, and only the shadows that follow the car's own outline (the footprint shadow needs the 3D fit).
- `weak-fit`: a loose match; the tyres are not locked to the floor.
- `yaw-unsure`, `yaw-vs-label`, `no-view-label`: check the angle. (`flip-ambiguous`: front and back fitted equally well and the hints decided; usually fine.)
- `invented`: much of the frame, or some of it near the car, was never in the background picture.
- `merged-object`, `neighbour-car`, `occluder`, `mask-reflection`: the cut-out has a sizeable extra or missing part (more than 3% of the car); check `cutouts\`.
- `contact-mismatch`: the tyres miss their floor contacts by more than 1% of the output height.
- `camera-outside-room`: the photo's camera would be outside that background's room, so it was moved inside (the floor's perspective then differs a little from the car's).
- `car-on-slope` (the car stood on a sloping road; levelled to it), `possible-slope`, `partial-level` (levelling reduced to keep a cut edge on the border).
- Camera evidence: `focal-prior`, `focal-weak`, `lens-change`, `roll-from-silhouette`, `roll-from-neighbours`, `no-lines`, `lines-error`, `camera-height-at-bound`.
- `body-guessed`, `body-ambiguous`: no `body` in `cars.json`.
- `generic-room`, `no-room`, `soft-floor`, `reflection-skipped`, `fit-slow`.

`flags` in flat mode: `cut-left` / `cut-right` / `cut-top` / `cut-bottom`; `low-conf-label` (classifier confidence under 60%); `no-view-label`; `hw-mismatch` (often a label far from the true angle); `scale-capped` (the car is smaller than the stage needs); `near-edge` / `cropped`; `contact-unsnapped` / `contact-partial`.

Other notes:
- `cutouts\` is the quickest way to check a cut-out: open one in any viewer that shows transparency.
- Running again overwrites files with the same name. The script refuses to write into the images folder.
- `results.csv` is written even when a run stops part-way (an error, Ctrl+C). In room mode it then lists every photo cut out so far, the ones not placed yet with a `note`; a photo the run never reached is not listed.

## Use from Python

Room mode, for one photo (`fit_photo` measures nothing about the background, so one fit serves every background):

```python
from pathlib import Path

import cv2
from bg_changer import CarSegmenter, load_cars, select_car
from bg_changer.scenes import SCENES
from car_fit_to_bg import place_in_room
from car_fit_to_bg.batch import choose_focals, fit_measured, measure

shape = select_car(load_cars("cars.json"), "Maruti Suzuki Baleno 2022")
path = Path("0012.0s_front_92pct.jpg")
image = cv2.imread(str(path))                                             # BGR
car = CarSegmenter().cut_out(image)                                       # car.alpha: the full photo's alpha
m = measure(path, image, car.alpha)                                       # its background lines
choose_focals([m])                                                        # the lens (pass every photo of a video)
fit = fit_measured(m, image, car.alpha, car.cut_sides, shape, [m], None)  # the camera and the car's pose
background = cv2.imread("backgrounds/background_1.webp")
placed = place_in_room(fit, "background_1", background, SCENES["background_1"])
cv2.imwrite("car_studio.webp", placed.image)                              # placed.metrics, placed.flags
```

Flat mode:

```python
import cv2
from bg_changer import BackgroundChanger, CarShape

changer = BackgroundChanger(car=CarShape(4.684, 1.834, 1.667, 2.829))   # metres; loads the model once
image = cv2.imread("0012.0s_right_95pct.jpg")                           # BGR
background = cv2.imread("backgrounds/background_1.webp")
result, plan = changer.process(image, background, "background_1", stem="0012.0s_right_95pct")
cv2.imwrite("car_studio.webp", result)                                  # plan.scale, plan.box_out, plan.flags
```

To put one car on several backgrounds in flat mode, cut it out once with `car = changer.cut_out(image)`, then call `changer.place(image, car, background, "background_2", stem=...)` for each. `BackgroundChanger` always uses the full-size background; for small photos, shrink the scene with `background_scaler` first (`BackgroundScaler(...).apply(scene, background, k)`).

## Tests

```powershell
.venv\Scripts\python -m pytest tests
```

The tests need no model. They check:
- **Car pixels:** every opaque car pixel is exactly one rotation and one plain downscale of the original, with shadows and reflection on. Room mode: an unrotated car on all three rooms and a generic room, a rotated one on background_1, a rotated car cropped at its cut edge on all three rooms; rotated edges get no new colours or dark fringes. Flat mode: a plain downscale on every scene.
- **Room mode through `change_background.py`:** a synthetic photo goes through stages 2-3 onto the three rooms, a generic room and a background without a room (no model, so stage 1's cut-out is the synthetic one); the written images are the rendered ones; cut-off photos are skipped without a fit; a failed fit placed on a background without a room keeps its flags; a photo whose lines cannot be measured is still fitted; `results.csv` columns, flat mode's unchanged.
- **3D rooms:** background_1 re-renders its own picture exactly from its own camera and background_2 within 1 grey level (background_3's picture is de-vignetted first, so it is not compared); a textured box seen from a new camera matches the expected picture; a camera above a room's ceiling is caught.
- **3D fit:** a known pose is recovered, the same way on every run; a rolled photo is levelled once and that is tracked everywhere; synthetic vanishing points give back roll, pitch and focal; two lenses at one image size are not pooled; blank, full-frame and uniform photos fail cleanly.
- **Flat mode:** placement, scenes, grounding, the background scaler and the car model (as before).
- **Files:** `cars.json` and config validation (room settings included); background selection.
- **`car_isolation`** (fake detector and SAM, no download): with the step off the cut-out is bit-identical; a touching neighbour car is removed while the car's inside stays opaque; a window hole is filled and a mirror gap stays open; a piece sticking out is removed only with evidence (another object's box, a free photo edge), so a spoiler SAM missed stays; a cached 8-bit mask gives the same cut-out; YOLOX decoding and NMS; the `[car_isolation]` config block.

`tools\check_flat.py` checks that flat mode still reproduces the pinned baseline pixel for pixel.

## Speed

Measured on this laptop (Intel i5-1240P, CPU only):

| Step | Time |
|---|---|
| Load the model | ~8-13 s, once per run |
| Cut the car out | ~12-17 s per photo, typically; the first photos and a busy or hot laptop can take up to ~45 s |
| `car_isolation` (when on) | ~10-13 s more per photo (`sample_vid`), plus ~2 s to load the models once |
| Room mode: measure the lines | under 1 s per photo |
| Room mode: fit the car in 3D | ~10-20 s per photo, once whatever the number of backgrounds |
| Room mode: render one background, with shadows, and save | ~0.2-1 s (the first render of background_3 in a run takes ~10 s more to prepare its picture) |
| Flat mode: place on one background and save | about 1 s |

Small photos are not much faster to cut out: the model works at its own fixed size. Cutting out the car and the 3D fit are nearly all of the time, and both happen once per photo however many backgrounds you choose. `birefnet-general` takes 4-5× longer for no visible gain. Fewer `threads` is faster on this CPU (see [findings](docs/findings.md#speed-fewer-threads-is-faster-on-this-laptop)).

## Known limits

- **Areas the background never showed are filled, not real.** A portrait phone frame looking down sees more floor and more ceiling or sky than the landscape background picture has; those parts are smooth fills (up to about half of some `sample_vid` frames, mostly at the top and bottom; on four close-ups also 5-15% of the area around the car). Check rows flagged `invented`.
- **The 3D outline is a simple shape** (SUV, hatch or sedan, stretched to the car's dimensions). A half-car close-up or an unusual body can be fitted some degrees off (flagged `yaw-unsure` or `weak-fit`).
- **Original lighting and reflections stay on the car**, on purpose: changing them would change how damage looks. A car shot in sun keeps its sunlit look in the studio.
- **Cut-off cars stay cut off.** Nothing is invented for the missing part; the cut edge stays on the output's border. Cars cut off at the bottom or on two sides are not placed.
- **Small photos give small results.** The car is never enlarged: a 464×832 video gives outputs of at most 464×832. Film the car whole, in landscape, at 1080p or more for bigger results.
- **The background pictures are flat photos.** Seen from a camera far from where they were taken (very high, very close), their baked-in light streaks and far scenery can look stretched; `camera-outside-room` marks the extreme cases.
- **End views ground more softly.** A car seen straight from the front or back often hides its tyres behind its bumpers; its contact then comes from its outline only.
- **A neighbouring car touching the main car in the photo** may be kept with it (often flagged `neighbour-car`). Check `cutouts\`.
- **Thin posts behind the car** (lamp posts, sign poles) are removed, but only thin vertical ones (up to ~1.2 % of the image width). A thick pillar touching the car stays.
- **Flat mode keeps the photo's own viewpoint** (the car can look like it floats on photos not shot like the Mercedes walkaround) and its tilt.
