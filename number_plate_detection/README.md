# Number plate masking for car images

Takes an image of a car (or a folder of them), finds the number plate if one is visible, and covers exactly the plate: either with a sample image warped onto the plate's 4 corners (so it sits right even on angled views), or with a solid colour. Images without a plate are copied unchanged.

Built for very low compute: a tiny YOLOv9 plate detector from [open-image-models](https://github.com/ankandrew/open-image-models) (MIT) runs on ONNX Runtime on the CPU, and the plate's outline is found with plain OpenCV. No GPU and no torch. Research notes and the other options considered are in [`docs/findings.md`](docs/findings.md).

## Setup (once)

Needs Python 3.10+. In PowerShell, from this `number_plate_detection` folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The first run downloads the model weights (7.4 MB) from the open-image-models GitHub releases into `%USERPROFILE%\.cache\open-image-models\`. After that it works offline. If the download fails (e.g. a certificate error on a company network), the script prints the download link and how to point `model` at a file you saved yourself.

## Run

1. Edit [`config.toml`](config.toml): set the input image or folder, the output folder, and whether to use the overlay image.
2. From this folder (`F:\PROJECTS\car_demo_img\number_plate_detection`), run:

```powershell
.venv\Scripts\python mask_plates.py
```

The default input is the frames made by `frame_view_type` (`..\frame_view_type\output\Final_zoomed_car`).

### Settings in `config.toml`

| Setting | Default | Meaning |
|---|---|---|
| `input` | `"../frame_view_type/output/Final_zoomed_car"` | One image, or a folder of `.jpg .jpeg .png .bmp .webp` images (not subfolders) |
| `output` | `"output"` | Base folder for results. Each input gets its own subfolder: `output\<input name>\` |
| `overlay` | `true` | `true` = cover each plate with `overlay_image`, warped onto its 4 corners. `false` = paint it with `color`. |
| `overlay_image` | `"assets/RC.png"` | The image to put on the plates. Any size, ideally plate-shaped (about 4.5:1); a PNG's transparency is kept. `assets/sample_plate.png` is a neutral placeholder. |
| `auto_brightness` | `true` | `true` = for every plate, [`plate_brightness_detector`](plate_brightness_detector/README.md) picks the overlay's brightness itself, for any plate colour (white, yellow, green, blue, red, black). `false` = use `lighting` below. |
| `brightness_smoothing` | `0` | Video only, with `auto_brightness`: smooth the main plate's brightness over this many recent frames so the overlay doesn't flicker. `0` = off (keep it off for separate photos). |
| `lighting` | `100` | Used when `auto_brightness = false`. 0–100 %: how far to match the overlay to the original plate's lighting. `0` = the overlay at its own full brightness (can look pasted on). `100` = its white looks like the original plate's white in that photo, including shade and a gradient across the plate. `50` = halfway. Overlay only. |
| `color` | `[0, 0, 0]` | Fill colour as `[red, green, blue]`. Used when `overlay = false`, and for any plate whose corners can't be found. |
| `tight_fit` | `true` | `true` = follow each plate's own outline. `false` = use the detector's straight box (bigger on angled views). |
| `padding` | `0.0` | Extra margin beyond the plate's edge, as a share of the plate's height. `0` = exactly the plate. |
| `model` | `"yolo-v9-t-384-license-plate-end2end"` | Detector. Bigger input size finds smaller plates but is slower (see [findings](docs/findings.md#the-chosen-detector)). A local `.onnx` path also works. |
| `confidence` | `0.25` | Keep plates the model is at least this sure about. Lower it if plates are missed. |
| `threads` | `4` | CPU threads for the detector. `0` = every core (slower on this laptop); 1–2 leaves the CPU free for other work. |

- Relative paths are relative to the `number_plate_detection` folder.
- For a full Windows path, use single quotes so the backslashes are kept: `input = 'D:\photos\car.jpg'`. Forward slashes in double quotes also work.

### How a plate is covered

1. The detector finds a straight box around the plate.
2. Inside that box, the plate's own 4 corners are found, out to its outer edge. First the code works out which colour is the plate's background (the one connected area round the letters). Light plates (white, yellow) are found by their bright background, in about 4–8 ms. Dark or coloured plates (blue, green, red, black with light letters) are found by their background colour, in about 20 ms.
   - **If the plate's edge can't be seen**, the outline is worked out from its row of letters. This happens with a dark plate on a dark car, or a plate so small that the background round its letters is only a pixel or two wide. Lines are fitted along the letters' tops and bottoms (the plate's tilt and perspective), the sides follow the letters' upright strokes, and a margin is added (0.35 × letter height above and below, 4.5 % of the row's length at each end). It's rougher, and the console and `detections.csv` show `letters` instead of `corners` for these plates. Example: the black plates with yellow letters in `sample_vid` (letters about 11 px tall).
3. With `overlay = true`, the overlay image is warped so its corners land exactly on the plate's corners, and darkened to the plate's lighting:
   - `auto_brightness = true`: [`plate_brightness_detector`](plate_brightness_detector/README.md) reads the light from something white on the plate (background or letters), else from a light paint such as yellow (corrected for its colour), else from the photo. About 2–8 ms per plate.
   - `auto_brightness = false`: the plate's light parts are measured and the overlay is darkened towards them by the `lighting` percentage. This is right for white plates, but too dark on yellow ones.

   With `overlay = false`, the plate's shape is filled with `color`.
4. **If neither the edge nor a row of letters can be found** (a false hit on a grille or chrome trim, a badly blurred plate), that plate gets a solid box in `color` instead, even with `overlay = true`. An image stretched over a straight box would sit wrong. The console and `detections.csv` show `box` for these.

### One-off overrides

Command-line options win over `config.toml`:

```powershell
.venv\Scripts\python mask_plates.py D:\photos\car.jpg --no-overlay --color 255,255,255
```

| Option | Overrides |
|---|---|
| `<image or folder>` | `input` |
| `--out <folder>` | `output` |
| `--overlay` / `--no-overlay` | `overlay` |
| `--overlay-image <file>` | `overlay_image` |
| `--auto-brightness` / `--no-auto-brightness` | `auto_brightness` |
| `--brightness-smoothing N` | `brightness_smoothing` |
| `--lighting N` | `lighting` (0–100) |
| `--color R,G,B` | `color` |
| `--tight-fit` / `--no-tight-fit` | `tight_fit` |
| `--pad N` | `padding` |
| `--model <name or .onnx>` | `model` |
| `--conf N` | `confidence` |
| `--threads N` | `threads` |
| `--config <file>` | Use a different config file |

## Output

```
output\Final_zoomed_car\
  0000.0s_front_92pct.jpg      same file names as the input, plate covered
  ...
  detections.csv
```

- `detections.csv` has one row per plate: file, number of plates in that image, plate number, `fit` (`corners`, `letters` or `box`), the detector box `x1, y1, x2, y2`, confidence, `corners` (the plate's outline as `x,y` pairs: top-left, top-right, bottom-right, bottom-left), and with `auto_brightness` also `brightness_pct` (how bright the overlay's white was made), `brightness_source` (what it was read from, e.g. `white background`, `yellow background (/0.75)`, `photo (...)`) and `plate_colours` (background/letters, e.g. `blue/white`). An image with no plate gets one row with `plates = 0`.
- For a single image, the subfolder is named after the image (`output\car\car.jpg`).
- Running again writes into the same folder, overwriting files with the same name. The script refuses to write into the input folder.

## Use from Python

```python
import cv2
from plate_brightness_detector import BrightnessDetector
from plate_masker import PlateMasker

overlay = cv2.imread("assets/sample_plate.png", cv2.IMREAD_UNCHANGED)
masker = PlateMasker(overlay=overlay, brightness_detector=BrightnessDetector())  # or lighting=100 instead
image = cv2.imread("car.jpg")             # BGR
masked, plates = masker.process(image)    # plates: Plate(x1, y1, x2, y2, confidence, corners), .fit
cv2.imwrite("car_masked.jpg", masked)
```

`PlateDetector` (find only), `fit_plate()` (find corners) and `mask_plates()` (mask only) can also be used on their own. Mask styles are small functions registered by name in `MASKERS` in [`plate_masker/masking.py`](plate_masker/masking.py), so a new style (e.g. blur) is one function.

## Testing plate colours

The test video only has white plates, so there is a test set with every plate colour. Sample plates in [`assets/test_plates/`](assets/test_plates/README.md) (white, white/red, yellow taxi, green/white, green/yellow, green/black, blue/white, red/white, black/white, and black/yellow without a border line like the Indian rental plates in `sample_vid`) are pasted onto 3 views of the car (front, angled front, angled rear) at the real plate's spot and lighting. That way the right outline and brightness are known for every test photo. A night case (dark scene, plate glowing in a flash) is added too.

1. Make the test photos once (into `assets/test_photos/`, about 27 MB; it needs `frame_view_type`'s frames):

   ```powershell
   .venv\Scripts\python tests\make_test_photos.py
   ```

2. Test every colour, one after the other:

   ```powershell
   .venv\Scripts\python tests\run_colour_tests.py
   ```

   - `--pause` waits for Enter after each colour, so you can look at its results before the next.
   - `--case yellow_black` tests one colour (repeat `--case` for several).

For every photo the runner masks the plate with auto brightness on and off, and checks:
- the plate is found
- its corners are found
- its outline overlaps the true one by at least 90 %
- the logo's white is within ±5 points of the true brightness

Results go to `output\colour_tests\<colour>\`: the masked photos, `compare.jpg` (test photo | auto off | auto on) and `results.csv`, with a `summary.csv` for all colours.

To try one colour through the normal script instead: `.venv\Scripts\python mask_plates.py assets\test_photos\yellow_black`.

**Your own tests:**
- **A new plate design:** drop a plate image into `assets/test_plates/` and run `make_test_photos.py` again.
- **Real photos:** put them in a folder inside `assets/test_photos/`. They are run too, with numbers only (no right answer is known).

**Latest result: all 11 cases pass on all 3 views.**

| Plate | Outline overlap | True brightness | Auto ON | Auto OFF |
|---|---|---|---|---|
| white, black letters | 93–97 % | 75–76 % | 75–76 % | 75 % |
| white, red letters | 96–97 % | 75–76 % | 75–76 % | 75 % |
| yellow taxi | 94–95 % | 75–76 % | 76–77 % | 57 % (too dark) |
| green, white letters | 94–96 % | 75–76 % | 75–76 % | 71–73 % |
| green, yellow letters | 94–95 % | 75–76 % | 76–77 % | 55 % (too dark) |
| green, black letters | 96–97 % | 75–76 % | 78–85 % (photo reading, rougher) | 23 % (far too dark) |
| blue, white letters | 93–96 % | 75–76 % | 75–76 % | 71–72 % |
| red, white letters | 94–97 % | 75–76 % | 75–76 % | 71–73 % |
| black, white letters | 91–97 % | 75–76 % | 75–76 % | 69–72 % |
| black, yellow letters, no border (outline from the letters) | 92–93 % | 75–76 % | 76 % | 53–56 % (too dark) |
| night, plate glowing in a flash | 93–98 % | dark scene | 19–22 % (photo reading) | 98–100 % (glowing) |

## Speed on CPU

Measured on this PC (Intel i5-1240P, no GPU) with the 20 4K frames from `frame_view_type`:

- **Whole run with overlay:** 20 frames in about 4.3 s. The model loads in about 1 s.
- **Per frame:** about 85 ms to find, fit and cover the plate: detector about 50 ms, corner fit about 8 ms per plate, overlay about 18 ms (solid fill about 10 ms). Reading and writing the 4K JPEG takes longer (about 80 ms + 47 ms).
- **Bigger models:** 256 → 30 ms, 384 → 63 ms, 512 → 102 ms, 640 → 151 ms per frame (detector only).
- **Threads:** 4 was fastest. Letting ONNX Runtime use every core was about 4× slower on this laptop, so `threads = 4` is the default.

## Results on the test frames

- The main car's plate was found in all 12 frames where it is visible. The corners were found on all 12, so the overlay sits on the plate's own edge: straight, tilted and strongly angled views alike. No part of the original plate shows around it.
- The two non-plates fell back to a small solid box: a false hit on chrome trim (33 % confidence) and a dark dealer plate holder.
- Two small background plates behind the showroom glass were missed at the default `confidence = 0.25` and caught at `0.15`.

Details are in [`docs/findings.md`](docs/findings.md#results-on-our-frames).

## Known limits

- **A missed plate is not masked.** Always check the output. Lower `confidence` first if plates are missed (it costs no time); a bigger model helps less.
- **Coloured plates are tested on generated photos only.** The real test video has white plates only. Yellow, green, blue, red and black plates were tested with sample plates pasted onto the car (see [Testing plate colours](#testing-plate-colours)). Real coloured plates (reflective sheeting, other fonts, dirt, stickers) may behave differently, so add real photos to the test set when you have them.
- **Outlines from the letters are estimates.** When a plate's edge can't be seen, its size comes from standard margins round the letters. Plates whose letters are small compared with the plate (e.g. US plates) will get a logo smaller than the plate. The letters themselves are always covered.
- **Small plates get their brightness from the photo.** When the letters are only 2–3 px thick, every letter pixel is blended with the background, so their paint can't be read. On `sample_vid` the reading comes from the photo (67–78 %, steady from frame to frame).
- **The straight box remains for a badly blurred plate**, or one with fewer than 3 readable letters and no visible edge.
- **Plate corners are sharp in the fit**, while real plates have slightly rounded corners. At the corners the overlay can reach a pixel or two onto the plate holder.
- **Small, distant or dim plates** (e.g. cars behind glass in the background) are the weak spot for detection.
- **Tested on one video only** (one car, daylight, one plate style). Night, rain, blur and other countries' plates are untested.
- **The model's training data is not published**, so plates from some countries or in unusual formats may be found less reliably.
