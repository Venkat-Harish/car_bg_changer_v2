# Number plate detection and masking: findings

**Question:** Given an image of a car, can we find the number plate (if one is visible) and mask it, using very little compute (CPU only, no GPU) and open-source parts?

Status: researched 2026-11-04; built as [`mask_plates.py`](../mask_plates.py) + [`plate_masker/`](../plate_masker/); tight plate-shaped fit and image overlay added 2026-11-05. Tested on the 20 frames from `frame_view_type`.

## Short answer

Yes. A tiny YOLOv9 plate detector from [open-image-models](https://github.com/ankandrew/open-image-models) (MIT) runs through ONNX Runtime on the CPU, with no torch and no separate car detector. It finds a box around the plate. Plain OpenCV then finds the plate's own 4 corners inside that box, and we cover exactly that shape: with a sample image warped onto the corners, or with a solid colour.

On our 20 test frames (4K) the default 384 px model found the main car's plate in **all 12 frames where it is visible**, and the corners were found on all 12. A whole frame takes about **85 ms** on a laptop CPU (4 threads), with the detector about 50 ms of that. The model file is 7.4 MB. That is about 10× cheaper than the side classifier in `frame_view_type` (0.8 s per frame).

## Options compared

| Option | Licence | Model / compute | Verdict |
|---|---|---|---|
| **[open-image-models](https://github.com/ankandrew/open-image-models)** plate detectors | MIT | YOLOv9-tiny/small, ONNX. Six sizes, 256 to 640 px input. Runs on ONNX Runtime CPU; GPU, OpenVINO and DirectML are optional extras. | **Chosen.** Smallest compute, permissive licence, plain pip install, API returns boxes directly. |
| [fast-alpr](https://github.com/ankandrew/fast-alpr) | MIT | Same detector plus an OCR model that reads the plate text | Not needed: we only mask, we don't read. Using the detector package directly skips the OCR model. |
| [platedet](https://pypi.org/project/platedet) | Not checked | YOLO converted to ONNX, CPU by default | Trained mostly on Brazilian plates |
| [LiteALPR](https://huggingface.co/anhone3/LiteALPR) | AGPL-3.0 | YOLOv8n with Ghost convolutions, about 23 FPS on CPU (ONNX FP32) | AGPL is a problem for closed products. Trained mainly on Vietnamese plates. |
| [Meta EgoBlur](https://projectaria.com/tools/egoblur) | Apache-2.0, behind a model licence agreement | Faster R-CNN, about 104M parameters, about 400 MB. Meta's own figure: about 8 s per 2 MP image on CPU. | Accurate, but hundreds of times too heavy for "low compute" |
| [understand-ai/anonymizer](https://github.com/understand-ai/anonymizer) | Apache-2.0 | Faces and plates | Archived, no updates. Its own notes say it struggles with low-quality images. |
| [PeekingDuck](https://peekingduck.readthedocs.io/en/stable/use_cases/privacy_protection_license_plates.html) | Apache-2.0 | YOLOv4 family, pipeline framework | Older model, and you adopt a whole framework |
| [ippen/image-anonymizer](https://github.com/ippen/image-anonymizer) | Not checked | YOLO + Gaussian blur, Streamlit app | An app, not a module to import |
| OpenCV Haar cascade (`haarcascade_russian_plate_number.xml`, ships with OpenCV) | BSD | Classic cascade, near-zero compute, nothing to download | Tuned for Russian plates; many false hits and misses. Only as a last-resort fallback. |

## The chosen detector

All six plate models from open-image-models are single-class ("License Plate") YOLOv9 models with NMS built into the ONNX graph ("end2end"). Accuracy figures are the project's own:

| Model | Input | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `yolo-v9-t-256-license-plate-end2end` | 256 | 0.937 | 0.797 | 0.858 | 0.606 |
| **`yolo-v9-t-384-license-plate-end2end`** (our default) | 384 | 0.942 | 0.863 | 0.920 | 0.687 |
| `yolo-v9-t-416-license-plate-end2end` | 416 | 0.940 | 0.894 | 0.940 | 0.702 |
| `yolo-v9-t-512-license-plate-end2end` | 512 | 0.955 | 0.901 | 0.948 | 0.724 |
| `yolo-v9-t-640-license-plate-end2end` | 640 | 0.966 | 0.896 | 0.958 | 0.758 |
| `yolo-v9-s-608-license-plate-end2end` | 608 | 0.957 | 0.917 | 0.966 | 0.772 |

- The number in the name is the input size. The whole image is shrunk to that size, so a bigger size finds smaller (more distant) plates but costs more time.
- Weights download once from the project's GitHub releases into `%USERPROFILE%\.cache\open-image-models\<model>\`. After that it works offline. A local `.onnx` path also works.
- **Not published:** which countries' plates it was trained on, and CPU speed. We measured speed ourselves (below).
- The same package also has general COCO detectors (RF-DETR), which could find the car first if we ever need it. Not needed now: the plate model runs on the whole image.

## Results on our frames

**Test set:** the 20 frames from `frame_view_type` (`Final_zoomed_car.mp4`, 3840×2160, a white Mercedes EQB filmed while walking round it on a dealer lot). Every detection was checked by eye at full resolution.

What the frames contain:
- **The main car's plate** is visible in 12 frames (straight and diagonal front and back views). It is out of view in the 5 side views and in 3 diagonal frames where it is round the corner (`0008`, `0014`, `0026`).
- **Two readable plates in the background**, on dark cars behind the showroom glass (`0014`, `0016`). Small (about 150 px wide in a 3840 px frame) and dim.
- **Dealer plate holders** ("Van Mossel") on lot cars. Plate-shaped but not registration numbers; masking them does no harm.

| Model, threshold | Main car plates found (of 12) | Background plates found (of 2) | False hits | Dealer holders masked |
|---|---|---|---|---|
| 256, 0.25 | 12 | 1 (loose box) | 3 (chrome trim ×2, grille) | 0 |
| **384, 0.25 (default)** | **12** | **0** | **1** (chrome trim, 33 %) | 1 |
| 512, 0.25 | 12 | 1 | 1 (grille) | 0 |
| 640, 0.25 | 12 | 1 | 1 (grille) | 2 |
| 384, 0.15 | 12 | 2 | 3 (chrome trim, car body, badge) | 1 |

- Every model found every visible main-car plate. (This comparison used the first version's straight box plus 10 % padding; the tight fit came later and doesn't change which plates are found.)
- False hits are small patches on chrome trim, grilles or badges, all below 70 % confidence and most below 35 %.
- **Only 384 at 0.15 caught both background plates.** The bigger models do not fix this at 0.25. A lower threshold does, at no extra time.
- **Choice:** the default stays 384 at 0.25: the cheapest setting that finds every main-car plate with one small false patch. Where background plates must be hidden too, set `confidence = 0.15`.

### Speed

Measured on this PC (Intel i5-1240P laptop, no GPU, other apps open), one 4K frame, median of 48 runs at 4 threads:

| Model | Detect time per frame |
|---|---|
| 256 | 30 ms |
| **384** | **63 ms** |
| 512 | 102 ms |
| 640 | 151 ms |

- **Full run, default settings:** 20 4K frames in about 4.3 s with the tight fit and overlay (3.0 s with the first version's plain box). Per frame: detector about 50 ms, corner fit about 8 ms per plate, overlay about 18 ms (solid fill about 10 ms). The rest is JPEG reading (about 80 ms per 4K frame) and writing (about 47 ms).
- **Frame size barely changes detect time,** because every image is shrunk to the model's input size first. Only file reading and writing grow with resolution.
- **Model load:** 0.7 s. The very first run of each model also downloads its 7.4 MB file (it took about 24 s here).
- **Install size:** about 250 MB for the whole venv (onnxruntime, OpenCV, numpy). No torch.

**Threads matter more than model size.** Same 384 model, same frame, runs interleaved:

| ONNX Runtime threads | Median | Best |
|---|---|---|
| 1 | 215 ms | 104 ms |
| 2 | 79 ms | 60 ms |
| **4 (default)** | **56 ms** | 49 ms |
| all cores (ONNX Runtime default) | 229 ms | 130 ms |

This CPU has 4 performance and 8 efficiency cores. Using all of them makes threads wait on the slow cores and fight other apps for time. So the module caps threads at 4 (`threads` in `config.toml`). Use 1–2 to leave the CPU free for other work. On a different machine, try 2, 4 and 0 (all cores).

## How the module works

1. Load the model once (`PlateMasker()` or `PlateDetector()`), on ONNX Runtime's CPU provider with 4 threads.
2. `detect()` runs the ONNX model on the full image and returns plate boxes in original-image pixels, highest confidence first.
3. `fit_plate()` ([`corners.py`](../plate_masker/corners.py)) finds the plate's own outline inside each box (next section). If it can't, the plate keeps its box.
4. `mask_plates()` covers each plate's shape with the chosen mask style: `"image"` (overlay warped onto the 4 corners) or `"solid"`. Plates left with only a box always get `"solid"`. Styles are one function each in [`masking.py`](../plate_masker/masking.py), looked up by name (`MASKERS`).

## Fitting the plate's own outline

**Why it's needed.** The detector's box is a straight rectangle. On angled views the plate is a tilted 4-sided shape, so the box covered 2–3× the plate's area (the plate's area was only 32 % of the box on `0034` and 33 % on `0006`, against 66 % on the straight `0000`). With 10 % padding on top, the old black box was clearly bigger than the plate.

**How it works** (plain OpenCV, about 8 ms per plate, no extra model):

1. Threshold the area around the box (Otsu) and take the bright blob under the box centre: the plate's white face. A few blur and clean-up settings are tried in turn, scaled to the plate's size, because a clean-up strong enough to cut the plate free of a white bumper also erases the 1–2 px white margin round the characters on small plates.
2. Simplify the blob's outline to 4 corners: convex hull, then `approxPolyDP`.
3. **Grow to the outer trim, edge by edge.** Brightness profiles across every edge of our plates showed the same structure: white face → thin dark border line → light rim (the plate's raised edge) → bumper or holder. The rim ended 0.13–0.15 × plate height outside the white face on almost every edge (0.09–0.23 overall). So each edge is measured on its own from 7 profiles: find the border line and the end of the rim, take the median, and cap it at 0.22 × height. A plate with no border line ends where its white face ends. If an edge can't be measured, 0.14 × height is used.
4. **Sanity checks**, or the plate keeps its box: convex, no tiny edges, width/height 1.0–8.0, area 0.2–1.6 × the box, spans at least 60 % of the box, and top/bottom edges within 40° of horizontal (sides within 40° of vertical). Our most tilted real plate leans about 25°. The tilt check was added after a false hit on chrome trim produced a sideways shape that passed every other check.

**Results on our frames:**

- Corners found on all 12 visible main-car plates (straight, tilted, angled front and rear). Checked at 3× zoom: no white face, border line or rim of the original plate shows outside the overlay or fill. Only the car's black plastic plate holder is visible around it, as it should be.
- Both non-plates fell back to a box: the chrome-trim false hit (`0008`) and the dark dealer plate holder (`0022`, it has no light face).
- On the same frames shrunk to 640–1920 px wide, the straight front and the angled rear still fit at every size. The strongly angled front (`0034`) fits from 1280 px wide; below that its plate is about 40 px tall and it falls back to the box.
- **Not handled:** plates with rounded corners are fitted with sharp corners, so at the very corners the overlay can reach a pixel or two onto the holder. Dark plates (light text on a dark face) always fall back to the box.

## Limits and risks

- **A missed plate is a privacy leak; a wrong mask is cheap.** The default threshold (0.25) is the model's own. On our frames it missed two small background plates behind glass; `confidence = 0.15` caught them. Check the output.
- **Small, distant or dim plates** are the weak spot. Lower the threshold first (free). A bigger model helps less than expected (see the results table) and costs 2–3× the time.
- **Tested on one video only:** one car, one dealer lot, daylight. Night shots, rain, motion blur and other plate formats are untested.
- **Unknown training data.** Plates from countries or formats it has not seen (e.g. square two-line plates, unusual colours) may be found less reliably. Test on real footage from each market.
- **Corner fitting relies on a light plate face** (white or yellow) and on the plate being at least about 40 px tall at steep angles. Anything else falls back to the straight box with a solid fill.
- **Not tracked over time.** Each image is handled on its own. For video, a plate found in one frame and missed in the next will flicker; carrying boxes over from neighbouring frames would fix that.

## Replacing the plate with an image

Built (`overlay = true` in `config.toml`). The overlay image is perspective-warped so its 4 corners land on the plate's 4 fitted corners, then blended in with an anti-aliased mask; a PNG's transparency is kept. Only the small area around the plate is warped (about 18 ms per plate on a 4K frame). The placeholder `assets/sample_plate.png` is 1040×224 (EU plate proportions), so swap in your own image at about that shape.

Rules that came out of testing:

| Case | What happens |
|---|---|
| Straight, tilted or angled plate with corners found | Overlay warped onto the plate's own outline. Default `padding = 0`, because the fitted outline already includes the plate's border and rim. |
| Corners not found (false hit, dark dealer holder, tiny plate) | Solid box in `color`. An image stretched over a straight box would sit wrong on a tilted plate, a grille or a badge. |
| `tight_fit = false` | The overlay goes on the detector's straight box (useful only for straight-on views). |

### Matching the plate's lighting

At full brightness the overlay looked pasted on, most of all on the shaded rear plates. The camera never records these plates as pure white: their white face measured **170–187 of 255** (front plates in sun 182–187, rear plates in shade 170–181), varying by about ±10 across a plate. The overlay's 255 white was brighter than anything else on the car.

So before painting, the original plate is measured: the inner part of the fitted plate (clear of border and rim), light pixels only (the face, not the characters), fitted with a plane so a gradient across the plate is kept, plus one refit to drop glare. A full match scales the overlay so its white becomes the plate's white (about 70 % of its own brightness here). Darker colours in the overlay scale by the same factor. Brightness only, no colour, so a yellow plate won't tint the logo.

The `lighting` setting (0–100 %) blends between no change and that full match: `factor = 1 − lighting/100 × (1 − plate white/255)`. Measured on 4 plates, the overlay's white came out at 255 at 0 %, 214–221 at 50 %, and 173–188 at 100 %, against 186–190 in the original plates. By eye, 100 % looks like part of the photo on both sunlit and shaded plates, so it is the default.

### Any plate colour (added 2026-11-06)

The fixed method above reads "the light parts of the plate" and assumes they are white. In a test that recoloured the real plate in `0000`, it held up on plates with white anywhere: white letters on blue, green, red or black read 182–183 against the correct 186. It failed in two ways:

- **Brightness on yellow:** a yellow plate or yellow letters read 140, so the board came out about 25 % too dark.
- **Outline on dark plates:** light letters on a dark background gave a crooked outline (82–84 % overlap with the true outline), because the "bright blob" there is the lettering, not the plate.

What was built:

1. **Which side is the background:** the plate is split into its light and dark parts. The background is the part that forms one connected area; the letters are separate islands. On all 12 real plates the white background's biggest connected piece held 91–98 % of its pixels, the letters' only 15–18 %. (Area alone is unreliable: this bold font covers about half the plate, and the dark part was the majority in 5 of 12.)
2. **Outline by background colour** ([`corners.py`](../plate_masker/corners.py)): light-background plates are fitted as before (the 12 real plates are unchanged, within 0.5 px). Dark or coloured backgrounds are found by colour distance to the background colour (two-colour k-means in Lab), trying the darker colour first and then the lighter one. Simulated green, blue, red and black plates now fit at 98 % overlap. A black plate on a black holder and dark bumper (angled rear) falls back to the box.
3. **[`plate_brightness_detector`](../plate_brightness_detector/README.md)** (toggle `auto_brightness`): reads the light from white paint (background or letters), else from a light paint divided by its factor (yellow 0.75), else from the photo (95th percentile of the lower two-thirds). A plate brighter than 2× the photo reading is treated as glare or flash. Optional median smoothing over frames for video.

Results with `auto_brightness = true` (correct answer 73–74 % for every simulated plate): real white plates 73–75 %; yellow taxi 75 %; green/white, blue/white, red/white and black/white 72–73 %; green/yellow 74–75 %. Green with black letters (no white or light paint) falls back to the photo reading: 85 %, rougher. Night glow → 22 % (photo); muddy plate → photo. With `auto_brightness = false` the yellow plates came out at 53–56 %. With `auto_brightness = false`, output is pixel-identical to before the change.

Costs: corner fit 4–8 ms for light plates, about 20 ms for coloured plates, up to about 70 ms for green with black letters (both colours are tried). Brightness 2–8 ms per plate, plus about 2 ms per image.

### The colour test set (added 2026-11-06)

The checks above recoloured the real plate in memory, which left nothing to re-test with. The test set in [`assets/test_plates`](../assets/test_plates/README.md) and `assets/test_photos` replaces that: 9 flat sample plates are pasted onto 3 views (front, angled front, angled rear) at the real plate's outline and lighting, plus a night-flash case. [`tests/run_colour_tests.py`](../tests/run_colour_tests.py) scores each photo. It passes when the plate is found, its corners are found, the outline overlaps the truth by at least 90 %, and the brightness is within ±5 points.

The first run failed 7 of the 10 cases, every time because of the outline; brightness was right wherever the outline was. What it found, and the fixes:

- **The fixed 0.14 × height trim on coloured plates.** That trim was only right by luck. On a straight view, the colour search can pick up the plate's outer rim, so the blob is already the whole plate, and adding 0.14 × height made it too big (+0.14 on every edge; 68–74 % overlap). Fix: on coloured plates the trim is now measured. Walking outward, the code looks for "different from the background (border line), background colour again (rim), different (the car)". A thin, blurred line never matches the letter colour exactly, but it still shows up as a bump. When the rim's end can't be seen, a typical rim (0.06 × height) is used.
- **A faint border line merges the plate with the bumper.** A thin red line, blurred, no longer separated a white plate from the white bumper (51 % overlap). Fix: a background outline more than 10 % outside the detector's box is rejected, and the next attempt is used. Real plates' outer edges reach at most 2 % / 6 % past the box.
- **Black plates in a black holder.** The black background merges with the holder and dark bumper. Fix: a light border line around a dark background is accepted as the plate's outline, with only the rim added.
- **The sample plates' border line was thinner than on real plates** (3.6 % of height against about 5 %). Perspective squeezed it to about 2 px on the angled view and blur removed it. Redrawn at 5 %.

After the fixes, all 10 cases pass on all 3 views: outline overlap 91–98 %, brightness within ±1 point (green/black, which has no white or light paint, uses the photo reading at 78–85 % against a true 75–76 %). The 12 real plates' outlines are unchanged (within 0.5 px), and `mask_plates.py`'s output on the 20 frames is pixel-identical to before.

### A second real video: `sample_vid` (added 2026-11-07)

`sample_vid.mp4` is a low-resolution phone video (464×832 portrait) of a dark grey hatchback with **black Indian rental plates with yellow letters** (`KA 03 AM 1227`). The plates are about 80×25 px, with letters about 11 px tall and 2–3 px thick.

The detector found the plate in all 8 frames where one is visible (the other 6 are side views and close-ups with no plate). But every plate fell back to the plain black box, so the logo never appeared. All three outline methods failed:
- the plate's black background is almost the same shade as the dark car round it, so its edge can't be traced
- there is no light border line
- the only bright parts are the letters

**Fix: outline from the letters** (fit `letters`), the last method before the box:
- Find the row of characters: at least 3 shapes of similar height and similar ink on one line, at most 40° from horizontal, light or dark.
- Fit lines along their tops and bottoms (tilt and perspective), and take the plate's sides along the letters' upright strokes. At an angle these stay near upright in the photo instead of square to the slanted row.
- Add margins: 0.35 × the letter height above and below, and 4.5 % of the row's length at each end, scaled by the letter height at that end for perspective. These were measured from the real plates (0.27–0.4 vertically, about 3–5 % at the ends) and kept within the detector's box left to right.

Things the test set caught along the way:
- End margins sized from the letter height were too wide on angled views (the row is squeezed, the letters' height isn't).
- The sides must follow the letters' strokes, not run square to the row.
- A thin highlight at the plate's end was being counted as a letter; a minimum of 30 % of the median letter's ink filters it.

**Brightness on small plates.** Reading the light from 2–3 px letters swung from 65 % to 100 % between frames of the same car, because every pixel is blended with the black background, and the yellow was read sometimes as yellow (÷0.75), sometimes as orange (÷0.6). New rule in `plate_brightness_detector`: if less than 30 % of the light paint survives shaving 1 px off its edges, use the photo reading. Readable plates keep 45–83 %; these keep 8–16 %. Now 67–78 %, steady.

**Results:**
- All 8 `sample_vid` plates get the logo, following tilt and perspective, with even brightness.
- A black/yellow test plate without a border line (condensed letters filling the plate, like the Indian plates) was added to the colour test set. All 11 cases pass on all 3 views; black/yellow overlaps the truth by 92–93 %.
- The 4K video's output is pixel-identical to before.

**Limit:** outlines from the letters assume the letters fill most of the plate. Plates with small letters on a big plate (e.g. US) would get a logo smaller than the plate. The letters themselves are always covered.

Possible next steps: check `PAINT_FACTORS` and the colour boundaries on real photos of yellow, green, blue and red plates; put the branded image only on the main car's plate (the largest, most confident one) and give background plates a solid fill; round the overlay's corners to match real plates.

## Sources

- [open-image-models (GitHub)](https://github.com/ankandrew/open-image-models), [PyPI](https://pypi.org/project/open-image-models/), model registry in [`hub.py`](https://github.com/ankandrew/open-image-models/blob/main/open_image_models/detection/core/hub.py)
- [fast-alpr (GitHub)](https://github.com/ankandrew/fast-alpr)
- [platedet (PyPI)](https://pypi.org/project/platedet)
- [LiteALPR (Hugging Face)](https://huggingface.co/anhone3/LiteALPR)
- [EgoBlur (Project Aria)](https://projectaria.com/tools/egoblur), [EgoBlur paper (arXiv)](https://arxiv.org/pdf/2308.13093v2.pdf), [EgoBlur on PyPI](https://pypi.org/project/egoblur/)
- [Nvidia LPDNet vs Meta EgoBlur (Starschema blog)](https://medium.com/starschema-blog/nvidia-lpdnet-vs-meta-egoblur-blurring-license-plates-with-deep-learning-d7cc2d770685)
- [understand-ai/anonymizer](https://github.com/understand-ai/anonymizer)
- [PeekingDuck: privacy protection (licence plates)](https://peekingduck.readthedocs.io/en/stable/use_cases/privacy_protection_license_plates.html)
- [ippen/image-anonymizer](https://github.com/ippen/image-anonymizer)
- [Autoware: data anonymization](https://docs.autoware.org/main/datasets/data-anonymization/)
