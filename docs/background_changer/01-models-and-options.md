# Models and options

What model background_changer uses, what each algorithm around it does, and which other models and placement approaches were considered.

## In short

- **One learned model runs inside background_changer:** BiRefNet general-lite, through rembg on ONNX Runtime, CPU only, 4 threads. It gives the car's soft cut-out mask. Everything after it is classical code.
- **Two learned models run upstream, not here.** number_plate_detection's YOLOv9 plate detector has already masked the plates in the input frames. frame_view_type's ConvNeXt-Small classifier wrote the view label into each file name and a `results.csv`. background_changer only reads those, as a hint for which way round the car is.
- **There is no generative model and no second segmentation or refinement model. No neural network is used in rendering.** The mask clean-up, the camera from lines, the 3D car fit and the 3D room render all use geometry, morphology and classical optimisation.
- **Placement uses room mode by default.** It re-renders the background from each photo's own camera. It replaced v1 "fit to a box". The older flat mode is still available. A full 3D rebuild of the car, as Spyne does, was studied and not built.

## How it works

Where each model or algorithm sits in one room-mode run ([change_background.py:8-13](../../background_changer/change_background.py#L8-L13)):

1. **Input.** The photos are frame_view_type's frames with the plates already masked by number_plate_detection. That folder is the default `images_folder` ([config.toml:11-13](../../background_changer/config.toml#L11-L13)). The view label is in the file name, e.g. `0012.0s_right_95pct.jpg` ([probe.py:16-33](../../background_changer/bg_changer/probe.py#L16-L33)).
2. **Load the cut-out model once per run.** `CarSegmenter` opens one rembg session on the CPU provider with `intra_op_num_threads = 4` ([segment.py:162-171](../../background_changer/bg_changer/segment.py#L162-L171), [change_background.py:837-843](../../background_changer/change_background.py#L837-L843)).
3. **Cut out each photo.** One `rembg.remove(..., only_mask=True)` call returns the soft alpha ([segment.py:173-182](../../background_changer/bg_changer/segment.py#L173-L182)). `clean_alpha` then cuts it down to this one car with OpenCV morphology ([segment.py:115-147](../../background_changer/bg_changer/segment.py#L115-L147)). See [02](02-car-cutout-model.md) and [03](03-cutout-refinement.md).
4. **Measure the photo's camera from its background lines.** OpenCV LSD finds the line segments, with the car masked out. MSAC finds the vertical vanishing point. A Manhattan search gives the focal length, and a `soft_l1` least-squares refit settles roll, pitch and focal ([lines.py:1-14](../../background_changer/car_fit_to_bg/lines.py#L1-L14)). See [05](05-camera-angle.md).
5. **Free the model.** After every photo is cut out, the segmenter is deleted before fitting and rendering ([change_background.py:897-899](../../background_changer/change_background.py#L897-L899)).
6. **Fit the car's pose.** A hand-built 3D proxy, made from the car's `cars.json` dimensions, is drawn through the photo's camera and fitted to the mask with Nelder-Mead ([pose.py:394-395](../../background_changer/car_fit_to_bg/pose.py#L394-L395)). The classifier's probabilities (or the file-name label) and a red-tail-lamp cue settle front versus back ([pose.py:497-499](../../background_changer/car_fit_to_bg/pose.py#L497-L499)). See [04](04-car-dimensions-and-orientation.md). A tyre contact lock then nudges the camera with `least_squares` ([ground.py:179-184](../../background_changer/car_fit_to_bg/ground.py#L179-L184)). See [05](05-camera-angle.md).
7. **Render the room.** A CPU ray-caster re-renders the background picture from the fitted camera. It fills unseen areas by fixed rules and shades the floor ([render.py:1-13](../../background_changer/room3d/render.py#L1-L13)). See [06](06-3d-room-rendering.md).
8. **Paste the car.** Opaque car pixels are copied unchanged, and only the soft edge is blended ([compose.py:28-43](../../background_changer/bg_changer/compose.py#L28-L43)).

Flat mode runs steps 1-3 and 8 only. In between, it scales and moves the car onto the background picture as it is, through one fixed "capture camera" (see [Placement approaches](#placement-approaches)).

## Models used

| Model or algorithm | Kind | Role | Licence | Runtime | Code |
|---|---|---|---|---|---|
| **BiRefNet general-lite** (rembg name `birefnet-general-lite`) | Learned (neural network). **The only model run here.** | Soft 0..1 alpha of the salient object: the car cut-out | MIT (BiRefNet); rembg MIT | rembg 2.0.86 on ONNX Runtime 1.31.1, `CPUExecutionProvider`, 4 threads. ~220 MB weights, Swin-Tiny backbone. ~17 s per 4K photo, warm, at 4 threads (measured on an Intel i5-1240P laptop, CPU, [findings](../../background_changer/docs/findings.md#cutting-the-car-out)) | [segment.py:22](../../background_changer/bg_changer/segment.py#L22), [segment.py:162-182](../../background_changer/bg_changer/segment.py#L162-L182), [requirements.txt:3-4](../../background_changer/requirements.txt#L3-L4) |
| **View classifier** `veerupandey/view_classifier` (ConvNeXt-Small, `joint_v2_best.pt`) | Learned, **upstream, not run here** | Its 8-bin label (in the file name) and probabilities (in its `results.csv`) are a yaw hint. Flat mode uses the label's bin centre as the yaw. Room mode uses the probabilities, or the label, as a von Mises prior in the pose score. | MIT (code) | PyTorch 2.9.1 on CPU, inside frame_view_type only. ~198 MB, ~0.8 s per 4K frame ([frame_view_type README](../../frame_view_type/README.md)) | [yaw.py:16-17](../../background_changer/bg_changer/yaw.py#L16-L17), [batch.py:56-69](../../background_changer/car_fit_to_bg/batch.py#L56-L69), [pose.py:176-206](../../background_changer/car_fit_to_bg/pose.py#L176-L206) |
| **Plate detector** `yolo-v9-t-384-license-plate-end2end` (open-image-models) | Learned, **upstream, not run here** | Masks the number plates before background_changer sees the photo. The masked plate pixels are part of the car from then on. | MIT | ONNX Runtime on CPU, in number_plate_detection | [detector.py:1-19](../../number_plate_detection/plate_masker/detector.py#L1-L19), [config.toml:11-13](../../background_changer/config.toml#L11-L13) |
| **`clean_alpha`** mask clean-up | Classical (OpenCV/NumPy morphology) | Threshold at 0.5, fill windows, drop thin posts, keep the main car, keep a thin soft edge, opaque core | n/a (own code) | CPU, part of the cut-out time | [segment.py:115-147](../../background_changer/bg_changer/segment.py#L115-L147), see [03](03-cutout-refinement.md) |
| **OpenCV LSD** line segment detector | Classical | Straight background edges for the camera's roll, pitch and focal. Segments on the car are dropped. | OpenCV (Apache-2.0) | Photo downscaled to at most 1280 px | [lines.py:41-45](../../background_changer/car_fit_to_bg/lines.py#L41-L45), [lines.py:66-78](../../background_changer/car_fit_to_bg/lines.py#L66-L78), [lines.py:253-262](../../background_changer/car_fit_to_bg/lines.py#L253-L262) |
| **MSAC** vertical vanishing point and **Manhattan** focal search | Classical robust estimation | Vertical direction (gives roll and pitch), then the focal that best explains the horizontal lines | n/a | 800 two-line hypotheses, fixed seed 0 | [lines.py:100-101](../../background_changer/car_fit_to_bg/lines.py#L100-L101), [lines.py:141-174](../../background_changer/car_fit_to_bg/lines.py#L141-L174) |
| **scipy `least_squares`** | Classical optimiser | `soft_l1` refit of roll, pitch and focal; the tyre contact lock (camera height and pitch) | SciPy (BSD-3-Clause) | CPU | [lines.py:197](../../background_changer/car_fit_to_bg/lines.py#L197), [ground.py:179-184](../../background_changer/car_fit_to_bg/ground.py#L179-L184) |
| **Hand-built 3D car proxy** | Geometry, not learned | An EQB-shaped body stretched to the car's length, width, height and wheelbase. Room mode fits its silhouette to the mask. Flat mode views it through the capture camera V. | n/a | NumPy; drawn with `cv2.fillConvexPoly` | [carmodel.py:11-44](../../background_changer/bg_changer/carmodel.py#L11-L44), [pose.py:60-78](../../background_changer/car_fit_to_bg/pose.py#L60-L78), [pose.py:140-156](../../background_changer/car_fit_to_bg/pose.py#L140-L156), see [04](04-car-dimensions-and-orientation.md) |
| **scipy `minimize`, Nelder-Mead** | Classical optimiser (derivative-free) | Pose fit over yaw, roll, log distance, camera height and aim. The IoU comes from a rasterised render, so there is no gradient. | SciPy (BSD-3-Clause) | 36-yaw sweep, then at most 5 candidates refined; ~10-20 s per photo (i5-1240P, [README](../../background_changer/README.md#speed)) | [pose.py:394-395](../../background_changer/car_fit_to_bg/pose.py#L394-L395), [pose.py:501-534](../../background_changer/car_fit_to_bg/pose.py#L501-L534) |
| **Background calibration** | Measured once from each picture (read-only pass), stored as constants | The plate camera and floor of background_1/2/3: rim ellipse, LED strips and seams; floor-tile repeat; 0.80 m jersey barriers | n/a | None at run time | [background_1.py:3-12](../../background_changer/bg_changer/scenes/background_1.py#L3-L12), [background_2.py:3-10](../../background_changer/bg_changer/scenes/background_2.py#L3-L10), [background_3.py:3-10](../../background_changer/bg_changer/scenes/background_3.py#L3-L10), [rooms.py:37-97](../../background_changer/car_fit_to_bg/rooms.py#L37-L97) |
| **Room ray-caster** (`room3d`) | Classical geometry plus Gaussian-pyramid sampling | Re-renders each background from the photo's camera. Fills unseen surfaces by fixed per-surface rules. | n/a (NumPy + OpenCV) | ~0.2-1 s per background with shadows (i5-1240P, [README](../../background_changer/README.md#speed)) | [render.py:1-13](../../background_changer/room3d/render.py#L1-L13), [render.py:40-64](../../background_changer/room3d/render.py#L40-L64), [room.py:1-13](../../background_changer/room3d/room.py#L1-L13), see [06](06-3d-room-rendering.md) |

What the code does **not** contain:

- **No generative model.** Nothing draws, inpaints or "enhances" pixels with a learned model. Background areas the plate never showed are filled by fixed rules in each surface's own coordinates: hybrid floor, band-fold walls, low-pass ceiling and sky ([render.py:7-12](../../background_changer/room3d/render.py#L7-L12)).
  > Note: README.md:110 says these parts are "filled smoothly". In code only the ceiling, the sky and walls moved in height are smooth (low-pass). The floor and side walls use the hybrid and band fills.
- **No neural network in rendering, shading or compositing.** `room3d`, `place.py`, `compose.py` and `level.py` use only NumPy, OpenCV and SciPy. The segmenter is deleted before any fitting or rendering starts ([change_background.py:897-899](../../background_changer/change_background.py#L897-L899)).
- **No second segmentation or refinement model.** There is one `remove` call per photo, with `only_mask=True` and no other option, so rembg's own alpha matting is not used ([segment.py:178](../../background_changer/bg_changer/segment.py#L178)). `requirements.txt` has no torch and no other model package ([requirements.txt:3-9](../../background_changer/requirements.txt#L3-L9)). rembg and onnxruntime are imported only in `segment.py` ([segment.py:164-175](../../background_changer/bg_changer/segment.py#L164-L175)) and in an install check ([change_background.py:796-797](../../background_changer/change_background.py#L796-L797)).

## The maths and constants

| Name | Value | Meaning | Code |
|---|---|---|---|
| `DEFAULT_MODEL` | `"birefnet-general-lite"` | rembg model name. Config `model`, CLI `--model`. | [segment.py:22](../../background_changer/bg_changer/segment.py#L22), [config.toml:87-89](../../background_changer/config.toml#L87-L89), [change_background.py:715](../../background_changer/change_background.py#L715) |
| `DEFAULT_THREADS` | `4` | ONNX Runtime `intra_op_num_threads`; `0` lets ONNX Runtime decide. Config `threads`, CLI `--threads`. | [segment.py:23](../../background_changer/bg_changer/segment.py#L23), [config.toml:91-92](../../background_changer/config.toml#L91-L92), [segment.py:169-171](../../background_changer/bg_changer/segment.py#L169-L171) |
| providers | `["CPUExecutionProvider"]` | CPU only. No GPU provider is ever requested. | [segment.py:171](../../background_changer/bg_changer/segment.py#L171) |
| `only_mask` | `True` | The only option passed to `rembg.remove` | [segment.py:178](../../background_changer/bg_changer/segment.py#L178) |
| `BIN_DEG` | front 0, front-right 45, right 90, back-right 135, back 180, back-left 225, left 270, front-left 315 | The classifier's 8 view bins as yaw angles (degrees) | [yaw.py:16-17](../../background_changer/bg_changer/yaw.py#L16-L17) |
| view prior `sigma`, `floor` | 25°, 0.05 | Width of each bin's von Mises bump, and the floor under the mixture | [pose.py:176-183](../../background_changer/car_fit_to_bg/pose.py#L176-L183) |
| label fallback | `p = min(0.95, max(0.3, conf/100))`, `conf` 60 if missing | Probabilities from the file-name label when `results.csv` has no row for the photo | [pose.py:200-206](../../background_changer/car_fit_to_bg/pose.py#L200-L206), [batch.py:63-69](../../background_changer/car_fit_to_bg/batch.py#L63-L69) |
| `TAU` | 0.02 | IoU per nat: converts outline overlap into the same units as the priors | [pose.py:39](../../background_changer/car_fit_to_bg/pose.py#L39) |
| `CAPTURE_DISTANCE_M`, `CAPTURE_ELEVATION_DEG` | 4.70 m, 16.08° | Flat mode's fixed capture camera V (about 1.30 m high) | [carmodel.py:31-32](../../background_changer/bg_changer/carmodel.py#L31-L32) |
| `MIN_DEG` | 0.5° | Room mode: a levelling rotation smaller than this is skipped, so the car's pixels stay bit-identical | [level.py:25](../../background_changer/car_fit_to_bg/level.py#L25), [level.py:57-58](../../background_changer/car_fit_to_bg/level.py#L57-L58) |
| classifier `results.csv` path | `<repo>/frame_view_type/output/<images folder name>/results.csv` | Read when it exists; the `prob_<label>` columns are used. `angle_deg` is parsed but never used. | [batch.py:56-60](../../background_changer/car_fit_to_bg/batch.py#L56-L60), [pose.py:186-197](../../background_changer/car_fit_to_bg/pose.py#L186-L197) |

Formulas:

- **Raw alpha:** `raw_alpha = mask_u8 / 255`, resized back to the photo with `INTER_LINEAR` if rembg returns another size ([segment.py:179-181](../../background_changer/bg_changer/segment.py#L179-L181)).
- **View prior (room mode):** `logprior(yaw) = log(0.05 + sum over bins of p_bin * exp(k * (cos(yaw - bin_deg) - 1)))`, with `k = 1 / (25° in radians)^2 ≈ 5.25`. It is 0 when no probabilities are known ([pose.py:176-183](../../background_changer/car_fit_to_bg/pose.py#L176-L183)).
- **Label fallback:** the label's bin gets `p` and every other bin gets `(1 - p) / 7` ([pose.py:203-205](../../background_changer/car_fit_to_bg/pose.py#L203-L205)).
- **Pose score in nats (lower is better):** `score = e / 0.02 - logprior(yaw) + red_penalty`, where `e = -IoU + prior terms` ([pose.py:497-499](../../background_changer/car_fit_to_bg/pose.py#L497-L499)).
- **How much the classifier can move a pose:** `logprior` runs from `log 0.05 ≈ -3.0` to `log 1.05 ≈ +0.05`. That is at most about 3 nats, or 0.06 IoU. The outline decides, and the hint breaks near-ties such as front versus back.
- **Flat mode scale:** `s = min(1, px_per_m * h_rel(yaw) / H_photo)`, with `px_per_m = side_px * size / side_length` ([placement.py:109-114](../../background_changer/bg_changer/placement.py#L109-L114), [placement.py:133-137](../../background_changer/bg_changer/placement.py#L133-L137), [base.py:63](../../background_changer/bg_changer/scenes/base.py#L63)). For the EQB, `side_length ≈ 5.786` anchor metres ([test_carmodel.py:24](../../background_changer/tests/test_carmodel.py#L24)).
  > Note: the `side_length` docstring says "EQB: 5.809" ([carmodel.py:260](../../background_changer/bg_changer/carmodel.py#L260)). The unit test and a hand projection give ≈ 5.786.

## Options we had

### Cut-out model

All the candidates run on the CPU. The full comparison, speeds and thread findings are in [02-car-cutout-model.md](02-car-cutout-model.md).

| Option | Licence | Verdict | Why |
|---|---|---|---|
| **BiRefNet general-lite** | MIT | **Chosen (default)** | Clean wheel arches, mirrors and bonnet line. ~17 s per 4K photo at 4 threads, 220 MB. |
| BiRefNet general | MIT | Selectable (`model = "birefnet-general"`), not default | Looks the same as lite at the 1920×1080 output size, compared side by side and zoomed. 66-122 s per photo and 970 MB. Worth it only if the output size is raised a lot. |
| ISNet general-use | Apache-2.0 | Not used | Softer edges; can nibble thin parts |
| U²-Net | Apache-2.0 | Not used | Blobby edges; older, superseded |
| BRIA RMBG-2.0 | CC BY-NC 4.0 | Rejected | Very good edges, but a non-commercial licence |
| YOLO-seg (Ultralytics) | AGPL-3.0 | Rejected | Copyleft licence, and coarse edges from a low-resolution mask head |
| SAM / SAM 2 | Apache-2.0 | Not used | Coarse at 4K, slow encoder, needs a point or box prompt. Salient-object removal plus `clean_alpha` already isolates the car. |

Speeds were measured on the Intel i5-1240P laptop (CPU) ([findings](../../background_changer/docs/findings.md#speed-fewer-threads-is-faster-on-this-laptop)).

> Note: config.toml:88 and README.md:201 say `birefnet-general` is "about 4x slower". README.md:389 says "4-5×". The measured 66-122 s against ~17 s is about 4-7×.

> Note: the notes disagree on the slowest cut-out. README.md:383 gives ~12-17 s per photo typically, up to ~45 s. spyne-360-spin-findings.md:495 gives ≈ 13-17 s per frame, up to ~31 s.

### Placement approaches

| Option | Licence / cost | Verdict | Why |
|---|---|---|---|
| v1 "fit to a box" | Own code | **Dropped** | Stood every car on the wrong row (907, not the turntable centre at 764) and forced every view to one height |
| Flat mode: scale and move through a fixed capture camera | Own code, ~1 s per background | **Kept, opt-in** (`background_view = "flat"`), and the fallback for a background without a 3D room | Car pixels only scaled and moved. Right only for photos shot like the Mercedes walkaround. |
| **Room mode:** re-render the background from each photo's camera | Own code; fit ~10-20 s per photo, render ~0.2-1 s per background | **Chosen (default)** | Changes the floor's viewpoint, not the car's, so any camera height, distance or roll works. Car pixels get at most one rotation and one downscale. |
| Track A: 2D "virtual turntable" from SfM | COLMAP (BSD) etc. | Partly used | Its background calibration, true-centre anchor and grounding layers are in the code. SfM, frame choice by angle and the rotating disc were not built. |
| Track B: full 3D reconstruction (NeRF / Gaussian splatting), like Spyne | GPU or cloud; Inria 3DGS research-only, OpenSplat AGPL | **Not done** | Needs a CUDA GPU (~1.5-2 days per car on this CPU), fails on glossy paint and glass, and re-renders the car's pixels |

Speeds are for the i5-1240P laptop, CPU ([README](../../background_changer/README.md#speed)).

**v1: fit to a box (dropped).** The first version scaled each frame to fit a `max_width × max_height` box and stood its lowest pixel on one row. On background_1, 18 of 20 frames came out ≈ 648 px tall (the height cap) with the lowest pixel on row 907. The other two hit the width cap, and the scale swung 0.38-0.49 from frame to frame ([spyne notes §1, §5](../../background_changer/docs/spyne-360-spin-findings.md#5-why-our-output-looks-copy-pasted-gap-analysis)). It looked wrong for two reasons ([findings](../../background_changer/docs/findings.md#why-fit-to-a-box-looked-wrong)):
- **Wrong row.** Row 907 is 2.3 m in front of the turntable's true centre at row 764, where its 8 seams meet. The rim ellipse's centre, row 834, is itself 1.4 m nearer the camera. At row 907 the floor scale is 403 px/m instead of 261, so every car looked too big and too low.
- **One height for every view.** A real car turning in front of one camera changes height with the view. Spyne's spin of the same video shows side ≈ 520-560 px, front ≈ 676 and rear ≈ 776. Forcing every view to ≈ 648 px flattens that, and the car looks pasted on.

Nothing of v1 is left in the code. The background_1 scene docstring still records the old row ([background_1.py:6-7](../../background_changer/bg_changer/scenes/background_1.py#L6-L7)).
> Note: the gap table in spyne-360-spin-findings.md §5 ("Our `background_changer` today") describes v1, as it was on 2026-11-05. It is not the current code.

**Flat mode: scale and move (kept).** The car's pixels are only scaled down and moved. So the car keeps the walkaround's own viewpoint, close (≈ 3.4-4.6 m) and about 16° down ([findings](../../background_changer/docs/findings.md#two-cameras-that-meet-at-the-stage-centre)):
- **Car geometry:** the 3D proxy viewed through the fixed capture camera V, 4.70 m from the footprint centre and 16.08° down ([carmodel.py:3-9](../../background_changer/bg_changer/carmodel.py#L3-L9)). It reproduces Spyne's frames to ≈ 6 px RMS.
- **Angle:** the file-name label's bin centre, which can be off by up to ~24° ([yaw.py:1-5](../../background_changer/bg_changer/yaw.py#L1-L5)).
- **Size:** `s = min(1, px_per_m * h_rel(yaw) / H_photo)` (see the maths above). A side view of any car spans the scene's `side_px`.
- **Why it is not the default:** it works only for photos shot like the Mercedes walkaround ([README](../../background_changer/README.md#flat-mode)). `sample_vid` is 464×832 portrait, with an ultra-wide lens (f ≈ 366 px), shot 2-5 m away, 21-47° down and rolled up to 7°. A car seen from 20-47° above, on a floor seen from about 9°, floats whatever the scale and position ([findings](../../background_changer/docs/findings.md#room-mode-fitting-each-photo-and-re-rendering-the-background)).
- **Where it still runs:** `background_view = "flat"`. In room mode, a background without a 3D room is placed flat with the fitted yaw (sigma 10°) ([change_background.py:612-616](../../background_changer/change_background.py#L612-L616)).

**Room mode: re-render the background (chosen).** This is the default ([config.toml:35](../../background_changer/config.toml#L35), [change_background.py:721](../../background_changer/change_background.py#L721)). Each photo's camera is measured from its own lines, and the car's pose from its outline. Each background is rebuilt once as a simple 3D room from its measured camera: a box for background_1, curved walls for background_2, asphalt plus a backdrop for background_3. It is then ray-cast from the photo's camera. Details: [04](04-car-dimensions-and-orientation.md), [05](05-camera-angle.md), [06](06-3d-room-rendering.md).
- **Why chosen:** it changes the floor's viewpoint, not the car's. That is the only way to match a photo taken from high above or close up without resampling the car.
- **Checks:** background_1 re-renders its own picture exactly from its own camera, and background_2 within 1 grey level ([test_room3d.py:16-23](../../background_changer/tests/test_room3d.py#L16-L23)).
- **Fit accuracy (planning tests):** 0 front/back flips on 32 placeable frames. Yaw within 2.9° RMS (Baleno) and 1.4-1.7° (EQB) of a video-pooled fit. The worst case, a half-car close-up, was about 30° off and was caught by the margin test ([findings](../../background_changer/docs/findings.md#room-mode-fitting-each-photo-and-re-rendering-the-background)).
  > Note: the pose.py docstring says "tested on 34 frames without a front/back flip" ([pose.py:8](../../background_changer/car_fit_to_bg/pose.py#L8)), while findings.md says 32 placeable frames.
- **Price:** areas the plate never showed are invented fills. That is up to about half of some `sample_vid` frames, and 0-6% of the EQB frames. The plates are flat photos, so baked-in light streaks and far scenery stretch when seen from far away.
- **This is the compromise the Spyne notes suggest:** 3D for geometry only (poses, a proxy for shadows and reflections), never for car pixels ([spyne notes §6.3](../../background_changer/docs/spyne-360-spin-findings.md#63-track-b-a-full-3d-rebuild-like-spyne)).

**Track A: 2D virtual turntable (partly used).** The Spyne notes planned these steps ([§6.2](../../background_changer/docs/spyne-360-spin-findings.md#62-track-a-a-2d-virtual-turntable-from-real-frames)):
- calibrate each background once (GeoCalib, Perspective Fields or fSpy);
- measure every frame's viewpoint with COLMAP SfM;
- choose frames by angle;
- scale by `s_i = m_bg * z_i / f_i`;
- anchor the footprint centre on the true disc centre;
- rotate the turntable texture;
- ground the car with layers.

What was built:
- **Background calibration:** a read-only measurement from each picture's own lines and objects, with no learned calibration model ([findings](../../background_changer/docs/findings.md#calibrating-the-backgrounds-from-the-pictures-alone)).
- **Anchor:** the true disc centre (958.8, 764.0) ([background_1.py:35](../../background_changer/bg_changer/scenes/background_1.py#L35)).
- **Grounding layers:** drawn under the car (see [06](06-3d-room-rendering.md)).

What was not built:
- **SfM:** each photo is fitted on its own, from its lines and outline, so no overlapping frames are needed.
- **Frame choice by angle:** frame_view_type still takes one frame every N seconds.
- **Disc rotation:** no code rotates the turntable texture.

**Track B: full 3D reconstruction, like Spyne (not done).** Spyne's own wiki names Gaussian splatting and NeRF for its spins. A 3D rebuild would give the most natural result and could show angles the video cut off. It was not built for these reasons ([§6.3](../../background_changer/docs/spyne-360-spin-findings.md#63-track-b-a-full-3d-rebuild-like-spyne), [§7.7](../../background_changer/docs/spyne-360-spin-findings.md#77-3d-reconstruction-track-b-evidence), [§7.9](../../background_changer/docs/spyne-360-spin-findings.md#79-cpu-feasibility-on-this-laptop-i5-1240p-16-gb-no-gpu), [§8](../../background_changer/docs/spyne-360-spin-findings.md#8-open-decisions-for-the-team)):
- **Compute.** It needs a CUDA GPU or a cloud service. On the i5-1240P laptop it would take roughly 1.5-2 days per car. Inria 3DGS needs CUDA (24 GB for paper quality) and is research-only. OpenSplat runs on CPU but is ~100× slower and AGPL. Brush (Apache-2.0, WebGPU) is unbenchmarked on the laptop's Iris Xe. In the feasibility table, "VGGT, 3DGS training, IC-Light, diffusion" are all "No: need CUDA, or days on CPU".
- **Our cars are the worst case.** Glossy paint and glass are known failures. In 3DGS-DR, a front window converges to a purely reflective surface and the side windows to purely transparent. A white glossy car with glass is the worst case for photogrammetry too.
- **It re-renders the car's pixels.** Rendered views reach only ~28.7 dB PSNR (28.66 dB in a 2026 dealership pipeline with 14 fixed 4K cameras). At that level renders are visibly not identical to the photos, so fine dents and scratches can be softened or changed. Spyne's own output changes the car: in its frame 18 the input's sky reflections are gone ([§3.3](../../background_changer/docs/spyne-360-spin-findings.md#33-appearance-spyne-changes-the-car-itself)).
- **Still open.** "Trial a GPU or cloud 3D rebuild at all?" is a team decision. The suggested order is to trial it later, only with a GPU and a damage-fidelity check (§6.4, §8).

### Learned camera and viewpoint helpers (looked at, not used)

| Option | Licence | Verdict | Why |
|---|---|---|---|
| GeoCalib (ECCV 2024) | Weights CC-BY-4.0 (code licence not checked) | Not used | Intrinsics and gravity from one image, with a CPU fallback. The three backgrounds were measured once by hand, and each photo uses LSD + MSAC ([05](05-camera-angle.md)). |
| Perspective Fields (CVPR 2023) | Not established | Not used | Roll, pitch and FOV from one image. Its demo hard-codes `.cuda()`. |
| fSpy | GPL | Not used | Manual vanishing points |
| COLMAP / pycolmap SfM (GLOMAP mapper) | BSD | Not used | Track A's per-frame viewpoint. Sparse SfM runs on CPU, but mapping time is unmeasured and a glossy close-range clip may not register (spyne notes §9). |
| VGGT | Only its "Commercial" checkpoint | Not used | Needs CUDA |
| Render for CNN; Orient Anything V2 | Not checked | Not used | ~6° and ~15° median viewpoint error; both confuse front and back. On our frames the silhouette fit, with the classifier prior and the tail-lamp cue, had 0 front/back flips and 1.4-2.9° RMS yaw error (see room mode above). |

Sources: [spyne notes §7.2-§7.3, §7.9-§7.10](../../background_changer/docs/spyne-360-spin-findings.md#72-geometry-calibration-the-turntable-model-and-the-limits-of-2d).

## What the "car pixels must not change" rule rules out

**The rule.** The car's exterior must not change. Dents, scratches, smudges and dirt are the point of these photos ([findings](../../background_changer/docs/findings.md#the-job)). On glossy paint, a dent is seen as a distortion of the reflected scene. So anything that re-renders shading or reflections can hide, move or invent dents ([spyne notes §7.6](../../background_changer/docs/spyne-360-spin-findings.md#76-what-not-to-use-and-why-damage-fidelity)).

| Ruled out | Why |
|---|---|
| Generative fill, inpainting, "AI enhance" | Invents pixels. It can smooth away a dent or add a reflection that was never there. Outpainting and diffusion novel views also invent pixels. Stable Virtual Camera's model card says it is "not trained to be factual". |
| Relighting and harmonisation (IC-Light and similar) | Re-renders the car's shading to match the new scene, which changes how damage looks. IC-Light is latent diffusion: detail loss, colour bleeding, per-frame flicker; it needs a GPU, and V2 is non-commercial. DoveNet regenerates the whole image at 256 px; PCT-Net applies per-pixel colour transforms; Harmonizer applies only global filters but is non-commercial. |
| Colour or exposure matching to the background | Changes the car's colour, and so how scratches and paint defects read. Reinhard-style colour transfer pulls the paint toward the studio's colour. |
| Sharpening, denoising, upscaling (super-resolution) | Changes texture and can hide or exaggerate fine scratches. Super-resolution generates detail. |
| Dehaze and window tint (Spyne's car edits) | They replace car pixels. A cracked windscreen is damage. |
| Frame interpolation (FILM, RIFE) to fill missing angles | Synthesises regions never seen. Optical flow breaks on specular paint, glass, grilles and spokes. |
| Stitching neighbouring frames, flow-guided video completion (ProPainter) | Stitching is exact only for planar scenes or pure rotation: wheels and mirrors ghost, and a seam can duplicate or cut a scratch. ProPainter still hallucinates what was never seen; it is non-commercial and needs about 19-28 GB of GPU memory. |

Sources: [findings](../../background_changer/docs/findings.md#the-job) (first four rows), [spyne notes §3.3, §7.5, §7.6](../../background_changer/docs/spyne-360-spin-findings.md#76-what-not-to-use-and-why-damage-fidelity).

**What the car's pixels do get.** Every edit apart from these goes on the background, before the car is pasted.

| Mode | Operation on car pixels | Code | Checked by |
|---|---|---|---|
| Flat | One uniform `INTER_AREA` downscale, never up. Alpha above `1 - 1e-4` snaps to 1, so those pixels are copied exactly. | [compose.py:46-57](../../background_changer/bg_changer/compose.py#L46-L57), [compose.py:41-42](../../background_changer/bg_changer/compose.py#L41-L42) | [test_compose.py:61](../../background_changer/tests/test_compose.py#L61) |
| Room | At most one rigid rotation about the frame centre to level the photo, skipped below 0.5°. Colour is premultiplied Lanczos-4, clamped to the 3×3 source range (no new colours, no dark fringes); alpha is bilinear. Then one `INTER_AREA` downscale, never up. On the soft edge only, colour is alpha-weighted. | [level.py:1-13](../../background_changer/car_fit_to_bg/level.py#L1-L13), [level.py:53-94](../../background_changer/car_fit_to_bg/level.py#L53-L94), [level.py:97-118](../../background_changer/car_fit_to_bg/level.py#L97-L118) | [test_car_fit.py:57](../../background_changer/tests/test_car_fit.py#L57), [test_car_fit.py:100](../../background_changer/tests/test_car_fit.py#L100), [test_room_mode.py:143](../../background_changer/tests/test_room_mode.py#L143) |

The plates arrive already masked by number_plate_detection. That is the one upstream edit to the car.

> Note: findings.md:84 says the car's pixels may only be scaled and moved, and findings.md:157 says tilted frames cannot be straightened under that rule. Both describe flat mode. Room mode now allows one levelling rotation (findings.md:20; [level.py:1-2](../../background_changer/car_fit_to_bg/level.py#L1-L2)).

**Open decisions from the Spyne notes ([§8](../../background_changer/docs/spyne-360-spin-findings.md#8-open-decisions-for-the-team)) and where the code stands:**

| Decision | In the code today |
|---|---|
| (a) Rotation-homography re-aiming | Roll only: room mode's one in-plane rotation (`Rz` about the optical axis, [level.py:10-12](../../background_changer/car_fit_to_bg/level.py#L10-L12)). No pitch or yaw re-aim. |
| (b) One global brightness gain per frame | Not built. No gain is applied to car pixels. |
| (c) Colour clean-up of the semi-transparent edge band only | Not built (see the next section) |
| (d) Glass tint or see-through | Not built. Windows are filled solid, so the original glass pixels are kept ([segment.py:122-124](../../background_changer/bg_changer/segment.py#L122-L124)). |
| (e) Mirrored edge fill near the car | Not built. Cut-off cars stay cut off. |

## Edge matting and glass: noted, not built

The Spyne notes list ways to improve the cut-out's edge and glass ([§7.5](../../background_changer/docs/spyne-360-spin-findings.md#75-grounding-and-appearance-with-the-car-layer-locked)). The code has no matting model, no decontamination step, no guided filter and no glass segmentation. All edge handling is classical: see [03-cutout-refinement.md](03-cutout-refinement.md).

| Idea | Tools named in the notes | Status in code |
|---|---|---|
| Recompute colour only in the `0 < alpha < 1` band (foreground estimation). `alpha = 1` pixels come out unchanged, so the notes call it safe for damage. | Photoroom fast-foreground-estimation, PyMatting, BiRefNet's `refine_foreground`, rembg's `decontaminate` | Not built. rembg is called with `only_mask=True` only ([segment.py:178](../../background_changer/bg_changer/segment.py#L178)). Open decision (c). |
| Refine the mask at 4K with a guided filter instead of eroding the edge | Guided filter | Not built. The soft edge is eroded by `edge_shrink` = 2 px instead ([segment.py:134-136](../../background_changer/bg_changer/segment.py#L134-L136)), which the notes say trims real edge pixels. |
| Resample premultiplied | n/a | Room mode only: alpha-weighted Lanczos-4 rotation and alpha-weighted `INTER_AREA` on the edge ([level.py:76-87](../../background_changer/car_fit_to_bg/level.py#L76-L87), [level.py:110-116](../../background_changer/car_fit_to_bg/level.py#L110-L116)). Flat mode still resizes colour and straight alpha separately ([compose.py:52-54](../../background_changer/bg_changer/compose.py#L52-L54)). |
| Clear semi-transparent old-ground haze below the contact line; keep the gap under the body transparent | n/a | No separate step. Soft alpha is only kept within `EDGE_KEEP` of the outline ([03](03-cutout-refinement.md)). |
| Glass masks, for tint or see-through windows | HITL car-parts dataset (CC0; window, plate and damage classes), Ultralytics carparts-seg (AGPL; front and back glass only), CLIPSeg or Grounded-SAM zero-shot on CPU | Not built; the notes mark it "needs approval". Windows are filled solid instead. |

> Note: spyne-360-spin-findings.md:426 says "our code today resizes RGB and straight alpha separately, then blends in sRGB". That is now true for flat mode only.

> Note: the "1024² mask" in spyne-360-spin-findings.md:429, like README.md:389's "the model works at its own fixed size", comes from the notes. No code in the repo sets the model's input size.

## Limits

- **Edge quality rests on one model.** No matting or refinement model sharpens BiRefNet's mask. The notes put its working size at about 1024², so a 4K mask is upsampled.
- **The view hint is only as good as frame_view_type.** Labels are off by up to ~24° on the EQB video and by up to 177° on `sample_vid` ([findings](../../background_changer/docs/findings.md#room-mode-fitting-each-photo-and-re-rendering-the-background)). Room mode fits the yaw itself and gives the hint at most ~3 nats. With neither a label nor `results.csv`, the outline alone decides, and the photo is flagged `no-view-label` ([batch.py:92-93](../../background_changer/car_fit_to_bg/batch.py#L92-L93)).
- **No relighting.** The car keeps its original lighting and reflections, so a car shot in sunshine looks sunlit on a soft-lit studio background ([findings](../../background_changer/docs/findings.md#known-limits)).
- **No synthesis.** Cut-off cars stay cut off. In room mode, a car cut off at the bottom or on two sides is not placed ([batch.py:124-131](../../background_changer/car_fit_to_bg/batch.py#L124-L131)). Only a 3D rebuild (Track B) could show those angles.
- **The proxy is one shape.** The EQB shape is stretched to each car, with three roof presets (suv, hatch, sedan). Unusual bodies and half-car close-ups can be fitted some degrees off.
- **The rooms are simple.** A box, extruded walls, one ceiling plane or a far backdrop. Everything the plate never showed is a rule-based fill, flagged `invented` when there is a lot of it.
- **CPU only.** The cut-out (~12-17 s per photo) and the 3D fit (~10-20 s per photo) take nearly all of the time (i5-1240P, [README](../../background_changer/README.md#speed)).

## Further reading

- [docs/background_changer/README.md](README.md): overview and index of these docs.
- [02-car-cutout-model.md](02-car-cutout-model.md): the BiRefNet cut-out model, speeds and threads.
- [03-cutout-refinement.md](03-cutout-refinement.md): the classical mask clean-up.
- [04-car-dimensions-and-orientation.md](04-car-dimensions-and-orientation.md): `cars.json`, the 3D proxy, yaw.
- [05-camera-angle.md](05-camera-angle.md): roll, pitch, focal, distance, height, levelling, tyre lock.
- [06-3d-room-rendering.md](06-3d-room-rendering.md): rooms, output camera, ray-casting, fills, grounding, composite.
- [background_changer/docs/findings.md](../../background_changer/docs/findings.md): the rule, the cut-out comparison, and the flat and room mode history.
- [background_changer/docs/spyne-360-spin-findings.md](../../background_changer/docs/spyne-360-spin-findings.md): the Spyne study, Tracks A and B, and the research on what not to use.
- [background_changer/README.md](../../background_changer/README.md): how to run, settings, speed and known limits.
- [frame_view_type/01-model-and-options.md](../frame_view_type/01-model-and-options.md): the view classifier.
- [number_plate_detection/01-model-and-options.md](../number_plate_detection/01-model-and-options.md): the plate detector.
- [video_pipeline/README.md](../video_pipeline/README.md) and [docs/README.md](../README.md): how the three projects chain.
