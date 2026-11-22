# Plate detector: model and options

## In short

- One small model finds the plates: `yolo-v9-t-384-license-plate-end2end` from [open-image-models](https://github.com/ankandrew/open-image-models) (MIT). It is a single-class ("License Plate") YOLOv9-tiny with NMS inside the ONNX graph. The file is 7.4 MB.
- It runs on ONNX Runtime's `CPUExecutionProvider` with 4 intra-op threads and a confidence threshold of 0.25. No GPU, no torch.
- The whole image goes in. The model returns straight boxes; plain OpenCV then finds the plate's own corners ([02-exact-plate-fit.md](02-exact-plate-fit.md)).
- On the 20 test frames (4K) it found the main car's plate in all 12 frames where it is visible, in about 50-63 ms per frame on a laptop CPU. Two small background plates need `confidence = 0.15`.

## How it works

1. **Settings.** `model`, `confidence` and `threads` come from `config.toml` ([config.toml:45-55](../../number_plate_detection/config.toml#L45-L55)). Command-line `--model`, `--conf` and `--threads` win over the file ([mask_plates.py:165-168](../../number_plate_detection/mask_plates.py#L165-L168)). A missing key falls back to the code default ([mask_plates.py:189-192](../../number_plate_detection/mask_plates.py#L189-L192)).
2. **Checks.** `confidence` must be strictly between 0 and 1, and `threads` must be 0 or more ([mask_plates.py:200-205](../../number_plate_detection/mask_plates.py#L200-L205)). In `config.toml`, `model` must be quoted text and `threads` a whole number ([mask_plates.py:71-73](../../number_plate_detection/mask_plates.py#L71-L73), [mask_plates.py:87-89](../../number_plate_detection/mask_plates.py#L87-L89)). A `.onnx` path in `config.toml` is taken relative to the config's folder; a model name is left as it is ([mask_plates.py:80-82](../../number_plate_detection/mask_plates.py#L80-L82)).
3. **Load once.** `PlateMasker` builds one `PlateDetector` and keeps it for every image ([\_\_init\_\_.py:94](../../number_plate_detection/plate_masker/__init__.py#L94)). `onnxruntime` and `open_image_models` are imported only here ([detector.py:81-82](../../number_plate_detection/plate_masker/detector.py#L81-L82)).
   - **Local file or registered name.** A model ending in `.onnx` (any case) is a local file; it must exist, or `FileNotFoundError` is raised ([detector.py:84-86](../../number_plate_detection/plate_masker/detector.py#L84-L86)). Any other value is an open-image-models name.
   - **Threads.** With `threads > 0` an `ort.SessionOptions` is made with `intra_op_num_threads = threads` and `inter_op_num_threads = 1`. With `threads = 0` no options are passed, so ONNX Runtime picks its own thread count (all cores) ([detector.py:89-93](../../number_plate_detection/plate_masker/detector.py#L89-L93)).
   - **Create.** `open_image_models.create_detector(model, conf_thresh=conf, providers=["CPUExecutionProvider"], sess_options=...)`. A local file also gets `backend="yolo_v9"` and `class_labels=["License Plate"]` ([detector.py:94-101](../../number_plate_detection/plate_masker/detector.py#L94-L101)).
   - **First run downloads the weights** (7.4 MB) from the open-image-models GitHub releases into `%USERPROFILE%\.cache\open-image-models\<model>\`. After that it works offline ([README.md:16](../../number_plate_detection/README.md#L16), [findings.md:41](../../number_plate_detection/docs/findings.md#L41)). The download took about 24 s here ([findings.md:80](../../number_plate_detection/docs/findings.md#L80)).
   - **Download fails** (for example a certificate error on a company network): the `urllib.error.URLError` becomes a `RuntimeError` whose message gives the URL from open-image-models' `DETECTION_MODELS`, and says to save the file as `models\<file name>` in the `number_plate_detection` folder and set `model = "models/<file name>"` ([detector.py:102-103](../../number_plate_detection/plate_masker/detector.py#L102-L103), [detector.py:118-135](../../number_plate_detection/plate_masker/detector.py#L118-L135)). `mask_plates.py` prints that message and stops ([mask_plates.py:224-230](../../number_plate_detection/mask_plates.py#L224-L230)).
4. **Detect.** `detect(image_bgr)` passes the full BGR image to the model ([detector.py:105-109](../../number_plate_detection/plate_masker/detector.py#L105-L109)). Inside open-image-models the whole image is shrunk to the model's input size (384 px for the default), so frame resolution barely changes detection time ([findings.md:40](../../number_plate_detection/docs/findings.md#L40), [findings.md:79](../../number_plate_detection/docs/findings.md#L79)). NMS runs inside the ONNX graph ("end2end"), and boxes below `conf_thresh` are dropped by the package.
5. **Clean the boxes.** Each box is cut to whole pixels with `int()` and clipped to the image. Boxes with no width or no height are dropped. The rest are sorted by confidence, highest first ([detector.py:110-115](../../number_plate_detection/plate_masker/detector.py#L110-L115)). There is no size or shape filter at this stage; the shape checks happen in the corner fit.
6. **Hand on.** `PlateMasker.process()` calls `detect()`, then the corner fit, the brightness reading and the mask ([\_\_init\_\_.py:100-114](../../number_plate_detection/plate_masker/__init__.py#L100-L114)). Those steps are in [02-exact-plate-fit.md](02-exact-plate-fit.md), [03-multi-colour-plates.md](03-multi-colour-plates.md) and [04-auto-brightness.md](04-auto-brightness.md).

### What `detect()` returns

A `list[Plate]`, highest confidence first. An image with no plate gives an empty list. `Plate` is a frozen dataclass ([detector.py:22-64](../../number_plate_detection/plate_masker/detector.py#L22-L64)):

| Field | Straight from `detect()` | Meaning |
|---|---|---|
| `x1, y1, x2, y2` | `int`, clipped to the image | Detector box in original-image pixels; `x2`/`y2` exclusive ([detector.py:26](../../number_plate_detection/plate_masker/detector.py#L26)) |
| `confidence` | `float` | Model score; boxes under `conf` never reach this list |
| `corners` | `None` | Filled in later by `fit_plate()`: 4 points, top-left, top-right, bottom-right, bottom-left |
| `brightness` | `None` | Filled in later when auto brightness is on |
| `outline_from` | `""` | Filled in later: `"corners"` or `"letters"` |
| `.fit` (property) | `"box"` | `"box"` while `corners` is `None`, else `outline_from` or `"corners"` ([detector.py:51-56](../../number_plate_detection/plate_masker/detector.py#L51-L56)) |
| `.quad` (property) | the box's 4 corners | `float32` shape (4, 2): the fitted corners, or `(x1, y1), (x2-1, y1), (x2-1, y2-1), (x1, y2-1)` ([detector.py:58-64](../../number_plate_detection/plate_masker/detector.py#L58-L64)) |

`PlateDetector` can be used on its own (find only, no masking).

## The maths and constants

| Name | Value | Meaning | Code link |
|---|---|---|---|
| `DEFAULT_MODEL` | `"yolo-v9-t-384-license-plate-end2end"` | Default detector; 384 = input size in px | [detector.py:14](../../number_plate_detection/plate_masker/detector.py#L14), [config.toml:48](../../number_plate_detection/config.toml#L48) |
| `DEFAULT_CONF` | `0.25` | Passed as `conf_thresh`; open-image-models drops boxes scoring below it. It is the model's own default. | [detector.py:16](../../number_plate_detection/plate_masker/detector.py#L16), [config.toml:51](../../number_plate_detection/config.toml#L51) |
| `confidence` range | `0 < conf < 1` | Checked by `mask_plates.py` only, not by `PlateDetector` | [mask_plates.py:200-201](../../number_plate_detection/mask_plates.py#L200-L201) |
| `DEFAULT_THREADS` | `4` | ONNX Runtime intra-op threads | [detector.py:19](../../number_plate_detection/plate_masker/detector.py#L19), [config.toml:55](../../number_plate_detection/config.toml#L55) |
| `threads = 0` | no `SessionOptions` | ONNX Runtime's own default: every core | [detector.py:89-90](../../number_plate_detection/plate_masker/detector.py#L89-L90) |
| `inter_op_num_threads` | `1` | Set whenever `threads > 0` | [detector.py:93](../../number_plate_detection/plate_masker/detector.py#L93) |
| `providers` | `["CPUExecutionProvider"]` | CPU only, always | [detector.py:98](../../number_plate_detection/plate_masker/detector.py#L98) |
| Local model test | name ends in `.onnx` (case-insensitive) | Treated as a file; must exist | [detector.py:84-86](../../number_plate_detection/plate_masker/detector.py#L84-L86) |
| Local model settings | `backend="yolo_v9"`, `class_labels=["License Plate"]` | Tells open-image-models how to read a plain `.onnx` | [detector.py:100](../../number_plate_detection/plate_masker/detector.py#L100) |
| Box clip | `x = max(0, min(w, int(b.x)))`, `y = max(0, min(h, int(b.y)))` | Keeps boxes inside the image | [detector.py:111-112](../../number_plate_detection/plate_masker/detector.py#L111-L112) |
| Empty-box rule | keep only if `x2 > x1 and y2 > y1` | Drops boxes clipped to nothing | [detector.py:113](../../number_plate_detection/plate_masker/detector.py#L113) |
| Sort | `confidence`, descending | Plate 1 in `detections.csv` is the surest one | [detector.py:115](../../number_plate_detection/plate_masker/detector.py#L115) |
| Model names allowed | `yolo-v9-t-256 / -384 / -416 / -512 / -640-license-plate-end2end`, `yolo-v9-s-608-license-plate-end2end`, or a `.onnx` path | Set with `model` or `--model` | [config.toml:45-48](../../number_plate_detection/config.toml#L45-L48) |
| Dependencies | `open-image-models[onnx]==0.6.0`, `opencv-python-headless`, `numpy` | No torch. The whole venv is about 250 MB ([findings.md:81](../../number_plate_detection/docs/findings.md#L81)). | [requirements.txt:2-4](../../number_plate_detection/requirements.txt#L2-L4) |

### The six model sizes

All six are single-class ("License Plate") YOLOv9 models with NMS in the ONNX graph. The number in the name is the input size: the whole image is shrunk to it, so a bigger size finds smaller plates but costs more time. Precision, recall and mAP are the open-image-models project's own figures, not ours ([findings.md:29-38](../../number_plate_detection/docs/findings.md#L29-L38)).

| Model | Input | Precision | Recall | mAP50 | mAP50-95 | Detect time, 4K frame (ours) |
|---|---|---|---|---|---|---|
| `yolo-v9-t-256-license-plate-end2end` | 256 | 0.937 | 0.797 | 0.858 | 0.606 | 30 ms |
| **`yolo-v9-t-384-license-plate-end2end`** (default) | 384 | 0.942 | 0.863 | 0.920 | 0.687 | **63 ms** |
| `yolo-v9-t-416-license-plate-end2end` | 416 | 0.940 | 0.894 | 0.940 | 0.702 | not measured |
| `yolo-v9-t-512-license-plate-end2end` | 512 | 0.955 | 0.901 | 0.948 | 0.724 | 102 ms |
| `yolo-v9-t-640-license-plate-end2end` | 640 | 0.966 | 0.896 | 0.958 | 0.758 | 151 ms |
| `yolo-v9-s-608-license-plate-end2end` | 608 | 0.957 | 0.917 | 0.966 | 0.772 | not measured |

Times: Intel i5-1240P laptop, CPU, 4 threads, median of 48 runs, other apps open ([findings.md:69-76](../../number_plate_detection/docs/findings.md#L69-L76)). The 416 and s-608 models were never run on our frames.

## Results on our frames

**Test set:** the 20 frames from `frame_view_type` (`Final_zoomed_car.mp4`, 3840x2160, a white Mercedes EQB on a dealer lot). Every detection was checked by eye at full resolution ([findings.md:47-52](../../number_plate_detection/docs/findings.md#L47-L52)).

- The **main car's plate** is visible in 12 frames.
- **Two readable background plates** sit on dark cars behind showroom glass (`0014`, `0016`). They are about 150 px wide in a 3840 px frame, and dim.
- **Dealer plate holders** are plate-shaped but carry no registration; masking them does no harm.

| Model, threshold | Main plates found (of 12) | Background plates found (of 2) | False hits | Dealer holders masked |
|---|---|---|---|---|
| 256, 0.25 | 12 | 1 (loose box) | 3 (chrome trim x2, grille) | 0 |
| **384, 0.25 (default)** | **12** | **0** | **1** (chrome trim, 33 %) | 1 |
| 512, 0.25 | 12 | 1 | 1 (grille) | 0 |
| 640, 0.25 | 12 | 1 | 1 (grille) | 2 |
| 384, 0.15 | 12 | 2 | 3 (chrome trim, car body, badge) | 1 |

Source: [findings.md:54-60](../../number_plate_detection/docs/findings.md#L54-L60).

- **Every model found every visible main-car plate.**
- **False hits** are small patches on chrome trim, grilles or badges. All were below 70 % confidence, most below 35 % ([findings.md:63](../../number_plate_detection/docs/findings.md#L63)).
- **Only 384 at 0.15 caught both background plates.** The bigger models do not fix this at 0.25. A lower threshold does, at no extra time ([findings.md:64](../../number_plate_detection/docs/findings.md#L64)).
- **Why 384 at 0.25:** the cheapest setting that finds every main-car plate, with one small false patch. Where background plates must be hidden too, set `confidence = 0.15` ([findings.md:65](../../number_plate_detection/docs/findings.md#L65)).
- **After the fit:** the corners were found on all 12 main plates. The two non-plates (the chrome-trim hit and a dark dealer holder) fell back to a small solid box ([README.md:176-178](../../number_plate_detection/README.md#L176-L178)).
- **Second video, `sample_vid`** (464x832 phone video, black Indian plates about 80x25 px): the plate was found in all 8 frames where one is visible ([findings.md:180-182](../../number_plate_detection/docs/findings.md#L180-L182)).

> Note: the model comparison above was run with the first version's straight box plus 10 % padding. The tight fit came later and does not change which plates are found ([findings.md:62](../../number_plate_detection/docs/findings.md#L62)).

## Speed

All figures were measured on an Intel i5-1240P laptop, CPU only (4 performance + 8 efficiency cores), with other apps open.

**Full run, default settings** ([findings.md:78](../../number_plate_detection/docs/findings.md#L78), [README.md:167-172](../../number_plate_detection/README.md#L167-L172)):

| Step | Time |
|---|---|
| Find + fit + cover, per 4K frame | about 85 ms |
| of which: detector | about 50 ms |
| of which: corner fit | about 8 ms per plate |
| of which: overlay (solid fill) | about 18 ms (about 10 ms) |
| JPEG read / write, per 4K frame | about 80 ms / about 47 ms |
| 20 frames, whole run | about 4.3 s |
| Model load | 0.7 s ([findings.md:80](../../number_plate_detection/docs/findings.md#L80)) |

> Note: README.md:169 says the model loads "in about 1 s". The detector time is about 50 ms in the full-run breakdown, 63 ms in the model-size table and 56 ms in the thread table below. These are separate measurement runs, not a code difference.

**Threads matter more than model size.** Same 384 model, same frame, runs interleaved ([findings.md:83-92](../../number_plate_detection/docs/findings.md#L83-L92)):

| ONNX Runtime threads | Median | Best |
|---|---|---|
| 1 | 215 ms | 104 ms |
| 2 | 79 ms | 60 ms |
| **4 (default)** | **56 ms** | 49 ms |
| all cores (`threads = 0`, ONNX Runtime default) | 229 ms | 130 ms |

- **Why 4:** with all 12 cores the fast cores wait on the slow efficiency cores and fight other apps for time. All cores was about 4x slower ([detector.py:17-18](../../number_plate_detection/plate_masker/detector.py#L17-L18)).
- **Other machines:** try 2, 4 and 0. Use 1-2 to leave the CPU free for other work ([config.toml:53-55](../../number_plate_detection/config.toml#L53-L55)).
- The background changer's cut-out model showed the same effect on this laptop ([../background_changer/02-car-cutout-model.md](../background_changer/02-car-cutout-model.md)).

## Options we had

| Option | Licence | Model / compute | Verdict | Why |
|---|---|---|---|---|
| **open-image-models plate detectors** | MIT | YOLOv9-tiny/small, ONNX, six sizes (256-640 px input); 7.4 MB; about 50-63 ms per 4K frame at 4 threads | **Chosen** | Smallest compute, permissive licence, plain pip install, API returns boxes directly, no torch. GPU, OpenVINO and DirectML are optional extras. |
| [fast-alpr](https://github.com/ankandrew/fast-alpr) | MIT | Same detector plus an OCR model | Not used | We only mask, we don't read. Using the detector package directly skips the OCR model. |
| [platedet](https://pypi.org/project/platedet) | Not checked | YOLO converted to ONNX, CPU by default | Not used | Trained mostly on Brazilian plates. |
| [LiteALPR](https://huggingface.co/anhone3/LiteALPR) | **AGPL-3.0** | YOLOv8n with Ghost convolutions, about 23 FPS on CPU (ONNX FP32) | Rejected | AGPL is a problem for closed products. Trained mainly on Vietnamese plates. |
| [Meta EgoBlur](https://projectaria.com/tools/egoblur) | Apache-2.0, behind a model licence agreement | Faster R-CNN, about 104M parameters, about 400 MB; Meta's figure about 8 s per 2 MP image on CPU | Rejected | Accurate, but hundreds of times too heavy for low compute. |
| [understand-ai/anonymizer](https://github.com/understand-ai/anonymizer) | Apache-2.0 | Faces and plates | Rejected | Archived. Its own notes say it struggles with low-quality images. |
| [PeekingDuck](https://peekingduck.readthedocs.io/en/stable/use_cases/privacy_protection_license_plates.html) | Apache-2.0 | YOLOv4 family, pipeline framework | Rejected | Older model, and it means adopting a whole framework. |
| [ippen/image-anonymizer](https://github.com/ippen/image-anonymizer) | Not checked | YOLO + Gaussian blur, Streamlit app | Rejected | An app, not a module to import. |
| OpenCV Haar cascade `haarcascade_russian_plate_number.xml` | BSD (ships with OpenCV) | Classic cascade, near-zero compute, nothing to download | Not used | Tuned for Russian plates; many false hits and misses. |
| RF-DETR COCO detectors (in open-image-models) | Not checked (the package is MIT; the RF-DETR weights come from Roboflow) | General object detector that could find the car first | Not used | Not needed: the plate model runs on the whole image. |
| Bigger plate model (512 / 640) | MIT | 1.6x / 2.4x the 384 time | Not default | Still missed one background plate at 0.25. Lowering the threshold on 384 caught both, for free. |

Sources: [findings.md:13-25](../../number_plate_detection/docs/findings.md#L13-L25), [findings.md:43](../../number_plate_detection/docs/findings.md#L43), [findings.md:54-65](../../number_plate_detection/docs/findings.md#L54-L65).

> Note: findings.md:25 calls the Haar cascade a "last-resort fallback". No Haar fallback exists in the code; a missed plate is simply not masked.

## Unknowns

- **Training data is not published**: which countries' plates the model saw. Square two-line plates or unusual formats may be found less reliably. Test on real footage from each market ([findings.md:42](../../number_plate_detection/docs/findings.md#L42), [findings.md:124](../../number_plate_detection/docs/findings.md#L124)).
- **CPU speed is not published.** All times above are our own.
- **416 and s-608** were not run on our frames.
- **Best thread count on other CPUs** is unknown. The default of 4 is tuned for a 4 P + 8 E core laptop.

## Limits

- **A missed plate is a privacy leak; a wrong mask is cheap.** A plate the detector misses is not covered at all, and nothing downstream catches it. Always check the output. Lower `confidence` first (it costs no time); a bigger model helps less and costs 2-3x the time ([findings.md:121-122](../../number_plate_detection/docs/findings.md#L121-L122), [README.md:184](../../number_plate_detection/README.md#L184)).
- **No tracking over time.** Each frame is handled on its own. A plate found in one frame and missed in the next will flicker. Carrying boxes over from neighbouring frames would fix that; only the overlay brightness is smoothed over frames ([findings.md:126](../../number_plate_detection/docs/findings.md#L126), see [04-auto-brightness.md](04-auto-brightness.md)).
- **Small, distant or dim plates** are the weak spot ([README.md:190](../../number_plate_detection/README.md#L190)).
- **False hits get a solid box.** A false hit on trim or a grille usually fails the corner fit and is covered with a small solid box in `color` (black by default) ([02-exact-plate-fit.md](02-exact-plate-fit.md)).
- **Little test footage.** One 4K video (one car, one dealer lot, daylight) plus the low-resolution `sample_vid`. Night, rain, motion blur and other countries' plates are untested ([README.md:191](../../number_plate_detection/README.md#L191)).
- **The video pipeline inherits these settings.** `video_pipeline/make_video.py` passes no `--model`, `--conf` or `--threads`, so `number_plate_detection/config.toml` decides them for the video too ([make_video.py:133-134](../../video_pipeline/make_video.py#L133-L134); see [../video_pipeline/README.md](../video_pipeline/README.md)).
- **A `--model` path on the command line** is used as given (relative to the current folder). Only a `.onnx` path inside `config.toml` is resolved against the config's folder ([mask_plates.py:80-82](../../number_plate_detection/mask_plates.py#L80-L82), [mask_plates.py:189](../../number_plate_detection/mask_plates.py#L189)).

> Note: README.md:9 says "Needs Python 3.10+", but `mask_plates.py` imports `tomllib` ([mask_plates.py:20](../../number_plate_detection/mask_plates.py#L20)), which needs Python 3.11+. The setup commands use `py -3.11`.

## Further reading

- [README.md](README.md): overview of the plate-masking project and index of these notes.
- [02-exact-plate-fit.md](02-exact-plate-fit.md): from the detector box to the plate's 4 corners, and the warp.
- [03-multi-colour-plates.md](03-multi-colour-plates.md) and [04-auto-brightness.md](04-auto-brightness.md): plate colours and overlay brightness.
- [number_plate_detection/docs/findings.md](../../number_plate_detection/docs/findings.md): the original research notes, options compared, results and speed.
- [number_plate_detection/README.md](../../number_plate_detection/README.md): setup, settings, command-line options and known limits.
- [number_plate_detection/config.toml](../../number_plate_detection/config.toml): the detector settings with comments.
