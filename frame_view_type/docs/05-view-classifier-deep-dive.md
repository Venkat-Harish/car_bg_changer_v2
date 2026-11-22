# 5. Deep dive: `veerupandey/view_classifier`

Repo: <https://github.com/veerupandey/view_classifier>. Reviewed 2026-11-04 by reading the code, docs and result files on GitHub. Nothing was installed or run.

## At a glance

| | |
|---|---|
| **What it is** | Python code **plus a ready-trained weights file**. No training needed to use it. |
| **Weights** | `models/joint_v2_best.pt`, 198 MB, stored with Git LFS. The download link was checked and works (2026-11-04). |
| **Model** | ConvNeXt-Small (ImageNet-pretrained, ~50M parameters) with two small output heads |
| **Input** | One image (JPG/PNG/WEBP/BMP) |
| **Output** | One of 8 views, an angle in degrees (0–360), a confidence, and the probability of each of the 8 views |
| **Conventions** | 0° = front, clockwise, 90° = the car's **right**. These are the same 8 views and the same car's-own-left/right meaning as our [problem definition](01-problem-definition.md). |
| **Hardware** | CPU, NVIDIA GPU (CUDA) or Apple GPU (MPS). It picks automatically. |
| **Stack** | Python ≥ 3.12, PyTorch 2.9.1, torchvision 0.24.1, Pillow. Gradio for the optional demo app. |
| **License** | MIT for the code. The weights come from datasets with their own terms (see Risks). |
| **Maturity** | Personal research project. Created 2026-09-01, last change 2026-09-12. 1 star, 0 forks. It calls itself "not a product". |

## What the repo contains

```
view_classifier/          The Python package
  model.py                  Network: ConvNeXt-Small + pose head + angle head
  labels.py                 The 8 view names and angle <-> view conversion
  transforms_cfv.py         Image preprocessing (crop, square pad, resize)
  predict.py                predict_image(): one image in, result dict out
  infer_folder.py           CLI: run on a whole folder, write JSON
  app.py                    Gradio web demo (upload a photo, see the result)
  demo_selftest.py          Checks that the weights and demo work before a demo
  walkaround.py             Draws an 8-view grid for one car (QA figure)
  train.py, auto_resume.py  Full training code (Apple GPU / NVIDIA GPU)
  data.py, prepare_*.py,    Dataset download, split and export helpers
  export_*.py, apply_zenodo_cfv_boxes.py, memory.py, eval_per_car.py
models/joint_v2_best.pt   THE TRAINED MODEL (198 MB, Git LFS)
docs/                     Results, angle notes, dataset and training recipes,
  figures/, results/        confusion matrices, walk-around grids, raw metric JSONs
  demo_images/              12 Wikimedia car-damage photos for the demo
notebooks/                One demo notebook
tests/                    3 small unit-test files (labels, data, loss)
```

**Not included:** the training datasets (gitignored, about 1.7 GB if downloaded) and any video handling.

## How it works

```
photo
  │  1. (optional) crop to the car's bounding box
  │  2. pad to a square by mirroring the edges ("reflect pad")
  │  3. resize to 224×224, normalise colours (ImageNet stats)
  ▼
ConvNeXt-Small backbone  →  768-number image feature
  ├─► pose head  (8 outputs)  → softmax → view + confidence
  └─► angle head (2 outputs)  → (sin θ, cos θ) → atan2 → angle in degrees
```

- The two heads are trained together. The training loss is the view classification error plus 2 × the angle error. The angle head makes the errors "smooth": a wrong answer is usually the neighbouring view, not a random one.
- **No horizontal flip** in training, on purpose, because flipping swaps left and right.
- **Training data:**
  - CFV (walk-around phone videos of 66 cars in a parking garage, the [dataset closest to our footage](03-datasets.md)).
  - UnsupCar (Freiburg "Static Cars", 47 cars, angles re-aligned to the same convention).
  - Car crops come from bounding boxes published on Zenodo.

### Example output (from `infer_folder`)

```json
{
  "path": "frame_0042.jpg",
  "pred": "front_left",
  "pred_ang": 312.4,
  "conf": 0.93,
  "probs": { "front": 0.03, "front_right": 0.0, "...": "...", "front_left": 0.93 }
}
```

Its names map one-to-one onto ours:

| Repo | Ours |
|---|---|
| `front` | front |
| `front_right` | front-right |
| `right` | right |
| `rear_right` | back-right |
| `rear` | back |
| `rear_left` | back-left |
| `left` | left |
| `front_left` | front-left |

## How good is it

The numbers below are from the repo's own result files (`docs/results/joint_v2_metrics.json`).

| Test set | Frames (cars) | 8-view accuracy | Mean angle error | Correct **or** next-door view |
|---|---|---|---|---|
| CFV test (same kind of footage it trained on) | 2,527 (7 cars) | **94.1%** | 3.8° | 99.9% |
| CFV val | 4,693 (13 cars) | 93.6% | 3.8° | not reported |
| UnsupCar test (different photos, different place) | 619 | **82.9%** | 17.0° | 89.7% |
| 12 real damage photos from Wikimedia | 12 | no labels | no labels | 8 of 12 had ≥ 60% confidence |

What the confusion matrices show:

- **On CFV, almost every mistake is a neighbouring view** (e.g. front ↔ front-right). There are zero left ↔ right mix-ups.
- **On UnsupCar, some mistakes are big.** 26 of 95 front-left frames were called back-right (the opposite corner, ~180° off), and 14 of 89 left frames were also called back-right. That is where the 17° mean error comes from.
- **The test set is small:** CFV test is only 7 cars. Per car, the worst validation cars still scored 90–92%.

**What this means for us:** our video will be "new footage", so expect something **between the UnsupCar and CFV numbers**, roughly 83–94%. If our videos look like CFV (phone walk-around, car fills the frame), expect the higher end. The occasional ~180° flip is exactly the kind of one-off error that smoothing over neighbouring video frames removes.

## Scope: what it does and doesn't do

| Does | Doesn't |
|---|---|
| Classify one image into 8 views + angle + confidence | Handle video. We extract frames ourselves. |
| Run on CPU or GPU | Find the car in the frame. There is no detector, so it uses the whole image. |
| Batch a folder to JSON | Filter non-car frames (interiors, close-ups, VIN plates, documents). The README lists this as "not in scope". |
| Web demo (Gradio) and notebook | Smooth results over time |
| Full training code to retrain or fine-tune | Ship as a pip package or an ONNX/mobile model |

**Cropping matters.** The model was trained on tight crops around the car (box + 10%). For raw frames, `predict_image` and `infer_folder` skip the crop and just pad the whole frame to a square. That is fine when the car fills most of the frame. If the car is small in the frame, a car detector in front (any off-the-shelf YOLO-style detector) would bring the input closer to what it was trained on.

## How well can it be integrated?

**Easy.** The core is about 150 lines across three files plus the weights file. It depends only on PyTorch, torchvision and Pillow.

| Way to use it | Effort | Good for |
|---|---|---|
| **1. CLI on a folder of frames:** `python -m view_classifier.infer_folder --ckpt models/joint_v2_best.pt --images <frames> --out preds.json` | Minutes once installed | Our first test. Zero code. |
| **2. Gradio app:** `python -m view_classifier.app`, then open http://127.0.0.1:7860 | Minutes | Showing people, spot-checking frames by hand |
| **3. Python import:** `from view_classifier.predict import predict_image` → dict | Small | Calling it from our own Python script or service |
| **4. Copy the core:** `model.py`, `labels.py`, `transforms_cfv.py` + the `.pt` file into our own code | Small | A clean product integration without the training extras |
| **5. Export to ONNX** (not provided, but ConvNeXt exports cleanly) | Moderate | Non-Python apps (C#/.NET, JavaScript, mobile) or faster CPU inference |
| **6. Fine-tune on our own labelled frames** with the included `train.py` | Moderate, needs a GPU | If accuracy on our footage isn't good enough |

### Fit for our video use case

```
video ──ffmpeg──► frames (e.g. 2–5 per second)
   ──(optional) car detector crop──►
   ──view_classifier──► view + angle + confidence per frame
   ──smooth angle over neighbouring frames──►
   ──drop low-confidence frames──► final label per frame
```

Only the middle step is this repo. Frame extraction, the optional crop and the smoothing are small additions on our side. Because the model outputs an angle, smoothing is easy: take a circular average of the angle over the nearby frames, then convert to a view.

### Windows notes (our machine)

- The README commands are written for Mac/Linux (`source .venv/bin/activate`, `PYTHONPATH=.`). On Windows PowerShell, use `.venv\Scripts\Activate.ps1` and `$env:PYTHONPATH="."` instead.
- The repo asks for Python 3.12+, but **Python 3.11 works for inference** (verified 2026-11-04 with torch 2.9.1+cpu, see the root `README.md`). On Windows, the PyPI PyTorch wheel is the CPU build.
- Git LFS must be installed (`git lfs install`), or the `.pt` file will be a 134-byte placeholder instead of the 198 MB model.
- The Mac-specific memory helper is only used in training and switches itself off on other systems.
- **Speed (measured on an i5-1240P, no GPU):** about 0.8 s per 4K frame (pad + model). Reading a 38 s 4K video end to end adds about 19 s. A whole run at one frame per 2 s took 47 s. See the root `README.md`.

## Risks and caveats

- **One-person research project:** no releases, no issue history, not maintained as a product. Pin the exact commit if we depend on it.
- **Weights license:** the code is MIT, but the weights were trained on CFV (Apache-2.0) **and** UnsupCar / Freiburg Static Cars plus Zenodo labels, whose terms are not spelled out in the repo. Check these before any commercial use.
- **Loading the weights runs pickle code** (`torch.load(..., weights_only=False)`). Only load the `.pt` file from this repo, never one from an unknown source.
- **Small test set:** 7 test cars. The headline 94% has a wide margin of uncertainty.
- **Front/back flips on unfamiliar footage** (seen on UnsupCar). Smoothing over video frames should catch most of these.
- **No reject class:** close-ups and interiors still get one of the 8 views. A low confidence score is the only warning.

## Verdict

This is the strongest ready-made candidate. It uses exactly our 8 views with our left/right convention, ships trained weights that download today, was trained on walk-around phone video like ours, is small enough for a CPU, and has a one-command folder mode for testing. The work on our side is around it, not in it: extract frames, optionally crop the car, smooth over time, and filter non-car frames. **Next step:** run the folder CLI on 50–100 hand-labelled frames from our video, as in the [evaluation plan](04-recommendation-and-next-steps.md).

## Sources

- [veerupandey/view_classifier: README, code, docs (GitHub)](https://github.com/veerupandey/view_classifier)
- [docs/RESULTS.md](https://github.com/veerupandey/view_classifier/blob/main/docs/RESULTS.md) and [docs/results/joint_v2_metrics.json](https://github.com/veerupandey/view_classifier/blob/main/docs/results/joint_v2_metrics.json), accuracy and confusion matrices
- [docs/ANGLE.md](https://github.com/veerupandey/view_classifier/blob/main/docs/ANGLE.md), angle convention
- [docs/DATASETS.md](https://github.com/veerupandey/view_classifier/blob/main/docs/DATASETS.md), training data (CFV, UnsupCar, Zenodo boxes)
- [view_classifier/model.py](https://github.com/veerupandey/view_classifier/blob/main/view_classifier/model.py), [predict.py](https://github.com/veerupandey/view_classifier/blob/main/view_classifier/predict.py), [infer_folder.py](https://github.com/veerupandey/view_classifier/blob/main/view_classifier/infer_folder.py), [transforms_cfv.py](https://github.com/veerupandey/view_classifier/blob/main/view_classifier/transforms_cfv.py)
- [models/README.md](https://github.com/veerupandey/view_classifier/blob/main/models/README.md), weights file and Git LFS instructions
- [docs/figures/walkaround_v2_id003.png](https://github.com/veerupandey/view_classifier/blob/main/docs/figures/walkaround_v2_id003.png), example 8-view grid on a CFV car
- [UnsupCar harmonized angles (Zenodo)](https://doi.org/10.5281/zenodo.20173357)
- [CFV Dataset (Hugging Face)](https://huggingface.co/datasets/fort-cyber/CFV-Dataset)
