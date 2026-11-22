# 2. Options compared

Five kinds of options turned up, from "download and run" to "train our own".

## Summary table

| Option | Out-of-box fit to our 8 classes | Reported accuracy | License | Hardware | Cost |
|---|---|---|---|---|---|
| **A1** `veerupandey/view_classifier` | Exact 8 + angle | ~94% (CFV test), 82.9% (UnsupCar test) | MIT code. Data has its own terms | CPU likely OK, GPU faster | Free |
| **A2** `mitbersh/car-view-classification` | Exact 8 + "other" | Not published | Not stated | CPU (ResNet-18, small) | Free |
| **A3** `tiesenx14/vit-base-patch16-224-car-angle` | Exact 8 + "undetermined" | Not published | Not stated | CPU OK (ViT-base, 86M) | Free |
| **A4** `Nekshay/SWIN_Angle_Detection_Car` | 16 poses, collapse to 8 | Not published | Not stated | CPU OK, GPU faster (Swin-base) | Free |
| **A5** `ArrayDice/car_orientation_classification` | Labels not listed | 69% (own eval set) | Apache-2.0 | CPU OK (ViT-base) | Free |
| **B** Orient Anything V1 / V2 | Angle, bin into 8 | Zero-shot, general objects (no car-only number) | CC-BY-4.0 | Small model on CPU. Large model wants a GPU | Free |
| **C** CLIP / SigLIP prompts | Any classes via text | Near chance on left vs right in research | Open (varies) | CPU OK | Free |
| **D** Prompted vision-language model | Any classes via text | No published number for this task | Open (Qwen-VL) or paid API | GPU for open models. None for APIs | Free locally, or a small fee per image |
| **E** Fine-tune our own | Whatever we choose | Depends on data | Ours | GPU for training, CPU for inference | Free, plus our time |

---

## A. Ready-made fine-tuned models

These were trained specifically to tell car views apart. They are the fastest path to a result.

### A1. `veerupandey/view_classifier` (strongest candidate)

- **What it outputs:** an 8-way pose bin (front, front_right, right, rear_right, rear, rear_left, left, front_left) **and** a continuous azimuth angle (predicted as sin/cos so it wraps correctly around 360°).
- **Convention:** 0° = vehicle front, clockwise, 90° = the vehicle's right. This matches our [problem definition](01-problem-definition.md).
- **Training:** trained on the CFV dataset (walk-around video frames) plus UnsupCar. Horizontal flip augmentation is deliberately turned off, because flipping would swap left and right.
- **Reported results:**
  - CFV test: 94.1% 8-way accuracy, 3.77° mean angle error.
  - UnsupCar test: 82.9% accuracy, 17.0° mean angle error.
- **Ease of use:** has a Gradio web demo, batch inference over a folder (JSON output), and a demo notebook. Weights are stored in the repo via Git LFS.
- **Caveats:**
  - It is a personal research project.
  - It is not meant for close-ups, VIN plates, interiors or documents. A filter should remove those frames first.
  - The code is MIT, but the training datasets carry their own terms.

### A2. `mitbersh/car-view-classification`

- ResNet-18 (small and fast) with two heads: left / centre / right and front / centre / back. The two answers are combined into one view.
- Labels: back, back-left, back-right, front, front-left, front-right, left, right, **other**. This is our exact set plus a reject class.
- Part of the AutoInspect car-damage project. Its training set is on Hugging Face (gated) and partly built from Carvana images.
- No accuracy, license or inference code on the model card. The GitHub repo has the code.

### A3. `tiesenx14/vit-base-patch16-224-car-angle`

- ViT-base. The model card is empty, but its config lists Vietnamese labels that match our classes exactly:

| Label | Meaning |
|---|---|
| `chinh_dien_dau_xe` | front (straight-on) |
| `ben_phai_dau_xe` | front-right |
| `ben_phai_xe` | right |
| `ben_phai_duoi_xe` | back-right |
| `chinh_dien_duoi_xe` | back (straight-on) |
| `ben_trai_duoi_xe` | back-left |
| `ben_trai_xe` | left |
| `ben_trai_dau_xe` | front-left |
| `khong_xac_dinh` | undetermined |

- Runs with a one-line Hugging Face `pipeline("image-classification", ...)`.
- Unknown training data, no metrics, no license, 7 downloads last month. **To verify:** whether "phai/trai" (right/left) means the car's side or the viewer's side.

### A4. `Nekshay/SWIN_Angle_Detection_Car`

- Swin-base with 16 classes named `Angle1` … `Angle16`. This is almost certainly the 16 Carvana studio poses, 22.5° apart.
- To get 8 classes, merge pairs of neighbouring poses. **To verify:** which index is the front, and which way the numbering goes.
- No model card, metrics or license. The most-downloaded of these HF models (~240 a month).

### A5. `ArrayDice/car_orientation_classification`

- ViT-base fine-tune, Apache-2.0. It reports only 69% accuracy and doesn't list its labels.
- Listed for completeness only. Not recommended.

### Also seen, but not a fit

- [SyedaAimanAli/CNN-Classifier](https://github.com/SyedaAimanAli/CNN-Classifier): front / back / side only. No left vs right, no diagonals.
- [VoNet](https://dl.acm.org/doi/pdf/10.1145/3018009.3018045) (paper): 5 classes (front, rear, side, front-side, rear-side). It merges left and right.

---

## B. Zero-shot orientation model: Orient Anything

- Estimates the 3D orientation of **any** object from one image. It returns azimuth, polar angle, in-plane rotation and a confidence score. It was trained on about 2M rendered images of 3D models.
- **How we'd use it:** take the azimuth and bin it into the 8 classes from the [problem definition](01-problem-definition.md). **To verify:** which way its azimuth increases.
- **Sizes:** Small (23M parameters), Base (88M) and Large (305M). Small should run on CPU. Large is better on a GPU.
- **V2** (NeurIPS 2025 Spotlight) is much better than V1. For example, median error on Pascal3D+ fell from 22.9° to 15.0°. V2 also models symmetric objects.
- Works best on a single object per image. Crop the car first if the frame has other things in it.
- Code and a Gradio demo are on GitHub, and there is a Hugging Face Space to try it in the browser. License: CC-BY-4.0.
- **Why it's interesting:** no car-specific training is needed, so it may cope better with unusual cars or camera heights than the fine-tuned models. **Why it's second:** it's a general model with no car-only accuracy number, and its errors (15–20° median) are a big share of a 45° slice.

---

## C. CLIP / SigLIP text prompts (not recommended alone)

- The idea: compare each frame to text like "a photo of the front-left of a car" and pick the best match. Free, and no training.
- **The problem:** research shows contrastive vision-language models are poor at left vs right.
  - "What's left can't be right" (2023) finds CLIP at around chance on left/right relations.
  - The Visual Spatial Reasoning benchmark (2022) finds left/right especially hard when it depends on an object's own orientation, which is exactly our case.
- Likely fine for front vs back vs side, but unreliable for the left/right and diagonal splits.
- **Still useful as a backbone:** frozen CLIP, SigLIP or DINOv2 features with a small trained classifier on top is a cheap version of option E.

---

## D. Prompted vision-language models

- Ask a general vision chat model "Which side of this car is shown? Answer one of: front, front-right, …".
- **Open-source:** the Qwen-VL family, run locally (needs a GPU for decent speed).
- **Hosted:** Gemini, GPT or Claude APIs. No setup, and a small fee per image (check current pricing before running many frames).
- **No published accuracy for this exact task.** The nearest benchmark (choosing the right camera view in multi-camera driving scenes) is a different task, so it doesn't tell us much. It has to be measured on our frames.
- These models may also struggle with left vs right, for the same reasons as CLIP. Giving an example image per class in the prompt (few-shot) may help.

---

## E. Fine-tune our own small classifier (fallback)

Only needed if A and B fall short on our footage.

- **Model:** a small CNN (ResNet / EfficientNet / MobileNet), or a linear head on frozen DINOv2 or CLIP features. Training takes minutes to hours on one GPU, and inference runs on CPU.
- **Data:** CFV and/or Carvana (see [Datasets](03-datasets.md)), plus some hand-labelled frames of our own.
- **Augmentation trick:** a horizontal flip turns a left view into a right view. So flipping is allowed **only if the label is also swapped**: left↔right, front-left↔front-right, back-left↔back-right. Front and back stay the same. This doubles the data.
- **Predict an angle, not just a class:** like `view_classifier`, predicting sin/cos of the azimuth alongside the class gives smoother errors and makes temporal smoothing easier.

## Sources

- [veerupandey/view_classifier (GitHub)](https://github.com/veerupandey/view_classifier)
- [mitbersh/car-view-classification, model (Hugging Face)](https://huggingface.co/mitbersh/car-view-classification)
- [DedovInside/AutoInspect, project behind the mitbersh model (GitHub)](https://github.com/DedovInside/AutoInspect)
- [tiesenx14/vit-base-patch16-224-car-angle (Hugging Face)](https://huggingface.co/tiesenx14/vit-base-patch16-224-car-angle) and its [config.json with labels](https://huggingface.co/tiesenx14/vit-base-patch16-224-car-angle/raw/main/config.json)
- [Nekshay/SWIN_Angle_Detection_Car (Hugging Face)](https://huggingface.co/Nekshay/SWIN_Angle_Detection_Car) and its [config.json with labels](https://huggingface.co/Nekshay/SWIN_Angle_Detection_Car/raw/main/config.json)
- [ArrayDice/car_orientation_classification (Hugging Face)](https://huggingface.co/ArrayDice/car_orientation_classification)
- [Orient Anything, code and V2 release notes (GitHub)](https://github.com/SpatialVision/Orient-Anything)
- [Orient Anything V2, NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116168) and [arXiv 2601.05573](https://arxiv.org/abs/2601.05573)
- [What's left can't be right: positional incompetence of contrastive VLMs (arXiv 2311.11477)](https://arxiv.org/abs/2311.11477v1)
- [Visual Spatial Reasoning (arXiv 2205.00363)](https://arxiv.org/pdf/2205.00363)
- [Multi-view MLLM benchmark for autonomous driving, nearest VLM evidence (arXiv 2606.09644)](https://arxiv.org/pdf/2606.09644)
- [SyedaAimanAli/CNN-Classifier (GitHub)](https://github.com/SyedaAimanAli/CNN-Classifier)
- [VoNet: Vehicle Orientation Classification using CNN (ACM)](https://dl.acm.org/doi/pdf/10.1145/3018009.3018045)
