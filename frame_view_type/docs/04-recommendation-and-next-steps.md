# 4. Recommendation and next steps

## What to try first, once we have frames

| Order | Try | Why | Effort |
|---|---|---|---|
| 1 | [`veerupandey/view_classifier`](https://github.com/veerupandey/view_classifier) | Exactly our 8 classes plus an angle. ~94% reported. Trained on walk-around video. Has a folder-batch mode. | Clone, `git lfs pull`, run on a folder |
| 2 | [Orient Anything V2](https://github.com/SpatialVision/Orient-Anything) | Zero-shot, no car-specific training. A second opinion that may handle odd cars or angles better. | Try the Hugging Face Space first, then run locally |
| 3 | [`mitbersh`](https://huggingface.co/mitbersh/car-view-classification) and [`tiesenx14`](https://huggingface.co/tiesenx14/vit-base-patch16-224-car-angle) HF models | Labels already match our 8 classes. Cheap extra baselines. | A few lines with `transformers` |
| 4 | CFV's own best model ([`fort-cyber/car-orientation`](https://github.com/fort-cyber/car-orientation)) | Trained on the dataset closest to our footage. Apache-2.0. | Download weights from the repo's link |
| 5 | A prompted vision-language model | "No setup" baseline to compare against | Paste frames into a chat, or a short API script |

Skip CLIP/SigLIP prompts as a main approach (poor at left vs right). Keep option E (fine-tune our own) as the fallback if nothing above is good enough on our footage.

## How to judge them on our footage

A small, honest test is enough to pick a winner:

1. **Pick frames.** Take about 50–100 frames spread evenly through the video, so every one of the 8 classes has several examples.
2. **Label them by hand** using the [convention](01-problem-definition.md): the car's own left and right, with 45° slices. Mark unusable frames (blur, close-ups) as "other".
3. **Check conventions.** Before scoring, run each model on 3–4 obvious frames (straight front, straight left …) to confirm its left/right and angle direction match ours. Fix the mapping if not.
4. **Score each model:**
   - 8-way accuracy.
   - A **confusion matrix** (which classes get mixed up). Watch for left↔right swaps (a convention or mirror problem) and diagonal↔side mixes (border cases, less serious).
   - "Off by one neighbour" rate. Calling a front-right frame "front" is a much smaller error than calling it "back-left".
5. **Add temporal smoothing** (a majority vote over e.g. 5–9 neighbouring frames) and score again. On walk-around video this should remove most one-off errors.

## Open questions and risks

- **Licenses:** the `mitbersh`, `tiesenx14` and `Nekshay` models state no license. Fine for exploring. For any product use, prefer the MIT `view_classifier` code, CFV (Apache-2.0) or Orient Anything (CC-BY-4.0), and check the terms of their training data.
- **Angle conventions** are unverified for Orient Anything, CFV, Carvana and the HF models (see the "to verify" notes in the other docs).
- **Non-car frames:** close-ups, interiors and blurred frames need an "other" class or a filter, or they get forced into a wrong side.
- **Our footage vs training data:** camera height, distance, lighting and car type (e.g. pickups, vans) may differ from the training sets. `view_classifier` notes that hard cases like pickups improved a lot in its v2. Our own test above is the only real answer.

## Sources

- [veerupandey/view_classifier (GitHub)](https://github.com/veerupandey/view_classifier)
- [Orient Anything (GitHub)](https://github.com/SpatialVision/Orient-Anything)
- [mitbersh/car-view-classification (Hugging Face)](https://huggingface.co/mitbersh/car-view-classification)
- [tiesenx14/vit-base-patch16-224-car-angle (Hugging Face)](https://huggingface.co/tiesenx14/vit-base-patch16-224-car-angle)
- [fort-cyber/car-orientation (GitHub)](https://github.com/fort-cyber/car-orientation)
