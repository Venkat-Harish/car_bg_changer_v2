# Car viewpoint classification: exploration notes

**Question:** Given frames from a video of a car, can a cheap or open-source image classifier tell which side of the car each frame shows?

The 8 target classes:

| | | |
|---|---|---|
| front-left | **front** | front-right |
| **left** | (car) | **right** |
| back-left | **back** | back-right |

Status: research only (2026-11-03). Nothing has been built or tested on our own frames yet.

## Short answer

Yes. Several free models already do this, and one free dataset was recorded the same way as our video (walking around the car).

| Rank | Option | Why |
|---|---|---|
| 1 | [`veerupandey/view_classifier`](https://github.com/veerupandey/view_classifier) | Exactly our 8 classes plus a continuous angle. About 94% reported accuracy. Trained on walk-around video frames. MIT code. |
| 2 | [Orient Anything V2](https://github.com/SpatialVision/Orient-Anything) | Zero-shot "which way is this object facing" model. Gives an angle we can bin into 8 classes. A good second opinion. |
| 3 | [`mitbersh/car-view-classification`](https://huggingface.co/mitbersh/car-view-classification), [`tiesenx14/vit-base-patch16-224-car-angle`](https://huggingface.co/tiesenx14/vit-base-patch16-224-car-angle) | Hugging Face models whose labels already match our 8 classes plus "other". No accuracy or license published. |
| 4 | A prompted vision-language model (Qwen-VL, Gemini, GPT, Claude) | No setup needed. Accuracy on this task is unknown and has to be measured. |
| Avoid alone | CLIP / SigLIP text prompts | Research shows these models are close to chance at left vs right. |

Every option can be free (run locally) except the hosted vision-language APIs, which charge a small fee per image.

## Documents

1. [Problem definition](01-problem-definition.md): the 8 classes as angles, the left/right convention, and why video helps.
2. [Options compared](02-options-compared.md): every model found, with a comparison table.
3. [Datasets](03-datasets.md): free labelled data for testing or fine-tuning.
4. [Recommendation and next steps](04-recommendation-and-next-steps.md): what to try first and how to evaluate it on our frames.
5. [Deep dive: `view_classifier`](05-view-classifier-deep-dive.md): what the top candidate's repo contains, how it works, how good it is, and how to integrate it.

Each document ends with its sources.
