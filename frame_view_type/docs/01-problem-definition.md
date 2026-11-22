# 1. Problem definition

## The 8 classes as angles

The simplest way to define the classes is by **azimuth**: the angle of the camera around the car, seen from above. Here 0° means the camera is straight in front of the car, and the angle increases clockwise (towards the car's right side). This matches the convention used by [`veerupandey/view_classifier`](https://github.com/veerupandey/view_classifier).

Each class is a 45° slice centred on its main direction:

| Class | Centre | Range |
|---|---|---|
| front | 0° | 337.5° – 22.5° |
| front-right | 45° | 22.5° – 67.5° |
| right | 90° | 67.5° – 112.5° |
| back-right | 135° | 112.5° – 157.5° |
| back | 180° | 157.5° – 202.5° |
| back-left | 225° | 202.5° – 247.5° |
| left | 270° | 247.5° – 292.5° |
| front-left | 315° | 292.5° – 337.5° |

Thinking in angles helps for two reasons:
- Models that output an angle (Orient Anything, the azimuth head of `view_classifier`, the CFV dataset) map to our classes with one lookup.
- Models with a different number of classes (e.g. 16 Carvana poses) can be collapsed to 8 the same way.

## Convention to settle: whose "left"?

"Left" should mean **the car's own left**, i.e. the driver side on a left-hand-drive car. It should not mean the left side of the picture. A photo showing the car's left side has the car's nose pointing to the left of the image.

Every model and dataset below may use a different convention. Some may use "viewer's left", some start at 0° = front and go clockwise, others counter-clockwise. **Each one has to be checked on a few known frames before trusting its labels.** This is the most likely source of silly errors.

## Why this is harder than it looks

- **Left and right are mirror images.** A car's left side and right side look almost the same. The cues are small: steering-wheel side, fuel cap, how the asymmetric parts sit. Generic models are bad at this split (see [Options compared](02-options-compared.md), CLIP section).
- **Diagonal views sit on boundaries.** A frame at 22° is "front" and one at 23° is "front-right". Expect most mistakes at the borders between neighbouring classes. A model that also outputs an angle tells us how close to a border a frame is.
- **Frames that are not a car view.** Close-ups, wheels, interiors, licence plates and motion-blurred frames have no correct side. An "other / undetermined" class, or a filter in front of the classifier, handles these.
- **Camera height.** A walk-around at eye level is what most training data looks like. Very high or very low shots, or a partly cropped car, will hurt accuracy.

## What the video gives us for free

Our frames come from one continuous video, not random photos. That helps in two ways:

1. **The angle changes smoothly.** In a walk-around, the true class only ever moves to a neighbouring class (front → front-right → right …). A single frame labelled "back" between two "front-right" frames is almost surely wrong. A **majority vote or median over a small window of neighbouring frames** fixes these isolated mistakes cheaply.
2. **The order is known.** If the walk starts at the front and goes one way, the sequence of classes is fully predictable. This gives a strong sanity check, and could even be used to label frames with very little model help.

The [CFV dataset](03-datasets.md) was built by exactly this kind of walk-around, so models trained on it should match our footage well.

## Sources

- [`veerupandey/view_classifier`, angle convention (GitHub)](https://github.com/veerupandey/view_classifier)
- [CFV Dataset card, walk-around collection method (Hugging Face)](https://huggingface.co/datasets/fort-cyber/CFV-Dataset)
