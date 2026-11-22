# 3. Datasets

Free labelled data, in case we need to test models properly or fine-tune our own (option E in [Options compared](02-options-compared.md)).

## Summary

| Dataset | Labels | Size | License | Fit for us |
|---|---|---|---|---|
| **CFV (Car Full View)** | Angle 0–360° + bounding box | 23.8k frames, ~66 cars | Apache-2.0 | **Best.** Walk-around video, like ours |
| **Carvana** | 16 fixed poses (22.5° apart) from filenames | Thousands of cars × 16 | Kaggle competition terms | Good for training. License limits use |
| `mitbersh/car-view-classification` | Our 8 classes + "other" | 1k–10k images | Not stated (gated) | Ready-made 8 classes. Partly Carvana |
| CompCars | Coarse viewpoints (no left/right split) | 136k web images | Non-commercial research only | Poor fit |
| Stanford Cars | Make/model only, no views | 16k images | Research | No fit |

## CFV: Car Full View (best match)

- **How it was made:** the authors recorded each car on video while walking around it, starting at the front and going clockwise back to the start. A few frames were labelled by hand, and the angles of the frames in between were filled in automatically. **This is the same kind of footage as ours.**
- **Labels:** an integer angle 0–360° per frame, a bounding box around the car, and a car ID. Licence plates are blurred.
- **Size:** 23,826 frames at 512×512 (18.8k train, 5.05k test), about 66 cars, 1.66 GB.
- **License:** Apache-2.0 (commercial use OK).
- **Extras:** the official repo (`fort-cyber/car-orientation`) has training code and links to weights for its best model on Google Drive. This is one more ready-made model to try.
- **To verify:** the dataset card calls the angle "pitch", but from the collection method it is clearly the angle around the car (azimuth). Check which angle is the front, and the direction, on a few frames.
- Paper: Catruna et al., "Car Full View Dataset: Fine-Grained Predictions of Car Orientation from Images", *Electronics* 12(24), 2023.

## Carvana (Kaggle 2017)

- Built for a car-segmentation competition. Each car was photographed on a turntable in a studio at **16 fixed angles, evenly spread over 360°** (22.5° apart). The angle index is in the filename: `<car_id>_01.jpg` … `<car_id>_16.jpg`. So every image has a free angle label.
- Thousands of cars. The labelled training split alone is 318 cars × 16 images, and the larger test split has far more, also with the angle in the filename.
- **Mapping to 8 classes:** every second pose sits on a class centre (0°, 45°, 90° …). The poses in between sit exactly on class borders, so drop them or treat them as ambiguous.
- **To verify:** which index is the front, and which way the numbering turns. One project lists index 1 = 0°, 5 = 90°, 9 = 180°, 13 = 270°, but doesn't say which one is the front.
- **Caveats:**
  - Studio photos on a plain background, unlike outdoor video. Good for training, less good for testing.
  - Kaggle competition data. The terms likely restrict use outside the competition and commercial use. Check before using it.

## `mitbersh/car-view-classification` dataset

- Already split into our 8 classes plus "other", in image-folder format (147 MB).
- Partly built from Carvana. The card notes that redistribution may be restricted because of this.
- Gated: you have to request access on Hugging Face.

## Not a good fit

- **CompCars:** large and has viewpoint labels, but the views are coarse, with left and right merged (front, rear, side, front-side, rear-side, as used by VoNet; to verify against the CompCars paper). It is also non-commercial research only.
- **Stanford Cars:** labelled only by make, model and year. No viewpoint labels.

## Sources

- [CFV Dataset (Hugging Face)](https://huggingface.co/datasets/fort-cyber/CFV-Dataset)
- [fort-cyber/car-orientation, official CFV code and weights link (GitHub)](https://github.com/fort-cyber/car-orientation)
- [CFV paper, *Electronics* 12(24) 4947 (MDPI)](https://www.mdpi.com/2079-9292/12/24/4947)
- [Carvana Image Masking Challenge (Kaggle)](https://www.kaggle.com/c/carvana-image-masking-challenge)
- [Carvana dataset overview (Dataset Ninja)](https://datasetninja.com/carvana-image-masking)
- [lambertwx/carvana-pose, 16 poses over 360° of yaw (GitHub)](https://github.com/lambertwx/carvana-pose)
- [Carvana 1st-place winners' interview, studio capture setup (Kaggle blog)](https://medium.com/kaggle-blog/carvana-image-masking-challenge-1st-place-winners-interview-78fcc5c887a8)
- [mitbersh/car-view-classification dataset (Hugging Face)](https://huggingface.co/datasets/mitbersh/car-view-classification)
- [CompCars dataset page, license terms (CUHK MMLAB)](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)
- [VoNet, 5-view scheme (ACM)](https://dl.acm.org/doi/pdf/10.1145/3018009.3018045)
