# Glossary

Terms used in [pipeline-steps.md](pipeline-steps.md), in plain language.

| Term | Meaning |
|---|---|
| **alpha / mask** | A grey picture the size of the photo: white where the car is, black where it isn't, grey at soft edges. The cut-out *is* this mask. |
| **BiRefNet** | An open model that separates the main object from the background. It makes the first cut-out (step 3). |
| **SAM 3** | Meta's "Segment Anything" model, version 3. You give it a word such as "car" or "windshield" and it outlines every matching object. Used to keep only this car (step 4) and to find the glass (step 5). |
| **MoGe-3** | A Microsoft model that turns one photo into a 3D point cloud. A floor fitted to that cloud gives the camera's height and tilt (step 6). |
| **VGGT** | A model that solves several photos of the same scene together: where each shot was taken from, and the lens. It gives one shared focal length and the angle between shots (step 6). |
| **fal.ai** | A service that runs these open models on its own servers, paid per call. Nothing is generated; the models only measure and outline. |
| **focal length (px)** | How "zoomed" the lens is, measured in pixels. A wrong focal makes the car's 3D shape look stretched or flattened. |
| **pitch / roll** | How far the camera looks down (pitch) and how far it is tilted sideways (roll). |
| **yaw** | Which way the car faces the camera: 0° is front-on, 90° its right side, 180° the back. |
| **vanishing point / horizon** | Parallel lines in a photo, such as kerbs and window rows, meet at a point. Two such points give the horizon and so the camera's tilt. |
| **IoU** | Intersection over union: how well two outlines overlap. 1.0 is a perfect match. Below 0.90 the fit is flagged `weak-fit`, below 0.75 `fit-failed`. |
| **proxy / 3D car** | A simple block model of the car, built from its real length, width, height and wheelbase in `cars.json`. |
| **levelling** | The one rigid rotation of the car's pixels that makes it stand upright when the phone was held crooked. |
| **tyre contact lock** | Pinning the 3D car to where the real tyres touch the ground in the photo, so the car neither floats nor sinks. |
| **room / re-render** | Each studio background was measured once as a 3D room. Re-rendering draws that room again from the photo's own camera. |
| **invented** | Room areas the original background picture never showed, which had to be filled from nearby texture. Too much of it sets the `invented` flag. |
| **flag** | A short note in `results.csv` (e.g. `weak-fit`, `camera-lines-fallback`) that marks a photo for a human look. Photos are placed anyway: flag, don't drop. |
| **hosted / local** | Hosted = run on fal.ai. Local = run on the laptop's CPU. Every hosted step falls back to the local path if fal fails. |
| **e2e** | End to end: the whole chain from video to studio shots, as `e2e_steps.py` runs it. |
