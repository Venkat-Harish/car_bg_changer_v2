# Spyne 360° spin: what it does, why it looks natural, and what we are missing

*Findings from the Spyne demo (https://console.spyne.ai/virtualstudio/demo-v2, "360° Spin" section), plus fact-checked web research. Collected on 2026-11-05. This is information only: no code was changed.*

**How to read this**

| Section | Contents |
|---|---|
| §1 | The short answer |
| §2–§3 | What I saw first-hand |
| §4–§6 | The purpose, the gap, and what "fitting into 3D space" needs |
| §7 | Verified research, with sources |
| §8–§9 | Decisions and unknowns |
| §10 | Evidence log |

Every research claim in §7 was checked against its source by a separate reviewer; claims that could not be confirmed were dropped. *Inference* marks my own reasoning.

---

## 1. In short

**What it is.** Spyne turns **one phone walkaround video** into a **"360 Studio Turntable"**:
- **36 frames, 10° apart.** Their API also offers 72.
- The frames play as a drag-to-rotate spin: the car on a turntable in a branded studio.
- **For the dealer**, it replaces a physical turntable studio and photographer. Spyne's own material compares it with dealer turntables costing up to $150,000. It gives every listing the same premium look, and lets shoppers inspect every angle.

**Why it looks natural.** Every frame obeys **one 3D set-up**:
- One fixed camera and room.
- The car anchored on the turntable centre at real-world scale, with its perspective kept.
- The turntable disc turning with the car.

On top of that come even brightness across frames, floor shadow and reflection, tinted glass, and the number plate swapped for the dealer logo.

**How, in Spyne's own words.** Spyne's sales wiki says *"There are two ways to generate the Spin 360 output: Gaussian Splatting (medium stabilisation), NeRF (Neural Radiance Field) 3D modelling (high stabilisation)"*. Both are 3D reconstruction methods. It offers four spin types:
- **Original Walkaround:** *"created without any AI input"*. It picks, stitches and stabilises frames, in about 10 min.
- **Turntable Studio, Showroom and Outdoor:** *"360 Stabilization, transformation"* plus super-resolution (*"All colors are brightened"*), in about 30 min.

Their **3D 360-Spin API** returns 36 or 72 frames in about an hour and is *"recommended for unstable video inputs"*. What I saw fits a 3D path:
- complete views from a cropped video;
- "smeared into streaks" failure examples;
- 30–40 min processing.

Spyne does not say which method produced this particular demo.

**Why ours looks copy-pasted.** Our own `results.csv` on `background_1` shows it:
- **18 of 20 frames are forced to ≈ 648 px tall**, the `max_height` cap, with the lowest pixel at **y = 907**. The other two hit the width cap.
- The scale swings 0.38–0.49 from frame to frame.
- Spyne's car height varies **520–776 px** with the view, as a real turntable would.

**What "fitting into the background's 3D space" needs**, without editing car pixels:
1. A camera model for each background.
2. Each frame's real viewpoint, from structure-from-motion (SfM), a technique that recovers camera positions from a set of overlapping frames.
3. Frames chosen by angle, not time.
4. Scale from one virtual camera distance, not fit-to-box.
5. The car's footprint centre anchored on the true disc centre.
6. Shadow, reflection and disc rotation as layers under the car.
7. At most one global brightness gain per frame.

**The catch.** Spyne's look also **changes the car's pixels**: super-resolution, "transformation", glass tint and dehaze. In their frame 18 the input's sky reflections are gone. Our rule is that dents and scratches stay exactly as photographed. So copy Spyne's **geometry**, not its pixel edits.

Some things **cannot be fixed in 2D**: viewing height, perspective strength, and cropped-off parts. Fix those at capture, with Spyne's own rules: 3–3.7 m away, 1.2–1.5 m high, car never cropped, one full loop. Our clip has **no uncut pure-left profile**, so no rule-compliant method can create that view.

---

## 2. What the demo page is

| Part | What it shows |
|---|---|
| Product | "Dealer OS by Spyne", sample car *2024 Mercedes-Benz EQB*. Steps: **Create → Promote → Publish**. |
| Media tabs | **Images** (stills), **360° Spin** (Exterior / Interior), **Video Tour**, plus a **"View input"** toggle |
| Create options | 9 studio backgrounds (ids 49577, 54321, 57826, 32087, 18541, …); 9 video-tour styles (Trinity, Vertex, Dare in 16:9 and 4:3); number plate replaced by the dealer logo, or blurred |
| "Fix with Studio Create" | Runs the processing. Its progress text reads **"Matching light and shadows"**. |
| Before/after samples | **Dehaze**, **Shadow & reflection**, **Super Resolution**, **Windows tint** |

"360° Spin enhancements" cards on the page:

| Card | Spyne's description | What it tells us |
|---|---|---|
| Colour cast | "Evens brightness across every frame" | Exposure and colour are matched across frames, so the spin doesn't flicker |
| Spin from cropped video | "Builds the spin straight from a walkaround clip" | Works even when the car is partly out of frame. Not in Spyne's public docs; the published validator *rejects* cropped videos. |
| Interior 360 | "Cabin spin without a Ricoh Theta" | Not in the public docs either; those require a 360° camera |
| Vehicle highlights | "Hotspots for wheels, lights and trim" | Clickable points on the spin; a documented add-on |
| Stable 360, original set | "Keeps the real shoot environment" | Matches the documented **Original Walkaround** type (no AI) |
| 8 to 36 frames (soon) | "Interpolated into a smooth, stable spin" | Building 10° steps from 45° steps needs 3D or generative view synthesis, which invents pixels |

---

## 3. First-hand evidence from the 360° spin

**Method.** I read the page, its iframe, and the public API data the page loads. All 36 output frames were analysed inside the browser: pixel comparison, an average image, measurements, and full-resolution crops. Nothing was downloaded. Our local frames were used for comparison.

### 3.1 Input and output

| Evidence | Meaning |
|---|---|
| "View input" plays **`Final_zoomed_car.mp4`**: 3840×2160, 38.17 s, 30 fps, handheld phone, close range | **The same video our `frame_view_type` frames come from** |
| The spin is an iframe, `https://assets.spyne.ai/360?sku_id=…`, showing **36 frames of 1920×1080 PNG** named `car_replace_bg_<id>.png` | Finished, pre-rendered composites. The viewer only flips through them. |
| The API data the page loads (`get-sku-image-data`) includes:<br>• `FRAME_INT` 1–36<br>• `AI_ANGLE: null`<br>• `skuName: "video…"`<br>• `skuCategory: "3D_Model"`<br>• `sequenceData: {first_frame_angle: 45, rotation: "clockwise"}` | Frames are ordered by viewing angle: 36 frames = 10° steps.<br>`3D_Model` is Spyne's general category name for spins: their wiki's iframe example uses it too.<br>The angle convention is undocumented: frame 1 is actually a left-side profile. |
| Contact sheet order: left side (1) → rear (7–10) → right side (16–19) → front (25–30) → left (34–36) | A smooth, complete 360° with no gaps |

### 3.2 Geometry: one camera, one turntable, one car

| Evidence | Meaning |
|---|---|
| The **top 20% of every frame** (walls, ceiling, logo) is **identical in all 36 frames**. A sampled wall pixel had the same value 36 times. | **One fixed background plate seen from one fixed virtual camera.** The room never moves. |
| The **turntable disc's texture changes every frame**, while the floor outside the disc stays put. The disc's rectangular hatch is on the right in the front view (frame 27) and on the left in the left view (frame 36). | **The turntable rotates with the car**, like a real one |
| The average of all 36 frames centres the car's blur on the disc. In the side view the car's centre (x ≈ 950) sits on the disc centre (x ≈ 956). | **One anchor point:** the car turns about the turntable centre |
| The car's roof row and its lowest dark row change **smoothly with the view** (table below). Rear and front views reach lower and look taller than side views. | **Perspective is kept:** the near end is closer to the camera, so it looks bigger. There are no fit-to-box jumps. |
| Turntable rim fit in frame 18: centre (956, 774), 1814 px wide, height/width **0.246** | The virtual camera looks down at roughly **14°** (asin 0.246; this slightly overestimates for a close camera, see §7.2) |
| The side-view car is about 1360 px long, **≈ 75% of the disc diameter** | The car is sized to the studio's real-world scale, roughly a 4.7 m EQB on a ~6 m disc |
| Against our 4K input: side view ≈ **0.37×**, front view ≈ **0.50×** | **A different scale per view.** It undoes the walker's changing distance, so every frame looks shot from one distance. |

Selected frame measurements (1920×1080 output). "Lowest row" is the lowest dark car-or-shadow row: bumper underside plus contact shadow, **not** the tyre contact. Real tyre contacts move much less between views, by tens of pixels (§7.2).

| Frame | View | Roof y | Lowest row y | Height (px) |
|---|---|---|---|---|
| 1 | left side | 292 | 852 | 560 |
| 5 | rear-left ¾ | 252 | 984 | 732 |
| 9 | rear | 220 | 996 | 776 |
| 14 | rear-right ¾ | 264 | 948 | 684 |
| 18 | right side | 300 | 820 | 520 |
| 23 | front-right ¾ | 316 | 976 | 660 |
| 27 | front | 320 | 996 | 676 |
| 32 | front-left ¾ | 312 | 968 | 656 |
| 36 | left side | 304 | 832 | 528 |

### 3.3 Appearance: Spyne changes the car itself

| Evidence | Meaning |
|---|---|
| **Frame 18 (right side), full-resolution crop.** The paint is neutral studio-white, without the input's sky and cloud reflections. The side glass is a uniform grey tint; the trees and cars it reflected are gone. The hub's Mercedes star is a soft blob (crisp in the input), and edges look slightly "waxy". | The car is **relit, cleaned and enhanced**, not just cut out |
| **Frame 36 (left side)** is a **complete, flat profile with equal-size wheels**. In the video the pure left side is **cut off at the front** (≈ 26–31 s). At 31.2 s the whole car fits, but the view has already turned toward front-left, with the wide-angle "big front wheel" look. | That angle was **synthesized or re-rendered**, not copied from one video frame |
| Before/after **Shadow & reflection**: car moved from asphalt onto the marble turntable; contact shadow and soft floor reflection added; plate became a logo plate | Grounding is drawn on the floor |
| Before/after **Windows tint**: see-through glass that showed buildings and the cabin becomes a uniform dark tint | Glass pixels are replaced |
| Before/after **Dehaze**: sun haze and lens flare removed from the car | Car pixels are changed |
| Before/after **Super Resolution**: wheel and tyre detail sharpened | Detail is **generated**, which is risky for real scratches |

### 3.4 Spyne's capture rules (their linked guideline video)

From `https://resources.spyne.ai/360-guidline-video-v5.mp4` (2 min 23 s):

- **Distance and height:** phone **4–5 ft (≈ 1.2–1.5 m)** high, **10–12 ft (≈ 3–3.7 m)** from the car.
- **Pace:** finish the walk in **60–90 s**.
- **Framing:** *"Ensure that the car doesn't crop in any frame."*
- **Checklist:** no background movement around the car, and stay as stable as possible; then *Confirm* or *Reshoot*.
- **Output:** called the **"360 Studio Turntable"**, ready in **about 30–40 min**.
- **Things to avoid:**
  - **High or low exposure:** the output car is dark.
  - **Crowded location:** the output car is **smeared into streaks**.
  - **Unstable camera:** the output car is **smeared into streaks**.

**Our video breaks two rules:** the car is cropped at several angles, and 38 s is under the 60–90 s guidance (though inside the validator's 30–90 s). That is presumably why the demo files it under "Spin from cropped video".

### 3.5 How Spyne makes the studio spin

**Sourced (Spyne's own material, §7.1):**
- Spin 360 output is generated with **Gaussian Splatting** or **NeRF 3D modelling**.
- The studio tiers add "transformation" and super-resolution.
- The 3D endpoint is recommended for unstable videos and takes about an hour.

**Inference.** The visible evidence fits a **3D reconstruction of the car from the whole walkaround, rendered on a virtual turntable by one virtual camera**:
- the fixed plate and rotating disc;
- complete views where every input frame is cropped;
- streak-smear failures;
- the flatter profile;
- the slightly waxy detail.

It does **not** fit 2D frame-by-frame pasting. The `car_replace_bg_*` file names suggest a background-replacement step at the end. Spyne does not name the method used for this demo.

---

## 4. The main reason for the exercise

1. **The business goal.** Every car in stock gets the same branded, showroom-quality, **interactive 360° view** from a quick phone walkaround. That replaces a physical turntable studio and photographer, and the listings look consistent and premium. Shoppers can inspect every angle, which is meant to build trust. Hotspots, interior 360 and a video tour come from the same capture.
2. **Why "natural" matters so much in a spin.** Dragging plays the frames one after another. Any change between frames is seen at once as the car **jumping, shrinking, sliding or floating, or the room shaking**: the camera, scale, ground point, background or brightness. A single still can hide these problems; a spin cannot.
3. **So the exercise** is to make a set of independent handheld frames behave like **one physical car on one turntable in front of one camera**. That is what "fit the images into a 3D space" means. The car's size and position in each frame must come from the background's 3D geometry and the frame's real viewpoint, not from "make it fit the box".

   A walk around a parked car is geometrically the same motion as a car turning on a turntable (§7.2). What differs is the background and the lighting, which is fixed to the world in a walkaround.

---

## 5. Why our output looks copy-pasted (gap analysis)

| Aspect | Our `background_changer` today | Spyne 360° | Visible effect for us |
|---|---|---|---|
| Frame choice | One frame every 2 s by **time**, 8 coarse view classes | 36 frames at **even 10° angles** | Uneven rotation steps; a jerky spin |
| Scale | Each frame fitted into `max_width × max_height` | **One virtual camera distance** for all frames | Every view is forced to the same height (≈ 648 px), so front, side and rear look the same size |
| Anchor | Bounding-box centre to canvas centre; lowest pixel on one fixed `ground` row (907) | **Footprint centre on the turntable centre**; contacts follow perspective | The car slides and hops; its near end never comes toward the camera |
| Camera match | Not considered | Fixed studio camera; the car's real perspective is kept | The car looks pasted at the wrong depth or height |
| Cropped frames | Kept, flagged in `results.csv` (5 of 20: 10 s, 14 s, 26 s, 28 s, 30 s) | Complete car in every frame | A sliced car standing in the studio |
| Turntable | Static disc in `background_1` | Disc texture rotates with the car | The car "skates" over a fixed floor pattern |
| Grounding | Contact shadow rebuilt from each frame's outline (changes shape frame to frame) | Contact shadow **and** floor reflection, one floor model | Less grounded; the shadow shape flickers |
| Cut-out edges | RGB and straight alpha resized separately, blended in sRGB; `edge_shrink` trims real edge pixels | Clean edges | Possible fringe of the old background, or slightly eaten edges |
| Appearance | Car untouched (our rule): outdoor sky reflections, see-through glass | Relit, dehazed, glass tinted, even brightness | Outdoor reflections look out of place in a studio |
| How a frame is made | 2D paste of one video frame | Most likely rendered from a 3D reconstruction (Spyne's wiki names NeRF / Gaussian Splatting) | Spyne shows angles the video never fully captured |

**Measured side by side** (our numbers from `output/Final_zoomed_car/results.csv`, `background_1`; Spyne's from §3.2):

| View | Our scale | Our placed car height | Our lowest row | Spyne car height |
|---|---|---|---|---|
| Front (`0000.0s`) | 0.474 | **648 px** (height cap) | 907 | 676 px |
| Right side (`0012.0s`) | 0.431 | **608 px** (width cap) | 907 | 520 px |
| Rear (`0020.0s`) | 0.469 | **648 px** (height cap) | 908 | 776 px |
| Left side (`0030.0s`, cut off) | 0.409 | **644 px** (width cap) | 907 | 528 px |
| All 20 frames | 0.38–0.49 | 608–649 px | 907–908 | 520–776 px |

Our layout makes every view the same height on the same line. A fixed camera watching a car turn sees the ends swing toward it and grow. That flattening is the "copy-paste" feeling.

**Our backgrounds can be measured the same way.** In `backgrounds/background_1.webp` the turntable rim is centred at (957, 836), is 1782 px wide, and has height/width **0.169**. Its camera therefore looks down at roughly **9.7°**, lower than Spyne's ~14°. `background_2` and `background_3` have no disc; they need a horizon or vanishing-line calibration.

---

## 6. What "fitting the images into 3D space" needs

### 6.1 What 2D can and cannot fix
(Sources in §7.2.)

| Can be corrected in 2D on real pixels | Cannot be corrected in 2D |
|---|---|
| Camera **rotation**, exactly, for any scene (re-aim, de-roll, re-pitch with a rotation homography) | **Viewing height**: how much roof, bonnet or sill shows |
| Focal length or crop (uniform scale) | **Perspective strength**, which is set only by camera-to-car distance |
| Camera distance at the anchor point, approximately (uniform scale) | Occlusions: parts never seen from that position |
| Small viewpoint offsets on floor-level points (ground-plane homography) | **Cropped-off parts** of the car |

The consequence: choose or build a background whose camera height and distance match the capture, and **fix cropping at capture**.

### 6.2 Track A: a 2D "virtual turntable" from real frames
This keeps the photographed pixels, so damage stays as shot. It fits our CPU-only laptop.

1. **Calibrate each background once.**
   - Get horizon, roll, pitch and field of view with GeoCalib or Perspective Fields (CPU), or with manual vanishing points in fSpy.
   - For `background_1`, start from the disc ellipse. The **true disc centre is not the ellipse centre**: it is the pole of the horizon line with respect to the ellipse, a little above the ellipse centre.
   - Pixels per metre at the centre ≈ major·√(1−k²)/D, for a chosen real disc diameter D (real car turntables are about 4.5–6 m).
   - Store the floor homography with the result.
2. **Measure each frame's real viewpoint** with COLMAP/pycolmap (SfM) on CPU, using sequential matching with loop detection, or the global mapper.
   - Derive the up-axis, the car's **footprint centre**, and each frame's azimuth, elevation and distance.
   - Get metric scale from the EQB's **2.829 m** wheelbase or its 1.667 m height.
   - Use the existing 8-class classifier only to label 0° and the direction of travel. Single-image estimators are off by ~6–15° and confuse front with back.
3. **Choose frames by angle, not time.** The clip has ≈ 1,145 frames, about 3 per degree. For each 10° (or 5°) target, take the **uncut**, sharp frame with the smallest angle error and the closest elevation and distance. Reject frames whose car mask touches the photo border.
4. **Scale from geometry**: `s_i = m_bg · z_i / f_i`.
   - `z_i` = depth of the footprint centre in frame i, from SfM.
   - `f_i` = that frame's focal length in px.
   - `m_bg` = background px/m at the disc centre.

   Keep each frame's own perspective. Keep the net scale ≤ 1: 4K → 1080p leaves ~2× headroom, so damage is never upsampled. *Do not* scale by "wheelbase in pixels": it collapses near the front and rear views.
5. **Anchor the footprint centre on the true disc centre.** Find it from triangulated wheel hubs or tyre contacts, not from a raw point centroid or the centre of the walking loop. Add a small roll correction. If keystone shows, re-aim with the rotation homography. Smooth the centre and scale tracks around the closed loop.
6. **Rotate the turntable**: warp only the disc texture by each frame's angle, `H_floor · R(θ) · H_floor⁻¹` about the true centre. Keep the studio's shading fixed. Or use plates with a plain disc.
7. **Ground the car with layers under it**, driven by the same floor model:
   - **Footprint shadow:** the car's 4.68 × 1.83 m rectangle, rotated by θ and projected onto the floor.
   - **Tyre-contact shading.**
   - **A dark core under the body.** Without it the car floats.
   - On glossy floors, a faint **per-column mirrored reflection**.
8. **Brightness consistency (needs a ruling):** at most one bounded global gain per frame, estimated from the car itself across neighbouring frames and solved around the loop. Never colour-match to the background.
9. **Off-limits under our rule:** relighting, harmonisation networks, super-resolution, dehaze on the car, frame interpolation, and generative or mirrored fill on the car (§7.6).

### 6.3 Track B: a full 3D rebuild, like Spyne
Reconstruct the car (Gaussian splatting or NeRF, or photogrammetry) and render the turntable from the studio camera.

- **Gains:** the most natural result, and complete views from cropped video.
- **Costs:** needs a CUDA GPU or a cloud service. On this laptop's CPU it would take roughly 1.5–2 days per car.
- **Risks:**
  - Glossy paint and glass are known failure cases.
  - Rendered views reach only ~28.7 dB PSNR, a pixel-accuracy score; at that level renders are visibly not identical to the photos, so **fine dents and scratches can be softened or changed**.

Possible compromise: use 3D only for **geometry** (poses, a proxy for shadows and reflections), not for car pixels.

### 6.4 Practical order
1. **Fix capture first**, following Spyne's own rules: 3–3.7 m away, 1.2–1.5 m high, 1× lens, steady, one closed loop, car never cropped, overcast or diffuse light. Then every angle has a real frame.
2. **Calibrate `background_1`** and prototype Track A on 3–4 hand-picked frames (side, ¾, rear). Check anchor drift and tyre contacts.
3. **Add SfM-based angle selection** for a full 36-frame spin.
4. **Trial Track B** later, only if a GPU is available, with a damage-fidelity check.

---

## 7. Verified research notes

### 7.1 What Spyne itself publishes

**Spin types** ([Spin 360 wiki](https://sites.google.com/spyne.ai/spin-360/spyne-spin-360)):

| Type | Method | Time |
|---|---|---|
| **Original Walkaround** | *"created without any AI input. It only picks different frames … stiches them together, and stabilizes the output"* | ≈ 10 min |
| **Turntable Studio** ("virtual replica of a car studio"), **Showroom** (built from a walkaround of the dealer's real showroom), **Outdoor** | "360 Stabilization, transformation" and "Super-resolution – All colors are brightened" | ≈ 30 min each |

**Generation methods.** The same wiki: *"two ways to generate the Spin 360 output: 1. Gaussian Splatting (medium stabilisation) 2. NeRF (Neural Radiance Field) 3D modelling (high stabilisation)"*. It also mentions separate endpoints for original, studio, and "3D walkaround (Showroom)". It does not say explicitly which method the Turntable Studio uses. A Spyne research job ad asks for NeRF, Gaussian Splatting and multi-view geometry work.

**API** ([3D 360-Spin](https://docs.spyne.ai/reference/post_360-v1-3d-2), [merchandise process](https://docs.spyne.ai/reference/merchandise-process.md)):
- `POST https://api.spyne.ai/360/v1/3d` takes `videoUrl`, a Studio-type `backgroundId`, `exteriorFrameCount` (36 or 72), `generatePhotos.catalogCount` (8 or 12 stills) and `interior360Image`.
- It *"Takes an hour"* and is *"Recommended for unstable video inputs"*.
- The v2 merchandise process has `spinFrameCount` "36" (default) or "72", plus a hotspot flag.
- Spins are delivered as an iframe `…/360?sku_id=…&image_category=3D_Model`, with the response API polled about every 20 min.
- Frame metadata such as `first_frame_angle` is **not** publicly documented.

**Video validation (all videos):**
- car present;
- **car not cropped in any frame**;
- **30–90 s**;
- **start point = end point**.

`POST /360/v1/validate-video` checks "Cropping, Angle, Car Present". Thresholds are not published.

**Capture guidance:**
- Camera at chest level and stable; full circle in 60–90 s; car in frame the whole time. The guideline decks add **10–12 ft away** and **4–5 ft high**.
- The app extracts **8 equidistant angles** from an exterior video.
- The [photo-angle guide](https://www.spyne.ai/blogs/?p=28329): ¾ views from just below hood height at about 40–45°; sides at door-handle height; front and rear at eye level; ~50 mm for the hero shot and 70–100 mm for sides; on phones *"Step back instead of using digital zoom"*; the same camera height across inventory.

**[dos-donts.pdf](https://resources.spyne.ai/dos-donts.pdf)** is a 21-page image-only deck with 19 rules, including:
- car body not cropped;
- no person in or within ~6 ft of the car;
- car clean, on flat solid ground, windows up, wipers parked, doors closed, wheels straight, headlights off;
- no rain;
- avoid harsh or low light;
- avoid colourful billboards (they cause a **colour cast** on the body) and sunset reflections;
- shoot only when in focus.

Spyne handles dirt by **asking for a clean car**, not by retouching.

**Stills** ([shoot-recommendations.pdf](https://resources.spyne.ai/spyne-input-shoot-recommendations.pdf)): 8 input→output pairs at 1620×1080.
- The studio wall is (near-)identical in all outputs, but the **wall-to-floor line moves per image (0.60–0.72 of the height)** with the view.
- The car is centred, ~0.46–0.48 of the width for front and rear and ~0.63–0.65 for ¾ and side.

For stills, Spyne **adapts the background's floor and horizon to each photo**. That changes the background, not the car, so it is allowed under our rule.

**Still pipeline steps** (sales decks): background replacement; "AI generates the floor beneath the car"; drop shadow; window tint and see-through masking; auto tilt correction; plate mask or logo; resolution enhancement; marketplace resizing; "50+ models"; three QC stages, including manual QC.

**Processing and backgrounds:**
- GPU batch processing, 45–60 min per car ("30 mins/output … 16 outputs a day"). Original Walkaround ≈ 10 min.
- Background library: Studio 01–29, Multiwall 30–34, Outdoor 35–42. Custom turntable studios cost up to $400.

### 7.2 Geometry: calibration, the turntable model, and the limits of 2D

**Background camera from one image:**
- [Perspective Fields](https://arxiv.org/abs/2212.03239) (CVPR 2023) predicts roll, pitch and field of view. Its perspective-agreement score (APFD) had the **highest correlation with human judgement of composites** (median Pearson 0.87).
- [GeoCalib](https://github.com/cvg/GeoCalib) (ECCV 2024) gives intrinsics and gravity, with a CPU fallback.
- [fSpy](https://github.com/stuffmatic/fSpy) uses manual vanishing points.
- Seamless studio plates usually have no straight lines, which limits classic vanishing-point metrology.

**Horizon, height and size** ([Zhu et al., ECCV 2020](https://arxiv.org/abs/2007.09529); Hoiem et al.):
- Pitch θ = arctan((v_c − v₀)/f).
- For an object on the floor (small pitch): h_obj = h_cam·(v_t − v_b)/(v₀ − v_b).
- So car image height ≈ H_car·(v_contact − v_horizon)/h_cam. Once the contact row is fixed, **the car's pixel size is fixed**; it cannot be freely "fitted to a box".

**Turntable ellipse** ([OMEGA paper](https://arxiv.org/abs/1808.00530); [Mariyanayagam et al.](https://arxiv.org/abs/1804.04922)):
- Minor/major = sin(δ) under parallel projection, where δ is the elevation of the line of sight to the disc centre.
- Under perspective, b/a = sin δ / √(1−k²) with k = r·cos δ / d. So asin(b/a) slightly **overestimates** δ for close cameras.
- The image of the disc centre is the **pole of the horizon line** with respect to the ellipse, about k × semi-minor **above** the ellipse centre. One ellipse alone cannot reveal it.
- Physical turntables: a 5 m plate (PhotoRobot Carousel 5000); a studio turntable of 4.5 m that stops every 10° for 36 images (WebRotate360).

**Car dimensions** (2024 EQB): 4684 × 1834 × 1667 mm, wheelbase 2829 mm. Tyres are 235/55R18, ≈ 716 mm in diameter.

**Turntable equivalence** ([Kosaka et al., CVPR 2026 workshop](https://research.lycorp.co.jp/en/publications/2678)):
- A camera orbiting a static car and a fixed camera watching a turntable are the same single-axis motion.
- In a walkaround, the sun and reflections are fixed to the world, so they **swing around the car** across the spin. On a studio turntable they stay fixed to the camera.

**Why raw walkaround frames have the wrong sizes.** A walker keeping a roughly constant gap *s* from the body is ≈ 0.92 + s m from the centre at the sides and ≈ 2.34 + s m at the ends. A turntable spin needs one constant distance to the centre.

**Near end lower and larger** (pinhole):
- y − y_horizon = f·h/Z, and px/m = f/Z.
- Visible **tyre contacts** in end views drop only tens of pixels versus side views. The bumper underside and the shadow drop more.
- Uniform scaling matches size at the anchor, but **cannot change perspective strength**, which is set by distance alone ([perspective distortion](https://en.wikipedia.org/wiki/Perspective_distortion)).
- Example: a rear bumper 2.34 m nearer than the anchor, shot from 4 m but shown on a plate "seen" from 9 m, looks ~78% too large after anchor-matched scaling.
- Viewers notice stronger perspective on the object than on the background, and roll errors beyond about ±2.5° ([Hold-Geoffroy et al.](https://arxiv.org/abs/1712.01259)).

**Homographies** ([OpenCV tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)):
- `K R K⁻¹` re-aims a camera **exactly for any scene**.
- A plane-induced homography is exact only for that plane, here the floor. Body points above the floor get parallax.
- **Viewing elevation cannot be changed by warping a single image.**

### 7.3 Measuring each frame's viewpoint and choosing frames

**COLMAP** ([FAQ](https://colmap.github.io/faq.html), [tutorial](https://colmap.github.io/tutorial.html)):
- Runs the whole **sparse** pipeline on CPU (slower); only dense stereo needs CUDA.
- Sequential matching for video, with vocabulary-tree loop detection.
- One shared camera if the zoom is constant.
- Masks exclude glass and shiny panels.
- Warns about textureless and specular surfaces.
- **COLMAP 4.0** (Mar 2026) integrates the GLOMAP global mapper (claimed 1–2 orders of magnitude faster) and ALIKED + LightGlue via ONNX.

**Precedents:**
- [3DRealCar](https://arxiv.org/abs/2406.04875) (ICCV 2025): phone orbits of real cars, ~200 views each. COLMAP poses were used because ARKit poses were inaccurate. PCA alignment. *"Glossy materials bring huge challenges."*
- [CO3D](https://ai.meta.com/blog/common-objects-in-3d-dataset-for-3d-reconstruction/): ~19,000 phone "turntable" videos posed with COLMAP; poorly reconstructed videos were filtered out. Each run needs automatic validation.

**Single-image viewpoint models are too coarse for 10° spacing:**
- Render for CNN: ~6° median error on cars.
- Orient Anything V2: ~15° median 3D angle error.
- Both confuse front and back.
- Wheel ellipses give the yaw off a pure side view (b/a ≈ |cos ψ|) and distance from the tyre size, but they are poorly conditioned near a pure side view.

**Selecting frames.**
- Choose by **relative camera angle, not playback time**: CloudPano, *"normalize rotation, not playback time"*.
- 38.17 s × 30 fps ≈ 1,145 frames ≈ 3 per degree, so density is not the problem; cropping is.
- Smooth the centre and scale tracks as a path: [L1-optimal camera paths](https://research.google/pubs/pub37041/), the YouTube stabiliser method.

### 7.4 Placement maths
- **Similarity version:**
  1. Scale by `s_i = m_bg · z_i / f_i`.
  2. Translate so the footprint centre's image lands on the true disc centre.
  3. Rotate so the frame's vertical matches the plate's up-vector.

  Perspective Fields aligns sprites to backgrounds with the same similarity transform.
- **Exact-rotation version** (if pitch or keystone mismatch shows): remove lens distortion, then warp with `H_i = K_V · R_V · R_iᵀ · K_i⁻¹`, using focal length `f_V = f_bg · ‖C_i − G‖ / d_bg`, the *distance* to the footprint centre G rather than its depth.
- Rotation re-aiming is a resample of real pixels. It would, however, break today's bit-exact "only downscaled" test, so it needs a rule decision (§8).

### 7.5 Grounding and appearance with the car layer locked

**Layer order** (linear light): plate → rotated disc texture → floor reflection → shadow layers (multiplied) → car (premultiplied alpha, "over"). Premultiplied alpha avoids colour fringes; blend in linear light ([alpha compositing](https://en.wikipedia.org/wiki/Alpha_compositing)).

**Turntable rotation.** Warp only the disc texture with `H_floor·R(θ)·H_floor⁻¹` about the **true** centre. This is exact for floor-plane texture. Divide out a blurred shading layer first, so the studio's light fall-off doesn't rotate.

**Shadows:**
- Can be built from the 2D silhouette, as in SSN (CVPR 2021), Pixel Height (ECCV 2022) and remove.bg's "car" shadow.
- **Recipe:**
  - The footprint rectangle (4.684 × 1.834 m) rotated by θ and projected through the floor homography.
  - Tyre-contact shading.
  - A **dark core under the body** between the tyres; without it the car floats.
  - Combine as `floor × ∏(1 − oᵢ·mᵢ)`. A sum can go negative.
- Soft-shadow physics (PCSS): **sharp at the tyres, softer toward the sills**.
- Studio cars are lit by a large overhead source, so expect a soft shadow centred under the car and no long cast shadow.

**Floor reflection** (glossy floors only):
- Mirror **each column about its own contact point** with slight vertical compression (~0.75–0.95). A whole-car flip about one line fails in ¾ views.
- Opacity from Schlick's Fresnel approximation (F₀ ≈ 0.04): camera 20° / 15° / 10° above the floor gives F ≈ 0.16 / 0.25 / 0.41. Times the floor's glossiness, that is roughly 10–30%.
- Fade and blur with distance.

**Edge fringe** (our code today resizes RGB and straight alpha separately, then blends in sRGB):
- Resample premultiplied.
- Recompute colour **only in the 0 < α < 1 band** with foreground estimation: [Photoroom fast-foreground-estimation](https://github.com/Photoroom/fast-foreground-estimation), [PyMatting](https://github.com/pymatting/pymatting), BiRefNet's `refine_foreground`, or rembg's `decontaminate`. **α = 1 pixels come out unchanged**, so this is safe for damage.
- Refine the 1024² mask at 4K with a guided filter, instead of eroding the edge (`edge_shrink` trims real edge pixels).
- Clear semi-transparent old-ground haze below the contact line, and keep the gap under the body transparent.

**Brightness and colour consistency** (if allowed):
- One bounded, global, per-channel (von Kries) gain per frame, in linear RGB.
- Estimate it from **matched car-surface points in neighbouring frames** (highlights excluded), solve jointly around the closed 36-frame loop, and smooth circularly. This is panorama gain compensation: Brown & Lowe; [OpenCV GainCompensator](https://docs.opencv.org/4.5.4/d0/de1/classcv_1_1detail_1_1GainCompensator.html).
- **Never** equalise each frame's mean car brightness, and **never** transfer colour statistics from the background: Reinhard-style transfer pulls the paint colour toward the studio's.
- Note: after clipping and 8-bit encoding the gain is not perfectly invertible.

**Glass** (needs approval):
- Commercial tools treat car windows specially: remove.bg's semi-transparency exists only for car windows, and Spyne lists "Window See-through (for foreground windows)".
- Masks:
  - [HITL car-parts dataset](https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/): CC0, with window, plate **and damage** classes.
  - Ultralytics carparts-seg: front and back glass only.
  - Zero-shot: CLIPSeg or Grounded-SAM, on CPU.
- Any tint should keep part of the glass's fine detail: **a cracked windscreen is damage**.

**Plate to logo.** Find the plate's quadrilateral (WPOD-NET-style rectification, or contours), not just the detector box. Perspective-warp an opaque logo panel, keep the plate's low-frequency shading, and confine the edit to that quadrilateral.

### 7.6 What not to use, and why (damage fidelity)

- **On glossy paint, dents are seen as distortions of the reflected scene.** That is how industrial deflectometry finds them. Anything that re-renders shading or reflections can **hide, move or invent dents**, even if fine texture is restored afterwards.
- **IC-Light** (relighting) is latent diffusion. Known issues: detail loss, colour bleeding, per-frame flicker. It needs a GPU, and V2 is non-commercial.
- **Harmonisation networks:**
  - DoveNet regenerates the whole image at 256 px.
  - PCT-Net applies per-pixel colour transforms.
  - Harmonizer applies only global filters, the closest to safe, but it is non-commercial.

  The safe pattern is to use any network only as an *estimator*: fit one global colour map, then apply it to the untouched car. Deep White-Balance Editing is a precedent.
- **Frame interpolation** (FILM, RIFE) synthesises regions that were never seen (*"realistic inpaintings of large pre-occluded regions"*). Optical flow breaks on specular paint, glass, grilles and spokes.
- **Outpainting and diffusion novel views** invent pixels. Stable Virtual Camera's model card: *"not trained to be factual"*.
- **Stitching neighbouring frames** to complete a cropped view is exact only for planar scenes or pure rotation. Wheels and mirrors ghost; panels are nearly planar, but a seam can duplicate or cut a scratch.
- **Flow-guided video completion** (FGVC; ProPainter, which is non-commercial and needs about 19–28 GB of GPU memory) still hallucinates whatever was never seen.

### 7.7 3D reconstruction (Track B) evidence

- **A white glossy car with glass is the worst case.** Photogrammetry guides say to avoid untextured, shiny, reflective and transparent objects (Agisoft, COLMAP).
- **Radiance fields** fail on gloss and glass. [Ref-NeRF](https://dorverbin.github.io/refnerf/) documents the gloss problem. In [3DGS-DR](https://arxiv.org/html/2404.18454), *"The front window converges to a purely reflective surface and the side windows … purely transparent"*.
- **Car-specific results:**
  - [Car-GS](https://arxiv.org/html/2501.11020v1) targets reflective paint and transparent windows.
  - A [2026 dealership drive-through pipeline](https://arxiv.org/html/2603.26638v1) (14 fixed 4K cameras) reaches **PSNR 28.66 dB** on held-out views, with remaining errors in "highly specular, low-texture regions".
- **Building a car from few photos** is established research: [NeRS](https://jasonyzhang.com/ners/) builds one from 8–16 unposed photos. The output is a re-rendered appearance. [2DGS](https://surfsplatting.github.io/) gives cleaner geometry, which is useful as a proxy for shadows and reflections.
- **Compute:**
  - Inria 3DGS needs CUDA (24 GB for paper quality) and is research-only.
  - OpenSplat (AGPL) runs on CPU at ~100× slower.
  - Brush (Apache-2.0, WebGPU) is unbenchmarked on Iris Xe.
  - Cloud services such as Polycam and KIRI turn phone video into splats.
- **Shadow-catcher compositing.** Blender Cycles' **shadow catcher** renders shadows from a proxy object for compositing real pixels on top. Karsch et al. (SIGGRAPH Asia 2011) is the classic "insert objects into a photo" method.

### 7.8 How other vendors describe it

- **Impel (SpinCar):**
  - Sensor fusion plus optical stabilisation.
  - Gyroscope-guided capture with live prompts (speed, levelling, direction).
  - Automatic spin completion.
  - Its "AI Virtual Booth" *"automatically adjusts the image for perspective, vehicle size and background"* regardless of camera distance, pitch or height.
- **Fyusion** (acquired by Cox Automotive, 2021): multiplane-image view synthesis (co-authors of LLFF).
- **CloudPano Spin API:** frames selected by relative camera angle; centring for "consistent position and apparent scale"; "mirrored edge fill" for small gaps; 96 frames by default.
- **Glo3D, Dealer Image Pro, CarStudio AI, CarCutter, Phyron:** feature lists only, no methods.

### 7.9 CPU feasibility on this laptop (i5-1240P, 16 GB, no GPU)

| Step | Feasible? | Note |
|---|---|---|
| COLMAP / pycolmap sparse SfM | Yes | CPU mode is slower; use frames at ~1600 px with masks; mapping time unmeasured |
| Background calibration (GeoCalib, Perspective Fields, fSpy, ellipse fit) | Yes | Perspective Fields' demo hard-codes `.cuda()` and needs a small change |
| Cut-out (rembg `birefnet-general-lite`) | Yes | ≈ 13–17 s per frame measured here (up to ~31 s) → ≈ 8–10 min for 36 frames |
| Warps, shadows, reflection, disc rotation, gains | Yes | Plain OpenCV / NumPy |
| Blender Cycles shadow-catcher plates | Probably | Unbenchmarked |
| VGGT, 3DGS training, IC-Light, diffusion | No | Need CUDA, or days on CPU |

The integrated Iris Xe GPU could be tried through ONNX Runtime DirectML or OpenVINO. Any speed-up is unverified.

### 7.10 Licences to check before using anything

| Licence | Tools |
|---|---|
| **Non-commercial or research-only** | Inria 3DGS, Harmonizer, MASt3R, ProPainter, Deep White-Balance, IC-Light V2, Depth Anything V2 Base/Large, Stable Virtual Camera |
| **Copyleft** | OpenSplat (AGPL), Ultralytics (AGPL), fSpy (GPL), WPOD-NET PyTorch port (GPL) |
| **Conditional** | VGGT (only its "Commercial" checkpoint), GeoCalib weights (CC-BY-4.0) |
| **Permissive** | COLMAP / GLOMAP (BSD), LightGlue and DISK (Apache-2.0), ALIKED (BSD-3), Brush (Apache-2.0), HITL car-parts dataset (CC0) |
| **Not established** | Perspective Fields |

---

## 8. Open decisions for the team

- **Where exactly is the "don't alter the exterior" line?** Each of these touches car pixels in some way. (Number plates are already altered.)
  - (a) Rotation-homography re-aiming (a resample of real pixels; it breaks today's bit-exact test).
  - (b) One global brightness gain per frame.
  - (c) Colour clean-up of the semi-transparent edge band only.
  - (d) Glass tint or see-through, where cracks count as damage.
  - (e) Mirrored edge fill near the car.
- **Cropped angles:** re-shoot with the whole car in frame (recommended), skip those angles, or allow synthesis?
- **Deliverable:**
  - 36 or 72 frames?
  - Stills too (Spyne gives 8 or 12)?
  - Which aspect ratios (Spyne: 16:9, 3:2, 4:3)?
  - Which start angle and direction? Spyne's "first_frame_angle 45, clockwise" is undocumented, and its frame 1 is a left profile.
- **Backgrounds:** calibrate the flat plates, or rebuild them as simple 3D scenes (floor, disc, walls, camera matched to the capture)? Would the outdoor `background_3` suit an outdoor clip's lighting better than the white studio?
- **Track B:** trial a GPU or cloud 3D rebuild at all?

## 9. Unknowns to measure next

1. **Angle coverage:** which 10° targets have at least one uncut, sharp frame? `results.csv` flags 5 of the 20 samples as cut off, and the pure left side has none. This decides whether a re-shoot is mandatory.
2. **The clip's camera path:** does CPU SfM register this glossy, close-range clip? Is it one closed loop? How much do height and pitch drift?
3. **Background cameras:**
   - `background_1`: horizon row, field of view, true disc centre, and a chosen disc size.
   - `background_2` and `background_3`: vanishing lines or a learned horizon.
4. **Capture vs plate elevation:** a chest-height walk 3–3.7 m from the body looks down roughly 11–21°, compared with ≈ 9.7° for `background_1`.
5. **Zoom and stabilisation:** is the zoom in "Final_zoomed_car" constant, and was electronic stabilisation on? Does an un-zoomed original exist?
6. **Lighting:** direct sun or overcast, and from which direction? Which plate suits it?
7. **Footprint centre and scale:** test triangulated wheel hubs and tyre contacts.
8. **Spyne's virtual camera:** solve it from its 36 frames (horizon, focal length, distance), re-measured with tyre contacts only.
9. **Costs and acceptance tests:** SfM time on ~380–570 frames. Checks for "looks natural":
   - anchor drift from the disc centre;
   - scale smoothness versus angle;
   - tyre contacts on the projected footprint;
   - flicker;
   - Perspective Fields' APFD score.

---

## 10. Evidence log

| Item | Where / how |
|---|---|
| Demo page | https://console.spyne.ai/virtualstudio/demo-v2: 360° Spin tab, "View input", enhancement cards, before/after sets |
| Spin viewer | `https://assets.spyne.ai/360?sku_id=…` (iframe on the demo page) |
| Frame list and metadata | `api.spyne.ai/api/v1/hotspot/get-sku-image-data?skuId=…&type=output` (`FRAME_INT`, `AI_ANGLE`, `skuCategory`, `sequenceData`) |
| Input video | `Final_zoomed_car.mp4` on Spyne's CDN: 3840×2160, 38.17 s, 30 fps (the same as our local copy) |
| Guideline video | https://resources.spyne.ai/360-guidline-video-v5.mp4 (frames read in the browser) |
| Help resources | https://resources.spyne.ai/spyne-input-shoot-recommendations.pdf · https://resources.spyne.ai/shooting-postions-app-shoots.pdf · https://resources.spyne.ai/dos-donts.pdf |
| Spyne documentation | https://sites.google.com/spyne.ai/spin-360/spyne-spin-360 · https://docs.spyne.ai/reference/post_360-v1-3d-2 · https://docs.spyne.ai/reference/merchandise-process.md · https://www.spyne.ai/blogs/?p=28329 |
| Pixel analysis | All 36 frames, downscaled 4× in the browser:<br>• per-pixel range across frames (static map)<br>• average image<br>• roof and lowest rows per frame<br>• turntable rim ellipse fit<br>• full-resolution crops of frames 18, 27 and 36 |
| Local comparison | `frame_view_type/output/Final_zoomed_car/` frames 0000, 0006, 0012, 0028, 0030, 0032. Video frames at 26–33 s read in the browser. `background_changer/output/Final_zoomed_car/results.csv` (our placement numbers). |
| Our background | `background_changer/backgrounds/background_1.webp`: rim ellipse fit with OpenCV (read-only) |
| Research | A 9-agent read-only research run (4 areas, each fact-checked, plus a completeness review). Only claims rated supported or partly supported are used; two unconfirmable claims (a deleted Spyne blog post, a patent search) were left out. |
