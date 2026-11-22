# See-through car glass: what CarDekho listings do, and our options

*Findings from two CarDekho used-car listings (a 2022 Nissan Magnite photographed by Spinny and a 2021 Tata Nexon photographed by Cars24), plus web research. Collected on 2026-11-14. This is information only: no code was changed.*

**How to read this**

| Section | Contents |
|---|---|
| §1 | The short answer |
| §2 | What the two reference listings show, with pixel measurements |
| §3 | Can we see through to the far-side windows? |
| §4 | Options, cheapest and safest first |
| §5 | Where vision-language models help |
| §6 | Fit with our rules and this laptop |
| §7 | If we try it later |
| §8 | Sources |

*Inference* marks my own reasoning.

---

## 1. In short

**The look.** On CarDekho's turntable-studio photos:
- the glass carries no picture of the surroundings, only a faint, even haze;
- the cabin's shapes stay visible: headrests, seat backs, steering wheel, pillars, the far-side mirror;
- the backdrop shows through the **far-side** windows, slightly darker and tinted.

**How it is made.** Across many postings the reflection on the glass is **the same on every car, whatever the model**: one stock "sample reflection" (the team noticed this; I did not re-check it). So the glass is built, not photographed:
1. mask the glass;
2. replace the old outside view with the backdrop, darkened;
3. lay one fixed reflection texture over the glass at low opacity.

**Can we do it?** Yes. The background seen through both panes is the new background at the same pixel position, and we already render that. The hard part is the **mask**: deciding, for every glass pixel, whether it shows the outside (replace it), a part of the cabin (keep it), or a reflection (cover it).

**Conflict with our rule.** Today the car's pixels are untouched and windows keep their original glass ([`bg_changer/segment.py`](../../background_changer/bg_changer/segment.py#L117-L133) `fill_holes`; [findings.md](../../background_changer/docs/findings.md#cutting-the-car-out)). Every option below except Option 0 needs an exception **for the glass area only**, with cracks, chips and stickers kept. [spyne-360-spin-findings.md](../../background_changer/docs/spyne-360-spin-findings.md) §7.5 already marks glass as "needs approval": a cracked windscreen is damage.

---

## 2. What the two reference listings show

### 2.1 2022 Nissan Magnite (Spinny, 40 photos at 1500×844)

- **Side profile:** through the near windows you see the driver headrest, the steering wheel, the far B-pillar and the rear headrests. The far windows show the pale backdrop. The glass shows no sky, trees or photographer.
- **Front view:** through the windshield, the seat backs hide the lower part of the rear glass. Above them the backdrop shows as a bright patch with a sharp, stepped lower edge.
- **Rear three-quarter view:** the same pattern. A sticker on a rear door window is kept.

Pixel check, front view (9×9 averages):

| Where | RGB |
|---|---|
| Backdrop beside the car | (197, 207, 217) |
| Backdrop seen through the windshield and rear glass | (217, 242, 249) |
| Dark windshield over the dashboard (≈ the reflection haze) | (88, 101, 113) |
| Prediction: 0.65 × backdrop + haze | (216, 236, 254) |

*Inference:* this fits a **real photo in a physical white turntable bay**. The reflections are there, but they reflect a featureless white room, so they read as a clean haze. One compressed JPEG is not proof.

### 2.2 2021 Tata Nexon (Cars24, 35 photos at 1920×1080)

- **Side profile:** the near glass is dark grey with clear cabin shapes: seat edge, steering wheel, A- and B-pillars, and the far-side mirror seen through the cabin. The far windows are light grey "holes" with crisp, anti-aliased edges that follow the far window frame, the wheel and the mirror.
- **Pixel check:**
  - the holes are about (213, 218, 212), roughly 0.93 × the backdrop (226–234);
  - they are **perfectly flat**: standard deviation 0.0–0.14 over 13×13 px, against 0.5 on the backdrop;
  - they have the **same value in both windows**.
- **Front three-quarter view:** the windshield is a light haze with faint streaks. The seats and headrests show through the side window, and the far window shows the backdrop.

*Inference:* this looks **processed**: a cut-out on a clean studio backdrop, with the far-window areas filled or made transparent and only lightly darkened. The faint streaks fit the shared stock reflection described in §1.

### 2.3 What both have in common (the look to copy)

1. The near glass carries only a **featureless haze**.
2. **Cabin parts** stay: headrests, seat backs, wheel, pillars, far mirror, stickers.
3. The far glass shows the **new backdrop, slightly darker and tinted**: 0.65× plus haze on the Magnite (physical), 0.93× and flat on the Nexon (processed).

---

## 3. Can we see through to the far-side windows? Yes

- **The geometry is free.** Light passes almost straight through thin glass. What shows through both panes is the **new background at the same pixel position**, which we already render (flat or 3D room). No 3D model of the car is needed.
- **The blend is simple.** For a pixel that sees through both panes: `out = T·B_new + H`
  - `T` ≈ 0.6–0.9 is the transmission of two panes, with a slight cyan or green tint.
  - `H` is a small, smooth haze taken from the new scene's brightness. It stands in for the studio reflection.
- **The hard part is the mask.** Every glass pixel is one of three kinds:
  1. **Hole:** the old background shows through both panes. Replace it with `T·B_new + H`.
  2. **Cabin part:** keep the photo's pixels, cleaned if allowed.
  3. **Reflection on the near glass over the cabin:** clean it, or cover it.
- **Where it works well:** the far window shows sky, a wall or distant background that contrasts with the cabin. This is the common lot photo.
- **Where it fails, so the photo is flagged "unsure" and keeps its original glass:**
  - privacy or very dark tint, where nothing shows through;
  - another car or a person right behind the car;
  - strong sky reflections that hide the cabin;
  - night or flash shots;
  - cracked glass (damage must stay visible).

---

## 4. Options, cheapest and safest first

### Option 0: Capture side (no pixel edits)

- **A circular polarising filter on the phone, at roughly 50–60° to the glass.** This is near Brewster's angle (about 56° for glass). It cuts most glass glare and lets the cabin show. It never removes all of it, and it is weak head-on (the windshield from the front).
- **Keep the sky out of the glass:** shoot in shade, in overcast light, or under a canopy.
- This creates no see-through on its own, but it makes every option below more reliable: fewer reflections are mistaken for cabin parts.

### Option 1: Reuse our own cut-out's holes (no new model, CPU, repeatable)

- **Idea:** BiRefNet (`birefnet-general-lite`) already marks the far-window areas as background when they show the same scene as outside the car. `fill_holes` then fills them back in on purpose.
- **What it would take:**
  - keep those holes open where they lie inside a window (`car_isolation`'s `glass_ok` already marks window holes);
  - composite them as `T·B_new + H`;
  - clean the hole edges with foreground estimation ([PyMatting](https://github.com/pymatting/pymatting), [Photoroom fast-foreground-estimation](https://github.com/Photoroom/fast-foreground-estimation)), as already planned for the outer edge.
- **Gives:** the far-side see-through.
- **Does not give:** clean near glass; the reflections over the cabin stay.

### Option 2: Masks from a model

- **Glass mask, from any one of:**
  - **SAM 3** (Meta, November 2025, ICLR 2026) with text prompts such as "car window" and "windshield". Code and weights are open; the licence still needs checking. A GPU is recommended; on CPU it is slow.
  - A small segmenter trained on the **Humans in the Loop car-parts dataset** (CC0; window, windshield **and damage** classes).
  - **Gemini 2.5** segmentation (paid API), which can return masks for a text prompt. Not tested here.
- **Cabin parts:** SAM 3 prompts such as "seat", "headrest", "steering wheel", "side mirror".
- **Holes** = glass − cabin parts − pillars. Cross-check against Option 1's holes, and against colour continuity with the background just outside the car.
- **When they disagree:** mark the photo "unsure" and keep the original glass, or fall back to Option 5.

### Option 3: Remove reflections from the near glass (glass crop only)

These single-image reflection-removal models output the "transmission layer": the cabin and the far view, without reflections.

| Model | Base / type | Licence | Hardware |
|---|---|---|---|
| **RDNet** | Won the NTIRE 2025 reflection-removal challenge ([code](https://github.com/lime-j/RDNet)) | to check | GPU |
| **XReflection** toolbox | DSRNet, DSIT and others | Apache-2.0 | GPU (CPU slow) |
| **WindowSeat** (Huawei Bayer Lab, Dec 2025) | Qwen-Image-Edit-2509 (20B) fine-tuned for window reflections | Apache-2.0 | One large GPU (cloud) |
| **GenSIRR / SIRR-LMM** | FLUX.1 Kontext-dev | FLUX dev weights are non-commercial | GPU |
| **Adobe Camera Raw "Reflection Removal"** | Manual tool | Commercial | Works best when the glass fills the frame; weak on small side windows |

- **Risk:** the generative ones (WindowSeat, GenSIRR) can invent cabin detail, and can soften or hide **cracks and chips**.

### Option 4: General image editors, copying back only the glass

- **Paid:** Gemini 2.5 Flash Image ("Nano Banana"), OpenAI gpt-image-1, FLUX.1 Kontext pro (Black Forest Labs API).
- **Open:** Qwen-Image-Edit-2509 (Apache-2.0, 20B, GPU), FLUX.1 Kontext-dev (non-commercial weights).
- **How it would be used:**
  1. Composite the car onto the new background.
  2. Prompt: "clear glass, no reflections, keep seats and headrests, show the background through the far windows".
  3. Copy back **only the glass-mask pixels** from the edit. The body stays our bit-exact composite.
- **Risks:**
  - an invented or mismatched cabin (wrong headrest shapes);
  - hidden cracks;
  - a different result on every run;
  - lost resolution and compression. One reviewer says Nano Banana "takes certain creative liberties" and compresses the image.
  - FLUX Kontext tends to change only what it is told to; Qwen edits more freely.
- **Best use:** quick prototypes of the look, or a fallback. Not the main path for condition photos.

### Option 5: Dark fill where the cabin is unclear

- Where the cabin cannot be made out, fill the glass with a **uniform dark tint**, and open holes only where the mask is confident.
- Do **not** draw synthetic headrests: that would be invented content.
- Repeatable and safe; it reads like tinted glass.

### Option 6: Buy it (useful as a benchmark)

- **remove.bg / ImageKit:** semi-transparency exists **only for car windows**.
- **imgix:** `bg-remove-semi-transparency`, Premium plan, car windows only.
- **Spyne:** "Window See-through (for foreground windows)". [spyne-360-spin-findings.md](../../background_changer/docs/spyne-360-spin-findings.md) found that Spyne **replaces** the glass pixels (uniform tint, reflections gone).
- **Use:** send 10–20 of our own photos and compare against Options 1–3. The photos leave the machine, so this needs approval.

### Option 7: One stock reflection on all glass (what the listings appear to do)

- On the glass mask, lay **one fixed studio-reflection texture** (a soft gradient with a few streaks), warped to each window's shape, at low opacity (screen or add blend). Apply it after Option 1 or 5 has handled the holes and the cabin.
- **Gains:** it hides the photo's own reflections and gives every car the same studio look. No model, the same result every run, milliseconds on CPU.
- **Costs:** it is plainly not the car's real glass. It must stay faint enough for cracks and chips to show, or be skipped when damage is found.

---

## 5. Where vision-language models help

- **Masks:** Gemini 2.5 segmentation. SAM 3 also takes text prompts, but it is not a VLM.
- **Checking (Claude, GPT, Gemini):**
  - "Does this window show the outside through the far glass?"
  - "Is there a crack or chip in the windscreen?"
  - "Does the edited glass match the original cabin layout?"

  Use them to **flag** photos, not to paint pixels.
- **Generating pixels:** Option 4. The highest risk for condition photos.

---

## 6. Fit with our rules and this laptop (i5-1240P, 16 GB, no GPU)

- **On CPU today:** Options 0, 1, 5 and 7, plus the blend maths. SAM 3 would be slow.
- **Need a cloud GPU or a paid API:** SAM 3 at speed, RDNet, WindowSeat, the Qwen and FLUX editors, Gemini, gpt-image.
- **Rule change needed:** allow edits **inside the glass mask only**.
  - Keep cracks, chips and stickers. Look for damage (the Humans in the Loop damage classes, or a VLM check) and keep the original glass when it is found.
  - Record per photo whether the glass was edited or marked unsure.
  - Always keep the untouched export.

---

## 7. If we try it later (smallest experiment first)

1. **Options 1 + 7** on about 20 existing photos (holes from the current model, `T·B_new + H`, one stock reflection). Check whether the far-side see-through looks right, and how often the holes are wrong.
2. **Add SAM 3 glass and cabin masks** (Option 2) and compare hole accuracy.
3. **Run reflection removal** (Option 3: RDNet, then WindowSeat on a rented GPU) on the glass crops. Look closely at cracked or stickered glass.
4. **Benchmark a paid service** (Option 6: remove.bg or Spyne) on the same 20 photos.

---

## 8. Sources

**Reference listings** (viewed on cardekho.com on 2026-11-14): used 2022 Nissan Magnite XL (Spinny, Hyderabad); used 2021 Tata Nexon XMA AMT S (Cars24, Hyderabad).

**Commercial tools**
- [Spyne feature list](https://sites.google.com/spyne.ai/image-merchandising/home)
- [imgix background-removal semi-transparency](https://docs.imgix.com/apis/rendering/background/background-removal-semi-transparency)
- [ImageKit background removal](https://docs.imagekit.io/extensions/overview/background-removal)

**Reflection removal**
- [NTIRE 2025 Challenge on Single Image Reflection Removal in the Wild](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Yang_NTIRE_2025_Challenge_on_Single_Image_Reflection_Removal_in_the_CVPRW_2025_paper.html)
- [RDNet: Reversible Decoupling Network](https://huggingface.co/papers/2410.08063)
- [XReflection toolbox](https://xreflection.readthedocs.io/en/latest/)
- [WindowSeat announcement](https://huggingface.co/posts/toshas/756417426105326) (paper: arXiv 2512.05000)
- [GenSIRR](https://huggingface.co/lime-j/GenSIRR), [SIRR-LMM](https://arxiv.org/html/2601.07209)

**Segmentation**
- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719), [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3)
- [Humans in the Loop car-parts and car-damages dataset](https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/)

**Image editors**
- [Qwen-Image-Edit vs FLUX.1 Kontext](https://diffusiondoodles.substack.com/p/qwen-image-edit-vs-flux1-kontext)
- [Fixing window reflections (Nano Banana, Camera Raw)](https://fstoppers.com/lightroom/how-fix-window-reflections-lightroom-cant-remove-905001)

**Capture**
- [Polarising filters in car photography](https://www.autopix.no/blog/10-powerful-reasons-to-use-a-polarization-filter-in-car-photography-and-how-to-do-it-right)
