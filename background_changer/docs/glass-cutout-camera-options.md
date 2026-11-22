# Glass reflections, cut-out and camera angle: model options and costs

Research note, 23 November 2026. This is information for analysis only: nothing has been built or tested on our photos.
Prices are list prices, mostly from third-party price trackers. Check them on each vendor's own page before
budgeting.

## 1. The goal, and what it means for choosing models

**The goal:** take each frame of a car, cut the car out neatly, clean up the glass, and put it into a professional
studio. Every frame of the same car must sit in **the same room**.

That last point decides most of the choices:
- A generative model (OpenAI, Gemini, Qwen, Seedream and so on) draws a new picture every time it is called. It cannot
  promise the same room, floor and lights across 36 frames.
- So the room is a fixed 3D model, re-drawn from each frame's **measured** camera position. This is what room mode
  already does.
- Chat/vision models can't measure that camera either:
  - On a 2026 benchmark of camera movement between image pairs (VRRPI-Bench), the best model scored only 59.7%.
  - Another 2026 paper reports a standard vision-language model at 5% on a camera-pose accuracy score (AUC30) before
    special training.

**Three jobs, three kinds of model:**

| Job | Right kind of model | Touches the car's pixels? |
|---|---|---|
| Cut the car out | Segmentation: returns a mask only | No. We keep our own pixels and only use the mask |
| Clean the glass | Tint or replace the glass, or generative clean-up of the glass area | **Yes.** This is the only one of the three that changes the car itself |
| Camera angle | 3D geometry: camera calibration, depth, multi-view pose | No |

## 2. Can OpenAI, Gemini, Chinese models or Claude do these jobs?

| | Cut-out | Glass reflections | Camera angle |
|---|---|---|---|
| **OpenAI GPT Image 2** | **No.** It redraws the whole picture, so the car you get back is a redrawn car, not the photo. The edit mask is only a hint: developers report the model redraws everything | **Possible, with conditions** (§4) | **No** |
| **Google Nano Banana 2 / Pro** (Gemini image models) | No, same reason | Possible, with conditions | No |
| **Chinese models:** Qwen-Image-Edit (Alibaba), Seedream (ByteDance) | No, same reason | Possible, with conditions. Qwen-Image-Edit-2511's weights are open (Apache-2.0), so it can run on our own rented GPU | No |
| **FLUX.1 Kontext** (Black Forest Labs) | No | Possible, with conditions | No |
| **Claude** (Anthropic) | **Can't produce images.** It can only look at them | Can only **check** results, for example "is there still a reflection?" | No. It can catch an obvious front/back mix-up, but can't measure angles |
| **Gemini chat (spatial mode)** | It can draw rough masks, but the edges are not good enough | No | Experimental 3D boxes; coarse |

**In short:** generative image models are worth testing **only for the glass**. The cut-out and the camera need
specialised models, which are also far cheaper.

## 3. Cut-out options and prices

| Option | What it gives | Price per frame | Notes |
|---|---|---|---|
| BiRefNet-lite on the laptop (today) | Good edges | $0 | About 17 s per frame on the CPU. Picks up whatever stands out, so other cars and posts can join the car |
| **SAM 3** on fal.ai, text prompt "car" | Finds **this car only**: separates neighbouring cars, posts and people | **$0.002–0.005** (fal's pages disagree). SAM 3.1 is $0.05 | Meta's SAM licence allows commercial use, with military/ITAR exclusions. Edges are coarser than BiRefNet's, so combine the two |
| BiRefNet v2 on fal.ai | Same edges as today, but fast on a GPU | About $0.001–0.003 (billed at $0.0008 per GPU second) | MIT licence |
| Pixelcut / Bria RMBG-3.0 on fal.ai | General product cut-out | $0.016 / $0.018 | Check that Bria's licence covers our use |
| **remove.bg API** | Has car-specific options: semi-transparent car windows and a car shadow | **$0.07–0.23** on subscriptions; $0.21–0.90 on one-off credit packs | Keep only its mask, never its colours |
| Photoroom API | Product cut-out; its mask-only output skips their edge clean-up | About $0.02–0.07 (not confirmed) | |
| Spyne, CarCutter, VinAudit | Trained on vehicle photos | Quote-based | No independent tests found. Spyne also changes the car itself |

**Best value:** SAM 3 to pick the car, plus BiRefNet for the edge, comes to under **$0.01 per frame**. remove.bg is
the premium option if its car-window handling wins a side-by-side test on our frames.

## 4. Glass reflections

### Why glass is harder than it looks
Car glass shows two things at once:
1. **Reflections** on its surface: sky, buildings, the photographer.
2. **The old background seen through it.** Through the side windows you see the car park behind the car.

If only the reflections are removed, the old car park still shows through the windows, which looks wrong in a studio.
Professional car-photo tools handle glass in one of three ways:

| Approach | Looks | Same across frames? | Cost | Risk |
|---|---|---|---|---|
| **a. Replace the glass with a clean dark tint** (a smooth gradient, keeping a faint trace of the original) | Clean, studio-like; the interior is hidden | **Yes.** It's a fixed rule, not a generated image | About $0: needs only a window mask (SAM 3 prompt "car window", about $0.005) | Hides the interior, and may hide a cracked windscreen |
| **b. Semi-transparent glass**, so the new studio shows through (what remove.bg offers for car windows) | Natural, but some old reflection remains | Yes | Included in the remove.bg price | Old reflections partly visible |
| **c. Generative clean-up** of the glass area only (clear glass showing the interior) | Best single-frame look | **No.** Each frame is drawn separately, so the glass changes slightly from frame to frame | $0.02–0.24 per frame (table below) | Can invent interior details or hide damage |

Dedicated reflection-removal research models exist, made for photos taken through windows: **WindowSeat** (Dec 2025),
**SIRR-LMM** (Jan 2026), and **FIRM** (code public, but it needs clicks on the image). None has been tried on car glass,
and I couldn't confirm released weights for the first two. Commercial tools that do car-window reflection removal
include Magic Eraser and Imagen AI (a Lightroom plug-in). They are per-seat apps, not APIs.

### Generative model prices (one edit = one frame)
| Model | Price per edit | Max output size | Notes |
|---|---|---|---|
| **OpenAI GPT Image 2** | Low $0.005–0.006 · medium **$0.04–0.05** · high **$0.17–0.21** at about 1–1.5 MP, plus the input image | About 8.3 MP (3840×2160); above about 2560 px is reported as experimental | Batch is half price. Newer 2.5/2.6 variants are reported at similar prices (not confirmed). GPT Image 1.5 retires on 1 Dec 2026 |
| **Gemini Nano Banana 2** (3.1 Flash Image) | $0.067 (1K) · $0.101 (2K) · $0.151 (4K) | 4K | Batch is about half. A "Lite" version costs $0.034 at 1K |
| **Gemini Nano Banana Pro** (3 Pro Image) | $0.134 (1K/2K) · $0.24 (4K) | 4K | Batch is about half |
| **Qwen-Image-Edit** (Alibaba) | $0.02–0.075 by API (Alibaba Cloud or fal) | | 2511 weights are Apache-2.0, so we can run it on our own GPU (§6) |
| **Seedream 5.0** (ByteDance) | Lite edit $0.032–0.035 · Pro $0.045 (up to 2.36 MP) / $0.09 above | | API only |
| **FLUX.1 Kontext** | dev $0.025 · pro $0.04 · max $0.08 | | The dev weights are non-commercial without a licence; the API is fine |
| **Claude** | Can't edit images. As a checker it costs about **$0.0003** (Haiku 5.5), **$0.005** (Sonnet 5.5) or **$0.01** (Opus 5.5) per photo | | A photo is about 1,600 input tokens (Anthropic counts width × height / 750, after resizing) |

### If a generative model is used for glass, these conditions must hold
1. **Send the glass mask, and paste back only the glass.** These models redraw the whole picture even with a mask, so
   the body must always come from the original photo.
2. **Size and alignment.** The output is at most about 4K and has to be scaled back to the photo. Expect small
   misalignment at the glass edges.
3. **Flicker.** Each frame is drawn on its own, so a 36-frame spin will show the glass changing slightly. Approach (a)
   doesn't have this problem.
4. **Evidence (needs a business decision).** The photos document the car's condition, and **glass is part of the
   car**: a chipped or cracked windscreen is damage.
   - The project's rule today forbids any generated content on the car.
   - Possible middle grounds: generative clean-up on side and rear windows only, keeping the windscreen as shot; or
     always storing the untouched original frame next to the studio version.

## 5. Camera angle options and prices
This summarises the earlier research.

| Option | What it measures | Price | Licence |
|---|---|---|---|
| **GeoCalib** | Focal length and camera tilt from one image; a backup where our line detection is weak | $0 (runs on the laptop CPU) | Apache-2.0 code, CC-BY-4.0 weights |
| **MoGe-3** (Microsoft, Aug 2026) | A 3D picture of the scene in metres, plus focal length. Fitting the floor gives the camera's real height and tilt | About **$0.03 per frame** on fal | MIT |
| **SAM 3D Objects** (Meta) | The car's rotation and position from one photo plus its mask: a starting guess for which way it faces | **$0.02 per frame** on fal | SAM licence (commercial OK) |
| **VGGT-1B-Commercial** (Meta) | **All frames of a car solved together**: the angle between shots is measured, and the room stays consistent | About $0.001 per GPU second on fal: a few cents per car (estimate) | Commercial checkpoint, gated |
| **Depth Anything 3** Base/Small (ByteDance) | The same multi-view solve; its authors report better camera accuracy than VGGT | Self-host on a GPU (§6) | Apache-2.0 for Base/Small only (Large/Giant are non-commercial) |
| **MapAnything** (Meta) | Multi-view, in metres; can use known camera settings | On Replicate | Use the Apache checkpoint |

Shiny paint and glass are a known weak spot for the multi-view models. Masking the car and letting the floor and walls
carry the solve helps.

## 6. Cost analysis

**Assumptions:**
- One car = **36 frames**.
- Glass handled once per frame. Tint or a generative clean-up doesn't depend on the background. If the glass has to
  show the *new* background through it, multiply the glass cost by the number of backgrounds.
- GPU rental, 2026: an L4 costs about $0.44–0.80/h, an A100 about $1.19–2.10/h, an H100 about $1.99–3.99/h. Serverless
  is billed per second.
- Self-hosted timings are **assumed, not measured**.

| Set-up | What's in it | Per frame | Per car (36 frames) |
|---|---|---|---|
| **A. Today** | BiRefNet-lite on the laptop; line camera; no glass handling | $0 | $0, but about 10 min of CPU per car for the cut-out alone |
| **B. Hosted open models, nothing generated** | SAM 3 car + window masks ($0.01) + BiRefNet on fal (~$0.002) + glass tint ($0) + MoGe-3 ($0.03) + VGGT per car (~$0.001/frame) | **~$0.04** | **~$1.50** |
| **C. B + low-cost generative glass** | B + Qwen-Image-Edit / Seedream Lite / FLUX Kontext pro / Nano Banana 2 at 1K ($0.03–0.07) | **~$0.07–0.11** | **~$2.50–4** |
| **D. Premium vendors** | remove.bg cut-out ($0.07–0.23) + Nano Banana Pro 2K ($0.134) or GPT Image 2 high (~$0.17–0.21) for glass + MoGe-3 + SAM 3D ($0.05) | **~$0.25–0.49** | **~$9–18** |
| **E. Self-hosted on one rented GPU** | Qwen-Image-Edit-2511 + SAM 3 + BiRefNet + Depth Anything 3 Base on an H100 (~$2–4/h), assuming ~10–20 s per frame for everything | **~$0.006–0.02** | **~$0.20–0.80**, plus setup work and paying for idle time |
| Add-on: Claude as a checker | One look per frame: reflection left? front/back right? | +$0.0003–0.01 | +$0.01–0.36 |

**What the numbers say:**
- The cut-out and the camera are cheap everywhere: cents per car.
- **The glass step decides the bill.** A generative model there costs 2–10× everything else combined.
- Self-hosting only pays off at volume: hundreds of cars a month, so the GPU isn't sitting idle.

## 7. Recommendation
1. **Cut-out:** SAM 3 (this car only) + BiRefNet (clean edges), hosted: under $0.01 per frame. Test remove.bg next to
   it on our hard frames (other cars, posts, glass).
2. **Glass:** start with **approach (a), the dark tint**. It's consistent across frames, costs nothing and generates
   nothing. Before allowing any generative glass clean-up, decide the evidence question in §4. If it's allowed, run a
   blind test of Qwen-Image-Edit, Seedream Lite and Nano Banana 2 on side and rear windows only, with paste-back.
3. **Camera:** geometry models, not chat models. Solve all frames together (VGGT or Depth Anything 3), use MoGe-3 for
   the floor and height, and keep GeoCalib as a free backup.
4. **Checks:** Claude Haiku or Sonnet, or Gemini Flash, as a cheap per-frame quality check.

## 8. Decisions and unknowns
- **Evidence rule for glass:** is any change to the glass allowed? If so, on which windows?
- **Sending photos to outside APIs:** plates are already covered by then, but check each vendor's data-retention terms.
- **Licences** to avoid for commercial use: FLUX.1 Kontext dev weights, Depth Anything 3 Large/Giant, the original
  VGGT and MapAnything checkpoints, RMBG-2.0.
- **Prices:** most come from trackers, and some disagree (SAM 3 on fal, Photoroom, GPT Image 2.x names). A 50-frame
  pilot per option would give real costs and quality.

## Sources
- **Generative image models and prices:**
  [OpenAI GPT Image 2 model page](https://developers.openai.com/api/docs/models/gpt-image-2),
  [GPT Image 2 cost per image](https://yingtu.ai/en/blog/gpt-image-2-cost-per-image),
  [OpenAI image pricing calculator](https://costgoat.com/pricing/openai-images),
  [OpenAI image pricing 2026](https://www.developersdigest.tech/models/image-generation/openai),
  [GPT Image 2 sizes (OpenAI cookbook)](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide),
  [GPT Image 2 on Azure](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-openais-gpt-image-2-in-microsoft-foundry/4500571),
  [mask behaviour reports](https://community.openai.com/t/understanding-how-gpt-image-models-on-edits-see-mask-and-transparency/1381752),
  [mask ignored report](https://community.openai.com/t/mask-is-completely-ignored-in-the-image-edit-api/1350867),
  [Nano Banana Pro on Google Cloud](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-pro-image),
  [Nano Banana pricing](https://benchlm.ai/media-pricing/nano-banana),
  [Nano Banana 2](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-flash-image),
  [AI image model pricing (Aug 2026)](https://invideo.io/blog/ai-image-model-pricing/),
  [image API prices (Oct 2026)](https://movcl.com/blog/comparisons/ai-image-api-prices-compared/),
  [Replicate edit model comparison](https://replicate.com/blog/compare-image-editing-models),
  [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).
- **Glass:** [Magic Eraser for dealerships](https://magiceraser.live/en/blog/ai-photo-editing-for-auto-dealerships),
  [Imagen AI car window tool](https://imagen-ai.com/solution/the-ai-car-window-reflection-removal-tool/),
  [WindowSeat](https://arxiv.org/pdf/2512.05000), [SIRR-LMM](https://arxiv.org/html/2601.07209),
  [FIRM](https://arxiv.org/html/2406.01555v1),
  [remove.bg API options](https://www.remove.bg/help/a/how-can-i-customize-my-api-results).
- **Cut-out:** [remove.bg pricing](https://remove.bg/pricing), [fal background removers](https://fal.ai/explore/background-remover-apis),
  [fal BiRefNet](https://fal.ai/models/fal-ai/birefnet), [fal SAM 3](https://fal.ai/models/fal-ai/sam-3/image),
  [SAM 3 licence summary](https://roboflow.com/model-licenses/segment-anything-3),
  [Photoroom mask output](https://docs.photoroom.com/remove-background-api-basic-plan/download-the-segmentation-mask),
  [Photoroom vs remove.bg cost](https://www.photoroom.com/blog/remove-bg-api-cost-leonardo-ai-vs-photoroom).
- **Camera:** [VRRPI-Bench](https://arxiv.org/abs/2601.22228v1), [VLM-to-3D paper](https://arxiv.org/pdf/2605.30561.pdf),
  [GeoCalib](https://github.com/cvg/GeoCalib), [MoGe-3](https://microsoft.github.io/MoGe-3/),
  [fal MoGe-3](https://fal.ai/models/microsoft/moge-3/api), [fal SAM 3D Objects](https://fal.ai/models/fal-ai/sam-3/3d-objects),
  [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial), [fal VGGT](https://fal.ai/models/fal-ai/vggt-1b/llms.txt),
  [Depth Anything 3](https://arxiv.org/abs/2511.10647v1), [MapAnything](https://arxiv.org/abs/2509.13414v3).
- **GPU rental:** [cloud GPU prices 2026](https://www.hostinger.com/tutorials/?p=158956), [RunPod pricing](https://www.runpod.io/gpu-cloud/pricing).
- **Claude prices:** Anthropic API price list (Claude Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10, Opus 5.5 $4/$20 per
  million input/output tokens).
