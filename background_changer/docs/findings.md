# Findings: putting the car on a studio background

## The job

Input: frames from the walk-around video (3840×2160 JPG, plates already masked by `number_plate_detection`). The car stands in a dealer car park with other cars, buildings and sky around it.

Output: the same car, centred and sized to fit, on one of the studio/track backgrounds in `backgrounds\` (`background_1/2/3.webp`, flat 1920×1080 pictures), as WEBP.

**Hard rule: the car's exterior must not change.** Dents, scratches, smudges and dirt are the point of these photos, so they must survive. That rules out anything that *redraws* car pixels:

| Not done | Why |
|---|---|
| Generative fill / inpainting / "AI enhance" | Invents pixels. It can smooth away a dent or add a reflection that was never there. |
| Relighting / harmonisation models (IC-Light and similar) | Re-renders the car's shading to match the new scene. That changes how damage looks. |
| Colour or exposure matching to the background | Changes the car's colour, and so how scratches and paint defects read. |
| Sharpening, denoising, upscaling | Changes texture and can hide or exaggerate fine scratches. |

What *is* done to car pixels:
- **flat mode: one uniform downscale with `INTER_AREA`** (plain area averaging). Where the car is fully opaque (everything except a thin outer edge), the output pixel is exactly that downscaled pixel. `tests/test_compose.py::test_car_pixels_are_only_downscaled_on_every_scene` checks this bit for bit.
- **room mode: at most one rigid rotation, then that same downscale**, and cropping at a cut edge. The rotation levels a photo taken with a rolled phone. It is skipped below 0.5°, so many cars are pasted exactly as in flat mode. Where it happens: premultiplied Lanczos-4 colour, clamped to the 3×3 source pixels' range (no new colours, no dark fringes), bilinear alpha. `tests/test_car_fit.py` and `tests/test_room_mode.py` check every opaque output pixel bit for bit.

## Cutting the car out

Background-removal models give a soft alpha (0 = background, 1 = object) for the most salient object. All of the candidates below run on the CPU through ONNX Runtime via [rembg](https://github.com/danielgatis/rembg) (MIT), so no torch or GPU is needed.

| Model | Licence | Edges | CPU speed (i5-1240P) | Notes |
|---|---|---|---|---|
| **BiRefNet general-lite** (default) | MIT | Clean wheel arches, mirrors, bonnet line | ~17 s/image warm at 4 threads; 220 MB | Swin-Tiny backbone |
| BiRefNet general | MIT | Same as lite at 1920×1080 output (compared side by side, zoomed) | 66-122 s/image; 970 MB | Swin-Large backbone. Use it if the output size is raised a lot. |
| ISNet general-use | Apache-2.0 | Softer, can nibble thin parts | Fast; ~170 MB | |
| U²-Net | Apache-2.0 | Blobby | Fast | Older, superseded by the above |
| BRIA RMBG-2.0 | **CC BY-NC 4.0** | Very good | Medium | **Rejected: non-commercial licence** |
| YOLO-seg (Ultralytics) | **AGPL-3.0** | Coarse (low-res mask head) | Fast | Rejected: licence and edge quality |
| SAM / SAM 2 | Apache-2.0 | Coarse at 4K | Slow encoder | Needs a point/box prompt; not needed here |

### Speed: fewer threads is faster on this laptop

The i5-1240P has 4 performance cores and 8 efficiency cores. ONNX Runtime by default spreads the model over all of them, and the slow cores hold the fast ones back. Measured on one 4K frame, after a warm-up run:

| Model | 4 threads | 8 threads | ONNX default |
|---|---|---|---|
| birefnet-general-lite | **~17 s** | ~20 s | ~35 s |
| birefnet-general | ~66 s | ~100-120 s | ~81 s |

So the defaults are `model = "birefnet-general-lite"` and `threads = 4`. Laptop timings vary a lot with heat and power mode; the first image after loading is always slower (warm-up). On a desktop CPU with only performance cores, try `threads = 0`.

### From model alpha to "this car only"

The raw alpha has three problems on these photos. `bg_changer/segment.py` (`clean_alpha`) fixes each one:

1. **Other cars.** The car park has parked cars that the model may also pick up (frame 0024 has one on each side).
   - Threshold the alpha at 0.5 and split it into connected blobs.
   - Keep the blob with the best score of *area × closeness to the centre*. The walk-around keeps the main car big and central.
   - Small detached pieces lying wholly inside that blob's box are kept too.
   - Soft alpha is kept only within a few pixels of the kept outline.
   - **Posts behind the car.** A lamp post or sign pole standing behind the car touches its outline in the photo and joins the same blob (frame 0012: a post sticking out of the roof). These are removed with a *horizontal* opening: everything narrower than 1.2 % of the image width (≈46 px at 4K) *side to side* is taken off. Any removed piece that reaches further than that out of the car is dropped whole, base included. Shorter pieces are the car's own edge details (rounded tyre bottoms, corners) and go back in.
     - The rule is deliberately one-directional. A plain (round) opening also caught posts, but it would cut off a wing mirror whose stalk looks thin, for example on a car further from the camera. Mirrors, wipers and roof rails stick out sideways, so the horizontal rule never touches them.
     - Window holes are filled *before* this step, so thin pillars between see-through windows are never mistaken for posts.
2. **Windows.** Models often make glass partly transparent. On a new background that would show the studio *through* the car, which is wrong: the photo shows the car's real glass, reflections and interior. Holes inside the outline are filled, so glass keeps its original pixels.
3. **Soft interior.** Inside the outline, except for a thin edge band (0.3 % of the image's longer side, ≈12 px at 4K, ≈5 px after downscaling), alpha is forced to 1. That is the "fidelity core": no blending with the background happens there. `edge_shrink` (default 2 px) pulls the soft edge in a little to drop a fringe of the old background, such as blue sky on a white roof.

### Keeping only the target car (`car_isolation`, optional)

`clean_alpha` alone cannot tell which vehicle is the target: BiRefNet returns "the salient object(s)", so a car parked behind ours that touches its outline in the photo, a person, or the cameraman's hand joins the main blob. `fill_holes` also fills every enclosed gap, not only windows. `car_isolation/` (switch: `[car_isolation] enabled`) adds an instance step between the raw alpha and `clean_alpha`:

| | Model | Licence | Time on this laptop |
|---|---|---|---|
| Which objects are where | YOLOX-s, official ONNX (COCO, 80 classes) | Apache-2.0 | ~0.2-0.3 s at 640 px |
| The target car's and other vehicles' masks | SAM ViT-B, the ONNX pair rembg publishes; encoder once per photo (on a crop around the car), decoder per box | Apache-2.0 | ~9-13 s per photo |

- **Target:** the vehicle box that best matches BiRefNet's main blob (share of the blob in the box × how full the box is). If none matches, or SAM's car disagrees with the blob (less than 60% of SAM's mask on the blob, or covering under 40% of it), nothing changes (`no-vehicle`, `unsure`).
- **Other vehicles:** their SAM masks, grown by 0.3% of the long side, are cleared from the alpha; never inside SAM's target car (shrunk by the same band).
- **Other objects:** a piece sticking out of SAM's car by more than 1.2% of the long side is removed only with evidence: half of it lies in another detected object's box, or it reaches a photo edge where the car is not cut off. The first version removed pieces on shape alone and cut off part of the Baleno's spoiler in `sample_vid` 0022 (SAM's car had left the spoiler out). With the evidence rule it stays.
- **Parts of our car detected as vehicles:** YOLOX labels the EQB's door mirrors "car" in front views. The first EQB run removed both mirrors as other vehicles (`Final_zoomed_car` 0002, 0036). A detection that may be a part of our car is now never removed and never counts as evidence. That is a box at least 80% inside the target car's own box, or a small box (under 3% of the car box's area) at least 30% inside it, such as the far-side mirror in the ¾ view 0036, which sticks out sideways. A car really parked behind ours is bigger and sticks out of our car's box: in `sample_vid` 0000 the two are 5.4% and 6.1% of the car box, about 60% inside. When unsure, the cut-out stays as it was.
- **Holes:** with the step on, only holes lying mostly (60%) on SAM's car are filled.
- **Memory:** BiRefNet and SAM loaded together made BiRefNet ~3× slower on the 16 GB laptop (paging). The run therefore cuts out every photo first (rembg's mask is 8-bit, cached losslessly), frees BiRefNet, then loads YOLOX and SAM.

`sample_vid` with the step on:
- 0000: the car parked behind the roof is gone.
- 0008 and 0010: the cameraman's hand is gone. Before, it made the car look cut off at the bottom, so these frames were not placed at all; now they are.
- 0022: the spoiler is intact.
- 0014 and 0024: mirrors and antenna are kept.
- 3D fits: within ±0.01 IoU of before, slightly better on 0000.

## Placing the car: one scene per background

### Why "fit to a box" looked wrong

The first version scaled each frame to fit a `max_width × max_height` box and stood its lowest pixel on one row. On background_1 that made 18 of 20 frames ≈ 648 px tall with their tyres on row 907. Two things were wrong with that.

1. **Row 907 is 2.3 m in front of the turntable's centre.** The disc's true centre is at row 764, where its 8 seams meet. The rim ellipse's centre (row 834) is 1.4 m nearer the camera, because the camera is close. At row 907 the floor's scale is 403 px/m instead of 261, so every car looked too big and too low.
2. **A real car turning in front of one camera changes height with the view.** Its near end swings toward the camera. Spyne's spin of the same video shows side ≈ 520-560 px, front ≈ 676 and rear ≈ 776, of which roughly 594 and 661 is the car itself and the rest shadow (`docs/spyne-360-spin-findings.md`). Forcing every view to one height flattens that, and the car looks pasted on.

### Calibrating the backgrounds (from the pictures alone)

A read-only measurement pass fitted a camera and floor model to each picture. No camera details are needed.

| | background_1 (turntable) | background_2 (grey studio) | background_3 (race track) |
|---|---|---|---|
| How | Rim ellipse (0.15 px rms) + side-wall LED-strip vanishing point + 8 disc seams | Square floor-tile repeat + plinth lines + panel seams | Nine 0.80 m jersey barriers (h_cam = 0.80·(y_b − y_h)/(y_b − y_t)) + truck check |
| Camera | level, f 1702.5 px, horizon 502.5, roll 0 | f 1213 px, horizon 546.8, pitch +1.6° | f ≈ 1400 px, horizon 397, camera 1.24 m |
| Stage centre | (958.8, 764.0) | (960, 780) | (960, 685) |
| Floor | glossy (mirrored LED streaks), soft light | matte, soft overhead light | matte asphalt, hazy daylight, warm-neutral contact shadows |

### Two cameras that meet at the stage centre

The car's pixels may only be scaled and moved, so the car always keeps the walkaround's own viewpoint: close (≈ 3.4-4.6 m) and looking down ≈ 16°.

- **Car geometry:** an EQB-shaped 3D outline, projected through a fixed **capture camera** (4.70 m from the car's footprint centre, 16.08° down). It reproduces Spyne's frames to ≈ 6 px RMS.
- **Why not the plates' own cameras:** they are farther and flatter (6.5 m, 8.7° on background_1), and would make the photos' height and width disagree by up to 23% (1.109 ± 0.091 vs 1.053 ± 0.038 through the capture camera, over 15 uncut frames).
- **Floor in front of the car:** follows each plate's own camera ("pinned floor map"), so shadows stay compact.

### Per frame

- **Angle:** from the view label (bin centre). Measured against a perspective fit, labels are off by up to ~24° (e.g. `0014 back-right` is really ≈ 111°, `0032 left 47%` ≈ 297°). The user accepted this.
- **Size:** `scale = min(1, px_per_m · h_rel(yaw) / H_photo)`.
  - Height is used, not width, because width is cut off in 5 of the 20 frames and height scatters less (2.4% vs 2.9%).
  - `px_per_m` comes from each scene's stage size: a side view spans the same pixels whatever the car (≈ 73% of background_1's disc). Every car fills the stage the same way; the dimensions set the proportions.
- **Vertical:** the lowest pixel lands where the 3D outline's lowest point is. Front and rear views come lower and look taller (on background_1: side ≈ 500, front ≈ 580, rear ≈ 640 px).
- **Horizontal:** the outline's box offset, averaged over the label's uncertainty, so a wrong label near an end view does not shove the car sideways. Cut frames put their uncut end where the whole car would end.

### Grounding

Drawn on the background in linear light, before the car is pasted:

- **Tracking the pasted car.** The 3D outline is drawn at the label's angle, which can be off. So floor effects are mapped onto the pasted car:
  - its two ends, plus its lowest point, are matched;
  - tyre contacts are matched to the real tyres (the local lowest points of the car's bottom outline) in side and ¾ views.
- **Contact and gap layers** come from the car's own bottom outline. The floor line is its **lower envelope**: the hull under the outline, through the points where it touches (the tyres), kept level past the outermost ones. Mirrors and other parts more than 1.3 m above the lowest point are left out.
  - A dark line marks where the outline touches that line.
  - The floor seen under the body, between the underside and the line, is darkened, but only up to 0.45 m of underbody height.

  These follow the real car whatever its angle.
- **A footprint shadow** (core plus soft skirt) is projected from the car's real footprint, shifted onto the matched tyres.
- **On background_1's glossy floor**, a faint blurred mirror image is added, column by column about each column's own contact line.

**Problems found and fixed while tuning:**
- Contact patches at the predicted (not real) tyres floated next to the car.
- A floor line extended diagonally past the near tyre drew a dark wedge to the frame edge.
- Calibrated "plate" shadow strengths (taken from wall-base shading) were too faint to ground a car.

**Found by an independent review and fixed** (each with a regression test):
- A floor line built from the matched tyres went flat when only one tyre matched, so the other tyre floated over a dark slab (5 of 20 frames). It is now the outline's lower envelope.
- In ¾ views, apron and bumper corners were mistaken for tyres. Only near-side tyres touching the floor line, away from the bumper ends, are matched now, and a lone match's correction fades with distance.
- Mirrors and receding flanks got underbody shading; it is now height-capped.
- On cut frames, the short edge-to-tyre slope was stretched across the whole car. Beyond the tyre the pasted scale is used now.
- Tall vans could be pushed past the turntable's front edge. Every scene now checks that all views fit, and shrinks with a warning.
- Smaller fixes:
  - anchor overrides re-pin the plate floor;
  - bottom-cut photos are placed by the roof;
  - reflection seams at the car's ends are gone;
  - `cars.json` unit and encoding errors get clear messages.

## Low-resolution input: `sample_vid` and the background scaler

**The car.** It is a 2022+ Maruti Suzuki Baleno (3rd generation). The evidence:
- the BALENO badge on the tailgate and the Suzuki logo;
- the 2022 grille, headlamps and tail lamps;
- steel wheels with covers (a lower trim);
- a yellow KA 03 (Bengaluru) plate and a revv.co.in rental sticker.

Indian-market dimensions are 3990 × 1745 × 1500 mm, wheelbase 2520 mm (Autocar India, Rushlane, CarDekho). The Australian-market car differs: 3995 mm long, 1470 mm tall. It is in `cars.json` as `Maruti Suzuki Baleno 2022`.

**The problem.** The video is 464×832 portrait (59.97 fps), and frame_view_type took 14 frames from it. The car is 210-474 px tall in them, while a full-size background_1 stage wants 582-652 px for those views. The car is never enlarged, so at full size every frame was `scale-capped`. The car stood at 33-73% of its stage size, a different share in each frame.

**The fix: shrink the background, not enlarge the car** (`background_scaler/`).
- **Every pixel measurement shrinks with the picture.** At k of full size the scene scales:
  - the stage centre and the rim centre (pixel centres, like `cv2.resize`), and `side_px`;
  - the floor camera: M·H for a homography; horizon and k for a level floor;
  - the floor mask (drawn at full size, then INTER_AREA);
  - the lowest-row limit and the reflection blur.

  Shadow sizes are in metres, so they follow px/m by themselves. The few fixed pixel sizes in the code (mask feather, guard shift, debug marks) follow `SceneSpec.px_scale`.
- **Check.** A car placed on background_1 or background_3 shrunk to 960×540 or 653×367 matches the full-size result downscaled. The mean difference is 0.05-0.15 grey levels, where the shadows are up to ~130 levels deep.
- **Per folder (default).** k is the smallest need over the folder's photos. Here that is 0026, the far rear shot: 34% on background_1 (653×367), 708×398 on background_2 and 715×402 on background_3.
- **Per photo (`background_scale = "photo"`).** k is 34-73% (653×367 to ~1400×790 on background_1), and every car is pasted pixel for pixel.

**What placement cannot fix in this video:**
- **Cut-off cars.** 10 of the 14 frames run off the photo edge, because the phone was held upright and close. They show a straight cut. 0008 and 0010 are cut on two sides and are not placed.
- **Tilted frames.** Several frames are rolled (0004, 0018, 0024: the phone was tilted), so the car leans on the stage. Straightening it would mean rotating, so resampling, the car's pixels, which the "only scale and move" rule does not allow.
- **A merged neighbour.** In 0000 a car parked down the street touches the Baleno's roof outline in the photo and is kept with it.
- **Narrow front and rear views.** These frames are flagged `hw-mismatch` (1.20-1.27): the car is ~20% narrower for its height than the 3D outline predicts. Most likely the phone looked down on this lower car more steeply than the capture camera's 16°, which shows more bonnet and roof. Since size comes from height, these views come out slightly narrow.

Room mode (next section) was built for exactly these frames: it measures each frame's own camera (21-47° down, rolled up to 7°, 2-5 m away) and re-renders the background from it.

## Room mode: fitting each photo and re-rendering the background

**Why the flat placement floats.** It only scales and moves the car through one fixed camera fitted on the Mercedes walkaround (4.7 m away, 16° down). `sample_vid` breaks every part of that: 464×832 portrait, an ultra-wide lens (f ≈ 366 px), 2-5 m away, 21-47° down, rolled up to 7°, 10 of 14 frames cut off, view labels off by up to 177°. A car seen from 20-47° above, pasted on a floor seen from about 9°, floats whatever the scale and position. So room mode changes the floor's viewpoint, not the car's.

**Per photo** (`car_fit_to_bg/`):
1. **Camera from the background lines** (`lines.py`). LSD line segments with the car masked out, at most 1280 px; the vertical vanishing point by MSAC; the focal length by a Manhattan search; a robust refit; a bootstrap for confidence. The focal comes out within ±3-7% where the lines constrain it, which is about 3 frames in 4. So it is pooled: photos of the same size with confident lines (at least 3, within ±8%) share their weighted median, and a photo whose own confident focal is more than 20% off keeps it (`lens-change`). Otherwise: the photo's own lines, EXIF, uncertain lines, or a 70° field of view (flagged). Roll and pitch are re-solved at the chosen focal. Video frames with too few lines borrow roll and pitch from confident neighbours within ±8 s.
2. **The car's pose** (`pose.py`). A convex-piece proxy built from the car's dimensions and body preset (suv, hatch, sedan), rendered with near-plane clipping. A sweep over 36 view angles, then up to 5 candidates refined (angle, roll, distance, height, aim) with priors from the lines. Scored on outline overlap, minus the classifier's (or label's) log-prior, plus red-tail-lamp penalties (red at the end facing the camera decides front vs back). The outline alone is nearly symmetric front-to-back, which is why those hints matter: in the planning tests, 0 front/back flips on 32 placeable frames, and yaw within 2.9° RMS (Baleno) / 1.4-1.7° (EQB) of a video-pooled fit. The worst, a half-car close-up (Baleno 0006), was about 30° off and was caught by the margin test (`yaw-unsure`: under 2 nats to the flipped pose).
3. **Level** (`level.py`). One rotation by the camera's roll to the car's floor. A car on a cross-slope (Baleno 0014 stands on about 7.5°) is refitted with that roll, so the room's floor becomes the car's floor.
4. **Tyre bottoms and contact lock** (`ground.py`). Only near-side tyres (and both facing tyres in end views) are searched; a bottom is accepted only where the outline has a real tyre bump below the predicted contact, never on a cut edge. (Searching far tyres snapped them to bumpers and aprons.) With 2 contacts the camera's height and pitch are adjusted (regularised, ±15% / ±3°), with 1 only its height.

**Per background** (`room3d/`, `car_fit_to_bg/place.py`). Each background picture is a 3D room seen by its own camera, built from the scene modules' measurements: background_1 re-renders its own picture exactly from its own camera, background_2 within 1 grey level (background_3's picture is de-vignetted first, so it is not compared). The output camera is the photo's levelled, locked camera, placed so the car's footprint lands on the scene's stage at the stage-relative size flat mode uses, on the plate camera's side. Every output pixel is ray-cast to the floor, a wall or the ceiling and sampled from the picture through a Gaussian pyramid (trilinear by footprint, bicubic), so a floor seen from above is neither aliased nor blurred. What the picture never showed is filled per surface: the floor with its blurred lighting plus a tiled clean patch, walls mirrored within a band, the ceiling and sky as a smooth low-pass extension (an earlier column-clamp fill gave repeated light bars and sky "beams"). background_3's picture is de-vignetted first (brightness ≈ 1 − 0.18 (d/960)², measured on the asphalt).

**Output size.** The car is never enlarged, and the background shouldn't be either: `s = min(1, max_output_size / long side, texels at the stage)`. At their own 3840×2160 the EQB photos' backgrounds would sit at 0.24-0.38 picture pixels per output pixel at the car (visibly soft); kept at ≥ 1, the outputs come out 907-1442 px wide, with the car about as large as in flat mode. `keep_background_sharp = false` gives 1920 px wide outputs at 0.47-0.75. The rule also shrinks a close-up from a small photo: `sample_vid` 0006 comes out at 346-365 px wide (by background) instead of 464, 0020 at 428-452.

**Invented areas.** The plate pictures are landscape and look nearly level; a portrait frame looking 30-47° down sees beyond them at the top (ceiling, sky) and the bottom (the floor under the phone). In `sample_vid` up to about half of a full frame is fill, mostly at the top and bottom; on four close-ups (0006, 0020, 0022, 0024) 5-15% of the area around the car is fill too (flagged `invented`). In the EQB frames 0-6%, at most 1.7% around the car.

## Known limits

- **The car keeps its original lighting and reflections**, by design (see the rule above). A car shot in sunshine, with sky and buildings reflected in the paint, will look sunlit on a soft-lit studio background.
- **A neighbouring car that touches the main car in the photo** joins the same blob and is kept. Check `cutouts\` for this.
- **Posts wider than ~1.2 % of the image width** (a thick pillar, a tree trunk) that touch the car are not removed. **A thin car part standing straight up** (an old-style whip antenna) would be removed like a post. Modern shark-fin antennas are wide enough to stay.
- **Cut-off cars stay cut off.** Nothing is invented for the missing part.
- **Small photos give small results.** With `background_scale` the output is only as large as the photos' car allows (653×367 for `sample_vid`). Filming the car whole, in landscape, at 1080p or more gives full-size results.
- **Flat mode: viewpoint height and perspective strength cannot be changed in 2D.** The car keeps the photo's close, 16°-down viewpoint; the plates look down 9-12°. The car's eye level sits about 53 px (background_1) and 57 px (background_2) above the plate horizon; background_3 matches. Room mode re-renders the background from the photo's viewpoint instead.
- **Flat mode: the angle comes from 8 labels**, so it can be off by up to ~24°. Room mode fits it.
- **Room mode: the plate pictures are flat photos.** Seen from far from where they were taken, their baked-in light streaks (background_1's LED reflections on the floor) and far scenery (background_3's trees, power line and lamp post) stretch or repeat. background_3 can show a sky seam on steep left-looking frames.
- **Room mode: end views ground more softly.** A car seen straight from the front or back often hides its tyres behind its bumpers (both facing tyres are searched); without a found tyre its contact comes from the outline only.
