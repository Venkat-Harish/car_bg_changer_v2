# plate_brightness_detector

Picks how bright the overlay should be on a number plate, automatically, for **any plate colour**: white, yellow (taxis), green, blue, red, black.

It is used by `mask_plates.py` when `auto_brightness = true` in [`../config.toml`](../config.toml). With `false`, the fixed `lighting` setting is used instead. The module only needs an image and a plate's 4 corners, so it can be used on its own.

## The idea, in plain words

The camera can't see "how much light falls on the plate". It only sees **light × the paint's own colour**. A black wall in the sun still looks dark, so you can only judge the light from a wall if you know its true colour.

So the module looks for **something on the plate whose true colour it knows**, sees how bright it looks, and works out the light from that. The overlay's white is then made exactly that bright.

## The steps

1. **Split the plate into its two colours:** background and characters. The background is the one that forms a single connected area around everything; the characters are separate islands. (Which one covers more area is not reliable: bold letters can cover half the plate.)
2. **Read the light from the best reference available**, best first:
   1. **Something white**, either the background (most plates) or the letters (blue, green, red, black plates with white letters). How bright it looks *is* the light.
   2. **A light coloured paint**, when nothing is white (a yellow taxi plate, yellow letters). Its brightness is divided by how bright that paint is next to white: yellow 0.75 (`PAINT_FACTORS` in [`colours.py`](colours.py)).
   3. **The photo**, when the plate can't be read (dirty, glare, too small, only dark colours). This uses the brightest parts of the lower two-thirds of the image, meaning the car, not the sky. It is rougher.
3. **Sanity checks**, each falling back to the photo reading:
   - **Glow:** the plate is far brighter (more than 2×) than the photo reading, as happens at night when a reflective plate glows in the flash.
   - **Paint too thin to read:** after shaving 1 px off the edges of the light paint, less than 30 % of it is left (`MIN_CORE_SHARE`). That means thin letters on a small plate, where every pixel is blended with the background, so neither the paint's brightness nor its colour can be read. Readable plates keep 45–83 %; the 25 px-tall plates in `sample_vid` keep 8–16 %. Before this check, their readings swung from 65 % to 100 % between frames of the same car; from the photo they are a steady 67–78 %.
4. **Video (optional):** the main plate's reading is smoothed over the last few frames, so the overlay doesn't flicker.

The reading is a gentle gradient across the plate, not one number, so a plate that is half in shade keeps that shade.

## Results

Results on the colour test set ([`assets/test_plates`](../assets/test_plates/README.md)): sample plates pasted onto 3 views of the car, all lit like the real plate, so the true brightness is 75–76 % everywhere. Run it yourself with `tests\run_colour_tests.py` (see the main [README](../README.md#testing-plate-colours)).

| Plate | Reading from | Board brightness (auto) | With `auto_brightness = false` |
|---|---|---|---|
| Your real white plates (12 frames of the video) | white background | 73–75 % | 67–73 % |
| White, black letters | white background | 75–76 % | 75 % |
| White, red letters | white background | 75–76 % | 75 % |
| Yellow taxi, black letters | yellow background ÷ 0.75 | 76–77 % | 57 % (too dark) |
| Green, white letters | white letters | 75–76 % | 71–73 % |
| Green, yellow letters | yellow letters ÷ 0.75 | 76–77 % | 55 % (too dark) |
| Blue, white letters | white letters | 75–76 % | 71–72 % |
| Red, white letters | white letters | 75–76 % | 71–73 % |
| Black, white letters | white letters | 75–76 % | 69–72 % |
| Black, yellow letters (no border) | yellow letters ÷ 0.75 | 76 % | 53–56 % (too dark) |
| Real black/yellow plates in `sample_vid` (25 px tall) | photo (letters too thin to read) | 67–78 % | — |
| Green, black letters | photo (no white or light paint) | 78–85 %, a rougher estimate | 23 % |
| Night: plate glowing in a flash, photo dark | photo (plate too bright) | 19–22 % | 98–100 % |

With the old fixed method (`auto_brightness = false`), yellow plates come out about 20 points too dark, because yellow paint is read as white in shade.

Speed: about 2–8 ms per plate on a 4K frame, plus about 2 ms per image for the photo reading.

## Use from Python

```python
from plate_brightness_detector import BrightnessDetector

detector = BrightnessDetector()                 # BrightnessDetector(smoothing=5) for video
reading = detector.measure(image_bgr, corners)  # corners: 4 points, TL TR BR BL, image pixels
reading.percent      # 73.0 -> make the overlay's white 73 % bright
reading.source       # "white background", "yellow background (/0.75)", "photo (...)", ...
reading.background   # "white", "yellow", "blue", ...
reading.characters   # "black", "white", ...
reading.plane        # (a, b, c): brightness a + b*x + c*y across the plate

readings = detector.measure_all(image_bgr, [corners1, corners2])  # all plates in one image
```

## Tuning

- **`PAINT_FACTORS`** in [`colours.py`](colours.py) are first estimates. To check them for a plate type, photograph a white plate and that coloured plate side by side in the same light, and divide their brightness. Only factors of 0.5 or more are used; darker paints are too uncertain to divide by.
- **`NEUTRAL_SATURATION`** (60) decides what counts as white or grey rather than a colour. Real white plates measured 4–12; yellows are 200+.
- **`glow_ratio`** (2.0): how much brighter than the photo a plate may be before it's treated as glare or flash.

## Limits

- **Dirty white plates** look like white plates in shade, so the overlay is made a little too dark. No photo can tell the two apart.
- **Plates with only dark colours** (e.g. green with black letters) fall back to the photo reading, which is rougher.
- The yellow and orange factors and the colour boundaries **come from simulations**. They need checking on real photos of each plate type.
- **Smoothing** follows the largest plate in each frame. Keep it at 0 for unrelated photos.
