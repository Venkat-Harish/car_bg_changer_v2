# Sample plates for the colour tests

One flat plate image per plate colour (1040×224 px, plate proportions, generic text `TEST 123`). [`tests/make_test_photos.py`](../../tests/make_test_photos.py) pastes each one onto 3 views of the car, at the real plate's spot and in its lighting. That makes `assets/test_photos/<name>/` plus `expected.csv`, so the right answer is known for every test photo.

| File | Plate | Stands for | Light is read from |
|---|---|---|---|
| `white_black.png` | white, black letters | most of Europe, India private cars | white background |
| `white_red.png` | white, red letters | Belgium | white background |
| `yellow_black.png` | yellow, black letters | taxis and commercial, UK rear, Netherlands | yellow background ÷ 0.75 |
| `green_white.png` | green, white letters | electric cars (e.g. India) | white letters |
| `green_yellow.png` | green, yellow letters | electric taxis or commercial (e.g. India) | yellow letters ÷ 0.75 |
| `green_black.png` | green, black letters | tests the fallback: no white or light paint | the photo (rougher) |
| `blue_white.png` | blue, white letters | e.g. China | white letters |
| `red_white.png` | red, white letters | e.g. temporary or diplomatic plates | white letters |
| `black_white.png` | black, white letters | e.g. rental or vintage cars | white letters |
| `black_yellow.png` | black, yellow letters, no border line, long condensed text | Indian self-drive rental plates (like `sample_vid`); its edge can't be seen on a black holder, so the outline comes from the letters | yellow letters ÷ 0.75 |

Each plate (except `black_yellow`) has a border line in the letter colour (about 5 % of the plate's height), with a thin rim of background colour outside it, like the real plate in the test video.

## Adding your own plate

Drop any plate image (PNG or JPG, about 4.7:1, no transparency) into this folder and run `tests\make_test_photos.py` again. It becomes a test case named after the file. Its expected reading is worked out from its two main colours: "white" if one is white or grey, "light paint" if one is yellow or orange, otherwise "photo".

The standard images are drawn only if missing, so your edits to them are kept. To draw them again from scratch:

```powershell
.venv\Scripts\python tests\make_test_photos.py --redraw
```
