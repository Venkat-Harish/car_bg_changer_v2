"""background_scaler: shrink a background, with its scene measurements, to the resolution of the car photos.

The car's pixels may only be scaled down, never up. A frame from a low-resolution video (a phone
walkaround at 464x832, say) holds a car only 200-480 px tall, while a 1920x1080 background's stage
wants it 580-650 px tall. So instead of enlarging the car, the background is shrunk until the car
fills the stage at its own resolution. That means the picture (INTER_AREA) and every pixel measurement
of its scene module: stage centre, stage size, floor camera, floor mask, turntable rim, lowest-row
limit and blur sizes. The car then stands on the smaller stage exactly as it would on the full-size
one, with fewer pixels around it.

How small, per run (config.toml `background_scale`):
  "folder"  one size for all photos of the folder, set by the photo with the smallest car (default);
  "photo"   each photo gets the size its own car needs (the car is pasted pixel for pixel);
  "off"     always the full-size background.
Photos that already hold enough pixels (e.g. 4K frames) keep the full-size background in every mode.

    from background_scaler import BackgroundScaler
    scaler = BackgroundScaler("photo")
    k = scaler.photo_scale(scene, probe, yaw)                  # or scaler.folder_scale(needs)
    small_scene, small_plate = scaler.apply(scene, plate, k)
    out, plan = place_car(image, car, small_plate, small_scene, probe, yaw)

Files: sizing.py (how small), rescale.py (the smaller scene), scaler.py (per-run choice and cache).
"""

from .rescale import scale_scene, scale_spec
from .scaler import BackgroundScaler, crop_to_car
from .sizing import DEFAULT_MIN_WIDTH, DEFAULT_MODE, MODES, background_size, folder_scale, needed_scale

__all__ = [
    "DEFAULT_MIN_WIDTH", "DEFAULT_MODE", "MODES", "BackgroundScaler", "background_size", "crop_to_car",
    "folder_scale", "needed_scale", "scale_scene", "scale_spec",
]
