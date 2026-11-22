"""Pick how bright an overlay should be on a number plate, for any plate colour.

    from plate_brightness_detector import BrightnessDetector

    detector = BrightnessDetector()
    reading = detector.measure(image_bgr, plate_corners)   # 4 corners, TL TR BR BL
    reading.percent    # e.g. 73.0: make the overlay's white 73 % bright
    reading.source     # e.g. "white background", "yellow background (/0.75)", "photo (...)"

Independent of plate_masker: it only needs an image and the plate's 4 corners. See README.md.
"""

from .colours import MIN_FACTOR, PAINT_FACTORS, name_colour
from .detector import BrightnessDetector, BrightnessReading

__all__ = ["BrightnessDetector", "BrightnessReading", "MIN_FACTOR", "PAINT_FACTORS", "name_colour"]
