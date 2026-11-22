"""Optional step between the background-removal model and the car's mask: keep only the target car.

Turned on with `[car_isolation] enabled = true` in config.toml (off by default; about 10-20 s more per
frame on a laptop CPU). YOLOX-s finds every vehicle, SAM ViT-B masks the target and its neighbours,
and bg_changer.segment.clean_alpha then drops the neighbours (and other things stuck to the car's
outline) and fills only window holes. Only the mask changes, never the car's pixels.

    from bg_changer import CarSegmenter
    from car_isolation import CarIsolator

    car = CarSegmenter().cut_out(image_bgr, isolator=CarIsolator())
"""

from .detect import Detection, VehicleDetector
from .isolate import DEFAULT_MIN_SCORE, CarIsolator, IsolationResult, overlay, plan_removal

__all__ = ["DEFAULT_MIN_SCORE", "CarIsolator", "Detection", "IsolationResult", "VehicleDetector", "overlay",
           "plan_removal"]
