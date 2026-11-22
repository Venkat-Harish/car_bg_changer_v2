"""Idea B: hosted open models on fal.ai, nothing generated (docs/glass-cutout-camera-options.md, set-up B).

- fal.py    : FalClient (queue REST API, disk cache, cost guard), load_key
- masks.py  : SAM 3 "car" instances + BiRefNet on fal -> raw alpha and a HostedIsolator for CarSegmenter.finish
- glass.py  : SAM 3 "car window" mask + deterministic dark glass tint (the only step that changes car pixels)
- camera.py : MoGe-3 per-frame focal / floor (height, pitch, roll) + VGGT per-car joint solve -> HostedCamera

Only imported when `[hosted] enabled = true`.
"""
