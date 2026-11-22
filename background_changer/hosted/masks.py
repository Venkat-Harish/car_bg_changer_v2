"""Hosted cut-out: BiRefNet on fal for the edge, SAM 3 (text prompt "car") to keep only this car.

Contract (stub; Agent A implements):
- birefnet_alpha(client, image_bgr, max_side=2048) -> float32 HxW in 0..1 at the photo's size (INTER_LINEAR up).
- sam3_instances(client, image_bgr, prompt, max_side=2048) -> list[Instance] (bool HxW masks at the photo's size).
- HostedIsolator(client).isolate(image_bgr, raw) -> car_isolation.IsolationResult, same contract as
  car_isolation/isolate.py CarIsolator.isolate, so bg_changer.segment.CarSegmenter.finish(image, raw, isolator)
  works unchanged. Raises nothing for "no car found": returns status "no-vehicle" / "unsure" with remove=None.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fal import FalClient

SAM3_ENDPOINT = "fal-ai/sam-3/image"
BIREFNET_ENDPOINT = "fal-ai/birefnet/v2"


@dataclass
class Instance:
    mask: np.ndarray  # bool HxW at the photo's size
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 px at the photo's size
    label: str = ""


def birefnet_alpha(client: FalClient, image: np.ndarray, max_side: int = 2048) -> np.ndarray:
    raise NotImplementedError


def sam3_instances(client: FalClient, image: np.ndarray, prompt: str, max_side: int = 2048) -> list[Instance]:
    raise NotImplementedError


class HostedIsolator:
    def __init__(self, client: FalClient, prompt: str = "car", other_prompts: tuple[str, ...] = ("person",)):
        self.client = client
        self.prompt = prompt
        self.other_prompts = other_prompts

    def isolate(self, image: np.ndarray, raw: np.ndarray):
        raise NotImplementedError


__all__ = ["BIREFNET_ENDPOINT", "SAM3_ENDPOINT", "HostedIsolator", "Instance", "birefnet_alpha", "sam3_instances"]
