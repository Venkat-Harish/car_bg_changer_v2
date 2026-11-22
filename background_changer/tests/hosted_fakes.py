"""Shared test double for hosted.fal.FalClient: no network, no key.

    fake = FakeFal({"fal-ai/sam-3/image": lambda payload, images: {...}})
    fake.files["mem://mask0.png"] = mask_uint8      # what fetch_image / fetch_bytes return for that URL

Handlers get (payload, images) where images is the dict passed to run() (ndarrays, untouched) and return the JSON
result the real endpoint would return. Every call is recorded in fake.calls.
"""

from __future__ import annotations

import cv2
import numpy as np

from hosted.fal import FalError, Spend


class FakeFal:
    def __init__(self, handlers: dict | None = None):
        self.handlers = dict(handlers or {})
        self.files: dict[str, np.ndarray | bytes] = {}
        self.calls: list[tuple[str, dict, dict]] = []
        self.spend = Spend()

    def run(self, endpoint, payload=None, images=None, cost=None, image_format=".jpg", max_side=0):
        self.calls.append((endpoint, dict(payload or {}), dict(images or {})))
        if endpoint not in self.handlers:
            raise FalError(f"{endpoint}: no fake handler")
        self.spend.calls += 1
        return self.handlers[endpoint](dict(payload or {}), dict(images or {}))

    def add_file(self, url: str, data) -> str:
        self.files[url] = data
        return url

    def fetch_bytes(self, url):
        data = self.files[url]
        if isinstance(data, np.ndarray):
            ok, buf = cv2.imencode(".png", data)
            return buf.tobytes()
        return data

    def fetch_image(self, url, flags=cv2.IMREAD_UNCHANGED):
        data = self.files[url]
        if isinstance(data, np.ndarray):
            return data.copy()
        return cv2.imdecode(np.frombuffer(data, np.uint8), flags)
