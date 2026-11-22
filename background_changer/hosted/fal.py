"""Tiny fal.ai client over the queue REST API (no fal-client dependency), with a disk cache and a cost guard.

    client = FalClient(load_key(), cache_dir=Path("output/_hosted_cache"), max_cost=3.0)
    out = client.run("fal-ai/birefnet/v2", {"model": "General Use (Light)"}, images={"image_url": image_bgr})
    alpha = client.fetch_image(out["image"]["url"])

Images are passed separately from the JSON payload: the cache key is sha1(endpoint + payload + image bytes), so an
image is only uploaded on a cache miss and a rerun of the same folder costs $0. Every result file a call returns
should be read through `fetch_bytes` / `fetch_image`, which are cached as well (fal's result URLs expire).

The key comes from the FAL_KEY environment variable or a `.env` file (`FAL_KEY=...`). It is never printed, logged or
written anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

QUEUE_URL = "https://queue.fal.run"
UPLOAD_INITIATE_URL = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"

# List-price estimates per call (USD), from docs/glass-cutout-camera-options.md; used only for the cost guard and the
# cost_usd column. Unknown endpoints count DEFAULT_COST.
COST = {
    "fal-ai/sam-3/image": 0.005,
    "fal-ai/birefnet": 0.002,
    "fal-ai/birefnet/v2": 0.002,
    "microsoft/moge-3": 0.03,
    "fal-ai/moge-3": 0.03,
    "fal-ai/vggt-1b": 0.001,  # per frame; callers pass cost= for multi-frame calls
}
DEFAULT_COST = 0.01


class FalError(RuntimeError):
    """A fal call failed (HTTP error, failed request, timeout, bad response)."""


class BudgetExceeded(FalError):
    """The run's estimated spend would pass max_cost; callers fall back to the local path."""


def load_key(env_file: Path | None = None) -> str | None:
    """FAL_KEY from the environment, else from env_file (KEY=VALUE lines, # comments, optional quotes)."""
    key = os.environ.get("FAL_KEY", "").strip()
    if key:
        return key
    if env_file is not None and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().removeprefix("export ").strip() == "FAL_KEY":
                value = value.strip().strip('"').strip("'")
                return value or None
    return None


def encode_image(image: np.ndarray, max_side: int = 0, fmt: str = ".jpg", quality: int = 95) -> tuple[bytes, float]:
    """(encoded bytes, scale) of a BGR(A) image, downscaled with INTER_AREA so its long side <= max_side
    (0 = full size). scale = sent size / original size. Masks: send as PNG."""
    h, w = image.shape[:2]
    s = 1.0
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        image = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, image, params)
    if not ok:
        raise FalError(f"could not encode image as {fmt}")
    return buf.tobytes(), s


@dataclass
class Spend:
    calls: int = 0
    cached: int = 0
    usd: float = 0.0
    by_endpoint: dict = field(default_factory=dict)


class FalClient:
    def __init__(self, key: str | None, cache_dir: Path | None = None, timeout: float = 120.0,
                 max_cost: float = 0.0, poll_s: float = 0.5, session=None, offline: bool = False):
        """max_cost 0 = no limit. offline=True serves only the cache (tests, reruns) and raises FalError on a miss."""
        self._key = key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.max_cost = max_cost
        self.poll_s = poll_s
        self.offline = offline
        self.spend = Spend()
        self._session = session
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # never show the key
        return f"FalClient(cache_dir={self.cache_dir!s}, spent=${self.spend.usd:.3f})"

    # ------------------------------------------------------------------ public
    def run(self, endpoint: str, payload: dict | None = None, images: dict | None = None, cost: float | None = None,
            image_format: str = ".jpg", max_side: int = 0) -> dict:
        """Call endpoint and return its JSON result. images maps payload field -> BGR(A) ndarray, list of ndarrays,
        or already-encoded bytes; each is uploaded and its URL put in that field. cost overrides the COST table
        for this call (e.g. per-frame models). Results are cached on disk."""
        payload = dict(payload or {})
        images = images or {}
        encoded = {k: ([self._enc(i, image_format, max_side) for i in v] if isinstance(v, list)
                       else self._enc(v, image_format, max_side)) for k, v in images.items()}
        key = self._cache_key(endpoint, payload, encoded)
        hit = self._cache_get(key + ".json")
        if hit is not None:
            self.spend.cached += 1
            return json.loads(hit)
        if self.offline:
            raise FalError(f"{endpoint}: not in the cache (offline)")
        est = COST.get(endpoint, DEFAULT_COST) if cost is None else cost
        if self.max_cost and self.spend.usd + est > self.max_cost + 1e-9:
            raise BudgetExceeded(f"{endpoint}: estimated spend ${self.spend.usd + est:.2f} would pass "
                                 f"max_cost_per_car ${self.max_cost:.2f}")
        if not self._key:
            raise FalError("FAL_KEY is not set (environment variable or background_changer\\.env)")
        for k, v in encoded.items():
            payload[k] = [self._upload(b, image_format) for b in v] if isinstance(v, list) else self._upload(v, image_format)
        result = self._queue(endpoint, payload)
        self.spend.calls += 1
        self.spend.usd += est
        self.spend.by_endpoint[endpoint] = round(self.spend.by_endpoint.get(endpoint, 0.0) + est, 4)
        self._cache_put(key + ".json", json.dumps(result).encode())
        return result

    def fetch_bytes(self, url: str) -> bytes:
        """Bytes of a result file (cached by URL). data: URIs are decoded locally."""
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        key = "url_" + hashlib.sha1(url.encode()).hexdigest()
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        if self.offline:
            raise FalError("result file not in the cache (offline)")
        r = self._http().get(url, timeout=self.timeout)
        if r.status_code != 200:
            raise FalError(f"download failed: HTTP {r.status_code}")
        self._cache_put(key, r.content)
        return r.content

    def fetch_image(self, url: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(self.fetch_bytes(url), np.uint8), flags)
        if img is None:
            raise FalError("result file is not an image")
        return img

    # ------------------------------------------------------------------ internals
    def _http(self):
        if self._session is None:
            import requests  # noqa: PLC0415 - only needed on a cache miss

            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        return {"Authorization": f"Key {self._key}", "Content-Type": "application/json"}

    @staticmethod
    def _enc(img, fmt: str, max_side: int) -> bytes:
        if isinstance(img, (bytes, bytearray)):
            return bytes(img)
        f = ".png" if (img.dtype == bool or img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 4)) else fmt
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        return encode_image(img, max_side, f)[0]

    @staticmethod
    def _cache_key(endpoint: str, payload: dict, encoded: dict) -> str:
        h = hashlib.sha1(endpoint.encode())
        h.update(json.dumps(payload, sort_keys=True).encode())
        for k in sorted(encoded):
            h.update(k.encode())
            for b in (encoded[k] if isinstance(encoded[k], list) else [encoded[k]]):
                h.update(hashlib.sha1(b).digest())
        return endpoint.replace("/", "_") + "_" + h.hexdigest()

    def _cache_get(self, name: str) -> bytes | None:
        if not self.cache_dir:
            return None
        p = self.cache_dir / name
        return p.read_bytes() if p.is_file() else None

    def _cache_put(self, name: str, data: bytes) -> None:
        if self.cache_dir:
            tmp = self.cache_dir / (name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(self.cache_dir / name)

    def _upload(self, data: bytes, fmt: str) -> str:
        """fal CDN URL for data; falls back to a data URI when the storage API refuses."""
        ctype = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        try:
            r = self._http().post(UPLOAD_INITIATE_URL, headers=self._headers(), timeout=self.timeout,
                                  json={"content_type": ctype, "file_name": "upload" + (".png" if ctype == "image/png" else ".jpg")})
            if r.status_code == 200:
                info = r.json()
                put = self._http().put(info["upload_url"], data=data, headers={"Content-Type": ctype},
                                       timeout=self.timeout)
                if put.status_code in (200, 201, 204):
                    return info["file_url"]
        except Exception:  # noqa: BLE001 - storage is an optimisation; a data URI always works for small files
            pass
        return f"data:{ctype};base64," + base64.b64encode(data).decode()

    def _queue(self, endpoint: str, payload: dict) -> dict:
        s = self._http()
        last = None
        for attempt in range(3):
            try:
                r = s.post(f"{QUEUE_URL}/{endpoint}", headers=self._headers(), json=payload, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise FalError(f"{endpoint}: HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise FalError(f"{endpoint}: HTTP {r.status_code}: {r.text[:300]}")  # not retried
                sub = r.json()
                return self._wait(endpoint, sub)
            except FalError as e:
                last = e
                if "HTTP 4" in str(e) and "HTTP 429" not in str(e):
                    raise
            except Exception as e:  # noqa: BLE001 - network error: retry
                last = FalError(f"{endpoint}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
        raise last

    def _wait(self, endpoint: str, sub: dict) -> dict:
        s = self._http()
        status_url = sub.get("status_url") or f"{QUEUE_URL}/{endpoint}/requests/{sub['request_id']}/status"
        response_url = sub.get("response_url") or f"{QUEUE_URL}/{endpoint}/requests/{sub['request_id']}"
        t0 = time.monotonic()
        while True:
            r = s.get(status_url, headers=self._headers(), timeout=self.timeout)
            if r.status_code >= 400 and r.status_code != 202:
                raise FalError(f"{endpoint}: status HTTP {r.status_code}: {r.text[:300]}")
            st = r.json().get("status")
            if st == "COMPLETED":
                break
            if st in ("FAILED", "ERROR", "CANCELLED"):
                raise FalError(f"{endpoint}: request {st}")
            if time.monotonic() - t0 > self.timeout:
                raise FalError(f"{endpoint}: timed out after {self.timeout:.0f} s")
            time.sleep(self.poll_s)
        r = s.get(response_url, headers=self._headers(), timeout=self.timeout)
        if r.status_code >= 400:
            raise FalError(f"{endpoint}: result HTTP {r.status_code}: {r.text[:300]}")
        out = r.json()
        return out.get("response", out) if isinstance(out, dict) and "response" in out and len(out) <= 3 else out


__all__ = ["COST", "BudgetExceeded", "FalClient", "FalError", "Spend", "encode_image", "load_key"]
