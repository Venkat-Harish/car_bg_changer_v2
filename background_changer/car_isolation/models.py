"""Where the two models live, and fetching them once on first use.

Both sit next to rembg's own models (`~/.rembg/models/<name>/`, or under REMBG_HOME / U2NET_HOME):

- YOLOX-s (Megvii, Apache-2.0), the official ONNX release asset, pinned by SHA-256 here.
- SAM ViT-B (Meta, Apache-2.0), the ONNX encoder/decoder pair rembg publishes; rembg pins their md5.
"""

from __future__ import annotations

from pathlib import Path

YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
YOLOX_FILE = "yolox_s.onnx"
YOLOX_SHA256 = "sha256:c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SAM_MODEL = "sam_vit_b_01ec64"


def _home() -> Path:
    from rembg.sessions.base import BaseSession

    return Path(BaseSession.rembg_home()) / "models"


def yolox_path() -> Path:
    """The YOLOX-s ONNX file, downloaded (about 35 MB) on first use and checked against its hash."""
    import pooch

    folder = _home() / "yolox"
    path = folder / YOLOX_FILE
    if path.exists():
        return path
    print(f"car_isolation: downloading {YOLOX_FILE} (about 35 MB, once) to {folder}", flush=True)
    return Path(pooch.retrieve(YOLOX_URL, YOLOX_SHA256, fname=YOLOX_FILE, path=folder, progressbar=True))


def sam_paths() -> tuple[Path, Path]:
    """SAM ViT-B (encoder, decoder) ONNX files via rembg's own download and checksums (about 360 MB, once)."""
    from rembg.sessions.sam import SamSession

    have = SamSession.resolve_existing(f"{SAM_MODEL}.encoder.onnx") and SamSession.resolve_existing(
        f"{SAM_MODEL}.decoder.onnx")
    if not have:
        print(f"car_isolation: downloading {SAM_MODEL} encoder + decoder (about 360 MB, once)", flush=True)
    enc, dec = SamSession.download_models(sam_model=SAM_MODEL)
    return Path(enc), Path(dec)


def downloaded() -> dict[str, bool]:
    """Which model files are already on disk (no download)."""
    from rembg.sessions.sam import SamSession

    return {
        YOLOX_FILE: (_home() / "yolox" / YOLOX_FILE).exists(),
        f"{SAM_MODEL}.encoder.onnx": SamSession.resolve_existing(f"{SAM_MODEL}.encoder.onnx") is not None,
        f"{SAM_MODEL}.decoder.onnx": SamSession.resolve_existing(f"{SAM_MODEL}.decoder.onnx") is not None,
    }
