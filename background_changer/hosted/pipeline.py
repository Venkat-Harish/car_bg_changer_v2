"""Glue between change_background.py and the hosted models: one HostedRun per run.

Stage 1 (per photo):  raw alpha (BiRefNet on fal, local BiRefNet as fallback) -> CarSegmenter.finish with a
                      HostedIsolator (SAM 3 "car") -> window mask (SAM 3 "car window") -> tinted frame.
                      The untouched file is copied to <out>/originals/; the tinted frame is written losslessly to
                      <out>/_frames/<stem>.png and is "the photo" from then on (stage 3 re-reads it).
Between 1 and 2:      MoGe-3 per frame + one VGGT call per folder -> HostedCamera per photo (Measured.external).

Every hosted step falls back to the local path on a FalError (incl. BudgetExceeded), flagged; a run never stops
because fal is down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bg_changer.segment import CarSegmenter

from .fal import FalClient, FalError, load_key


@dataclass
class HostedInfo:
    """Per-photo hosted results for results.csv (HOSTED_CSV_FIELDS)."""
    cutout: str = ""  # fal | local
    sam3_status: str = ""
    glass_pct: float | str = ""
    camera_source: str = ""
    moge_f: float | str = ""
    moge_h: float | str = ""
    vggt_ok: str = ""
    cost_usd: float = 0.0
    flags: list[str] = field(default_factory=list)

    def cols(self) -> dict:
        return {"hosted": self.cutout or "local", "sam3_status": self.sam3_status, "glass_pct": self.glass_pct,
                "camera_source": self.camera_source, "moge_f": self.moge_f, "moge_h": self.moge_h,
                "vggt_ok": self.vggt_ok, "cost_usd": round(self.cost_usd, 4)}


class HostedSegmenter(CarSegmenter):
    """CarSegmenter whose raw alpha comes from BiRefNet on fal; the local model loads only if fal fails."""

    def __init__(self, client: FalClient, edge_shrink: int, local_args: tuple, max_side: int = 2048,
                 use_fal: bool = True):
        self.client = client
        self.edge_shrink = edge_shrink
        self.last_isolation = None
        self.local_args = local_args
        self.max_side = max_side
        self.use_fal = use_fal
        self._local: CarSegmenter | None = None
        self.last_source = ""
        self.last_error = ""

    def raw_alpha(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.use_fal:
            from .masks import birefnet_alpha

            try:
                a = birefnet_alpha(self.client, image_bgr, self.max_side)
                self.last_source, self.last_error = "fal", ""
                return a
            except FalError as exc:
                self.last_error = str(exc)
        if self._local is None:
            self._local = CarSegmenter(*self.local_args)
        self.last_source = "local"
        return self._local.raw_alpha(image_bgr)

    def release(self) -> None:
        if self._local is not None:
            self._local.release()


class HostedRun:
    def __init__(self, settings: dict, out_dir: Path, cache_dir: Path, env_file: Path, local_args: tuple,
                 edge_shrink: int, client: FalClient | None = None):
        self.s = settings
        self.out_dir = out_dir
        if client is None:
            key = load_key(env_file)
            if not key:
                raise SystemExit("[hosted] enabled = true needs a fal.ai key: set the FAL_KEY environment variable or "
                                 f"put FAL_KEY=... in {env_file} (or set [hosted] enabled = false)")
            client = FalClient(key, cache_dir=cache_dir, timeout=float(settings["timeout_s"]),
                               max_cost=float(settings["max_cost_per_car"]))
        self.client = client
        self.segmenter = HostedSegmenter(client, edge_shrink, local_args, settings["max_side"], settings["cutout"])
        from .masks import HostedIsolator

        self.isolator = HostedIsolator(client, max_side=settings["max_side"])
        self.info: dict[str, HostedInfo] = {}

    # ------------------------------------------------------------------ stage 1
    def cut_out(self, path: Path, image: np.ndarray, trace: dict | None = None):
        """(CarMask, raw alpha) for one photo, the hosted way. Raises NoCarFound like CarSegmenter."""
        info = self.info.setdefault(path.stem, HostedInfo())
        spent0 = self.client.spend.usd
        raw = self.segmenter.raw_alpha(image)
        info.cutout = self.segmenter.last_source
        if self.segmenter.last_source != "fal" and self.s["cutout"]:
            info.flags.append("hosted-cutout-failed")
        try:
            car = self.segmenter.finish(image, raw, self.isolator, trace)
            found = self.segmenter.last_isolation
            info.sam3_status = found.status if found is not None else ""
        except FalError:
            car = self.segmenter.finish(image, raw, None, trace)
            info.sam3_status = "failed"
            info.flags.append("sam3-failed")
        info.cost_usd += self.client.spend.usd - spent0
        return car, raw

    def glass(self, path: Path, image: np.ndarray, car_alpha: np.ndarray) -> tuple[np.ndarray, Path | None,
                                                                                    np.ndarray | None]:
        """(frame to use from now on, its file for stage 3 or None, window mask or None). Saves the original."""
        info = self.info.setdefault(path.stem, HostedInfo())
        if not self.s["glass_tint"]:
            return image, None, None
        from .glass import glass_pct, save_original, tint_glass, window_mask

        spent0 = self.client.spend.usd
        car_mask = car_alpha >= 0.5
        try:
            wm = window_mask(self.client, image, car_mask, self.s["max_side"])
        except FalError:
            info.flags.append("glass-mask-failed")
            info.cost_usd += self.client.spend.usd - spent0
            return image, None, None
        info.cost_usd += self.client.spend.usd - spent0
        info.glass_pct = round(glass_pct(wm, car_mask), 2)
        save_original(path, self.out_dir / "originals")
        if not wm.any():
            return image, None, wm
        tinted = tint_glass(image, wm, self.s["tint_strength"], self.s["tint_keep"])
        frame_path = self.out_dir / "_frames" / f"{path.stem}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", tinted)
        if not ok:
            raise OSError(f"cannot encode {frame_path}")
        buf.tofile(str(frame_path))
        return tinted, frame_path, wm

    def cols(self, stem: str) -> dict:
        info = self.info.get(stem)
        return info.cols() if info else {}

    def flags(self, stem: str) -> list[str]:
        info = self.info.get(stem)
        return list(info.flags) if info else []

    # ------------------------------------------------------------------ between stages 1 and 2
    def camera(self, photos: list, read_frame, frame_alpha) -> dict[str, dict]:
        """MoGe-3 per photo + one VGGT call for the folder -> photo.measured.external (HostedCamera).
        read_frame(path) -> BGR, frame_alpha(photo) -> (full-frame alpha, crop xy). Returns view-prior probs
        {stem: probs} from VGGT's relative yaws anchored on the photos' view labels (empty when not possible).
        Fills the camera columns of photo.hosted in place."""
        from .camera import combine, moge_frame, vggt_car

        if not photos or not self.s["camera"]:
            return {}
        moge, small, small_masks, scales, sizes, keep = [], [], [], [], [], []
        print(f"hosted camera: MoGe-3 on {len(photos)} photo(s)"
              + (" + one VGGT solve over all of them" if self.s["vggt"] else ""), flush=True)
        for photo in photos:
            stem = photo.path.stem
            info = self.info.setdefault(stem, HostedInfo())
            try:
                frame = read_frame(photo.frame_path or photo.path)
            except OSError:
                frame = None
            if frame is None:
                moge.append(None)
                continue
            alpha, _ = frame_alpha(photo)
            mask = alpha >= 0.5
            spent0 = self.client.spend.usd
            try:
                fc = moge_frame(self.client, frame, mask)
            except FalError as exc:
                fc = None
                info.flags.append("moge-failed")
                print(f"  {photo.path.name}: MoGe-3 failed ({exc})", flush=True)
            info.cost_usd += self.client.spend.usd - spent0
            moge.append(fc)
            if fc is not None:
                info.moge_f = round(fc.f_px, 1) if fc.f_px else ""
                info.moge_h = round(fc.height_m, 3) if fc.height_m else ""
            H, W = frame.shape[:2]
            s = min(1.0, 1036 / max(H, W))  # VGGT works at 518 px: keep small copies, not the full frames
            if s < 1:
                frame = cv2.resize(frame, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            small.append(frame)
            small_masks.append(mask)
            scales.append(frame.shape[1] / W)
            keep.append(len(moge) - 1)
            del alpha
        sizes = [p.measured.size for p in photos]
        vggt = None
        if self.s["vggt"] and len(small) >= 2:
            spent0 = self.client.spend.usd
            try:
                got = vggt_car(self.client, small, small_masks)
                vggt = [None] * len(photos)
                for idx, fc, s in zip(keep, got, scales):
                    if fc is not None and fc.f_px:
                        fc.f_px = fc.f_px / s  # focal back to the photo's full size
                    vggt[idx] = fc
            except FalError as exc:
                print(f"  VGGT failed ({exc}); MoGe-3 alone", flush=True)
            share = (self.client.spend.usd - spent0) / max(1, len(keep))
            for idx in keep:
                self.info[photos[idx].path.stem].cost_usd += share
        del small, small_masks
        cams = combine(moge, vggt, sizes)
        for photo, cam, v in zip(photos, cams, vggt or [None] * len(photos)):
            info = self.info[photo.path.stem]
            info.vggt_ok = "" if not self.s["vggt"] else ("yes" if v is not None else "no")
            photo.measured.external = cam
            if photo.hosted is not None:
                photo.hosted.update(info.cols())
        return self._yaw_probs(photos, cams)

    @staticmethod
    def _yaw_probs(photos: list, cams: list) -> dict[str, dict]:
        """VGGT's relative yaws, anchored on the view labels (circular median over all labelled photos, so one
        wrong label does not matter), as a peaked view prior per photo. Needs 3+ photos with both."""
        from bg_changer.probe import parse_name
        from bg_changer.yaw import BIN_DEG

        from .camera import anchor_yaws, yaw_probs

        rel = [c.rel_yaw_deg if c is not None else None for c in cams]
        labels = []
        for p in photos:
            info = parse_name(p.path.stem)
            labels.append(BIN_DEG.get(info.label) if info and info.label in BIN_DEG else None)
        if sum(r is not None and lab is not None for r, lab in zip(rel, labels)) < 3:
            return {}
        offset = anchor_yaws(rel, labels)
        if offset is None:
            return {}
        return {p.path.stem: yaw_probs((offset + r) % 360.0, sigma=20.0) for p, r in zip(photos, rel) if r is not None}

    def apply_focals(self, items: list, config_focal: float | None) -> None:
        """After choose_focals: the hosted focal wins over the lines (not over a [fit] focal from the config)."""
        if config_focal:
            return
        for m in items:
            ext = getattr(m, "external", None)
            if ext is not None and ext.f_px:
                m.focal, m.focal_source = float(ext.f_px), "hosted"
