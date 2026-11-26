"""hosted.redact + the privacy hooks: people blurred in what is uploaded (never in the photo), the plate gate,
cache retention. Fake detector and fake HTTP session: no model, no network."""

import base64
import os
import time

import cv2
import numpy as np

import change_background as cb
from hosted.fal import FalClient
from hosted.redact import PeopleRedactor, blur_boxes
from test_hosted_fal import FakeSession

BOX = (20.0, 10.0, 50.0, 40.0)  # x0, y0, x1, y1


def noise(h=80, w=120, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_blur_changes_only_the_padded_box():
    img = noise()
    out = blur_boxes(img, [BOX], pad=0.0)
    assert out is not img and np.array_equal(img, noise())  # the input is never edited
    inside = (slice(10, 40), slice(20, 50))
    assert not np.array_equal(out[inside], img[inside])
    keep = np.ones(img.shape[:2], bool)
    keep[inside] = False
    assert np.array_equal(out[keep], img[keep])
    assert blur_boxes(img, []) is img
    # pixelated: far less detail left than in the noise (a face would not be readable)
    assert out[inside].std() < img[inside].std() / 3


def test_redactor_detects_once_per_image_and_skips_masks():
    calls = []
    red = PeopleRedactor(detect=lambda im: calls.append(1) or [BOX])
    img = noise()
    a, b = red(img), red(img.copy())
    assert np.array_equal(a, b) and len(calls) == 1 and (red.images, red.people) == (1, 1)
    mask = np.zeros((80, 120), np.uint8)
    assert red(mask) is mask and red(mask.astype(bool)) is not None and len(calls) == 1
    none = PeopleRedactor(detect=lambda im: [])
    img2 = noise(seed=1)
    assert none(img2) is img2


def sent_image(session) -> np.ndarray:
    url = session.posts[-1][1]["image_url"]
    data = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_client_uploads_the_blurred_copy_and_leaves_the_photo(tmp_path):
    img = noise(200, 300)
    keep = img.copy()
    s = FakeSession({"ok": 1})
    c = FalClient("k", cache_dir=tmp_path / "a", session=s, poll_s=0,
                  redact=PeopleRedactor(detect=lambda im: [(100.0, 50.0, 160.0, 150.0)]))
    c.run("fal-ai/sam-3/image", {"prompt": "car"}, images={"image_url": img})
    assert np.array_equal(img, keep)  # the caller's photo is untouched
    blurred = sent_image(s)
    s2 = FakeSession({"ok": 1})
    FalClient("k", cache_dir=tmp_path / "b", session=s2, poll_s=0).run(
        "fal-ai/sam-3/image", {"prompt": "car"}, images={"image_url": img})
    plain = sent_image(s2)
    box = (slice(50, 150), slice(100, 160))
    assert np.abs(blurred[box].astype(int) - img[box]).mean() > 30  # the person is gone from the upload
    assert plain[box].astype(int).std() > 2 * blurred[box].astype(int).std()
    far = (slice(0, 20), slice(0, 60))  # away from the box: same as an unredacted upload (same JPEG)
    assert np.abs(blurred[far].astype(int) - plain[far]).max() <= 2


def test_client_redacts_every_frame_of_a_list(tmp_path):
    s = FakeSession({"ok": 1})
    seen = []
    red = PeopleRedactor(detect=lambda im: seen.append(im.shape) or [])
    FalClient("k", cache_dir=tmp_path, session=s, poll_s=0, redact=red).run(
        "fal-ai/vggt-1b", {}, images={"image_urls": [noise(seed=1), noise(seed=2)]}, cost=0.002)
    assert len(seen) == 2


def test_prune_cache_removes_only_old_files(tmp_path):
    c = FalClient("k", cache_dir=tmp_path)
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text("{}")
    new.write_text("{}")
    t = time.time()
    os.utime(old, (t - 40 * 86400, t - 40 * 86400))
    assert c.prune_cache(0) == 0 and old.exists()  # 0 = keep forever
    assert c.prune_cache(30, now=t) == 1
    assert not old.exists() and new.exists()


def test_plate_gate(tmp_path):
    on = dict(cb.HOSTED_DEFAULTS)
    assert not cb.plates_masked(tmp_path, on)
    (tmp_path / "detections.csv").write_text("file\n", encoding="utf-8")
    assert cb.plates_masked(tmp_path, on)
    assert cb.plates_masked(tmp_path / "nowhere", {**on, "require_masked_plates": False})
