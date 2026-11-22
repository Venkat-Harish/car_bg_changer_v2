"""hosted.fal: key loading, the disk cache, the cost guard and the queue protocol (fake HTTP session, no network)."""

import numpy as np
import pytest

from hosted.fal import BudgetExceeded, FalClient, FalError, load_key


class Resp:
    def __init__(self, status=200, data=None, content=b""):
        self.status_code, self._data, self.content = status, data, content
        self.text = str(data)

    def json(self):
        return self._data


class FakeSession:
    """queue.fal.run: submit -> status (IN_PROGRESS once, then COMPLETED) -> result. Storage upload refused."""

    def __init__(self, result=None):
        self.result = result or {"ok": 1}
        self.posts, self.gets, self.status_calls = [], [], 0
        self.headers_seen = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, json))
        self.headers_seen.append(headers)
        if "storage" in url:
            return Resp(403, {})
        return Resp(200, {"request_id": "r1", "status_url": "S", "response_url": "R"})

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        if url == "S":
            self.status_calls += 1
            return Resp(200, {"status": "COMPLETED" if self.status_calls > 1 else "IN_PROGRESS"})
        if url == "R":
            return Resp(200, self.result)
        return Resp(200, None, b"filebytes")

    def put(self, *a, **k):
        return Resp(500)


def image():
    return np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)


def test_key_from_env_then_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# fal\nOTHER=1\nFAL_KEY = 'abc:def'\n", encoding="utf-8")
    monkeypatch.delenv("FAL_KEY", raising=False)
    assert load_key(env) == "abc:def"
    assert load_key(tmp_path / "missing.env") is None
    env.write_text("FAL_KEY=\n", encoding="utf-8")
    assert load_key(env) is None
    monkeypatch.setenv("FAL_KEY", "from-env")
    assert load_key(env) == "from-env"


def test_run_uploads_polls_and_caches(tmp_path):
    s = FakeSession({"masks": []})
    c = FalClient("k", cache_dir=tmp_path, session=s, poll_s=0)
    out = c.run("fal-ai/sam-3/image", {"prompt": "car"}, images={"image_url": image()})
    assert out == {"masks": []}
    url, payload = s.posts[-1]
    assert url.endswith("/fal-ai/sam-3/image") and payload["image_url"].startswith("data:image/jpeg;base64,")
    assert s.headers_seen[-1]["Authorization"] == "Key k"
    assert c.spend.calls == 1 and c.spend.usd == pytest.approx(0.005)
    # same image + payload: served from disk, nothing sent, nothing billed; also offline
    again = FalClient(None, cache_dir=tmp_path, session=FakeSession(), offline=True)
    assert again.run("fal-ai/sam-3/image", {"prompt": "car"}, images={"image_url": image()}) == {"masks": []}
    assert again.spend.cached == 1 and again.spend.usd == 0
    with pytest.raises(FalError):  # another prompt is another request
        again.run("fal-ai/sam-3/image", {"prompt": "window"}, images={"image_url": image()})


def test_budget_and_missing_key(tmp_path):
    c = FalClient("k", cache_dir=tmp_path, session=FakeSession(), max_cost=0.04, poll_s=0)
    c.run("microsoft/moge-3", {}, images={"image_url": image()})
    with pytest.raises(BudgetExceeded):
        c.run("microsoft/moge-3", {"x": 2}, images={"image_url": image()})
    with pytest.raises(FalError, match="FAL_KEY"):
        FalClient(None, session=FakeSession()).run("fal-ai/birefnet/v2", {})


def test_fetch_is_cached_and_key_is_never_shown(tmp_path):
    c = FalClient("secret-key", cache_dir=tmp_path, session=FakeSession(), poll_s=0)
    assert c.fetch_bytes("https://v3.fal.media/x.png") == b"filebytes"
    off = FalClient(None, cache_dir=tmp_path, offline=True)
    assert off.fetch_bytes("https://v3.fal.media/x.png") == b"filebytes"
    assert "secret-key" not in repr(c)
    assert not any(b"secret-key" in p.read_bytes() for p in tmp_path.iterdir())
