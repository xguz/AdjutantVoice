"""Tests for the adjutantvoice.mcp tool functions.

`@mcp.tool(...)` leaves the underlying function directly callable, so these
tests call `tts_file` / `tts_stream` / `tts_speak` as plain functions rather
than going through a real MCP transport. `tts.synthesize` itself is mocked
directly (see test_tts.py for coverage of that layer).
"""

import subprocess

import pytest

from adjutantvoice import mcp as mcp_module
from adjutantvoice import tts


@pytest.fixture(autouse=True)
def fake_synthesize(monkeypatch):
    monkeypatch.setattr(tts, "synthesize", lambda text: b"FAKE_MP3")
    monkeypatch.setattr(tts, "synthesize_with_duration", lambda text: (b"FAKE_MP3", 3.2))


# ---------------------------------------------------------------------------
# tts_file
# ---------------------------------------------------------------------------

def test_tts_file_writes_mp3_to_output_dir(tmp_settings, tmp_path):
    # In production, mcp.py's lifespan creates tts_output_dir on startup
    # (see _lifespan in mcp.py) — replicate that here since we call the
    # tool function directly without going through the real lifespan.
    tmp_settings.tts_output_dir.mkdir(parents=True, exist_ok=True)

    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello there", encoding="utf-8")

    out_path_str = mcp_module.tts_file(str(text_file), ctx=None)

    out_path = tmp_path
    assert out_path_str.endswith(".mp3")
    from pathlib import Path
    written = Path(out_path_str)
    assert written.exists()
    assert written.read_bytes() == b"FAKE_MP3"
    assert written.parent == tmp_settings.tts_output_dir.resolve()


def test_tts_file_raises_if_missing(tmp_settings):
    with pytest.raises(FileNotFoundError):
        mcp_module.tts_file("/nonexistent/path/notes.txt", ctx=None)


# ---------------------------------------------------------------------------
# tts_stream
# ---------------------------------------------------------------------------

def test_tts_stream_returns_raw_bytes():
    result = mcp_module.tts_stream("hello there", ctx=None)
    assert result == b"FAKE_MP3"


# ---------------------------------------------------------------------------
# tts_speak
# ---------------------------------------------------------------------------

def test_tts_speak_plays_and_confirms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_module.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or subprocess.CompletedProcess(a, 0),
    )

    result = mcp_module.tts_speak("hello there", ctx=None)

    assert result == "Played 3.2 s of audio."
    assert len(calls) == 1
    assert calls[0][0][0][0] == "ffplay"


def test_tts_speak_raises_runtime_error_if_ffplay_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(mcp_module.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="ffplay"):
        mcp_module.tts_speak("hello there", ctx=None)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def test_start_http_uses_selector_loop_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_module.sys, "platform", "win32")
    monkeypatch.setattr(mcp_module.asyncio, "set_event_loop_policy", lambda p: calls.append(type(p).__name__))
    monkeypatch.setattr(mcp_module.asyncio, "WindowsSelectorEventLoopPolicy", type("WindowsSelectorEventLoopPolicy", (), {}), raising=False)
    monkeypatch.setattr(mcp_module.mcp, "run", lambda **kw: calls.append(kw))

    mcp_module.start(transport="streamable-http", port=1234)

    assert calls == ["WindowsSelectorEventLoopPolicy", {"transport": "streamable-http", "port": 1234}]


def test_start_stdio_leaves_event_loop_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_module.asyncio, "set_event_loop_policy", lambda p: calls.append("policy"))
    monkeypatch.setattr(mcp_module.mcp, "run", lambda **kw: calls.append(kw))

    mcp_module.start(transport="stdio")

    assert calls == [{}]


# ---------------------------------------------------------------------------
# tts_speak playback queue
# ---------------------------------------------------------------------------

@pytest.fixture
def playback(monkeypatch, tmp_settings):
    """Fresh queue, short gap, and a fake ffplay that records when each clip played."""
    import threading
    import time

    monkeypatch.setattr(mcp_module, "_playback", mcp_module._PlaybackQueue())
    monkeypatch.setattr(tmp_settings, "speak_gap_seconds", 0.3)
    played = []
    lock = threading.Lock()

    def fake_synth(text):
        if text == "boom":
            raise RuntimeError("synthesis failed")
        if text == "slow to synthesize":
            time.sleep(0.4)
        return text.encode(), 1.0

    def fake_ffplay(cmd, **kwargs):
        start = time.monotonic()
        time.sleep(0.2)
        with lock:
            played.append((open(cmd[-1], "rb").read().decode(), start, time.monotonic()))

    monkeypatch.setattr(tts, "synthesize_with_duration", fake_synth)
    monkeypatch.setattr(mcp_module.subprocess, "run", fake_ffplay)
    return played


def _speak_concurrently(*texts):
    import threading
    import time

    threads = []
    for text in texts:
        t = threading.Thread(target=lambda s=text: _ignore_errors(mcp_module.tts_speak, s, ctx=None))
        t.start()
        threads.append(t)
        time.sleep(0.05)  # fixes the arrival order
    for t in threads:
        t.join(10)


def _ignore_errors(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except RuntimeError:
        pass


def test_tts_speak_plays_one_at_a_time_in_arrival_order_with_a_gap(playback):
    # The first caller is slower to synthesize, but it arrived first, so it plays first.
    _speak_concurrently("slow to synthesize", "second", "third")

    assert [text for text, _, _ in playback] == ["slow to synthesize", "second", "third"]
    for (_, _, prev_end), (_, next_start, _) in zip(playback, playback[1:]):
        assert next_start - prev_end >= 0.3 - 0.02  # no overlap, gap respected


def test_tts_speak_failure_does_not_block_the_queue(playback):
    _speak_concurrently("first", "boom", "third")

    assert [text for text, _, _ in playback] == ["first", "third"]


def test_tts_speak_no_gap_when_nothing_played_recently(playback):
    import time

    mcp_module.tts_speak("hello", ctx=None)
    time.sleep(0.4)  # longer than the gap
    t = time.monotonic()
    mcp_module.tts_speak("later", ctx=None)

    assert playback[1][1] - t < 0.1  # played right away
