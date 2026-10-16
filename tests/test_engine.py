"""Tests for adjutantvoice.engine in worker mode, with a fake worker process.

In-process mode is exercised indirectly by test_server.py / test_mcp.py.
Here `engine._worker_main` is swapped for `fake_worker` below, which speaks
the same protocol as `adjutantvoice.worker.main` without importing torch, so
these tests spawn real child processes but never load a model.
"""

import time

import pytest

from adjutantvoice import engine


def fake_worker(conn, voice_clone_path=None):
    conn.send(("ok", voice_clone_path is not None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        name, (text,) = msg
        if not text.strip():
            conn.send(("err", ValueError("text must not be empty")))
            continue
        if text == "hang":  # a worker that never finishes (e.g. the machine is paging)
            time.sleep(60)
        if name == "synthesize":
            conn.send(("ok", f"MP3:{text}".encode()))
        else:
            conn.send(("ok", (b"MP3", 1.5)))


def never_ready_worker(conn, voice_clone_path=None):
    time.sleep(60)  # stuck loading the model


@pytest.fixture
def worker_mode(monkeypatch):
    from adjutantvoice.config import settings

    # Patched (not just read) so tests may change them directly and still get them restored.
    monkeypatch.setattr(settings, "idle_unload_seconds", 60)
    monkeypatch.setattr(settings, "worker_timeout_seconds", 60)
    monkeypatch.setattr(settings, "min_available_memory_gb", 0)
    monkeypatch.setattr(engine, "_worker_main", lambda: fake_worker)
    return settings


def test_worker_starts_on_first_request(worker_mode):
    engine.load()

    assert engine.is_loaded() is True
    assert engine._proc is None  # load() doesn't spawn the worker
    assert engine.synthesize("hello") == b"MP3:hello"
    assert engine._proc.is_alive()
    assert engine.synthesize_with_duration("hello") == (b"MP3", 1.5)
    assert engine.synthesize_to_buffer("hi").read() == b"MP3:hi"


def test_synthesize_before_load_raises(worker_mode):
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.synthesize("hello")


def test_worker_errors_are_reraised(worker_mode):
    engine.load()

    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ")
    assert engine.synthesize("still works") == b"MP3:still works"


def test_using_voice_clone_is_known_before_worker_starts(worker_mode, tmp_settings):
    clone_path = tmp_settings.voice_clone_path
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    clone_path.write_bytes(b"clone")

    engine.load()

    assert engine._proc is None
    assert engine.using_voice_clone() is True


def test_voice_clone_path_is_passed_to_worker(worker_mode, tmp_settings, tmp_path):
    engine.load(voice_clone_path=tmp_path / "clone.pkl")  # file doesn't exist
    assert engine.using_voice_clone() is False

    engine.synthesize("hello")

    assert engine.using_voice_clone() is True  # the fake worker reports the path it got


def test_idle_worker_is_stopped_and_respawned(worker_mode):
    worker_mode.idle_unload_seconds = 1
    engine.load()
    engine.synthesize("first")
    proc = engine._proc

    time.sleep(2)

    assert engine._proc is None
    assert not proc.is_alive()
    assert engine.is_loaded() is True  # still "started" — just not resident
    assert engine.synthesize("again") == b"MP3:again"
    assert engine._proc.is_alive()


def test_unload_stops_worker(worker_mode):
    engine.load()
    engine.synthesize("hello")
    proc = engine._proc

    engine.unload()

    assert not proc.is_alive()
    assert engine.is_loaded() is False


def test_stuck_request_times_out_and_worker_is_replaced(worker_mode):
    engine.load()
    engine.synthesize("warm up")  # start the worker under the normal budget
    stuck = engine._proc
    worker_mode.worker_timeout_seconds = 2

    with pytest.raises(RuntimeError, match="longer than 2s"):
        engine.synthesize("hang")

    assert not stuck.is_alive()
    worker_mode.worker_timeout_seconds = 60
    assert engine.synthesize("next") == b"MP3:next"  # a fresh worker


def test_worker_that_never_finishes_loading_times_out(worker_mode, monkeypatch):
    worker_mode.worker_timeout_seconds = 2
    monkeypatch.setattr(engine, "_worker_main", lambda: never_ready_worker)
    engine.load()

    with pytest.raises(RuntimeError, match="didn't finish loading"):
        engine.synthesize("hello")

    assert engine._proc is None


def test_worker_not_started_when_memory_is_low(worker_mode, monkeypatch):
    worker_mode.min_available_memory_gb = 4
    monkeypatch.setattr(engine, "_available_memory_gb", lambda: 1.5)
    engine.load()

    with pytest.raises(RuntimeError, match="only 1.5 GB of memory is available"):
        engine.synthesize("hello")

    assert engine._proc is None
    monkeypatch.setattr(engine, "_available_memory_gb", lambda: 8.0)
    assert engine.synthesize("hello") == b"MP3:hello"
