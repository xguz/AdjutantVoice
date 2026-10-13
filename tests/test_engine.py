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
        elif name == "synthesize":
            conn.send(("ok", f"MP3:{text}".encode()))
        else:
            conn.send(("ok", (b"MP3", 1.5)))


@pytest.fixture
def worker_mode(monkeypatch):
    from adjutantvoice.config import settings

    monkeypatch.setattr(settings, "idle_unload_seconds", 60)
    monkeypatch.setattr(engine, "_worker_main", lambda: fake_worker)
    return settings


def test_load_starts_worker_and_synthesizes(worker_mode):
    engine.load()

    assert engine.is_loaded() is True
    assert engine._proc.is_alive()
    assert engine.synthesize("hello") == b"MP3:hello"
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


def test_voice_clone_path_is_passed_to_worker(worker_mode, tmp_path):
    engine.load(voice_clone_path=tmp_path / "clone.pkl")

    assert engine.using_voice_clone() is True


def test_idle_worker_is_stopped_and_respawned(worker_mode):
    worker_mode.idle_unload_seconds = 1
    engine.load()
    proc = engine._proc

    time.sleep(2)

    assert engine._proc is None
    assert not proc.is_alive()
    assert engine.is_loaded() is True  # still "started" — just not resident
    assert engine.synthesize("again") == b"MP3:again"
    assert engine._proc.is_alive()


def test_unload_stops_worker(worker_mode):
    engine.load()
    proc = engine._proc

    engine.unload()

    assert not proc.is_alive()
    assert engine.is_loaded() is False
