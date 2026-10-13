"""
Engine facade — what the HTTP and MCP servers call to synthesize speech.

Mirrors the public API of :mod:`tts` (load / unload / is_loaded /
using_voice_clone / synthesize*), in one of two modes chosen by
``settings.idle_unload_seconds``:

* ``0`` (default) — in-process: every call goes straight to :mod:`tts`.
* ``> 0`` — worker: the model lives in a child process (:mod:`worker`) that
  is stopped after that many idle seconds and respawned on the next request.
  Ending the process is the only way to hand everything back to the OS —
  model weights, the torch runtime and compiled GPU kernels (~4 GB on an
  Intel iGPU even after the weights are freed in-process).

Like :mod:`worker`, this module must not import torch or :mod:`tts` at module
level, so the long-running server process stays small.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Optional

from adjutantvoice.config import settings

logger = logging.getLogger(__name__)


def _tts():
    from adjutantvoice import tts

    return tts


def _use_worker() -> bool:
    return settings.idle_unload_seconds > 0


# ---------------------------------------------------------------------------
# Worker-mode state (all guarded by _lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_started = False
_voice_clone_path: Optional[Path] = None
_using_clone = False
_proc = None
_conn = None
_last_used = 0.0
_idle_timer: Optional[threading.Timer] = None


def _worker_main():
    """The worker entry point (a seam for tests, which swap in a fake)."""
    from adjutantvoice import worker

    return worker.main


def _spawn() -> None:
    global _proc, _conn, _using_clone

    logger.info("Starting TTS worker process …")
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_worker_main(),
        args=(child_conn, _voice_clone_path),
        name="adjutantvoice-worker",
        daemon=True,
    )
    proc.start()
    child_conn.close()  # so recv() sees EOF if the worker dies
    try:
        status, result = parent_conn.recv()
    except EOFError:
        proc.join(5)
        raise RuntimeError(f"TTS worker exited during startup (exit code {proc.exitcode})") from None
    if status == "err":
        proc.join(5)
        raise result
    _proc, _conn, _using_clone = proc, parent_conn, result


def _stop() -> None:
    global _proc, _conn
    if _conn is not None:
        try:
            _conn.send(None)
        except OSError:
            pass
        _conn.close()
    if _proc is not None:
        _proc.join(10)
        if _proc.is_alive():
            _proc.terminate()
            _proc.join(5)
    _proc = _conn = None


def _touch() -> None:
    """Record a use of the worker and (re)arm the idle timer."""
    global _last_used, _idle_timer
    _last_used = time.monotonic()
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(settings.idle_unload_seconds, _idle_stop)
    _idle_timer.daemon = True
    _idle_timer.start()


def _idle_stop() -> None:
    with _lock:
        # A request may have run while this timer waited on the lock.
        if _conn is None or time.monotonic() - _last_used < settings.idle_unload_seconds:
            return
        logger.info("TTS idle for %ss — stopping worker process.", settings.idle_unload_seconds)
        _stop()


def _call(name: str, *args):
    with _lock:
        if not _started:
            raise RuntimeError("TTS model is not loaded. Call engine.load() first.")
        if _conn is None:
            _spawn()
        try:
            _conn.send((name, args))
            status, result = _conn.recv()
        except (EOFError, OSError):
            _stop()
            raise RuntimeError("TTS worker process died; it will be restarted on the next request.") from None
        _touch()
    if status == "err":
        raise result
    return result


# ---------------------------------------------------------------------------
# Public API (same shape as tts)
# ---------------------------------------------------------------------------

def load(voice_clone_path: Optional[Path] = None) -> None:
    """Load the model — in a worker process if idle unloading is enabled."""
    global _started, _voice_clone_path
    if not _use_worker():
        _tts().load(voice_clone_path)
        return
    with _lock:
        if _started:
            return
        _voice_clone_path = voice_clone_path
        _spawn()  # load eagerly so the first request after startup is fast
        _started = True
        _touch()


def unload() -> None:
    """Stop the worker (if any) and release in-process model state."""
    global _started, _idle_timer
    with _lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        _stop()
        _started = False
    _tts().unload()


def is_loaded() -> bool:
    """True once :func:`load` has run — the worker may be idle-stopped."""
    if _use_worker():
        return _started
    return _tts().is_loaded()


def using_voice_clone() -> bool:
    if _use_worker():
        return _started and _using_clone
    return _tts().using_voice_clone()


def synthesize(text: str) -> bytes:
    if _use_worker():
        return _call("synthesize", text)
    return _tts().synthesize(text)


def synthesize_to_buffer(text: str) -> io.BytesIO:
    if not _use_worker():
        return _tts().synthesize_to_buffer(text)
    buf = io.BytesIO(synthesize(text))
    buf.seek(0)
    return buf


def synthesize_with_duration(text: str) -> tuple[bytes, float]:
    if _use_worker():
        return _call("synthesize_with_duration", text)
    return _tts().synthesize_with_duration(text)
