"""
Model worker process — hosts the OmniVoice model on behalf of :mod:`engine`.

Started by ``engine`` via ``multiprocessing`` (spawn). Protocol over the
connection:

  worker → parent  ("ok", using_voice_clone) once the model is loaded,
                   or ("err", exc) if loading failed
  parent → worker  (tts_function_name, args) → ("ok", result) | ("err", exc)
  parent → worker  None → worker exits

This module must not import torch (or ``tts``) at module level: the parent
imports it to reference :func:`main`, and the whole point of the worker is
keeping the torch runtime out of the parent process.
"""

from __future__ import annotations

import logging
import os
import sys


def _send_error(conn, exc: BaseException) -> None:
    try:
        conn.send(("err", exc))
    except Exception:  # exception not picklable
        conn.send(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))


def main(conn, voice_clone_path=None) -> None:
    # Under pythonw (e.g. a logon task) there are no console streams at all;
    # log to AV_LOG_FILE instead, as the parent does.
    if sys.stderr is None:
        sys.stderr = open(os.environ.get("AV_LOG_FILE", os.devnull), "a", buffering=1, encoding="utf-8")
    # Keep stdout clean — with the MCP stdio transport it carries protocol
    # messages, and model libraries print progress bars and warnings.
    os.dup2(sys.stderr.fileno(), 1)
    sys.stdout = sys.stderr
    logging.basicConfig(level=logging.INFO, format="[av-worker] %(message)s")

    from adjutantvoice import tts

    try:
        tts.load(voice_clone_path)
    except BaseException as exc:
        _send_error(conn, exc)
        return
    conn.send(("ok", tts.using_voice_clone()))

    while True:
        try:
            msg = conn.recv()
        except EOFError:  # parent went away
            return
        if msg is None:
            return
        name, args = msg
        try:
            conn.send(("ok", getattr(tts, name)(*args)))
        except Exception as exc:
            _send_error(conn, exc)
