"""Multiprocessing worker IPC tests and dataset priors calculation tests.

Verifies:
1. vision_worker_process IPC under Windows spawn multiprocessing.
2. reasoning_worker_process IPC under Windows spawn multiprocessing.
3. build_priors_ravdess process_dataset on empty or uncalibrated directories.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Dict

import pytest

from dce.pipeline import reasoning_worker_process, vision_worker_process
from dce.schemas import TurnTranscript
from scripts.build_priors_ravdess import process_dataset


def test_windows_spawn_vision_worker_ipc() -> None:
    """Verify vision_worker_process operates cleanly under multiprocessing spawn context."""
    ctx = multiprocessing.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()

    proc = ctx.Process(target=vision_worker_process, args=(in_q, out_q))
    proc.start()

    try:
        # Synthetic video frame packet: (pts_ns, frame)
        pts_test = 500_000_000
        dummy_frame = [0.5, 0.5, 0.5]
        in_q.put((pts_test, dummy_frame))

        res = out_q.get(timeout=5.0)
        assert res is not None
        assert res.get("status") == "OK"
        assert res.get("pts_ns") == pts_test
        assert res.get("face_detected") is False
        assert res.get("mean_lip_aperture") is None
        assert res.get("blink_rate_per_min") is None
        assert res.get("quality_flag") in ("no_vision_backend", "no_face", "extraction_error")

        # Send STOP
        in_q.put("STOP")
        stop_res = out_q.get(timeout=5.0)
        assert stop_res.get("status") == "STOPPED"

    finally:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        assert not proc.is_alive()


def test_windows_spawn_reasoning_worker_ipc() -> None:
    """Verify reasoning_worker_process operates cleanly under multiprocessing spawn context."""
    ctx = multiprocessing.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()

    proc = ctx.Process(target=reasoning_worker_process, args=(in_q, out_q))
    proc.start()

    try:
        transcript = TurnTranscript(
            turn_id=42,
            speaker_id="SPEAKER_00",
            text="I arrived in Munich yesterday afternoon.",
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )
        in_q.put(transcript.model_dump())

        res = out_q.get(timeout=5.0)
        assert res is not None
        assert res.get("status") == "OK"
        assert res.get("turn_id") == 42
        assert res.get("verdict") is None
        assert res.get("quality_flag") == "no_llm_backend"

        # Send STOP
        in_q.put("STOP")
        stop_res = out_q.get(timeout=5.0)
        assert stop_res.get("status") == "STOPPED"

    finally:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        assert not proc.is_alive()


def test_build_priors_empty_directory(tmp_path: Path) -> None:
    """Verify build_priors returns valid literature prior container when directory is empty."""
    empty_dir = tmp_path / "empty_audio"
    empty_dir.mkdir()

    priors_data = process_dataset(empty_dir)
    assert priors_data.calibrated is False
    assert priors_data.source == "literature_initial_estimates"
    assert "f0_semitone_sd" in priors_data.features
    assert "jitter_local_pct" in priors_data.features
    assert len(priors_data.feature_order) == 8
