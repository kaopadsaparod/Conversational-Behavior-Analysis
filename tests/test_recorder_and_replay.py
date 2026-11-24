"""Unit tests for FeatureRecorder and ReplayLoader deterministic serialization.

Verifies:
1. FeatureRecorder context manager entry and exit file safety.
2. Bit-exact serialization and round-trip deserialization via ReplayLoader.
3. ReplayLoader line-by-line streaming iteration and error handling on corrupted records.
4. FileNotFoundError handling on missing replay files.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dce.recorder import FeatureRecorder, ReplayLoader
from dce.schemas import (
    ProsodyFeatures,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    VisionFeatures,
)


def _make_dummy_turn(turn_id: int, speaker_id: str) -> RecorderTurnRecord:
    """Create a populated RecorderTurnRecord."""
    return RecorderTurnRecord(
        turn_id=turn_id,
        speaker_assignment=SpeakerAssignment(
            speaker_id=speaker_id,
            similarity=0.92,
            is_provisional=False,
            low_confidence=False,
        ),
        features=TurnFeatures(
            turn_id=turn_id,
            speaker_id=speaker_id,
            start_ns=turn_id * 1_000_000_000,
            end_ns=(turn_id + 1) * 1_000_000_000,
            timing=TimingFeatures(
                response_latency_open=0.35 + (turn_id * 0.05),
                turn_duration_ms=1000.0,
            ),
            prosody=ProsodyFeatures(
                f0_median_hz=145.0,
                articulation_rate=4.3,
            ),
            vision=VisionFeatures(
                blink_rate_per_min=16.0,
            ),
        ),
    )


def test_feature_recorder_context_manager_and_roundtrip(tmp_path: Path) -> None:
    """Verify FeatureRecorder context manager and full ReplayLoader roundtrip."""
    session_file = tmp_path / "replay_test.jsonl"

    record_1 = _make_dummy_turn(1, "SPEAKER_00")
    record_2 = _make_dummy_turn(2, "SPEAKER_01")

    with FeatureRecorder(session_file) as recorder:
        assert recorder.path == session_file
        recorder.write_record(record_1)
        recorder.write_record(record_2)
        recorder.flush()

    assert session_file.is_file()

    # Read back via ReplayLoader
    loader = ReplayLoader(session_file)
    assert loader.path == session_file

    loaded_records = loader.load_all()
    assert len(loaded_records) == 2

    r1 = loaded_records[0]
    assert r1.turn_id == 1
    assert r1.speaker_assignment.speaker_id == "SPEAKER_00"
    assert math.isclose(r1.features.timing.response_latency_open, 0.40, abs_tol=1e-6)

    r2 = loaded_records[1]
    assert r2.turn_id == 2
    assert r2.speaker_assignment.speaker_id == "SPEAKER_01"
    assert math.isclose(r2.features.timing.response_latency_open, 0.45, abs_tol=1e-6)


def test_replay_loader_streaming_and_empty_lines(tmp_path: Path) -> None:
    """Verify ReplayLoader skips empty lines and yields records sequentially."""
    test_file = tmp_path / "sparse.jsonl"
    r1 = _make_dummy_turn(1, "SPEAKER_00")

    # Write record with empty lines interspersed
    with test_file.open("w", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(r1.model_dump_json() + "\n")
        f.write("\n   \n")

    loader = ReplayLoader(test_file)
    streamed = list(loader)
    assert len(streamed) == 1
    assert streamed[0].turn_id == 1


def test_replay_loader_error_handling(tmp_path: Path) -> None:
    """Verify ReplayLoader raises FileNotFoundError and ValueError on malformed JSON."""
    missing_file = tmp_path / "non_existent.jsonl"
    with pytest.raises(FileNotFoundError):
        ReplayLoader(missing_file)

    corrupted_file = tmp_path / "corrupted.jsonl"
    corrupted_file.write_text("{\"turn_id\": 1, invalid_json\n", encoding="utf-8")

    loader = ReplayLoader(corrupted_file)
    with pytest.raises(ValueError, match="Failed parsing record at line 1"):
        list(loader)
