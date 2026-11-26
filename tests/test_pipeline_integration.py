"""Integration and end-to-end lifecycle tests for DiscourseConsistencyPipeline.

Verifies:
1. Real-time streaming chunk ingestion, state transitions, and event emission.
2. Turn commit execution: automated acoustic extraction, timing assembly,
   baseline scoring, and regularized Block-Diagonal Mahalanobis computation.
3. Provisional speaker promotion triggering baseline and claims ledger merging.
4. End-of-file (EOF) buffer flush emitting trailing turn records.
5. JSONL recording integration through FeatureRecorder and bit-exact replay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from dce.clock import ClockMode
from dce.pipeline import DiscourseConsistencyPipeline
from dce.recorder import ReplayLoader
from dce.schemas import (
    AppConfig,
    ClockConfig,
    DirectionLabel,
    VadConfig,
    VisionFeatures,
)


def _generate_pcm_chunk(samples: int = 320, amplitude: float = 0.1) -> List[float]:
    """Generate synthetic PCM chunk of float32 samples."""
    return [amplitude] * samples


def test_pipeline_streaming_turn_lifecycle(tmp_path: Path) -> None:
    """Test full turn lifecycle from speech onset to hard EOT commit and JSONL recording."""
    record_file = tmp_path / "stream_session.jsonl"
    cfg = AppConfig(
        mode=ClockMode.OFFLINE,
        vad=VadConfig(
            speech_prob_on=0.55,
            speech_prob_off=0.35,
            min_speech_ms=160.0,
            soft_pause_ms=400.0,
            hard_eot_ms=800.0,
            min_speculative_ms=600.0,
        ),
    )

    pipeline = DiscourseConsistencyPipeline(config=cfg, recorder_path=record_file)
    pcm_speech = _generate_pcm_chunk(320, amplitude=0.08)
    pcm_silence = _generate_pcm_chunk(320, amplitude=0.0)

    # 1. Feed 40 speech chunks (40 * 20ms = 800ms)
    pts_ns = 0
    for _ in range(40):
        events, rec = pipeline.process_chunk(
            audio_pcm=pcm_speech,
            vad_prob=0.85,
            pts_ns=pts_ns,
        )
        assert rec is None
        pts_ns += 20_000_000

    # 2. Feed 45 silence chunks (45 * 20ms = 900ms) to trigger 800ms hard EOT
    committed_record = None
    sample_vision = VisionFeatures(
        blink_rate_per_min=18.0,
        mean_lip_aperture=0.25,
        face_detected=True,
        quality_flag="ok",
    )
    for _ in range(45):
        events, rec = pipeline.process_chunk(
            audio_pcm=pcm_silence,
            vad_prob=0.10,
            pts_ns=pts_ns,
            vision_features=sample_vision,
        )
        if rec is not None:
            committed_record = rec
        pts_ns += 20_000_000

    assert committed_record is not None, "Pipeline failed to commit turn after hard EOT!"
    assert committed_record.turn_id == 1
    assert committed_record.speaker_assignment is not None
    assert committed_record.features is not None
    assert committed_record.mahalanobis is not None
    assert committed_record.mahalanobis.degrees_of_freedom > 0

    # Verify pipeline state tracking
    assert len(pipeline.completed_turns) == 1
    assert pipeline.completed_turns[0].turn_id == 1

    # 3. Clean close and verify JSONL file written
    pipeline.close()
    assert record_file.is_file()

    loader = ReplayLoader(record_file)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].turn_id == 1
    assert records[0].speaker_assignment.speaker_id == committed_record.speaker_assignment.speaker_id


def test_pipeline_eof_flush_commits_trailing_speech() -> None:
    """Test pipeline.flush() at EOF commits pending audio buffer exceeding minimum speech."""
    cfg = AppConfig(
        vad=VadConfig(
            speech_prob_on=0.55,
            min_speech_ms=160.0,
            hard_eot_ms=800.0,
        )
    )
    pipeline = DiscourseConsistencyPipeline(config=cfg)
    pcm_speech = _generate_pcm_chunk(320, amplitude=0.08)

    # Feed 20 speech chunks (400ms speech), no trailing silence
    pts_ns = 0
    for _ in range(20):
        pipeline.process_chunk(
            audio_pcm=pcm_speech,
            vad_prob=0.80,
            pts_ns=pts_ns,
        )
        pts_ns += 20_000_000

    # Stream ends abruptly without silence: call flush()
    flushed_record = pipeline.flush()
    assert flushed_record is not None
    assert flushed_record.turn_id == 1
    assert flushed_record.features.timing.speech_duration_ms >= 380.0
    assert len(pipeline.completed_turns) == 1

    pipeline.close()


def test_pipeline_provisional_promotion_synchronization() -> None:
    """Test provisional speaker promotion synchronizes BaselineStore and ClaimsLedger."""
    pipeline = DiscourseConsistencyPipeline()
    pcm_speech = _generate_pcm_chunk(320, amplitude=0.05)
    pcm_silence = _generate_pcm_chunk(320, amplitude=0.0)

    # Normalized fixed speaker embedding
    spk_emb = np.zeros(192, dtype=np.float64)
    spk_emb[0] = 1.0

    pts_ns = 0

    # Turn 1: Short turn (25 chunks * 20ms = 500ms speech) -> creates provisional speaker
    for _ in range(25):
        pipeline.process_chunk(pcm_speech, vad_prob=0.85, pts_ns=pts_ns)
        pts_ns += 20_000_000

    sample_vision = VisionFeatures(
        blink_rate_per_min=18.0,
        mean_lip_aperture=0.25,
        face_detected=True,
        quality_flag="ok",
    )

    rec1 = None
    for _ in range(45):
        _, r = pipeline.process_chunk(
            pcm_silence,
            vad_prob=0.10,
            pts_ns=pts_ns,
            voice_embedding=spk_emb,
            vision_features=sample_vision,
        )
        if r is not None:
            rec1 = r
        pts_ns += 20_000_000

    assert rec1 is not None
    assert rec1.speaker_assignment.is_provisional is True
    prov_id = rec1.speaker_assignment.speaker_id
    assert prov_id.startswith("PROV_")

    # Turn 2: Long turn (120 chunks * 20ms = 2400ms >= min_confirm_sec=2.0s) -> promotes to SPEAKER_00
    for _ in range(120):
        pipeline.process_chunk(pcm_speech, vad_prob=0.85, pts_ns=pts_ns)
        pts_ns += 20_000_000

    rec2 = None
    for _ in range(45):
        _, r = pipeline.process_chunk(
            pcm_silence,
            vad_prob=0.10,
            pts_ns=pts_ns,
            voice_embedding=spk_emb,
            vision_features=sample_vision,
        )
        if r is not None:
            rec2 = r
        pts_ns += 20_000_000

    assert rec2 is not None
    assert rec2.speaker_assignment.is_provisional is False
    assert rec2.speaker_assignment.promoted_from == prov_id
    perm_id = rec2.speaker_assignment.speaker_id
    assert perm_id.startswith("SPEAKER_")

    # Verify BaselineStore merged prov_id into perm_id
    assert prov_id not in pipeline.baseline_store._history
    assert perm_id in pipeline.baseline_store._history

    pipeline.close()


def _run_turn(
    pipeline: DiscourseConsistencyPipeline,
    start_ns: int,
    speech_chunks: int = 20,
    silence_chunks: int = 45,
    transcript_text: str = "Hello.",
    vision_features: VisionFeatures | None = None,
):
    """Helper to feed one full turn of speech + hard EOT silence and return (record, next_pts_ns)."""
    pcm_speech = _generate_pcm_chunk(320, amplitude=0.08)
    pcm_silence = _generate_pcm_chunk(320, amplitude=0.0)
    pts_ns = start_ns
    for _ in range(speech_chunks):
        pipeline.process_chunk(pcm_speech, vad_prob=0.85, pts_ns=pts_ns)
        pts_ns += 20_000_000

    committed = None
    for _ in range(silence_chunks):
        _, rec = pipeline.process_chunk(
            pcm_silence,
            vad_prob=0.10,
            pts_ns=pts_ns,
            transcript_text=transcript_text,
            vision_features=vision_features,
        )
        if rec is not None:
            committed = rec
        pts_ns += 20_000_000
    return committed, pts_ns


def test_pipeline_metamorphic_response_latency_and_question_classification() -> None:
    """Verify open/closed latency classification from prior turn and metamorphic gap sensitivity."""
    from unittest.mock import MagicMock
    from dce.reasoning.claims_ledger import ClaimsLedger
    from dce.reasoning.llm_adapter import LLMAdapter
    from dce.schemas import ClaimExtractionResponse

    cfg = AppConfig(
        mode=ClockMode.OFFLINE,
        vad=VadConfig(
            speech_prob_on=0.55,
            speech_prob_off=0.35,
            min_speech_ms=160.0,
            soft_pause_ms=400.0,
            hard_eot_ms=800.0,
            min_speculative_ms=600.0,
        ),
    )
    pipeline = DiscourseConsistencyPipeline(config=cfg)
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ClaimExtractionResponse(claims=[])
    pipeline._claims_ledger = ClaimsLedger(llm_adapter=mock_llm)

    # Turn 1: Open wh-question
    r1, pts = _run_turn(pipeline, start_ns=0, transcript_text="Where did you go yesterday?")
    assert r1 is not None
    assert r1.features.timing.response_latency_open is None
    assert r1.features.timing.response_latency_closed is None

    # Turn 2: Answer after 500ms gap -> response_latency_open ~ 0.5s
    r2, pts = _run_turn(
        pipeline,
        start_ns=r1.features.end_ns + 500_000_000,
        transcript_text="Did you check the server room?",
    )
    assert r2 is not None
    assert r2.features.timing.response_latency_open is not None
    assert math.isclose(r2.features.timing.response_latency_open, 0.5, abs_tol=0.05)
    assert r2.features.timing.response_latency_closed is None

    # Turn 3: Answer to closed aux/modal question after 1500ms gap -> response_latency_closed ~ 1.5s
    r3, pts = _run_turn(
        pipeline,
        start_ns=r2.features.end_ns + 1_500_000_000,
        transcript_text="Yes, I checked it.",
    )
    assert r3 is not None
    assert r3.features.timing.response_latency_closed is not None
    assert math.isclose(r3.features.timing.response_latency_closed, 1.5, abs_tol=0.05)
    assert r3.features.timing.response_latency_open is None

    # Turn 4: Prior turn was NOT a question -> both latencies must be None
    r4, _ = _run_turn(
        pipeline,
        start_ns=r3.features.end_ns + 600_000_000,
        transcript_text="Okay.",
    )
    assert r4 is not None
    assert r4.features.timing.response_latency_open is None
    assert r4.features.timing.response_latency_closed is None

    pipeline.close()


def test_pipeline_metamorphic_vision_disabled_and_blink_transcript() -> None:
    """Verify disabling vision sets blink_rate_per_min=None and 'blink' in transcript does not crash."""
    from unittest.mock import MagicMock
    from dce.reasoning.claims_ledger import ClaimsLedger
    from dce.reasoning.llm_adapter import LLMAdapter
    from dce.schemas import ClaimExtractionResponse

    pipeline = DiscourseConsistencyPipeline()
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ClaimExtractionResponse(claims=[])
    pipeline._claims_ledger = ClaimsLedger(llm_adapter=mock_llm)
    rec, _ = _run_turn(
        pipeline,
        start_ns=0,
        transcript_text="I did not blink when the light flashed.",
        vision_features=None,
    )
    assert rec is not None
    assert rec.features.vision.blink_rate_per_min is None
    assert rec.features.vision.face_detected is False
    pipeline.close()
