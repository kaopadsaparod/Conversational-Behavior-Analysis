"""Unit tests for Stage 1 Foundation First components.

Tests MasterClock, PolicyQueue, Pydantic schemas, default YAML configuration,
and strict anti-stub / NaN-safety guarantees.
"""

from __future__ import annotations

import ast
import math
import queue
import time
from pathlib import Path

import pytest
from dce.clock import ClockMode, MasterClock, PolicyQueue, QueuePolicy
from dce.schemas import (
    AppConfig,
    CancelSpeculativeEvent,
    ClaimExtractionResponse,
    ClaimItem,
    ConsistencyVerificationResponse,
    DirectionLabel,
    HardTurnCommitEvent,
    MahalanobisResult,
    PauseRecord,
    ProsodyFeatures,
    SpeakerAssignment,
    SpeculativeTriggerEvent,
    TurnFeatures,
    TurnTranscript,
    VADChunkResult,
    VADState,
    VerdictStatus,
    VisionFeatures,
    load_config,
    sanitize_nan_inf,
)


def test_default_config_loading() -> None:
    """Verify default.yaml loads completely and matches all architectural requirements."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    cfg = load_config(config_path)

    assert cfg.mode == "offline"
    assert cfg.clock.av_offset_ns == 0
    assert cfg.queues.offline_policy == "block"
    assert cfg.queues.live_vision_policy == "drop_oldest"
    assert cfg.queues.live_audio_policy == "never_drop"
    assert cfg.queues.max_size == 128

    assert cfg.hardware.asr_device == "cpu"
    assert cfg.hardware.llm_provider == "ollama"
    assert cfg.hardware.ollama_model == "qwen2.5:7b-instruct-q4_K_M"
    assert cfg.hardware.ollama_fallback_model == "qwen2.5:3b-instruct-q4_K_M"
    assert cfg.hardware.ollama_num_ctx == 4096
    assert cfg.hardware.gemini_model == "gemini-2.5-flash-lite"

    assert cfg.vad.sample_rate == 16000
    assert cfg.vad.speech_prob_on == 0.55
    assert cfg.vad.speech_prob_off == 0.35
    assert cfg.vad.min_speech_ms == 160.0
    assert cfg.vad.min_micro_pause_ms == 200.0
    assert cfg.vad.soft_pause_ms == 400.0
    assert cfg.vad.hard_eot_ms == 800.0
    assert cfg.vad.min_speculative_ms == 600.0
    assert cfg.vad.max_turn_ms == 25000.0
    assert cfg.vad.overlap_ratio_threshold == 0.15

    assert cfg.prosody.creak_trim_min_speech_ms == 600.0
    assert cfg.prosody.creak_trim_tail_ms == 200.0
    assert cfg.prosody.min_voiced_segment_ms == 75.0
    assert cfg.prosody.boundary_trim_ms == 15.0
    assert cfg.prosody.min_glottal_periods == 20
    assert cfg.prosody.default_pitch_floor_hz == 75.0
    assert cfg.prosody.pitch_floor_min_hz == 60.0
    assert cfg.prosody.pitch_floor_max_hz == 140.0

    assert cfg.baseline.rolling_window_turns == 40
    assert cfg.baseline.shrinkage_k == 6.0
    assert cfg.baseline.anchor_lock_turns == 8
    assert cfg.baseline.gate_full_z == 3.0
    assert cfg.baseline.gate_reduced_z == 5.0
    assert cfg.baseline.gate_reduced_weight == 0.2

    assert cfg.mahalanobis.reg_lambda == 0.30
    assert cfg.mahalanobis.eig_floor == 1.0e-6
    assert cfg.mahalanobis.cond_limit == 1.0e+12
    assert cfg.mahalanobis.tail_score_threshold == 0.05
    assert cfg.mahalanobis.tension_z_threshold == 1.5
    assert cfg.mahalanobis.hnr_tension_z_threshold == -1.5
    assert cfg.mahalanobis.calm_z_threshold == -1.5

    assert cfg.av_sync.grid_hz == 50.0
    assert cfg.av_sync.min_overlap_sec == 0.5
    assert cfg.av_sync.max_lag_ms == 200.0
    assert cfg.av_sync.min_std == 1.0e-6

    assert cfg.speaker_registry.sim_high == 0.65
    assert cfg.speaker_registry.sim_low == 0.45
    assert cfg.speaker_registry.face_sim_floor == 0.35
    assert cfg.speaker_registry.min_confirm_sec == 2.0
    assert cfg.speaker_registry.provisional_ttl_sec == 60.0
    assert cfg.speaker_registry.max_speakers == 4
    assert cfg.speaker_registry.max_provisional_buffers == 6
    assert cfg.speaker_registry.min_embedding_norm == 1.0e-8

    assert cfg.reasoning.anomaly_guided_search is False
    assert cfg.reasoning.max_llm_retries == 1


def test_master_clock_offline_mode() -> None:
    """Test MasterClock offline mode PTS stepping and conversions."""
    clock = MasterClock(mode=ClockMode.OFFLINE)
    assert clock.mode == ClockMode.OFFLINE
    assert clock.get_time_ns() == 0
    assert clock.get_pts_ns() == 0
    assert clock.get_time_sec() == 0.0

    assert clock.advance_to_pts(500_000_000) == 500_000_000
    assert clock.get_time_ns() == 500_000_000
    assert clock.get_time_sec() == 0.5

    # advancing to earlier pts does not regress monotonic pts
    assert clock.advance_to_pts(400_000_000) == 500_000_000
    assert clock.get_time_ns() == 500_000_000

    clock.reset(start_pts_ns=1_000_000_000)
    assert clock.get_time_ns() == 1_000_000_000
    assert clock.get_time_sec() == 1.0


def test_master_clock_live_mode() -> None:
    """Test MasterClock live mode monotonic timing with A/V offset."""
    clock = MasterClock(mode="live", av_offset_ns=50_000_000)
    assert clock.mode == ClockMode.LIVE
    assert clock.av_offset_ns == 50_000_000

    t1 = clock.get_time_ns()
    time.sleep(0.005)
    t2 = clock.get_time_ns()
    assert t2 > t1

    clock.set_av_offset_ns(100_000_000)
    assert clock.av_offset_ns == 100_000_000


def test_policy_queue_block() -> None:
    """Test PolicyQueue block policy when full and when empty."""
    q = PolicyQueue(max_size=2, policy="block")
    assert q.max_size == 2
    assert q.policy == QueuePolicy.BLOCK
    assert q.empty()

    assert q.put("first") is True
    assert q.put("second") is True
    assert q.full() is True

    # put_nowait raises queue.Full when at capacity
    with pytest.raises(queue.Full):
        q.put_nowait("third")

    with pytest.raises(queue.Full):
        q.put("third", block=True, timeout=0.01)

    assert q.get() == "first"
    assert q.get() == "second"
    assert q.empty() is True

    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_policy_queue_drop_oldest() -> None:
    """Test PolicyQueue drop_oldest policy discarding oldest element on overflow."""
    q = PolicyQueue(max_size=3, policy="drop_oldest")
    assert q.dropped_count == 0

    for i in range(5):
        q.put(i)

    assert q.dropped_count == 2
    assert q.qsize() == 3
    # Items 0 and 1 dropped, remaining should be 2, 3, 4
    assert q.get() == 2
    assert q.get() == 3
    assert q.get() == 4
    assert q.empty() is True


def test_policy_queue_never_drop() -> None:
    """Test PolicyQueue never_drop policy expanding capacity indefinitely."""
    q = PolicyQueue(max_size=2, policy="never_drop")

    for i in range(10):
        q.put(i)

    assert q.qsize() == 10
    assert q.full() is False
    assert q.dropped_count == 0
    assert q.get() == 0
    assert q.clear() == 9
    assert q.empty() is True


def test_schemas_nan_inf_sanitization() -> None:
    """Verify that NaN and Inf floats are sanitized to None across schemas."""
    prosody = ProsodyFeatures(
        f0_median_hz=float("nan"),
        f0_semitone_sd=float("inf"),
        jitter_local_pct=float("-inf"),
        shimmer_local_pct=3.14,
        hnr_db=None,
    )
    assert prosody.f0_median_hz is None
    assert prosody.f0_semitone_sd is None
    assert prosody.jitter_local_pct is None
    assert prosody.shimmer_local_pct == 3.14
    assert prosody.hnr_db is None

    # Serialization must not produce NaN or Infinity
    json_output = prosody.model_dump_json()
    assert "NaN" not in json_output
    assert "Infinity" not in json_output


def test_schemas_enums_and_contracts() -> None:
    """Verify enum states and event schema contracts."""
    assert DirectionLabel.NORMAL == "normal"
    assert DirectionLabel.COGNITIVE_LOAD_ANOMALY == "cognitive_load_anomaly"
    assert DirectionLabel.CALM_DEVIATION == "calm_deviation"
    assert DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION == "mixed_subthreshold_deviation"

    assert VerdictStatus.CONSISTENT == "CONSISTENT"
    assert VerdictStatus.DIRECT_CONTRADICTION == "DIRECT_CONTRADICTION"
    assert VerdictStatus.TIMELINE_SHIFT == "TIMELINE_SHIFT"
    assert VerdictStatus.UNVERIFIABLE == "UNVERIFIABLE"
    assert VerdictStatus.REJECTED_INVALID_REF == "REJECTED_INVALID_REF"

    # Event instantiation
    pause = PauseRecord(start_ns=1000, duration_ms=250.0)
    assert pause.start_ns == 1000
    assert pause.duration_ms == 250.0

    trigger = SpeculativeTriggerEvent(
        turn_id=1,
        epoch_id=1,
        start_ns=100,
        last_voiced_ns=700,
    )
    assert trigger.turn_id == 1
    assert trigger.epoch_id == 1

    cancel = CancelSpeculativeEvent(
        turn_id=1,
        invalidated_epoch_id=1,
        hesitation_pause=pause,
    )
    assert cancel.invalidated_epoch_id == 1
    assert cancel.hesitation_pause.duration_ms == 250.0

    commit = HardTurnCommitEvent(
        turn_id=1,
        start_ns=100,
        end_ns=900,
        voiced_chunks=10,
        overlap_voiced_chunks=2,
        overlap_ratio=0.2,
        overlap_flag=True,
        forced_split=False,
        intra_turn_pauses=[pause],
    )
    assert commit.overlap_flag is True
    assert len(commit.intra_turn_pauses) == 1

    # Speaker Assignment
    assign = SpeakerAssignment(
        speaker_id="SPEAKER_00",
        similarity=0.88,
        is_provisional=False,
        low_confidence=False,
    )
    assert assign.speaker_id == "SPEAKER_00"

    # Reasoning verification response
    resp = ConsistencyVerificationResponse(
        verdict=VerdictStatus.REJECTED_INVALID_REF,
        source_turn_id=3,
        prior_turn_id=999,
        explanation="Prior turn does not exist in ledger.",
    )
    assert resp.verdict == VerdictStatus.REJECTED_INVALID_REF


def test_anti_stub_compliance_stage1() -> None:
    """AST check verifying zero stubbing in Stage 1 source files."""
    pkg_dir = Path(__file__).resolve().parent.parent / "src" / "dce"
    for filename in ["clock.py", "schemas.py", "__init__.py"]:
        path = pkg_dir / filename
        source = path.read_text(encoding="utf-8")
        assert "TODO" not in source, f"Found TODO in {filename}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in node.body:
                    assert not isinstance(child, ast.Pass), f"Found pass in {filename}:{node.name}"
                    if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                        assert child.value.value is not Ellipsis, f"Found Ellipsis in {filename}:{node.name}"
                    if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Name):
                        assert child.exc.id != "NotImplementedError", f"Found NotImplementedError in {filename}:{node.name}"
