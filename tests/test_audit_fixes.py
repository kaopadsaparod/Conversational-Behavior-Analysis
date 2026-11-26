"""Regression tests for all P0, P1, and P2 audit fixes."""

from __future__ import annotations

import math
import sys
import types
from typing import Any, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmarks.run_ablation import evaluate_loso
from dce.baseline_store import BaselineStore, get_default_priors
from dce.clock import ClockMode
from dce.extractors.audio_dsp import transcribe_audio
from dce.extractors.vision_face import estimate_head_pose_euler
from dce.math_core import (
    classify_direction,
    compute_mahalanobis,
    normalized_av_xcorr,
    weighted_median,
)
from dce.pipeline import DiscourseConsistencyPipeline
from dce.reasoning.claims_ledger import (
    ClaimsLedger,
    assert_firewall_compliance,
    should_skip_extraction,
)
from dce.reasoning.llm_adapter import LLMAdapter, sanitize_gemini_schema
from dce.schemas import (
    AppConfig,
    ClaimExtractionResponse,
    ClaimItem,
    ClaimPolarity,
    ConsistencyVerificationResponse,
    DirectionLabel,
    HardwareConfig,
    ProsodyFeatures,
    ReasoningConfig,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    TurnTranscript,
    VadConfig,
    VerdictStatus,
    VisionFeatures,
)


# ============================================================================
# P0 Tests: Math Core, BaselineStore, ClaimsLedger Firewall & Ref Validation
# ============================================================================


def test_normalized_av_xcorr_positive_and_negative_lags_100ms() -> None:
    """Verify R_ve(tau) = corr(v(t), e(t + tau)) for +100ms and -100ms lags."""
    t = np.linspace(0.0, 2.0, 201)  # 100 Hz input resampled to 50 Hz
    v = np.sin(2.0 * np.pi * 1.5 * t)

    # Case 1: Audio lags video by +100ms -> e(t) = v(t - 0.1), so e(t + 0.1) = v(t)
    e_lagged = np.sin(2.0 * np.pi * 1.5 * (t - 0.1))
    corr_pos, lag_pos = normalized_av_xcorr(t, v, t, e_lagged, max_lag_ms=200.0)
    assert corr_pos is not None and lag_pos is not None
    assert corr_pos > 0.99
    assert math.isclose(lag_pos, 100.0, abs_tol=1.0)

    # Case 2: Audio leads video by 100ms -> e(t) = v(t + 0.1), so e(t - 0.1) = v(t)
    e_leading = np.sin(2.0 * np.pi * 1.5 * (t + 0.1))
    corr_neg, lag_neg = normalized_av_xcorr(t, v, t, e_leading, max_lag_ms=200.0)
    assert corr_neg is not None and lag_neg is not None
    assert corr_neg > 0.99
    assert math.isclose(lag_neg, -100.0, abs_tol=1.0)


def test_compute_mahalanobis_indefinite_nonfinite_and_overflow_guards() -> None:
    """Verify indefinite matrix triggers eigen fallback, non-finite corr is sanitized, and d_sq never overflows."""
    priors = get_default_priors()
    order = ["jitter_local_pct", "shimmer_local_pct"]

    # Indefinite matrix with well-conditioned eigenvalues (+4 and -2, cond=2 <= 1e12)
    indef_corr = np.array([[1.0, 3.0], [3.0, 1.0]], dtype=np.float64)
    res_indef = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 1.0, "shimmer_local_pct": 1.0},
        correlation_matrix=indef_corr,
        feature_order=order,
        priors=priors,
        reg_lambda=0.0,
    )
    assert res_indef.eigen_fallback_used is True
    assert math.isfinite(res_indef.d_sq) and res_indef.d_sq >= 0.0

    # Non-finite correlation matrix
    nan_corr = np.array([[1.0, float("nan")], [float("inf"), 1.0]], dtype=np.float64)
    res_nan = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 2.0, "shimmer_local_pct": 1.0},
        correlation_matrix=nan_corr,
        feature_order=order,
        priors=priors,
    )
    assert math.isfinite(res_nan.d_sq)
    assert math.isfinite(res_nan.chi2_tail_score)

    # Overflow guard on huge z-scores
    res_huge = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 1e200, "shimmer_local_pct": 1e200},
        correlation_matrix=np.eye(2),
        feature_order=order,
        priors=priors,
    )
    assert math.isfinite(res_huge.d_sq)
    assert math.isfinite(res_huge.mahalanobis_distance)
    assert res_huge.chi2_tail_score == 0.0


def test_baseline_store_quality_flag_gating_excludes_degraded_channels() -> None:
    """Verify quality_flag != 'ok' returns z_score=None and excludes channel from Mahalanobis."""
    store = BaselineStore()
    bad_score = store.score_feature("SPK_01", "jitter_local_pct", 2.5, quality_flag="no_audio_backend")
    assert bad_score.z_score is None
    assert bad_score.confidence_weight == 0.0

    tf = TurnFeatures(
        turn_id=1,
        speaker_id="SPK_01",
        start_ns=0,
        end_ns=1_000_000_000,
        timing=TimingFeatures(response_latency_open=0.5),
        prosody=ProsodyFeatures(jitter_local_pct=2.5, quality_flag="insufficient_data"),
        vision=VisionFeatures(blink_rate_per_min=18.0, quality_flag="extreme_head_pose"),
    )
    scores = store.score_and_update_turn("SPK_01", tf)
    assert scores["response_latency_open"].z_score is not None
    assert scores["jitter_local_pct"].z_score is None
    assert scores["blink_rate_per_min"].z_score is None

    m_res = compute_mahalanobis(
        feature_scores=scores,
        correlation_matrix=np.eye(8),
        feature_order=list(store.priors.keys()),
        priors=store.priors,
    )
    assert m_res.evaluated_features == ["response_latency_open"]
    assert m_res.degrees_of_freedom == 1


def test_firewall_catches_underscores_and_allows_clean_speech() -> None:
    """Verify two-sided firewall fix: catches underscore identifiers and allows 'blink'/'latency' in speech."""
    for bad in ["jitter_local_pct", "f0_semitone_sd", "z_score", "semitones", "mean_hnr_db"]:
        with pytest.raises(ValueError, match="Neuro-Symbolic Firewall violation"):
            assert_firewall_compliance(f"Metric {bad} was elevated.")

    # Speech containing 'blink' or 'latency' must NOT crash extract_claims
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ClaimExtractionResponse(claims=[])
    ledger = ClaimsLedger(llm_adapter=mock_llm)

    t = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="I didn't even blink when I saw the network latency spike.",
        start_ns=0,
        end_ns=1_000_000,
    )
    claims = ledger.extract_claims(t)
    assert claims == []
    assert mock_llm.generate_structured.call_count == 1

    # Framing leak must still raise ValueError
    t_bad_framing = TurnTranscript(
        turn_id=2,
        speaker_id="jitter_local_pct",
        text="I went home.",
        start_ns=0,
        end_ns=1_000_000,
    )
    with pytest.raises(ValueError, match="Neuro-Symbolic Firewall violation"):
        ledger.extract_claims(t_bad_framing)


def test_verify_consistency_hardened_reference_validation() -> None:
    """Verify source_turn_id, prior_turn_id < current, required refs, and prior_claim_id membership."""
    mock_llm = MagicMock(spec=LLMAdapter)
    ledger = ClaimsLedger(llm_adapter=mock_llm)

    c1 = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPK_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="office",
        time_ref_raw="9 AM",
    )
    c2 = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPK_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="9 AM",
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    # 1. Wrong source_turn_id
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.CONSISTENT,
        source_turn_id=99,
        prior_turn_id=1,
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0001_01",
        explanation="Wrong source turn.",
    )
    assert ledger.verify_consistency(c2).verdict == VerdictStatus.REJECTED_INVALID_REF

    # 2. prior_turn_id >= current_claim.turn_id
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.CONSISTENT,
        source_turn_id=2,
        prior_turn_id=2,
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0002_01",
        explanation="Self-reference.",
    )
    assert ledger.verify_consistency(c2).verdict == VerdictStatus.REJECTED_INVALID_REF

    # 3. Missing prior_turn_id on DIRECT_CONTRADICTION
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=None,
        source_claim_id="CLM_0002_01",
        prior_claim_id=None,
        explanation="Missing prior ref.",
    )
    assert ledger.verify_consistency(c2).verdict == VerdictStatus.REJECTED_INVALID_REF

    # 4. Unknown prior_claim_id
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.TIMELINE_SHIFT,
        source_turn_id=2,
        prior_turn_id=1,
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_FAKE_99",
        explanation="Fake claim id.",
    )
    assert ledger.verify_consistency(c2).verdict == VerdictStatus.REJECTED_INVALID_REF


# ============================================================================
# P1 & P2 Tests: Pipeline, Ledger Edge Cases, Audio/Vision, Schema, Ablation
# ============================================================================


def test_calm_z_threshold_and_name_based_group_map() -> None:
    """Verify calm_z_threshold wiring and name-based group_map with permuted feature_order."""
    priors = get_default_priors()
    # z=-1.6 is calm under threshold -1.5, but mixed under -2.0
    _, lbl_default = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": -1.6},
        priors=priors,
        calm_z_threshold=-1.5,
    )
    assert lbl_default == DirectionLabel.CALM_DEVIATION

    _, lbl_strict = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": -1.6},
        priors=priors,
        calm_z_threshold=-2.0,
    )
    assert lbl_strict == DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION

    # Permuted order: [vision, voice, timing] -> cross-group corr must be zeroed by name
    perm_order = ["blink_rate_per_min", "jitter_local_pct", "response_latency_open"]
    full_corr = np.full((3, 3), 0.8, dtype=np.float64)
    np.fill_diagonal(full_corr, 1.0)
    res_perm = compute_mahalanobis(
        feature_scores={"blink_rate_per_min": 1.0, "jitter_local_pct": 1.0, "response_latency_open": 1.0},
        correlation_matrix=full_corr,
        feature_order=perm_order,
        priors=priors,
        reg_lambda=0.0,
    )
    # Since all 3 are in different groups, Sigma_V == I_3, so d_sq == 1^2 + 1^2 + 1^2 == 3.0
    assert math.isclose(res_perm.d_sq, 3.0, rel_tol=1e-6)


def test_weighted_median_small_weights_no_false_tie() -> None:
    """Verify small weights (1e-13) do not trigger false 0.5 tie due to abs_tol."""
    val = weighted_median([10.0, 20.0], [0.4e-12, 0.6e-12])
    assert val == 20.0

    # Exact 0.5 tie at small scale still interpolates
    tie_val = weighted_median([10.0, 20.0], [0.5e-12, 0.5e-12])
    assert tie_val == 15.0


def test_claims_ledger_alias_time_matching_and_skip_edge_cases() -> None:
    """Verify whitespace collapse in register_alias, time_ref candidate matching, and skip gate."""
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.CONSISTENT,
        source_turn_id=2,
        prior_turn_id=1,
        source_claim_id="CLM_2",
        prior_claim_id="CLM_1",
        explanation="Same time ref.",
    )
    ledger = ClaimsLedger(llm_adapter=mock_llm)
    ledger.register_alias("  The   Server   Room  ", "  server   room  ")

    c1 = ClaimItem(
        claim_id="CLM_1",
        turn_id=1,
        speaker_id="SPK_00",
        raw_subject="The Server Room",
        canonical_entity="The   Server   Room",
        predicate="accessed",
        object_value="yes",
        time_ref_raw="yesterday at 3pm",
        time_ref_normalized="2026-11-25T15:00",
    )
    ledger.add_claim(c1)
    assert c1.canonical_entity == "server room"

    # Different entity, matching time_ref_normalized
    c2 = ClaimItem(
        claim_id="CLM_2",
        turn_id=2,
        speaker_id="SPK_00",
        raw_subject="Bob",
        canonical_entity="bob",
        predicate="left",
        object_value="building",
        time_ref_raw="3pm yesterday",
        time_ref_normalized="2026-11-25T15:00",
    )
    ledger.add_claim(c2)
    res = ledger.verify_consistency(c2)
    assert res.verdict == VerdictStatus.CONSISTENT

    # should_skip_extraction edge cases
    assert should_skip_extraction(TurnTranscript(turn_id=1, speaker_id="S", text="London.", start_ns=0, end_ns=1)) is False
    assert should_skip_extraction(TurnTranscript(turn_id=2, speaker_id="S", text="Yeah I'm", start_ns=0, end_ns=1)) is True
    assert should_skip_extraction(TurnTranscript(turn_id=3, speaker_id="S", text="I'm", start_ns=0, end_ns=1)) is True


def test_llm_adapter_inline_defs_preserves_siblings_and_fallback_model() -> None:
    """Verify _inline_defs preserves sibling keys and LLMAdapter uses ollama_fallback_model on retry."""
    raw_schema = {
        "$defs": {"Sub": {"type": "string"}},
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Sub", "description": "Sibling description"}
        },
    }
    sanitized = sanitize_gemini_schema(raw_schema)
    assert sanitized["properties"]["item"]["type"] == "string"
    assert sanitized["properties"]["item"]["description"] == "Sibling description"

    hw = HardwareConfig(ollama_model="primary-model", ollama_fallback_model="fallback-model")
    rc = ReasoningConfig(max_llm_retries=1)
    adapter = LLMAdapter(reasoning_config=rc, hardware_config=hw)
    seen_models: List[str] = []

    def fake_ollama(prompt, response_model, system_prompt):
        seen_models.append(adapter._active_ollama_model)
        if len(seen_models) == 1:
            raise RuntimeError("Primary failed")
        return ClaimExtractionResponse(claims=[])

    adapter._generate_ollama = fake_ollama  # type: ignore
    res = adapter.generate_structured("test", ClaimExtractionResponse)
    assert isinstance(res, ClaimExtractionResponse)
    assert seen_models == ["primary-model", "fallback-model"]


def test_pipeline_cancel_speculative_priors_context_and_verifications() -> None:
    """Verify CancelSpeculativeEvent, priors_en.json loading, context_turns, and multiple verifications."""
    cfg = AppConfig(
        mode=ClockMode.OFFLINE,
        vad=VadConfig(
            speech_prob_on=0.55,
            speech_prob_off=0.35,
            min_speech_ms=100.0,
            soft_pause_ms=200.0,
            hard_eot_ms=400.0,
            min_speculative_ms=200.0,
        ),
    )
    pipeline = DiscourseConsistencyPipeline(config=cfg)

    # Default correlation matrix loaded from data/priors_en.json (non-identity)
    assert not np.allclose(pipeline._correlation_matrix, np.eye(8))

    # CancelSpeculativeEvent removes matching epoch from mailbox
    pcm_speech = [0.05] * 320
    pcm_silence = [0.0] * 320
    pts_ns = 0
    for _ in range(15):  # 300ms speech
        pipeline.process_chunk(pcm_speech, vad_prob=0.85, pts_ns=pts_ns)
        pts_ns += 20_000_000
    for _ in range(11):  # 220ms silence -> triggers SpeculativeTriggerEvent
        pipeline.process_chunk(pcm_silence, vad_prob=0.10, pts_ns=pts_ns)
        pts_ns += 20_000_000
    # Resume speech -> triggers CancelSpeculativeEvent and discards from mailbox
    pipeline.process_chunk(pcm_speech, vad_prob=0.85, pts_ns=pts_ns)
    assert pipeline._mailbox.pop_latest() is None

    # Mock LLM to verify context_turns and multiple verifications preservation
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.side_effect = [
        ClaimExtractionResponse(
            claims=[
                ClaimItem(claim_id="C1", turn_id=1, speaker_id="S", raw_subject="A", canonical_entity="a", predicate="p", object_value="1"),
                ClaimItem(claim_id="C2", turn_id=1, speaker_id="S", raw_subject="B", canonical_entity="b", predicate="q", object_value="2"),
            ]
        ),
        ConsistencyVerificationResponse(verdict=VerdictStatus.UNVERIFIABLE, source_turn_id=1, explanation="v1"),
        ConsistencyVerificationResponse(verdict=VerdictStatus.UNVERIFIABLE, source_turn_id=1, explanation="v2"),
    ]
    pipeline._claims_ledger = ClaimsLedger(llm_adapter=mock_llm)

    # Flush trailing speech with transcript_text
    rec = pipeline.flush(transcript_text="Alice went to London and Bob stayed home.")
    assert rec is not None
    assert len(rec.claims) == 2
    assert len(rec.verifications) == 2
    assert rec.features.vision.quality_flag == "unavailable"
    assert rec.features.vision.blink_rate_per_min is None
    assert rec.features.timing.response_latency_open is None
    pipeline.close()


def test_transcribe_audio_hallucination_filtering_and_head_pose_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify transcribe_audio uses word_timestamps=True and filters hallucinations, and head-pose gate works."""
    captured_kwargs = {}

    class FakeSegment:
        def __init__(self, text, no_speech_prob=0.1, avg_logprob=-0.2, compression_ratio=1.2):
            self.text = text
            self.no_speech_prob = no_speech_prob
            self.avg_logprob = avg_logprob
            self.compression_ratio = compression_ratio

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                FakeSegment("Valid speech."),
                FakeSegment("Hallucinated silence", no_speech_prob=0.95),
                FakeSegment("Low logprob", avg_logprob=-1.5),
                FakeSegment("Repeated text", compression_ratio=3.0),
            ], None

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    out = transcribe_audio([0.1] * 1600)
    assert captured_kwargs.get("word_timestamps") is True
    assert out == "Valid speech."

    # Test estimate_head_pose_euler with mocked cv2
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.SOLVEPNP_ITERATIVE = 0  # type: ignore[attr-defined]
    fake_cv2.solvePnP = lambda *args, **kwargs: (True, np.zeros((3, 1)), np.zeros((3, 1)))  # type: ignore[attr-defined]
    fake_cv2.Rodrigues = lambda rvec: (np.eye(3), None)  # type: ignore[attr-defined]
    fake_cv2.RQDecomp3x3 = lambda rmat: ((5.0, 35.0, 0.0), None, None, None, None, None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    coords = np.zeros((300, 2), dtype=np.float64)
    euler = estimate_head_pose_euler(coords, 640, 480)
    assert euler == (5.0, 35.0, 0.0)


def test_ablation_positive_rate_and_schema_enum_types() -> None:
    """Verify AblationMetrics.positive_rate and ClaimPolarity/ClockMode typing on schemas."""
    cfg = AppConfig(mode="live")
    assert cfg.mode == ClockMode.LIVE

    clm = ClaimItem(
        claim_id="C1",
        turn_id=1,
        speaker_id="S1",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="is",
        object_value="here",
        polarity="negative",
    )
    assert clm.polarity == ClaimPolarity.NEGATIVE

    rec = RecorderTurnRecord(
        turn_id=1,
        speaker_assignment=SpeakerAssignment(speaker_id="S1", similarity=1.0),
        features=TurnFeatures(turn_id=1, speaker_id="S1", start_ns=0, end_ns=1),
        mahalanobis=None,
        verification=ConsistencyVerificationResponse(
            verdict=VerdictStatus.DIRECT_CONTRADICTION,
            source_turn_id=1,
            prior_turn_id=0,
            explanation="contradiction",
        ),
    )
    metrics = evaluate_loso([rec])
    assert metrics["llm_only"].positive_rate == 1.0
    assert metrics["hybrid"].positive_rate == 1.0


def test_step_b_allows_spoken_firewall_words_and_pipeline_priors_from_other_cwd(tmp_path: Path, monkeypatch) -> None:
    """Verify Step B does not crash on spoken 'latency'/'blink' and pipeline loads priors from any CWD."""
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=1,
        source_claim_id="C2",
        prior_claim_id="C1",
        explanation="Network latency changed.",
        confidence=1.0,
    )
    ledger = ClaimsLedger(llm_adapter=mock_llm)
    c1 = ClaimItem(
        claim_id="C1",
        turn_id=1,
        speaker_id="S1",
        raw_subject="network latency",
        canonical_entity="network latency",
        predicate="was",
        object_value="200ms",
    )
    c2 = ClaimItem(
        claim_id="C2",
        turn_id=2,
        speaker_id="S1",
        raw_subject="network latency",
        canonical_entity="network latency",
        predicate="was",
        object_value="10ms",
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)
    res = ledger.verify_consistency(c2)
    assert res.verdict == VerdictStatus.DIRECT_CONTRADICTION

    # Pipeline loads data/priors_en.json even when CWD is tmp_path
    monkeypatch.chdir(tmp_path)
    pipeline = DiscourseConsistencyPipeline(config=AppConfig(mode=ClockMode.OFFLINE))
    assert not np.allclose(pipeline._correlation_matrix, np.eye(8))

    # evaluate_loso checks all verifications in record.verifications
    rec_multi = RecorderTurnRecord(
        turn_id=2,
        speaker_assignment=SpeakerAssignment(speaker_id="S1", similarity=1.0),
        features=TurnFeatures(turn_id=2, speaker_id="S1", start_ns=0, end_ns=1),
        mahalanobis=None,
        verification=ConsistencyVerificationResponse(
            verdict=VerdictStatus.CONSISTENT,
            source_turn_id=2,
            prior_turn_id=1,
            explanation="ok",
        ),
        verifications=[
            ConsistencyVerificationResponse(
                verdict=VerdictStatus.CONSISTENT,
                source_turn_id=2,
                prior_turn_id=1,
                explanation="ok",
            ),
            ConsistencyVerificationResponse(
                verdict=VerdictStatus.DIRECT_CONTRADICTION,
                source_turn_id=2,
                prior_turn_id=1,
                explanation="second claim contradicts",
            ),
        ],
    )
    metrics_multi = evaluate_loso([rec_multi])
    assert metrics_multi["llm_only"].contradictions_flagged == 1
    assert metrics_multi["llm_only"].positive_rate == 1.0
