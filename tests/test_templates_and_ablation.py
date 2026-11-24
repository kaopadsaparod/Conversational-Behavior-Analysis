"""Unit tests for explanation templates and LOSO ablation benchmark runner.

Verifies:
1. Physical explanation template generation across all 4 direction states.
2. Suppression of stress warnings under DirectionLabel.CALM_DEVIATION.
3. Caveat inclusion under DirectionLabel.COGNITIVE_LOAD_ANOMALY.
4. evaluate_loso metric calculations across math-only, llm-only, and hybrid conditions.
"""

from __future__ import annotations

import pytest

from benchmarks.run_ablation import evaluate_loso
from dce.reasoning.templates import (
    MANDATORY_PSYCHOLOGICAL_CAVEAT,
    generate_physical_explanation,
)
from dce.schemas import (
    ConsistencyVerificationResponse,
    DirectionLabel,
    MahalanobisResult,
    ProsodyFeatures,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    VerdictStatus,
    VisionFeatures,
)


def _make_dummy_features(turn_id: int, speaker_id: str) -> TurnFeatures:
    """Create basic TurnFeatures instance for testing."""
    return TurnFeatures(
        turn_id=turn_id,
        speaker_id=speaker_id,
        start_ns=turn_id * 1_000_000_000,
        end_ns=(turn_id + 1) * 1_000_000_000,
        timing=TimingFeatures(response_latency_open=0.25),
        prosody=ProsodyFeatures(articulation_rate=4.4),
        vision=VisionFeatures(blink_rate_per_min=18.0),
    )


# ============================================================================
# 1. Physical Explanation Templates Tests
# ============================================================================


def test_generate_physical_explanation_normal() -> None:
    """Verify NORMAL state returns baseline stability message."""
    text = generate_physical_explanation(
        direction_label=DirectionLabel.NORMAL,
        evaluated_features=["f0_semitone_sd", "jitter_local_pct"],
        feature_z_scores={"f0_semitone_sd": 0.2, "jitter_local_pct": 0.1},
        vocal_tension_flag=False,
    )
    assert "within expected individual baseline variance" in text
    assert MANDATORY_PSYCHOLOGICAL_CAVEAT not in text


def test_generate_physical_explanation_calm_deviation() -> None:
    """Verify CALM_DEVIATION explicitly suppresses stress and highlights composure."""
    text = generate_physical_explanation(
        direction_label=DirectionLabel.CALM_DEVIATION,
        evaluated_features=["f0_semitone_sd", "jitter_local_pct"],
        feature_z_scores={"f0_semitone_sd": -1.8, "jitter_local_pct": -1.6},
        vocal_tension_flag=False,
    )
    assert "heightened vocal stability and composure" in text
    assert "relaxed physiological control" in text
    # MUST NOT contain cognitive load or stress warnings
    assert "Cognitive load anomaly" not in text
    assert MANDATORY_PSYCHOLOGICAL_CAVEAT not in text


def test_generate_physical_explanation_cognitive_load_anomaly() -> None:
    """Verify COGNITIVE_LOAD_ANOMALY details deviations and includes caveat."""
    text_tension = generate_physical_explanation(
        direction_label=DirectionLabel.COGNITIVE_LOAD_ANOMALY,
        evaluated_features=["jitter_local_pct", "shimmer_local_pct", "hnr_db"],
        feature_z_scores={"jitter_local_pct": 2.1, "shimmer_local_pct": 1.9, "hnr_db": -2.0},
        vocal_tension_flag=True,
    )
    assert "Cognitive load anomaly detected" in text_tension
    assert "elevated vocal fold tension" in text_tension
    assert "elevated jitter local pct" in text_tension
    assert MANDATORY_PSYCHOLOGICAL_CAVEAT in text_tension

    # Subcase: general divergence without individual z >= 1.5
    text_general = generate_physical_explanation(
        direction_label=DirectionLabel.COGNITIVE_LOAD_ANOMALY,
        evaluated_features=["f0_semitone_sd"],
        feature_z_scores={"f0_semitone_sd": 1.2},
        vocal_tension_flag=False,
    )
    assert "general multi-modal baseline divergence" in text_general
    assert MANDATORY_PSYCHOLOGICAL_CAVEAT in text_general


def test_generate_physical_explanation_mixed_subthreshold() -> None:
    """Verify MIXED_SUBTHRESHOLD_DEVIATION returns sub-threshold explanation."""
    text = generate_physical_explanation(
        direction_label=DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION,
        evaluated_features=["f0_semitone_sd", "articulation_rate"],
        feature_z_scores={"f0_semitone_sd": 1.1, "articulation_rate": -1.2},
        vocal_tension_flag=False,
    )
    assert "Sub-threshold multimodal deviations observed" in text
    assert MANDATORY_PSYCHOLOGICAL_CAVEAT not in text


# ============================================================================
# 2. LOSO Ablation Metric Evaluation Tests
# ============================================================================


def test_evaluate_loso_ablation_conditions() -> None:
    """Verify evaluate_loso correctly isolates math_only, llm_only, and hybrid."""
    # Turn 1: Math anomaly only (Cognitive Load Anomaly)
    rec_math = RecorderTurnRecord(
        turn_id=1,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_01", similarity=1.0),
        features=_make_dummy_features(1, "SPK_01"),
        mahalanobis=MahalanobisResult(
            d_sq=15.0,
            mahalanobis_distance=3.87,
            chi2_tail_score=0.001,
            degrees_of_freedom=3,
            vocal_tension_flag=True,
            direction_label=DirectionLabel.COGNITIVE_LOAD_ANOMALY,
            eigen_fallback_used=False,
            evaluated_features=["f0_semitone_sd"],
            feature_z_scores={"f0_semitone_sd": 2.2},
        ),
        verification=ConsistencyVerificationResponse(
            verdict=VerdictStatus.CONSISTENT,
            source_turn_id=1,
            explanation="Coherent statement.",
            confidence=1.0,
        ),
    )

    # Turn 2: Calm deviation (Must NOT be flagged as cognitive load anomaly!)
    rec_calm = RecorderTurnRecord(
        turn_id=2,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_01", similarity=1.0),
        features=_make_dummy_features(2, "SPK_01"),
        mahalanobis=MahalanobisResult(
            d_sq=14.0,
            mahalanobis_distance=3.74,
            chi2_tail_score=0.002,
            degrees_of_freedom=3,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.CALM_DEVIATION,
            eigen_fallback_used=False,
            evaluated_features=["f0_semitone_sd"],
            feature_z_scores={"f0_semitone_sd": -2.0},
        ),
    )

    # Turn 3: LLM contradiction only
    rec_llm = RecorderTurnRecord(
        turn_id=3,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_02", similarity=1.0),
        features=_make_dummy_features(3, "SPK_02"),
        mahalanobis=MahalanobisResult(
            d_sq=2.0,
            mahalanobis_distance=1.41,
            chi2_tail_score=0.45,
            degrees_of_freedom=3,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.NORMAL,
            eigen_fallback_used=False,
            evaluated_features=["f0_semitone_sd"],
            feature_z_scores={"f0_semitone_sd": 0.1},
        ),
        verification=ConsistencyVerificationResponse(
            verdict=VerdictStatus.DIRECT_CONTRADICTION,
            source_turn_id=3,
            prior_turn_id=1,
            explanation="Contradicts earlier timeline.",
            confidence=0.9,
        ),
    )

    # Turn 4: Completely normal turn
    rec_norm = RecorderTurnRecord(
        turn_id=4,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_02", similarity=1.0),
        features=_make_dummy_features(4, "SPK_02"),
        mahalanobis=MahalanobisResult(
            d_sq=1.0,
            mahalanobis_distance=1.0,
            chi2_tail_score=0.70,
            degrees_of_freedom=3,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.NORMAL,
            eigen_fallback_used=False,
            evaluated_features=["f0_semitone_sd"],
            feature_z_scores={"f0_semitone_sd": 0.0},
        ),
    )

    records = [rec_math, rec_calm, rec_llm, rec_norm]
    results = evaluate_loso(records)

    math_metrics = results["math_only"]
    assert math_metrics.total_turns == 4
    assert math_metrics.anomalies_detected == 1  # Only rec_math; rec_calm is calm deviation!

    llm_metrics = results["llm_only"]
    assert llm_metrics.total_turns == 4
    assert llm_metrics.contradictions_flagged == 1  # Only rec_llm

    hybrid_metrics = results["hybrid"]
    assert hybrid_metrics.total_turns == 4
    assert hybrid_metrics.anomalies_detected == 1
    assert hybrid_metrics.contradictions_flagged == 1
    assert hybrid_metrics.f1_score == 2 / 4.0  # 2 turns flagged (rec_math or rec_llm)


def test_evaluate_loso_empty_records() -> None:
    """Verify evaluate_loso handles empty records gracefully."""
    results = evaluate_loso([])
    assert results["math_only"].total_turns == 0
    assert results["math_only"].anomalies_detected == 0
    assert results["llm_only"].total_turns == 0
    assert results["hybrid"].total_turns == 0
