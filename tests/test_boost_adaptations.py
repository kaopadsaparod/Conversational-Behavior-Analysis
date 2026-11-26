"""Comprehensive tests for all 6 /boost research-inspired DCE adaptations."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.dashboard import assess_modality_reliability, compute_late_fusion_score
from benchmarks.run_ablation import (
    evaluate_decode_benchmark,
    evaluate_stratified_false_alarms,
    load_decode_dataset,
)
from dce.baseline_store import (
    BaselineStore,
    compute_noise_robust_voice_reliability,
    get_egemaps_priors,
)
from dce.extractors.audio_dsp import extract_egemaps_spectral_features
from dce.extractors.vision_face import (
    compute_eye_gaze_offset,
    compute_facial_action_units,
    evaluate_head_pose_quality,
)
from dce.reasoning.claims_ledger import ClaimsLedger, NLIPreFilter
from dce.schemas import (
    ClaimItem,
    ClaimPolarity,
    ConsistencyVerificationResponse,
    DirectionLabel,
    MahalanobisResult,
    ProsodyFeatures,
    ReasoningConfig,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    TurnTranscript,
    VerdictStatus,
    VisionFeatures,
    load_config,
)
from scripts.build_priors_ravdess import (
    calibrate_thresholds_from_d_sq_samples,
    load_besst_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# 1. DECODE & Dialogue NLI Pre-Filter + Benchmark
# ============================================================================


def test_nli_prefilter_skips_llm_when_no_contradiction() -> None:
    """NLIPreFilter should short-circuit consistent pairs without calling LLM Step B."""
    mock_llm = MagicMock()
    nli = NLIPreFilter(scorer_fn=lambda p, h: 0.10, use_hf_model=False)
    cfg = ReasoningConfig(nli_prefilter_enabled=True, nli_contradiction_threshold=0.50)
    ledger = ClaimsLedger(llm_adapter=mock_llm, config=cfg, nli_prefilter=nli)

    c1 = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPK_0",
        raw_subject="meeting",
        canonical_entity="meeting",
        predicate="starts_at",
        object_value="9am",
        polarity=ClaimPolarity.AFFIRMATIVE,
    )
    c2 = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPK_0",
        raw_subject="meeting",
        canonical_entity="meeting",
        predicate="starts_at",
        object_value="9am",
        polarity=ClaimPolarity.AFFIRMATIVE,
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    res = ledger.verify_consistency(c2)
    assert res.response.verdict == VerdictStatus.CONSISTENT
    assert ledger.stats["nli_prefilter_skipped"] == 1
    assert ledger.stats["nli_prefilter_passed"] == 0
    mock_llm.generate_structured.assert_not_called()


def test_nli_prefilter_passes_contradiction_to_llm_and_falls_back_without_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NLIPreFilter passes contradictions to LLM and falls back cleanly if transformers is missing."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    nli = NLIPreFilter(use_hf_model=True)

    mock_llm = MagicMock()
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=1,
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0001_01",
        explanation="Opposite polarity on same meeting claim.",
        confidence=0.95,
    )
    cfg = ReasoningConfig(nli_prefilter_enabled=True, nli_contradiction_threshold=0.50)
    ledger = ClaimsLedger(llm_adapter=mock_llm, config=cfg, nli_prefilter=nli)

    c1 = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPK_0",
        raw_subject="meeting",
        canonical_entity="meeting",
        predicate="attended",
        object_value="the board meeting",
        polarity=ClaimPolarity.AFFIRMATIVE,
    )
    c2 = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPK_0",
        raw_subject="meeting",
        canonical_entity="meeting",
        predicate="attended",
        object_value="the board meeting",
        polarity=ClaimPolarity.NEGATIVE,
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    res = ledger.verify_consistency(c2)
    assert res.response.verdict == VerdictStatus.DIRECT_CONTRADICTION
    assert ledger.stats["nli_prefilter_passed"] == 1
    mock_llm.generate_structured.assert_called_once()


def test_decode_benchmark_loader_and_f1(tmp_path: Path) -> None:
    """load_decode_dataset + evaluate_decode_benchmark should compute exact F1 and saved LLM calls."""
    decode_file = tmp_path / "decode_sample.jsonl"
    rows = [
        {
            "id": "d1",
            "turns": [
                {"speaker": "A", "text": "I arrived at 9am"},
                {"speaker": "A", "text": "I did not arrive at 9am"},
            ],
            "is_contradiction": True,
        },
        {
            "id": "d2",
            "turns": [
                {"speaker": "A", "text": "I live in Bangkok"},
                {"speaker": "A", "text": "I live in Bangkok"},
            ],
            "is_contradiction": False,
        },
    ]
    decode_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    samples = load_decode_dataset(decode_file)
    assert len(samples) == 2

    mock_llm = MagicMock()
    mock_llm.generate_structured.side_effect = [
        ConsistencyVerificationResponse(
            verdict=VerdictStatus.DIRECT_CONTRADICTION,
            source_turn_id=2,
            prior_turn_id=1,
            source_claim_id="d1_C2",
            prior_claim_id="d1_C1",
            explanation="Contradiction",
            confidence=0.95,
        )
    ]

    metrics = evaluate_decode_benchmark(
        samples,
        ledger_factory=lambda: ClaimsLedger(
            llm_adapter=mock_llm,
            config=ReasoningConfig(nli_prefilter_enabled=True),
            nli_prefilter=NLIPreFilter(use_hf_model=False),
        ),
    )
    assert metrics.total_samples == 2
    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.tn == 1
    assert math.isclose(metrics.f1_score, 1.0, abs_tol=1e-9)
    assert metrics.llm_calls_saved == 1


# ============================================================================
# 2. openSMILE (eGeMAPS) Noise-Robust Features & Weighting
# ============================================================================


def test_egemaps_spectral_features_and_opensmile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """extract_egemaps_spectral_features computes finite metrics and falls back when opensmile missing."""
    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    # 200 Hz harmonic speech-like signal + slight noise
    sig = 0.5 * np.sin(2.0 * np.pi * 200.0 * t) + 0.2 * np.sin(2.0 * np.pi * 400.0 * t)
    monkeypatch.setitem(sys.modules, "opensmile", None)

    feats = extract_egemaps_spectral_features(sig, sample_rate=sr, use_opensmile=True)
    assert feats["alpha_ratio_db"] is not None and feats["alpha_ratio_db"] > 0.0
    assert feats["spectral_slope_db"] is not None and math.isfinite(feats["spectral_slope_db"])
    assert feats["loudness_db"] is not None and feats["loudness_db"] < 0.0
    assert feats["acoustic_snr_db"] is not None and math.isfinite(feats["acoustic_snr_db"])

    empty = extract_egemaps_spectral_features([], sample_rate=sr)
    assert empty["alpha_ratio_db"] is None


def test_noise_robust_voice_weighting_in_baseline_store() -> None:
    """Low SNR should attenuate jitter/shimmer z-scores while keeping eGeMAPS features at full weight."""
    assert math.isclose(compute_noise_robust_voice_reliability(None), 1.0)
    assert math.isclose(compute_noise_robust_voice_reliability(18.0, snr_threshold_db=12.0), 1.0)
    low_rel = compute_noise_robust_voice_reliability(4.0, snr_threshold_db=12.0)
    assert 0.25 <= low_rel < 1.0

    store = BaselineStore(priors=get_egemaps_priors())
    tf_noisy = TurnFeatures(
        turn_id=1,
        speaker_id="SPK_NOISY",
        start_ns=0,
        end_ns=2_000_000_000,
        prosody=ProsodyFeatures(
            jitter_local_pct=2.0,
            shimmer_local_pct=8.0,
            alpha_ratio_db=10.0,
            loudness_db=-24.0,
            acoustic_snr_db=2.0,
            quality_flag="ok",
        ),
    )
    scores_noisy = store.score_and_update_turn("SPK_NOISY", tf_noisy)

    store_clean = BaselineStore(priors=get_egemaps_priors())
    tf_clean = TurnFeatures(
        turn_id=1,
        speaker_id="SPK_CLEAN",
        start_ns=0,
        end_ns=2_000_000_000,
        prosody=ProsodyFeatures(
            jitter_local_pct=2.0,
            shimmer_local_pct=8.0,
            alpha_ratio_db=10.0,
            loudness_db=-24.0,
            acoustic_snr_db=25.0,
            quality_flag="ok",
        ),
    )
    scores_clean = store_clean.score_and_update_turn("SPK_CLEAN", tf_clean)

    assert abs(scores_noisy["jitter_local_pct"].z_score) < abs(scores_clean["jitter_local_pct"].z_score)
    assert "alpha_ratio_db" in scores_noisy
    assert "loudness_db" in scores_noisy


# ============================================================================
# 3. OpenFace & Py-FEAT Head Pose Quality Gate, Eye Gaze & Action Units
# ============================================================================


def test_head_pose_quality_gate_and_au_gaze() -> None:
    """Head pose gate rejects yaw > 30 or pitch > 25; gaze and AUs compute cleanly."""
    assert evaluate_head_pose_quality(0.0, 0.0, 0.0) == (True, "ok")
    assert evaluate_head_pose_quality(0.0, 35.0, 0.0) == (False, "extreme_head_pose")
    assert evaluate_head_pose_quality(-28.0, 5.0, 0.0) == (False, "extreme_head_pose")
    assert evaluate_head_pose_quality(float("nan"), 0.0, 0.0) == (False, "extreme_head_pose")

    # 478 landmarks synthetic face
    coords = np.zeros((478, 2), dtype=np.float64)
    coords[33] = [100.0, 100.0]
    coords[133] = [140.0, 100.0]
    coords[159] = [120.0, 95.0]
    coords[145] = [120.0, 105.0]
    coords[160] = [110.0, 96.0]
    coords[158] = [130.0, 96.0]
    coords[153] = [110.0, 104.0]
    coords[144] = [130.0, 104.0]
    coords[263] = [220.0, 100.0]
    coords[362] = [180.0, 100.0]
    coords[385] = [190.0, 96.0]
    coords[387] = [210.0, 96.0]
    coords[373] = [190.0, 104.0]
    coords[380] = [210.0, 104.0]
    coords[107] = [145.0, 80.0]
    coords[336] = [175.0, 80.0]
    coords[61] = [130.0, 160.0]
    coords[291] = [190.0, 160.0]
    coords[13] = [160.0, 155.0]
    coords[14] = [160.0, 165.0]
    coords[468] = [125.0, 100.0]  # slight right gaze

    gaze = compute_eye_gaze_offset(coords)
    assert gaze is not None
    assert -1.0 <= gaze[0] <= 1.0 and -1.0 <= gaze[1] <= 1.0
    assert compute_eye_gaze_offset(coords[:468]) is None

    aus = compute_facial_action_units(coords)
    assert set(aus.keys()) == {
        "AU01_inner_brow_raise",
        "AU04_brow_lowerer",
        "AU12_lip_corner_puller",
        "AU25_lips_part",
        "AU45_blink",
    }
    assert aus["AU45_blink"] == 0.0


# ============================================================================
# 4. DOLOS & LCUNet Stratified False Alarm Rate
# ============================================================================


def test_stratified_false_alarm_rate_report() -> None:
    """evaluate_stratified_false_alarms reports FAR across turn_length, pitch_group, question_type."""
    records = [
        RecorderTurnRecord(
            turn_id=1,
            speaker_assignment=SpeakerAssignment(speaker_id="M1", similarity=0.9),
            transcript=TurnTranscript(turn_id=1, speaker_id="M1", text="Did you go?", start_ns=0, end_ns=1_000_000_000, is_question=True),
            features=TurnFeatures(
                turn_id=1,
                speaker_id="M1",
                start_ns=0,
                end_ns=1_000_000_000,
                timing=TimingFeatures(turn_duration_ms=1000.0, response_latency_closed=0.3),
                prosody=ProsodyFeatures(f0_median_hz=115.0),
            ),
            mahalanobis=MahalanobisResult(
                d_sq=12.0,
                mahalanobis_distance=3.46,
                chi2_tail_score=0.01,
                degrees_of_freedom=4,
                vocal_tension_flag=True,
                direction_label=DirectionLabel.COGNITIVE_LOAD_ANOMALY,
            ),
        ),
        RecorderTurnRecord(
            turn_id=2,
            speaker_assignment=SpeakerAssignment(speaker_id="F1", similarity=0.9),
            transcript=TurnTranscript(turn_id=2, speaker_id="F1", text="Where were you?", start_ns=0, end_ns=5_000_000_000, is_question=True),
            features=TurnFeatures(
                turn_id=2,
                speaker_id="F1",
                start_ns=0,
                end_ns=5_000_000_000,
                timing=TimingFeatures(turn_duration_ms=5000.0, response_latency_open=0.5),
                prosody=ProsodyFeatures(f0_median_hz=210.0),
            ),
            mahalanobis=MahalanobisResult(
                d_sq=2.0,
                mahalanobis_distance=1.41,
                chi2_tail_score=0.73,
                degrees_of_freedom=4,
                vocal_tension_flag=False,
                direction_label=DirectionLabel.NORMAL,
            ),
        ),
    ]

    report = evaluate_stratified_false_alarms(records)
    assert report["turn_length"]["short"].false_alarm_rate == 1.0
    assert report["turn_length"]["long"].false_alarm_rate == 0.0
    assert report["pitch_group"]["low_pitch"].false_alarm_rate == 1.0
    assert report["pitch_group"]["high_pitch"].false_alarm_rate == 0.0
    assert report["question_type"]["closed"].false_alarm_rate == 1.0
    assert report["question_type"]["open"].false_alarm_rate == 0.0
    assert math.isclose(report["bias_disparity"]["pitch_group"], 1.0, abs_tol=1e-9)


# ============================================================================
# 5. BESST Cognitive Load Calibration & Config
# ============================================================================


def test_besst_calibration_and_default_config(tmp_path: Path) -> None:
    """BESST calibration separates low/high d_sq distributions and default.yaml loads cleanly."""
    cal = calibrate_thresholds_from_d_sq_samples(
        low_d_sq=[1.5, 2.0, 2.5, 3.0, 3.2],
        high_d_sq=[10.0, 12.5, 14.0, 16.0, 18.0],
        df=4,
    )
    assert cal["calibration_source"] == "ravdess+besst"
    assert cal["besst_low_load_d_sq_p95"] < cal["cognitive_load_d_sq_threshold"] < cal["besst_high_load_d_sq_p50"]
    assert cal["tail_score_threshold"] in (0.01, 0.025, 0.05, 0.10, 0.15, 0.20)

    empty_cal = calibrate_thresholds_from_d_sq_samples([], [])
    assert empty_cal["calibration_source"] == "ravdess_only"

    csv_path = tmp_path / "besst_labels.csv"
    csv_path.write_text("filename,cognitive_load\nsample1.wav,low\nsample2.wav,high\n", encoding="utf-8")
    lbls = load_besst_labels(csv_path)
    assert lbls == {"sample1.wav": "low", "sample2.wav": "high"}

    cfg = load_config(REPO_ROOT / "config" / "default.yaml")
    assert cfg.mahalanobis.calibration_source == "ravdess+besst"
    assert cfg.baseline.noise_robust_voice_weighting is True


# ============================================================================
# 6. Real-Life Trial Dashboard Modality Reliability & Late Fusion
# ============================================================================


def test_dashboard_modality_reliability_and_late_fusion() -> None:
    """Broken vision (Low-light) is skipped so Audio + Text drive Late Fusion cleanly."""
    rec = RecorderTurnRecord(
        turn_id=5,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_0", similarity=0.92),
        transcript=TurnTranscript(
            turn_id=5,
            speaker_id="SPK_0",
            text="I never signed that invoice.",
            start_ns=0,
            end_ns=2_000_000_000,
            confidence=0.94,
        ),
        features=TurnFeatures(
            turn_id=5,
            speaker_id="SPK_0",
            start_ns=0,
            end_ns=2_000_000_000,
            prosody=ProsodyFeatures(f0_semitone_sd=3.0, acoustic_snr_db=22.0, quality_flag="ok"),
            vision=VisionFeatures(face_detected=False, low_light_flag=True, quality_flag="low_light"),
        ),
        mahalanobis=MahalanobisResult(
            d_sq=5.0,
            mahalanobis_distance=2.23,
            chi2_tail_score=0.28,
            degrees_of_freedom=3,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.NORMAL,
            feature_z_scores={"f0_semitone_sd": 1.5},
        ),
        verifications=[
            ConsistencyVerificationResponse(
                verdict=VerdictStatus.DIRECT_CONTRADICTION,
                source_turn_id=5,
                prior_turn_id=2,
                explanation="Contradicts Turn 2",
                confidence=0.95,
            )
        ],
    )

    rel = assess_modality_reliability(rec)
    assert rel["audio"]["status"] == "OK"
    assert rel["vision"]["status"] == "Skipped/Low-light"
    assert rel["vision"]["weight"] == 0.0
    assert rel["text"]["status"] == "High"

    fusion = compute_late_fusion_score(rec)
    assert "vision" in fusion["skipped_modalities"]
    assert set(fusion["active_modalities"]) == {"audio", "text"}
    # Max-dominant Review Priority: DIRECT_CONTRADICTION (1.0) is never diluted by audio (0.5)
    assert math.isclose(fusion["review_priority"], 1.0, abs_tol=1e-4)
    assert math.isclose(fusion["fused_risk_score"], 1.0, abs_tol=1e-4)


def test_pipeline_wires_reasoning_config_nli_prefilter() -> None:
    """DiscourseConsistencyPipeline must forward AppConfig.reasoning to ClaimsLedger."""
    from dce.pipeline import DiscourseConsistencyPipeline
    from dce.schemas import AppConfig

    cfg = AppConfig(reasoning=ReasoningConfig(nli_prefilter_enabled=True))
    pipe = DiscourseConsistencyPipeline(config=cfg)
    assert pipe._claims_ledger._nli_prefilter is not None


def test_nli_heuristic_handles_attached_numbers_and_avoids_boilerplate_false_positives() -> None:
    """Heuristic NLI should catch 9am vs 10am and not falsely flag unrelated same-speaker claims."""
    nli = NLIPreFilter(use_hf_model=False)
    c_9am = ClaimItem(
        claim_id="C1", turn_id=1, speaker_id="SPK_0",
        raw_subject="meeting", canonical_entity="meeting",
        predicate="starts_at", object_value="9am", polarity=ClaimPolarity.AFFIRMATIVE,
    )
    c_10am = ClaimItem(
        claim_id="C2", turn_id=2, speaker_id="SPK_0",
        raw_subject="meeting", canonical_entity="meeting",
        predicate="starts_at", object_value="10am", polarity=ClaimPolarity.AFFIRMATIVE,
    )
    assert nli.predict_contradiction_score(
        nli.format_utterance_claim(c_9am), nli.format_utterance_claim(c_10am)
    ) >= 0.50

    c_car = ClaimItem(
        claim_id="C3", turn_id=1, speaker_id="SPK_0",
        raw_subject="user", canonical_entity="user",
        predicate="owns", object_value="a car", polarity=ClaimPolarity.AFFIRMATIVE,
    )
    c_city = ClaimItem(
        claim_id="C4", turn_id=2, speaker_id="SPK_0",
        raw_subject="user", canonical_entity="user",
        predicate="lives_in", object_value="London", polarity=ClaimPolarity.AFFIRMATIVE,
    )
    assert nli.predict_contradiction_score(
        nli.format_utterance_claim(c_car), nli.format_utterance_claim(c_city)
    ) < 0.50


def test_egemaps_clean_tone_vs_white_noise_snr_and_stereo_safety() -> None:
    """Clean continuous harmonic audio must get high SNR, white noise low SNR, and 2D/NaN safe."""
    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    clean_sig = 0.5 * np.sin(2.0 * np.pi * 200.0 * t) + 0.2 * np.sin(2.0 * np.pi * 400.0 * t)
    clean_feats = extract_egemaps_spectral_features(clean_sig, sample_rate=sr)
    assert clean_feats["acoustic_snr_db"] is not None and clean_feats["acoustic_snr_db"] >= 12.0

    rng = np.random.default_rng(42)
    noise_sig = rng.normal(0.0, 0.2, size=sr)
    noise_feats = extract_egemaps_spectral_features(noise_sig, sample_rate=sr)
    assert noise_feats["acoustic_snr_db"] is not None and noise_feats["acoustic_snr_db"] < 12.0

    stereo_sig = np.column_stack([clean_sig, clean_sig])
    stereo_feats = extract_egemaps_spectral_features(stereo_sig, sample_rate=sr)
    assert stereo_feats["loudness_db"] is not None and math.isfinite(stereo_feats["loudness_db"])

    nan_sig = clean_sig.copy()
    nan_sig[10] = float("nan")
    nan_feats = extract_egemaps_spectral_features(nan_sig, sample_rate=sr)
    assert nan_feats["loudness_db"] is not None and math.isfinite(nan_feats["loudness_db"])


def test_noisy_mic_does_not_corrupt_speaker_baseline_history() -> None:
    """Low-SNR jitter/shimmer observations must not pollute BaselineStore._history."""
    store = BaselineStore(priors=get_egemaps_priors())
    tf_very_noisy = TurnFeatures(
        turn_id=1,
        speaker_id="SPK_BAD_MIC",
        start_ns=0,
        end_ns=2_000_000_000,
        prosody=ProsodyFeatures(
            jitter_local_pct=2.8,
            shimmer_local_pct=11.0,
            alpha_ratio_db=9.5,
            loudness_db=-24.0,
            acoustic_snr_db=0.0,
            quality_flag="ok",
        ),
    )
    scores = store.score_and_update_turn("SPK_BAD_MIC", tf_very_noisy)
    assert "jitter_local_pct" in scores
    hist = store._history.get("SPK_BAD_MIC", {})
    assert "jitter_local_pct" not in hist
    assert "shimmer_local_pct" not in hist
    assert "alpha_ratio_db" in hist


def test_nli_prefilter_label2id_and_temporal_preservation() -> None:
    """NLI must use label2id['contradiction'], run on CPU, and never drop temporal claims."""
    nli = NLIPreFilter(scorer_fn=lambda p, h: 0.02, use_hf_model=False)
    assert nli._device == "cpu"
    assert ReasoningConfig().nli_contradiction_threshold == pytest.approx(0.10)

    c_time_prior = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPK_0",
        raw_subject="I",
        canonical_entity="speaker_0",
        predicate="watched_news",
        object_value="at home",
        time_ref_raw="6:30pm",
    )
    c_time_curr = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPK_0",
        raw_subject="I",
        canonical_entity="speaker_0",
        predicate="arrived",
        object_value="airport",
        time_ref_raw="7:00pm",
    )
    kept = nli.filter_candidates(c_time_curr, [c_time_prior], threshold=0.10)
    assert len(kept) == 1


def test_besst_dimension_invariant_threshold_across_4_and_8_features() -> None:
    """BESST calibration must normalize by df so 8-feature normal turns are not falsely flagged."""
    from scripts.build_priors_ravdess import is_cognitive_load_elevated

    cal = calibrate_thresholds_from_d_sq_samples(
        low_d_sq=[2.5, 3.0, 3.8, 4.2],
        high_d_sq=[12.0, 14.0, 16.0, 18.0],
        df=4,
    )
    assert cal["calibration_df"] == 4
    assert cal["cognitive_load_d_sq_per_df_threshold"] == pytest.approx(cal["cognitive_load_d_sq_threshold"] / 4.0, rel=1e-2)

    # Normal 8-feature turn: d_sq=8.0 (d_sq/df = 1.0, tail_score=0.433) -> NOT elevated!
    assert is_cognitive_load_elevated(
        d_sq=8.0,
        df=8,
        tail_score=0.433,
        tail_threshold=cal["tail_score_threshold"],
        d_sq_per_df_threshold=cal["cognitive_load_d_sq_per_df_threshold"],
    ) is False


def test_head_pose_euler_folding_prevents_180_deg_frontal_rejection() -> None:
    """Frontal faces returning +-178 deg pitch/roll from RQDecomp3x3 must fold to +-2 deg and pass."""
    from dce.extractors.vision_face import fold_euler_angle_deg

    assert fold_euler_angle_deg(178.0) == pytest.approx(-2.0)
    assert fold_euler_angle_deg(-178.0) == pytest.approx(2.0)
    ok, flag = evaluate_head_pose_quality(pitch_deg=178.0, yaw_deg=2.0, roll_deg=-179.0)
    assert ok is True
    assert flag == "ok"


def test_review_priority_does_not_dilute_text_contradiction_with_calm_av() -> None:
    """A clear text contradiction with calm audio/vision (0.0) must keep review_priority=1.0."""
    from app.dashboard import compute_review_priority

    rec = RecorderTurnRecord(
        turn_id=5,
        speaker_assignment=SpeakerAssignment(speaker_id="SPK_0", similarity=0.95),
        transcript=TurnTranscript(
            turn_id=5,
            speaker_id="SPK_0",
            text="I never went to London.",
            start_ns=0,
            end_ns=1_000_000_000,
            confidence=0.95,
        ),
        features=TurnFeatures(
            turn_id=5,
            speaker_id="SPK_0",
            start_ns=0,
            end_ns=1_000_000_000,
            prosody=ProsodyFeatures(f0_semitone_sd=2.5, acoustic_snr_db=20.0, quality_flag="ok"),
            vision=VisionFeatures(blink_rate_per_min=18.0, face_detected=True, quality_flag="ok"),
        ),
        mahalanobis=MahalanobisResult(
            d_sq=0.0,
            mahalanobis_distance=0.0,
            chi2_tail_score=1.0,
            degrees_of_freedom=5,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.NORMAL,
            feature_z_scores={
                "f0_semitone_sd": 0.0,
                "jitter_local_pct": 0.0,
                "shimmer_local_pct": 0.0,
                "hnr_db": 0.0,
                "blink_rate_per_min": 0.0,
            },
        ),
        verifications=[
            ConsistencyVerificationResponse(
                verdict=VerdictStatus.DIRECT_CONTRADICTION,
                source_turn_id=5,
                prior_turn_id=1,
                explanation="Contradicts Turn 1.",
            )
        ],
    )
    res = compute_review_priority(rec)
    assert res["review_priority"] == pytest.approx(1.0)
    assert res["multimodal_anomaly"] == pytest.approx(0.0)
