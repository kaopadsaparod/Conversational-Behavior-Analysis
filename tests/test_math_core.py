"""Analytical QA unit tests for Discourse Consistency Engine mathematical core.

All expected numerical outputs are hand-derived from first principles equations:
- Semitone conversion: S(t) = 12 * log2(F0(t) / F0_tilde)
- Weighted median with exact 0.5 tie-breaking: 0.5 * (x_k + x_{k+1})
- Weighted MAD: weighted_median(|x - med|, w)
- Regularized Block-Diagonal Mahalanobis distance with eigen-safe fallback
- 4-state cognitive load direction classification (normal, cognitive_load_anomaly,
  calm_deviation without neutral blocking, mixed_subthreshold_deviation)
- 50Hz zero-normalized cross correlation with per-lag overlap windowing
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from dce.baseline_store import get_default_priors
from dce.math_core import (
    classify_direction,
    compute_mahalanobis,
    f0_to_semitones,
    normalized_av_xcorr,
    weighted_mad,
    weighted_median,
)
from dce.schemas import DirectionLabel, PriorParameter


# ============================================================================
# 1. Semitone Conversion Tests (Hand-Calculated)
# ============================================================================


def test_f0_to_semitones_hand_calculated() -> None:
    """Verify semitone conversion against exact hand calculations.

    Formula: S(t) = 12 * log2(F0 / F0_tilde)
    - F0 = 440, F0_tilde = 440 -> 12 * log2(1) = 0.0
    - F0 = 220, F0_tilde = 440 -> 12 * log2(0.5) = 12 * (-1) = -12.0
    - F0 = 880, F0_tilde = 440 -> 12 * log2(2) = 12 * 1 = +12.0
    - F0 <= 0 or NaN -> NaN
    - F0_tilde <= 0 -> all NaN
    """
    f0_inputs = [440.0, 220.0, 880.0, 0.0, -50.0, float("nan")]
    median_f0 = 440.0

    out = f0_to_semitones(f0_inputs, median_f0)

    # Hand-calculated assertions
    assert math.isclose(out[0], 0.0, abs_tol=1e-9)
    assert math.isclose(out[1], -12.0, abs_tol=1e-9)
    assert math.isclose(out[2], 12.0, abs_tol=1e-9)
    assert np.isnan(out[3])
    assert np.isnan(out[4])
    assert np.isnan(out[5])

    # Non-positive or non-finite median_f0 returns all NaN
    invalid_out = f0_to_semitones([440.0, 220.0], median_f0=0.0)
    assert np.all(np.isnan(invalid_out))

    invalid_neg = f0_to_semitones([440.0, 220.0], median_f0=-100.0)
    assert np.all(np.isnan(invalid_neg))

    invalid_nan = f0_to_semitones([440.0, 220.0], median_f0=float("nan"))
    assert np.all(np.isnan(invalid_nan))


# ============================================================================
# 2. Weighted Median & Weighted MAD Tests (Hand-Calculated)
# ============================================================================


def test_weighted_median_exact_half_weight_tie() -> None:
    """Verify exact 0.5W cumulative tie-break returns 0.5 * (x_k + x_{k+1}).

    Hand Calculation 1:
      values = [10.0, 20.0], weights = [1.0, 1.0]
      W = 2.0, half_W = 1.0
      k=0: C_0 = 1.0 == half_W and k < 1 -> returns 0.5 * (10.0 + 20.0) = 15.0

    Hand Calculation 2:
      values = [30.0, 10.0, 20.0], weights = [1.0, 0.5, 0.5]
      Sorted: (10.0, 0.5), (20.0, 0.5), (30.0, 1.0)
      W = 2.0, half_W = 1.0
      k=0: C_0 = 0.5 < 1.0
      k=1: C_1 = 1.0 == half_W and k < 2 -> returns 0.5 * (20.0 + 30.0) = 25.0
    """
    res1 = weighted_median([10.0, 20.0], [1.0, 1.0])
    assert math.isclose(res1, 15.0, abs_tol=1e-12)

    res2 = weighted_median([30.0, 10.0, 20.0], [1.0, 0.5, 0.5])
    assert math.isclose(res2, 25.0, abs_tol=1e-12)


def test_weighted_median_non_tie() -> None:
    """Verify non-tie weighted median returns first x_k with C_k > 0.5W.

    Hand Calculation:
      values = [1.0, 2.0, 3.0], weights = [0.2, 0.5, 0.3]
      Sorted: (1.0, 0.2), (2.0, 0.5), (3.0, 0.3)
      W = 1.0, half_W = 0.5
      k=0: C_0 = 0.2 < 0.5
      k=1: C_1 = 0.7 > 0.5 -> returns x_1 = 2.0
    """
    res = weighted_median([1.0, 2.0, 3.0], [0.2, 0.5, 0.3])
    assert math.isclose(res, 2.0, abs_tol=1e-12)


def test_weighted_median_edge_cases() -> None:
    """Verify weighted median handles zero weights and empty inputs gracefully."""
    assert np.isnan(weighted_median([], []))
    assert np.isnan(weighted_median([10.0, 20.0], [0.0, 0.0]))
    assert np.isnan(weighted_median([10.0, 20.0], [-1.0, -2.0]))

    # Single valid element
    assert math.isclose(weighted_median([42.0], [1.0]), 42.0, abs_tol=1e-12)


def test_weighted_median_tiny_weights_e13() -> None:
    """Verify rel_tol=1e-12, abs_tol=0.0 avoids false ties for weights ~1e-13."""
    # Non-tie: 0.4e-13 < 0.5 * 1.0e-13, must return 20.0 (not 15.0)
    assert weighted_median([10.0, 20.0], [0.4e-13, 0.6e-13]) == 20.0
    # Exact tie: 0.5e-13 == 0.5 * 1.0e-13, must interpolate to 15.0
    assert math.isclose(weighted_median([10.0, 20.0], [0.5e-13, 0.5e-13]), 15.0, abs_tol=1e-12)


def test_weighted_mad_hand_calculated() -> None:
    """Verify weighted MAD using the same weighted median function.

    Hand Calculation 1:
      values = [1.0, 2.0, 3.0], weights = [0.2, 0.5, 0.3]
      med = 2.0
      abs_diffs: |1.0 - 2.0| = 1.0 (w=0.2), |2.0 - 2.0| = 0.0 (w=0.5), |3.0 - 2.0| = 1.0 (w=0.3)
      Sorted: (0.0, 0.5), (1.0, 0.2), (1.0, 0.3)
      W = 1.0, half_W = 0.5
      k=0: C_0 = 0.5 == half_W and k < 2 -> returns 0.5 * (0.0 + 1.0) = 0.5

    Hand Calculation 2:
      values = [10.0, 20.0, 30.0, 40.0], weights = [1.0, 1.0, 1.0, 1.0]
      med = 25.0
      abs_diffs: 15.0 (w=1.0), 5.0 (w=1.0), 5.0 (w=1.0), 15.0 (w=1.0)
      Sorted: (5.0, 1.0), (5.0, 1.0), (15.0, 1.0), (15.0, 1.0)
      W = 4.0, half_W = 2.0
      k=0: C_0 = 1.0
      k=1: C_1 = 2.0 == half_W and k < 3 -> returns 0.5 * (5.0 + 15.0) = 10.0
    """
    mad1 = weighted_mad([1.0, 2.0, 3.0], [0.2, 0.5, 0.3])
    assert math.isclose(mad1, 0.5, abs_tol=1e-12)

    mad2 = weighted_mad([10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0])
    assert math.isclose(mad2, 10.0, abs_tol=1e-12)


# ============================================================================
# 3. Block-Diagonal Mahalanobis Distance & Eigen Fallback Tests
# ============================================================================


def test_compute_mahalanobis_hand_calculated_identity() -> None:
    """Verify Mahalanobis distance with identity correlation matrix.

    Hand Calculation:
      Two features: 'f0_semitone_sd' and 'jitter_local_pct'
      z = [2.0, 0.0]^T
      Sigma_V = I_2
      d_sq = z^T I^-1 z = 2.0^2 + 0.0^2 = 4.0
      D_M = sqrt(4.0) = 2.0
      df = 2
      chi2_tail_score = sf(4.0, df=2) = exp(-4.0 / 2) = exp(-2) approx 0.1353352832366127
      tail_score >= 0.05 -> direction is NORMAL
    """
    priors = get_default_priors()
    feature_order = list(priors.keys())
    eye_corr = np.eye(len(feature_order), dtype=np.float64)

    res = compute_mahalanobis(
        feature_scores={"f0_semitone_sd": 2.0, "jitter_local_pct": 0.0},
        correlation_matrix=eye_corr,
        feature_order=feature_order,
        priors=priors,
        reg_lambda=0.30,
    )

    assert math.isclose(res.d_sq, 4.0, abs_tol=1e-7)
    assert math.isclose(res.mahalanobis_distance, 2.0, abs_tol=1e-7)
    expected_tail = math.exp(-2.0)
    assert math.isclose(res.chi2_tail_score, expected_tail, abs_tol=1e-6)
    assert res.degrees_of_freedom == 2
    assert res.direction_label == DirectionLabel.NORMAL
    assert not res.eigen_fallback_used
    assert 0.0 <= res.chi2_tail_score <= 1.0


def test_compute_mahalanobis_cross_group_zeroing() -> None:
    """Verify that cross-group correlations are strictly zeroed out."""
    priors = get_default_priors()
    feature_order = list(priors.keys())

    # Build matrix with 0.8 correlation everywhere
    dense_corr = np.full((8, 8), 0.8, dtype=np.float64)
    np.fill_diagonal(dense_corr, 1.0)

    # Index 0 is response_latency_open (group 0: timing)
    # Index 3 is f0_semitone_sd (group 1: voice)
    # Cross group between index 0 and index 3 must be zeroed out!
    res = compute_mahalanobis(
        feature_scores={"response_latency_open": 1.0, "f0_semitone_sd": 1.0},
        correlation_matrix=dense_corr,
        feature_order=feature_order,
        priors=priors,
        reg_lambda=0.0,  # Zero regularization to observe pure block-diagonal structure
    )

    # In block-diagonal, cross group (0, 3) is 0.0.
    # Therefore, Sigma_V for these two features from different groups is exactly I_2.
    # z = [1.0, 1.0]^T -> d_sq = 1.0^2 + 1.0^2 = 2.0
    assert math.isclose(res.d_sq, 2.0, abs_tol=1e-7)
    assert math.isclose(res.mahalanobis_distance, math.sqrt(2.0), abs_tol=1e-7)


def test_compute_mahalanobis_singular_fallback() -> None:
    """Verify that singular submatrix triggers eigen-floor (1e-6) fallback gracefully."""
    priors = get_default_priors()
    feature_order = list(priors.keys())

    # Create singular correlation matrix with identical voice features (corr = 1.0)
    singular_corr = np.ones((8, 8), dtype=np.float64)

    res = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 2.0, "shimmer_local_pct": 2.0},
        correlation_matrix=singular_corr,
        feature_order=feature_order,
        priors=priors,
        reg_lambda=0.0,
        cond_limit=1e6,
        eig_floor=1e-6,
    )

    assert res.eigen_fallback_used is True
    assert math.isfinite(res.d_sq)
    assert math.isfinite(res.mahalanobis_distance)
    assert 0.0 <= res.chi2_tail_score <= 1.0
    assert hasattr(res, "chi2_tail_score")
    assert not hasattr(res, "p_value")


def test_compute_mahalanobis_invalid_correlation_matrix_raises() -> None:
    """Shape mismatch or non-finite values in correlation_matrix must raise ValueError."""
    priors = get_default_priors()
    feature_order = ["jitter_local_pct", "shimmer_local_pct"]

    with pytest.raises(ValueError):
        compute_mahalanobis(
            feature_scores={"jitter_local_pct": 1.0, "shimmer_local_pct": 1.0},
            correlation_matrix=np.eye(3),
            feature_order=feature_order,
            priors=priors,
        )

    nan_corr = np.array([[1.0, float("nan")], [float("nan"), 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_mahalanobis(
            feature_scores={"jitter_local_pct": 1.0, "shimmer_local_pct": 1.0},
            correlation_matrix=nan_corr,
            feature_order=feature_order,
            priors=priors,
        )

    inf_corr = np.array([[1.0, float("inf")], [float("inf"), 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_mahalanobis(
            feature_scores={"jitter_local_pct": 1.0, "shimmer_local_pct": 1.0},
            correlation_matrix=inf_corr,
            feature_order=feature_order,
            priors=priors,
        )


def test_compute_mahalanobis_non_pd_3x3_eigen_fallback() -> None:
    """Non-PD 3x3 voice submatrix triggers Cholesky failure -> eigen-clamp fallback."""
    priors = get_default_priors()
    order = ["jitter_local_pct", "shimmer_local_pct", "hnr_db"]
    # (jitter,shimmer)=0.9, (jitter,hnr)=0.9, (shimmer,hnr)=-0.9 -> smallest eig < 0
    non_pd = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ],
        dtype=np.float64,
    )
    res = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 1.5, "shimmer_local_pct": -0.5, "hnr_db": 1.0},
        correlation_matrix=non_pd,
        feature_order=order,
        priors=priors,
        reg_lambda=0.0,
    )
    assert res.eigen_fallback_used is True
    assert math.isfinite(res.d_sq) and res.d_sq >= 0.0


def test_compute_mahalanobis_z_clip_prevents_overflow() -> None:
    """Huge z=1e200 is clipped by z_clip (default 50.0) so d_sq is finite and bounded."""
    priors = get_default_priors()
    order = ["jitter_local_pct", "shimmer_local_pct"]
    res = compute_mahalanobis(
        feature_scores={"jitter_local_pct": 1e200, "shimmer_local_pct": -1e200},
        correlation_matrix=np.eye(2),
        feature_order=order,
        priors=priors,
        z_clip=50.0,
    )
    assert math.isfinite(res.d_sq)
    assert math.isclose(res.d_sq, 50.0**2 + 50.0**2, rel_tol=1e-12)
    assert math.isfinite(res.mahalanobis_distance)


def test_compute_mahalanobis_closed_form_identity_df2() -> None:
    """Closed-form identity check: d_sq = sum(z^2) and df=2 -> chi2_tail_score = exp(-d_sq/2) within 1e-12."""
    priors = get_default_priors()
    order = ["f0_semitone_sd", "jitter_local_pct"]
    z1, z2 = 1.8, -0.9
    res = compute_mahalanobis(
        feature_scores={"f0_semitone_sd": z1, "jitter_local_pct": z2},
        correlation_matrix=np.eye(2),
        feature_order=order,
        priors=priors,
    )
    expected_d_sq = z1**2 + z2**2
    expected_tail = math.exp(-expected_d_sq / 2.0)
    assert math.isclose(res.d_sq, expected_d_sq, abs_tol=1e-12)
    assert math.isclose(res.chi2_tail_score, expected_tail, abs_tol=1e-12)


def test_classify_direction_uses_positive_calm_and_stress_thresholds() -> None:
    """Verify stress_z_threshold and calm_z_threshold (positive magnitude) are respected."""
    priors = get_default_priors()
    # Positive calm_z_threshold=1.5: stress_z=-1.6 <= -1.5 -> CALM_DEVIATION
    _, lbl_calm = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": -1.6},
        priors=priors,
        calm_z_threshold=1.5,
    )
    assert lbl_calm == DirectionLabel.CALM_DEVIATION

    # Positive calm_z_threshold=2.0: stress_z=-1.6 > -2.0 -> MIXED_SUBTHRESHOLD_DEVIATION
    _, lbl_mixed = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": -1.6},
        priors=priors,
        calm_z_threshold=2.0,
    )
    assert lbl_mixed == DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION

    # stress_z_threshold=2.0: z=1.6 < 2.0 -> MIXED_SUBTHRESHOLD_DEVIATION (not anomaly)
    _, lbl_stress = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": 1.6},
        priors=priors,
        stress_z_threshold=2.0,
    )
    assert lbl_stress == DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION


# ============================================================================
# 4. 4-State Direction Classification Tests (Hand-Calculated)
# ============================================================================


def test_direction_state_1_normal() -> None:
    """State 1: tail_score >= 0.05 -> 'normal'."""
    priors = get_default_priors()
    tension, direction = classify_direction(
        chi2_tail_score=0.08,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": 2.5},
        priors=priors,
        tail_score_threshold=0.05,
    )
    assert direction == DirectionLabel.NORMAL


def test_direction_state_2_cognitive_load_anomaly() -> None:
    """State 2: tail_score < 0.05 and either max stress z >= 1.5 or neutral |z| >= 1.5."""
    priors = get_default_priors()

    # Case 2a: Directional stress feature >= 1.5
    tension, direction = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=["jitter_local_pct"],
        feature_z_scores={"jitter_local_pct": 1.6},  # polarity = +1 -> stress_z = 1.6 >= 1.5
        priors=priors,
        tail_score_threshold=0.05,
    )
    assert direction == DirectionLabel.COGNITIVE_LOAD_ANOMALY
    assert tension is True  # jitter > 1.5

    # Case 2b: Neutral feature |z| >= 1.5
    tension, direction = classify_direction(
        chi2_tail_score=0.02,
        evaluated_features=["articulation_rate"],
        feature_z_scores={"articulation_rate": 2.0},  # polarity = 0 -> |z| = 2.0 >= 1.5
        priors=priors,
        tail_score_threshold=0.05,
    )
    assert direction == DirectionLabel.COGNITIVE_LOAD_ANOMALY
    assert tension is False


def test_direction_state_3_calm_deviation_not_blocked_by_neutral() -> None:
    """State 3: calm_deviation when tail_score < 0.05, directional features have max < 0 and min <= -1.5,
    and neutral |z| < 1.5 (specifically verifying articulation_rate does NOT block calm_deviation).
    """
    priors = get_default_priors()

    # Directional features:
    # jitter_local_pct (polarity=+1): z = -1.6 -> stress_z = -1.6
    # hnr_db (polarity=-1): z = +1.6 -> stress_z = -1.6
    # Neutral feature:
    # articulation_rate (polarity=0): z = +0.6 -> |z| = 0.6 < 1.5
    evaluated = ["jitter_local_pct", "hnr_db", "articulation_rate"]
    z_scores = {
        "jitter_local_pct": -1.6,
        "hnr_db": 1.6,
        "articulation_rate": 0.6,
    }

    tension, direction = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=evaluated,
        feature_z_scores=z_scores,
        priors=priors,
        tail_score_threshold=0.05,
    )

    assert direction == DirectionLabel.CALM_DEVIATION
    assert tension is False


def test_direction_state_4_mixed_subthreshold_deviation() -> None:
    """State 4: tail_score < 0.05 but neither condition 2 nor 3 holds."""
    priors = get_default_priors()

    # Directional feature: jitter_local_pct (polarity=+1) has z = 0.8 -> stress_z = 0.8
    # Not >= 1.5, and not < 0.0
    # Neutral feature: articulation_rate has z = 0.5
    evaluated = ["jitter_local_pct", "articulation_rate"]
    z_scores = {
        "jitter_local_pct": 0.8,
        "articulation_rate": 0.5,
    }

    tension, direction = classify_direction(
        chi2_tail_score=0.01,
        evaluated_features=evaluated,
        feature_z_scores=z_scores,
        priors=priors,
        tail_score_threshold=0.05,
    )

    assert direction == DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION


# ============================================================================
# 5. 50Hz Audio-Visual Cross-Correlation Tests (Per-Lag Windowing)
# ============================================================================


def test_normalized_av_xcorr_insufficient_overlap() -> None:
    """Verify returns (None, None) when mutual overlap duration is less than 0.5s."""
    t_vis = [0.0, 0.1, 0.2, 0.3]
    v_sig = [1.0, 2.0, 1.5, 1.2]
    t_aud = [0.0, 0.1, 0.2, 0.3]
    a_sig = [0.5, 0.8, 0.6, 0.4]

    corr, lag = normalized_av_xcorr(
        t_vis, v_sig, t_aud, a_sig, grid_hz=50.0, min_overlap_sec=0.5
    )
    assert corr is None
    assert lag is None


def test_normalized_av_xcorr_zero_std_rejection() -> None:
    """Verify returns (None, None) when signal is flat (std < 1e-6)."""
    t_vis = np.linspace(0.0, 1.0, 51)
    v_flat = np.full(51, 2.0)  # zero std
    t_aud = np.linspace(0.0, 1.0, 51)
    a_sig = np.sin(2 * np.pi * 2.0 * t_aud)

    corr, lag = normalized_av_xcorr(
        t_vis, v_flat, t_aud, a_sig, grid_hz=50.0, min_overlap_sec=0.5, min_std=1e-6
    )
    assert corr is None
    assert lag is None


def test_normalized_av_xcorr_exact_overlap_per_lag() -> None:
    """Verify identical signals produce correlation close to 1.0 at lag 0ms."""
    t = np.linspace(0.0, 1.0, 51)
    sig = np.sin(2 * np.pi * 3.0 * t) + 1.0

    corr, lag = normalized_av_xcorr(
        t, sig, t, sig, grid_hz=50.0, min_overlap_sec=0.5, max_lag_ms=200.0
    )

    assert corr is not None
    assert lag is not None
    assert math.isclose(corr, 1.0, abs_tol=1e-4)
    assert math.isclose(lag, 0.0, abs_tol=1e-4)


def test_normalized_av_xcorr_moving_average_lags() -> None:
    """Verify rho(tau) = corr(v[n], e[n+tau]) with +100ms, -100ms, and 0ms lags at 50 Hz."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(500)
    v = np.convolve(raw, np.ones(5, dtype=np.float64) / 5.0, mode="valid")
    t = np.arange(len(v), dtype=np.float64) / 50.0

    # Audio lags video by 100ms (5 samples at 50 Hz)
    e_lag = np.empty_like(v)
    e_lag[5:] = v[:-5]
    e_lag[:5] = v[:5]
    corr_pos, lag_pos = normalized_av_xcorr(t, v, t, e_lag, grid_hz=50.0, max_lag_ms=200.0)
    assert corr_pos is not None and lag_pos is not None
    assert math.isclose(corr_pos, 1.0, abs_tol=1e-9)
    assert math.isclose(lag_pos, 100.0, abs_tol=1e-9)

    # Video lags audio by 100ms (audio leads by 100ms)
    e_lead = np.empty_like(v)
    e_lead[:-5] = v[5:]
    e_lead[-5:] = v[-5:]
    corr_neg, lag_neg = normalized_av_xcorr(t, v, t, e_lead, grid_hz=50.0, max_lag_ms=200.0)
    assert corr_neg is not None and lag_neg is not None
    assert math.isclose(corr_neg, 1.0, abs_tol=1e-9)
    assert math.isclose(lag_neg, -100.0, abs_tol=1e-9)

    # Identical signal
    corr_id, lag_id = normalized_av_xcorr(t, v, t, v, grid_hz=50.0, max_lag_ms=200.0)
    assert corr_id is not None and lag_id is not None
    assert math.isclose(corr_id, 1.0, abs_tol=1e-9)
    assert math.isclose(lag_id, 0.0, abs_tol=1e-9)
