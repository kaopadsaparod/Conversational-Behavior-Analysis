"""Discourse Consistency Engine - Mathematical Core.

Implements semitone conversions, exact-definition weighted median and MAD,
regularized Block-Diagonal Mahalanobis distance with eigen-safe fallback,
4-state cognitive load direction classification, and 50Hz audio-visual
zero-normalized cross-correlation with per-lag overlap normalization.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2

from dce.schemas import DirectionLabel, FeatureScore, MahalanobisResult, PriorParameter

logger = logging.getLogger(__name__)


def f0_to_semitones(
    f0_series: Union[Sequence[float], np.ndarray],
    median_f0: float,
) -> np.ndarray:
    """Convert fundamental frequency series (Hz) to semitones relative to median pitch.

    Formula: S(t) = 12 * log2(F0(t) / median_f0) for F0(t) > 0 and median_f0 > 0.
    Unvoiced frames (F0 <= 0 or non-finite) are set to np.nan.

    Args:
        f0_series: Sequence or array of fundamental frequencies in Hz.
        median_f0: Speaker median reference pitch in Hz.

    Returns:
        np.ndarray of float64 semitone values with np.nan for unvoiced frames.
    """
    f0_arr = np.asarray(f0_series, dtype=np.float64)
    out = np.full(f0_arr.shape, np.nan, dtype=np.float64)

    if not math.isfinite(median_f0) or median_f0 <= 0.0:
        return out

    valid_mask = np.isfinite(f0_arr) & (f0_arr > 0.0)
    if np.any(valid_mask):
        ratio = f0_arr[valid_mask] / float(median_f0)
        out[valid_mask] = 12.0 * np.log2(ratio)

    return out


def weighted_median(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Compute weighted median following exact cumulative weight specification.

    Pairs (x_k, w_k) are sorted by x_k ascending for w_k > 0.
    Cumulative weight C_k = sum_{j=1}^k w_j, and total weight W = C_n.
    - If there exists an index k < n-1 where C_k == 0.5 * W (within 1e-12 tolerance),
      returns 0.5 * (x_k + x_{k+1}).
    - Otherwise, returns the first x_k where C_k > 0.5 * W.

    Args:
        values: Sequence of data values.
        weights: Sequence of positive weights.

    Returns:
        Weighted median float, or np.nan if no valid positive weights.
    """
    valid_pairs: List[Tuple[float, float]] = []
    for v, w in zip(values, weights):
        if math.isfinite(v) and math.isfinite(w) and w > 0.0:
            valid_pairs.append((float(v), float(w)))

    if not valid_pairs:
        return float("nan")

    # Sort ascending by value
    valid_pairs.sort(key=lambda item: item[0])

    total_weight = sum(w for _, w in valid_pairs)
    if total_weight <= 0.0:
        return float("nan")

    half_weight = 0.5 * total_weight
    cum_weight = 0.0
    n = len(valid_pairs)

    for k in range(n):
        cum_weight += valid_pairs[k][1]
        # Check exact half-weight tie condition with next element
        if k < n - 1 and math.isclose(cum_weight, half_weight, rel_tol=1e-12, abs_tol=1e-12):
            return 0.5 * (valid_pairs[k][0] + valid_pairs[k + 1][0])
        if cum_weight > half_weight:
            return valid_pairs[k][0]

    return valid_pairs[-1][0]


def weighted_mad(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Compute weighted Median Absolute Deviation (MAD_w).

    MAD_w = weighted_median(|x_k - x_tilde_w|, w_k) using the same weighted median function.

    Args:
        values: Sequence of data values.
        weights: Sequence of positive weights.

    Returns:
        Weighted MAD float, or np.nan if no valid data.
    """
    med = weighted_median(values, weights)
    if not math.isfinite(med):
        return float("nan")

    abs_diffs = [abs(float(v) - med) for v in values]
    return weighted_median(abs_diffs, weights)


def classify_direction(
    chi2_tail_score: float,
    evaluated_features: Sequence[str],
    feature_z_scores: Dict[str, float],
    priors: Dict[str, PriorParameter],
    tail_score_threshold: float = 0.05,
    tension_z_threshold: float = 1.5,
    hnr_tension_z_threshold: float = -1.5,
) -> Tuple[bool, DirectionLabel]:
    """Classify 4-state cognitive load direction and vocal tension flag.

    Classification rules:
      1. vocal_tension_flag is True iff:
         (z_jitter > tension_z_threshold) or (z_shimmer > tension_z_threshold) or (z_hnr < hnr_tension_z_threshold)
      2. direction_label:
         - If chi2_tail_score >= tail_score_threshold: 'normal'
         - If chi2_tail_score < tail_score_threshold and
           (max(z_stress, directional) >= 1.5 or max(|z_neutral|) >= 1.5): 'cognitive_load_anomaly'
         - If chi2_tail_score < tail_score_threshold and len(directional) > 0 and
           max(z_stress, directional) < 0.0 and min(z_stress, directional) <= -1.5 and
           (all |z_neutral| < 1.5 for neutral features): 'calm_deviation'
         - Else: 'mixed_subthreshold_deviation'

    Args:
        chi2_tail_score: Tail probability of Mahalanobis distance.
        evaluated_features: List of feature names included in Mahalanobis calculation.
        feature_z_scores: Map of feature names to their z-scores.
        priors: Map of feature names to PriorParameter instances.
        tail_score_threshold: Threshold for anomalous tail score (default 0.05).
        tension_z_threshold: Positive z threshold for jitter/shimmer tension (1.5).
        hnr_tension_z_threshold: Negative z threshold for HNR tension (-1.5).

    Returns:
        Tuple of (vocal_tension_flag, DirectionLabel).
    """
    # 1. Evaluate vocal tension flag
    z_jitter = feature_z_scores.get("jitter_local_pct", 0.0)
    z_shimmer = feature_z_scores.get("shimmer_local_pct", 0.0)
    z_hnr = feature_z_scores.get("hnr_db", 0.0)

    has_jitter_tension = ("jitter_local_pct" in feature_z_scores) and (z_jitter > tension_z_threshold)
    has_shimmer_tension = ("shimmer_local_pct" in feature_z_scores) and (z_shimmer > tension_z_threshold)
    has_hnr_tension = ("hnr_db" in feature_z_scores) and (z_hnr < hnr_tension_z_threshold)

    vocal_tension_flag = bool(has_jitter_tension or has_shimmer_tension or has_hnr_tension)

    # 2. Partition evaluated features into directional and neutral
    directional_stress_z: List[float] = []
    neutral_abs_z: List[float] = []

    for feat in evaluated_features:
        if feat not in feature_z_scores:
            continue
        z = feature_z_scores[feat]
        prior = priors.get(feat)
        polarity = prior.stress_polarity if prior is not None else 0

        if polarity != 0:
            directional_stress_z.append(float(polarity * z))
        else:
            neutral_abs_z.append(abs(float(z)))

    # 3. Direction label assignment
    if chi2_tail_score >= tail_score_threshold:
        return vocal_tension_flag, DirectionLabel.NORMAL

    has_stress_anomaly = len(directional_stress_z) > 0 and max(directional_stress_z) >= 1.5
    has_neutral_anomaly = len(neutral_abs_z) > 0 and max(neutral_abs_z) >= 1.5

    if has_stress_anomaly or has_neutral_anomaly:
        return vocal_tension_flag, DirectionLabel.COGNITIVE_LOAD_ANOMALY

    # Check calm deviation
    if len(directional_stress_z) > 0:
        max_stress = max(directional_stress_z)
        min_stress = min(directional_stress_z)
        neutral_all_calm = len(neutral_abs_z) == 0 or all(az < 1.5 for az in neutral_abs_z)

        if max_stress < 0.0 and min_stress <= -1.5 and neutral_all_calm:
            return vocal_tension_flag, DirectionLabel.CALM_DEVIATION

    return vocal_tension_flag, DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION


def compute_mahalanobis(
    feature_scores: Dict[str, Union[FeatureScore, float]],
    correlation_matrix: Union[Sequence[Sequence[float]], np.ndarray],
    feature_order: Sequence[str],
    priors: Dict[str, PriorParameter],
    reg_lambda: float = 0.30,
    eig_floor: float = 1.0e-6,
    cond_limit: float = 1.0e+12,
    tail_score_threshold: float = 0.05,
    tension_z_threshold: float = 1.5,
    hnr_tension_z_threshold: float = -1.5,
) -> MahalanobisResult:
    """Compute regularized Block-Diagonal Mahalanobis distance with eigen-safe fallback.

    Structure:
      - Groups: Timing (0, 1, 2), Voice (3, 4, 5, 6), Vision (7).
      - Cross-group correlations zeroed; corr(0, 1) zeroed.
      - Symmetrized: Sigma = (M + M^T) / 2.
      - Regularized: Sigma_hat = (1 - lambda) * Sigma + lambda * I (lambda = 0.30).
      - Submatrix condition checked against cond_limit (1e12).
      - Distance D_M = sqrt(max(z_v^T Sigma_v^-1 z_v, 0.0)).
      - Tail score: chi2.sf(d_sq, df=len(V)).

    Args:
        feature_scores: Mapping from feature name to FeatureScore or raw z float.
        correlation_matrix: 8x8 empirical or literature correlation matrix.
        feature_order: Ordered list of 8 feature names.
        priors: PriorParameter database for evaluated features.
        reg_lambda: Regularization shrinkage toward identity (default 0.30).
        eig_floor: Eigenvalue floor for safe inversion fallback (default 1e-6).
        cond_limit: Matrix condition number limit before eigen-fallback (default 1e12).
        tail_score_threshold: Anomaly threshold on chi2_tail_score (default 0.05).
        tension_z_threshold: Vocal tension threshold for jitter/shimmer (default 1.5).
        hnr_tension_z_threshold: Vocal tension threshold for HNR (default -1.5).

    Returns:
        MahalanobisResult dataclass instance.
    """
    n_features = len(feature_order)
    raw_corr = np.array(correlation_matrix, dtype=np.float64)
    if raw_corr.shape != (n_features, n_features):
        raw_corr = np.eye(n_features, dtype=np.float64)

    # 1. Enforce block-diagonal group structure: timing(0,1,2), voice(3,4,5,6), vision(7)
    group_map: Dict[int, int] = {
        0: 0, 1: 0, 2: 0,  # timing
        3: 1, 4: 1, 5: 1, 6: 1,  # voice
        7: 2,  # vision
    }

    block_corr = np.copy(raw_corr)
    for i in range(n_features):
        for j in range(n_features):
            grp_i = group_map.get(i, i)
            grp_j = group_map.get(j, j)
            if grp_i != grp_j:
                block_corr[i, j] = 0.0

    # Specifically zero out correlation between latency_open(0) and latency_closed(1)
    if n_features > 1:
        block_corr[0, 1] = 0.0
        block_corr[1, 0] = 0.0

    # Symmetrize
    sym_corr = (block_corr + block_corr.T) / 2.0
    np.fill_diagonal(sym_corr, 1.0)

    # Regularization
    sigma_hat = (1.0 - reg_lambda) * sym_corr + reg_lambda * np.eye(n_features, dtype=np.float64)

    # 2. Extract valid features
    evaluated_features: List[str] = []
    evaluated_indices: List[int] = []
    evaluated_z: List[float] = []
    z_map: Dict[str, float] = {}

    for idx, feat in enumerate(feature_order):
        if feat not in feature_scores:
            continue
        val = feature_scores[feat]
        if isinstance(val, FeatureScore):
            z_val = val.z_score
        else:
            z_val = val

        if z_val is not None and math.isfinite(z_val):
            evaluated_features.append(feat)
            evaluated_indices.append(idx)
            evaluated_z.append(float(z_val))
            z_map[feat] = float(z_val)

    k = len(evaluated_indices)
    if k == 0:
        return MahalanobisResult(
            d_sq=0.0,
            mahalanobis_distance=0.0,
            chi2_tail_score=1.0,
            degrees_of_freedom=0,
            vocal_tension_flag=False,
            direction_label=DirectionLabel.NORMAL,
            eigen_fallback_used=False,
            evaluated_features=[],
            feature_z_scores={},
        )

    z_v = np.array(evaluated_z, dtype=np.float64)
    sigma_v = sigma_hat[np.ix_(evaluated_indices, evaluated_indices)]

    eigen_fallback_used = False
    solved_vec: Optional[np.ndarray] = None

    # Check condition limit
    try:
        cond_val = float(np.linalg.cond(sigma_v))
        if cond_val <= cond_limit:
            solved_vec = np.linalg.solve(sigma_v, z_v)
        else:
            eigen_fallback_used = True
    except np.linalg.LinAlgError:
        eigen_fallback_used = True

    if eigen_fallback_used or solved_vec is None:
        logger.warning("Submatrix ill-conditioned or singular; applying eigen-safe fallback.")
        eigvals, eigvecs = np.linalg.eigh(sigma_v)
        safe_eigs = np.maximum(eigvals, eig_floor)
        sigma_v_inv = eigvecs @ np.diag(1.0 / safe_eigs) @ eigvecs.T
        solved_vec = sigma_v_inv @ z_v
        eigen_fallback_used = True

    d_sq = max(float(np.dot(z_v, solved_vec)), 0.0)
    d_m = math.sqrt(d_sq)
    tail_score = float(chi2.sf(d_sq, df=k))

    vocal_tension, direction = classify_direction(
        chi2_tail_score=tail_score,
        evaluated_features=evaluated_features,
        feature_z_scores=z_map,
        priors=priors,
        tail_score_threshold=tail_score_threshold,
        tension_z_threshold=tension_z_threshold,
        hnr_tension_z_threshold=hnr_tension_z_threshold,
    )

    return MahalanobisResult(
        d_sq=d_sq,
        mahalanobis_distance=d_m,
        chi2_tail_score=tail_score,
        degrees_of_freedom=k,
        vocal_tension_flag=vocal_tension,
        direction_label=direction,
        eigen_fallback_used=eigen_fallback_used,
        evaluated_features=evaluated_features,
        feature_z_scores=z_map,
    )


def normalized_av_xcorr(
    t_vis_sec: Union[Sequence[float], np.ndarray],
    lip_aperture: Union[Sequence[float], np.ndarray],
    t_aud_sec: Union[Sequence[float], np.ndarray],
    audio_rms_envelope: Union[Sequence[float], np.ndarray],
    grid_hz: float = 50.0,
    min_overlap_sec: float = 0.5,
    max_lag_ms: float = 200.0,
    min_std: float = 1.0e-6,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute 50Hz zero-normalized cross-correlation between lip aperture and audio RMS.

    Per-lag mean subtraction and L2 norm strictly computed over that lag's
    overlapping window. Each lag evaluated must have >= min_overlap_sec actual overlap.

    Args:
        t_vis_sec: Visual signal timestamps in seconds.
        lip_aperture: Normalized lip aperture metric series.
        t_aud_sec: Audio signal timestamps in seconds.
        audio_rms_envelope: Audio RMS envelope series.
        grid_hz: Resampling grid frequency (default 50.0 Hz).
        min_overlap_sec: Minimum overlapping duration in seconds (default 0.5s -> 25 samples).
        max_lag_ms: Maximum search lag in milliseconds (default 200.0ms -> 10 samples).
        min_std: Minimum standard deviation threshold to reject flat signals (1e-6).

    Returns:
        Tuple of (max_correlation, best_lag_ms), or (None, None) if criteria not met.
    """
    tv = np.asarray(t_vis_sec, dtype=np.float64)
    lv = np.asarray(lip_aperture, dtype=np.float64)
    ta = np.asarray(t_aud_sec, dtype=np.float64)
    aa = np.asarray(audio_rms_envelope, dtype=np.float64)

    if len(tv) < 2 or len(ta) < 2 or len(lv) != len(tv) or len(aa) != len(ta):
        return None, None

    if not (np.all(np.isfinite(tv)) and np.all(np.isfinite(lv)) and
            np.all(np.isfinite(ta)) and np.all(np.isfinite(aa))):
        return None, None

    # Ensure strictly ascending timestamps
    if not np.all(np.diff(tv) > 0.0):
        sort_v = np.argsort(tv)
        tv = tv[sort_v]
        lv = lv[sort_v]
    if not np.all(np.diff(ta) > 0.0):
        sort_a = np.argsort(ta)
        ta = ta[sort_a]
        aa = aa[sort_a]

    # Find mutual time span
    start_t = max(float(tv[0]), float(ta[0]))
    end_t = min(float(tv[-1]), float(ta[-1]))
    span = end_t - start_t

    if span < min_overlap_sec:
        return None, None

    # Construct regular 50Hz grid
    dt = 1.0 / grid_hz
    n_pts = int(math.floor(span * grid_hz)) + 1
    if n_pts < 2:
        return None, None

    t_grid = start_t + np.arange(n_pts, dtype=np.float64) * dt

    # Interpolate signals onto grid
    v_grid = np.interp(t_grid, tv, lv)
    a_grid = np.interp(t_grid, ta, aa)

    max_lag_samples = int(round((max_lag_ms / 1000.0) * grid_hz))
    min_overlap_samples = int(round(min_overlap_sec * grid_hz))

    best_corr: Optional[float] = None
    best_lag_ms: Optional[float] = None

    for k in range(-max_lag_samples, max_lag_samples + 1):
        if k >= 0:
            v_slice = v_grid[k:]
            a_slice = a_grid[: n_pts - k]
        else:
            v_slice = v_grid[: n_pts + k]
            a_slice = a_grid[-k:]

        overlap_len = len(v_slice)
        if overlap_len < min_overlap_samples:
            continue

        std_v = float(np.std(v_slice))
        std_a = float(np.std(a_slice))
        if std_v < min_std or std_a < min_std:
            continue

        v_centered = v_slice - np.mean(v_slice)
        a_centered = a_slice - np.mean(a_slice)

        norm_v = float(np.linalg.norm(v_centered))
        norm_a = float(np.linalg.norm(a_centered))

        if norm_v < 1e-12 or norm_a < 1e-12:
            continue

        corr = float(np.dot(v_centered, a_centered) / (norm_v * norm_a))
        lag_ms = float(k * (1000.0 / grid_hz))

        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_lag_ms = lag_ms

    if best_corr is None or best_lag_ms is None:
        return None, None

    return best_corr, best_lag_ms
