"""Discourse Consistency Engine - Statistical Baseline Profiling Store.

Maintains rolling per-speaker baseline distributions with Bayesian shrinkage toward
literature population priors, gated sample updates, per-feature anchor locks, and
idempotent speaker identity merging.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

from dce.math_core import weighted_mad, weighted_median
from dce.schemas import (
    BaselineConfig,
    FeatureScore,
    PriorParameter,
    PriorsData,
    TurnFeatures,
)

logger = logging.getLogger(__name__)


def get_default_priors() -> Dict[str, PriorParameter]:
    """Construct standard literature population priors for all 8 core features."""
    return {
        "response_latency_open": PriorParameter(
            feature_name="response_latency_open",
            pop_median=-0.20,
            pop_sigma=0.55,
            sigma_floor=0.15,
            max_anchor_drift=0.60,
            log_transform=True,
            log_offset=0.05,
            stress_polarity=1,
        ),
        "response_latency_closed": PriorParameter(
            feature_name="response_latency_closed",
            pop_median=-0.80,
            pop_sigma=0.45,
            sigma_floor=0.15,
            max_anchor_drift=0.50,
            log_transform=True,
            log_offset=0.05,
            stress_polarity=1,
        ),
        "articulation_rate": PriorParameter(
            feature_name="articulation_rate",
            pop_median=4.40,
            pop_sigma=0.75,
            sigma_floor=0.20,
            max_anchor_drift=0.80,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=0,
        ),
        "f0_semitone_sd": PriorParameter(
            feature_name="f0_semitone_sd",
            pop_median=2.60,
            pop_sigma=0.85,
            sigma_floor=0.25,
            max_anchor_drift=0.90,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
        "jitter_local_pct": PriorParameter(
            feature_name="jitter_local_pct",
            pop_median=1.10,
            pop_sigma=0.45,
            sigma_floor=0.12,
            max_anchor_drift=0.45,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
        "shimmer_local_pct": PriorParameter(
            feature_name="shimmer_local_pct",
            pop_median=4.80,
            pop_sigma=1.60,
            sigma_floor=0.40,
            max_anchor_drift=1.60,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
        "hnr_db": PriorParameter(
            feature_name="hnr_db",
            pop_median=15.50,
            pop_sigma=3.20,
            sigma_floor=0.80,
            max_anchor_drift=3.00,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=-1,
        ),
        "blink_rate_per_min": PriorParameter(
            feature_name="blink_rate_per_min",
            pop_median=18.00,
            pop_sigma=7.50,
            sigma_floor=2.00,
            max_anchor_drift=8.00,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
    }


def get_egemaps_priors() -> Dict[str, PriorParameter]:
    """Return 8 core priors plus noise-robust eGeMAPS spectral/energy priors."""
    priors = get_default_priors()
    priors.update(
        {
            "spectral_slope_db": PriorParameter(
                feature_name="spectral_slope_db",
                pop_median=-4.50,
                pop_sigma=2.50,
                sigma_floor=0.50,
                max_anchor_drift=3.00,
                log_transform=False,
                log_offset=0.0,
                stress_polarity=0,
            ),
            "alpha_ratio_db": PriorParameter(
                feature_name="alpha_ratio_db",
                pop_median=10.00,
                pop_sigma=4.50,
                sigma_floor=1.00,
                max_anchor_drift=5.00,
                log_transform=False,
                log_offset=0.0,
                stress_polarity=-1,
            ),
            "loudness_db": PriorParameter(
                feature_name="loudness_db",
                pop_median=-24.00,
                pop_sigma=6.00,
                sigma_floor=1.50,
                max_anchor_drift=6.00,
                log_transform=False,
                log_offset=0.0,
                stress_polarity=1,
            ),
        }
    )
    return priors


def compute_noise_robust_voice_reliability(
    acoustic_snr_db: Optional[float],
    snr_threshold_db: float = 12.0,
) -> float:
    """Compute a reliability weight in [0.25, 1.0] for jitter/shimmer under noisy mics/VoIP."""
    if acoustic_snr_db is None or not math.isfinite(acoustic_snr_db):
        return 1.0
    if acoustic_snr_db >= snr_threshold_db:
        return 1.0
    # Linear ramp from 0.25 (at <= 0 dB) to 1.0 (at snr_threshold_db)
    ratio = max(0.0, acoustic_snr_db) / max(snr_threshold_db, 1.0)
    return 0.25 + 0.75 * ratio


class BaselineStore:
    """Per-speaker baseline profiling store with Bayesian shrinkage and anchor locks."""

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        priors: Optional[Union[Dict[str, PriorParameter], PriorsData, Path, str]] = None,
        allow_builtin_priors: bool = False,
    ) -> None:
        """Initialize BaselineStore.

        Args:
            config: Baseline configuration parameters.
            priors: Priors dictionary, PriorsData model, or path to priors_en.json.
            allow_builtin_priors: If True, fall back to get_default_priors() when no
                priors file is found on disk.
        """
        self._config: BaselineConfig = config if config is not None else BaselineConfig()
        self._priors: Dict[str, PriorParameter] = self._resolve_priors(
            priors, allow_builtin_priors=allow_builtin_priors
        )

        # Structure: _history[speaker_id][feature_name] -> Deque[Tuple[float, float]] (val, weight)
        self._history: Dict[str, Dict[str, Deque[Tuple[float, float]]]] = {}
        # Structure: _anchors[speaker_id][feature_name] -> float (anchor_median)
        self._anchors: Dict[str, Dict[str, float]] = {}
        # Structure: _anchor_locked[speaker_id][feature_name] -> bool
        self._anchor_locked: Dict[str, Dict[str, bool]] = {}
        # Structure: _ok_counts[speaker_id][feature_name] -> int
        self._ok_counts: Dict[str, Dict[str, int]] = {}

    @property
    def config(self) -> BaselineConfig:
        """Return baseline configuration."""
        return self._config

    @property
    def priors(self) -> Dict[str, PriorParameter]:
        """Return priors dictionary."""
        return self._priors

    def _resolve_priors(
        self,
        priors: Optional[Union[Dict[str, PriorParameter], PriorsData, Path, str]],
        allow_builtin_priors: bool = False,
    ) -> Dict[str, PriorParameter]:
        """Resolve priors via explicit argument -> DCE_PRIORS_PATH -> repo data/priors_en.json."""
        if priors is not None:
            if isinstance(priors, PriorsData):
                return priors.features

            if isinstance(priors, (str, Path)):
                path = Path(priors)
                if not path.is_file():
                    raise FileNotFoundError(f"Priors file not found: {path}")
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return PriorsData.model_validate(data).features

            if isinstance(priors, dict):
                return priors

            raise TypeError(f"Unsupported priors type: {type(priors).__name__}")

        env_priors = os.environ.get("DCE_PRIORS_PATH")
        if env_priors:
            candidate_path = Path(env_priors)
        else:
            candidate_path = Path(__file__).resolve().parents[2] / "data" / "priors_en.json"

        if candidate_path.is_file():
            with candidate_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return PriorsData.model_validate(data).features

        if allow_builtin_priors:
            logger.warning(
                "Priors file not found at %s; falling back to built-in literature priors",
                candidate_path,
            )
            return get_default_priors()

        raise FileNotFoundError(f"Priors file not found: {candidate_path}")

    def transform_value(self, feature_name: str, raw_value: float) -> float:
        """Apply feature-specific log-offset transformation if configured.

        Args:
            feature_name: Name of the feature.
            raw_value: Raw physiological metric value.

        Returns:
            Transformed feature value float.
        """
        prior = self._priors.get(feature_name)
        if prior is not None and prior.log_transform:
            return math.log(max(float(raw_value), 0.0) + prior.log_offset)
        return float(raw_value)

    def get_baseline_stats(
        self,
        speaker_id: str,
        feature_name: str,
    ) -> Tuple[float, float, float]:
        """Compute shrunk baseline stats (hat_mu, hat_sigma, confidence_c).

        Cold-start guard: If len(samples) == 0 or n_eff <= 0.0, returns
        (prior.pop_median, max(prior.pop_sigma, prior.sigma_floor), 0.0)
        immediately WITHOUT calling weighted_median.

        Args:
            speaker_id: Unique identifier of the speaker.
            feature_name: Name of the evaluated feature.

        Returns:
            Tuple of (hat_mu, hat_sigma, confidence_c).
        """
        prior = self._priors.get(feature_name)
        if prior is None:
            return 0.0, 1.0, 0.0

        samples: List[Tuple[float, float]] = []
        if speaker_id in self._history and feature_name in self._history[speaker_id]:
            samples = list(self._history[speaker_id][feature_name])

        n_eff = float(sum(w for _, w in samples))

        # Strict Cold-Start Guard
        if len(samples) == 0 or n_eff <= 0.0:
            return (
                prior.pop_median,
                max(prior.pop_sigma, prior.sigma_floor),
                0.0,
            )

        vals = [item[0] for item in samples]
        weights = [item[1] for item in samples]

        # Calculate sample weighted median
        med_w = weighted_median(vals, weights)

        # Apply anchor clamping if locked
        is_locked = self._anchor_locked.get(speaker_id, {}).get(feature_name, False)
        if is_locked and feature_name in self._anchors.get(speaker_id, {}):
            anchor = self._anchors[speaker_id][feature_name]
            drift = prior.max_anchor_drift
            med_w = max(anchor - drift, min(anchor + drift, med_w))

        mad_w = weighted_mad(vals, weights)
        if not math.isfinite(mad_w) or mad_w < 0.0:
            mad_w = 0.0

        # Bayesian shrinkage (decouple location k_mu and scale k_sigma so scale never collapses)
        k_sigma = max(self._config.scale_shrinkage_k, 1.0)
        if self._config.decoupled_shrinkage:
            if feature_name in (
                "response_latency_open",
                "response_latency_closed",
                "articulation_rate",
            ):
                k_mu = (
                    self._config.l2_location_shrinkage_k
                    if n_eff >= 3.0
                    else self._config.acoustic_location_shrinkage_k
                )
            else:
                k_mu = self._config.acoustic_location_shrinkage_k
        else:
            k_mu = self._config.shrinkage_k

        hat_mu = (n_eff * med_w + k_mu * prior.pop_median) / (n_eff + k_mu) if (n_eff + k_mu) > 0.0 else med_w
        sigma_est = (n_eff * (1.4826 * mad_w) + k_sigma * prior.pop_sigma) / (n_eff + k_sigma)
        hat_sigma = max(sigma_est, prior.sigma_floor)
        confidence_c = n_eff / (n_eff + k_sigma)

        return hat_mu, hat_sigma, confidence_c

    def score_feature(
        self,
        speaker_id: str,
        feature_name: str,
        raw_value: Optional[float],
        quality_flag: str = "ok",
    ) -> FeatureScore:
        """Score an individual feature against the speaker's baseline before updating.

        Args:
            speaker_id: Unique identifier of the speaker.
            feature_name: Name of the feature.
            raw_value: Observed metric value (or None).
            quality_flag: Upstream quality indicator ('ok', etc.).

        Returns:
            FeatureScore instance containing raw, transformed, z-score, and confidence.
        """
        prior = self._priors.get(feature_name)
        if raw_value is None or not math.isfinite(raw_value) or prior is None or quality_flag != "ok":
            safe_raw = float(raw_value) if (raw_value is not None and math.isfinite(raw_value)) else None
            return FeatureScore(
                feature_name=feature_name,
                raw_value=safe_raw,
                transformed_value=None,
                baseline_mean=None,
                baseline_sigma=None,
                z_score=None,
                stress_z=None,
                confidence_weight=0.0,
            )

        transformed = self.transform_value(feature_name, raw_value)
        hat_mu, hat_sigma, conf = self.get_baseline_stats(speaker_id, feature_name)

        z = (transformed - hat_mu) / hat_sigma if hat_sigma > 0.0 else 0.0
        stress_z = float(prior.stress_polarity * z)

        return FeatureScore(
            feature_name=feature_name,
            raw_value=float(raw_value),
            transformed_value=transformed,
            baseline_mean=hat_mu,
            baseline_sigma=hat_sigma,
            z_score=z,
            stress_z=stress_z,
            confidence_weight=conf,
        )

    def update_feature(
        self,
        speaker_id: str,
        feature_name: str,
        raw_value: Optional[float],
        quality_flag: str = "ok",
        z_score: Optional[float] = None,
        word_count: Optional[int] = None,
    ) -> float:
        """Gated update of speaker baseline with per-feature anchor locking.

        Gate rules:
          - |z| <= 3.0 -> w = 1.0
          - 3.0 < |z| <= 5.0 -> w = 0.2
          - |z| > 5.0 or quality_flag != 'ok' -> w = 0.0 (discarded)

        Args:
            speaker_id: Unique identifier of the speaker.
            feature_name: Name of the feature to update.
            raw_value: Observed metric value.
            quality_flag: Signal quality flag.
            z_score: Precomputed z-score, or None to compute internally.

        Returns:
            Update weight assigned to this observation (0.0, 0.2, or 1.0).
        """
        if raw_value is None or not math.isfinite(raw_value):
            return 0.0

        if z_score is None:
            score = self.score_feature(speaker_id, feature_name, raw_value, quality_flag)
            z_score = score.z_score

        if z_score is None or not math.isfinite(z_score):
            return 0.0

        existing_samples = self._history.get(speaker_id, {}).get(feature_name, [])
        n_eff = float(sum(w for _, w in existing_samples))

        # Warm-up starvation guard: while n_eff < 3.0, accept plausible room/mic values even if |z| > 5 vs studio prior
        plausible_warmup = True
        if feature_name == "jitter_local_pct" and raw_value > 5.0:
            plausible_warmup = False
        elif feature_name == "shimmer_local_pct" and raw_value > 18.0:
            plausible_warmup = False
        elif feature_name == "hnr_db" and raw_value < 2.0:
            plausible_warmup = False

        abs_z = abs(z_score)
        if quality_flag != "ok":
            weight = 0.0
        elif n_eff < 3.0 and plausible_warmup:
            weight = 1.0
        elif abs_z > self._config.gate_reduced_z:
            weight = 0.0
        elif abs_z > self._config.gate_full_z:
            weight = self._config.gate_reduced_weight
        else:
            weight = 1.0

        if weight <= 0.0:
            return 0.0

        transformed = self.transform_value(feature_name, raw_value)

        # Initialize containers if absent
        if speaker_id not in self._history:
            self._history[speaker_id] = {}
            self._anchors[speaker_id] = {}
            self._anchor_locked[speaker_id] = {}
            self._ok_counts[speaker_id] = {}

        if feature_name not in self._history[speaker_id]:
            self._history[speaker_id][feature_name] = collections.deque(
                maxlen=self._config.rolling_window_turns
            )
        if feature_name not in self._ok_counts[speaker_id]:
            self._ok_counts[speaker_id][feature_name] = 0

        # Append sample
        self._history[speaker_id][feature_name].append((transformed, weight))
        # Guard against warm-up trap: short greetings below min_anchor_words do not advance anchor lock
        if word_count is None or word_count >= self._config.min_anchor_words:
            self._ok_counts[speaker_id][feature_name] += 1

        # Check anchor lock
        is_locked = self._anchor_locked[speaker_id].get(feature_name, False)
        ok_count = self._ok_counts[speaker_id][feature_name]

        if not is_locked and ok_count >= self._config.anchor_lock_turns:
            current_vals = [item[0] for item in self._history[speaker_id][feature_name]]
            current_weights = [item[1] for item in self._history[speaker_id][feature_name]]
            locked_median = weighted_median(current_vals, current_weights)
            if math.isfinite(locked_median):
                self._anchors[speaker_id][feature_name] = locked_median
                self._anchor_locked[speaker_id][feature_name] = True
                logger.info(
                    "Locked anchor for speaker %s, feature %s at median=%.4f (after %d ok turns)",
                    speaker_id,
                    feature_name,
                    locked_median,
                    ok_count,
                )

        return weight

    def score_and_update_turn(
        self,
        speaker_id: str,
        turn_features: TurnFeatures,
        word_count: Optional[int] = None,
    ) -> Dict[str, FeatureScore]:
        """Convenience method to score and update all available features of a turn.

        Args:
            speaker_id: Unique identifier of the speaker.
            turn_features: TurnFeatures container.

        Returns:
            Dictionary mapping feature names to their FeatureScore instances.
        """
        if turn_features.prosody.voip_or_denoised_flag:
            fine_wave_flag = "voip_denoised"
        elif turn_features.prosody.fine_wave_quality_flag != "ok":
            fine_wave_flag = turn_features.prosody.fine_wave_quality_flag
        else:
            fine_wave_flag = turn_features.prosody.quality_flag
        extracted_map: Dict[str, Tuple[Optional[float], str]] = {
            "response_latency_open": (turn_features.timing.response_latency_open, "ok"),
            "response_latency_closed": (turn_features.timing.response_latency_closed, "ok"),
            "articulation_rate": (turn_features.prosody.articulation_rate, turn_features.prosody.quality_flag),
            "f0_semitone_sd": (turn_features.prosody.f0_semitone_sd, turn_features.prosody.quality_flag),
            "jitter_local_pct": (turn_features.prosody.jitter_local_pct, fine_wave_flag),
            "shimmer_local_pct": (turn_features.prosody.shimmer_local_pct, fine_wave_flag),
            "hnr_db": (turn_features.prosody.hnr_db, fine_wave_flag),
            "blink_rate_per_min": (turn_features.vision.blink_rate_per_min, turn_features.vision.quality_flag),
        }
        for egemaps_feat in ("spectral_slope_db", "alpha_ratio_db", "loudness_db"):
            if egemaps_feat in self._priors:
                extracted_map[egemaps_feat] = (
                    getattr(turn_features.prosody, egemaps_feat, None),
                    turn_features.prosody.quality_flag,
                )

        au_map = {
            "au01_inner_brow_raise": "AU01_inner_brow_raise",
            "au04_brow_lowerer": "AU04_brow_lowerer",
            "au12_lip_corner_puller": "AU12_lip_corner_puller",
            "au25_lips_part": "AU25_lips_part",
        }
        for feat_key, au_key in au_map.items():
            if feat_key in self._priors:
                extracted_map[feat_key] = (
                    turn_features.vision.action_units.get(au_key),
                    turn_features.vision.quality_flag,
                )
        if "eye_gaze_horizontal" in self._priors:
            extracted_map["eye_gaze_horizontal"] = (
                turn_features.vision.eye_gaze_horizontal,
                turn_features.vision.quality_flag,
            )

        noise_reliability = (
            compute_noise_robust_voice_reliability(
                turn_features.prosody.acoustic_snr_db,
                snr_threshold_db=self._config.jitter_shimmer_noise_floor_snr_db,
            )
            if self._config.noise_robust_voice_weighting
            else 1.0
        )

        scores: Dict[str, FeatureScore] = {}
        raw_z_scores: Dict[str, Optional[float]] = {}

        # 1. Score all features first (Score-before-Update)
        for feat, (val, qflag) in extracted_map.items():
            if val is not None and math.isfinite(val):
                fs = self.score_feature(speaker_id, feat, val, quality_flag=qflag)
                raw_z_scores[feat] = fs.z_score
                if feat in ("jitter_local_pct", "shimmer_local_pct") and noise_reliability < 1.0:
                    fs = fs.model_copy(
                        update={
                            "z_score": fs.z_score * noise_reliability if fs.z_score is not None else None,
                            "stress_z": fs.stress_z * noise_reliability if fs.stress_z is not None else None,
                            "confidence_weight": fs.confidence_weight * noise_reliability,
                        }
                    )
                scores[feat] = fs

        # 2. Gated update using unattenuated z-scores (prevent noisy mics from corrupting baseline)
        for feat, score in scores.items():
            if feat in ("jitter_local_pct", "shimmer_local_pct") and noise_reliability < 0.5:
                continue
            val, qflag = extracted_map[feat]
            self.update_feature(
                speaker_id=speaker_id,
                feature_name=feat,
                raw_value=val,
                quality_flag=qflag,
                z_score=raw_z_scores.get(feat),
                word_count=word_count,
            )

        return scores

    def reset_anchor(self, speaker_id: str, feature_name: Optional[str] = None) -> None:
        """Reset anchor lock and clear stored anchor median for speaker.

        Args:
            speaker_id: Target speaker identifier.
            feature_name: Specific feature name, or None to reset all features.
        """
        if feature_name is None:
            self._anchors.pop(speaker_id, None)
            self._anchor_locked.pop(speaker_id, None)
            if speaker_id in self._ok_counts:
                self._ok_counts[speaker_id].clear()
        else:
            if speaker_id in self._anchors:
                self._anchors[speaker_id].pop(feature_name, None)
            if speaker_id in self._anchor_locked:
                self._anchor_locked[speaker_id].pop(feature_name, None)
            if speaker_id in self._ok_counts:
                self._ok_counts[speaker_id].pop(feature_name, None)

    def merge_speaker_id(self, old_id: str, new_id: str) -> None:
        """Idempotently merge history, anchors, and ok_counts from old_id into new_id.

        Combines history by prepending old history to new, truncating to the last
        rolling_window_turns (40). Sums ok_count and locks anchor if total >= anchor_lock_turns.
        Cleans up old_id entries. Safe to invoke repeatedly.

        Args:
            old_id: Source speaker ID to merge from.
            new_id: Target speaker ID to merge into.
        """
        if old_id == new_id or old_id not in self._history:
            return

        if new_id not in self._history:
            self._history[new_id] = {}
            self._anchors[new_id] = {}
            self._anchor_locked[new_id] = {}
            self._ok_counts[new_id] = {}

        old_hist_dict = self._history[old_id]
        old_anchors = self._anchors.get(old_id, {})
        old_locked = self._anchor_locked.get(old_id, {})
        old_oks = self._ok_counts.get(old_id, {})

        for feat, old_deque in old_hist_dict.items():
            new_deque = self._history[new_id].get(feat)
            if new_deque is None:
                new_deque = collections.deque(maxlen=self._config.rolling_window_turns)
                self._history[new_id][feat] = new_deque

            # Prepend old samples to new samples, retaining the latest 40 items
            combined = list(old_deque) + list(new_deque)
            truncated = combined[-self._config.rolling_window_turns:]
            self._history[new_id][feat] = collections.deque(
                truncated, maxlen=self._config.rolling_window_turns
            )

            # Sum ok counts
            old_ok = old_oks.get(feat, 0)
            new_ok = self._ok_counts[new_id].get(feat, 0)
            total_ok = old_ok + new_ok
            self._ok_counts[new_id][feat] = total_ok

            # Handle anchor lock
            is_new_locked = self._anchor_locked[new_id].get(feat, False)
            is_old_locked = old_locked.get(feat, False)

            if not is_new_locked:
                if is_old_locked:
                    self._anchors[new_id][feat] = old_anchors[feat]
                    self._anchor_locked[new_id][feat] = True
                elif total_ok >= self._config.anchor_lock_turns:
                    merged_vals = [item[0] for item in self._history[new_id][feat]]
                    merged_weights = [item[1] for item in self._history[new_id][feat]]
                    med = weighted_median(merged_vals, merged_weights)
                    if math.isfinite(med):
                        self._anchors[new_id][feat] = med
                        self._anchor_locked[new_id][feat] = True

        # Remove old speaker entry
        self._history.pop(old_id, None)
        self._anchors.pop(old_id, None)
        self._anchor_locked.pop(old_id, None)
        self._ok_counts.pop(old_id, None)
        logger.info("Successfully merged speaker profile %s into %s", old_id, new_id)
