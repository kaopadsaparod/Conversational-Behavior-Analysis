"""Discourse Consistency Engine - Rule-Based Physical Explanation Templates.

Generates deterministic natural language explanations of physiological and prosodic
deviations from statistical baseline models.
Enforces psychological caveats and strictly suppresses stress warnings for calm deviations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from dce.schemas import DirectionLabel

MANDATORY_PSYCHOLOGICAL_CAVEAT = (
    "Caveat: Elevated physiological arousal or cognitive load can arise naturally "
    "from situational anxiety, conversational engagement, complex memory retrieval, "
    "or topic difficulty, and does NOT indicate deceptive intent or lack of veracity."
)


def generate_physical_explanation(
    direction_label: DirectionLabel,
    evaluated_features: Sequence[str],
    feature_z_scores: Dict[str, float],
    vocal_tension_flag: bool = False,
) -> str:
    """Generate rule-based explanation of physiological indicators based on directional status.

    Args:
        direction_label: 4-state DirectionLabel classification.
        evaluated_features: Names of features evaluated in the Mahalanobis model.
        feature_z_scores: Map from feature name to normalized baseline z-score.
        vocal_tension_flag: Whether jitter/shimmer/HNR breached tension thresholds.

    Returns:
        Structured explanatory paragraph including caveats where appropriate.
    """
    if direction_label == DirectionLabel.NORMAL:
        return (
            "Physiological and prosodic indicators are within expected individual baseline variance. "
            "No significant acoustic, timing, or kinematic anomalies detected."
        )

    if direction_label == DirectionLabel.CALM_DEVIATION:
        # Crucial: Stress warnings MUST be suppressed for calm deviations
        return (
            "Acoustic and kinematic patterns indicate heightened vocal stability and composure "
            "relative to speaker baseline. Perturbation metrics and latency are reduced, "
            "exhibiting relaxed physiological control."
        )

    if direction_label == DirectionLabel.COGNITIVE_LOAD_ANOMALY:
        deviations: List[str] = []

        if vocal_tension_flag:
            deviations.append("elevated vocal fold tension (perturbed micro-stability)")

        for feat in evaluated_features:
            z = feature_z_scores.get(feat, 0.0)
            if abs(z) >= 1.5:
                direction_str = "elevated" if z > 0.0 else "depressed"
                feat_readable = feat.replace("_", " ")
                deviations.append(f"{direction_str} {feat_readable} (z={z:+.2f})")

        dev_desc = "; ".join(deviations) if deviations else "general multi-modal baseline divergence"

        return (
            f"Cognitive load anomaly detected: observed {dev_desc}. "
            f"{MANDATORY_PSYCHOLOGICAL_CAVEAT}"
        )

    # direction_label == DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION:
    return (
        "Sub-threshold multimodal deviations observed across baseline metrics without a "
        "conclusive directional polarity. Individual features deviate moderately from personal "
        "norms but do not reach definitive arousal thresholds."
    )
