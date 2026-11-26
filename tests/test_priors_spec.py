"""Specification tests for DCE literature priors and default configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

from dce.baseline_store import get_default_priors
from dce.schemas import PriorsData, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exact specification table:
# feature_name -> (stress_polarity, log_transform, log_offset, pop_median, pop_sigma, sigma_floor, max_anchor_drift)
EXPECTED_PRIORS_SPEC: Dict[str, Tuple[int, bool, float, float, float, float, float]] = {
    "response_latency_open": (1, True, 0.05, -0.20, 0.55, 0.15, 0.60),
    "response_latency_closed": (1, True, 0.05, -0.80, 0.45, 0.15, 0.50),
    "articulation_rate": (0, False, 0.0, 4.40, 0.75, 0.20, 0.80),
    "f0_semitone_sd": (1, False, 0.0, 2.60, 0.85, 0.25, 0.90),
    "jitter_local_pct": (1, False, 0.0, 1.10, 0.45, 0.12, 0.45),
    "shimmer_local_pct": (1, False, 0.0, 4.80, 1.60, 0.40, 1.60),
    "hnr_db": (-1, False, 0.0, 15.50, 3.20, 0.80, 3.00),
    "blink_rate_per_min": (1, False, 0.0, 18.00, 7.50, 2.00, 8.00),
}


def test_priors_json_matches_spec() -> None:
    """Verify data/priors_en.json matches every column of the priors specification table."""
    priors_path = REPO_ROOT / "data" / "priors_en.json"
    with priors_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    priors_data = PriorsData.model_validate(data)

    assert set(priors_data.features.keys()) == set(EXPECTED_PRIORS_SPEC.keys())
    for feat, (pol, log_t, log_off, mu, sigma, eps, delta_max) in EXPECTED_PRIORS_SPEC.items():
        p = priors_data.features[feat]
        assert p.feature_name == feat
        assert p.stress_polarity == pol, f"{feat} stress_polarity mismatch"
        assert p.log_transform is log_t, f"{feat} log_transform mismatch"
        assert math.isclose(p.log_offset, log_off, abs_tol=1e-9), f"{feat} log_offset mismatch"
        assert math.isclose(p.pop_median, mu, abs_tol=1e-9), f"{feat} pop_median mismatch"
        assert math.isclose(p.pop_sigma, sigma, abs_tol=1e-9), f"{feat} pop_sigma mismatch"
        assert math.isclose(p.sigma_floor, eps, abs_tol=1e-9), f"{feat} sigma_floor mismatch"
        assert math.isclose(p.max_anchor_drift, delta_max, abs_tol=1e-9), f"{feat} max_anchor_drift mismatch"


def test_get_default_priors_matches_spec() -> None:
    """Verify get_default_priors() matches every column of the priors specification table."""
    defaults = get_default_priors()
    assert set(defaults.keys()) == set(EXPECTED_PRIORS_SPEC.keys())
    for feat, (pol, log_t, log_off, mu, sigma, eps, delta_max) in EXPECTED_PRIORS_SPEC.items():
        p = defaults[feat]
        assert p.feature_name == feat
        assert p.stress_polarity == pol, f"{feat} stress_polarity mismatch"
        assert p.log_transform is log_t, f"{feat} log_transform mismatch"
        assert math.isclose(p.log_offset, log_off, abs_tol=1e-9), f"{feat} log_offset mismatch"
        assert math.isclose(p.pop_median, mu, abs_tol=1e-9), f"{feat} pop_median mismatch"
        assert math.isclose(p.pop_sigma, sigma, abs_tol=1e-9), f"{feat} pop_sigma mismatch"
        assert math.isclose(p.sigma_floor, eps, abs_tol=1e-9), f"{feat} sigma_floor mismatch"
        assert math.isclose(p.max_anchor_drift, delta_max, abs_tol=1e-9), f"{feat} max_anchor_drift mismatch"


def test_default_yaml_has_round1_config_keys() -> None:
    """Verify config/default.yaml defines the updated Mahalanobis and Reasoning config keys."""
    cfg = load_config(REPO_ROOT / "config" / "default.yaml")
    assert math.isclose(cfg.mahalanobis.stress_z_threshold, 1.5, abs_tol=1e-9)
    assert math.isclose(cfg.mahalanobis.calm_z_threshold, 1.5, abs_tol=1e-9)
    assert math.isclose(cfg.mahalanobis.z_clip, 50.0, abs_tol=1e-9)
    assert cfg.reasoning.retrieval_top_k == 12
