"""Unit tests for Discourse Consistency Engine baseline store.

Tests cold-start guards (weighted_median never called on n_eff=0),
score-before-update semantics, per-feature independent anchor locking and reset,
and idempotent speaker profile merging with rolling window truncation.
"""

from __future__ import annotations

import collections
import math
from unittest.mock import MagicMock, patch

import pytest

from dce.baseline_store import BaselineStore, get_default_priors
from dce.schemas import BaselineConfig, PriorParameter


# ============================================================================
# 1. Cold-Start Guard Test
# ============================================================================


def test_cold_start_guard_never_calls_weighted_median() -> None:
    """Verify that when n_eff == 0.0 or samples are empty, weighted_median is NEVER called.

    Must return: (prior.pop_median, max(prior.pop_sigma, prior.sigma_floor), 0.0)
    """
    priors = get_default_priors()
    store = BaselineStore(priors=priors)

    prior_articulation = priors["articulation_rate"]
    expected_mu = prior_articulation.pop_median  # 4.40
    expected_sigma = max(prior_articulation.pop_sigma, prior_articulation.sigma_floor)  # 0.75

    with patch("dce.baseline_store.weighted_median") as mock_median:
        # Case A: Brand new speaker with zero history
        mu, sigma, conf = store.get_baseline_stats("NEW_SPEAKER", "articulation_rate")
        mock_median.assert_not_called()
        assert math.isclose(mu, expected_mu, abs_tol=1e-7)
        assert math.isclose(sigma, expected_sigma, abs_tol=1e-7)
        assert math.isclose(conf, 0.0, abs_tol=1e-7)

        # Case B: Speaker history exists but has zero effective weight
        store._history["ZERO_WEIGHT_SPEAKER"] = {
            "articulation_rate": collections.deque([(5.0, 0.0), (6.0, 0.0)], maxlen=40)
        }
        mu_zero, sigma_zero, conf_zero = store.get_baseline_stats(
            "ZERO_WEIGHT_SPEAKER", "articulation_rate"
        )
        mock_median.assert_not_called()
        assert math.isclose(mu_zero, expected_mu, abs_tol=1e-7)
        assert math.isclose(sigma_zero, expected_sigma, abs_tol=1e-7)
        assert math.isclose(conf_zero, 0.0, abs_tol=1e-7)


# ============================================================================
# 2. Score-Before-Update Test
# ============================================================================


def test_score_before_update_ordering() -> None:
    """Verify that an observation is scored against the baseline prior to updating the window.

    Pop prior for articulation_rate: pop_median = 4.40, pop_sigma = 0.75.
    Observation x = 5.90.
    Pre-update baseline: hat_mu = 4.40, hat_sigma = 0.75.
    Expected z-score = (5.90 - 4.40) / 0.75 = 1.50 / 0.75 = 2.0.
    """
    store = BaselineStore()
    speaker_id = "SPEAKER_TEST"
    feature = "articulation_rate"

    # Score observation on pristine speaker
    score_first = store.score_feature(speaker_id, feature, raw_value=5.90, quality_flag="ok")
    assert math.isclose(score_first.baseline_mean, 4.40, abs_tol=1e-6)
    assert math.isclose(score_first.baseline_sigma, 0.75, abs_tol=1e-6)
    assert math.isclose(score_first.z_score, 2.0, abs_tol=1e-6)
    assert math.isclose(score_first.confidence_weight, 0.0, abs_tol=1e-6)

    # Now execute the update
    w = store.update_feature(
        speaker_id, feature, raw_value=5.90, quality_flag="ok", z_score=score_first.z_score
    )
    assert w == 1.0  # |z| = 2.0 <= 3.0 -> full weight 1.0

    # Verify that the subsequent baseline has now changed due to the update
    mu_after, sigma_after, conf_after = store.get_baseline_stats(speaker_id, feature)
    # n_eff = 1.0, k = 6.0 -> conf = 1.0 / 7.0 approx 0.142857
    assert conf_after > 0.0
    # Shrunk mu: (1.0 * 5.90 + 6.0 * 4.40) / 7.0 = 32.30 / 7.0 approx 4.6142857
    expected_new_mu = (1.0 * 5.90 + 6.0 * 4.40) / 7.0
    assert math.isclose(mu_after, expected_new_mu, abs_tol=1e-5)


# ============================================================================
# 3. Per-Feature Anchor Locking and Reset Tests
# ============================================================================


def test_per_feature_independent_anchor_locking() -> None:
    """Verify anchor locking operates independently per (speaker_id, feature_name) at 8 ok turns."""
    config = BaselineConfig(anchor_lock_turns=8)
    store = BaselineStore(config=config)
    speaker_id = "SPK_ANCHOR"

    feat1 = "articulation_rate"
    feat2 = "jitter_local_pct"

    # Feed 8 ok turns for feat1
    for _ in range(8):
        store.update_feature(speaker_id, feat1, raw_value=4.50, quality_flag="ok", z_score=0.1)

    # Feed only 3 ok turns for feat2
    for _ in range(3):
        store.update_feature(speaker_id, feat2, raw_value=1.10, quality_flag="ok", z_score=0.0)

    # feat1 must be locked; feat2 must NOT be locked
    assert store._anchor_locked[speaker_id][feat1] is True
    assert feat1 in store._anchors[speaker_id]
    assert math.isclose(store._anchors[speaker_id][feat1], 4.50, abs_tol=1e-6)

    assert store._anchor_locked[speaker_id].get(feat2, False) is False
    assert feat2 not in store._anchors[speaker_id]

    # Test targeted reset of feat1
    store.reset_anchor(speaker_id, feature_name=feat1)
    assert store._anchor_locked[speaker_id].get(feat1, False) is False
    assert feat1 not in store._anchors[speaker_id]
    assert store._ok_counts[speaker_id].get(feat1, 0) == 0

    # Re-lock feat1 after 8 more turns
    for _ in range(8):
        store.update_feature(speaker_id, feat1, raw_value=4.80, quality_flag="ok", z_score=0.2)
    assert store._anchor_locked[speaker_id][feat1] is True

    # Test global reset for speaker_id
    store.reset_anchor(speaker_id, feature_name=None)
    assert speaker_id not in store._anchors
    assert speaker_id not in store._anchor_locked


# ============================================================================
# 4. Idempotent merge_speaker_id Tests
# ============================================================================


def test_idempotent_merge_speaker_id() -> None:
    """Verify merge_speaker_id is idempotent, combines history keeping latest 40, and sums ok_counts."""
    config = BaselineConfig(rolling_window_turns=40, anchor_lock_turns=8)
    store = BaselineStore(config=config)

    old_id = "PROV_01"
    new_id = "SPEAKER_00"
    feat = "articulation_rate"

    # Old speaker has 25 samples with value 4.0 (ok_count = 5)
    for _ in range(25):
        store.update_feature(old_id, feat, raw_value=4.0, quality_flag="ok", z_score=0.0)
    store._ok_counts[old_id][feat] = 5

    # New speaker has 20 samples with value 5.0 (ok_count = 4)
    for _ in range(20):
        store.update_feature(new_id, feat, raw_value=5.0, quality_flag="ok", z_score=0.0)
    store._ok_counts[new_id][feat] = 4

    # Merging same ID or non-existent ID should be clean no-op
    store.merge_speaker_id("NON_EXISTENT", new_id)
    store.merge_speaker_id(new_id, new_id)

    # Perform first merge
    store.merge_speaker_id(old_id, new_id)

    # Assertions on merged state:
    # 1. Total samples combined = 25 + 20 = 45, truncated to latest 40 entries
    assert len(store._history[new_id][feat]) == 40
    # The first 20 in the truncated list should be from old (4.0), the last 20 from new (5.0)
    samples = list(store._history[new_id][feat])
    assert math.isclose(samples[-1][0], 5.0, abs_tol=1e-6)
    assert math.isclose(samples[0][0], 4.0, abs_tol=1e-6)

    # 2. ok_count summed: 5 + 4 = 9 >= 8 -> triggers anchor lock!
    assert store._ok_counts[new_id][feat] == 9
    assert store._anchor_locked[new_id][feat] is True
    assert feat in store._anchors[new_id]

    # 3. Old ID deleted
    assert old_id not in store._history
    assert old_id not in store._anchors

    # 4. Perform second merge call with same old_id (now deleted) -> Must be idempotent no-op!
    history_snapshot = list(store._history[new_id][feat])
    ok_count_snapshot = store._ok_counts[new_id][feat]
    anchor_snapshot = store._anchors[new_id][feat]

    store.merge_speaker_id(old_id, new_id)

    assert list(store._history[new_id][feat]) == history_snapshot
    assert store._ok_counts[new_id][feat] == ok_count_snapshot
    assert store._anchors[new_id][feat] == anchor_snapshot
