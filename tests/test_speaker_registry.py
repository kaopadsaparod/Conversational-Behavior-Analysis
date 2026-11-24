"""Unit tests for Discourse Consistency Engine Speaker Registry and Identification.

Tests zero-norm/non-finite guards, duration-weighted unit centroid updates,
long-turn adoption of provisional buffers, short-turn cumulative duration promotion,
bounded capacity (max_speakers = 4) preventing zombie provisional buffers,
TTL expiration of stale provisional candidates, overlap handling, and strict
1-to-1 face hint bindings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dce.schemas import SpeakerRegistryConfig
from dce.speaker_registry import SpeakerRegistry


# ============================================================================
# 1. Zero-Norm and Non-Finite Guard Tests
# ============================================================================


def test_zero_norm_and_non_finite_guard() -> None:
    """Verify registry rejects zero-norm, sub-threshold norm, NaN, and Inf embeddings."""
    reg = SpeakerRegistry()

    # Exact zero vector
    res_zero = reg.assign(
        embedding=[0.0, 0.0, 0.0], duration_sec=2.5, timestamp_sec=1.0
    )
    assert res_zero.speaker_id == "UNKNOWN_INVALID"
    assert res_zero.low_confidence is True
    assert math.isclose(res_zero.similarity, 0.0, abs_tol=1e-7)

    # Sub-threshold norm (< min_embedding_norm = 1e-8)
    res_sub = reg.assign(
        embedding=[1e-9, 0.0, 0.0], duration_sec=2.5, timestamp_sec=2.0
    )
    assert res_sub.speaker_id == "UNKNOWN_INVALID"
    assert res_sub.low_confidence is True

    # NaN vector
    res_nan = reg.assign(
        embedding=[1.0, float("nan"), 0.5], duration_sec=2.5, timestamp_sec=3.0
    )
    assert res_nan.speaker_id == "UNKNOWN_INVALID"
    assert res_nan.low_confidence is True

    # Inf vector
    res_inf = reg.assign(
        embedding=[float("inf"), 1.0, 0.0], duration_sec=2.5, timestamp_sec=4.0
    )
    assert res_inf.speaker_id == "UNKNOWN_INVALID"
    assert res_inf.low_confidence is True

    # None embedding
    res_none = reg.assign(embedding=None, duration_sec=2.5, timestamp_sec=5.0)  # type: ignore
    assert res_none.speaker_id == "UNKNOWN_INVALID"
    assert res_none.low_confidence is True


# ============================================================================
# 2. Duration-Weighted Unit Centroid Update Tests
# ============================================================================


def test_duration_weighted_unit_centroid_update() -> None:
    """Verify centroid update: s_new = D_old*c_old + d_turn*e_hat, c_new = s_new / ||s_new||.

    Hand Calculation:
      Turn 1: e_1 = [1.0, 0.0], duration = 2.0s
              c_1 = [1.0, 0.0], D_1 = 2.0s
      Turn 2: e_2 = [0.8, 0.6] (norm = 1.0, sim = 0.8 >= sim_high=0.65), duration = 2.0s
              s_new = 2.0 * [1.0, 0.0] + 2.0 * [0.8, 0.6] = [3.6, 1.2]
              ||s_new|| = sqrt(3.6^2 + 1.2^2) = sqrt(12.96 + 1.44) = sqrt(14.4) approx 3.794733
              c_new = [3.6 / 3.794733, 1.2 / 3.794733] approx [0.9486833, 0.3162277]
              D_new = 4.0s
    """
    reg = SpeakerRegistry()

    # Turn 1
    res1 = reg.assign(embedding=[1.0, 0.0], duration_sec=2.0, timestamp_sec=1.0)
    assert res1.speaker_id == "SPEAKER_00"
    spk0 = reg.speakers["SPEAKER_00"]
    assert math.isclose(spk0.total_duration_sec, 2.0, abs_tol=1e-7)
    assert np.allclose(spk0.centroid, [1.0, 0.0], atol=1e-7)

    # Turn 2
    res2 = reg.assign(embedding=[0.8, 0.6], duration_sec=2.0, timestamp_sec=3.0)
    assert res2.speaker_id == "SPEAKER_00"
    assert res2.low_confidence is False

    expected_norm = math.sqrt(3.6**2 + 1.2**2)
    expected_centroid = np.array([3.6 / expected_norm, 1.2 / expected_norm])

    assert math.isclose(spk0.total_duration_sec, 4.0, abs_tol=1e-7)
    assert np.allclose(spk0.centroid, expected_centroid, atol=1e-6)
    assert math.isclose(float(np.linalg.norm(spk0.centroid)), 1.0, abs_tol=1e-7)


# ============================================================================
# 3. Long Turn Adopts Matching Provisional Buffer Tests
# ============================================================================


def test_long_turn_adopts_matching_provisional_buffer() -> None:
    """Verify long turn (>=2.0s) adopts matching PROV_XX and promotes to SPEAKER_XX with promoted_from."""
    reg = SpeakerRegistry()

    # Step 1: Short turn creates PROV_01
    res_short = reg.assign(embedding=[0.0, 1.0], duration_sec=1.0, timestamp_sec=1.0)
    assert res_short.speaker_id == "PROV_01"
    assert res_short.is_provisional is True
    assert "PROV_01" in reg.provisional

    # Step 2: Long turn with identical vector (sim = 1.0 >= sim_low = 0.45)
    res_long = reg.assign(embedding=[0.0, 1.0], duration_sec=2.5, timestamp_sec=3.0)
    assert res_long.speaker_id == "SPEAKER_00"
    assert res_long.is_provisional is False
    assert res_long.promoted_from == "PROV_01"
    assert res_long.low_confidence is False

    # PROV_01 should be deleted from provisional pool
    assert "PROV_01" not in reg.provisional
    assert "SPEAKER_00" in reg.speakers
    # Total duration: 1.0s (prov) + 2.5s (long turn) = 3.5s
    assert math.isclose(reg.speakers["SPEAKER_00"].total_duration_sec, 3.5, abs_tol=1e-7)


# ============================================================================
# 4. Short Turn Cumulative Provisional Promotion Tests
# ============================================================================


def test_short_turn_cumulative_promotion() -> None:
    """Verify short turns accumulate duration in provisional buffer and promote once >= 2.0s."""
    reg = SpeakerRegistry()

    # Turn 1: 1.2s -> creates PROV_01
    res1 = reg.assign(embedding=[1.0, 0.0], duration_sec=1.2, timestamp_sec=1.0)
    assert res1.speaker_id == "PROV_01"
    assert res1.is_provisional is True
    assert res1.promoted_from is None

    # Turn 2: 0.9s (cumulative 2.1s >= 2.0s) -> promotes to SPEAKER_00
    res2 = reg.assign(embedding=[1.0, 0.0], duration_sec=0.9, timestamp_sec=2.0)
    assert res2.speaker_id == "SPEAKER_00"
    assert res2.is_provisional is False
    assert res2.promoted_from == "PROV_01"
    assert "PROV_01" not in reg.provisional
    assert math.isclose(reg.speakers["SPEAKER_00"].total_duration_sec, 2.1, abs_tol=1e-7)


# ============================================================================
# 5. Capacity Bound (max_speakers = 4) and No Zombie Buffers
# ============================================================================


def test_max_speakers_capacity_limit_and_no_zombies() -> None:
    """Verify when 4 permanent speakers exist, no new PROV buffers are created."""
    config = SpeakerRegistryConfig(max_speakers=4)
    reg = SpeakerRegistry(config=config)

    # Populate 4 distinct permanent speakers with orthogonal embeddings
    reg.assign(embedding=[1.0, 0.0, 0.0, 0.0, 0.0], duration_sec=2.0, timestamp_sec=1.0)
    reg.assign(embedding=[0.0, 1.0, 0.0, 0.0, 0.0], duration_sec=2.0, timestamp_sec=2.0)
    reg.assign(embedding=[0.0, 0.0, 1.0, 0.0, 0.0], duration_sec=2.0, timestamp_sec=3.0)
    reg.assign(embedding=[0.0, 0.0, 0.0, 1.0, 0.0], duration_sec=2.0, timestamp_sec=4.0)

    assert len(reg.speakers) == 4

    # 5th distinct speaker attempts to register with short turn
    res_short_5 = reg.assign(
        embedding=[0.0, 0.0, 0.0, 0.0, 1.0], duration_sec=1.0, timestamp_sec=5.0
    )
    # Must NOT create PROV_XX; falls back to closest permanent with low_confidence=True
    assert len(reg.provisional) == 0
    assert res_short_5.is_provisional is False
    assert res_short_5.low_confidence is True
    assert res_short_5.speaker_id in reg.speakers

    # 5th distinct speaker attempts with long turn
    res_long_5 = reg.assign(
        embedding=[0.0, 0.0, 0.0, 0.0, 1.0], duration_sec=3.0, timestamp_sec=6.0
    )
    assert len(reg.speakers) == 4
    assert res_long_5.low_confidence is True
    assert res_long_5.speaker_id in reg.speakers


# ============================================================================
# 6. Provisional TTL Expiration Tests
# ============================================================================


def test_provisional_ttl_expiration() -> None:
    """Verify provisional candidates older than provisional_ttl_sec (60s) are purged."""
    config = SpeakerRegistryConfig(provisional_ttl_sec=60.0)
    reg = SpeakerRegistry(config=config)

    # Create PROV_01 at t = 10.0s
    reg.assign(embedding=[1.0, 0.0], duration_sec=1.0, timestamp_sec=10.0)
    assert "PROV_01" in reg.provisional

    # Advance time to t = 75.0s (delta = 65s > 60s)
    # Different speaker utterance triggers purge
    reg.assign(embedding=[0.0, 1.0], duration_sec=1.0, timestamp_sec=75.0)

    # PROV_01 must have been purged
    assert "PROV_01" not in reg.provisional
    assert "PROV_02" in reg.provisional


# ============================================================================
# 7. Overlap Turn Handling Tests
# ============================================================================


def test_overlap_turn_handling() -> None:
    """Verify overlap turn never updates centroids, returning closest permanent or UNKNOWN_OVERLAP."""
    reg = SpeakerRegistry()

    # Create SPEAKER_00 with [1.0, 0.0]
    reg.assign(embedding=[1.0, 0.0], duration_sec=2.0, timestamp_sec=1.0)
    initial_dur = reg.speakers["SPEAKER_00"].total_duration_sec
    initial_centroid = np.copy(reg.speakers["SPEAKER_00"].centroid)

    # Overlap turn matching SPEAKER_00 (sim >= sim_low = 0.45)
    res_match = reg.assign(
        embedding=[1.0, 0.0], duration_sec=1.5, timestamp_sec=2.0, overlap_flag=True
    )
    assert res_match.speaker_id == "SPEAKER_00"
    assert res_match.low_confidence is True
    # Ensure centroid and duration were NOT updated
    assert math.isclose(reg.speakers["SPEAKER_00"].total_duration_sec, initial_dur, abs_tol=1e-7)
    assert np.allclose(reg.speakers["SPEAKER_00"].centroid, initial_centroid, atol=1e-7)

    # Overlap turn with orthogonal voice (sim < 0.45)
    res_unmatch = reg.assign(
        embedding=[0.0, 1.0], duration_sec=1.5, timestamp_sec=3.0, overlap_flag=True
    )
    assert res_unmatch.speaker_id == "UNKNOWN_OVERLAP"
    assert res_unmatch.low_confidence is True


# ============================================================================
# 8. 1-to-1 Face Hint Binding Tests
# ============================================================================


def test_one_to_one_face_hint_binding() -> None:
    """Verify strict 1-to-1 face hint binding where reassigning clears prior binding."""
    reg = SpeakerRegistry()

    # Register SPEAKER_00 and bind to "face_user"
    reg.assign(
        embedding=[1.0, 0.0],
        duration_sec=2.0,
        timestamp_sec=1.0,
        face_hint="face_user",
    )
    assert reg.speakers["SPEAKER_00"].face_hint == "face_user"

    # Register SPEAKER_01 and bind to the SAME "face_user"
    reg.assign(
        embedding=[0.0, 1.0],
        duration_sec=2.0,
        timestamp_sec=2.0,
        face_hint="face_user",
    )
    # SPEAKER_00 face_hint must be cleared to None; SPEAKER_01 now owns "face_user"
    assert reg.speakers["SPEAKER_00"].face_hint is None
    assert reg.speakers["SPEAKER_01"].face_hint == "face_user"

    # Short turn shortcut using face hint with sim >= face_sim_floor (0.35)
    # Cosine sim with SPEAKER_01 is 0.40 (below sim_low=0.45, but above face_sim_floor=0.35)
    vec = [math.sqrt(1 - 0.40**2), 0.40]
    res_face = reg.assign(
        embedding=vec,
        duration_sec=1.0,
        timestamp_sec=3.0,
        face_hint="face_user",
    )
    assert res_face.speaker_id == "SPEAKER_01"
    assert res_face.low_confidence is False
