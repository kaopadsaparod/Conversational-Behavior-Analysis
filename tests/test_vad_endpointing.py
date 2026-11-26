"""Unit tests for Discourse Consistency Engine VAD, Dual-Level Endpointing, and Mailbox.

Tests:
1. Micro-pause detection (200-399ms) recorded in intra_turn_pauses in BOTH SPEAKING and SOFT_PAUSE states.
2. Soft pause: silence reaching 400ms emits single SpeculativeTriggerEvent if speech >= 600ms.
3. Noise rejection: prob 0.36 during soft pause does not resume to speaking.
4. Resume hysteresis: prob >= 0.55 resumes to SPEAKING, emits CancelSpeculativeEvent with hesitation_pause, increments epoch_id.
5. Hard EOT commit at 800ms silence.
6. Forced split at 25s turn duration (forced_split=True).
7. flush() method commits trailing turn at EOF.
8. LatestWinsSpeculativeMailbox: single-slot overwrite on put_latest, discard_if_matches, pop_latest.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from dce.schemas import (
    CancelSpeculativeEvent,
    HardTurnCommitEvent,
    PauseRecord,
    SpeculativeTriggerEvent,
    VADChunkResult,
    VadConfig,
    VADState,
)
from dce.vad_endpointing import (
    DualLevelEndpointTracker,
    LatestWinsSpeculativeMailbox,
    is_speculative_result_valid,
)


def _make_pcm_20ms() -> List[float]:
    """Return 320 samples of 16kHz audio (20ms)."""
    return [0.05] * 320


# ============================================================================
# 1. Micro-Pause, Soft Pause, Noise Rejection, Hysteresis, and Hard EOT
# ============================================================================


def test_dual_level_endpointing_lifecycle() -> None:
    """Comprehensive lifecycle test covering:
    - Micro-pause (200-399ms) in SPEAKING state
    - Soft pause (400ms) speculative trigger emission
    - Noise rejection (prob 0.36 in SOFT_PAUSE does not resume)
    - Resume hysteresis (prob 0.55 resumes, cancels speculative, increments epoch, records pause)
    - Hard EOT commit at 800ms silence
    """
    cfg = VadConfig(
        sample_rate=16000,
        speech_prob_on=0.55,
        speech_prob_off=0.35,
        min_speech_ms=160.0,
        min_micro_pause_ms=200.0,
        soft_pause_ms=400.0,
        hard_eot_ms=800.0,
        min_speculative_ms=600.0,
        max_turn_ms=25000.0,
    )
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()  # 20ms chunk = 20_000_000 ns

    current_ns = 0

    # 1. Start speech: 35 chunks * 20ms = 700ms speech
    for _ in range(35):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.85, is_speech=True, is_overlap=False)
        events = tracker.process_chunk(chunk, pcm)
        assert len(events) == 0
        current_ns += 20_000_000

    assert tracker.state == VADState.SPEAKING
    assert tracker.turn_id == 1
    assert tracker.epoch_id == 1

    # 2. Intra-turn micro-pause in SPEAKING state: 13 chunks * 20ms = 260ms (200ms <= 260ms < 400ms)
    for _ in range(13):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.10, is_speech=False, is_overlap=False)
        events = tracker.process_chunk(chunk, pcm)
        assert len(events) == 0
        current_ns += 20_000_000

    assert tracker.state == VADState.SPEAKING  # Still speaking because < 400ms

    # Resume speech after 260ms micro-pause
    chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.80, is_speech=True, is_overlap=False)
    events = tracker.process_chunk(chunk, pcm)
    assert len(events) == 0
    current_ns += 20_000_000

    # Micro-pause must be recorded in intra_turn_pauses
    assert len(tracker.intra_turn_pauses) == 1
    pause1 = tracker.intra_turn_pauses[0]
    assert math.isclose(pause1.duration_ms, 260.0, abs_tol=1e-3)

    # Continue speech for 5 chunks (100ms)
    for _ in range(5):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.80, is_speech=True, is_overlap=False)
        tracker.process_chunk(chunk, pcm)
        current_ns += 20_000_000

    # 3. Enter SOFT_PAUSE: 20 chunks * 20ms = 400ms silence
    spec_events: List[SpeculativeTriggerEvent] = []
    for _ in range(20):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.10, is_speech=False, is_overlap=False)
        evs = tracker.process_chunk(chunk, pcm)
        for ev in evs:
            if isinstance(ev, SpeculativeTriggerEvent):
                spec_events.append(ev)
        current_ns += 20_000_000

    assert tracker.state == VADState.SOFT_PAUSE
    assert len(spec_events) == 1
    assert spec_events[0].turn_id == 1
    assert spec_events[0].epoch_id == 1

    # 4. Noise Rejection: feed vad_prob = 0.36 (between off=0.35 and on=0.55)
    # Must NOT resume to SPEAKING!
    noise_chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.36, is_speech=False, is_overlap=False)
    noise_events = tracker.process_chunk(noise_chunk, pcm)
    current_ns += 20_000_000
    assert len(noise_events) == 0
    assert tracker.state == VADState.SOFT_PAUSE

    # 5. Resume Hysteresis: feed vad_prob = 0.60 >= speech_prob_on (0.55)
    # Resumes to SPEAKING, emits CancelSpeculativeEvent, increments epoch_id
    resume_chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.60, is_speech=True, is_overlap=False)
    resume_events = tracker.process_chunk(resume_chunk, pcm)
    current_ns += 20_000_000

    assert tracker.state == VADState.SPEAKING
    assert len(resume_events) == 1
    assert isinstance(resume_events[0], CancelSpeculativeEvent)
    cancel_ev = resume_events[0]
    assert cancel_ev.invalidated_epoch_id == 1
    assert cancel_ev.hesitation_pause is not None
    # 20 silence chunks (400ms) + 1 noise chunk (20ms) = 420ms hesitation pause
    assert math.isclose(cancel_ev.hesitation_pause.duration_ms, 420.0, abs_tol=1e-3)
    assert tracker.epoch_id == 2
    # Verify hesitation pause is also in intra_turn_pauses
    assert len(tracker.intra_turn_pauses) == 2

    # Speak for another 5 chunks (100ms)
    for _ in range(5):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.80, is_speech=True, is_overlap=False)
        tracker.process_chunk(chunk, pcm)
        current_ns += 20_000_000

    # 6. Hard EOT Commit at 800ms silence
    # First 400ms (20 chunks) transitions to SOFT_PAUSE, emitting speculative for epoch 2
    for _ in range(20):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.10, is_speech=False, is_overlap=False)
        tracker.process_chunk(chunk, pcm)
        current_ns += 20_000_000

    assert tracker.state == VADState.SOFT_PAUSE

    # Next 400ms (20 chunks) reaches 800ms total silence -> emits HardTurnCommitEvent
    commit_events: List[HardTurnCommitEvent] = []
    for _ in range(20):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.10, is_speech=False, is_overlap=False)
        evs = tracker.process_chunk(chunk, pcm)
        for ev in evs:
            if isinstance(ev, HardTurnCommitEvent):
                commit_events.append(ev)
        current_ns += 20_000_000

    assert len(commit_events) == 1
    commit = commit_events[0]
    assert commit.turn_id == 1
    assert commit.forced_split is False
    assert len(commit.intra_turn_pauses) == 2
    assert tracker.state == VADState.IDLE


# ============================================================================
# 2. Forced Split at 25s
# ============================================================================


def test_forced_split_at_max_turn_duration() -> None:
    """Verify continuous speech exceeding max_turn_ms (25000ms) triggers forced split commit."""
    cfg = VadConfig(max_turn_ms=25000.0, min_speech_ms=160.0)
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()

    current_ns = 0
    # 25,000ms / 20ms = 1250 chunks
    # Feed 1250 chunks of continuous speech
    commits: List[HardTurnCommitEvent] = []
    for i in range(1251):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.90, is_speech=True, is_overlap=False)
        evs = tracker.process_chunk(chunk, pcm)
        for ev in evs:
            if isinstance(ev, HardTurnCommitEvent):
                commits.append(ev)
        current_ns += 20_000_000
        if len(commits) > 0:
            break

    assert len(commits) == 1
    commit = commits[0]
    assert commit.forced_split is True
    assert commit.valid_epoch_id == 1
    # Triggering speech chunk must be preserved as the first chunk of the next turn
    assert tracker.state == VADState.SPEAKING
    assert tracker.turn_id == 2
    assert tracker.voiced_chunks == 1


# ============================================================================
# 3. flush() at EOF
# ============================================================================


def test_flush_commits_trailing_turn_at_eof() -> None:
    """Verify flush() commits pending turn on EOF if speech >= min_speech_ms."""
    cfg = VadConfig(min_speech_ms=160.0)
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()

    current_ns = 0
    # 15 chunks * 20ms = 300ms speech (>= 160ms)
    for _ in range(15):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.85, is_speech=True, is_overlap=False)
        tracker.process_chunk(chunk, pcm)
        current_ns += 20_000_000

    assert tracker.state == VADState.SPEAKING

    # Media ends abruptly (EOF)
    commit = tracker.flush()
    assert commit is not None
    assert commit.turn_id == 1
    assert commit.forced_split is False
    assert tracker.state == VADState.IDLE

    # If tracker is already IDLE, flush() returns None
    assert tracker.flush() is None


def test_flush_discards_subthreshold_trailing_speech() -> None:
    """Verify flush() discards turn if speech duration < min_speech_ms (160ms)."""
    cfg = VadConfig(min_speech_ms=160.0)
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()

    current_ns = 0
    # 4 chunks * 20ms = 80ms speech (< 160ms)
    for _ in range(4):
        chunk = VADChunkResult(pts_ns=current_ns, vad_prob=0.85, is_speech=True, is_overlap=False)
        tracker.process_chunk(chunk, pcm)
        current_ns += 20_000_000

    commit = tracker.flush()
    assert commit is None
    assert tracker.state == VADState.IDLE


# ============================================================================
# 4. LatestWinsSpeculativeMailbox Tests
# ============================================================================


def test_latest_wins_speculative_mailbox() -> None:
    """Verify single-slot overwrite on put_latest, discard_if_matches, and pop_latest."""
    mailbox = LatestWinsSpeculativeMailbox()

    assert mailbox.is_empty() is True
    assert mailbox.pop_latest() is None

    ev1 = SpeculativeTriggerEvent(turn_id=1, epoch_id=1, start_ns=100, last_voiced_ns=500)
    ev2 = SpeculativeTriggerEvent(turn_id=1, epoch_id=2, start_ns=100, last_voiced_ns=800)

    # Put ev1
    mailbox.put_latest(ev1)
    assert mailbox.is_empty() is False
    assert mailbox.peek() == ev1

    # Overwrite with ev2 in single-slot
    mailbox.put_latest(ev2)
    assert mailbox.peek() == ev2

    # Discard mismatch does nothing
    assert mailbox.discard_if_matches(turn_id=99, epoch_id=1) is False
    assert mailbox.discard_if_matches(turn_id=1, epoch_id=1) is False  # ev2 is epoch 2!
    assert mailbox.peek() == ev2

    # Discard match clears mailbox
    assert mailbox.discard_if_matches(turn_id=1, epoch_id=2) is True
    assert mailbox.is_empty() is True
    assert mailbox.pop_latest() is None

    # Put ev3 and pop_latest clears mailbox
    ev3 = SpeculativeTriggerEvent(turn_id=2, epoch_id=1, start_ns=200, last_voiced_ns=700)
    mailbox.put_latest(ev3)
    popped = mailbox.pop_latest()
    assert popped == ev3
    assert mailbox.is_empty() is True
    assert mailbox.pop_latest() is None


# ============================================================================
# 5. Audit Round 1: Speculative Epoch Validation, Numpy PCM, and 20ms Frames
# ============================================================================


def test_is_speculative_result_valid() -> None:
    """Verify pure epoch/turn reconciliation helper for speculative ASR results."""
    commit = HardTurnCommitEvent(
        turn_id=3,
        valid_epoch_id=2,
        start_ns=0,
        end_ns=1_000_000_000,
    )
    assert is_speculative_result_valid(3, 2, commit) is True
    assert is_speculative_result_valid(3, 1, commit) is False
    assert is_speculative_result_valid(2, 2, commit) is False


def test_20ms_speech_then_silence_speculative_and_commit() -> None:
    """1.0s speech -> 0.9s silence: Speculative(epoch=1) at 400ms, Commit(valid_epoch_id=1) at 800ms."""
    cfg = VadConfig()
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = np.full(320, 0.05, dtype=np.float32)

    current_ns = 0
    # 1.0s speech (50 * 20ms)
    for _ in range(50):
        evs = tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.90, is_speech=True),
            pcm,
        )
        assert evs == []
        current_ns += 20_000_000

    # 0.9s silence (45 * 20ms)
    spec_at_idx = None
    commit_at_idx = None
    spec_ev: SpeculativeTriggerEvent | None = None
    commit_ev: HardTurnCommitEvent | None = None

    for idx in range(45):
        evs = tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.05, is_speech=False),
            pcm,
        )
        for ev in evs:
            if isinstance(ev, SpeculativeTriggerEvent):
                spec_at_idx = idx + 1
                spec_ev = ev
            elif isinstance(ev, HardTurnCommitEvent):
                commit_at_idx = idx + 1
                commit_ev = ev
        current_ns += 20_000_000

    assert spec_at_idx == 20  # 20 * 20ms = 400ms
    assert spec_ev is not None
    assert spec_ev.epoch_id == 1

    assert commit_at_idx == 40  # 40 * 20ms = 800ms
    assert commit_ev is not None
    assert commit_ev.valid_epoch_id == 1
    assert is_speculative_result_valid(spec_ev.turn_id, spec_ev.epoch_id, commit_ev) is True
    assert tracker.state == VADState.IDLE


def test_20ms_speech_soft_pause_resume_then_commit() -> None:
    """1.0s speech -> 0.5s silence -> 1.0s speech -> 0.9s silence."""
    cfg = VadConfig()
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = np.full(320, 0.05, dtype=np.float32)
    current_ns = 0
    all_events = []

    def _feed(n_frames: int, prob: float, speech: bool) -> None:
        nonlocal current_ns
        for _ in range(n_frames):
            evs = tracker.process_chunk(
                VADChunkResult(pts_ns=current_ns, vad_prob=prob, is_speech=speech),
                pcm,
            )
            all_events.extend(evs)
            current_ns += 20_000_000

    _feed(50, 0.90, True)   # 1.0s speech
    _feed(25, 0.05, False)  # 0.5s silence -> Speculative(1)
    _feed(50, 0.90, True)   # 1.0s speech -> CancelSpeculative(1)
    _feed(45, 0.05, False)  # 0.9s silence -> Speculative(2), HardTurnCommit(valid_epoch_id=2)

    specs = [e for e in all_events if isinstance(e, SpeculativeTriggerEvent)]
    cancels = [e for e in all_events if isinstance(e, CancelSpeculativeEvent)]
    commits = [e for e in all_events if isinstance(e, HardTurnCommitEvent)]

    assert [s.epoch_id for s in specs] == [1, 2]
    assert len(cancels) == 1
    assert cancels[0].invalidated_epoch_id == 1
    assert len(commits) == 1
    assert commits[0].valid_epoch_id == 2
    assert len(commits[0].intra_turn_pauses) == 1
    assert math.isclose(commits[0].intra_turn_pauses[0].duration_ms, 500.0, abs_tol=1e-3)
    assert is_speculative_result_valid(specs[0].turn_id, specs[0].epoch_id, commits[0]) is False
    assert is_speculative_result_valid(specs[1].turn_id, specs[1].epoch_id, commits[0]) is True


def test_20ms_micro_pause_240ms_no_speculative() -> None:
    """~250ms silence (12 * 20ms = 240ms) then resume: no Speculative, 1 pause record."""
    cfg = VadConfig()
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()
    current_ns = 0
    all_events = []

    for _ in range(35):  # 700ms speech
        all_events.extend(
            tracker.process_chunk(VADChunkResult(pts_ns=current_ns, vad_prob=0.85, is_speech=True), pcm)
        )
        current_ns += 20_000_000

    for _ in range(12):  # 240ms silence (>= 200ms, < 400ms)
        all_events.extend(
            tracker.process_chunk(VADChunkResult(pts_ns=current_ns, vad_prob=0.05, is_speech=False), pcm)
        )
        current_ns += 20_000_000

    # Resume speech
    all_events.extend(
        tracker.process_chunk(VADChunkResult(pts_ns=current_ns, vad_prob=0.85, is_speech=True), pcm)
    )

    assert [e for e in all_events if isinstance(e, SpeculativeTriggerEvent)] == []
    assert len(tracker.intra_turn_pauses) == 1
    assert math.isclose(tracker.intra_turn_pauses[0].duration_ms, 240.0, abs_tol=1e-3)


def test_20ms_short_100ms_speech_does_not_commit() -> None:
    """100ms speech (< min_speech_ms=160ms) followed by 0.9s silence does not commit."""
    cfg = VadConfig()
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm = _make_pcm_20ms()
    current_ns = 0
    commits: List[HardTurnCommitEvent] = []

    for _ in range(5):  # 100ms speech
        evs = tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.90, is_speech=True),
            pcm,
        )
        commits.extend(e for e in evs if isinstance(e, HardTurnCommitEvent))
        current_ns += 20_000_000

    for _ in range(45):  # 0.9s silence
        evs = tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.05, is_speech=False),
            pcm,
        )
        commits.extend(e for e in evs if isinstance(e, HardTurnCommitEvent))
        current_ns += 20_000_000

    assert commits == []
    assert tracker.state == VADState.IDLE
    assert tracker.flush() is None


def test_max_turn_preserves_triggering_speech_chunk_for_next_turn() -> None:
    """Forced split at 25s preserves triggering speech chunk as first chunk of turn 2."""
    cfg = VadConfig(max_turn_ms=25000.0, min_speech_ms=160.0)
    tracker = DualLevelEndpointTracker(config=cfg)
    pcm_turn1 = np.full(320, 0.1, dtype=np.float32)
    pcm_trigger = np.full(320, 0.75, dtype=np.float32)

    current_ns = 0
    commits: List[HardTurnCommitEvent] = []
    for i in range(1250):
        chunk_pcm = pcm_trigger if i == 1249 else pcm_turn1
        evs = tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.90, is_speech=True),
            chunk_pcm,
        )
        commits.extend(e for e in evs if isinstance(e, HardTurnCommitEvent))
        current_ns += 20_000_000

    assert len(commits) == 1
    assert commits[0].turn_id == 1
    assert commits[0].valid_epoch_id == 1
    assert commits[0].forced_split is True
    assert tracker.state == VADState.SPEAKING
    assert tracker.turn_id == 2
    assert tracker.voiced_chunks == 1

    # Feed 7 more speech chunks (total 8 * 20ms = 160ms >= min_speech_ms) then flush
    for _ in range(7):
        tracker.process_chunk(
            VADChunkResult(pts_ns=current_ns, vad_prob=0.90, is_speech=True),
            pcm_turn1,
        )
        current_ns += 20_000_000

    commit2 = tracker.flush()
    assert commit2 is not None
    assert commit2.turn_id == 2
    assert commit2.valid_epoch_id == 1
    assert commit2.audio_pcm16k is not None
    assert len(commit2.audio_pcm16k) == 8 * 320
    # First 320 samples must come from the preserved triggering chunk
    assert math.isclose(commit2.audio_pcm16k[0], 0.75, abs_tol=1e-6)
