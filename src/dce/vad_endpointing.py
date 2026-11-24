"""Discourse Consistency Engine - Dual-Level VAD and Speculative Endpointing.

Implements sample-accurate PCM buffering, micro-pause tracking in both speaking
and soft pause states, speculative ASR triggers, hysteresis resume, hard turn
boundary commits, EOF flushing, and a single-slot LatestWinsSpeculativeMailbox.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence, Union

from dce.schemas import (
    CancelSpeculativeEvent,
    HardTurnCommitEvent,
    PauseRecord,
    SpeculativeTriggerEvent,
    VADChunkResult,
    VadConfig,
    VADState,
)

logger = logging.getLogger(__name__)


class DualLevelEndpointTracker:
    """State machine tracker for speech turns, speculative triggers, and micro-pauses."""

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        """Initialize DualLevelEndpointTracker.

        Args:
            config: VAD and turn endpointing configuration parameters.
        """
        self._config: VadConfig = config if config is not None else VadConfig()
        self._state: VADState = VADState.IDLE

        self._turn_id: int = 0
        self._epoch_id: int = 0
        self._turn_start_ns: int = 0
        self._last_voiced_end_ns: int = 0

        self._voiced_chunks: int = 0
        self._overlap_voiced_chunks: int = 0

        # Sample-accurate audio chunk buffering
        self._audio_chunks: List[List[float]] = []
        self._last_voiced_chunk_idx: int = -1

        # Silence and pause tracking
        self._silence_start_ns: Optional[int] = None
        self._speculative_emitted_this_pause: bool = False
        self._intra_turn_pauses: List[PauseRecord] = []

        self._last_speculative_event: Optional[SpeculativeTriggerEvent] = None
        self._last_commit_event: Optional[HardTurnCommitEvent] = None

    @property
    def config(self) -> VadConfig:
        """Return VAD configuration."""
        return self._config

    @property
    def state(self) -> VADState:
        """Return current operational state (IDLE, SPEAKING, SOFT_PAUSE)."""
        return self._state

    @property
    def turn_id(self) -> int:
        """Return active turn identifier."""
        return self._turn_id

    @property
    def epoch_id(self) -> int:
        """Return active speculative epoch identifier."""
        return self._epoch_id

    @property
    def intra_turn_pauses(self) -> List[PauseRecord]:
        """Return intra-turn micro-pause records accumulated so far."""
        return list(self._intra_turn_pauses)

    @property
    def voiced_chunks(self) -> int:
        """Return count of voiced chunks in the current turn."""
        return self._voiced_chunks

    @property
    def overlap_voiced_chunks(self) -> int:
        """Return count of overlap voiced chunks in the current turn."""
        return self._overlap_voiced_chunks

    @property
    def last_speculative_event(self) -> Optional[SpeculativeTriggerEvent]:
        """Return most recently emitted SpeculativeTriggerEvent."""
        return self._last_speculative_event

    @property
    def last_commit_event(self) -> Optional[HardTurnCommitEvent]:
        """Return most recently emitted HardTurnCommitEvent."""
        return self._last_commit_event

    def _get_trimmed_pcm(self) -> List[float]:
        """Return flattened PCM audio trimmed up to last_voiced_chunk_idx."""
        if self._last_voiced_chunk_idx < 0 or not self._audio_chunks:
            return []
        out: List[float] = []
        limit = min(self._last_voiced_chunk_idx + 1, len(self._audio_chunks))
        for chunk in self._audio_chunks[:limit]:
            out.extend(chunk)
        return out

    def _reset(self) -> None:
        """Reset internal buffers and return tracker to IDLE state."""
        self._state = VADState.IDLE
        self._turn_start_ns = 0
        self._last_voiced_end_ns = 0
        self._voiced_chunks = 0
        self._overlap_voiced_chunks = 0
        self._audio_chunks.clear()
        self._last_voiced_chunk_idx = -1
        self._silence_start_ns = None
        self._speculative_emitted_this_pause = False
        self._intra_turn_pauses.clear()

    def process_chunk(
        self,
        chunk: VADChunkResult,
        audio_chunk_pcm: Optional[Sequence[float]] = None,
    ) -> List[Union[SpeculativeTriggerEvent, CancelSpeculativeEvent, HardTurnCommitEvent]]:
        """Process a streaming audio chunk and return any emitted events.

        Args:
            chunk: VAD evaluation result containing vad_prob, pts_ns, and is_overlap.
            audio_chunk_pcm: Raw 16kHz float32 audio samples for this chunk.

        Returns:
            List of generated events (triggers, cancellations, or commits).
        """
        emitted_events: List[
            Union[SpeculativeTriggerEvent, CancelSpeculativeEvent, HardTurnCommitEvent]
        ] = []
        pcm_list = list(audio_chunk_pcm) if audio_chunk_pcm is not None else []

        # Determine chunk duration in nanoseconds
        if len(pcm_list) > 0:
            chunk_duration_ns = int((len(pcm_list) / float(self._config.sample_rate)) * 1_000_000_000)
        else:
            chunk_duration_ns = 20_000_000  # Default 20ms fallback

        chunk_end_ns = chunk.pts_ns + chunk_duration_ns

        # --------------------------------------------------------------------
        # 1. IDLE State Handling
        # --------------------------------------------------------------------
        if self._state == VADState.IDLE:
            if chunk.vad_prob >= self._config.speech_prob_on:
                self._state = VADState.SPEAKING
                self._turn_id += 1
                self._epoch_id = 1
                self._turn_start_ns = chunk.pts_ns
                self._last_voiced_end_ns = chunk_end_ns
                self._voiced_chunks = 1
                self._overlap_voiced_chunks = 1 if chunk.is_overlap else 0
                self._audio_chunks = [pcm_list]
                self._last_voiced_chunk_idx = 0
                self._silence_start_ns = None
                self._speculative_emitted_this_pause = False
                self._intra_turn_pauses = []
            return emitted_events

        # --------------------------------------------------------------------
        # 2. Check Maximum Turn Duration (Forced Split)
        # --------------------------------------------------------------------
        current_turn_duration_ms = (chunk_end_ns - self._turn_start_ns) / 1_000_000.0
        if current_turn_duration_ms >= self._config.max_turn_ms:
            # Commit turn immediately due to max duration
            trimmed_pcm = self._get_trimmed_pcm()
            overlap_ratio = float(self._overlap_voiced_chunks) / float(max(self._voiced_chunks, 1))
            overlap_flag = bool(overlap_ratio >= self._config.overlap_ratio_threshold)

            commit_event = HardTurnCommitEvent(
                turn_id=self._turn_id,
                start_ns=self._turn_start_ns,
                end_ns=self._last_voiced_end_ns,
                audio_pcm16k=trimmed_pcm,
                voiced_chunks=self._voiced_chunks,
                overlap_voiced_chunks=self._overlap_voiced_chunks,
                overlap_ratio=overlap_ratio,
                overlap_flag=overlap_flag,
                forced_split=True,
                intra_turn_pauses=list(self._intra_turn_pauses),
            )
            self._last_commit_event = commit_event
            emitted_events.append(commit_event)
            self._reset()
            return emitted_events

        # Always buffer audio during active turn
        self._audio_chunks.append(pcm_list)

        # --------------------------------------------------------------------
        # 3. SPEAKING State Handling
        # --------------------------------------------------------------------
        if self._state == VADState.SPEAKING:
            if chunk.vad_prob >= self._config.speech_prob_off:
                # Active speech frame
                if self._silence_start_ns is not None:
                    # Silence just concluded
                    pause_ms = (chunk.pts_ns - self._silence_start_ns) / 1_000_000.0
                    if pause_ms >= self._config.min_micro_pause_ms:
                        self._intra_turn_pauses.append(
                            PauseRecord(start_ns=self._silence_start_ns, duration_ms=pause_ms)
                        )
                    self._silence_start_ns = None

                self._last_voiced_end_ns = chunk_end_ns
                self._voiced_chunks += 1
                if chunk.is_overlap:
                    self._overlap_voiced_chunks += 1
                self._last_voiced_chunk_idx = len(self._audio_chunks) - 1

            else:
                # Silence frame in SPEAKING
                if self._silence_start_ns is None:
                    self._silence_start_ns = chunk.pts_ns

                silence_ms = (chunk_end_ns - self._silence_start_ns) / 1_000_000.0

                # Transition to SOFT_PAUSE at soft_pause_ms (400ms)
                if silence_ms >= self._config.soft_pause_ms:
                    self._state = VADState.SOFT_PAUSE
                    speech_duration_ms = (
                        self._last_voiced_end_ns - self._turn_start_ns
                    ) / 1_000_000.0

                    if (
                        speech_duration_ms >= self._config.min_speculative_ms
                        and not self._speculative_emitted_this_pause
                    ):
                        trimmed_pcm = self._get_trimmed_pcm()
                        spec_event = SpeculativeTriggerEvent(
                            turn_id=self._turn_id,
                            epoch_id=self._epoch_id,
                            start_ns=self._turn_start_ns,
                            last_voiced_ns=self._last_voiced_end_ns,
                            audio_pcm16k=trimmed_pcm,
                        )
                        self._last_speculative_event = spec_event
                        emitted_events.append(spec_event)
                        self._speculative_emitted_this_pause = True

            return emitted_events

        # --------------------------------------------------------------------
        # 4. SOFT_PAUSE State Handling
        # --------------------------------------------------------------------
        if self._state == VADState.SOFT_PAUSE:
            if chunk.vad_prob >= self._config.speech_prob_on:
                # Resume speech: Hysteresis exit back to SPEAKING
                silence_ms = 0.0
                hesitation: Optional[PauseRecord] = None
                if self._silence_start_ns is not None:
                    silence_ms = (chunk.pts_ns - self._silence_start_ns) / 1_000_000.0
                    if silence_ms >= self._config.min_micro_pause_ms:
                        hesitation = PauseRecord(
                            start_ns=self._silence_start_ns, duration_ms=silence_ms
                        )
                        self._intra_turn_pauses.append(hesitation)

                if self._speculative_emitted_this_pause:
                    cancel_event = CancelSpeculativeEvent(
                        turn_id=self._turn_id,
                        invalidated_epoch_id=self._epoch_id,
                        hesitation_pause=hesitation,
                    )
                    emitted_events.append(cancel_event)
                    self._epoch_id += 1
                    self._speculative_emitted_this_pause = False

                self._state = VADState.SPEAKING
                self._silence_start_ns = None
                self._last_voiced_end_ns = chunk_end_ns
                self._voiced_chunks += 1
                if chunk.is_overlap:
                    self._overlap_voiced_chunks += 1
                self._last_voiced_chunk_idx = len(self._audio_chunks) - 1

            else:
                # Continued silence or low-prob noise (0.35 to 0.54) -> Stay in SOFT_PAUSE
                if self._silence_start_ns is None:
                    self._silence_start_ns = chunk.pts_ns

                silence_ms = (chunk_end_ns - self._silence_start_ns) / 1_000_000.0

                # Check Hard EOT threshold (800ms)
                if silence_ms >= self._config.hard_eot_ms:
                    speech_duration_ms = (
                        self._last_voiced_end_ns - self._turn_start_ns
                    ) / 1_000_000.0

                    if speech_duration_ms >= self._config.min_speech_ms:
                        trimmed_pcm = self._get_trimmed_pcm()
                        overlap_ratio = float(self._overlap_voiced_chunks) / float(
                            max(self._voiced_chunks, 1)
                        )
                        overlap_flag = bool(overlap_ratio >= self._config.overlap_ratio_threshold)

                        commit_event = HardTurnCommitEvent(
                            turn_id=self._turn_id,
                            start_ns=self._turn_start_ns,
                            end_ns=self._last_voiced_end_ns,
                            audio_pcm16k=trimmed_pcm,
                            voiced_chunks=self._voiced_chunks,
                            overlap_voiced_chunks=self._overlap_voiced_chunks,
                            overlap_ratio=overlap_ratio,
                            overlap_flag=overlap_flag,
                            forced_split=False,
                            intra_turn_pauses=list(self._intra_turn_pauses),
                        )
                        self._last_commit_event = commit_event
                        emitted_events.append(commit_event)

                    self._reset()

            return emitted_events

        return emitted_events

    def flush(self) -> Optional[HardTurnCommitEvent]:
        """Commit and finalize the pending turn at end of file (EOF).

        Returns:
            HardTurnCommitEvent if speech met min_speech_ms, else None.
        """
        if self._state in (VADState.SPEAKING, VADState.SOFT_PAUSE):
            speech_duration_ms = (self._last_voiced_end_ns - self._turn_start_ns) / 1_000_000.0
            if speech_duration_ms >= self._config.min_speech_ms:
                trimmed_pcm = self._get_trimmed_pcm()
                overlap_ratio = float(self._overlap_voiced_chunks) / float(
                    max(self._voiced_chunks, 1)
                )
                overlap_flag = bool(overlap_ratio >= self._config.overlap_ratio_threshold)

                commit_event = HardTurnCommitEvent(
                    turn_id=self._turn_id,
                    start_ns=self._turn_start_ns,
                    end_ns=self._last_voiced_end_ns,
                    audio_pcm16k=trimmed_pcm,
                    voiced_chunks=self._voiced_chunks,
                    overlap_voiced_chunks=self._overlap_voiced_chunks,
                    overlap_ratio=overlap_ratio,
                    overlap_flag=overlap_flag,
                    forced_split=False,
                    intra_turn_pauses=list(self._intra_turn_pauses),
                )
                self._last_commit_event = commit_event
                self._reset()
                return commit_event

            self._reset()
            return None

        return None


class LatestWinsSpeculativeMailbox:
    """Thread-safe single-slot mailbox for managing speculative ASR triggers."""

    def __init__(self) -> None:
        """Initialize empty mailbox."""
        self._item: Optional[SpeculativeTriggerEvent] = None
        self._lock: threading.Lock = threading.Lock()

    def put_latest(self, event: SpeculativeTriggerEvent) -> None:
        """Enqueue new speculative event, unconditionally overwriting any unconsumed event."""
        with self._lock:
            self._item = event

    def discard_if_matches(self, turn_id: int, epoch_id: int) -> bool:
        """Discard held event if its turn_id and epoch_id match the cancellation target.

        Args:
            turn_id: Target turn ID to discard.
            epoch_id: Target epoch ID to discard.

        Returns:
            True if matching event was found and discarded, False otherwise.
        """
        with self._lock:
            if (
                self._item is not None
                and self._item.turn_id == turn_id
                and self._item.epoch_id == epoch_id
            ):
                self._item = None
                return True
            return False

    def pop_latest(self) -> Optional[SpeculativeTriggerEvent]:
        """Atomically retrieve and clear the latest speculative trigger event."""
        with self._lock:
            item = self._item
            self._item = None
            return item

    def peek(self) -> Optional[SpeculativeTriggerEvent]:
        """Inspect the current mailbox item without consuming it."""
        with self._lock:
            return self._item

    def is_empty(self) -> bool:
        """Return True if mailbox currently has no queued item."""
        with self._lock:
            return self._item is None
