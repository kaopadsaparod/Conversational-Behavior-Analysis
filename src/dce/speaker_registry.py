"""Discourse Consistency Engine - Speaker Registry and Identification.

Maintains duration-weighted unit centroid speaker profiles, 1-to-1 face hint bindings,
provisional speaker buffer promotion for short/long turns, and bounded capacity
to prevent zombie speaker proliferation.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dce.schemas import SpeakerAssignment, SpeakerRegistryConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpeakerProfile:
    """Internal representation of a confirmed permanent speaker."""

    speaker_id: str
    centroid: np.ndarray  # 1D unit vector
    total_duration_sec: float
    last_seen_sec: float
    face_hint: Optional[str] = None


@dataclasses.dataclass
class ProvisionalSpeaker:
    """Internal representation of a provisional candidate speaker buffer."""

    prov_id: str
    centroid: np.ndarray  # 1D unit vector
    total_duration_sec: float
    last_seen_sec: float


class SpeakerRegistry:
    """Adaptive speaker clustering registry with duration-weighted unit centroid updates."""

    def __init__(self, config: Optional[SpeakerRegistryConfig] = None) -> None:
        """Initialize SpeakerRegistry.

        Args:
            config: Speaker registry configuration options.
        """
        self._config: SpeakerRegistryConfig = (
            config if config is not None else SpeakerRegistryConfig()
        )
        self._speakers: Dict[str, SpeakerProfile] = {}
        self._provisional: Dict[str, ProvisionalSpeaker] = {}
        self._next_speaker_idx: int = 0
        self._next_prov_idx: int = 1

    @property
    def config(self) -> SpeakerRegistryConfig:
        """Return speaker registry configuration."""
        return self._config

    @property
    def speakers(self) -> Dict[str, SpeakerProfile]:
        """Return dictionary of permanent speaker profiles."""
        return self._speakers

    @property
    def provisional(self) -> Dict[str, ProvisionalSpeaker]:
        """Return dictionary of active provisional candidate buffers."""
        return self._provisional

    def _normalize_embedding(
        self, embedding: Union[Sequence[float], np.ndarray]
    ) -> Optional[np.ndarray]:
        """Validate and normalize embedding vector to unit L2 length.

        Returns None if embedding is invalid, non-finite, or has norm < min_embedding_norm.
        """
        if embedding is None:
            return None

        arr = np.asarray(embedding, dtype=np.float64).flatten()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None

        norm = float(np.linalg.norm(arr))
        if norm < self._config.min_embedding_norm:
            return None

        return arr / norm

    def _cosine_similarity(self, unit_a: np.ndarray, unit_b: np.ndarray) -> float:
        """Compute cosine similarity between two unit-normalized vectors."""
        dot = float(np.dot(unit_a, unit_b))
        return max(-1.0, min(1.0, dot))

    def _purge_expired_provisional(self, current_time_sec: float) -> None:
        """Remove provisional speaker buffers that exceeded provisional_ttl_sec."""
        ttl = self._config.provisional_ttl_sec
        expired_ids = [
            pid
            for pid, prov in self._provisional.items()
            if (current_time_sec - prov.last_seen_sec) > ttl
        ]
        for pid in expired_ids:
            del self._provisional[pid]

    def _bind_face_hint(self, speaker_id: str, face_hint: Optional[str]) -> None:
        """Enforce strict 1-to-1 binding between face_hint and speaker_id."""
        if face_hint is None:
            return
        # Unlink face_hint from any other speaker
        for spk in self._speakers.values():
            if spk.speaker_id != speaker_id and spk.face_hint == face_hint:
                spk.face_hint = None
        if speaker_id in self._speakers:
            self._speakers[speaker_id].face_hint = face_hint

    def _find_speaker_by_face(self, face_hint: str) -> Optional[SpeakerProfile]:
        """Find permanent speaker currently bound to the given face_hint."""
        for spk in self._speakers.values():
            if spk.face_hint == face_hint:
                return spk
        return None

    def _find_closest_permanent(
        self, unit_emb: np.ndarray
    ) -> Tuple[float, Optional[SpeakerProfile]]:
        """Find the permanent speaker with the highest cosine similarity."""
        if not self._speakers:
            return 0.0, None
        best_sim = -2.0
        best_spk: Optional[SpeakerProfile] = None
        for spk in self._speakers.values():
            sim = self._cosine_similarity(spk.centroid, unit_emb)
            if sim > best_sim:
                best_sim = sim
                best_spk = spk
        return best_sim, best_spk

    def _find_closest_provisional(
        self, unit_emb: np.ndarray
    ) -> Tuple[float, Optional[ProvisionalSpeaker]]:
        """Find the provisional speaker buffer with the highest cosine similarity."""
        if not self._provisional:
            return 0.0, None
        best_sim = -2.0
        best_prov: Optional[ProvisionalSpeaker] = None
        for prov in self._provisional.values():
            sim = self._cosine_similarity(prov.centroid, unit_emb)
            if sim > best_sim:
                best_sim = sim
                best_prov = prov
        return best_sim, best_prov

    def _update_permanent_centroid(
        self, spk: SpeakerProfile, unit_emb: np.ndarray, duration_sec: float
    ) -> None:
        """Update permanent speaker centroid with duration weighting: s_new = D_old*c + d*e."""
        s_new = spk.total_duration_sec * spk.centroid + duration_sec * unit_emb
        norm = float(np.linalg.norm(s_new))
        if norm > 1e-12:
            spk.centroid = s_new / norm
        spk.total_duration_sec += duration_sec

    def assign(
        self,
        embedding: Union[Sequence[float], np.ndarray],
        duration_sec: float,
        timestamp_sec: float,
        overlap_flag: bool = False,
        face_hint: Optional[str] = None,
    ) -> SpeakerAssignment:
        """Assign utterance to a speaker profile, updating centroids or provisional buffers.

        Follows the strict decision table defined in Directive §4.2:
          1. Zero-norm / non-finite guard -> UNKNOWN_INVALID with low_confidence=True.
          2. Purge provisional buffers exceeding TTL.
          3. Overlap utterances -> match closest permanent or UNKNOWN_OVERLAP, no updates.
          4. Clean long turns (>=2.0s):
             - sim >= sim_high -> match & update centroid (bind face_hint).
             - sim_low <= sim < sim_high -> match without centroid update.
             - sim < sim_low:
               - if capacity available: check provisional match to promote, or create new permanent.
               - if capacity full: match closest permanent with low_confidence=True (no zombie).
          5. Clean short turns (<2.0s):
             - face_hint match with sim >= face_sim_floor -> match without update.
             - permanent match with sim >= sim_low -> match without update.
             - if capacity full: match closest permanent (collapse provisional if it reached 2.0s).
             - if capacity available: update or create provisional buffer (promote if >=2.0s).

        Args:
            embedding: Speaker voice embedding vector.
            duration_sec: Utterance speech duration in seconds.
            timestamp_sec: Engine timestamp in seconds.
            overlap_flag: Whether the utterance overlaps with another speaker.
            face_hint: Optional face identity hint from visual tracking.

        Returns:
            SpeakerAssignment dataclass instance.
        """
        # Step 1: Zero-norm & Finite Guard
        unit_emb = self._normalize_embedding(embedding)
        if unit_emb is None:
            return SpeakerAssignment(
                speaker_id="UNKNOWN_INVALID",
                similarity=0.0,
                is_provisional=False,
                low_confidence=True,
                promoted_from=None,
            )

        # Step 2: Purge expired provisional speakers
        self._purge_expired_provisional(timestamp_sec)

        best_perm_sim, best_perm_spk = self._find_closest_permanent(unit_emb)

        # Step 3: Overlap turn handling
        if overlap_flag:
            if best_perm_spk is not None and best_perm_sim >= self._config.sim_low:
                return SpeakerAssignment(
                    speaker_id=best_perm_spk.speaker_id,
                    similarity=best_perm_sim,
                    is_provisional=False,
                    low_confidence=True,
                    promoted_from=None,
                )
            return SpeakerAssignment(
                speaker_id="UNKNOWN_OVERLAP",
                similarity=best_perm_sim,
                is_provisional=False,
                low_confidence=True,
                promoted_from=None,
            )

        # Step 4: Long turn handling (duration >= min_confirm_sec = 2.0s)
        if duration_sec >= self._config.min_confirm_sec:
            if best_perm_spk is not None and best_perm_sim >= self._config.sim_high:
                self._update_permanent_centroid(best_perm_spk, unit_emb, duration_sec)
                best_perm_spk.last_seen_sec = timestamp_sec
                self._bind_face_hint(best_perm_spk.speaker_id, face_hint)
                return SpeakerAssignment(
                    speaker_id=best_perm_spk.speaker_id,
                    similarity=best_perm_sim,
                    is_provisional=False,
                    low_confidence=False,
                    promoted_from=None,
                )

            if best_perm_spk is not None and best_perm_sim >= self._config.sim_low:
                best_perm_spk.last_seen_sec = timestamp_sec
                if best_perm_spk.face_hint is None:
                    self._bind_face_hint(best_perm_spk.speaker_id, face_hint)
                return SpeakerAssignment(
                    speaker_id=best_perm_spk.speaker_id,
                    similarity=best_perm_sim,
                    is_provisional=False,
                    low_confidence=False,
                    promoted_from=None,
                )

            # best_perm_sim < sim_low (or no permanent speakers yet)
            if len(self._speakers) < self._config.max_speakers:
                best_prov_sim, best_prov = self._find_closest_provisional(unit_emb)
                if best_prov is not None and best_prov_sim >= self._config.sim_low:
                    # Merge provisional buffer into brand new permanent speaker
                    new_spk_id = f"SPEAKER_{self._next_speaker_idx:02d}"
                    self._next_speaker_idx += 1
                    s_new = (
                        best_prov.total_duration_sec * best_prov.centroid
                        + duration_sec * unit_emb
                    )
                    norm = float(np.linalg.norm(s_new))
                    centroid_new = s_new / norm if norm > 1e-12 else unit_emb
                    total_dur = best_prov.total_duration_sec + duration_sec

                    self._speakers[new_spk_id] = SpeakerProfile(
                        speaker_id=new_spk_id,
                        centroid=centroid_new,
                        total_duration_sec=total_dur,
                        last_seen_sec=timestamp_sec,
                        face_hint=face_hint,
                    )
                    promoted_prov_id = best_prov.prov_id
                    del self._provisional[promoted_prov_id]
                    self._bind_face_hint(new_spk_id, face_hint)

                    return SpeakerAssignment(
                        speaker_id=new_spk_id,
                        similarity=best_prov_sim,
                        is_provisional=False,
                        low_confidence=False,
                        promoted_from=promoted_prov_id,
                    )

                # Create brand new permanent speaker
                new_spk_id = f"SPEAKER_{self._next_speaker_idx:02d}"
                self._next_speaker_idx += 1
                self._speakers[new_spk_id] = SpeakerProfile(
                    speaker_id=new_spk_id,
                    centroid=np.copy(unit_emb),
                    total_duration_sec=duration_sec,
                    last_seen_sec=timestamp_sec,
                    face_hint=face_hint,
                )
                self._bind_face_hint(new_spk_id, face_hint)
                return SpeakerAssignment(
                    speaker_id=new_spk_id,
                    similarity=1.0,
                    is_provisional=False,
                    low_confidence=False,
                    promoted_from=None,
                )

            # Max permanent speakers reached: assign closest without updating
            fallback_id = best_perm_spk.speaker_id if best_perm_spk is not None else "UNKNOWN_INVALID"
            return SpeakerAssignment(
                speaker_id=fallback_id,
                similarity=best_perm_sim,
                is_provisional=False,
                low_confidence=True,
                promoted_from=None,
            )

        # Step 5: Short turn handling (duration < min_confirm_sec = 2.0s)
        # Check face_hint shortcut to permanent speaker
        if face_hint is not None:
            bound_spk = self._find_speaker_by_face(face_hint)
            if bound_spk is not None:
                face_sim = self._cosine_similarity(bound_spk.centroid, unit_emb)
                if face_sim >= self._config.face_sim_floor:
                    bound_spk.last_seen_sec = timestamp_sec
                    return SpeakerAssignment(
                        speaker_id=bound_spk.speaker_id,
                        similarity=face_sim,
                        is_provisional=False,
                        low_confidence=False,
                        promoted_from=None,
                    )

        # Check closest permanent speaker
        if best_perm_spk is not None and best_perm_sim >= self._config.sim_low:
            best_perm_spk.last_seen_sec = timestamp_sec
            return SpeakerAssignment(
                speaker_id=best_perm_spk.speaker_id,
                similarity=best_perm_sim,
                is_provisional=False,
                low_confidence=False,
                promoted_from=None,
            )

        # If capacity full: no new provisional created
        if len(self._speakers) >= self._config.max_speakers:
            fallback_id = best_perm_spk.speaker_id if best_perm_spk is not None else "UNKNOWN_INVALID"
            best_prov_sim, best_prov = self._find_closest_provisional(unit_emb)
            if best_prov is not None and best_prov_sim >= self._config.sim_low:
                s_new = best_prov.total_duration_sec * best_prov.centroid + duration_sec * unit_emb
                norm = float(np.linalg.norm(s_new))
                if norm > 1e-12:
                    best_prov.centroid = s_new / norm
                best_prov.total_duration_sec += duration_sec
                best_prov.last_seen_sec = timestamp_sec

                if best_prov.total_duration_sec >= self._config.min_confirm_sec:
                    p_id = best_prov.prov_id
                    del self._provisional[p_id]
                    return SpeakerAssignment(
                        speaker_id=fallback_id,
                        similarity=best_prov_sim,
                        is_provisional=False,
                        low_confidence=True,
                        promoted_from=p_id,
                    )
            return SpeakerAssignment(
                speaker_id=fallback_id,
                similarity=best_perm_sim,
                is_provisional=False,
                low_confidence=True,
                promoted_from=None,
            )

        # Capacity available: update or create provisional speaker buffer
        best_prov_sim, best_prov = self._find_closest_provisional(unit_emb)
        if best_prov is not None and best_prov_sim >= self._config.sim_low:
            s_new = best_prov.total_duration_sec * best_prov.centroid + duration_sec * unit_emb
            norm = float(np.linalg.norm(s_new))
            if norm > 1e-12:
                best_prov.centroid = s_new / norm
            best_prov.total_duration_sec += duration_sec
            best_prov.last_seen_sec = timestamp_sec

            # Check promotion threshold
            if best_prov.total_duration_sec >= self._config.min_confirm_sec:
                new_spk_id = f"SPEAKER_{self._next_speaker_idx:02d}"
                self._next_speaker_idx += 1
                self._speakers[new_spk_id] = SpeakerProfile(
                    speaker_id=new_spk_id,
                    centroid=np.copy(best_prov.centroid),
                    total_duration_sec=best_prov.total_duration_sec,
                    last_seen_sec=timestamp_sec,
                    face_hint=face_hint,
                )
                promoted_id = best_prov.prov_id
                del self._provisional[promoted_id]
                self._bind_face_hint(new_spk_id, face_hint)

                return SpeakerAssignment(
                    speaker_id=new_spk_id,
                    similarity=best_prov_sim,
                    is_provisional=False,
                    low_confidence=False,
                    promoted_from=promoted_id,
                )

            return SpeakerAssignment(
                speaker_id=best_prov.prov_id,
                similarity=best_prov_sim,
                is_provisional=True,
                low_confidence=False,
                promoted_from=None,
            )

        # Create new provisional buffer
        if len(self._provisional) >= self._config.max_provisional_buffers:
            oldest_pid = min(self._provisional.keys(), key=lambda k: self._provisional[k].last_seen_sec)
            del self._provisional[oldest_pid]

        new_prov_id = f"PROV_{self._next_prov_idx:02d}"
        self._next_prov_idx += 1
        self._provisional[new_prov_id] = ProvisionalSpeaker(
            prov_id=new_prov_id,
            centroid=np.copy(unit_emb),
            total_duration_sec=duration_sec,
            last_seen_sec=timestamp_sec,
        )

        return SpeakerAssignment(
            speaker_id=new_prov_id,
            similarity=1.0,
            is_provisional=True,
            low_confidence=False,
            promoted_from=None,
        )
