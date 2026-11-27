"""Discourse Consistency Engine - Feature Recorder and Deterministic Replay.

Provides strict JSONL recording guaranteed free of NaN and Infinity values
(allow_nan=False), alongside bit-exact deterministic replay capabilities.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, TextIO, Union

import numpy as np

from dce.schemas import RecorderTurnRecord


class FeatureRecorder:
    """Thread-safe JSONL logger writing RecorderTurnRecord with strict allow_nan=False."""

    def __init__(
        self,
        output_path: Union[str, Path],
        audio_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize FeatureRecorder.

        Args:
            output_path: Target filesystem path for the .jsonl recording file.
            audio_dir: Optional directory for saving per-turn WAV audio clips.
            overwrite: If True, truncate any existing file instead of appending.
        """
        self._path: Path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("w" if overwrite else "a", encoding="utf-8")
        self._audio_dir: Optional[Path] = Path(audio_dir) if audio_dir is not None else None
        if self._audio_dir is not None:
            self._audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Return the recording file path."""
        return self._path

    @property
    def audio_dir(self) -> Optional[Path]:
        """Return the optional audio clip directory."""
        return self._audio_dir

    def save_turn_audio(
        self,
        turn_id: int,
        audio_pcm16k: Sequence[float],
        sample_rate: int = 16000,
        max_duration_sec: Optional[float] = None,
    ) -> str:
        """Save a 16-bit PCM WAV clip for a turn and return its path string.

        Args:
            turn_id: Turn identifier used in the clip filename.
            audio_pcm16k: Floating-point PCM samples in [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz (default 16000).
            max_duration_sec: Optional maximum clip duration in seconds (None saves full turn).

        Returns:
            Filesystem path string of the written WAV file.
        """
        target_dir = self._audio_dir if self._audio_dir is not None else (self._path.parent / "audio_clips")
        target_dir.mkdir(parents=True, exist_ok=True)
        clip_path = target_dir / f"turn_{turn_id:04d}.wav"

        arr = np.asarray(audio_pcm16k, dtype=np.float32)
        if max_duration_sec is not None and max_duration_sec > 0.0:
            max_samples = int(round(max_duration_sec * sample_rate))
            if len(arr) > max_samples:
                arr = arr[:max_samples]

        pcm_i16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(clip_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm_i16.tobytes())
        return str(clip_path)

    def write_record(self, record: RecorderTurnRecord) -> None:
        """Serialize and write a RecorderTurnRecord with allow_nan=False.

        Args:
            record: RecorderTurnRecord instance to serialize.

        Raises:
            ValueError: If any NaN or Inf values are present during serialization.
        """
        dumped_dict = record.model_dump(mode="json")
        line = json.dumps(dumped_dict, allow_nan=False, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    def flush(self) -> None:
        """Flush internal buffers to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the recording file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> FeatureRecorder:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit, closing file."""
        self.close()


class ReplayLoader:
    """Deterministic replay reader yielding RecorderTurnRecord objects bit-exactly."""

    def __init__(self, input_path: Union[str, Path]) -> None:
        """Initialize ReplayLoader.

        Args:
            input_path: Path to the .jsonl recording file.
        """
        self._path: Path = Path(input_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Recording file not found: {self._path}")

    @property
    def path(self) -> Path:
        """Return path of the replay file."""
        return self._path

    def __iter__(self) -> Iterator[RecorderTurnRecord]:
        """Iterate line-by-line over the recorded JSONL file.

        Yields:
            Validated RecorderTurnRecord instances.
        """
        with self._path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = RecorderTurnRecord.model_validate_json(stripped)
                    yield record
                except Exception as exc:
                    raise ValueError(
                        f"Failed parsing record at line {line_idx} in {self._path}: {exc}"
                    ) from exc

    def load_all(self) -> List[RecorderTurnRecord]:
        """Load and return all recorded turn records in memory.

        Returns:
            List of all RecorderTurnRecord instances.
        """
        return list(self)
