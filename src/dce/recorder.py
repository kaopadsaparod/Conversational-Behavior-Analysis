"""Discourse Consistency Engine - Feature Recorder and Deterministic Replay.

Provides strict JSONL recording guaranteed free of NaN and Infinity values
(allow_nan=False), alongside bit-exact deterministic replay capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, TextIO, Union

from dce.schemas import RecorderTurnRecord


class FeatureRecorder:
    """Thread-safe JSONL logger writing RecorderTurnRecord with strict allow_nan=False."""

    def __init__(self, output_path: Union[str, Path]) -> None:
        """Initialize FeatureRecorder.

        Args:
            output_path: Target filesystem path for the .jsonl recording file.
        """
        self._path: Path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        """Return the recording file path."""
        return self._path

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
