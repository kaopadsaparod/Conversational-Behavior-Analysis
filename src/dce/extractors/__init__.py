"""Discourse Consistency Engine - Multimodal Extractors.

Provides acoustic prosody (F0, jitter, shimmer, HNR), ASR transcription hooks,
and visual kinematic feature extractors (lip aperture, blink rate).
"""

from __future__ import annotations

from dce.extractors.audio_dsp import extract_prosody, transcribe_audio
from dce.extractors.vision_face import extract_vision_frame

__all__ = [
    "extract_prosody",
    "transcribe_audio",
    "extract_vision_frame",
]
