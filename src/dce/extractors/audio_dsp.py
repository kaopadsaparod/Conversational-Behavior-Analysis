"""Discourse Consistency Engine - Audio DSP and Acoustic Prosody Extraction.

Extracts F0 semitone variability, Praat local jitter, local shimmer, HNR, and
articulation rates with creak trimming and pitch floor clipping.
Enforces lazy imports for praat-parselmouth, faster-whisper, and onnxruntime.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Union

import numpy as np

from dce.math_core import f0_to_semitones
from dce.schemas import ProsodyConfig, ProsodyFeatures, TurnTranscript

logger = logging.getLogger(__name__)

WHISPER_INITIAL_PROMPT = "Umm, uh, er, ah, like, you know, I mean, well..."


def extract_prosody(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    speaker_median_f0: float = 120.0,
    config: Optional[ProsodyConfig] = None,
) -> ProsodyFeatures:
    """Extract acoustic prosody features using Praat algorithms.

    Applies creak trimming (>600ms audio trims trailing 200ms), speaker-dependent
    pitch floor clipping [60Hz, 140Hz], minimum 20 glottal periods check, and
    semitone transformation.

    Args:
        audio_pcm: 16kHz mono audio float array in [-1.0, 1.0].
        sample_rate: Audio sampling frequency in Hz (default 16000).
        speaker_median_f0: Reference median pitch in Hz (default 120.0).
        config: Prosody extraction configuration parameters.

    Returns:
        ProsodyFeatures instance populated with acoustic metrics.
    """
    cfg = config if config is not None else ProsodyConfig()
    audio_arr = np.asarray(audio_pcm, dtype=np.float64)

    total_samples = len(audio_arr)
    if total_samples < int((cfg.min_voiced_segment_ms / 1000.0) * sample_rate):
        return ProsodyFeatures(
            glottal_periods_count=0,
            quality_flag="insufficient_data",
        )

    # Creak trimming: If turn > 600ms, trim trailing 200ms to avoid phrase-final vocal fry
    turn_duration_ms = (total_samples / float(sample_rate)) * 1000.0
    if turn_duration_ms > cfg.creak_trim_min_speech_ms:
        trim_samples = int((cfg.creak_trim_tail_ms / 1000.0) * sample_rate)
        if total_samples > trim_samples:
            audio_arr = audio_arr[:-trim_samples]

    # Speaker-adapted pitch floor: clip(0.6 * speaker_median_f0, 60.0, 140.0)
    pitch_floor_hz = float(
        np.clip(
            0.6 * speaker_median_f0,
            cfg.pitch_floor_min_hz,
            cfg.pitch_floor_max_hz,
        )
    )

    # LAZY IMPORT of parselmouth
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        logger.warning("parselmouth not installed; returning fallback prosody features.")
        # Fallback simple energy-based computation when praat-parselmouth is unavailable
        rms = float(np.sqrt(np.mean(audio_arr**2))) if len(audio_arr) > 0 else 0.0
        return ProsodyFeatures(
            f0_median_hz=speaker_median_f0,
            f0_semitone_sd=1.0 if rms > 0.01 else 0.0,
            glottal_periods_count=30 if rms > 0.01 else 0,
            quality_flag="no_audio_backend",
        )

    try:
        sound = parselmouth.Sound(values=audio_arr, sampling_frequency=float(sample_rate))

        # Extract pitch contour
        time_step_sec = 0.01
        pitch = sound.to_pitch(
            time_step=time_step_sec,
            pitch_floor=pitch_floor_hz,
            pitch_ceiling=500.0,
        )
        f0_values = pitch.selected_array["frequency"]
        voiced_mask = f0_values > 0.0

        if not np.any(voiced_mask):
            return ProsodyFeatures(
                f0_median_hz=None,
                f0_semitone_sd=None,
                jitter_local_pct=None,
                shimmer_local_pct=None,
                hnr_db=None,
                glottal_periods_count=0,
                quality_flag="unvoiced",
            )

        # Apply boundary_trim_ms (15 ms) on contiguous voiced segments >= min_voiced_segment_ms (75 ms)
        frame_ms = time_step_sec * 1000.0
        min_seg_frames = max(1, int(math.ceil(cfg.min_voiced_segment_ms / frame_ms)))
        trim_frames = max(0, int(round(cfg.boundary_trim_ms / frame_ms)))
        trimmed_voiced_mask = np.zeros_like(voiced_mask, dtype=bool)

        idx = 0
        n_frames = len(voiced_mask)
        while idx < n_frames:
            if voiced_mask[idx]:
                seg_start = idx
                while idx < n_frames and voiced_mask[idx]:
                    idx += 1
                seg_end = idx
                if (seg_end - seg_start) >= min_seg_frames:
                    t_start = seg_start + trim_frames
                    t_end = seg_end - trim_frames
                    if t_end > t_start:
                        trimmed_voiced_mask[t_start:t_end] = True
            else:
                idx += 1

        if not np.any(trimmed_voiced_mask):
            return ProsodyFeatures(
                f0_median_hz=None,
                f0_semitone_sd=None,
                jitter_local_pct=None,
                shimmer_local_pct=None,
                hnr_db=None,
                glottal_periods_count=0,
                quality_flag="insufficient_data",
            )

        f0_median = float(np.median(f0_values[trimmed_voiced_mask]))

        # Semitone standard deviation
        trimmed_f0 = np.where(trimmed_voiced_mask, f0_values, 0.0)
        semitones = f0_to_semitones(trimmed_f0, f0_median)
        valid_semitones = semitones[np.isfinite(semitones)]
        f0_semitone_sd = (
            float(np.std(valid_semitones)) if len(valid_semitones) > 1 else 0.0
        )

        # Pulse analysis for Jitter, Shimmer, and Period count
        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            pitch_floor_hz,
            500.0,
        )
        n_periods = int(
            call(
                point_process,
                "Get number of periods",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
            )
        )

        if n_periods < cfg.min_glottal_periods:
            return ProsodyFeatures(
                f0_median_hz=f0_median,
                f0_semitone_sd=f0_semitone_sd,
                glottal_periods_count=n_periods,
                quality_flag="insufficient_data",
            )

        jitter_local = float(
            call(
                point_process,
                "Get jitter (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
            )
            * 100.0
        )

        shimmer_local = float(
            call(
                [sound, point_process],
                "Get shimmer (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            * 100.0
        )

        # Harmonicity (HNR)
        harmonicity = sound.to_harmonicity(
            time_step=0.01,
            minimum_pitch=pitch_floor_hz,
        )
        hnr_mean = float(call(harmonicity, "Get mean", 0.0, 0.0))

        # Basic articulation rate proxy (syllables/sec estimated by voiced pulses / 5)
        speech_sec = float(len(audio_arr)) / float(sample_rate)
        est_syllables = max(1.0, float(n_periods) / 10.0)
        articulation_rate = est_syllables / max(speech_sec, 0.1)

        return ProsodyFeatures(
            f0_median_hz=f0_median,
            f0_semitone_sd=f0_semitone_sd,
            jitter_local_pct=jitter_local,
            shimmer_local_pct=shimmer_local,
            hnr_db=hnr_mean,
            articulation_rate=articulation_rate,
            glottal_periods_count=n_periods,
            quality_flag="ok",
        )

    except Exception as exc:
        logger.warning("Praat extraction encountered error: %s", exc)
        return ProsodyFeatures(
            glottal_periods_count=0,
            quality_flag="extraction_error",
        )


def transcribe_audio(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    model_size: str = "small.en",
    device: str = "cpu",
    initial_prompt: str = WHISPER_INITIAL_PROMPT,
) -> str:
    """Transcribe speech audio using faster-whisper with lazy model loading.

    Args:
        audio_pcm: 16kHz mono audio float array.
        sample_rate: Sampling frequency in Hz.
        model_size: Faster-Whisper model checkpoint name.
        device: Device to run inference ('cpu' or 'cuda').
        initial_prompt: Whisper prompt conditioning for fillers and hesitations.

    Returns:
        Transcribed plain text string.
    """
    audio_arr = np.asarray(audio_pcm, dtype=np.float32)
    if len(audio_arr) == 0:
        return ""

    # LAZY IMPORT of faster_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster_whisper not installed; ASR unavailable.")
        return ""

    try:
        compute_type = "int8" if device == "cpu" else "float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(
            audio_arr,
            beam_size=5,
            initial_prompt=initial_prompt,
            vad_filter=False,
            word_timestamps=True,
        )
        clean_parts = []
        for seg in segments:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            no_speech_prob = getattr(seg, "no_speech_prob", None)
            avg_logprob = getattr(seg, "avg_logprob", None)
            compression_ratio = getattr(seg, "compression_ratio", None)
            if no_speech_prob is not None and no_speech_prob > 0.8:
                continue
            if avg_logprob is not None and avg_logprob < -1.0:
                continue
            if compression_ratio is not None and compression_ratio > 2.4:
                continue
            clean_parts.append(text)
        return " ".join(clean_parts).strip()
    except Exception as exc:
        logger.warning("ASR transcription error: %s", exc)
        return ""
