"""Discourse Consistency Engine - Audio DSP and Acoustic Prosody Extraction.

Extracts F0 semitone variability, Praat local jitter, local shimmer, HNR, and
articulation rates with creak trimming and pitch floor clipping.
Enforces lazy imports for praat-parselmouth, faster-whisper, and onnxruntime.
"""

from __future__ import annotations

import contextlib
import logging
import math
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dce.math_core import f0_to_semitones
from dce.schemas import ASRResult, ProsodyConfig, ProsodyFeatures, TurnTranscript, WordTimestamp

logger = logging.getLogger(__name__)

WHISPER_INITIAL_PROMPT = "Umm, uh, er, ah, like, you know, I mean, well..."

_WHISPER_MODEL_LOCK = threading.Lock()
_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str, Any], Any] = {}


def clear_whisper_model_cache() -> None:
    """Clear the module-level faster-whisper model cache."""
    with _WHISPER_MODEL_LOCK:
        _WHISPER_MODEL_CACHE.clear()


def _get_whisper_model(model_cls: Any, model_size: str, device: str, compute_type: str) -> Any:
    """Return a cached WhisperModel instance for the given configuration."""
    cache_key = (model_size, device, compute_type, model_cls)
    with _WHISPER_MODEL_LOCK:
        model = _WHISPER_MODEL_CACHE.get(cache_key)
        if model is None:
            model = model_cls(model_size, device=device, compute_type=compute_type)
            _WHISPER_MODEL_CACHE[cache_key] = model
        return model


def extract_egemaps_spectral_features(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    use_opensmile: bool = False,
) -> Dict[str, Optional[float]]:
    """Extract noise-robust eGeMAPS-style spectral & energy features.

    Computes Alpha Ratio (50-1000 Hz vs 1000-5000 Hz), Spectral Slope (50-4000 Hz),
    Loudness (dBFS), and frame-energy SNR (dB). Optionally uses openSMILE eGeMAPSv02
    via lazy import when ``use_opensmile=True`` and installed.
    """
    arr = np.asarray(audio_pcm, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if arr.size == 0 or sample_rate <= 0 or not np.any(np.isfinite(arr)):
        return {
            "spectral_slope_db": None,
            "alpha_ratio_db": None,
            "loudness_db": None,
            "acoustic_snr_db": None,
        }
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional lazy import of opensmile
    if use_opensmile:
        try:
            import opensmile  # type: ignore

            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            df = smile.process_signal(arr.astype(np.float32), sample_rate)
            if not df.empty:
                row = df.iloc[0]
                if "slopeV0-500_sma3nz_amean" in row.index and "alphaRatioV_sma3nz_amean" in row.index:
                    slope_val = float(row["slopeV0-500_sma3nz_amean"])
                    alpha_val = float(row["alphaRatioV_sma3nz_amean"])
                    if math.isfinite(slope_val) and math.isfinite(alpha_val):
                        builtin = extract_egemaps_spectral_features(arr, sample_rate, use_opensmile=False)
                        return {
                            "spectral_slope_db": slope_val,
                            "alpha_ratio_db": alpha_val,
                            "loudness_db": builtin["loudness_db"],
                            "acoustic_snr_db": builtin["acoustic_snr_db"],
                        }
        except ImportError:
            logger.debug("opensmile not installed; using built-in eGeMAPS DSP.")
        except Exception as exc:
            logger.debug("opensmile extraction failed (%s); falling back to built-in DSP.", exc)

    rms = float(np.sqrt(np.mean(arr ** 2)))
    loudness_db = float(20.0 * math.log10(max(rms, 1e-12)))

    # Power spectrum via Hann-windowed rFFT
    window = np.hanning(len(arr))
    spec = np.fft.rfft(arr * window)
    power = np.abs(spec) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / float(sample_rate))

    low_mask = (freqs >= 50.0) & (freqs <= 1000.0)
    high_mask = (freqs > 1000.0) & (freqs <= 5000.0)
    e_low = float(np.sum(power[low_mask])) if np.any(low_mask) else 1e-18
    e_high = float(np.sum(power[high_mask])) if np.any(high_mask) else 1e-18
    alpha_ratio_db = float(10.0 * math.log10(max(e_low, 1e-18) / max(e_high, 1e-18)))

    # Frame-level SNR estimate (25 ms frames, 10 ms hop), with spectral fallback for steady-state signals
    frame_len = max(1, int(0.025 * sample_rate))
    hop_len = max(1, int(0.010 * sample_rate))
    snr_db: Optional[float] = None
    if len(arr) >= frame_len:
        frame_rms = []
        for start in range(0, len(arr) - frame_len + 1, hop_len):
            seg = arr[start : start + frame_len]
            frame_rms.append(float(np.sqrt(np.mean(seg ** 2))))
        if len(frame_rms) >= 4:
            p90 = float(np.percentile(frame_rms, 90))
            p10 = float(np.percentile(frame_rms, 10))
            if p90 > 2.0 * max(p10, 1e-12):
                snr_db = float(np.clip(20.0 * math.log10(max(p90, 1e-12) / max(p10, 1e-12)), -10.0, 60.0))
    if snr_db is None:
        # No pause frames in turn (continuous speech or steady tone): use spectral alpha-ratio proxy
        snr_db = float(np.clip(alpha_ratio_db + 12.0, -10.0, 40.0))

    slope_mask = (freqs >= 50.0) & (freqs <= 4000.0)
    if np.sum(slope_mask) >= 2:
        f_khz = freqs[slope_mask] / 1000.0
        p_db = 10.0 * np.log10(power[slope_mask])
        slope, _ = np.polyfit(f_khz, p_db, 1)
        spectral_slope_db = float(slope) if math.isfinite(slope) else 0.0
    else:
        spectral_slope_db = 0.0

    return {
        "spectral_slope_db": spectral_slope_db,
        "alpha_ratio_db": alpha_ratio_db,
        "loudness_db": loudness_db,
        "acoustic_snr_db": snr_db,
    }


def apply_zero_phase_highpass(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    cutoff_hz: float = 65.0,
) -> np.ndarray:
    """Apply a zero-phase 4th-order Butterworth high-pass filter to strip 50/60 Hz AC/fan hum."""
    arr = np.asarray(audio_pcm, dtype=np.float64)
    if arr.size <= 18 or sample_rate <= 0 or cutoff_hz <= 0.0 or cutoff_hz >= 0.45 * float(sample_rate):
        return arr
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff_hz, btype="highpass", fs=float(sample_rate), output="sos")
    return np.asarray(sosfiltfilt(sos, arr), dtype=np.float64)


def detect_voip_or_denoised_audio(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    noise_gate_dbfs: float = -65.0,
    min_gated_ratio: float = 0.15,
    rolloff_db_threshold: float = -30.0,
) -> Tuple[bool, Dict[str, float]]:
    """Detect VoIP/AI-denoised audio via Digital Noise-Gate Floor + >4 kHz Bandwidth Roll-off."""
    arr = np.asarray(audio_pcm, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if arr.size == 0 or sample_rate <= 0:
        return False, {"gated_ratio": 0.0, "highband_rolloff_db": 0.0}

    frame_len = max(1, int(0.020 * sample_rate))
    hop_len = max(1, int(0.010 * sample_rate))
    gated_ratio = 0.0
    has_active_speech = False
    if len(arr) >= frame_len:
        gate_amp = 10.0 ** (noise_gate_dbfs / 20.0)
        frame_rms = [
            float(np.sqrt(np.mean(arr[s : s + frame_len] ** 2)))
            for s in range(0, len(arr) - frame_len + 1, hop_len)
        ]
        if frame_rms:
            has_active_speech = max(frame_rms) > 0.01
            gated_ratio = sum(1 for r in frame_rms if r < gate_amp) / float(len(frame_rms))

    window = np.hanning(len(arr))
    power = np.abs(np.fft.rfft(arr * window)) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / float(sample_rate))
    speech_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    high_mask = (freqs >= 4000.0) & (freqs <= min(7800.0, 0.49 * float(sample_rate)))

    highband_rolloff_db = 0.0
    if np.any(speech_mask) and np.any(high_mask):
        e_speech = float(np.sum(power[speech_mask]))
        e_high = float(np.sum(power[high_mask]))
        highband_rolloff_db = float(10.0 * math.log10(max(e_high, 1e-18) / max(e_speech, 1e-18)))

    is_voip_denoised = bool(
        has_active_speech
        and gated_ratio >= min_gated_ratio
        and np.any(high_mask)
        and highband_rolloff_db <= rolloff_db_threshold
    )
    return is_voip_denoised, {
        "gated_ratio": round(gated_ratio, 4),
        "highband_rolloff_db": round(highband_rolloff_db, 2),
    }


def compute_voiced_articulation_rate(
    audio_arr: np.ndarray,
    sample_rate: int,
    voiced_mask: np.ndarray,
    time_step_sec: float = 0.01,
    min_syllable_dist_ms: float = 90.0,
    min_prominence_db: float = 2.0,
    pause_threshold_ms: float = 200.0,
    min_speaking_sec: float = 0.8,
    min_syllable_peaks: int = 3,
) -> Optional[float]:
    """Compute speaker-invariant articulation rate (syllables/sec) via Voiced Intensity Envelope Peaks.

    Follows the De Jong & Wempe syllable-nuclei principle: smooths the short-term dB
    energy envelope, restricts peak detection to voiced frames (F0 > 0) with minimum
    inter-syllable spacing >= 90 ms and prominence >= 2.0 dB, and divides by total
    speaking duration excluding silent pauses >= 200 ms. Returns None when speaking
    duration < 0.8s or fewer than 3 syllable peaks are detected.
    """
    from scipy.signal import butter, find_peaks, sosfiltfilt

    if len(audio_arr) == 0 or sample_rate <= 0 or len(voiced_mask) == 0:
        return None

    hop = max(1, int(round(time_step_sec * sample_rate)))
    win = max(hop, int(round(0.025 * sample_rate)))
    n_frames = len(voiced_mask)
    env_db = np.full(n_frames, -120.0, dtype=np.float64)
    for i in range(n_frames):
        center = i * hop
        s = max(0, center - win // 2)
        e = min(len(audio_arr), center + win // 2)
        if e > s:
            rms_sq = float(np.mean(audio_arr[s:e] ** 2))
            env_db[i] = 10.0 * math.log10(max(rms_sq, 1e-12))

    # Smooth envelope with ~10 Hz zero-phase low-pass (or fallback moving average on tiny turns)
    fs_env = 1.0 / max(time_step_sec, 1e-4)
    if len(env_db) > 12 and fs_env > 25.0:
        sos = butter(2, 10.0, btype="low", fs=fs_env, output="sos")
        env_smooth = sosfiltfilt(sos, env_db)
    else:
        env_smooth = env_db

    # Mask unvoiced frames to floor so peaks only occur inside voiced speech
    floor_db = float(np.min(env_smooth)) - 5.0
    voiced_env = np.where(voiced_mask, env_smooth, floor_db)
    min_dist_frames = max(1, int(round((min_syllable_dist_ms / 1000.0) / time_step_sec)))
    peaks, _ = find_peaks(voiced_env, distance=min_dist_frames, prominence=min_prominence_db)

    # Speaking duration = total frames minus unvoiced runs >= pause_threshold_ms (200 ms)
    pause_frames = max(1, int(round((pause_threshold_ms / 1000.0) / time_step_sec)))
    speaking_frames = 0
    idx = 0
    while idx < n_frames:
        if voiced_mask[idx]:
            speaking_frames += 1
            idx += 1
        else:
            run_start = idx
            while idx < n_frames and not voiced_mask[idx]:
                idx += 1
            run_len = idx - run_start
            if run_len < pause_frames:
                speaking_frames += run_len

    speaking_sec = float(speaking_frames) * time_step_sec
    if speaking_sec < min_speaking_sec or len(peaks) < min_syllable_peaks:
        return None
    return float(len(peaks)) / speaking_sec


def decode_media_audio_and_frames(
    media_path: Union[str, Path],
    target_sr: int = 16000,
    video_fps: float = 15.0,
    max_frames: int = 9000,
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """Centralized media decoder returning (mono_float32_audio, 16000, sampled_rgb_frames).

    Uses Python wave for .wav, PyAV (av >= 11, compatible with av 19.0.2) directly for
    non-WAV containers, and falls back to faster_whisper.decode_audio / ffmpeg if needed.
    Samples video frames at ``video_fps`` (default 15 FPS) for blink/lip tracking.
    """
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")

    suffix = path.suffix.lower()
    audio: Optional[np.ndarray] = None

    if suffix == ".wav":
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        if src_sr != target_sr and arr.size > 0:
            n_out = max(1, int(round(arr.size * float(target_sr) / float(src_sr))))
            arr = np.interp(
                np.linspace(0.0, 1.0, n_out, endpoint=False),
                np.linspace(0.0, 1.0, arr.size, endpoint=False),
                arr,
            ).astype(np.float32)
        return arr.astype(np.float32), target_sr, []

    # Try PyAV directly first (av 19.0.2 compatible; no external ffmpeg.exe needed)
    try:
        import av  # type: ignore

        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        chunks: List[np.ndarray] = []
        with av.open(str(path), mode="r") as container:
            if container.streams.audio:
                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        chunks.append(resampled.to_ndarray().reshape(-1))
                for resampled in resampler.resample(None):
                    chunks.append(resampled.to_ndarray().reshape(-1))
        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
    except Exception as av_exc:
        logger.debug("PyAV direct decode failed (%s); trying decode_audio/ffmpeg.", av_exc)

    if audio is None:
        try:
            from faster_whisper.audio import decode_audio  # type: ignore

            audio = np.asarray(decode_audio(str(path), sampling_rate=target_sr), dtype=np.float32)
        except Exception as fw_exc:
            logger.debug("faster_whisper.decode_audio failed (%s); trying ffmpeg.", fw_exc)

    if audio is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-vn", "-ac", "1", "-ar", str(target_sr), "-f", "wav", str(tmp_wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0 or not tmp_wav.is_file() or tmp_wav.stat().st_size == 0:
                raise RuntimeError(
                    f"Failed to decode non-WAV media '{path}' via PyAV and ffmpeg "
                    f"(exit={proc.returncode}): {proc.stderr.decode('utf-8', errors='replace')[:300]}"
                )
            audio, _, _ = decode_media_audio_and_frames(tmp_wav, target_sr=target_sr)
        finally:
            with contextlib.suppress(OSError):
                tmp_wav.unlink()

    frames: List[np.ndarray] = []
    if suffix in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                step = max(1, int(round(src_fps / max(video_fps, 1.0))))
                idx = 0
                while len(frames) < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % step == 0:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    idx += 1
                cap.release()
        except Exception:
            pass

    return audio.astype(np.float32), target_sr, frames


def extract_prosody(
    audio_pcm: Union[Sequence[float], np.ndarray],
    sample_rate: int = 16000,
    speaker_median_f0: float = 120.0,
    config: Optional[ProsodyConfig] = None,
    use_opensmile: bool = False,
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
    raw_arr = np.asarray(audio_pcm, dtype=np.float64)
    if raw_arr.ndim > 1:
        raw_arr = np.mean(raw_arr, axis=1)

    total_samples = len(raw_arr)
    if total_samples < int((cfg.min_voiced_segment_ms / 1000.0) * sample_rate):
        return ProsodyFeatures(
            glottal_periods_count=0,
            quality_flag="insufficient_data",
        )

    # Detect VoIP/denoised audio on raw PCM before high-pass filtering
    voip_or_denoised, _ = detect_voip_or_denoised_audio(raw_arr, sample_rate=sample_rate)

    # Creak trimming: If turn > 600ms, trim trailing 200ms to avoid phrase-final vocal fry
    audio_arr = raw_arr
    turn_duration_ms = (total_samples / float(sample_rate)) * 1000.0
    if turn_duration_ms > cfg.creak_trim_min_speech_ms:
        trim_samples = int((cfg.creak_trim_tail_ms / 1000.0) * sample_rate)
        if total_samples > trim_samples:
            audio_arr = audio_arr[:-trim_samples]

    # Zero-phase Butterworth high-pass filter (65 Hz) to strip 50/60 Hz AC/fan hum
    audio_arr = apply_zero_phase_highpass(
        audio_arr,
        sample_rate=sample_rate,
        cutoff_hz=cfg.highpass_cutoff_hz,
    )

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
        logger.warning("parselmouth not installed; returning unavailable prosody features.")
        return ProsodyFeatures(
            f0_median_hz=None,
            f0_semitone_sd=None,
            jitter_local_pct=None,
            shimmer_local_pct=None,
            hnr_db=None,
            articulation_rate=None,
            glottal_periods_count=0,
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

        # Semitone standard deviation (require voiced duration >= 0.8s so 1-word replies don't poison baseline)
        voiced_sec = float(np.sum(voiced_mask)) * time_step_sec
        trimmed_f0 = np.where(trimmed_voiced_mask, f0_values, 0.0)
        semitones = f0_to_semitones(trimmed_f0, f0_median)
        valid_semitones = semitones[np.isfinite(semitones)]
        if voiced_sec >= 0.8 and len(valid_semitones) > 1:
            f0_semitone_sd: Optional[float] = float(np.std(valid_semitones))
        else:
            f0_semitone_sd = None

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

        # Speaker-invariant articulation rate (syllables/sec via Voiced Intensity Envelope Peaks)
        articulation_rate = compute_voiced_articulation_rate(
            audio_arr=audio_arr,
            sample_rate=sample_rate,
            voiced_mask=voiced_mask,
            time_step_sec=time_step_sec,
        )

        egemaps = extract_egemaps_spectral_features(
            audio_arr, sample_rate=sample_rate, use_opensmile=use_opensmile
        )

        # Physiological ceiling & octave-jump gate for reverberant rooms
        voiced_st = valid_semitones
        octave_jump_ratio = 0.0
        if len(voiced_st) >= 2:
            diffs = np.abs(np.diff(voiced_st))
            octave_jump_ratio = float(np.mean(diffs >= 10.0))

        reverb_corrupted = (
            jitter_local > cfg.max_physiological_jitter_pct
            or shimmer_local > cfg.max_physiological_shimmer_pct
            or octave_jump_ratio > 0.20
        )

        if reverb_corrupted:
            return ProsodyFeatures(
                f0_median_hz=f0_median,
                f0_semitone_sd=f0_semitone_sd,
                jitter_local_pct=None,
                shimmer_local_pct=None,
                hnr_db=None,
                articulation_rate=articulation_rate,
                spectral_slope_db=egemaps["spectral_slope_db"],
                alpha_ratio_db=egemaps["alpha_ratio_db"],
                loudness_db=egemaps["loudness_db"],
                acoustic_snr_db=egemaps["acoustic_snr_db"],
                glottal_periods_count=n_periods,
                voip_or_denoised_flag=voip_or_denoised,
                fine_wave_quality_flag="reverberant_or_corrupted",
                quality_flag="ok",
            )

        return ProsodyFeatures(
            f0_median_hz=f0_median,
            f0_semitone_sd=f0_semitone_sd,
            jitter_local_pct=None if voip_or_denoised else jitter_local,
            shimmer_local_pct=None if voip_or_denoised else shimmer_local,
            hnr_db=None if voip_or_denoised else hnr_mean,
            articulation_rate=articulation_rate,
            spectral_slope_db=egemaps["spectral_slope_db"],
            alpha_ratio_db=egemaps["alpha_ratio_db"],
            loudness_db=egemaps["loudness_db"],
            acoustic_snr_db=egemaps["acoustic_snr_db"],
            glottal_periods_count=n_periods,
            voip_or_denoised_flag=voip_or_denoised,
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
    hotwords: Optional[str] = None,
) -> ASRResult:
    """Transcribe speech audio using faster-whisper with cached model loading.

    Args:
        audio_pcm: 16kHz mono audio float array.
        sample_rate: Sampling frequency in Hz.
        model_size: Faster-Whisper model checkpoint name.
        device: Device to run inference ('cpu' or 'cuda').
        initial_prompt: Whisper prompt conditioning for fillers and hesitations.
        hotwords: Optional comma/space-separated hotwords for rolling vocabulary biasing.

    Returns:
        ASRResult containing transcribed text and word-level timestamps.
    """
    audio_arr = np.asarray(audio_pcm, dtype=np.float32)
    if len(audio_arr) == 0:
        return ASRResult(text="", words=[])

    # LAZY IMPORT of faster_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster_whisper not installed; ASR unavailable.")
        return ASRResult(text="", words=[])

    try:
        compute_type = "int8" if device == "cpu" else "float16"
        model = _get_whisper_model(WhisperModel, model_size, device, compute_type)
        transcribe_kwargs: Dict[str, Any] = {
            "beam_size": 5,
            "initial_prompt": initial_prompt,
            "vad_filter": False,
            "word_timestamps": True,
        }
        if hotwords is not None:
            transcribe_kwargs["hotwords"] = hotwords

        segments, info = model.transcribe(audio_arr, **transcribe_kwargs)
        clean_parts: List[str] = []
        collected_words: List[WordTimestamp] = []
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
            for w in getattr(seg, "words", None) or []:
                w_text = getattr(w, "word", None)
                w_start = getattr(w, "start", None)
                w_end = getattr(w, "end", None)
                if w_text is None or w_start is None or w_end is None:
                    continue
                collected_words.append(
                    WordTimestamp(
                        word=str(w_text),
                        start=float(w_start),
                        end=float(w_end),
                        probability=(
                            float(w.probability)
                            if getattr(w, "probability", None) is not None
                            else None
                        ),
                    )
                )

        language = getattr(info, "language", None) if info is not None else None
        return ASRResult(
            text=" ".join(clean_parts).strip(),
            words=collected_words,
            language=language,
        )
    except Exception as exc:
        logger.warning("ASR transcription error: %s", exc)
        return ASRResult(text="", words=[])
