"""Deterministic Audio Degradation Stress Test for DCE.

Tests 10 real clean speech clips (RAVDESS Actors 01-10) under:
  (a) 50 Hz AC hum + broadband fan noise at SNR = 20 dB, 10 dB, 5 dB
  (b) Synthetic room reverberation (exponential RIR) at RT60 = 0.3s, 0.8s
  (c) Simulated Zoom/VoIP noise suppression (130-3800 Hz bandpass + digital noise-gate)

Saves full per-clip and aggregate metrics to benchmarks/results/audio_robustness.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import signal

from dce.baseline_store import BaselineStore
from dce.extractors.audio_dsp import (
    apply_zero_phase_highpass,
    decode_media_audio_and_frames,
    detect_voip_or_denoised_audio,
    extract_prosody,
)
from dce.schemas import (
    BaselineConfig,
    ProsodyFeatures,
    TimingFeatures,
    TurnFeatures,
    VisionFeatures,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "audio_robustness.json"


def _select_clean_clips(n: int = 10) -> List[Path]:
    clips: List[Path] = []
    for actor_idx in range(1, n + 1):
        actor_dir = REPO_ROOT / "dataset" / "Audio_Song_Actors_01-24" / f"Actor_{actor_idx:02d}"
        wav_path = actor_dir / f"03-02-01-01-01-01-{actor_idx:02d}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing RAVDESS clip: {wav_path}")
        clips.append(wav_path)
    return clips


def _active_speech_rms(arr: np.ndarray, sr: int = 16000) -> float:
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    rms_vals = [
        float(np.sqrt(np.mean(arr[s : s + frame_len] ** 2)))
        for s in range(0, max(1, len(arr) - frame_len + 1), hop_len)
    ]
    if not rms_vals:
        return float(np.sqrt(np.mean(arr ** 2)))
    p90 = float(np.percentile(rms_vals, 90))
    active = [r for r in rms_vals if r >= 0.2 * p90]
    return float(np.mean(active)) if active else float(np.sqrt(np.mean(arr ** 2)))


def add_hum_and_fan_noise(
    clean: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    t = np.arange(len(clean), dtype=np.float64) / float(sr)
    # 50 Hz AC hum + 100 Hz / 150 Hz harmonics + low-pass filtered fan noise (< 1200 Hz)
    hum = (
        0.65 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 100.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 150.0 * t)
    )
    white = rng.standard_normal(len(clean))
    sos_fan = signal.butter(4, 1200.0 / (0.5 * sr), btype="lowpass", output="sos")
    fan = signal.sosfiltfilt(sos_fan, white)
    fan = fan / max(float(np.sqrt(np.mean(fan ** 2))), 1e-12)
    hum = hum / max(float(np.sqrt(np.mean(hum ** 2))), 1e-12)

    noise_raw = 0.6 * hum + 0.4 * fan
    noise_rms = float(np.sqrt(np.mean(noise_raw ** 2)))
    speech_rms = _active_speech_rms(clean, sr)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise_raw * (target_noise_rms / max(noise_rms, 1e-12))
    out = clean + scaled_noise
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def add_reverb(
    clean: np.ndarray, sr: int, rt60_sec: float, rng: np.random.Generator
) -> np.ndarray:
    rir_len = int(min(rt60_sec * 1.2, 1.5) * sr)
    t = np.arange(rir_len, dtype=np.float64) / float(sr)
    decay = np.exp(-6.907755 * t / max(rt60_sec, 0.05))
    rir = rng.standard_normal(rir_len) * decay
    rir[0] = 1.0  # direct path
    rir = rir / max(float(np.sqrt(np.sum(rir ** 2))), 1e-12)
    rev = signal.fftconvolve(clean, rir, mode="full")[: len(clean)]
    # Normalize RMS to match original speech RMS
    orig_rms = float(np.sqrt(np.mean(clean ** 2)))
    rev_rms = float(np.sqrt(np.mean(rev ** 2)))
    if rev_rms > 1e-12:
        rev = rev * (orig_rms / rev_rms)
    return np.clip(rev, -0.99, 0.99)


def simulate_voip_denoised(clean: np.ndarray, sr: int) -> np.ndarray:
    # 130-3800 Hz bandpass + digital noise-gate zeroing quiet frames to spoof high SNR
    sos = signal.butter(6, [130.0 / (0.5 * sr), 3800.0 / (0.5 * sr)], btype="bandpass", output="sos")
    bp = signal.sosfiltfilt(sos, clean)
    frame_len = int(0.020 * sr)
    hop_len = int(0.010 * sr)
    out = bp.copy()
    speech_rms = _active_speech_rms(bp, sr)
    gate_thresh = 0.08 * speech_rms
    for s in range(0, len(out) - frame_len + 1, hop_len):
        seg = out[s : s + frame_len]
        if float(np.sqrt(np.mean(seg ** 2))) < gate_thresh:
            out[s : s + frame_len] = 0.0
    return out


def _check_snr_gated_in_baseline(pros: ProsodyFeatures, clean_pros: ProsodyFeatures) -> Dict[str, Any]:
    """Verify BaselineStore excludes/downweights jitter/shimmer when SNR < 12 dB or quality != ok.

    Warms up spk_test with 3 clean baseline turns first so cold-start (z_score=None)
    does not mask the real SNR / fine_wave_quality / VoIP gating signal.
    """
    cfg = BaselineConfig(noise_robust_voice_weighting=True, jitter_shimmer_noise_floor_snr_db=12.0)
    priors_path = REPO_ROOT / "data" / "priors_en.json"
    store = BaselineStore(config=cfg, priors=priors_path if priors_path.is_file() else None)
    for w_idx in range(1, 4):
        warmup_tf = TurnFeatures(
            turn_id=w_idx,
            speaker_id="spk_test",
            start_ns=(w_idx - 1) * 3_000_000_000,
            end_ns=w_idx * 3_000_000_000,
            prosody=clean_pros,
            timing=TimingFeatures(response_latency_open=300.0, response_latency_closed=300.0, speech_duration_ms=2500.0),
            vision=VisionFeatures(face_detected=False, quality_flag="no_video"),
        )
        store.score_and_update_turn("spk_test", warmup_tf)

    tf = TurnFeatures(
        turn_id=4,
        speaker_id="spk_test",
        start_ns=9_000_000_000,
        end_ns=12_000_000_000,
        prosody=pros,
        timing=TimingFeatures(response_latency_open=300.0, response_latency_closed=300.0, speech_duration_ms=2500.0),
        vision=VisionFeatures(face_detected=False, quality_flag="no_video"),
    )
    scores = store.score_and_update_turn("spk_test", tf)
    j = scores.get("jitter_local_pct")
    s = scores.get("shimmer_local_pct")
    j_gated = j is None or j.z_score is None or j.confidence_weight <= 0.0
    s_gated = s is None or s.z_score is None or s.confidence_weight <= 0.0
    return {
        "gated": bool(j_gated and s_gated),
        "jitter_weight": round(float(j.confidence_weight), 4) if j is not None else 0.0,
    }


def _summarize_condition(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(key: str) -> Any:
        vals = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "n_clips": len(rows),
        "quality_ok_count": sum(1 for r in rows if r["quality_flag"] == "ok"),
        "f0_sd_valid_count": sum(1 for r in rows if r["f0_semitone_sd"] is not None),
        "art_rate_valid_count": sum(1 for r in rows if r["articulation_rate"] is not None),
        "mean_f0_median_hz": _mean("f0_median_hz"),
        "mean_f0_semitone_sd": _mean("f0_semitone_sd"),
        "mean_articulation_rate": _mean("articulation_rate"),
        "mean_acoustic_snr_db": _mean("acoustic_snr_db"),
        "mean_jitter_pct": _mean("jitter_local_pct"),
        "mean_shimmer_pct": _mean("shimmer_local_pct"),
        "mean_hnr_db": _mean("hnr_db"),
        "reverb_flagged_count": sum(
            1 for r in rows if r["fine_wave_quality_flag"] == "reverberant_or_corrupted"
        ),
        "voip_detected_count": sum(1 for r in rows if r["voip_or_denoised_flag"]),
        "jitter_shimmer_disabled_count": sum(
            1 for r in rows if r["jitter_local_pct"] is None and r["shimmer_local_pct"] is None
        ),
        "baseline_noise_gated_count": sum(1 for r in rows if r["baseline_fine_wave_gated"]),
    }


def main() -> None:
    t0 = time.perf_counter()
    clips = _select_clean_clips(10)
    rng = np.random.default_rng(42)

    conditions = [
        "clean",
        "hum_fan_20db",
        "hum_fan_10db",
        "hum_fan_5db",
        "reverb_0.3s",
        "reverb_0.8s",
        "voip_denoised",
    ]
    per_condition: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}

    for idx, clip_path in enumerate(clips, start=1):
        audio, sr, _ = decode_media_audio_and_frames(clip_path)
        variants = {
            "clean": audio,
            "hum_fan_20db": add_hum_and_fan_noise(audio, sr, 20.0, rng),
            "hum_fan_10db": add_hum_and_fan_noise(audio, sr, 10.0, rng),
            "hum_fan_5db": add_hum_and_fan_noise(audio, sr, 5.0, rng),
            "reverb_0.3s": add_reverb(audio, sr, 0.3, rng),
            "reverb_0.8s": add_reverb(audio, sr, 0.8, rng),
            "voip_denoised": simulate_voip_denoised(audio, sr),
        }
        clean_pros = extract_prosody(audio, sample_rate=sr)
        for cond, sig in variants.items():
            # Verify highpass stability explicitly
            hp = apply_zero_phase_highpass(sig, sample_rate=sr, cutoff_hz=65.0)
            hp_finite = bool(np.all(np.isfinite(hp)))
            voip_flag, voip_meta = detect_voip_or_denoised_audio(sig, sample_rate=sr)
            pros = clean_pros if cond == "clean" else extract_prosody(sig, sample_rate=sr)
            gated_info = _check_snr_gated_in_baseline(pros, clean_pros)
            per_condition[cond].append(
                {
                    "clip_id": f"ravdess_actor_{idx:02d}",
                    "file": clip_path.name,
                    "hp_finite": hp_finite,
                    "voip_detect_raw": voip_flag,
                    "voip_gated_ratio": voip_meta["gated_ratio"],
                    "voip_highband_rolloff_db": voip_meta["highband_rolloff_db"],
                    "quality_flag": pros.quality_flag,
                    "fine_wave_quality_flag": pros.fine_wave_quality_flag,
                    "voip_or_denoised_flag": pros.voip_or_denoised_flag,
                    "f0_median_hz": pros.f0_median_hz,
                    "f0_semitone_sd": pros.f0_semitone_sd,
                    "articulation_rate": pros.articulation_rate,
                    "jitter_local_pct": pros.jitter_local_pct,
                    "shimmer_local_pct": pros.shimmer_local_pct,
                    "hnr_db": pros.hnr_db,
                    "acoustic_snr_db": pros.acoustic_snr_db,
                    "baseline_fine_wave_gated": gated_info["gated"],
                    "baseline_jitter_weight": gated_info["jitter_weight"],
                }
            )

    summary = {cond: _summarize_condition(rows) for cond, rows in per_condition.items()}
    elapsed = round(time.perf_counter() - t0, 3)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_sec": elapsed,
        "summary": summary,
        "per_clip": per_condition,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Audio robustness completed in {elapsed}s -> {RESULTS_PATH}")
    for cond, s in summary.items():
        print(
            f"  {cond:15s} | ok={s['quality_ok_count']}/10 | snr={s['mean_acoustic_snr_db']} | "
            f"f0_sd={s['mean_f0_semitone_sd']} | art={s['mean_articulation_rate']} | "
            f"reverb={s['reverb_flagged_count']}/10 | voip={s['voip_detected_count']}/10 | "
            f"jw_off={s['jitter_shimmer_disabled_count']}/10 | base_gated={s['baseline_noise_gated_count']}/10"
        )


if __name__ == "__main__":
    main()
