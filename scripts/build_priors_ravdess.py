"""Discourse Consistency Engine - Script to Compute Voice Priors from RAVDESS Dataset.

Parses WAV audio files, extracts acoustic prosody features using Praat,
computes empirical median and sigma for vocal features, and saves a PriorsData JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from dce.extractors.audio_dsp import extract_prosody
from dce.math_core import compute_mahalanobis
from dce.schemas import PriorParameter, PriorsData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_priors")


def process_dataset(audio_dir: Path) -> PriorsData:
    """Analyze all WAV files in dataset directory and compute feature prior parameters."""
    f0_sds: List[float] = []
    jitters: List[float] = []
    shimmers: List[float] = []
    hnrs: List[float] = []

    wav_files = list(audio_dir.glob("**/*.wav"))
    logger.info("Found %d audio files in %s", len(wav_files), audio_dir)

    for wav_path in wav_files:
        try:
            # Read PCM samples (lazy import scipy.io.wavfile)
            from scipy.io import wavfile

            sr, data = wavfile.read(str(wav_path))
            if data.dtype == np.int16:
                pcm = data.astype(np.float64) / 32768.0
            else:
                pcm = data.astype(np.float64)

            if pcm.ndim > 1:
                pcm = np.mean(pcm, axis=1)

            features = extract_prosody(pcm, sample_rate=sr)
            if features.quality_flag == "ok":
                if features.f0_semitone_sd is not None:
                    f0_sds.append(features.f0_semitone_sd)
                if features.jitter_local_pct is not None:
                    jitters.append(features.jitter_local_pct)
                if features.shimmer_local_pct is not None:
                    shimmers.append(features.shimmer_local_pct)
                if features.hnr_db is not None:
                    hnrs.append(features.hnr_db)
        except Exception as exc:
            logger.debug("Skipping %s due to error: %s", wav_path.name, exc)

    logger.info("Extracted %d valid speech samples.", len(f0_sds))

    # Compute empirical median and robust sigma (1.4826 * MAD)
    def calc_stats(vals: List[float], def_med: float, def_sig: float) -> tuple[float, float]:
        if not vals:
            return def_med, def_sig
        arr = np.array(vals)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sig = max(1.4826 * mad, 0.05)
        return med, sig

    med_f0, sig_f0 = calc_stats(f0_sds, 2.60, 0.85)
    med_jit, sig_jit = calc_stats(jitters, 1.10, 0.45)
    med_shim, sig_shim = calc_stats(shimmers, 4.80, 1.60)
    med_hnr, sig_hnr = calc_stats(hnrs, 15.50, 3.20)

    features_dict = {
        "response_latency_open": PriorParameter(
            feature_name="response_latency_open",
            pop_median=-0.20,
            pop_sigma=0.55,
            sigma_floor=0.15,
            max_anchor_drift=0.85,
            log_transform=True,
            log_offset=0.05,
            stress_polarity=1,
        ),
        "response_latency_closed": PriorParameter(
            feature_name="response_latency_closed",
            pop_median=-0.80,
            pop_sigma=0.45,
            sigma_floor=0.15,
            max_anchor_drift=0.70,
            log_transform=True,
            log_offset=0.05,
            stress_polarity=1,
        ),
        "articulation_rate": PriorParameter(
            feature_name="articulation_rate",
            pop_median=4.40,
            pop_sigma=0.75,
            sigma_floor=0.20,
            max_anchor_drift=1.15,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=0,
        ),
        "f0_semitone_sd": PriorParameter(
            feature_name="f0_semitone_sd",
            pop_median=med_f0,
            pop_sigma=sig_f0,
            sigma_floor=0.25,
            max_anchor_drift=1.25,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=0,
        ),
        "jitter_local_pct": PriorParameter(
            feature_name="jitter_local_pct",
            pop_median=med_jit,
            pop_sigma=sig_jit,
            sigma_floor=0.12,
            max_anchor_drift=0.65,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
        "shimmer_local_pct": PriorParameter(
            feature_name="shimmer_local_pct",
            pop_median=med_shim,
            pop_sigma=sig_shim,
            sigma_floor=0.40,
            max_anchor_drift=2.40,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=1,
        ),
        "hnr_db": PriorParameter(
            feature_name="hnr_db",
            pop_median=med_hnr,
            pop_sigma=sig_hnr,
            sigma_floor=0.80,
            max_anchor_drift=4.80,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=-1,
        ),
        "blink_rate_per_min": PriorParameter(
            feature_name="blink_rate_per_min",
            pop_median=18.00,
            pop_sigma=7.50,
            sigma_floor=2.00,
            max_anchor_drift=11.25,
            log_transform=False,
            log_offset=0.0,
            stress_polarity=0,
        ),
    }

    feature_order = list(features_dict.keys())
    corr_matrix = np.eye(len(feature_order), dtype=np.float64).tolist()

    return PriorsData(
        calibrated=len(f0_sds) > 50,
        source="ravdess_empirical_calibration" if len(f0_sds) > 50 else "literature_initial_estimates",
        features=features_dict,
        correlation_matrix=corr_matrix,
        feature_order=feature_order,
    )


def load_besst_labels(labels_path: Path) -> Dict[str, str]:
    """Load BESST cognitive load labels (low/medium/high) from CSV or JSON."""
    if not labels_path.is_file():
        return {}
    if labels_path.suffix.lower() == ".json":
        with labels_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): str(v).strip().lower() for k, v in raw.items()}

    mapping: Dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("file") or row.get("id") or ""
            lbl = row.get("cognitive_load") or row.get("label") or ""
            if fname and lbl:
                mapping[str(fname).strip()] = str(lbl).strip().lower()
    return mapping


def _infer_besst_load_label(wav_path: Path, labels_map: Dict[str, str]) -> Optional[str]:
    if wav_path.name in labels_map:
        return labels_map[wav_path.name]
    if wav_path.stem in labels_map:
        return labels_map[wav_path.stem]
    lower_parts = (wav_path.stem.lower(), wav_path.parent.name.lower())
    for level in ("low", "medium", "high"):
        if any(level in p for p in lower_parts):
            return level
    return None


def calibrate_thresholds_from_d_sq_samples(
    low_d_sq: Sequence[float],
    high_d_sq: Sequence[float],
    df: int = 4,
) -> Dict[str, Any]:
    """Calibrate Mahalanobis thresholds from BESST low vs high cognitive load d_sq distributions."""
    if not low_d_sq or not high_d_sq:
        return {
            "calibration_source": "ravdess_only",
            "besst_low_load_d_sq_p95": None,
            "besst_high_load_d_sq_p50": None,
            "cognitive_load_d_sq_threshold": None,
            "tail_score_threshold": 0.05,
        }

    low_arr = np.asarray(low_d_sq, dtype=np.float64)
    high_arr = np.asarray(high_d_sq, dtype=np.float64)
    low_p95 = float(np.percentile(low_arr, 95))
    high_p50 = float(np.percentile(high_arr, 50))
    opt_d_sq = 0.5 * (low_p95 + high_p50)

    # Sweep candidate chi2 tail thresholds to maximize F1 separating high vs low load
    best_tail = 0.05
    best_f1 = -1.0
    for cand_tail in (0.01, 0.025, 0.05, 0.10, 0.15, 0.20):
        low_tails = chi2.sf(low_arr, df=max(df, 1))
        high_tails = chi2.sf(high_arr, df=max(df, 1))
        tp = int(np.sum(high_tails < cand_tail))
        fp = int(np.sum(low_tails < cand_tail))
        fn = int(np.sum(high_tails >= cand_tail))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_tail = float(cand_tail)

    return {
        "calibration_source": "ravdess+besst",
        "calibration_df": max(df, 1),
        "besst_low_load_d_sq_p95": round(low_p95, 4),
        "besst_high_load_d_sq_p50": round(high_p50, 4),
        "cognitive_load_d_sq_threshold": round(opt_d_sq, 4),
        "cognitive_load_d_sq_per_df_threshold": round(opt_d_sq / max(df, 1), 4),
        "tail_score_threshold": best_tail,
    }


def is_cognitive_load_elevated(
    d_sq: float,
    df: int,
    tail_score: float,
    tail_threshold: float = 0.05,
    d_sq_per_df_threshold: Optional[float] = None,
) -> bool:
    """Dimension-invariant cognitive load check across 4-feature audio and 8-feature multimodal turns."""
    if df <= 0:
        return False
    if tail_score < tail_threshold:
        return True
    if d_sq_per_df_threshold is not None and (d_sq / float(df)) >= d_sq_per_df_threshold:
        return True
    return False


def calibrate_mahalanobis_from_besst(
    besst_dir: Path,
    priors: PriorsData,
    labels_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calibrate Mahalanobis thresholds using BESST ECG/GSR-labeled cognitive load speech."""
    labels_map = load_besst_labels(labels_path) if labels_path is not None else {}
    low_d_sq: List[float] = []
    high_d_sq: List[float] = []

    if not besst_dir.is_dir():
        return calibrate_thresholds_from_d_sq_samples(low_d_sq, high_d_sq)

    wav_files = list(besst_dir.glob("**/*.wav"))
    for wav_path in wav_files:
        lbl = _infer_besst_load_label(wav_path, labels_map)
        if lbl not in ("low", "high"):
            continue
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(str(wav_path))
            pcm = data.astype(np.float64) / 32768.0 if data.dtype == np.int16 else data.astype(np.float64)
            if pcm.ndim > 1:
                pcm = np.mean(pcm, axis=1)
            pros = extract_prosody(pcm, sample_rate=sr)
            if pros.quality_flag != "ok":
                continue

            z_scores: Dict[str, float] = {}
            for fname in ("f0_semitone_sd", "jitter_local_pct", "shimmer_local_pct", "hnr_db"):
                val = getattr(pros, fname, None)
                prior = priors.features.get(fname)
                if val is not None and prior is not None and prior.pop_sigma > 0:
                    z_scores[fname] = (float(val) - prior.pop_median) / prior.pop_sigma
            if not z_scores:
                continue

            mah = compute_mahalanobis(
                feature_scores=z_scores,
                correlation_matrix=priors.correlation_matrix,
                feature_order=priors.feature_order,
                priors=priors.features,
            )
            if lbl == "low":
                low_d_sq.append(float(mah.d_sq))
            else:
                high_d_sq.append(float(mah.d_sq))
        except Exception as exc:
            logger.debug("Skipping BESST file %s: %s", wav_path.name, exc)

    return calibrate_thresholds_from_d_sq_samples(low_d_sq, high_d_sq, df=4)


def main() -> None:
    """CLI entry point for building priors and optional BESST threshold calibration."""
    parser = argparse.ArgumentParser(description="Build empirical acoustic priors.")
    parser.add_argument("--data-dir", type=str, default="data/ravdess", help="Directory of RAVDESS WAV files")
    parser.add_argument("--besst-dir", type=str, default="", help="Optional BESST WAV directory for load calibration")
    parser.add_argument("--besst-labels", type=str, default="", help="Optional BESST CSV/JSON cognitive load labels")
    parser.add_argument("--out", type=str, default="data/priors_en.json", help="Output path for priors JSON")
    parser.add_argument("--out-calibration", type=str, default="", help="Optional JSON output for BESST thresholds")
    args = parser.parse_args()

    audio_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    priors_data = process_dataset(audio_dir)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(priors_data.model_dump(), f, indent=2, allow_nan=False)

    logger.info("Saved prior distribution to %s", out_path)

    if args.besst_dir:
        cal = calibrate_mahalanobis_from_besst(
            besst_dir=Path(args.besst_dir),
            priors=priors_data,
            labels_path=Path(args.besst_labels) if args.besst_labels else None,
        )
        logger.info("BESST calibration result: %s", cal)
        if args.out_calibration:
            cal_path = Path(args.out_calibration)
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            with cal_path.open("w", encoding="utf-8") as cf:
                json.dump(cal, cf, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
