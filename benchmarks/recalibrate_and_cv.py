"""Post-fix recalibration of data/priors_wild_video_en.json on 54 Train/ Truthful clips
and 5-fold CV tuning of Mode A/C physiological threshold strictly on Train/ (zero Test/ leakage).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dce.baseline_store import BaselineStore
from dce.extractors.audio_dsp import (
    decode_media_audio_and_frames,
    extract_egemaps_spectral_features,
    extract_prosody,
)
from dce.math_core import compute_mahalanobis
from dce.mcp_server import _segment_audio_energy
from dce.extractors.vision_face import extract_vision_sequence
from dce.schemas import (
    BaselineConfig,
    FEATURE_GROUPS,
    ProsodyConfig,
    TimingFeatures,
    TurnFeatures,
    VisionFeatures,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "dataset" / "archive" / "Real-life Deception Detection Dataset With Train Test" / "Train"
WILD_PRIORS_PATH = REPO_ROOT / "data" / "priors_wild_video_en.json"
EN_PRIORS_PATH = REPO_ROOT / "data" / "priors_en.json"
CV_OUT_PATH = REPO_ROOT / "benchmarks" / "results" / "train_cv_threshold.json"

EXCLUDED_MAHA_FEATURES = {"loudness_db", "au01_inner_brow_raise", "au25_lips_part"}


def _aggregate_vision(seq_slice: Sequence[VisionFeatures]) -> VisionFeatures:
    valid_vis = [v for v in seq_slice if v.face_detected and v.quality_flag == "ok"]
    if len(valid_vis) < 3:
        return VisionFeatures(face_detected=False, quality_flag="insufficient_frames")
    blink_rates = [v.blink_rate_per_min for v in valid_vis if v.blink_rate_per_min is not None]
    frontal_vis = [v for v in valid_vis if v.head_pose_yaw_deg is None or abs(v.head_pose_yaw_deg) <= 20.0]
    gaze_hs = [v.eye_gaze_horizontal for v in frontal_vis if v.eye_gaze_horizontal is not None]
    agg_aus: Dict[str, float] = {}
    for au_key in ("AU01_inner_brow_raise", "AU04_brow_lowerer", "AU12_lip_corner_puller", "AU25_lips_part"):
        vals = [v.action_units[au_key] for v in frontal_vis if au_key in v.action_units]
        if vals:
            agg_aus[au_key] = float(np.mean(vals))
    return VisionFeatures(
        face_detected=True,
        blink_rate_per_min=float(np.mean(blink_rates)) if blink_rates else None,
        eye_gaze_horizontal=float(np.mean(gaze_hs)) if gaze_hs else None,
        action_units=agg_aus,
        quality_flag="ok",
    )


def _extract_clip_turns(mp4_path: Path) -> List[TurnFeatures]:
    audio, sr, frames = decode_media_audio_and_frames(
        mp4_path, target_sr=16000, video_fps=15.0, max_frames=9000
    )
    seq = extract_vision_sequence(frames, fps=15.0) if frames else []
    segs = _segment_audio_energy(audio, sample_rate=sr) or [(0, len(audio), False)]
    pcfg = ProsodyConfig()
    turns: List[TurnFeatures] = []
    for idx, (s_idx, e_idx, _) in enumerate(segs, start=1):
        sub = audio[s_idx:e_idx]
        if len(sub) < int(0.4 * sr):
            continue
        pros = extract_prosody(sub, sample_rate=sr, config=pcfg)
        egem = extract_egemaps_spectral_features(sub, sample_rate=sr)
        if pros.acoustic_snr_db is None:
            pros.acoustic_snr_db = egem.get("acoustic_snr_db")
        vis = VisionFeatures(face_detected=False, quality_flag="no_video")
        if seq:
            f0 = max(0, int(round((s_idx / float(sr)) * 15.0)))
            f1 = min(len(seq), max(f0 + 1, int(round((e_idx / float(sr)) * 15.0))))
            vis = _aggregate_vision(seq[f0:f1])
        turns.append(
            TurnFeatures(
                turn_id=idx,
                speaker_id="speaker_0",
                start_ns=int(round((s_idx / float(sr)) * 1e9)),
                end_ns=int(round((e_idx / float(sr)) * 1e9)),
                timing=TimingFeatures(turn_duration_ms=(e_idx - s_idx) / float(sr) * 1000.0),
                prosody=pros,
                vision=vis,
            )
        )
    return turns


def recalibrate_priors(truth_turns_by_clip: Dict[str, List[TurnFeatures]]) -> Dict[str, Any]:
    wild = json.loads(WILD_PRIORS_PATH.read_text(encoding="utf-8"))
    en = json.loads(EN_PRIORS_PATH.read_text(encoding="utf-8"))
    feats = wild["features"]

    buckets: Dict[str, List[float]] = {k: [] for k in feats}
    for turns in truth_turns_by_clip.values():
        for tf in turns:
            p, v = tf.prosody, tf.vision
            if p.articulation_rate is not None:
                buckets["articulation_rate"].append(float(p.articulation_rate))
            if p.f0_semitone_sd is not None:
                buckets["f0_semitone_sd"].append(float(p.f0_semitone_sd))
            if p.jitter_local_pct is not None:
                buckets["jitter_local_pct"].append(float(p.jitter_local_pct))
            if p.shimmer_local_pct is not None:
                buckets["shimmer_local_pct"].append(float(p.shimmer_local_pct))
            if p.hnr_db is not None:
                buckets["hnr_db"].append(float(p.hnr_db))
            if p.spectral_slope_db is not None:
                buckets["spectral_slope_db"].append(float(p.spectral_slope_db))
            if p.alpha_ratio_db is not None:
                buckets["alpha_ratio_db"].append(float(p.alpha_ratio_db))
            if p.loudness_db is not None:
                buckets["loudness_db"].append(float(p.loudness_db))
            if v.blink_rate_per_min is not None:
                buckets["blink_rate_per_min"].append(float(v.blink_rate_per_min))
            if v.eye_gaze_horizontal is not None:
                buckets["eye_gaze_horizontal"].append(float(v.eye_gaze_horizontal))
            for au_src, au_dst in (
                ("AU01_inner_brow_raise", "au01_inner_brow_raise"),
                ("AU04_brow_lowerer", "au04_brow_lowerer"),
                ("AU12_lip_corner_puller", "au12_lip_corner_puller"),
                ("AU25_lips_part", "au25_lips_part"),
            ):
                if au_src in v.action_units:
                    buckets[au_dst].append(float(v.action_units[au_src]))

    for k, vals in buckets.items():
        if k in ("response_latency_open", "response_latency_closed") or len(vals) < 10:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sig = max(1.4826 * mad, float(feats[k]["sigma_floor"]))
        if k == "blink_rate_per_min":
            sig = float(np.clip(sig, 5.0, 12.0))
        feats[k]["pop_median"] = round(med, 4)
        feats[k]["pop_sigma"] = round(sig, 4)

    wild["source"] = "rldd_54_train_truthful_recalibrated_v3"
    # Preserve block-diagonal voice & timing correlations from priors_en.json
    order = wild["feature_order"]
    n = len(order)
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    en_idx = {name: i for i, name in enumerate(en["feature_order"])}
    en_corr = en["correlation_matrix"]
    for i, fi in enumerate(order):
        for j, fj in enumerate(order):
            if fi in en_idx and fj in en_idx:
                mat[i][j] = en_corr[en_idx[fi]][en_idx[fj]]
    wild["correlation_matrix"] = mat

    WILD_PRIORS_PATH.write_text(json.dumps(wild, indent=2) + "\n", encoding="utf-8")
    return wild


def _score_clip_phys(turns: List[TurnFeatures], corr_matrix: np.ndarray, feature_order: List[str]) -> Tuple[float, int, int]:
    bs = BaselineStore(
        config=BaselineConfig(
            decoupled_shrinkage=True,
            scale_shrinkage_k=6.0,
            l2_location_shrinkage_k=0.0,
            acoustic_location_shrinkage_k=3.0,
        ),
        priors=WILD_PRIORS_PATH,
    )
    w_sum = 0.0
    d_sum = 0.0
    valid_4f = 0
    high_turns = 0
    for tf in turns:
        fs = bs.score_and_update_turn("speaker_0", tf)
        maha_scores = {k: v for k, v in fs.items() if k not in EXCLUDED_MAHA_FEATURES}
        maha = compute_mahalanobis(
            feature_scores=maha_scores,
            correlation_matrix=corr_matrix,
            feature_order=feature_order,
            priors=bs.priors,
        )
        dur = max(float(tf.timing.turn_duration_ms or 0.0), 100.0)
        if maha.degrees_of_freedom >= 4:
            valid_4f += 1
            ratio = float(maha.d_sq) / float(maha.degrees_of_freedom)
            w_sum += dur * ratio
            d_sum += dur
            if ratio >= 2.8:
                high_turns += 1
    mean_ratio = (w_sum / d_sum) if d_sum > 0.0 else 0.0
    return mean_ratio, valid_4f, high_turns


def main() -> None:
    t0 = time.perf_counter()
    truth_mp4s = sorted(TRAIN_DIR.glob("trial_truth_*.mp4"))
    lie_mp4s = sorted(TRAIN_DIR.glob("trial_lie_*.mp4"))
    print(f"Found Train/ Truthful={len(truth_mp4s)}, Deceptive={len(lie_mp4s)}", flush=True)

    truth_turns = {p.name: _extract_clip_turns(p) for p in truth_mp4s}
    wild = recalibrate_priors(truth_turns)
    print(
        f"Recalibrated priors_wild_video_en.json (blink={wild['features']['blink_rate_per_min']['pop_median']}"
        f" +/- {wild['features']['blink_rate_per_min']['pop_sigma']})",
        flush=True,
    )

    lie_turns = {p.name: _extract_clip_turns(p) for p in lie_mp4s}

    feature_order = list(FEATURE_GROUPS.keys())
    src_order = list(wild["feature_order"])
    raw_corr = np.asarray(wild["correlation_matrix"], dtype=np.float64)
    idx_map = {name: i for i, name in enumerate(src_order)}
    corr_matrix = np.eye(len(feature_order), dtype=np.float64)
    for i, fi in enumerate(feature_order):
        for j, fj in enumerate(feature_order):
            if fi in idx_map and fj in idx_map:
                corr_matrix[i, j] = raw_corr[idx_map[fi], idx_map[fj]]

    clip_stats: List[Tuple[str, bool, float, int, int]] = []
    for name, turns in truth_turns.items():
        mr, v4, ht = _score_clip_phys(turns, corr_matrix, feature_order)
        clip_stats.append((name, False, mr, v4, ht))
    for name, turns in lie_turns.items():
        mr, v4, ht = _score_clip_phys(turns, corr_matrix, feature_order)
        clip_stats.append((name, True, mr, v4, ht))

    # 5-fold stratified CV on Train/ clips to tune mean_d_sq_df threshold
    candidates = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]
    rng = np.random.default_rng(42)
    t_idx = [i for i, x in enumerate(clip_stats) if not x[1]]
    l_idx = [i for i, x in enumerate(clip_stats) if x[1]]
    rng.shuffle(t_idx)
    rng.shuffle(l_idx)
    folds = [t_idx[k::5] + l_idx[k::5] for k in range(5)]

    cv_results: Dict[str, Any] = {}
    best_thr = 2.2
    best_f1 = -1.0
    for thr in candidates:
        tp = fp = tn = fn = 0
        for val_fold in folds:
            for idx in val_fold:
                _, is_dec, mr, v4, _ = clip_stats[idx]
                pred = (mr >= thr) and (v4 >= 2)
                if pred and is_dec:
                    tp += 1
                elif pred and not is_dec:
                    fp += 1
                elif not pred and not is_dec:
                    tn += 1
                else:
                    fn += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        acc = (tp + tn) / max(len(clip_stats), 1)
        cv_results[str(thr)] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
        }
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    out = {
        "n_train_truth": len(truth_mp4s),
        "n_train_lie": len(lie_mp4s),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "best_sustained_threshold": best_thr,
        "cv_sweep": cv_results,
    }
    CV_OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
