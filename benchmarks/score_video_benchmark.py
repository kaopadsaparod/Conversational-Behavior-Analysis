"""Part 3 Scoring: Evaluate Mode A, Mode B, and Mode C on 30 Anonymized Videos.

Reads:
  - benchmarks/results/_hidden_video_key.json
  - benchmarks/results/video_blind_transcripts.json
  - benchmarks/results/video_blind_verdicts.json
  - benchmarks/results/video_unlocked_signals.json

Outputs:
  - benchmarks/results/video_benchmark_scores.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
HIDDEN_KEY_PATH = RESULTS_DIR / "_hidden_video_key.json"
TRANSCRIPTS_PATH = RESULTS_DIR / "video_blind_transcripts.json"
VERDICTS_PATH = RESULTS_DIR / "video_blind_verdicts.json"
SIGNALS_PATH = RESULTS_DIR / "video_unlocked_signals.json"
SCORES_OUT_PATH = RESULTS_DIR / "video_benchmark_scores.json"


def _compute_metrics(preds: List[bool], gts: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for p, g in zip(preds, gts) if p and g)
    fp = sum(1 for p, g in zip(preds, gts) if p and not g)
    tn = sum(1 for p, g in zip(preds, gts) if not p and not g)
    fn = sum(1 for p, g in zip(preds, gts) if not p and g)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    acc = (tp + tn) / max(len(gts), 1)
    return {
        "n": len(gts),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def _safe_mean(vals: List[float]) -> Any:
    return round(float(np.mean(vals)), 3) if vals else None


def main() -> None:
    hidden_key: Dict[str, Dict[str, Any]] = json.loads(HIDDEN_KEY_PATH.read_text(encoding="utf-8"))
    transcripts: List[Dict[str, Any]] = json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
    verdicts_map: Dict[str, Dict[str, Any]] = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    signals_map: Dict[str, Any] = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))

    transcripts_by_id = {x["anon_id"]: x for x in transcripts}

    clip_rows: List[Dict[str, Any]] = []
    for anon_id in sorted(hidden_key.keys()):
        gt_info = hidden_key[anon_id]
        is_deceptive = gt_info["ground_truth"] == "deceptive"
        split = gt_info["split"]

        tr_info = transcripts_by_id[anon_id]
        raw_texts = [t.get("raw_text", "").strip() for t in tr_info["turns"] if t.get("raw_text", "").strip()]
        full_transcript = " ".join(raw_texts)
        annotated_all = " ".join(t.get("annotated_text", "") for t in tr_info["turns"])
        unclear_tags = re.findall(r"\[[^\]]+\? p=[0-9.]+[^\]]*\]", annotated_all)

        sig_entry = signals_map[anon_id]
        turns_unlocked = sig_entry["unlock_result"]["turns"]
        locked_verdicts = sig_entry["unlock_result"]["locked_verdicts"]

        # Extract per-clip physiological summary
        art_rates = [t["prosody"]["articulation_rate"] for t in turns_unlocked if t["prosody"].get("articulation_rate") is not None]
        f0_sds = [t["prosody"]["f0_semitone_sd"] for t in turns_unlocked if t["prosody"].get("f0_semitone_sd") is not None]
        jitters = [t["prosody"]["jitter_local_pct"] for t in turns_unlocked if t["prosody"].get("jitter_local_pct") is not None]
        blinks = [t["vision"]["blink_rate_per_min"] for t in turns_unlocked if t["vision"].get("blink_rate_per_min") is not None]

        d_sq_per_df_list: List[float] = []
        has_cog_load_anomaly = False
        for t in turns_unlocked:
            m = t.get("mahalanobis") or {}
            df = m.get("degrees_of_freedom") or 0
            d_sq = m.get("d_sq")
            if df > 0 and d_sq is not None:
                d_sq_per_df_list.append(float(d_sq) / float(df))
            if m.get("direction_label") == "cognitive_load_anomaly":
                has_cog_load_anomaly = True

        mean_d_sq_df = _safe_mean(d_sq_per_df_list)
        max_d_sq_df = round(max(d_sq_per_df_list), 3) if d_sq_per_df_list else None

        # Mode A: Physiological Signals Only (mean d_sq/df >= 3.0 OR max d_sq/df >= 4.0)
        # Also track built-in direction_label == cognitive_load_anomaly
        mode_a_flag = bool(max_d_sq_df is not None and max_d_sq_df >= 3.5)
        mode_a_builtin = has_cog_load_anomaly

        # Mode B: Blind Transcript Semantic Reasoning Only
        raw_v = verdicts_map[anon_id]["verdicts"][0]
        locked_v = locked_verdicts[0] if locked_verdicts else {}
        verdict_label = locked_v.get("verdict", raw_v["verdict"])
        exp_str = locked_v.get("explanation", raw_v.get("explanation", ""))
        mode_b_flag = bool(
            verdict_label in ("DIRECT_CONTRADICTION", "TIMELINE_SHIFT", "NEEDS_AUDIO_REVIEW")
            or "[EVASION]" in exp_str
            or "[MODIFIER_EROSION]" in exp_str
        )

        # Mode C: Combined Neuro-Symbolic Triage Flag
        mode_c_flag = bool(mode_b_flag or mode_a_flag)

        clip_rows.append(
            {
                "anon_id": anon_id,
                "original_filename": gt_info["original_filename"],
                "split": split,
                "ground_truth": gt_info["ground_truth"],
                "is_deceptive": is_deceptive,
                "turn_count": tr_info["turn_count"],
                "transcript_snippet": full_transcript[:140] + ("..." if len(full_transcript) > 140 else ""),
                "full_transcript": full_transcript,
                "unclear_count": len(unclear_tags),
                "unclear_examples": unclear_tags[:5],
                "blind_verdict": verdict_label,
                "blind_explanation": exp_str,
                "art_rate": _safe_mean(art_rates),
                "f0_sd": _safe_mean(f0_sds),
                "jitter": _safe_mean(jitters),
                "blink_rpm": _safe_mean(blinks),
                "mean_d_sq_df": mean_d_sq_df,
                "max_d_sq_df": max_d_sq_df,
                "has_cog_load_anomaly": has_cog_load_anomaly,
                "mode_a_pred": mode_a_flag,
                "mode_a_builtin_pred": mode_a_builtin,
                "mode_b_pred": mode_b_flag,
                "mode_c_pred": mode_c_flag,
                "sliced_audio_count": len(sig_entry.get("sliced_audio", [])),
            }
        )

    def _eval_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        gts = [r["is_deceptive"] for r in rows]
        return {
            "mode_a_physio_d_sq_df_3_5": _compute_metrics([r["mode_a_pred"] for r in rows], gts),
            "mode_a_builtin_cog_load": _compute_metrics([r["mode_a_builtin_pred"] for r in rows], gts),
            "mode_b_semantic": _compute_metrics([r["mode_b_pred"] for r in rows], gts),
            "mode_c_combined": _compute_metrics([r["mode_c_pred"] for r in rows], gts),
        }

    test_rows = [r for r in clip_rows if r["split"] == "test_12"]
    ext_rows = [r for r in clip_rows if r["split"] == "train_ext_18"]

    summary = {
        "all_30": _eval_subset(clip_rows),
        "test_12": _eval_subset(test_rows),
        "train_ext_18": _eval_subset(ext_rows),
        "per_clip": clip_rows,
    }
    SCORES_OUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for subset_name in ("test_12", "all_30"):
        print(f"=== {subset_name} ===")
        for mode_name, m in summary[subset_name].items():
            print(
                f"  {mode_name:26s} | TP={m['tp']:2d} FP={m['fp']:2d} TN={m['tn']:2d} FN={m['fn']:2d} | "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} Acc={m['accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
