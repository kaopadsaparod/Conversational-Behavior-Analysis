"""Statistical Significance & Full-Power Benchmark Evaluation.

Computes:
  1. 10,000-iteration stratified bootstrap 95% confidence intervals [ci_low, ci_high]
     for Accuracy, F1, Precision, and Recall.
  2. Exact Binomial p-values against chance (0.5) and majority-class baselines.
  3. Exact McNemar's test p-values for paired classifier comparisons.
  4. Out-of-fold 5-fold cross-validated physiological metrics across ALL 121 RLDD videos (N=121)
     with strict per-fold threshold tuning (zero held-out leakage).
  5. Expanded longitudinal Werewolf-Among-Us evaluation across >= 20 complete games (N >= 100
     player-role instances).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, fisher_exact

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def compute_classification_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, and confusion counts."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    tn = int(np.sum((~yt) & (~yp)))
    n = int(len(yt))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    acc = (tp + tn) / max(n, 1)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + spec)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def stratified_bootstrap_ci(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 10,000-iteration stratified bootstrap 95% CIs and exact Binomial p-values."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    n = len(yt)
    if n == 0:
        raise ValueError("Cannot bootstrap empty predictions.")

    point = compute_classification_metrics(yt, yp)
    pos_idx = np.where(yt)[0]
    neg_idx = np.where(~yt)[0]

    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot, dtype=np.float64)
    bal_accs = np.empty(n_boot, dtype=np.float64)
    precs = np.empty(n_boot, dtype=np.float64)
    recs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([s_pos, s_neg])
        else:
            idx = rng.choice(n, size=n, replace=True)
        yt_b = yt[idx]
        yp_b = yp[idx]
        tp = int(np.sum(yt_b & yp_b))
        fp = int(np.sum((~yt_b) & yp_b))
        fn = int(np.sum(yt_b & (~yp_b)))
        tn = int(np.sum((~yt_b) & (~yp_b)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        sp = tn / max(tn + fp, 1)
        f = (2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0
        accs[b] = (tp + tn) / float(n)
        bal_accs[b] = 0.5 * (r + sp)
        precs[b] = p
        recs[b] = r
        f1s[b] = f

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100.0 * alpha, 100.0 * (1.0 - alpha)

    correct = int(point["tp"] + point["tn"])
    maj_rate = max(len(pos_idx), len(neg_idx)) / float(n)
    p_vs_majority = (
        float(binomtest(correct, n, p=min(max(maj_rate, 1e-6), 1.0 - 1e-6), alternative="greater").pvalue)
        if maj_rate < 1.0
        else 1.0
    )
    p_vs_chance = (
        float(binomtest(correct, n, p=0.5, alternative="greater").pvalue)
        if len(pos_idx) == len(neg_idx) or abs(len(pos_idx) - len(neg_idx)) <= 1
        else p_vs_majority
    )
    p_fisher = float(
        fisher_exact(
            [[point["tp"], point["fp"]], [point["fn"], point["tn"]]],
            alternative="two-sided",
        ).pvalue
    )

    def _ci_pair(arr: np.ndarray) -> List[float]:
        return [round(float(np.percentile(arr, lo_q)), 4), round(float(np.percentile(arr, hi_q)), 4)]

    return {
        "n": n,
        "n_pos": int(len(pos_idx)),
        "n_neg": int(len(neg_idx)),
        "majority_baseline": round(maj_rate, 4),
        "n_boot": n_boot,
        "accuracy": round(point["accuracy"], 4),
        "accuracy_ci95": _ci_pair(accs),
        "balanced_accuracy": round(point["balanced_accuracy"], 4),
        "balanced_accuracy_ci95": _ci_pair(bal_accs),
        "f1": round(point["f1"], 4),
        "f1_ci95": _ci_pair(f1s),
        "precision": round(point["precision"], 4),
        "precision_ci95": _ci_pair(precs),
        "recall": round(point["recall"], 4),
        "recall_ci95": _ci_pair(recs),
        "confusion": {"tp": point["tp"], "fp": point["fp"], "fn": point["fn"], "tn": point["tn"]},
        "p_value_vs_chance": p_vs_chance,
        "p_value_vs_majority": p_vs_majority,
        "p_fisher": p_fisher,
    }


def mcnemar_test(
    y_true: Sequence[bool],
    pred_a: Sequence[bool],
    pred_b: Sequence[bool],
) -> Dict[str, Any]:
    """Compute exact Binomial and continuity-corrected chi2 McNemar test between two classifiers."""
    yt = np.asarray(y_true, dtype=bool)
    pa = np.asarray(pred_a, dtype=bool)
    pb = np.asarray(pred_b, dtype=bool)
    if len(yt) != len(pa) or len(yt) != len(pb):
        raise ValueError("y_true, pred_a, and pred_b must have identical length.")

    a_ok = pa == yt
    b_ok = pb == yt
    n_11 = int(np.sum(a_ok & b_ok))
    n_10 = int(np.sum(a_ok & (~b_ok)))  # A right, B wrong
    n_01 = int(np.sum((~a_ok) & b_ok))  # A wrong, B right
    n_00 = int(np.sum((~a_ok) & (~b_ok)))

    discordant = n_10 + n_01
    if discordant == 0:
        p_exact = 1.0
        chi2_cc = 0.0
    else:
        p_exact = float(binomtest(min(n_10, n_01), discordant, p=0.5, alternative="two-sided").pvalue)
        chi2_cc = float((max(0.0, abs(n_10 - n_01) - 1.0) ** 2) / float(discordant))

    return {
        "n": int(len(yt)),
        "both_correct": n_11,
        "a_only_correct": n_10,
        "b_only_correct": n_01,
        "both_wrong": n_00,
        "chi2_cc": round(chi2_cc, 4),
        "p_value_exact": p_exact,
    }


def evaluate_rldd_all_121_5fold_oof(n_boot: int = 10000) -> Dict[str, Any]:
    """Run strict out-of-fold 5-fold CV physiological evaluation across all 121 RLDD videos."""
    from dce.schemas import FEATURE_GROUPS
    from benchmarks.recalibrate_and_cv import (
        WILD_PRIORS_PATH,
        _extract_clip_turns,
        _score_clip_phys,
    )

    rldd_root = REPO_ROOT / "dataset" / "archive" / "Real-life Deception Detection Dataset With Train Test"
    mp4s = sorted((rldd_root / "Train").glob("*.mp4")) + sorted((rldd_root / "Test").glob("*.mp4"))
    wild = json.loads(WILD_PRIORS_PATH.read_text(encoding="utf-8"))
    feature_order = list(FEATURE_GROUPS.keys())
    src_order = list(wild["feature_order"])
    raw_corr = np.asarray(wild["correlation_matrix"], dtype=np.float64)
    idx_map = {name: i for i, name in enumerate(src_order)}
    corr_matrix = np.eye(len(feature_order), dtype=np.float64)
    for i, fi in enumerate(feature_order):
        for j, fj in enumerate(feature_order):
            if fi in idx_map and fj in idx_map:
                corr_matrix[i, j] = raw_corr[idx_map[fi], idx_map[fj]]

    clip_records: List[Tuple[str, bool, float, int]] = []
    for p in mp4s:
        is_dec = "lie" in p.stem.lower()
        turns = _extract_clip_turns(p)
        mr, v4, _ = _score_clip_phys(turns, corr_matrix, feature_order)
        clip_records.append((p.name, is_dec, mr, v4))

    rng = np.random.default_rng(42)
    pos_idx = [i for i, r in enumerate(clip_records) if r[1]]
    neg_idx = [i for i, r in enumerate(clip_records) if not r[1]]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    folds = [pos_idx[k::5] + neg_idx[k::5] for k in range(5)]

    candidates = [1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    oof_true: List[bool] = [False] * len(clip_records)
    oof_pred: List[bool] = [False] * len(clip_records)
    fold_thresholds: List[float] = []

    for val_fold in folds:
        val_set = set(val_fold)
        train_idx = [i for i in range(len(clip_records)) if i not in val_set]
        best_thr = 2.2
        best_f1 = -1.0
        for thr in candidates:
            yt_tr = [clip_records[i][1] for i in train_idx]
            yp_tr = [(clip_records[i][2] >= thr and clip_records[i][3] >= 2) for i in train_idx]
            f1_tr = compute_classification_metrics(yt_tr, yp_tr)["f1"]
            if f1_tr > best_f1:
                best_f1 = f1_tr
                best_thr = thr
        fold_thresholds.append(best_thr)
        for i in val_fold:
            oof_true[i] = clip_records[i][1]
            oof_pred[i] = bool(clip_records[i][2] >= best_thr and clip_records[i][3] >= 2)

    stats = stratified_bootstrap_ci(oof_true, oof_pred, n_boot=n_boot, seed=42)
    stats["fold_thresholds"] = fold_thresholds
    return stats


def evaluate_werewolf_expanded(
    n_games: int = 24,
    n_boot: int = 10000,
    use_cached: bool = False,
) -> Dict[str, Any]:
    """Run longitudinal Werewolf-Among-Us evaluation across >= 20 complete games (N >= 100)."""
    from benchmarks.run_werewolf_full_games import OUT_PATH, evaluate_full_games

    if use_cached and OUT_PATH.is_file():
        res = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if int(res.get("n_games", 0)) >= n_games and res.get("players"):
            players = res["players"]
        else:
            res = evaluate_full_games(n_games=n_games)
            players = res["players"]
    else:
        res = evaluate_full_games(n_games=n_games)
        players = res["players"]

    y_true = [bool(p["is_deceptive"]) for p in players]
    pred_a = [bool(p["mode_a"]) for p in players]
    pred_b = [bool(p["mode_b"]) for p in players]
    pred_c = [bool(p["mode_c"]) for p in players]

    return {
        "games_evaluated": int(res["n_games"]),
        "total_players": int(res["n_players"]),
        "mode_a_phys": stratified_bootstrap_ci(y_true, pred_a, n_boot=n_boot, seed=42),
        "mode_b_nli": stratified_bootstrap_ci(y_true, pred_b, n_boot=n_boot, seed=42),
        "mode_c_hybrid": stratified_bootstrap_ci(y_true, pred_c, n_boot=n_boot, seed=42),
        "mcnemar_hybrid_vs_phys": mcnemar_test(y_true, pred_c, pred_a),
        "mcnemar_phys_vs_nli": mcnemar_test(y_true, pred_a, pred_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bootstrap 95% CIs, p-values, and full-power evaluations.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--skip-rldd-121", action="store_true")
    parser.add_argument("--werewolf-games", type=int, default=24)
    parser.add_argument("--use-cached-werewolf", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report: Dict[str, Any] = {"n_boot": args.n_boot}

    if not args.skip_rldd_121:
        print("Running 5-fold OOF physiological CV across all 121 RLDD videos...", flush=True)
        report["rldd_121_5fold_oof_phys"] = evaluate_rldd_all_121_5fold_oof(n_boot=args.n_boot)

    print(f"Running longitudinal Werewolf evaluation across {args.werewolf_games} games...", flush=True)
    report["werewolf_expanded"] = evaluate_werewolf_expanded(
        n_games=args.werewolf_games,
        n_boot=args.n_boot,
        use_cached=args.use_cached_werewolf,
    )

    report["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    out_path = RESULTS_DIR / "statistical_significance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
