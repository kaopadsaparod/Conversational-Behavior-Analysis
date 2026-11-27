"""Part 2 Step 1: DECODE Balanced 100-Sample Benchmark (Heuristic + DeBERTa Stage 1).

Exports Stage-1 flagged dialogues (threshold=0.72, same_speaker_only=True) to
benchmarks/results/decode_stage2_blind_inputs.json with ALL ground-truth labels
stripped out for strict blind Stage-2 semantic reasoning.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.run_ablation import DecodeSample, evaluate_decode_benchmark, load_decode_dataset
from dce.reasoning.claims_ledger import NLIPreFilter
from dce.schemas import ClaimItem, ClaimPolarity

REPO_ROOT = Path(__file__).resolve().parent.parent
DECODE_DEV = REPO_ROOT / "dataset" / "decode_v0.1" / "dev.jsonl"
BLIND_INPUTS_PATH = REPO_ROOT / "benchmarks" / "results" / "decode_stage2_blind_inputs.json"
HIDDEN_GT_PATH = REPO_ROOT / "benchmarks" / "results" / "_hidden_decode_gt.json"
STAGE1_METRICS_PATH = REPO_ROOT / "benchmarks" / "results" / "decode_stage1_metrics.json"


def sample_balanced_100(samples: List[DecodeSample], seed: int = 42) -> List[DecodeSample]:
    rng = random.Random(seed)
    pos = [s for s in samples if s.is_contradiction]
    neg = [s for s in samples if not s.is_contradiction]
    rng.shuffle(pos)
    rng.shuffle(neg)
    selected = pos[:50] + neg[:50]
    rng.shuffle(selected)
    return selected


def main() -> None:
    all_samples = load_decode_dataset(DECODE_DEV)
    if not all_samples:
        raise FileNotFoundError(f"DECODE dev.jsonl not found or empty: {DECODE_DEV}")
    samples = sample_balanced_100(all_samples, seed=42)

    # (a) Lexical Heuristic baseline (threshold=0.50)
    t0 = time.perf_counter()
    heur_nli = NLIPreFilter(use_hf_model=False)
    heur_metrics = evaluate_decode_benchmark(
        samples, nli_prefilter=heur_nli, backend="nli", nli_threshold=0.50
    )
    heur_sec = round(time.perf_counter() - t0, 3)

    # (b) Stage 1: DeBERTa-v3-xsmall at calibrated threshold 0.72, same_speaker_only=True
    nli_threshold = 0.72
    t1 = time.perf_counter()
    deberta_nli = NLIPreFilter(
        model_name="cross-encoder/nli-deberta-v3-xsmall", use_hf_model=True
    )
    # Warm up model load
    deberta_nli.predict_scores("I went to Paris.", "I never went to Paris.")

    tp = fp = fn = tn = 0
    total_same_speaker_pairs = 0
    flagged_pairs_count = 0
    blind_inputs: List[Dict[str, Any]] = []
    hidden_gt: Dict[str, bool] = {}

    for sample in samples:
        hidden_gt[sample.sample_id] = bool(sample.is_contradiction)
        prior_claims: List[ClaimItem] = []
        flagged_pairs: List[Dict[str, Any]] = []
        max_score = 0.0

        for t_idx, turn in enumerate(sample.turns, start=1):
            text = turn.get("text", "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "SPK_0")
            claim = ClaimItem(
                claim_id=f"{sample.sample_id}_C{t_idx}",
                turn_id=t_idx,
                speaker_id=speaker,
                raw_subject=speaker,
                canonical_entity="dialogue_fact",
                predicate="stated",
                object_value=text,
                polarity=ClaimPolarity.AFFIRMATIVE,
            )
            same_priors = [pc for pc in prior_claims if pc.speaker_id == speaker]
            hyp = deberta_nli.format_utterance_claim(claim)
            for pc in same_priors:
                total_same_speaker_pairs += 1
                prem = deberta_nli.format_utterance_claim(pc)
                c_score, e_score = deberta_nli.predict_scores(prem, hyp)
                if c_score > max_score:
                    max_score = c_score
                if c_score >= nli_threshold:
                    flagged_pairs_count += 1
                    flagged_pairs.append(
                        {
                            "prior_turn_id": pc.turn_id,
                            "source_turn_id": claim.turn_id,
                            "speaker_id": speaker,
                            "prior_text": prem,
                            "source_text": hyp,
                            "nli_contradiction_score": round(c_score, 4),
                            "nli_entailment_score": round(e_score, 4) if e_score is not None else None,
                        }
                    )
            prior_claims.append(claim)

        pred_pos = len(flagged_pairs) > 0
        if pred_pos and sample.is_contradiction:
            tp += 1
        elif pred_pos and not sample.is_contradiction:
            fp += 1
        elif not pred_pos and sample.is_contradiction:
            fn += 1
        else:
            tn += 1

        if pred_pos:
            # STRICT ANTI-LEAKAGE: NO ground-truth labels in blind_inputs!
            blind_inputs.append(
                {
                    "sample_id": sample.sample_id,
                    "turns": [
                        {"turn_id": idx + 1, "speaker": t["speaker"], "text": t["text"]}
                        for idx, t in enumerate(sample.turns)
                    ],
                    "flagged_pairs": flagged_pairs,
                    "max_nli_score": round(max_score, 4),
                }
            )

    deberta_sec = round(time.perf_counter() - t1, 3)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    acc = (tp + tn) / max(len(samples), 1)
    llm_calls_saved = total_same_speaker_pairs - flagged_pairs_count

    BLIND_INPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLIND_INPUTS_PATH.write_text(json.dumps(blind_inputs, indent=2, ensure_ascii=False), encoding="utf-8")
    HIDDEN_GT_PATH.write_text(json.dumps(hidden_gt, indent=2), encoding="utf-8")

    stage1_summary = {
        "n_samples": len(samples),
        "n_positive": sum(1 for s in samples if s.is_contradiction),
        "n_negative": sum(1 for s in samples if not s.is_contradiction),
        "heuristic": {
            "tp": heur_metrics.tp,
            "fp": heur_metrics.fp,
            "fn": heur_metrics.fn,
            "tn": heur_metrics.tn,
            "precision": round(heur_metrics.precision, 4),
            "recall": round(heur_metrics.recall, 4),
            "f1": round(heur_metrics.f1_score, 4),
            "accuracy": round(heur_metrics.accuracy, 4),
            "llm_calls_saved": heur_metrics.llm_calls_saved,
            "elapsed_sec": heur_sec,
        },
        "deberta_stage1": {
            "threshold": nli_threshold,
            "same_speaker_only": True,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "total_same_speaker_pairs": total_same_speaker_pairs,
            "flagged_pairs_count": flagged_pairs_count,
            "flagged_dialogues_count": len(blind_inputs),
            "llm_calls_saved": llm_calls_saved,
            "elapsed_sec": deberta_sec,
        },
    }
    STAGE1_METRICS_PATH.write_text(json.dumps(stage1_summary, indent=2), encoding="utf-8")
    print(json.dumps(stage1_summary, indent=2))


if __name__ == "__main__":
    main()
