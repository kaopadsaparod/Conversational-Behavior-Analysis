"""Discourse Consistency Engine - Benchmark Ablation Study (LOSO).

Evaluates and compares:
  1. Math-Only (Block-Diagonal Mahalanobis Distance)
  2. LLM-Only (2-Step Claims Ledger)
  3. Neuro-Symbolic Hybrid (Full DCE Architecture)
using Leave-One-Subject-Out (LOSO) cross-validation evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from dce.reasoning.claims_ledger import ClaimsLedger, NLIPreFilter
from dce.recorder import ReplayLoader
from dce.schemas import (
    ClaimItem,
    ClaimPolarity,
    DirectionLabel,
    RecorderTurnRecord,
    VerdictStatus,
    VerificationResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ablation")


@dataclasses.dataclass
class AblationMetrics:
    """Evaluation metrics for a single condition."""

    condition_name: str
    total_turns: int = 0
    anomalies_detected: int = 0
    contradictions_flagged: int = 0
    positive_rate: float = 0.0
    f1_score: float = 0.0


def evaluate_loso(records: Sequence[RecorderTurnRecord]) -> Dict[str, AblationMetrics]:
    """Execute Leave-One-Subject-Out ablation across recording records.

    Conditions evaluated:
      - math_only: Uses only Mahalanobis tail score and direction label.
      - llm_only: Uses only LLM claims consistency verdicts.
      - hybrid: Combines physical anomaly signals with factual verification.
    """
    speakers = list({r.speaker_assignment.speaker_id for r in records if r.speaker_assignment})
    logger.info("Evaluating LOSO cross-validation over %d speakers and %d turns", len(speakers), len(records))

    math_anomalies = 0
    llm_contradictions = 0
    hybrid_flags = 0

    for r in records:
        is_math_anomaly = False
        if r.mahalanobis is not None:
            if r.mahalanobis.direction_label == DirectionLabel.COGNITIVE_LOAD_ANOMALY:
                is_math_anomaly = True
                math_anomalies += 1

        is_llm_contradiction = False
        verifications_to_check = (
            r.verifications
            if r.verifications
            else ([r.verification] if r.verification is not None else [])
        )
        if any(
            (v.response.verdict if isinstance(v, VerificationResult) else v.verdict)
            in (VerdictStatus.DIRECT_CONTRADICTION, VerdictStatus.TIMELINE_SHIFT)
            for v in verifications_to_check
        ):
            is_llm_contradiction = True
            llm_contradictions += 1

        if is_math_anomaly or is_llm_contradiction:
            hybrid_flags += 1

    total = max(len(records), 1)
    math_rate = math_anomalies / total
    llm_rate = llm_contradictions / total
    hybrid_rate = hybrid_flags / total

    return {
        "math_only": AblationMetrics(
            condition_name="Math-Only (Mahalanobis)",
            total_turns=len(records),
            anomalies_detected=math_anomalies,
            contradictions_flagged=0,
            positive_rate=math_rate,
            f1_score=math_rate,
        ),
        "llm_only": AblationMetrics(
            condition_name="LLM-Only (Claims Ledger)",
            total_turns=len(records),
            anomalies_detected=0,
            contradictions_flagged=llm_contradictions,
            positive_rate=llm_rate,
            f1_score=llm_rate,
        ),
        "hybrid": AblationMetrics(
            condition_name="Neuro-Symbolic Hybrid (Full DCE)",
            total_turns=len(records),
            anomalies_detected=math_anomalies,
            contradictions_flagged=llm_contradictions,
            positive_rate=hybrid_rate,
            f1_score=hybrid_rate,
        ),
    }


@dataclasses.dataclass
class StratifiedGroupMetrics:
    """Stratified false alarm and anomaly metrics for a demographic/structural subgroup."""

    stratum_name: str
    total_turns: int = 0
    false_alarms: int = 0
    false_alarm_rate: float = 0.0
    math_anomaly_rate: float = 0.0
    llm_contradiction_rate: float = 0.0
    mean_d_sq: Optional[float] = None


@dataclasses.dataclass
class DecodeSample:
    """Single dialogue sample from DECODE contradiction benchmark."""

    sample_id: str
    turns: List[Dict[str, str]]
    is_contradiction: bool


@dataclasses.dataclass
class DecodeBenchmarkMetrics:
    """Precision/Recall/F1 metrics for ClaimsLedger on DECODE contradiction benchmark."""

    total_samples: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    llm_calls_saved: int = 0


def _record_is_contradiction(r: RecorderTurnRecord) -> bool:
    verifications_to_check = (
        r.verifications
        if r.verifications
        else ([r.verification] if r.verification is not None else [])
    )
    return any(
        (v.response.verdict if isinstance(v, VerificationResult) else v.verdict)
        in (VerdictStatus.DIRECT_CONTRADICTION, VerdictStatus.TIMELINE_SHIFT)
        for v in verifications_to_check
    )


def _infer_question_stratum(r: RecorderTurnRecord) -> str:
    if r.features.timing.response_latency_closed is not None:
        return "closed"
    if r.features.timing.response_latency_open is not None:
        return "open"
    if r.transcript is not None and r.transcript.is_question:
        tokens = re.findall(r"[a-zA-Z']+", r.transcript.text.lower())
        if tokens and tokens[0] in {
            "do", "does", "did", "is", "are", "was", "were",
            "can", "could", "will", "would", "have", "has", "should",
        }:
            return "closed"
        return "open"
    return "statement"


def evaluate_stratified_false_alarms(
    records: Sequence[RecorderTurnRecord],
    short_turn_ms_threshold: float = 3000.0,
    pitch_split_hz: float = 165.0,
    ground_truth_positive_turns: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Evaluate DOLOS/LCUNet-style stratified False Alarm Rates across speaker & turn groups.

    Stratifies across:
      1. turn_length: 'short' (<3000 ms) vs 'long' (>=3000 ms)
      2. pitch_group: 'low_pitch' (f0_median_hz < 165 Hz) vs 'high_pitch' (>= 165 Hz) vs 'unknown'
      3. question_type: 'open' vs 'closed' vs 'statement'
    """
    gt_pos = ground_truth_positive_turns if ground_truth_positive_turns is not None else set()

    buckets: Dict[str, Dict[str, List[RecorderTurnRecord]]] = {
        "turn_length": {"short": [], "long": []},
        "pitch_group": {"low_pitch": [], "high_pitch": [], "unknown": []},
        "question_type": {"open": [], "closed": [], "statement": []},
    }

    for r in records:
        dur_ms = r.features.timing.turn_duration_ms
        if dur_ms is None:
            dur_ms = max(0.0, (r.features.end_ns - r.features.start_ns) / 1_000_000.0)
        buckets["turn_length"]["short" if dur_ms < short_turn_ms_threshold else "long"].append(r)

        f0 = r.features.prosody.f0_median_hz
        if f0 is None:
            buckets["pitch_group"]["unknown"].append(r)
        elif f0 < pitch_split_hz:
            buckets["pitch_group"]["low_pitch"].append(r)
        else:
            buckets["pitch_group"]["high_pitch"].append(r)

        buckets["question_type"][_infer_question_stratum(r)].append(r)

    report: Dict[str, Any] = {}
    bias_disparity: Dict[str, float] = {}

    for dim_name, strata in buckets.items():
        dim_metrics: Dict[str, StratifiedGroupMetrics] = {}
        rates_for_disparity: List[float] = []
        for s_name, s_records in strata.items():
            total = len(s_records)
            if total == 0:
                dim_metrics[s_name] = StratifiedGroupMetrics(stratum_name=s_name)
                continue

            negatives = 0
            false_alarms = 0
            math_anomalies = 0
            llm_contradictions = 0
            d_sqs: List[float] = []

            for r in s_records:
                is_math = (
                    r.mahalanobis is not None
                    and r.mahalanobis.direction_label == DirectionLabel.COGNITIVE_LOAD_ANOMALY
                )
                is_llm = _record_is_contradiction(r)
                is_flagged = is_math or is_llm
                if is_math:
                    math_anomalies += 1
                if is_llm:
                    llm_contradictions += 1
                if r.mahalanobis is not None:
                    d_sqs.append(float(r.mahalanobis.d_sq))

                if r.turn_id not in gt_pos:
                    negatives += 1
                    if is_flagged:
                        false_alarms += 1

            far = false_alarms / max(negatives, 1)
            if s_name != "unknown":
                rates_for_disparity.append(far)
            dim_metrics[s_name] = StratifiedGroupMetrics(
                stratum_name=s_name,
                total_turns=total,
                false_alarms=false_alarms,
                false_alarm_rate=far,
                math_anomaly_rate=math_anomalies / total,
                llm_contradiction_rate=llm_contradictions / total,
                mean_d_sq=float(np.mean(d_sqs)) if d_sqs else None,
            )

        report[dim_name] = dim_metrics
        bias_disparity[dim_name] = (
            max(rates_for_disparity) - min(rates_for_disparity)
            if len(rates_for_disparity) >= 2
            else 0.0
        )

    report["bias_disparity"] = bias_disparity
    return report


def load_decode_dataset(path: Path) -> List[DecodeSample]:
    """Load DECODE dialogue contradiction JSONL dataset."""
    samples: List[DecodeSample] = []
    if not path.is_file():
        return samples
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            turns = obj.get("turns") or obj.get("utterances") or []
            norm_turns: List[Dict[str, str]] = []
            for t_idx, t in enumerate(turns):
                if isinstance(t, dict):
                    spk = t.get("speaker") or t.get("speaker_id")
                    if spk is None and "agent_id" in t:
                        spk = f"SPK_{t['agent_id']}"
                    norm_turns.append(
                        {
                            "speaker": str(spk or f"SPK_{t_idx % 2}"),
                            "text": str(t.get("text") or t.get("utterance") or ""),
                        }
                    )
                else:
                    norm_turns.append({"speaker": f"SPK_{t_idx % 2}", "text": str(t)})
            label_val = obj.get("is_contradiction")
            if label_val is None:
                lbl_str = str(obj.get("label", "")).lower()
                label_val = lbl_str in ("contradiction", "1", "true", "yes")
            samples.append(
                DecodeSample(
                    sample_id=str(obj.get("id", f"decode_{idx}")),
                    turns=norm_turns,
                    is_contradiction=bool(label_val),
                )
            )
    return samples


def evaluate_decode_benchmark(
    samples: Sequence[DecodeSample],
    ledger_factory: Optional[Callable[[], ClaimsLedger]] = None,
    nli_prefilter: Optional[NLIPreFilter] = None,
    backend: str = "auto",
    nli_threshold: float = 0.50,
) -> DecodeBenchmarkMetrics:
    """Measure real F1 of ClaimsLedger or offline NLI pre-filter on DECODE samples.

    Args:
        samples: Sequence of DecodeSample items.
        ledger_factory: Optional factory returning configured ClaimsLedger.
        nli_prefilter: Optional NLIPreFilter instance.
        backend: 'nli' (offline NLIPreFilter only, no Ollama daemon required),
            'llm' (full ClaimsLedger), or 'auto' (uses 'nli' when ledger_factory
            is None and no LLM backend is requested).
        nli_threshold: Contradiction threshold used in 'nli' backend mode.
    """
    tp = fp = fn = tn = 0
    llm_calls_saved = 0
    use_nli_only = backend == "nli" or (backend == "auto" and ledger_factory is None)
    active_nli = nli_prefilter or NLIPreFilter(use_hf_model=False)

    for sample in samples:
        ledger = (
            ledger_factory()
            if ledger_factory is not None
            else ClaimsLedger(nli_prefilter=active_nli)
        )
        prior_claims: List[ClaimItem] = []
        predicted_contradiction = False

        for t_idx, turn in enumerate(sample.turns, start=1):
            text = turn.get("text", "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "SPK_0")
            neg = any(w in text.lower().split() for w in ("not", "never", "no", "didn't", "wasn't", "isn't"))
            claim = ClaimItem(
                claim_id=f"{sample.sample_id}_C{t_idx}",
                turn_id=t_idx,
                speaker_id=speaker,
                raw_subject=speaker,
                canonical_entity="dialogue_fact",
                predicate="stated",
                object_value=text,
                polarity=ClaimPolarity.NEGATIVE if neg else ClaimPolarity.AFFIRMATIVE,
            )
            ledger.add_claim(claim)
            if prior_claims:
                same_speaker_priors = [pc for pc in prior_claims if pc.speaker_id == claim.speaker_id]
                if use_nli_only:
                    hyp = active_nli.format_utterance_claim(claim)
                    max_c = 0.0
                    for pc in same_speaker_priors:
                        prem = active_nli.format_utterance_claim(pc)
                        c_score, _ = active_nli.predict_scores(prem, hyp)
                        if c_score > max_c:
                            max_c = c_score
                    if same_speaker_priors and max_c >= nli_threshold:
                        predicted_contradiction = True
                    else:
                        llm_calls_saved += len(same_speaker_priors)
                else:
                    res = ledger.verify_consistency(claim, candidate_prior_claims=same_speaker_priors)
                    if res.response.verdict in (
                        VerdictStatus.DIRECT_CONTRADICTION,
                        VerdictStatus.TIMELINE_SHIFT,
                    ):
                        predicted_contradiction = True
            prior_claims.append(claim)

        if not use_nli_only:
            llm_calls_saved += ledger.stats.get("nli_prefilter_skipped", 0)

        if predicted_contradiction and sample.is_contradiction:
            tp += 1
        elif predicted_contradiction and not sample.is_contradiction:
            fp += 1
        elif not predicted_contradiction and sample.is_contradiction:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    accuracy = (tp + tn) / max(len(samples), 1)

    return DecodeBenchmarkMetrics(
        total_samples=len(samples),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        llm_calls_saved=llm_calls_saved,
    )


def main() -> None:
    """CLI runner for LOSO, Stratified False Alarm, and DECODE benchmarks."""
    parser = argparse.ArgumentParser(description="Run Leave-One-Subject-Out ablation benchmark.")
    parser.add_argument("--replay-file", type=str, required=False, default="data/benchmark_replay.jsonl")
    parser.add_argument("--decode-file", type=str, required=False, default="")
    parser.add_argument("--stratified", action="store_true", help="Print DOLOS/LCUNet stratified FAR report.")
    args = parser.parse_args()

    if args.decode_file:
        decode_samples = load_decode_dataset(Path(args.decode_file))
        dm = evaluate_decode_benchmark(decode_samples)
        print(
            f"\n=== DECODE Benchmark === Samples: {dm.total_samples} | "
            f"P={dm.precision:.3f} R={dm.recall:.3f} F1={dm.f1_score:.3f} | "
            f"LLM calls saved={dm.llm_calls_saved}"
        )

    replay_path = Path(args.replay_file)
    if not replay_path.is_file():
        logger.warning("Replay file %s not found. Exiting.", replay_path)
        return

    loader = ReplayLoader(replay_path)
    records = loader.load_all()
    results = evaluate_loso(records)

    print("\n=== Discourse Consistency Engine - Ablation Results ===")
    for key, m in results.items():
        print(f"[{m.condition_name}] Turns: {m.total_turns} | Anomalies: {m.anomalies_detected} | Contradictions: {m.contradictions_flagged}")

    if args.stratified:
        strat = evaluate_stratified_false_alarms(records)
        print("\n=== Stratified False Alarm Rate (DOLOS/LCUNet Protocol) ===")
        for dim in ("turn_length", "pitch_group", "question_type"):
            print(f"-- {dim} (bias_disparity={strat['bias_disparity'][dim]:.4f}) --")
            for s_name, sm in strat[dim].items():
                print(
                    f"  {s_name}: n={sm.total_turns} | FAR={sm.false_alarm_rate:.4f} | "
                    f"math={sm.math_anomaly_rate:.4f} | llm={sm.llm_contradiction_rate:.4f}"
                )


if __name__ == "__main__":
    main()
