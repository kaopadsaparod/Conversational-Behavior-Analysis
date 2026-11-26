"""Discourse Consistency Engine - Claims Ledger and Neuro-Symbolic Firewall.

Enforces a strict Neuro-Symbolic Firewall prohibiting physical signal names or numbers
in LLM prompts, context-aware question gating, structured Step A claim extraction,
Step B consistency verification with reference validation, and idempotent speaker merging.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from dce.reasoning.llm_adapter import LLMAdapter
from dce.schemas import (
    FEATURE_GROUPS,
    ClaimExtractionResponse,
    ClaimItem,
    ConsistencyVerificationResponse,
    ReasoningConfig,
    TurnTranscript,
    VerdictStatus,
    VerificationResult,
)

logger = logging.getLogger(__name__)

_FIREWALL_SINGLE_STEMS: Dict[str, Set[str]] = {
    "jitter": {"jitter", "jitters"},
    "shimmer": {"shimmer", "shimmers"},
    "hnr": {"hnr"},
    "f0": {"f0"},
    "semitone": {"semitone", "semitones"},
    "latency": {"latency", "latencies"},
    "blink": {"blink", "blinks"},
    "zscore": {"zscore", "zscores"},
    "mahalanobis": {"mahalanobis"},
    "chi2": {"chi2"},
    "polarity": {"polarity", "polarities"},
    "stress": {"stress", "stresses"},
    "anomaly": {"anomaly", "anomalies"},
}

_FIREWALL_PHRASES: Dict[Tuple[str, ...], str] = {
    ("vocal", "tension"): "vocal tension",
    ("vocal", "tensions"): "vocal tension",
    ("cognitive", "load"): "cognitive load",
    ("cognitive", "loads"): "cognitive load",
    ("z", "score"): "z_score",
    ("z", "scores"): "z_score",
    ("d", "sq"): "d_sq",
    **{tuple(feat.lower().split("_")): feat for feat in FEATURE_GROUPS},
}

PROHIBITED_FIREWALL_TERMS: Set[str] = {
    *_FIREWALL_SINGLE_STEMS.keys(),
    "semitones",
    "z_score",
    "mahalanobis_distance",
    "d_sq",
    "chi2_tail_score",
    "vocal_tension",
    "cognitive_load_anomaly",
    *FEATURE_GROUPS.keys(),
}

_SINGLE_WORD_FILLERS: Set[str] = {
    "yeah",
    "yep",
    "uh-huh",
    "mm-hmm",
    "mhm",
    "okay",
    "ok",
    "right",
    "sure",
    "thanks",
    "wow",
    "oh",
    "hmm",
    "um",
    "uh",
}

_MULTI_WORD_FILLERS: Set[Tuple[str, str]] = {
    ("i", "see"),
    ("got", "it"),
    ("thank", "you"),
}


def _tokenize_firewall(text: str) -> List[str]:
    """Tokenize text on non-alphanumeric boundaries (including '_') and camelCase."""
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    split_camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split_camel)
    return re.findall(r"[a-z0-9]+", split_camel.lower())


def assert_firewall_compliance(text: str) -> None:
    """Verify that text contains zero physical signal names or acoustic metric terms.

    Tokenizes by non-alphanumeric characters (including '_') and camelCase, then
    matches prohibited stems (allowing plurals), multi-word phrases, and all prior
    feature names.

    Args:
        text: Prompt framing or system instruction text destined for LLM ingestion.

    Raises:
        ValueError: If any prohibited physical signal name is detected.
    """
    tokens = _tokenize_firewall(text)
    for token in tokens:
        for stem, variants in _FIREWALL_SINGLE_STEMS.items():
            if token in variants:
                raise ValueError(
                    f"Neuro-Symbolic Firewall violation: LLM prompt contains prohibited physical signal term: '{stem}'"
                )

    n_tokens = len(tokens)
    for phrase, label in _FIREWALL_PHRASES.items():
        plen = len(phrase)
        if n_tokens >= plen:
            for idx in range(n_tokens - plen + 1):
                if tuple(tokens[idx : idx + plen]) == phrase:
                    raise ValueError(
                        f"Neuro-Symbolic Firewall violation: LLM prompt contains prohibited physical signal term: '{label}'"
                    )


def should_skip_extraction(
    transcript: TurnTranscript,
    prior_transcript: Optional[TurnTranscript] = None,
) -> bool:
    """Evaluate context-aware question gate for conversational utterances.

    Skips extraction ONLY if every token (lowercased, punctuation stripped while
    preserving apostrophes and internal hyphens) belongs to the acknowledgment/filler
    lexicon AND the preceding turn was NOT a question.

    Args:
        transcript: Current conversational turn transcript.
        prior_transcript: Preceding conversational turn transcript, if available.

    Returns:
        True if turn should be skipped from LLM claim extraction, False otherwise.
    """
    if prior_transcript is not None and prior_transcript.is_question:
        return False

    raw_words = transcript.text.strip().split()
    tokens = [re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", w).lower() for w in raw_words]
    tokens = [t for t in tokens if t]

    if not tokens:
        return True

    idx = 0
    while idx < len(tokens):
        if idx + 1 < len(tokens) and (tokens[idx], tokens[idx + 1]) in _MULTI_WORD_FILLERS:
            idx += 2
        elif tokens[idx] in _SINGLE_WORD_FILLERS:
            idx += 1
        else:
            return False

    return True


class ClaimsLedger:
    """In-memory verifiable proposition store and multi-speaker claim consistency tracker."""

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        retrieval_top_k: Optional[int] = None,
        config: Optional[ReasoningConfig] = None,
    ) -> None:
        """Initialize ClaimsLedger.

        Args:
            llm_adapter: Configured LLM adapter for structured extraction and verification.
            retrieval_top_k: Maximum number of prior claims retrieved per verification.
            config: Optional ReasoningConfig providing retrieval_top_k.
        """
        self._llm: LLMAdapter = llm_adapter if llm_adapter is not None else LLMAdapter()
        self._retrieval_top_k: int = (
            retrieval_top_k
            if retrieval_top_k is not None
            else (config.retrieval_top_k if config is not None else ReasoningConfig().retrieval_top_k)
        )
        self._claims: Dict[str, ClaimItem] = {}
        self._claims_by_turn: Dict[int, List[ClaimItem]] = {}
        self._claims_by_speaker: Dict[str, List[ClaimItem]] = {}
        self._entity_aliases: Dict[str, str] = {}
        self._next_claim_num: int = 1
        self.stats: Dict[str, int] = {"llm_errors": 0}

    @property
    def claims(self) -> Dict[str, ClaimItem]:
        """Return all registered claims."""
        return self._claims

    @property
    def claims_by_turn(self) -> Dict[int, List[ClaimItem]]:
        """Return claims indexed by conversational turn ID."""
        return self._claims_by_turn

    @property
    def claims_by_speaker(self) -> Dict[str, List[ClaimItem]]:
        """Return claims indexed by speaker ID."""
        return self._claims_by_speaker

    @property
    def entity_aliases(self) -> Dict[str, str]:
        """Return normalized entity alias dictionary."""
        return self._entity_aliases

    def normalize_entity(self, raw_entity: str) -> str:
        """Normalize entity text to canonical lowercase stripped representation."""
        clean = raw_entity.strip().lower()
        clean = re.sub(r"\s+", " ", clean)
        return self._entity_aliases.get(clean, clean)

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register an entity alias mapping to its canonical form."""
        norm_alias = re.sub(r"\s+", " ", alias.strip().lower())
        norm_canonical = re.sub(r"\s+", " ", canonical.strip().lower())
        self._entity_aliases[norm_alias] = norm_canonical

    def add_claim(self, claim: ClaimItem) -> None:
        """Store a structured claim into the ledger with normalized whitespace."""
        claim.raw_subject = re.sub(r"\s+", " ", claim.raw_subject.strip())
        claim.canonical_entity = self.normalize_entity(claim.canonical_entity or claim.raw_subject)
        if claim.time_ref_normalized is not None:
            norm_time = re.sub(r"\s+", " ", claim.time_ref_normalized.strip())
            claim.time_ref_normalized = norm_time if norm_time else None
        self._claims[claim.claim_id] = claim

        if claim.turn_id not in self._claims_by_turn:
            self._claims_by_turn[claim.turn_id] = []
        self._claims_by_turn[claim.turn_id].append(claim)

        if claim.speaker_id not in self._claims_by_speaker:
            self._claims_by_speaker[claim.speaker_id] = []
        self._claims_by_speaker[claim.speaker_id].append(claim)

    def extract_claims(
        self,
        transcript: TurnTranscript,
        context_turns: Optional[List[TurnTranscript]] = None,
    ) -> List[ClaimItem]:
        """Execute Step A: Extract verifiable factual propositions from speech transcript.

        Enforces Firewall verification on generated prompt.

        Args:
            transcript: Current conversational turn transcript.
            context_turns: Up to 2 preceding conversational turn transcripts.

        Returns:
            List of newly registered ClaimItem instances.
        """
        # Build prompt from separated framing and transcript-derived parts
        framing_context_str = ""
        transcript_context_str = ""
        if context_turns:
            recent_ctx = context_turns[-2:]
            lines = [f"{t.speaker_id}: {t.text}" for t in recent_ctx]
            framing_lines = [f"{t.speaker_id}: <transcript>" for t in recent_ctx]
            transcript_context_str = "Prior context:\n" + "\n".join(lines) + "\n\n"
            framing_context_str = "Prior context:\n" + "\n".join(framing_lines) + "\n\n"

        framing_template = (
            "Current utterance by {speaker_id} (Turn {turn_id}):\n"
            '"{utterance}"\n\n'
            "Extract all explicit, verifiable factual claims as structured propositions. "
            "Include canonical entities, predicates, object values, time references, and affirmative or negative status."
        )

        prompt_framing = framing_context_str + framing_template.format(
            speaker_id=transcript.speaker_id,
            turn_id=transcript.turn_id,
            utterance="<transcript>",
        )
        prompt = transcript_context_str + framing_template.format(
            speaker_id=transcript.speaker_id,
            turn_id=transcript.turn_id,
            utterance=transcript.text,
        )

        system_instruction = (
            "You are a factual claim extraction system. Extract structured subject-predicate-object "
            "propositions. Do not include opinions or filler words."
        )

        # Enforce Neuro-Symbolic Firewall ONLY on prompt framing and system instructions
        assert_firewall_compliance(prompt_framing)
        assert_firewall_compliance(system_instruction)

        extracted_claims: List[ClaimItem] = []
        try:
            response = self._llm.generate_structured(
                prompt=prompt,
                response_model=ClaimExtractionResponse,
                system_prompt=system_instruction,
            )

            for raw_claim in response.claims:
                claim_id = f"CLM_{transcript.turn_id:04d}_{self._next_claim_num:02d}"
                self._next_claim_num += 1

                canonical = self.normalize_entity(raw_claim.canonical_entity or raw_claim.raw_subject)

                claim_item = ClaimItem(
                    claim_id=claim_id,
                    turn_id=transcript.turn_id,
                    speaker_id=transcript.speaker_id,
                    raw_subject=raw_claim.raw_subject,
                    canonical_entity=canonical,
                    predicate=raw_claim.predicate,
                    object_value=raw_claim.object_value,
                    time_ref_raw=raw_claim.time_ref_raw,
                    time_ref_normalized=raw_claim.time_ref_normalized,
                    location_ref=raw_claim.location_ref,
                    polarity=raw_claim.polarity,
                    confidence=raw_claim.confidence,
                )
                self.add_claim(claim_item)
                extracted_claims.append(claim_item)

        except Exception as exc:
            self.stats["llm_errors"] += 1
            logger.warning("Step A claim extraction encountered error: %s", exc)

        return extracted_claims

    def verify_consistency(
        self,
        current_claim: ClaimItem,
        candidate_prior_claims: Optional[List[ClaimItem]] = None,
    ) -> VerificationResult:
        """Execute Step B: Verify consistency of a current claim against prior ledger claims.

        Validates referenced turn/claim existence against retrieved candidates, forcing
        REJECTED_INVALID_REF on invalid references, LLM_ERROR on backend failures, and
        computing dynamic conflict_scope ('self' vs 'cross_speaker').

        Args:
            current_claim: Newly extracted claim to verify.
            candidate_prior_claims: Optional subset of prior claims to compare against.

        Returns:
            VerificationResult wrapping ConsistencyVerificationResponse and conflict_scope.
        """
        def _norm_opt(val: Optional[str]) -> Optional[str]:
            if val is None:
                return None
            cleaned = re.sub(r"\s+", " ", val.strip().lower())
            return cleaned if cleaned else None

        if candidate_prior_claims is None:
            curr_entity = self.normalize_entity(current_claim.canonical_entity or current_claim.raw_subject)
            curr_subj = _norm_opt(current_claim.raw_subject)
            curr_time_norm = _norm_opt(current_claim.time_ref_normalized)
            curr_time_raw = _norm_opt(current_claim.time_ref_raw)

            candidates = [
                c
                for c in self._claims.values()
                if c.turn_id < current_claim.turn_id
                and (
                    c.canonical_entity == curr_entity
                    or (curr_subj is not None and _norm_opt(c.raw_subject) == curr_subj)
                    or (curr_time_norm is not None and _norm_opt(c.time_ref_normalized) == curr_time_norm)
                    or (curr_time_raw is not None and _norm_opt(c.time_ref_raw) == curr_time_raw)
                )
            ]
            candidates.sort(key=lambda c: (c.turn_id, c.claim_id), reverse=True)
            candidates = candidates[: self._retrieval_top_k]
        else:
            candidates = sorted(
                candidate_prior_claims,
                key=lambda c: (c.turn_id, c.claim_id),
                reverse=True,
            )[: self._retrieval_top_k]

        if not candidates:
            return VerificationResult(
                response=ConsistencyVerificationResponse(
                    verdict=VerdictStatus.UNVERIFIABLE,
                    source_turn_id=current_claim.turn_id,
                    prior_turn_id=None,
                    source_claim_id=current_claim.claim_id,
                    prior_claim_id=None,
                    explanation="No prior verifiable claims on this entity exist in the ledger.",
                    confidence=1.0,
                ),
                conflict_scope=None,
            )

        candidate_by_id: Dict[str, ClaimItem] = {c.claim_id: c for c in candidates}

        # Build prompt from separated framing and transcript-derived parts
        candidate_lines = [
            f"- Prior Claim [{c.claim_id}] (Turn {c.turn_id}, Speaker {c.speaker_id}): "
            f"{c.canonical_entity} | {c.predicate} | {c.object_value} | time: {c.time_ref_raw} | "
            f"stance: {getattr(c.polarity, 'value', c.polarity)}"
            for c in candidates
        ]
        candidates_str = "\n".join(candidate_lines)
        current_claim_str = (
            f"{current_claim.canonical_entity} | {current_claim.predicate} | {current_claim.object_value} | "
            f"time: {current_claim.time_ref_raw} | "
            f"stance: {getattr(current_claim.polarity, 'value', current_claim.polarity)}"
        )

        framing_template = (
            f"Target Claim [{current_claim.claim_id}] (Turn {current_claim.turn_id}, Speaker {current_claim.speaker_id}):\n"
            "{claim_body}\n\n"
            "Prior Claims in Ledger:\n{prior_claims_body}\n\n"
            "Evaluate consistency between the target claim and the prior claims. "
            "Return CONSISTENT, DIRECT_CONTRADICTION, TIMELINE_SHIFT, or UNVERIFIABLE. "
            "Provide the exact prior_turn_id and prior_claim_id referenced."
        )

        prompt_framing = framing_template.format(
            claim_body="<claim>",
            prior_claims_body="<prior_claims>",
        )
        prompt = framing_template.format(
            claim_body=current_claim_str,
            prior_claims_body=candidates_str,
        )

        system_instruction = (
            "You are a strict logical consistency verification auditor. Compare factual propositions. "
            "Identify direct contradictions or temporal shifts."
        )

        assert_firewall_compliance(prompt_framing)
        assert_firewall_compliance(system_instruction)

        try:
            resp = self._llm.generate_structured(
                prompt=prompt,
                response_model=ConsistencyVerificationResponse,
                system_prompt=system_instruction,
            )

            # CRITICAL REF VALIDATION
            invalid_reason: Optional[str] = None
            if resp.source_turn_id != current_claim.turn_id:
                invalid_reason = (
                    f"source_turn_id {resp.source_turn_id} does not match current_claim.turn_id {current_claim.turn_id}."
                )
            elif resp.verdict in (VerdictStatus.DIRECT_CONTRADICTION, VerdictStatus.TIMELINE_SHIFT) and (
                resp.prior_turn_id is None or resp.prior_claim_id is None
            ):
                invalid_reason = (
                    f"Verdict {resp.verdict.value} requires both prior_turn_id and prior_claim_id."
                )
            elif resp.prior_turn_id is not None and resp.prior_turn_id not in self._claims_by_turn:
                invalid_reason = f"Referenced prior turn {resp.prior_turn_id} does not exist in ledger."
            elif resp.prior_turn_id is not None and resp.prior_turn_id >= current_claim.turn_id:
                invalid_reason = (
                    f"Referenced prior_turn_id {resp.prior_turn_id} is not strictly before current_claim.turn_id {current_claim.turn_id}."
                )
            elif resp.prior_claim_id is not None and resp.prior_claim_id not in candidate_by_id:
                invalid_reason = (
                    f"Referenced prior_claim_id {resp.prior_claim_id} was not in the candidates sent to LLM."
                )
            elif (
                resp.prior_claim_id is not None
                and resp.prior_turn_id is not None
                and candidate_by_id[resp.prior_claim_id].turn_id != resp.prior_turn_id
            ):
                invalid_reason = (
                    f"Referenced prior_claim_id {resp.prior_claim_id} does not belong to prior_turn_id {resp.prior_turn_id}."
                )

            if invalid_reason is not None:
                logger.warning(
                    "LLM produced invalid reference (%s). Rejecting as REJECTED_INVALID_REF.",
                    invalid_reason,
                )
                return VerificationResult(
                    response=ConsistencyVerificationResponse(
                        verdict=VerdictStatus.REJECTED_INVALID_REF,
                        source_turn_id=current_claim.turn_id,
                        prior_turn_id=resp.prior_turn_id,
                        source_claim_id=current_claim.claim_id,
                        prior_claim_id=resp.prior_claim_id,
                        explanation=invalid_reason,
                        confidence=1.0,
                    ),
                    conflict_scope=None,
                )

            conflict_scope: Optional[str] = None
            if resp.prior_claim_id is not None and resp.prior_claim_id in candidate_by_id:
                prior_claim = candidate_by_id[resp.prior_claim_id]
                conflict_scope = (
                    "self" if prior_claim.speaker_id == current_claim.speaker_id else "cross_speaker"
                )

            return VerificationResult(response=resp, conflict_scope=conflict_scope)

        except Exception as exc:
            self.stats["llm_errors"] += 1
            logger.warning("Step B consistency verification error: %s", exc)
            return VerificationResult(
                response=ConsistencyVerificationResponse(
                    verdict=VerdictStatus.LLM_ERROR,
                    source_turn_id=current_claim.turn_id,
                    prior_turn_id=None,
                    source_claim_id=current_claim.claim_id,
                    prior_claim_id=None,
                    explanation=f"Verification failed due to backend error: {exc}",
                    confidence=0.0,
                ),
                conflict_scope=None,
            )

    def merge_speaker_id(self, old_id: str, new_id: str) -> None:
        """Idempotently reassign all claims belonging to old_id into new_id.

        Args:
            old_id: Source speaker identifier to merge from.
            new_id: Target speaker identifier to merge into.
        """
        if old_id == new_id or old_id not in self._claims_by_speaker:
            return

        old_claims = self._claims_by_speaker.pop(old_id, [])
        if new_id not in self._claims_by_speaker:
            self._claims_by_speaker[new_id] = []

        for claim in old_claims:
            claim.speaker_id = new_id
            self._claims_by_speaker[new_id].append(claim)

        logger.info(
            "Idempotently merged %d claims from %s into %s",
            len(old_claims),
            old_id,
            new_id,
        )
