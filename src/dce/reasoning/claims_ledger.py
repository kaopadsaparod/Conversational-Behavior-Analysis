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
    ClaimExtractionResponse,
    ClaimItem,
    ConsistencyVerificationResponse,
    TurnTranscript,
    VerdictStatus,
)

logger = logging.getLogger(__name__)

PROHIBITED_FIREWALL_TERMS: Set[str] = {
    "jitter",
    "shimmer",
    "hnr",
    "f0",
    "semitone",
    "semitones",
    "latency",
    "blink",
    "z_score",
    "mahalanobis",
    "mahalanobis_distance",
    "d_sq",
    "chi2_tail_score",
    "vocal_tension",
    "cognitive_load_anomaly",
    "articulation_rate",
}

_PRONOUN_CONTRACTIONS: Set[str] = {"i", "i'm", "i've", "i'll", "i'd"}

_COMMON_SINGLE_WORD_FILLERS: Set[str] = {
    "yeah", "yes", "no", "nope", "nah", "okay", "ok", "right", "sure", "fine",
    "well", "uh", "um", "hmm", "mhm", "hello", "hi", "hey", "bye", "thanks",
    "thank", "please", "sorry", "what", "why", "how", "who", "where", "when",
    "really", "great", "good", "cool", "nice", "alright", "true", "false",
    "maybe", "perhaps", "definitely", "absolutely", "exactly", "indeed",
    "agreed", "wait", "stop", "go", "never", "always", "nothing", "something",
    "everything", "negative", "affirmative",
}


def assert_firewall_compliance(text: str) -> None:
    """Verify that text contains zero physical signal names or acoustic metric terms.

    Matches both standalone terms and underscore-delimited metric identifiers
    (e.g. 'jitter_local_pct', 'f0_semitone_sd', 'z_score', 'semitones').

    Args:
        text: Prompt text destined for LLM ingestion.

    Raises:
        ValueError: If any prohibited physical signal name is detected.
    """
    lower_text = text.lower()
    for term in PROHIBITED_FIREWALL_TERMS:
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if re.search(pattern, lower_text):
            raise ValueError(
                f"Neuro-Symbolic Firewall violation: LLM prompt contains prohibited physical signal term: '{term}'"
            )


def should_skip_extraction(
    transcript: TurnTranscript,
    prior_transcript: Optional[TurnTranscript] = None,
) -> bool:
    """Evaluate context-aware question gate for conversational utterances.

    Short utterances (< 4 words without digits or proper nouns) are skipped ONLY
    when the preceding turn was NOT a question. If the preceding turn was a question,
    short responses (e.g. 'No.', 'Never.', 'I didn't.') MUST NOT be skipped.

    Args:
        transcript: Current conversational turn transcript.
        prior_transcript: Preceding conversational turn transcript, if available.

    Returns:
        True if turn should be skipped from LLM claim extraction, False otherwise.
    """
    clean_text = transcript.text.strip()
    words = clean_text.split()

    # If prior turn was a question, never skip
    if prior_transcript is not None and prior_transcript.is_question:
        return False

    # Check if utterance is short (< 4 words)
    if len(words) < 4:
        # Check if contains numbers/digits
        has_digits = bool(re.search(r"\d", clean_text))
        if has_digits:
            return False

        # Check if contains capitalized proper nouns
        has_proper_nouns = False
        if len(words) == 1:
            token = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", words[0])
            if (
                token
                and token[0].isupper()
                and token.lower() not in _PRONOUN_CONTRACTIONS
                and token.lower() not in _COMMON_SINGLE_WORD_FILLERS
            ):
                has_proper_nouns = True
        elif len(words) > 1:
            for w in words[1:]:
                token = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", w)
                if token and token[0].isupper() and token.lower() not in _PRONOUN_CONTRACTIONS:
                    has_proper_nouns = True
                    break

        if not has_proper_nouns:
            return True

    return False


class ClaimsLedger:
    """In-memory verifiable proposition store and multi-speaker claim consistency tracker."""

    def __init__(self, llm_adapter: Optional[LLMAdapter] = None) -> None:
        """Initialize ClaimsLedger.

        Args:
            llm_adapter: Configured LLM adapter for structured extraction and verification.
        """
        self._llm: LLMAdapter = llm_adapter if llm_adapter is not None else LLMAdapter()
        self._claims: Dict[str, ClaimItem] = {}
        self._claims_by_turn: Dict[int, List[ClaimItem]] = {}
        self._claims_by_speaker: Dict[str, List[ClaimItem]] = {}
        self._entity_aliases: Dict[str, str] = {}
        self._next_claim_num: int = 1

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
        """Store a structured claim into the ledger."""
        claim.canonical_entity = self.normalize_entity(claim.canonical_entity or claim.raw_subject)
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
        # Build prompt using clean transcripts
        context_str = ""
        framing_context_str = ""
        if context_turns:
            recent_ctx = context_turns[-2:]
            lines = [f"{t.speaker_id}: {t.text}" for t in recent_ctx]
            framing_lines = [f"{t.speaker_id}: <transcript>" for t in recent_ctx]
            context_str = "Prior context:\n" + "\n".join(lines) + "\n\n"
            framing_context_str = "Prior context:\n" + "\n".join(framing_lines) + "\n\n"

        prompt = (
            f"{context_str}"
            f"Current utterance by {transcript.speaker_id} (Turn {transcript.turn_id}):\n"
            f'"{transcript.text}"\n\n'
            f"Extract all explicit, verifiable factual claims as structured propositions. "
            f"Include canonical entities, predicates, object values, time references, and polarity."
        )

        # Audit non-transcript prompt framing and system instruction separately from spoken text
        # so legitimate English words (e.g. 'blink', 'latency') in speech do not crash extraction.
        prompt_framing = (
            f"{framing_context_str}"
            f"Current utterance by {transcript.speaker_id} (Turn {transcript.turn_id}):\n"
            f'"<transcript>"\n\n'
            f"Extract all explicit, verifiable factual claims as structured propositions. "
            f"Include canonical entities, predicates, object values, time references, and polarity."
        )

        system_instruction = (
            "You are a factual claim extraction system. Extract structured subject-predicate-object "
            "propositions. Do not include opinions or filler words."
        )

        # Enforce Neuro-Symbolic Firewall on prompt framing and system instructions
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
            logger.warning("Step A claim extraction encountered error: %s", exc)

        return extracted_claims

    def verify_consistency(
        self,
        current_claim: ClaimItem,
        candidate_prior_claims: Optional[List[ClaimItem]] = None,
    ) -> ConsistencyVerificationResponse:
        """Execute Step B: Verify consistency of a current claim against prior ledger claims.

        Validates referenced turn existence, forcing REJECTED_INVALID_REF if missing.

        Args:
            current_claim: Newly extracted claim to verify.
            candidate_prior_claims: Optional subset of prior claims to compare against.

        Returns:
            ConsistencyVerificationResponse instance.
        """
        # Find candidates if not provided
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
        else:
            candidates = candidate_prior_claims

        if not candidates:
            return ConsistencyVerificationResponse(
                verdict=VerdictStatus.UNVERIFIABLE,
                source_turn_id=current_claim.turn_id,
                prior_turn_id=None,
                source_claim_id=current_claim.claim_id,
                prior_claim_id=None,
                explanation="No prior verifiable claims on this entity exist in the ledger.",
                confidence=1.0,
            )

        # Build comparison prompt
        candidate_lines = [
            f"- Prior Claim [{c.claim_id}] (Turn {c.turn_id}, Speaker {c.speaker_id}): "
            f"{c.canonical_entity} | {c.predicate} | {c.object_value} | time: {c.time_ref_raw} | polarity: {c.polarity}"
            for c in candidates
        ]
        candidates_str = "\n".join(candidate_lines)

        prompt = (
            f"Target Claim [{current_claim.claim_id}] (Turn {current_claim.turn_id}, Speaker {current_claim.speaker_id}):\n"
            f"{current_claim.canonical_entity} | {current_claim.predicate} | {current_claim.object_value} | "
            f"time: {current_claim.time_ref_raw} | polarity: {current_claim.polarity}\n\n"
            f"Prior Claims in Ledger:\n{candidates_str}\n\n"
            f"Evaluate consistency between the target claim and the prior claims. "
            f"Return CONSISTENT, DIRECT_CONTRADICTION, TIMELINE_SHIFT, or UNVERIFIABLE. "
            f"Provide the exact prior_turn_id and prior_claim_id referenced."
        )

        prompt_framing = (
            f"Target Claim [{current_claim.claim_id}] (Turn {current_claim.turn_id}, Speaker {current_claim.speaker_id}):\n"
            f"<claim>\n\n"
            f"Prior Claims in Ledger:\n<prior_claims>\n\n"
            f"Evaluate consistency between the target claim and the prior claims. "
            f"Return CONSISTENT, DIRECT_CONTRADICTION, TIMELINE_SHIFT, or UNVERIFIABLE. "
            f"Provide the exact prior_turn_id and prior_claim_id referenced."
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
            # Validate source_turn_id, prior_turn_id, and prior_claim_id; fail safe to REJECTED_INVALID_REF
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
            elif resp.prior_claim_id is not None and resp.prior_claim_id not in self._claims:
                invalid_reason = f"Referenced prior_claim_id {resp.prior_claim_id} does not exist in ledger."
            elif (
                resp.prior_claim_id is not None
                and resp.prior_turn_id is not None
                and self._claims[resp.prior_claim_id].turn_id != resp.prior_turn_id
            ):
                invalid_reason = (
                    f"Referenced prior_claim_id {resp.prior_claim_id} does not belong to prior_turn_id {resp.prior_turn_id}."
                )

            if invalid_reason is not None:
                logger.warning(
                    "LLM produced invalid reference (%s). Rejecting as REJECTED_INVALID_REF.",
                    invalid_reason,
                )
                return ConsistencyVerificationResponse(
                    verdict=VerdictStatus.REJECTED_INVALID_REF,
                    source_turn_id=current_claim.turn_id,
                    prior_turn_id=resp.prior_turn_id,
                    source_claim_id=current_claim.claim_id,
                    prior_claim_id=resp.prior_claim_id,
                    explanation=invalid_reason,
                    confidence=1.0,
                )

            return resp

        except Exception as exc:
            logger.warning("Step B consistency verification error: %s", exc)
            return ConsistencyVerificationResponse(
                verdict=VerdictStatus.UNVERIFIABLE,
                source_turn_id=current_claim.turn_id,
                prior_turn_id=None,
                source_claim_id=current_claim.claim_id,
                prior_claim_id=None,
                explanation=f"Verification failed due to backend error: {exc}",
                confidence=0.0,
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
