"""Discourse Consistency Engine - Claims Ledger and Neuro-Symbolic Firewall.

Enforces a strict Neuro-Symbolic Firewall prohibiting physical signal names or numbers
in LLM prompts, context-aware question gating, structured Step A claim extraction,
Step B consistency verification with reference validation, and idempotent speaker merging.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

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

_PHONETIC_FRAGILE_PAIRS: Set[Tuple[str, str]] = {
    ("13", "30"), ("thirteen", "thirty"),
    ("14", "40"), ("fourteen", "forty"),
    ("15", "50"), ("fifteen", "fifty"),
    ("16", "60"), ("sixteen", "sixty"),
    ("17", "70"), ("seventeen", "seventy"),
    ("18", "80"), ("eighteen", "eighty"),
    ("19", "90"), ("nineteen", "ninety"),
    ("can", "can't"), ("can", "cannot"), ("won't", "want"),
    ("did", "didn't"), ("do", "don't"), ("is", "isn't"),
    ("was", "wasn't"), ("were", "weren't"), ("have", "haven't"),
    ("has", "hasn't"), ("had", "hadn't"), ("would", "wouldn't"),
    ("could", "couldn't"), ("should", "shouldn't"),
    ("leave", "live"), ("left", "lived"),
    ("two", "to"), ("too", "to"), ("no", "know"),
    ("hear", "here"), ("their", "there"), ("sent", "cent"),
    ("wait", "weight"), ("week", "weak"), ("right", "write"),
    ("buy", "by"), ("night", "knight"), ("eight", "ate"),
    ("for", "four"), ("one", "won"), ("wear", "where"),
    ("knew", "new"), ("weather", "whether"), ("plane", "plain"),
    ("principal", "principle"), ("bare", "bear"), ("break", "brake"),
}
_PHONETIC_FRAGILE_SET: Set[frozenset] = {frozenset(pair) for pair in _PHONETIC_FRAGILE_PAIRS}


_PHONETIC_CANONICAL_MAP: Dict[str, str] = {
    "1": "one",
    "2": "two",
    "4": "four",
    "8": "eight",
    "13": "thirteen",
    "30": "thirty",
    "14": "fourteen",
    "40": "forty",
    "15": "fifteen",
    "50": "fifty",
    "16": "sixteen",
    "60": "sixty",
    "17": "seventeen",
    "70": "seventy",
    "18": "eighteen",
    "80": "eighty",
    "19": "nineteen",
    "90": "ninety",
    "cannot": "can't",
    "cant": "can't",
    "wont": "won't",
    "didnt": "didn't",
    "dont": "don't",
    "isnt": "isn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "wouldnt": "wouldn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
}


def _normalize_phonetic_tokens(text: Optional[str]) -> List[str]:
    """Tokenize and normalize numbers/contractions for phonetic-fragile matching."""
    if not text:
        return []
    lowered = text.lower()
    lowered = re.sub(r"\bcan\s+not\b", "cannot", lowered)
    lowered = re.sub(r"\bwill\s+not\b", "wont", lowered)
    lowered = re.sub(r"\bdid\s+not\b", "didnt", lowered)
    lowered = re.sub(r"\bdo\s+not\b", "dont", lowered)
    raw_tokens = re.findall(r"[a-z0-9']+", lowered)
    return [_PHONETIC_CANONICAL_MAP.get(tok, tok) for tok in raw_tokens]


def is_phonetic_fragile_difference(claim_a: ClaimItem, claim_b: ClaimItem) -> bool:
    """Return True if two claims differ only by known phonetic-fragile / ASR-confusable tokens."""
    tokens_a = _normalize_phonetic_tokens(claim_a.object_value) + _normalize_phonetic_tokens(claim_a.time_ref_raw)
    tokens_b = _normalize_phonetic_tokens(claim_b.object_value) + _normalize_phonetic_tokens(claim_b.time_ref_raw)

    if not tokens_a or not tokens_b or tokens_a == tokens_b:
        return False

    # Compare token-by-token when aligned in length, or single-token difference in sets
    if len(tokens_a) == len(tokens_b):
        diffs = [(a, b) for a, b in zip(tokens_a, tokens_b) if a != b]
        return bool(diffs) and all(frozenset(pair) in _PHONETIC_FRAGILE_SET for pair in diffs)

    set_a, set_b = set(tokens_a), set(tokens_b)
    only_a = set_a - set_b
    only_b = set_b - set_a
    if len(only_a) == 1 and len(only_b) == 1:
        return frozenset((next(iter(only_a)), next(iter(only_b)))) in _PHONETIC_FRAGILE_SET
    return False


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


class NLIPreFilter:
    """Utterance-structured NLI pre-filter (e.g., DeBERTa-v3-xsmall) before LLM Step B.

    Lazily imports transformers/torch when loading a HuggingFace NLI model and
    falls back to a deterministic lexical/polarity heuristic when unavailable.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        scorer_fn: Optional[Callable[[str, str], float]] = None,
        use_hf_model: bool = True,
    ) -> None:
        self.model_name = model_name
        self._scorer_fn = scorer_fn
        self._use_hf_model = use_hf_model
        self._tokenizer: object = None
        self._model: object = None
        self._hf_load_attempted: bool = False
        self._contradiction_idx: int = 0
        self._entailment_idx: int = 1
        self._device: str = "cpu"

    @staticmethod
    def format_utterance_claim(claim: ClaimItem) -> str:
        """Format a ClaimItem as clean raw proposition text (no artificial speaker wrapper)."""
        if claim.predicate == "stated" and claim.canonical_entity == "dialogue_fact":
            return claim.object_value.strip()
        stance = getattr(claim.polarity, "value", claim.polarity)
        time_part = f" at {claim.time_ref_raw}" if claim.time_ref_raw else ""
        loc_part = f" in {claim.location_ref}" if claim.location_ref else ""
        neg_prefix = "not " if str(stance).lower() == "negative" else ""
        return (
            f"{claim.canonical_entity} {neg_prefix}{claim.predicate} "
            f"{claim.object_value}{time_part}{loc_part}."
        ).strip()

    @staticmethod
    def _heuristic_contradiction_score(premise: str, hypothesis: str) -> float:
        """Lightweight fallback NLI contradiction scorer when transformers is unavailable."""
        def _strip_boilerplate(text: str) -> str:
            low = text.strip().lower()
            if " stated that " in low:
                return low.split(" stated that ", 1)[1]
            return low

        p_low = _strip_boilerplate(premise)
        h_low = _strip_boilerplate(hypothesis)
        p_tokens = set(re.findall(r"[a-z0-9']+", p_low))
        h_tokens = set(re.findall(r"[a-z0-9']+", h_low))

        neg_markers = {"not", "no", "never", "neither", "none", "didn't", "didnt", "wasn't", "isnt", "negative"}
        p_neg = bool(p_tokens & neg_markers)
        h_neg = bool(h_tokens & neg_markers)

        p_content = p_tokens - neg_markers
        h_content = h_tokens - neg_markers
        content_overlap = len(p_content & h_content) / max(len(p_content | h_content), 1)

        if p_neg != h_neg and content_overlap >= 0.30:
            return 0.85

        # Differing numbers or times (handles attached units like '9am' vs '10am')
        p_nums = set(re.findall(r"\d+(?::\d+)?", p_low))
        h_nums = set(re.findall(r"\d+(?::\d+)?", h_low))
        if p_nums and h_nums and p_nums != h_nums and content_overlap >= 0.25:
            return 0.80

        if content_overlap >= 0.85 and p_neg == h_neg:
            return 0.05
        if p_content != h_content and content_overlap >= 0.45:
            return 0.60
        return 0.15

    def _ensure_hf_model(self) -> bool:
        """Lazily load HuggingFace NLI model and tokenizer; return True if ready."""
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._hf_load_attempted or not self._use_hf_model:
            return False
        self._hf_load_attempted = True
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            label2id = getattr(self._model.config, "label2id", {}) or {}
            for lbl, idx in label2id.items():
                low_lbl = str(lbl).lower()
                if "contradict" in low_lbl:
                    self._contradiction_idx = int(idx)
                elif "entail" in low_lbl:
                    self._entailment_idx = int(idx)
            if not label2id:
                id2label = getattr(self._model.config, "id2label", {}) or {}
                for idx, lbl in id2label.items():
                    low_lbl = str(lbl).lower()
                    if "contradict" in low_lbl:
                        self._contradiction_idx = int(idx)
                    elif "entail" in low_lbl:
                        self._entailment_idx = int(idx)
            return True
        except Exception as exc:
            logger.warning("NLI model '%s' unavailable (%s); using heuristic fallback.", self.model_name, exc)
            self._tokenizer = None
            self._model = None
            return False

    @staticmethod
    def split_clauses(text: str) -> List[str]:
        """Split a multi-sentence turn into individual clauses/sentences for fine-grained NLI."""
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+|;\s+", cleaned)
        clauses = [p.strip() for p in parts if len(p.strip()) >= 3]
        return clauses if clauses else [cleaned]

    def _predict_single_hf(self, premise: str, hypothesis: str) -> Tuple[float, Optional[float]]:
        """Run HF cross-encoder on a single (premise, hypothesis) pair."""
        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            c_prob = float(probs[self._contradiction_idx].item())
            e_prob = (
                float(probs[self._entailment_idx].item())
                if self._entailment_idx < len(probs)
                else None
            )
            return c_prob, e_prob

    def predict_scores(self, premise: str, hypothesis: str) -> Tuple[float, Optional[float]]:
        """Return (contradiction_prob, entailment_prob_or_none) for (premise, hypothesis).

        Splits multi-sentence turns into clauses (when not using a custom scorer_fn)
        and returns the strongest clause-level contradiction pair.
        """
        if self._scorer_fn is not None:
            score = float(self._scorer_fn(premise, hypothesis))
            return max(0.0, min(1.0, score)), None

        p_clauses = self.split_clauses(premise) or [premise]
        h_clauses = self.split_clauses(hypothesis) or [hypothesis]
        pairs = [(pc, hc) for pc in p_clauses for hc in h_clauses][:25]

        if self._ensure_hf_model():
            try:
                if len(pairs) == 1:
                    return self._predict_single_hf(pairs[0][0], pairs[0][1])

                import torch

                p_list = [p for p, _ in pairs]
                h_list = [h for _, h in pairs]
                inputs = self._tokenizer(
                    p_list,
                    h_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    c_probs = probs[:, self._contradiction_idx]
                    best_idx = int(torch.argmax(c_probs).item())
                    c_prob = float(c_probs[best_idx].item())
                    e_prob = (
                        float(probs[best_idx, self._entailment_idx].item())
                        if self._entailment_idx < probs.shape[-1]
                        else None
                    )
                    return c_prob, e_prob
            except Exception as exc:
                logger.warning("NLI inference failed (%s); falling back to heuristic.", exc)

        best_c = 0.0
        for pc, hc in pairs:
            c_val = self._heuristic_contradiction_score(pc, hc)
            if c_val > best_c:
                best_c = c_val
        return best_c, None

    def predict_contradiction_score(self, premise: str, hypothesis: str) -> float:
        """Predict contradiction probability in [0.0, 1.0] for (premise, hypothesis)."""
        c_prob, _ = self.predict_scores(premise, hypothesis)
        return c_prob

    def filter_candidates(
        self,
        current_claim: ClaimItem,
        candidates: List[ClaimItem],
        threshold: float = 0.10,
        entailment_skip_threshold: float = 0.85,
        same_speaker_only: bool = True,
    ) -> List[ClaimItem]:
        """Return prior claim candidates for Step B, preserving temporal claims and low-entailment pairs."""
        if not candidates:
            return []
        hypothesis = self.format_utterance_claim(current_claim)
        curr_has_time = bool(current_claim.time_ref_raw or current_claim.time_ref_normalized)
        scored: List[Tuple[float, ClaimItem]] = []
        for cand in candidates:
            if same_speaker_only and cand.speaker_id != current_claim.speaker_id:
                continue
            cand_has_time = bool(cand.time_ref_raw or cand.time_ref_normalized)
            premise = self.format_utterance_claim(cand)
            c_score, e_score = self.predict_scores(premise, hypothesis)
            # Never drop temporal pairs prematurely (NLI often scores indirect TIMELINE_SHIFT as neutral)
            if curr_has_time or cand_has_time:
                scored.append((max(c_score, threshold), cand))
            elif c_score >= threshold or (e_score is not None and e_score <= entailment_skip_threshold):
                scored.append((c_score, cand))
        scored.sort(key=lambda item: (item[0], item[1].turn_id), reverse=True)
        return [item[1] for item in scored]


class ClaimsLedger:
    """In-memory verifiable proposition store and multi-speaker claim consistency tracker."""

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        retrieval_top_k: Optional[int] = None,
        config: Optional[ReasoningConfig] = None,
        nli_prefilter: Optional[NLIPreFilter] = None,
    ) -> None:
        """Initialize ClaimsLedger.

        Args:
            llm_adapter: Configured LLM adapter for structured extraction and verification.
            retrieval_top_k: Maximum number of prior claims retrieved per verification.
            config: Optional ReasoningConfig providing retrieval_top_k and NLI settings.
            nli_prefilter: Optional NLIPreFilter instance for pre-filtering before Step B.
        """
        self._config: ReasoningConfig = config if config is not None else ReasoningConfig()
        self._llm: LLMAdapter = llm_adapter if llm_adapter is not None else LLMAdapter()
        self._retrieval_top_k: int = (
            retrieval_top_k
            if retrieval_top_k is not None
            else self._config.retrieval_top_k
        )
        if nli_prefilter is not None:
            self._nli_prefilter: Optional[NLIPreFilter] = nli_prefilter
        elif self._config.nli_prefilter_enabled:
            self._nli_prefilter = NLIPreFilter(model_name=self._config.nli_model_name)
        else:
            self._nli_prefilter = None

        self._claims: Dict[str, ClaimItem] = {}
        self._claims_by_turn: Dict[int, List[ClaimItem]] = {}
        self._claims_by_speaker: Dict[str, List[ClaimItem]] = {}
        self._entity_aliases: Dict[str, str] = {}
        self._next_claim_num: int = 1
        self.stats: Dict[str, int] = {
            "llm_errors": 0,
            "nli_prefilter_skipped": 0,
            "nli_prefilter_passed": 0,
        }

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
            "Include canonical entities, predicates, object values, time references, and affirmative or negative status. "
            "When the prior turn is a negative question (e.g., 'Didn't you...?') and the current utterance is a short "
            "'Yes' or 'No' reply, resolve affirmation or negation using the full clause and agreement context rather than "
            "blindly assuming 'Yes' means an affirmative event."
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
            "propositions. Do not include opinions or filler words. Resolve L2 yes/no answers to negative questions "
            "using prior question agreement context."
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

        if self._nli_prefilter is not None:
            filtered_candidates = self._nli_prefilter.filter_candidates(
                current_claim,
                candidates,
                threshold=self._config.nli_contradiction_threshold,
            )
            if not filtered_candidates:
                self.stats["nli_prefilter_skipped"] += 1
                top_prior = candidates[0]
                return VerificationResult(
                    response=ConsistencyVerificationResponse(
                        verdict=VerdictStatus.CONSISTENT,
                        source_turn_id=current_claim.turn_id,
                        prior_turn_id=top_prior.turn_id,
                        source_claim_id=current_claim.claim_id,
                        prior_claim_id=top_prior.claim_id,
                        explanation="NLI pre-filter found no contradiction candidates; LLM Step B skipped.",
                        confidence=0.9,
                    ),
                    conflict_scope=(
                        "self" if top_prior.speaker_id == current_claim.speaker_id else "cross_speaker"
                    ),
                )
            self.stats["nli_prefilter_passed"] += 1
            candidates = filtered_candidates

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
            "Evaluate consistency between the target claim and the prior claims.\n"
            "Distinguish carefully:\n"
            "- CONSISTENT: compatible facts, or a natural progression where an explicit new event/time explains a state change.\n"
            "- DIRECT_CONTRADICTION: incompatible factual claims about the same static fact, attribute, or current status.\n"
            "- TIMELINE_SHIFT: conflicting event times/sequences or an unmotivated reversal of prior status without a coherent timeline update.\n"
            "- UNVERIFIABLE: insufficient factual overlap to judge.\n"
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
            "Identify direct contradictions and distinguish genuine timeline updates from timeline contradictions/shifts."
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
                if resp.verdict in (
                    VerdictStatus.DIRECT_CONTRADICTION,
                    VerdictStatus.TIMELINE_SHIFT,
                ) and is_phonetic_fragile_difference(current_claim, prior_claim):
                    resp = resp.model_copy(
                        update={
                            "verdict": VerdictStatus.NEEDS_AUDIO_REVIEW,
                            "explanation": (
                                f"{resp.explanation} "
                                "[Audio review recommended: difference may stem from phonetic-fragile ASR confusion]"
                            ).strip(),
                        }
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
