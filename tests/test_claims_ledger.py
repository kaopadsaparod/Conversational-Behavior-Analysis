"""Unit tests for Discourse Consistency Engine reasoning ledger, firewall, and schema sanitizer.

Tests:
1. Neuro-Symbolic Firewall: inspect Step A and Step B prompts to verify ZERO physiological signal
   terms/numbers appear, and assert_firewall_compliance raises ValueError on violations.
2. Context-aware question gate: short answers (< 4 words) are NOT skipped if prior turn was a question.
3. Schema sanitizer: 3-step sanitize_gemini_schema resolving $defs, collapsing anyOf with null
   into nullable, stripping metadata while strictly preserving user properties named 'title'.
4. Invalid reference handling: ConsistencyVerificationResponse referencing a non-existent turn_id
   has its status converted to VerdictStatus.REJECTED_INVALID_REF.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from dce.reasoning.claims_ledger import (
    PROHIBITED_FIREWALL_TERMS,
    ClaimsLedger,
    assert_firewall_compliance,
    should_skip_extraction,
)
from dce.reasoning.llm_adapter import LLMAdapter, sanitize_gemini_schema
from dce.schemas import (
    FEATURE_GROUPS,
    ClaimExtractionResponse,
    ClaimItem,
    ConsistencyVerificationResponse,
    TurnTranscript,
    VerdictStatus,
    VerificationResult,
)


# ============================================================================
# 1. Neuro-Symbolic Firewall Tests
# ============================================================================


def test_firewall_rejects_prohibited_terms() -> None:
    """Verify assert_firewall_compliance detects and raises ValueError for any physiological term."""
    # Test all prohibited terms
    for term in PROHIBITED_FIREWALL_TERMS:
        bad_prompt = f"The speaker showed elevated {term} during the turn."
        with pytest.raises(ValueError, match="Neuro-Symbolic Firewall violation"):
            assert_firewall_compliance(bad_prompt)

    # Clean text should pass without exception
    clean_prompt = "The speaker stated: 'I parked my vehicle in the garage at 5 PM yesterday.'"
    assert_firewall_compliance(clean_prompt)


def test_firewall_tokenizes_snake_camel_plurals_and_allows_spoken_text() -> None:
    """Verify firewall catches snake_case, camelCase, and plurals in framing, while allowing spoken words."""
    bad_framings = [
        "Check jitter_local_pct for this turn.",
        "Check f0_semitone_sd for this turn.",
        "Check blink_rate_per_min for this turn.",
        "Check response_latency_open for this turn.",
        "Check semitones for this turn.",
        "Check jitterLocalPct and cognitiveLoad.",
        "Check vocalTension and articulation_rate.",
        "Check zScore, chi2, anomalies, and latencies.",
    ] + [f"Feature {feat} value" for feat in FEATURE_GROUPS]

    for bad in bad_framings:
        with pytest.raises(ValueError, match="Neuro-Symbolic Firewall violation"):
            assert_firewall_compliance(bad)

    # Spoken text containing 'latency', 'blinked', 'f0' MUST succeed in extract_claims
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ClaimExtractionResponse(claims=[])
    ledger = ClaimsLedger(llm_adapter=mock_llm)
    spoken = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="the latency was awful and I blinked, my f0 key",
        start_ns=100_000,
        end_ns=900_000,
        is_question=False,
    )
    claims = ledger.extract_claims(spoken)
    assert claims == []
    mock_llm.generate_structured.assert_called_once()


def test_firewall_prompt_inspection_step_a_and_step_b() -> None:
    """Inspect Step A and Step B generated prompts to guarantee ZERO physical signal metrics appear."""
    mock_llm = MagicMock(spec=LLMAdapter)
    captured_prompts: List[str] = []

    def mock_generate_structured(prompt: str, response_model: type, system_prompt: str):
        captured_prompts.append(prompt)
        if response_model == ClaimExtractionResponse:
            return ClaimExtractionResponse(
                claims=[
                    ClaimItem(
                        claim_id="CLM_MOCK_01",
                        turn_id=1,
                        speaker_id="SPEAKER_00",
                        raw_subject="Alice",
                        canonical_entity="Alice",
                        predicate="visited",
                        object_value="the library",
                        time_ref_raw="yesterday",
                        polarity="affirmative",
                        confidence=1.0,
                    )
                ]
            )
        elif response_model == ConsistencyVerificationResponse:
            return ConsistencyVerificationResponse(
                verdict=VerdictStatus.CONSISTENT,
                source_turn_id=2,
                prior_turn_id=1,
                source_claim_id="CLM_0002_01",
                prior_claim_id="CLM_0001_01",
                explanation="Statements align logically.",
                confidence=1.0,
            )
        raise ValueError(f"Unexpected model: {response_model}")

    mock_llm.generate_structured.side_effect = mock_generate_structured

    ledger = ClaimsLedger(llm_adapter=mock_llm)

    # Step A execution
    t1 = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="I went to the library yesterday afternoon.",
        start_ns=100_000,
        end_ns=900_000,
        is_question=False,
    )
    claims1 = ledger.extract_claims(t1)
    assert len(claims1) == 1

    # Verify Step A prompt
    assert len(captured_prompts) == 1
    prompt_step_a = captured_prompts[0]
    for term in PROHIBITED_FIREWALL_TERMS:
        assert term not in prompt_step_a.lower(), f"Step A prompt leaked physical term: {term}"

    # Step B execution
    t2 = TurnTranscript(
        turn_id=2,
        speaker_id="SPEAKER_00",
        text="Alice was at the library yesterday.",
        start_ns=1_000_000,
        end_ns=1_900_000,
        is_question=False,
    )
    claims2 = ledger.extract_claims(t2)
    assert len(claims2) == 1

    prompt_step_a_2 = captured_prompts[1]
    for term in PROHIBITED_FIREWALL_TERMS:
        assert term not in prompt_step_a_2.lower()

    # Verify consistency
    res_b = ledger.verify_consistency(claims2[0])
    assert len(captured_prompts) == 3
    prompt_step_b = captured_prompts[2]

    # Verify Step B prompt
    for term in PROHIBITED_FIREWALL_TERMS:
        assert term not in prompt_step_b.lower(), f"Step B prompt leaked physical term: {term}"


# ============================================================================
# 2. Context-Aware Question Gate Tests
# ============================================================================


def test_question_gate_short_answers_not_skipped_after_question() -> None:
    """Verify short utterances (< 4 words) are NOT skipped if prior turn was a question."""
    prior_question = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="Did you see the vehicle leave the parking lot?",
        start_ns=100_000,
        end_ns=900_000,
        is_question=True,
    )

    # Single-word or short answers following a question MUST NOT be skipped
    short_answers = [
        "No.",
        "Never.",
        "I didn't.",
        "Yes, absolutely.",
        "Negative.",
    ]

    for ans in short_answers:
        curr = TurnTranscript(
            turn_id=2,
            speaker_id="SPEAKER_01",
            text=ans,
            start_ns=1_000_000,
            end_ns=1_500_000,
            is_question=False,
        )
        assert not should_skip_extraction(curr, prior_question), (
            f"Short answer '{ans}' following question was incorrectly skipped!"
        )


def test_question_gate_skips_short_acknowledgments_without_prior_question() -> None:
    """Verify conversational filler acknowledgments are skipped when prior was NOT a question."""
    prior_statement = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="The weather was quite rainy throughout the entire morning.",
        start_ns=100_000,
        end_ns=900_000,
        is_question=False,
    )

    fillers = [
        "Yeah.",
        "Yep.",
        "Uh-huh.",
        "Mm-hmm.",
        "Mhm.",
        "Okay.",
        "Ok.",
        "Right.",
        "Sure.",
        "I see.",
        "Got it.",
        "Thanks.",
        "Thank you.",
        "Wow.",
        "Oh.",
        "Hmm.",
        "Um.",
        "Uh.",
    ]

    for filler in fillers:
        curr = TurnTranscript(
            turn_id=2,
            speaker_id="SPEAKER_01",
            text=filler,
            start_ns=1_000_000,
            end_ns=1_500_000,
            is_question=False,
        )
        assert should_skip_extraction(curr, prior_statement) is True, (
            f"Filler '{filler}' should have been skipped."
        )

    # Non-lexicon short utterances MUST NOT be skipped even without prior question
    for non_filler in ["No, I'm fine.", "London."]:
        curr = TurnTranscript(
            turn_id=3,
            speaker_id="SPEAKER_01",
            text=non_filler,
            start_ns=1_600_000,
            end_ns=2_000_000,
            is_question=False,
        )
        assert should_skip_extraction(curr, prior_statement) is False, (
            f"Non-filler '{non_filler}' should NOT have been skipped."
        )


def test_question_gate_does_not_skip_utterances_with_numbers_or_proper_nouns() -> None:
    """Verify short utterances containing numbers or proper nouns are retained even without questions."""
    prior_statement = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="We gathered together at the conference hall.",
        start_ns=100_000,
        end_ns=900_000,
        is_question=False,
    )

    # Has numbers
    with_numbers = TurnTranscript(
        turn_id=2,
        speaker_id="SPEAKER_01",
        text="Room 402",
        start_ns=1_000_000,
        end_ns=1_500_000,
        is_question=False,
    )
    assert not should_skip_extraction(with_numbers, prior_statement)

    # Has proper noun (capitalized non-first word)
    with_proper = TurnTranscript(
        turn_id=3,
        speaker_id="SPEAKER_01",
        text="saw Bob",
        start_ns=2_000_000,
        end_ns=2_500_000,
        is_question=False,
    )
    assert not should_skip_extraction(with_proper, prior_statement)


# ============================================================================
# 3. 3-Step Gemini Schema Sanitizer Tests
# ============================================================================


def test_sanitize_gemini_schema_resolves_defs_nullables_and_preserves_title() -> None:
    """Verify 3-step sanitize_gemini_schema:
    Step 1: inlines $ref pointers from $defs
    Step 2: collapses anyOf with null into primary type with nullable=True
    Step 3: strips metadata ($defs, default, additionalProperties, title) while strictly
            preserving user properties named 'title'.
    """

    class SubModel(BaseModel):
        title: str = Field(description="User field named title")
        priority: int = 1

    class RootModel(BaseModel):
        sub_item: Optional[SubModel] = None
        title: str = "Root user title"
        description: Optional[str] = None

    raw_schema = RootModel.model_json_schema()

    # Verify raw schema has $defs, title metadata, and anyOf
    assert "$defs" in raw_schema
    assert "title" in raw_schema  # Root title metadata

    sanitized = sanitize_gemini_schema(raw_schema)

    # 1. Step 1 & 3: $defs must be completely resolved and stripped
    assert "$defs" not in sanitized

    # 2. Step 3: Top-level and sub-level title METADATA must be stripped
    # But user fields named "title" in properties MUST BE PRESERVED!
    assert "properties" in sanitized
    assert "title" in sanitized["properties"], "User property named 'title' was removed!"
    user_title_prop = sanitized["properties"]["title"]
    assert user_title_prop.get("type") == "string"
    assert "title" not in user_title_prop, "Metadata title was not stripped inside property schema"

    # 3. Step 2: anyOf with null collapsed to nullable=True
    sub_item_schema = sanitized["properties"]["sub_item"]
    assert "anyOf" not in sub_item_schema
    assert sub_item_schema.get("type") == "object"
    assert sub_item_schema.get("nullable") is True

    # User property "title" inside sub_item must also be preserved!
    assert "title" in sub_item_schema["properties"]
    assert "default" not in sub_item_schema["properties"]["priority"]


# ============================================================================
# 4. Invalid Reference Handling Tests
# ============================================================================


def test_verify_consistency_invalid_turn_reference_converted_to_rejected() -> None:
    """Verify LLM response referencing non-existent turn is converted to REJECTED_INVALID_REF."""
    mock_llm = MagicMock(spec=LLMAdapter)

    # Simulate LLM producing DIRECT_CONTRADICTION referencing non-existent prior turn 999
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=999,  # Does not exist in ledger!
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0999_01",
        explanation="Claims directly contradict each other.",
        confidence=0.95,
    )

    ledger = ClaimsLedger(llm_adapter=mock_llm)

    # Add a legitimate prior claim for turn 1
    valid_prior_claim = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="noon",
        polarity="affirmative",
    )
    ledger.add_claim(valid_prior_claim)

    current_claim = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="departed",
        object_value="home",
        time_ref_raw="noon",
        polarity="affirmative",
    )
    ledger.add_claim(current_claim)

    # Execute verification
    result = ledger.verify_consistency(current_claim)

    # Must be forced to REJECTED_INVALID_REF
    assert isinstance(result, VerificationResult)
    assert result.response.verdict == VerdictStatus.REJECTED_INVALID_REF
    assert result.response.prior_turn_id == 999
    assert "does not exist in ledger" in result.response.explanation or "not in" in result.response.explanation


def test_verify_consistency_valid_turn_reference_accepted() -> None:
    """Verify LLM response referencing an existing prior turn is retained as-is."""
    mock_llm = MagicMock(spec=LLMAdapter)

    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=1,  # Exists in ledger!
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0001_01",
        explanation="Claims directly contradict each other.",
        confidence=0.95,
    )

    ledger = ClaimsLedger(llm_adapter=mock_llm)

    valid_prior = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="noon",
        polarity="affirmative",
    )
    ledger.add_claim(valid_prior)

    current_claim = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="departed",
        object_value="home",
        time_ref_raw="noon",
        polarity="affirmative",
    )
    ledger.add_claim(current_claim)

    result = ledger.verify_consistency(current_claim)
    assert isinstance(result, VerificationResult)
    assert result.response.verdict == VerdictStatus.DIRECT_CONTRADICTION
    assert result.response.prior_turn_id == 1
    assert result.conflict_scope == "self"


def test_verify_consistency_all_invalid_ref_cases() -> None:
    """Test each invalid reference rejection condition individually."""
    def _make_ledger(resp: ConsistencyVerificationResponse) -> tuple[ClaimsLedger, ClaimItem]:
        mock_llm = MagicMock(spec=LLMAdapter)
        mock_llm.generate_structured.return_value = resp
        ledger = ClaimsLedger(llm_adapter=mock_llm)
        ledger.add_claim(ClaimItem(
            claim_id="CLM_0001_01", turn_id=1, speaker_id="SPEAKER_00",
            raw_subject="Alice", canonical_entity="alice", predicate="arrived",
            object_value="home", time_ref_raw="noon",
        ))
        ledger.add_claim(ClaimItem(
            claim_id="CLM_0001_02", turn_id=1, speaker_id="SPEAKER_00",
            raw_subject="Bob", canonical_entity="bob", predicate="slept",
            object_value="room", time_ref_raw="midnight",
        ))
        ledger.add_claim(ClaimItem(
            claim_id="CLM_0003_01", turn_id=3, speaker_id="SPEAKER_00",
            raw_subject="Alice", canonical_entity="alice", predicate="stayed",
            object_value="home", time_ref_raw="noon",
        ))
        curr = ClaimItem(
            claim_id="CLM_0002_01", turn_id=2, speaker_id="SPEAKER_00",
            raw_subject="Alice", canonical_entity="alice", predicate="departed",
            object_value="home", time_ref_raw="noon",
        )
        ledger.add_claim(curr)
        return ledger, curr

    # 1. source_turn_id != current_claim.turn_id
    ledger, curr = _make_ledger(ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=99, prior_turn_id=1,
        source_claim_id="CLM_0002_01", prior_claim_id="CLM_0001_01",
        explanation="Mismatch source turn",
    ))
    assert ledger.verify_consistency(curr).response.verdict == VerdictStatus.REJECTED_INVALID_REF

    # 2. prior_turn_id is None for DIRECT_CONTRADICTION / TIMELINE_SHIFT
    for v in (VerdictStatus.DIRECT_CONTRADICTION, VerdictStatus.TIMELINE_SHIFT):
        ledger, curr = _make_ledger(ConsistencyVerificationResponse(
            verdict=v, source_turn_id=2, prior_turn_id=None,
            source_claim_id="CLM_0002_01", prior_claim_id="CLM_0001_01",
            explanation="Missing prior turn",
        ))
        assert ledger.verify_consistency(curr).response.verdict == VerdictStatus.REJECTED_INVALID_REF

    # 3. prior_turn_id >= current_claim.turn_id
    ledger, curr = _make_ledger(ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2, prior_turn_id=3,
        source_claim_id="CLM_0002_01", prior_claim_id="CLM_0003_01",
        explanation="Future turn",
    ))
    assert ledger.verify_consistency(curr).response.verdict == VerdictStatus.REJECTED_INVALID_REF

    # 4. prior_claim_id exists in ledger but NOT in candidates sent to LLM
    ledger, curr = _make_ledger(ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2, prior_turn_id=1,
        source_claim_id="CLM_0002_01", prior_claim_id="CLM_0001_02",
        explanation="Unrelated claim not in candidates",
    ))
    assert ledger.verify_consistency(curr).response.verdict == VerdictStatus.REJECTED_INVALID_REF

    # 5. prior_turn_id does not match referenced claim's turn_id
    ledger, curr = _make_ledger(ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2, prior_turn_id=2,
        source_claim_id="CLM_0002_01", prior_claim_id="CLM_0001_01",
        explanation="Wrong turn_id for claim",
    ))
    assert ledger.verify_consistency(curr).response.verdict == VerdictStatus.REJECTED_INVALID_REF


# ============================================================================
# 5. ClaimsLedger Speaker Merging and LLM Adapter Retry Tests
# ============================================================================


def test_claims_ledger_merge_speaker_id() -> None:
    """Verify merge_speaker_id idempotently reassigns claims to new speaker ID."""
    ledger = ClaimsLedger()

    c1 = ClaimItem(
        claim_id="CLM_01",
        turn_id=1,
        speaker_id="PROV_01",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="noon",
    )
    c2 = ClaimItem(
        claim_id="CLM_02",
        turn_id=2,
        speaker_id="PROV_01",
        raw_subject="Bob",
        canonical_entity="bob",
        predicate="left",
        object_value="office",
        time_ref_raw="1pm",
    )
    ledger.add_claim(c1)
    ledger.add_claim(c2)

    assert len(ledger.claims_by_speaker.get("PROV_01", [])) == 2

    # Execute merge
    ledger.merge_speaker_id("PROV_01", "SPEAKER_00")

    assert len(ledger.claims_by_speaker.get("PROV_01", [])) == 0
    merged_claims = ledger.claims_by_speaker.get("SPEAKER_00", [])
    assert len(merged_claims) == 2
    assert all(c.speaker_id == "SPEAKER_00" for c in merged_claims)

    # Idempotent second call
    ledger.merge_speaker_id("PROV_01", "SPEAKER_00")
    assert len(ledger.claims_by_speaker.get("SPEAKER_00", [])) == 2


def test_verify_consistency_backend_exception_fallback() -> None:
    """Verify LLM exception during verify_consistency falls back to UNVERIFIABLE."""
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.side_effect = RuntimeError("Connection timeout")

    ledger = ClaimsLedger(llm_adapter=mock_llm)
    c1 = ClaimItem(
        claim_id="CLM_ERR_01",
        turn_id=1,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="noon",
    )
    ledger.add_claim(c1)

    c2 = ClaimItem(
        claim_id="CLM_ERR_02",
        turn_id=2,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="departed",
        object_value="home",
        time_ref_raw="noon",
    )

    res = ledger.verify_consistency(c2)
    assert isinstance(res, VerificationResult)
    assert res.response.verdict == VerdictStatus.LLM_ERROR
    assert res.response.confidence == 0.0
    assert "Connection timeout" in res.response.explanation


def test_extract_claims_exception_increments_llm_errors_stat() -> None:
    """Verify Step A exception increments ledger.stats['llm_errors']."""
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.side_effect = RuntimeError("API failure")
    ledger = ClaimsLedger(llm_adapter=mock_llm)

    t1 = TurnTranscript(
        turn_id=1,
        speaker_id="SPEAKER_00",
        text="I arrived at the station at 9 AM.",
        start_ns=100_000,
        end_ns=900_000,
        is_question=False,
    )
    claims = ledger.extract_claims(t1)
    assert claims == []
    assert ledger.stats["llm_errors"] == 1


def test_conflict_scope_dynamic_before_and_after_speaker_merge() -> None:
    """Verify conflict_scope is 'cross_speaker' before merge_speaker_id and 'self' after."""
    mock_llm = MagicMock(spec=LLMAdapter)
    mock_llm.generate_structured.return_value = ConsistencyVerificationResponse(
        verdict=VerdictStatus.DIRECT_CONTRADICTION,
        source_turn_id=2,
        prior_turn_id=1,
        source_claim_id="CLM_0002_01",
        prior_claim_id="CLM_0001_01",
        explanation="Contradictory arrival time.",
    )
    ledger = ClaimsLedger(llm_adapter=mock_llm)

    prior = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="PROV_01",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="noon",
    )
    ledger.add_claim(prior)

    curr = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPEAKER_00",
        raw_subject="Alice",
        canonical_entity="alice",
        predicate="arrived",
        object_value="home",
        time_ref_raw="3pm",
    )
    ledger.add_claim(curr)

    res_before = ledger.verify_consistency(curr)
    assert res_before.conflict_scope == "cross_speaker"

    ledger.merge_speaker_id("PROV_01", "SPEAKER_00")

    res_after = ledger.verify_consistency(curr)
    assert res_after.conflict_scope == "self"


def test_retrieval_top_k_newest_first_and_time_ref_normalized() -> None:
    """Verify retrieval caps at 12 newest candidates, matches time_ref_normalized, and normalizes whitespace."""
    mock_llm = MagicMock(spec=LLMAdapter)
    captured_prompts: List[str] = []

    def _mock_gen(prompt: str, response_model: type, system_prompt: str):
        captured_prompts.append(prompt)
        return ConsistencyVerificationResponse(
            verdict=VerdictStatus.CONSISTENT,
            source_turn_id=31,
            prior_turn_id=30,
            source_claim_id="CLM_0031_01",
            prior_claim_id="CLM_0030_01",
            explanation="Consistent.",
        )

    mock_llm.generate_structured.side_effect = _mock_gen
    ledger = ClaimsLedger(llm_adapter=mock_llm)

    # Whitespace normalization in register_alias and add_claim
    ledger.register_alias("  Alice   Smith  ", "  alice   smith ")
    assert ledger.entity_aliases["alice smith"] == "alice smith"

    for i in range(1, 31):
        ledger.add_claim(ClaimItem(
            claim_id=f"CLM_{i:04d}_01",
            turn_id=i,
            speaker_id=f"SPEAKER_{i % 2:02d}",
            raw_subject="  Alice   Smith ",
            canonical_entity="  Alice   Smith ",
            predicate="visited",
            object_value=f"place_{i}",
            time_ref_raw="yesterday",
        ))

    curr = ClaimItem(
        claim_id="CLM_0031_01",
        turn_id=31,
        speaker_id="SPEAKER_00",
        raw_subject="Alice Smith",
        canonical_entity="alice smith",
        predicate="visited",
        object_value="place_30",
        time_ref_raw="yesterday",
    )
    ledger.add_claim(curr)

    ledger.verify_consistency(curr)
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    candidate_lines = [line for line in prompt.splitlines() if line.startswith("- Prior Claim [")]
    assert len(candidate_lines) <= 12
    assert len(candidate_lines) == 12
    # Newest first: turn 30 down to turn 19
    assert "CLM_0030_01" in candidate_lines[0]
    assert "CLM_0019_01" in candidate_lines[-1]

    # Verify time_ref_normalized matching across different entities
    ledger2 = ClaimsLedger(llm_adapter=mock_llm)
    ledger2.add_claim(ClaimItem(
        claim_id="CLM_T1",
        turn_id=1,
        speaker_id="SPEAKER_00",
        raw_subject="Bob",
        canonical_entity="bob",
        predicate="left",
        object_value="office",
        time_ref_raw="at 5",
        time_ref_normalized="2026-01-01T17:00",
    ))
    c_time = ClaimItem(
        claim_id="CLM_T2",
        turn_id=2,
        speaker_id="SPEAKER_01",
        raw_subject="Carol",
        canonical_entity="carol",
        predicate="arrived",
        object_value="station",
        time_ref_raw="five pm",
        time_ref_normalized="  2026-01-01T17:00 ",
    )
    res_time = ledger2.verify_consistency(c_time)
    assert res_time.response.verdict != VerdictStatus.UNVERIFIABLE or len(captured_prompts) == 2


def test_llm_adapter_retry_on_transient_failure() -> None:
    """Verify LLMAdapter retries on transient error and succeeds."""
    adapter = LLMAdapter()
    calls = []

    def mock_generate(prompt, model, sys):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Transient network blip")
        return ClaimExtractionResponse(claims=[])

    adapter._generate_ollama = mock_generate  # type: ignore

    res = adapter.generate_structured(
        prompt="Extract claims",
        response_model=ClaimExtractionResponse,
    )
    assert isinstance(res, ClaimExtractionResponse)
    assert len(calls) == 2  # 1 failure + 1 retry

