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
    ClaimExtractionResponse,
    ClaimItem,
    ConsistencyVerificationResponse,
    TurnTranscript,
    VerdictStatus,
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
        "Uh huh.",
        "Okay.",
        "Right.",
        "I see.",
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
    verdict_resp = ledger.verify_consistency(current_claim)

    # Must be forced to REJECTED_INVALID_REF
    assert verdict_resp.verdict == VerdictStatus.REJECTED_INVALID_REF
    assert verdict_resp.prior_turn_id == 999
    assert "does not exist in ledger" in verdict_resp.explanation


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

    verdict_resp = ledger.verify_consistency(current_claim)
    assert verdict_resp.verdict == VerdictStatus.DIRECT_CONTRADICTION
    assert verdict_resp.prior_turn_id == 1
