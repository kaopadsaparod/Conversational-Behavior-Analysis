"""Discourse Consistency Engine - Neuro-Symbolic Reasoning Subsystem.

Provides rule-based physical explanation templates, structured LLM adapters
with Gemini schema sanitization, and a Claims Ledger protected by the
Neuro-Symbolic Firewall.
"""

from __future__ import annotations

from dce.reasoning.claims_ledger import ClaimsLedger
from dce.reasoning.llm_adapter import LLMAdapter, sanitize_gemini_schema
from dce.reasoning.templates import generate_physical_explanation

__all__ = [
    "generate_physical_explanation",
    "LLMAdapter",
    "sanitize_gemini_schema",
    "ClaimsLedger",
]
