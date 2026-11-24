"""Discourse Consistency Engine - Structured LLM Adapters and Schema Sanitizer.

Provides structured output integration for Ollama and Gemini backends,
a 3-step Gemini JSON schema sanitizer with reference inlining and nullable preservation,
and automatic 1x retry logic.
Enforces lazy imports for httpx, ollama, and google.genai.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from dce.schemas import HardwareConfig, ReasoningConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _inline_defs(node: Any, defs: Dict[str, Any], visited: Optional[set] = None) -> Any:
    """Step 1: Recursively replace $ref pointers with their actual contents from $defs."""
    if visited is None:
        visited = set()

    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref_path = node["$ref"]
            if ref_path.startswith("#/$defs/"):
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs and ref_name not in visited:
                    target = copy.deepcopy(defs[ref_name])
                    node.clear()
                    node.update(target)
                    return _inline_defs(node, defs, visited | {ref_name})

        for k, v in list(node.items()):
            node[k] = _inline_defs(v, defs, visited)
        return node

    if isinstance(node, list):
        return [_inline_defs(item, defs, visited) for item in node]

    return node


def _collapse_nullables(node: Any) -> Any:
    """Step 2: Collapse anyOf containing null type into primary schema with nullable=True."""
    if isinstance(node, dict):
        if "anyOf" in node and isinstance(node["anyOf"], list):
            non_null = [
                item
                for item in node["anyOf"]
                if isinstance(item, dict) and item.get("type") != "null"
            ]
            has_null = len(non_null) < len(node["anyOf"])
            if has_null and len(non_null) == 1:
                primary = copy.deepcopy(non_null[0])
                del node["anyOf"]
                node.update(primary)
                node["nullable"] = True

        for k, v in list(node.items()):
            node[k] = _collapse_nullables(v)
        return node

    if isinstance(node, list):
        return [_collapse_nullables(item) for item in node]

    return node


def _strip_gemini_metadata(node: Any) -> Any:
    """Step 3: Strip unsupported Gemini metadata while strictly preserving user property names."""
    if isinstance(node, dict):
        # Remove unsupported metadata keys from current object
        for key in ["$defs", "default", "additionalProperties", "title"]:
            node.pop(key, None)

        # Recurse into user properties dictionary, preserving the property key names
        if "properties" in node and isinstance(node["properties"], dict):
            for prop_name, prop_schema in list(node["properties"].items()):
                node["properties"][prop_name] = _strip_gemini_metadata(prop_schema)

        # Recurse into all other dictionary values
        for k, v in list(node.items()):
            if k != "properties":
                node[k] = _strip_gemini_metadata(v)

        return node

    if isinstance(node, list):
        return [_strip_gemini_metadata(item) for item in node]

    return node


def sanitize_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the mandatory 3-step Gemini schema sanitization pipeline.

    Step 1: Inlines all `{"$ref": "#/$defs/..."}` at any depth using `$defs`.
    Step 2: Collapses `anyOf` with null types (including Optional[SubModel]) to primary type with nullable=True.
    Step 3: Strips metadata ($defs, default, additionalProperties, title) while preserving user properties named 'title'.

    Args:
        schema: Raw JSON schema dictionary produced by Pydantic.

    Returns:
        Sanitized JSON schema compliant with the Gemini API.
    """
    cleaned = copy.deepcopy(schema)
    defs = cleaned.get("$defs", {})

    # Step 1: Inline definitions
    cleaned = _inline_defs(cleaned, defs)

    # Step 2: Collapse nullables
    cleaned = _collapse_nullables(cleaned)

    # Step 3: Strip metadata while preserving user property names
    cleaned = _strip_gemini_metadata(cleaned)

    return cleaned


class LLMAdapter:
    """Adaptive LLM interface supporting Ollama and Gemini with retries and structured output."""

    def __init__(
        self,
        reasoning_config: Optional[ReasoningConfig] = None,
        hardware_config: Optional[HardwareConfig] = None,
    ) -> None:
        """Initialize LLMAdapter.

        Args:
            reasoning_config: Reasoning parameters (max_retries, etc.).
            hardware_config: Hardware options (provider, model names).
        """
        self._reasoning_cfg: ReasoningConfig = (
            reasoning_config if reasoning_config is not None else ReasoningConfig()
        )
        self._hardware_cfg: HardwareConfig = (
            hardware_config if hardware_config is not None else HardwareConfig()
        )
        self._provider: str = self._hardware_cfg.llm_provider.lower()
        self._max_retries: int = self._reasoning_cfg.max_llm_retries

    @property
    def provider(self) -> str:
        """Return active LLM provider name."""
        return self._provider

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Generate structured Pydantic response with automatic 1x retry on failure.

        Args:
            prompt: User prompt text.
            response_model: Expected Pydantic model class.
            system_prompt: Optional system prompt instructions.

        Returns:
            Validated instance of response_model.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                if self._provider == "gemini":
                    return self._generate_gemini(prompt, response_model, system_prompt)
                return self._generate_ollama(prompt, response_model, system_prompt)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM generation attempt %d failed: %s. Retrying...",
                    attempt + 1,
                    exc,
                )

        raise RuntimeError(
            f"LLM structured generation failed after {self._max_retries + 1} attempts: {last_error}"
        )

    def _generate_ollama(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Query Ollama backend using structured JSON format."""
        # LAZY IMPORT of httpx or ollama
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for Ollama integration.")

        url = "http://localhost:11434/api/chat"
        schema = response_model.model_json_schema()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._hardware_cfg.ollama_model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {
                "num_ctx": self._hardware_cfg.ollama_num_ctx,
                "temperature": 0.0,
            },
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("message", {}).get("content", "{}")
            return response_model.model_validate_json(raw_text)

    def _generate_gemini(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Query Gemini backend with 3-step sanitized JSON schema."""
        # LAZY IMPORT of google.genai
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai is required for Gemini integration.")

        raw_schema = response_model.model_json_schema()
        clean_schema = sanitize_gemini_schema(raw_schema)

        client = genai.Client()
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=clean_schema,
            temperature=0.0,
            system_instruction=system_prompt,
        )

        response = client.models.generate_content(
            model=self._hardware_cfg.gemini_model,
            contents=prompt,
            config=config,
        )
        return response_model.model_validate_json(response.text)
