"""Discourse Consistency Engine (dce).

Neuro-symbolic conversational cognitive load analysis and consistency verification.

Uses PEP 562 lazy exports so lightweight entrypoints such as ``python -m dce.mcp_server``
start in <0.2s without eagerly importing numpy/scipy/pydantic/pipeline modules.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

__version__ = "0.1.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Clock & Queues
    "MasterClock": ("dce.clock", "MasterClock"),
    "ClockMode": ("dce.clock", "ClockMode"),
    "PolicyQueue": ("dce.clock", "PolicyQueue"),
    "QueuePolicy": ("dce.clock", "QueuePolicy"),
    # Schemas
    "DCEBaseModel": ("dce.schemas", "DCEBaseModel"),
    "AppConfig": ("dce.schemas", "AppConfig"),
    "ClockConfig": ("dce.schemas", "ClockConfig"),
    "QueuesConfig": ("dce.schemas", "QueuesConfig"),
    "HardwareConfig": ("dce.schemas", "HardwareConfig"),
    "VadConfig": ("dce.schemas", "VadConfig"),
    "ProsodyConfig": ("dce.schemas", "ProsodyConfig"),
    "BaselineConfig": ("dce.schemas", "BaselineConfig"),
    "MahalanobisConfig": ("dce.schemas", "MahalanobisConfig"),
    "AvSyncConfig": ("dce.schemas", "AvSyncConfig"),
    "SpeakerRegistryConfig": ("dce.schemas", "SpeakerRegistryConfig"),
    "ReasoningConfig": ("dce.schemas", "ReasoningConfig"),
    "load_config": ("dce.schemas", "load_config"),
    "sanitize_nan_inf": ("dce.schemas", "sanitize_nan_inf"),
    "VADState": ("dce.schemas", "VADState"),
    "PauseRecord": ("dce.schemas", "PauseRecord"),
    "SpeculativeTriggerEvent": ("dce.schemas", "SpeculativeTriggerEvent"),
    "CancelSpeculativeEvent": ("dce.schemas", "CancelSpeculativeEvent"),
    "HardTurnCommitEvent": ("dce.schemas", "HardTurnCommitEvent"),
    "VADChunkResult": ("dce.schemas", "VADChunkResult"),
    "TurnTranscript": ("dce.schemas", "TurnTranscript"),
    "ProsodyFeatures": ("dce.schemas", "ProsodyFeatures"),
    "VisionFeatures": ("dce.schemas", "VisionFeatures"),
    "TimingFeatures": ("dce.schemas", "TimingFeatures"),
    "TurnFeatures": ("dce.schemas", "TurnFeatures"),
    "DirectionLabel": ("dce.schemas", "DirectionLabel"),
    "PriorParameter": ("dce.schemas", "PriorParameter"),
    "PriorsData": ("dce.schemas", "PriorsData"),
    "FeatureScore": ("dce.schemas", "FeatureScore"),
    "MahalanobisResult": ("dce.schemas", "MahalanobisResult"),
    "SpeakerAssignment": ("dce.schemas", "SpeakerAssignment"),
    "ClaimPolarity": ("dce.schemas", "ClaimPolarity"),
    "ClaimItem": ("dce.schemas", "ClaimItem"),
    "ClaimExtractionResponse": ("dce.schemas", "ClaimExtractionResponse"),
    "VerdictStatus": ("dce.schemas", "VerdictStatus"),
    "ConsistencyVerificationResponse": ("dce.schemas", "ConsistencyVerificationResponse"),
    "RecorderTurnRecord": ("dce.schemas", "RecorderTurnRecord"),
    # Math Core
    "f0_to_semitones": ("dce.math_core", "f0_to_semitones"),
    "weighted_median": ("dce.math_core", "weighted_median"),
    "weighted_mad": ("dce.math_core", "weighted_mad"),
    "compute_mahalanobis": ("dce.math_core", "compute_mahalanobis"),
    "classify_direction": ("dce.math_core", "classify_direction"),
    "normalized_av_xcorr": ("dce.math_core", "normalized_av_xcorr"),
    # Baseline Store
    "BaselineStore": ("dce.baseline_store", "BaselineStore"),
    "get_default_priors": ("dce.baseline_store", "get_default_priors"),
    # Speaker Registry
    "SpeakerRegistry": ("dce.speaker_registry", "SpeakerRegistry"),
    # VAD & Endpointing
    "DualLevelEndpointTracker": ("dce.vad_endpointing", "DualLevelEndpointTracker"),
    "LatestWinsSpeculativeMailbox": ("dce.vad_endpointing", "LatestWinsSpeculativeMailbox"),
    # Feature Recorder & Replay
    "FeatureRecorder": ("dce.recorder", "FeatureRecorder"),
    "ReplayLoader": ("dce.recorder", "ReplayLoader"),
    # Reasoning Subsystem
    "generate_physical_explanation": ("dce.reasoning.templates", "generate_physical_explanation"),
    "LLMAdapter": ("dce.reasoning.llm_adapter", "LLMAdapter"),
    "sanitize_gemini_schema": ("dce.reasoning.llm_adapter", "sanitize_gemini_schema"),
    "ClaimsLedger": ("dce.reasoning.claims_ledger", "ClaimsLedger"),
    "assert_firewall_compliance": ("dce.reasoning.claims_ledger", "assert_firewall_compliance"),
    "should_skip_extraction": ("dce.reasoning.claims_ledger", "should_skip_extraction"),
    # Pipeline Orchestration & Multiprocessing Workers
    "DiscourseConsistencyPipeline": ("dce.pipeline", "DiscourseConsistencyPipeline"),
    "audio_worker_process": ("dce.pipeline", "audio_worker_process"),
    "vision_worker_process": ("dce.pipeline", "vision_worker_process"),
    "reasoning_worker_process": ("dce.pipeline", "reasoning_worker_process"),
}

__all__: List[str] = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(__all__)
