"""Discourse Consistency Engine (dce).

Neuro-symbolic conversational cognitive load analysis and consistency verification.
"""

from __future__ import annotations

from dce.baseline_store import BaselineStore, get_default_priors
from dce.clock import ClockMode, MasterClock, PolicyQueue, QueuePolicy
from dce.math_core import (
    classify_direction,
    compute_mahalanobis,
    f0_to_semitones,
    normalized_av_xcorr,
    weighted_mad,
    weighted_median,
)
from dce.pipeline import (
    DiscourseConsistencyPipeline,
    audio_worker_process,
    reasoning_worker_process,
    vision_worker_process,
)
from dce.reasoning.claims_ledger import ClaimsLedger, assert_firewall_compliance, should_skip_extraction
from dce.reasoning.llm_adapter import LLMAdapter, sanitize_gemini_schema
from dce.reasoning.templates import generate_physical_explanation
from dce.recorder import FeatureRecorder, ReplayLoader
from dce.schemas import (
    AppConfig,
    AvSyncConfig,
    BaselineConfig,
    CancelSpeculativeEvent,
    ClaimExtractionResponse,
    ClaimItem,
    ClaimPolarity,
    ClockConfig,
    ConsistencyVerificationResponse,
    DCEBaseModel,
    DirectionLabel,
    FeatureScore,
    HardTurnCommitEvent,
    HardwareConfig,
    MahalanobisConfig,
    MahalanobisResult,
    PauseRecord,
    PriorParameter,
    PriorsData,
    ProsodyConfig,
    ProsodyFeatures,
    QueuesConfig,
    ReasoningConfig,
    RecorderTurnRecord,
    SpeakerAssignment,
    SpeakerRegistryConfig,
    SpeculativeTriggerEvent,
    TimingFeatures,
    TurnFeatures,
    TurnTranscript,
    VADChunkResult,
    VadConfig,
    VADState,
    VerdictStatus,
    VisionFeatures,
    load_config,
    sanitize_nan_inf,
)
from dce.speaker_registry import SpeakerRegistry
from dce.vad_endpointing import DualLevelEndpointTracker, LatestWinsSpeculativeMailbox

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Clock & Queues
    "MasterClock",
    "ClockMode",
    "PolicyQueue",
    "QueuePolicy",
    # Schemas
    "DCEBaseModel",
    "AppConfig",
    "ClockConfig",
    "QueuesConfig",
    "HardwareConfig",
    "VadConfig",
    "ProsodyConfig",
    "BaselineConfig",
    "MahalanobisConfig",
    "AvSyncConfig",
    "SpeakerRegistryConfig",
    "ReasoningConfig",
    "load_config",
    "sanitize_nan_inf",
    "VADState",
    "PauseRecord",
    "SpeculativeTriggerEvent",
    "CancelSpeculativeEvent",
    "HardTurnCommitEvent",
    "VADChunkResult",
    "TurnTranscript",
    "ProsodyFeatures",
    "VisionFeatures",
    "TimingFeatures",
    "TurnFeatures",
    "DirectionLabel",
    "PriorParameter",
    "PriorsData",
    "FeatureScore",
    "MahalanobisResult",
    "SpeakerAssignment",
    "ClaimPolarity",
    "ClaimItem",
    "ClaimExtractionResponse",
    "VerdictStatus",
    "ConsistencyVerificationResponse",
    "RecorderTurnRecord",
    # Math Core
    "f0_to_semitones",
    "weighted_median",
    "weighted_mad",
    "compute_mahalanobis",
    "classify_direction",
    "normalized_av_xcorr",
    # Baseline Store
    "BaselineStore",
    "get_default_priors",
    # Speaker Registry
    "SpeakerRegistry",
    # VAD & Endpointing
    "DualLevelEndpointTracker",
    "LatestWinsSpeculativeMailbox",
    # Feature Recorder & Replay
    "FeatureRecorder",
    "ReplayLoader",
    # Reasoning Subsystem
    "generate_physical_explanation",
    "LLMAdapter",
    "sanitize_gemini_schema",
    "ClaimsLedger",
    "assert_firewall_compliance",
    "should_skip_extraction",
    # Pipeline Orchestration & Multiprocessing Workers
    "DiscourseConsistencyPipeline",
    "audio_worker_process",
    "vision_worker_process",
    "reasoning_worker_process",
]
