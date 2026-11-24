"""Discourse Consistency Engine - Pydantic v2 Schemas and Data Contracts.

Defines all core schemas, event objects, configurations, feature representations,
statistical baseline structures, speaker identity contracts, and reasoning models.
Strictly enforces NaN/Infinity sanitization to None and allow_inf_nan=False.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def sanitize_nan_inf(value: Any) -> Any:
    """Recursively sanitize non-finite floats (NaN, +Inf, -Inf) to None.

    Args:
        value: Any input data structure.

    Returns:
        Data structure with all NaN/Inf values replaced by None.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_nan_inf(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_nan_inf(v) for v in value]
    return value


class DCEBaseModel(BaseModel):
    """Base Pydantic model for DCE with strict NaN/Inf safety and extra key tolerance."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="ignore",
        validate_assignment=True,
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _sanitize_inputs(cls, data: Any) -> Any:
        """Sanitize non-finite float inputs to None before validation."""
        return sanitize_nan_inf(data)


# ============================================================================
# Configuration Schemas
# ============================================================================


class ClockConfig(DCEBaseModel):
    """Configuration for MasterClock."""

    av_offset_ns: int = 0


class QueuesConfig(DCEBaseModel):
    """Configuration for engine policy queues."""

    offline_policy: str = "block"
    live_vision_policy: str = "drop_oldest"
    live_audio_policy: str = "never_drop"
    max_size: int = 128


class HardwareConfig(DCEBaseModel):
    """Configuration for compute hardware and model backends."""

    asr_device: str = "cpu"
    llm_provider: str = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_fallback_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_num_ctx: int = 4096
    gemini_model: str = "gemini-2.5-flash-lite"


class VadConfig(DCEBaseModel):
    """Configuration for Voice Activity Detection and turn boundary segmentation."""

    sample_rate: int = 16000
    speech_prob_on: float = 0.55
    speech_prob_off: float = 0.35
    min_speech_ms: float = 160.0
    min_micro_pause_ms: float = 200.0
    soft_pause_ms: float = 400.0
    hard_eot_ms: float = 800.0
    min_speculative_ms: float = 600.0
    max_turn_ms: float = 25000.0
    overlap_ratio_threshold: float = 0.15


class ProsodyConfig(DCEBaseModel):
    """Configuration for acoustic prosody extraction."""

    creak_trim_min_speech_ms: float = 600.0
    creak_trim_tail_ms: float = 200.0
    min_voiced_segment_ms: float = 75.0
    boundary_trim_ms: float = 15.0
    min_glottal_periods: int = 20
    default_pitch_floor_hz: float = 75.0
    pitch_floor_min_hz: float = 60.0
    pitch_floor_max_hz: float = 140.0


class BaselineConfig(DCEBaseModel):
    """Configuration for rolling baseline shrinkage and anchor locking."""

    rolling_window_turns: int = 40
    shrinkage_k: float = 6.0
    anchor_lock_turns: int = 8
    gate_full_z: float = 3.0
    gate_reduced_z: float = 5.0
    gate_reduced_weight: float = 0.2


class MahalanobisConfig(DCEBaseModel):
    """Configuration for Mahalanobis anomaly detection and regularized covariance."""

    reg_lambda: float = 0.30
    eig_floor: float = 1.0e-6
    cond_limit: float = 1.0e+12
    tail_score_threshold: float = 0.05
    tension_z_threshold: float = 1.5
    hnr_tension_z_threshold: float = -1.5
    calm_z_threshold: float = -1.5


class AvSyncConfig(DCEBaseModel):
    """Configuration for audio-visual cross-correlation synchronization."""

    grid_hz: float = 50.0
    min_overlap_sec: float = 0.5
    max_lag_ms: float = 200.0
    min_std: float = 1.0e-6


class SpeakerRegistryConfig(DCEBaseModel):
    """Configuration for speaker clustering and provisional buffer promotion."""

    sim_high: float = 0.65
    sim_low: float = 0.45
    face_sim_floor: float = 0.35
    min_confirm_sec: float = 2.0
    provisional_ttl_sec: float = 60.0
    max_speakers: int = 4
    max_provisional_buffers: int = 6
    min_embedding_norm: float = 1.0e-8


class ReasoningConfig(DCEBaseModel):
    """Configuration for LLM reasoning and claim verification."""

    anomaly_guided_search: bool = False
    max_llm_retries: int = 1


class AppConfig(DCEBaseModel):
    """Root configuration model combining all DCE subsystem options."""

    mode: str = "offline"
    clock: ClockConfig = Field(default_factory=ClockConfig)
    queues: QueuesConfig = Field(default_factory=QueuesConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    mahalanobis: MahalanobisConfig = Field(default_factory=MahalanobisConfig)
    av_sync: AvSyncConfig = Field(default_factory=AvSyncConfig)
    speaker_registry: SpeakerRegistryConfig = Field(default_factory=SpeakerRegistryConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load and validate DCE configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw_dict)


# ============================================================================
# VAD & Endpointing Event Schemas
# ============================================================================


class VADState(str, Enum):
    """Operational states of the dual-level VAD endpoint tracker."""

    IDLE = "IDLE"
    SPEAKING = "SPEAKING"
    SOFT_PAUSE = "SOFT_PAUSE"


class PauseRecord(DCEBaseModel):
    """Record of an intra-turn hesitation or micro-pause."""

    start_ns: int
    duration_ms: float


class SpeculativeTriggerEvent(DCEBaseModel):
    """Event emitted at soft pause to initiate early speculative ASR transcription."""

    turn_id: int
    epoch_id: int
    start_ns: int
    last_voiced_ns: int
    audio_pcm16k: Optional[List[float]] = None


class CancelSpeculativeEvent(DCEBaseModel):
    """Event emitted when speaker resumes speech after a soft pause, cancelling speculative ASR."""

    turn_id: int
    invalidated_epoch_id: int
    hesitation_pause: Optional[PauseRecord] = None


class HardTurnCommitEvent(DCEBaseModel):
    """Event emitted when a turn has definitively completed or reached maximum duration."""

    turn_id: int
    start_ns: int
    end_ns: int
    audio_pcm16k: Optional[List[float]] = None
    voiced_chunks: int = 0
    overlap_voiced_chunks: int = 0
    overlap_ratio: float = 0.0
    overlap_flag: bool = False
    forced_split: bool = False
    intra_turn_pauses: List[PauseRecord] = Field(default_factory=list)


class VADChunkResult(DCEBaseModel):
    """Result of evaluating a single audio chunk through VAD."""

    vad_prob: float
    is_speech: bool
    is_overlap: bool = False
    pts_ns: int = 0


class TurnTranscript(DCEBaseModel):
    """Completed textual transcription of a conversational turn."""

    turn_id: int
    speaker_id: str
    text: str
    start_ns: int
    end_ns: int
    is_question: bool = False
    confidence: Optional[float] = None


# ============================================================================
# Feature Extraction Schemas
# ============================================================================


class ProsodyFeatures(DCEBaseModel):
    """Acoustic prosody metrics extracted from voiced speech."""

    f0_median_hz: Optional[float] = None
    f0_semitone_sd: Optional[float] = None
    jitter_local_pct: Optional[float] = None
    shimmer_local_pct: Optional[float] = None
    hnr_db: Optional[float] = None
    articulation_rate: Optional[float] = None
    glottal_periods_count: int = 0
    quality_flag: str = "ok"


class VisionFeatures(DCEBaseModel):
    """Visual kinematic features extracted from facial tracking."""

    blink_rate_per_min: Optional[float] = None
    mean_lip_aperture: Optional[float] = None
    face_detected: bool = False
    face_id_hint: Optional[str] = None
    quality_flag: str = "ok"


class TimingFeatures(DCEBaseModel):
    """Conversational timing, latency, and hesitation metrics."""

    response_latency_open: Optional[float] = None
    response_latency_closed: Optional[float] = None
    turn_duration_ms: Optional[float] = None
    speech_duration_ms: Optional[float] = None
    total_pause_ms: Optional[float] = None
    pause_count: int = 0


class TurnFeatures(DCEBaseModel):
    """Unified container of all multimodal feature extractions for a turn."""

    turn_id: int
    speaker_id: str
    start_ns: int
    end_ns: int
    timing: TimingFeatures = Field(default_factory=TimingFeatures)
    prosody: ProsodyFeatures = Field(default_factory=ProsodyFeatures)
    vision: VisionFeatures = Field(default_factory=VisionFeatures)
    av_sync_lag_ms: Optional[float] = None
    av_sync_corr: Optional[float] = None


# ============================================================================
# Statistical Baseline & Anomaly Schemas
# ============================================================================


class DirectionLabel(str, Enum):
    """Four-state direction classification of cognitive load and physiological arousal."""

    NORMAL = "normal"
    COGNITIVE_LOAD_ANOMALY = "cognitive_load_anomaly"
    CALM_DEVIATION = "calm_deviation"
    MIXED_SUBTHRESHOLD_DEVIATION = "mixed_subthreshold_deviation"


class PriorParameter(DCEBaseModel):
    """Literature population prior parameters for a single feature."""

    feature_name: str
    pop_median: float
    pop_sigma: float
    sigma_floor: float
    max_anchor_drift: float
    log_transform: bool = False
    log_offset: float = 0.0
    stress_polarity: int = 0


class PriorsData(DCEBaseModel):
    """Schema for literature priors database (data/priors_en.json)."""

    calibrated: bool = False
    source: str = "literature_initial_estimates"
    features: Dict[str, PriorParameter] = Field(default_factory=dict)
    correlation_matrix: List[List[float]] = Field(default_factory=list)
    feature_order: List[str] = Field(default_factory=list)


class FeatureScore(DCEBaseModel):
    """Individual normalized feature deviation score against speaker baseline."""

    feature_name: str
    raw_value: Optional[float] = None
    transformed_value: Optional[float] = None
    baseline_mean: Optional[float] = None
    baseline_sigma: Optional[float] = None
    z_score: Optional[float] = None
    stress_z: Optional[float] = None
    confidence_weight: float = 1.0


class MahalanobisResult(DCEBaseModel):
    """Result of regularized Block-Diagonal Mahalanobis distance computation."""

    d_sq: float
    mahalanobis_distance: float
    chi2_tail_score: float
    degrees_of_freedom: int
    vocal_tension_flag: bool
    direction_label: DirectionLabel
    eigen_fallback_used: bool = False
    evaluated_features: List[str] = Field(default_factory=list)
    feature_z_scores: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# Speaker Registry Schemas
# ============================================================================


class SpeakerAssignment(DCEBaseModel):
    """Speaker identity assignment for an utterance."""

    speaker_id: str
    similarity: float
    is_provisional: bool = False
    low_confidence: bool = False
    promoted_from: Optional[str] = None


# ============================================================================
# Reasoning & Claims Verification Schemas
# ============================================================================


class ClaimPolarity(str, Enum):
    """Polarity of an extracted factual claim."""

    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class ClaimItem(DCEBaseModel):
    """Structured factual proposition extracted from speech."""

    claim_id: str
    turn_id: int
    speaker_id: str
    raw_subject: str
    canonical_entity: str
    predicate: str
    object_value: str
    time_ref_raw: Optional[str] = None
    time_ref_normalized: Optional[str] = None
    location_ref: Optional[str] = None
    polarity: str = "affirmative"
    confidence: float = 1.0


class ClaimExtractionResponse(DCEBaseModel):
    """Output schema for Step A LLM claim extraction."""

    claims: List[ClaimItem] = Field(default_factory=list)
    extracted_entities: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


class VerdictStatus(str, Enum):
    """Verdict classification for conversational consistency verification."""

    CONSISTENT = "CONSISTENT"
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    TIMELINE_SHIFT = "TIMELINE_SHIFT"
    UNVERIFIABLE = "UNVERIFIABLE"
    REJECTED_INVALID_REF = "REJECTED_INVALID_REF"


class ConsistencyVerificationResponse(DCEBaseModel):
    """Output schema for Step B LLM consistency verification."""

    verdict: VerdictStatus
    source_turn_id: int
    prior_turn_id: Optional[int] = None
    source_claim_id: Optional[str] = None
    prior_claim_id: Optional[str] = None
    explanation: str
    confidence: float = 1.0


# ============================================================================
# Feature Recorder & Replay Record Schema
# ============================================================================


class RecorderTurnRecord(DCEBaseModel):
    """Bit-exact record of a complete conversational turn for offline replay."""

    turn_id: int
    speaker_assignment: SpeakerAssignment
    transcript: Optional[TurnTranscript] = None
    features: TurnFeatures
    mahalanobis: Optional[MahalanobisResult] = None
    claims: List[ClaimItem] = Field(default_factory=list)
    verification: Optional[ConsistencyVerificationResponse] = None
