"""Discourse Consistency Engine - Windows-Safe Stdio MCP Server.

Implements a zero-dependency JSON-RPC 2.0 stdio Model Context Protocol (MCP) server
compatible with Antigravity, Claude Code, and Codex.

Enforces:
1. Lazy model loading (<0.2s startup for initialize / tools/list).
2. Strict Windows stdio safety: all Python logging, sys.stdout prints, and C/C++
   fd-1 writes are redirected to sys.stderr so sys.stdout carries ONLY JSON-RPC frames.
3. 4-Tool State-Locked Neuro-Symbolic Firewall:
   - dce_ingest_media: extracts audio/vision/prosody/Mahalanobis into backend session
     memory and returns ONLY annotated transcripts + incomplete-sentence flags.
   - dce_submit_verdicts: validates turn_ids, downgrades phonetic-fragile conflicts
     to NEEDS_AUDIO_REVIEW (NEEDS_HUMAN_REVIEW), and locks text verdicts.
   - dce_unlock_signals: refuses until dce_submit_verdicts has locked verdicts; then
     unlocks BaselineStore (k_mu=0 default), Mahalanobis, and templates.py explanations
     and writes deterministic .jsonl replay logs.
   - dce_slice_turn_audio: slices a 3-second WAV clip for a target turn for human review.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("dce.mcp_server")

SERVER_NAME = "dce-analyzer"
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

_TRAILING_INCOMPLETE_WORDS = {
    "and",
    "or",
    "but",
    "because",
    "so",
    "if",
    "when",
    "while",
    "although",
    "though",
    "to",
    "for",
    "with",
    "from",
    "at",
    "by",
    "in",
    "on",
    "of",
    "the",
    "a",
    "an",
    "that",
    "which",
    "who",
}

# Lightweight phonetic-fragile lookup for annotation and verdict validation
# (mirrored from claims_ledger so annotation works without heavy imports).
_PHONETIC_ALT_PAIRS: List[Tuple[str, str]] = [
    ("13", "30"), ("thirteen", "thirty"),
    ("14", "40"), ("fourteen", "forty"),
    ("15", "50"), ("fifteen", "fifty"),
    ("16", "60"), ("sixteen", "sixty"),
    ("17", "70"), ("seventeen", "seventy"),
    ("18", "80"), ("eighteen", "eighty"),
    ("19", "90"), ("nineteen", "ninety"),
    ("can", "can't"), ("cannot", "can"), ("won't", "want"),
    ("did", "didn't"), ("do", "don't"), ("is", "isn't"),
    ("was", "wasn't"), ("were", "weren't"), ("have", "haven't"),
    ("has", "hasn't"), ("had", "hadn't"), ("would", "wouldn't"),
    ("could", "couldn't"), ("should", "shouldn't"),
    ("leave", "live"), ("left", "lived"),
    ("two", "to"), ("no", "know"), ("hear", "here"),
    ("their", "there"), ("wait", "weight"), ("week", "weak"),
    ("right", "write"), ("buy", "by"), ("night", "knight"),
    ("eight", "ate"), ("for", "four"), ("one", "won"),
    ("wear", "where"), ("knew", "new"), ("weather", "whether"),
]

_PHONETIC_ALT_MAP: Dict[str, str] = {}
for _a, _b in _PHONETIC_ALT_PAIRS:
    _PHONETIC_ALT_MAP.setdefault(_a, _b)
    _PHONETIC_ALT_MAP.setdefault(_b, _a)


MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "dce_ingest_media",
        "description": (
            "Ingest an audio/video file, compute acoustic prosody, vision kinematics, "
            "and Mahalanobis cognitive-load scores into locked backend session memory, "
            "and return ONLY firewall-safe annotated transcripts with low-confidence "
            "word tags ([word? p=0.52 | alt: ...]) and incomplete-sentence flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "media_path": {
                    "type": "string",
                    "description": "Path to the audio (.wav, .mp3, .flac) or video (.mp4, .mkv, .mov) file.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional custom session identifier. Auto-generated if omitted.",
                },
                "shrinkage_k": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Bayesian shrinkage k_mu for BaselineStore (default 0.0 for slow speakers).",
                },
                "low_conf_threshold": {
                    "type": "number",
                    "default": 0.70,
                    "description": "Word probability threshold below which words are annotated with [word? p=...].",
                },
                "asr_device": {
                    "type": "string",
                    "default": "cpu",
                    "enum": ["cpu", "cuda"],
                    "description": "Device for faster-whisper ASR inference.",
                },
                "asr_model": {
                    "type": "string",
                    "default": "small.en",
                    "description": "Faster-whisper model checkpoint name.",
                },
                "use_asr": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to run faster-whisper ASR if transcript_override is not supplied.",
                },
                "transcript_override": {
                    "type": "array",
                    "description": "Optional pre-transcribed turn list for testing or external ASR.",
                    "items": {"type": "object"},
                },
            },
            "required": ["media_path"],
        },
    },
    {
        "name": "dce_submit_verdicts",
        "description": (
            "Submit LLM-analyzed factual claims and consistency verdicts for an ingested "
            "session. Validates turn_id references against actual session turns, checks "
            "for phonetic-fragile / low-confidence ASR collisions (15/50, did/didn't, "
            "can/can't) and downgrades them to NEEDS_AUDIO_REVIEW (NEEDS_HUMAN_REVIEW), "
            "then locks text verdicts so signals can be unlocked."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier returned by dce_ingest_media.",
                },
                "verdicts": {
                    "type": "array",
                    "description": "List of consistency verdict objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": [
                                    "CONSISTENT",
                                    "DIRECT_CONTRADICTION",
                                    "TIMELINE_SHIFT",
                                    "NEEDS_AUDIO_REVIEW",
                                    "UNVERIFIABLE",
                                ],
                            },
                            "source_turn_id": {"type": "integer"},
                            "prior_turn_id": {"type": ["integer", "null"]},
                            "source_claim_id": {"type": ["string", "null"]},
                            "prior_claim_id": {"type": ["string", "null"]},
                            "explanation": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["verdict", "source_turn_id", "explanation"],
                    },
                },
                "claims": {
                    "type": "array",
                    "description": "Optional list of extracted ClaimItem objects.",
                    "items": {"type": "object"},
                },
            },
            "required": ["session_id", "verdicts"],
        },
    },
    {
        "name": "dce_unlock_signals",
        "description": (
            "Unlock precomputed BaselineStore (k_mu=0), Mahalanobis anomaly scores, "
            "and templates.py physical explanations paired with locked text verdicts, "
            "and write a deterministic .jsonl replay file. Refuses to run unless "
            "dce_submit_verdicts has already locked the session verdicts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier from dce_ingest_media.",
                },
                "output_jsonl_path": {
                    "type": "string",
                    "description": "Optional output path for the .jsonl replay file.",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "dce_slice_turn_audio",
        "description": (
            "Slice a short WAV audio clip (default 3.0 seconds) for a specific turn_id "
            "into the workspace so the human reviewer can click and listen to verify "
            "ambiguous words (e.g., can vs can't, 15 vs 50, did vs didn't)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session identifier from dce_ingest_media.",
                },
                "turn_id": {
                    "type": "integer",
                    "description": "Target turn_id to slice.",
                },
                "duration_sec": {
                    "type": "number",
                    "default": 3.0,
                    "description": "Maximum duration of the sliced WAV clip in seconds.",
                },
                "start_offset_sec": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Offset in seconds from the start of the turn.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional directory where the sliced .wav file will be saved.",
                },
            },
            "required": ["session_id", "turn_id"],
        },
    },
]


@dataclasses.dataclass
class _StoredTurn:
    """Backend session storage for a single turn (never exposed raw before unlock)."""

    turn_id: int
    speaker_id: str
    start_ns: int
    end_ns: int
    audio_pcm16k: Any  # np.ndarray float32
    raw_text: str
    annotated_text: str
    low_confidence_words: List[Dict[str, Any]]
    incomplete_sentence: bool
    forced_split: bool
    overlap_flag: bool
    transcript_obj: Any  # TurnTranscript
    features_obj: Any  # TurnFeatures
    mahalanobis_obj: Any  # MahalanobisResult
    physical_explanation: str
    speaker_assignment_obj: Any  # SpeakerAssignment


@dataclasses.dataclass
class _DCESession:
    """State-locked backend session for the 4-step MCP workflow."""

    session_id: str
    state: str  # "INGESTED" | "VERDICTS_LOCKED" | "SIGNALS_UNLOCKED"
    media_path: str
    sample_rate: int
    shrinkage_k: float
    turns: Dict[int, _StoredTurn]
    turn_order: List[int]
    submitted_claims: List[Any] = dataclasses.field(default_factory=list)
    locked_verdicts: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    validated_verdict_objs: List[Any] = dataclasses.field(default_factory=list)
    replay_jsonl_path: Optional[str] = None


def _load_wav_mono_16k(wav_path: Path) -> Tuple[Any, int]:
    """Load a WAV file using stdlib wave + numpy and normalize to 16kHz mono float32."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if n_frames == 0 or len(raw_bytes) == 0:
        return np.zeros(0, dtype=np.float32), 16000

    if sampwidth == 1:
        arr = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw_u8[:, 0].astype(np.int32)
            | (raw_u8[:, 1].astype(np.int32) << 8)
            | (raw_u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)

    if framerate != 16000 and len(arr) > 0:
        target_len = max(1, int(round(len(arr) * (16000.0 / float(framerate)))))
        src_x = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        arr = np.interp(dst_x, src_x, arr).astype(np.float32)

    return arr.astype(np.float32), 16000


def _load_media_audio_and_vision(media_path: Path) -> Tuple[Any, int, List[Any]]:
    """Load 16kHz mono float32 audio and 15 FPS RGB video frames via shared decoder."""
    from dce.extractors.audio_dsp import decode_media_audio_and_frames

    return decode_media_audio_and_frames(
        media_path=media_path,
        target_sr=16000,
        video_fps=15.0,
        max_frames=450,
    )


def _segment_audio_energy(
    audio: Any,
    sample_rate: int = 16000,
    silence_split_ms: float = 600.0,
    max_turn_sec: float = 25.0,
) -> List[Tuple[int, int, bool]]:
    """Segment 16kHz audio into (start_sample, end_sample, forced_split) tuples."""
    n_samples = len(audio)
    if n_samples == 0:
        return []

    import numpy as np

    frame_len = int(0.025 * sample_rate)
    hop_len = int(0.020 * sample_rate)
    if n_samples <= frame_len:
        return [(0, n_samples, False)]

    rms_vals: List[float] = []
    starts: List[int] = []
    for s in range(0, n_samples - frame_len + 1, hop_len):
        seg = audio[s : s + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(seg * seg))))
        starts.append(s)

    max_rms = max(rms_vals) if rms_vals else 0.0
    if max_rms < 1e-6:
        return [(0, n_samples, False)]

    thresh = max(1e-4, max_rms * 0.08)
    min_silence_frames = max(1, int(round((silence_split_ms / 1000.0) / 0.020)))
    max_turn_samples = int(round(max_turn_sec * sample_rate))

    segments: List[Tuple[int, int, bool]] = []
    in_speech = False
    seg_start = 0
    silence_run = 0
    last_speech_end = 0

    for idx, rms in enumerate(rms_vals):
        s_pos = starts[idx]
        e_pos = min(n_samples, s_pos + frame_len)
        if rms >= thresh:
            if not in_speech:
                in_speech = True
                seg_start = s_pos
            silence_run = 0
            last_speech_end = e_pos
            if (e_pos - seg_start) >= max_turn_samples:
                segments.append((seg_start, e_pos, True))
                in_speech = False
                silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    if last_speech_end > seg_start:
                        segments.append((seg_start, last_speech_end, False))
                    in_speech = False
                    silence_run = 0

    if in_speech and last_speech_end > seg_start:
        segments.append((seg_start, last_speech_end, False))

    if not segments:
        return [(0, n_samples, False)]
    return segments


def _annotate_transcript(
    raw_text: str,
    words: Sequence[Any],
    low_conf_threshold: float,
    word_alternatives: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Annotate low-confidence or phonetic-fragile words as [word? p=0.52 | alt: ...]."""
    alt_lookup = dict(_PHONETIC_ALT_MAP)
    if word_alternatives:
        for k, v in word_alternatives.items():
            alt_lookup[k.lower()] = v

    low_conf_list: List[Dict[str, Any]] = []
    if not words:
        return raw_text.strip(), low_conf_list

    annotated_tokens: List[str] = []
    for w in words:
        w_str = str(getattr(w, "word", "") if not isinstance(w, dict) else w.get("word", "")).strip()
        if not w_str:
            continue
        prob = getattr(w, "probability", None) if not isinstance(w, dict) else w.get("probability")
        alt_explicit = getattr(w, "alternative", None) if not isinstance(w, dict) else w.get("alternative")
        clean_key = re.sub(r"^[^\w']+|[^\w']+$", "", w_str.lower())
        alt_word = alt_explicit or alt_lookup.get(clean_key)

        is_low = prob is not None and float(prob) < low_conf_threshold
        is_fragile_mid = prob is not None and (clean_key in alt_lookup) and float(prob) < 0.85

        if is_low or is_fragile_mid:
            p_val = round(float(prob), 2)
            low_conf_list.append(
                {
                    "word": w_str,
                    "probability": p_val,
                    "alternative": alt_word,
                }
            )
            if alt_word:
                annotated_tokens.append(f"[{w_str}? p={p_val:.2f} | alt: {alt_word}]")
            else:
                annotated_tokens.append(f"[{w_str}? p={p_val:.2f}]")
        else:
            annotated_tokens.append(w_str)

    if annotated_tokens:
        return " ".join(annotated_tokens), low_conf_list
    return raw_text.strip(), low_conf_list


def _is_incomplete_sentence(text: str, forced_split: bool) -> bool:
    """Detect cut-off or incomplete sentences."""
    if forced_split:
        return True
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "-", ",")):
        return True
    tokens = re.findall(r"[a-zA-Z0-9']+", stripped.lower())
    if tokens and tokens[-1] in _TRAILING_INCOMPLETE_WORDS:
        return True
    return False


def _extract_turn_tokens(text: str) -> List[str]:
    """Normalize text tokens for phonetic-fragile collision detection."""
    from dce.reasoning.claims_ledger import _normalize_phonetic_tokens

    return _normalize_phonetic_tokens(text)


class DCESessionStore:
    """Thread-safe state-locked session store backing the 4 MCP tools."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, _DCESession] = {}

    def get_session_state(self, session_id: str) -> Optional[str]:
        with self._lock:
            sess = self._sessions.get(session_id)
            return sess.state if sess is not None else None

    def ingest_media(
        self,
        media_path: str,
        session_id: Optional[str] = None,
        shrinkage_k: float = 0.0,
        low_conf_threshold: float = 0.70,
        asr_device: str = "cpu",
        asr_model: str = "small.en",
        use_asr: bool = True,
        transcript_override: Optional[List[Dict[str, Any]]] = None,
        priors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Step 1: Ingest media, lock acoustic/vision/Mahalanobis math, return text-only."""
        import numpy as np

        from dce.baseline_store import BaselineStore
        from dce.extractors.audio_dsp import (
            extract_egemaps_spectral_features,
            extract_prosody,
            transcribe_audio,
        )
        from dce.extractors.vision_face import extract_vision_sequence
        from dce.math_core import compute_mahalanobis
        from dce.reasoning.claims_ledger import assert_firewall_compliance
        from dce.reasoning.templates import generate_physical_explanation
        from dce.schemas import (
            BaselineConfig,
            FEATURE_GROUPS,
            MahalanobisConfig,
            PriorsData,
            ProsodyConfig,
            SpeakerAssignment,
            TimingFeatures,
            TurnFeatures,
            TurnTranscript,
            VisionFeatures,
            WordTimestamp,
        )

        path = Path(media_path)
        audio, sr, video_frames = _load_media_audio_and_vision(path)
        sid = session_id or f"dce-{uuid.uuid4().hex[:12]}"

        # Load priors correlation matrix (same logic as DiscourseConsistencyPipeline)
        feature_order = list(FEATURE_GROUPS.keys())
        corr_matrix = np.eye(len(feature_order), dtype=np.float64)
        priors_candidates = [
            Path("data/priors_en.json"),
            Path(__file__).resolve().parent.parent.parent / "data" / "priors_en.json",
        ]
        env_priors = os.environ.get("DCE_PRIORS_PATH")
        if env_priors:
            priors_candidates.insert(0, Path(env_priors))
        for p_cand in priors_candidates:
            if p_cand.is_file():
                try:
                    pdata = PriorsData.model_validate(json.loads(p_cand.read_text(encoding="utf-8")))
                    src_order = list(pdata.feature_order)
                    raw_corr = np.asarray(pdata.correlation_matrix, dtype=np.float64)
                    if src_order and raw_corr.shape == (len(src_order), len(src_order)):
                        idx_map = {name: idx for idx, name in enumerate(src_order)}
                        mapped = np.eye(len(feature_order), dtype=np.float64)
                        for i, fi in enumerate(feature_order):
                            for j, fj in enumerate(feature_order):
                                if fi in idx_map and fj in idx_map:
                                    mapped[i, j] = raw_corr[idx_map[fi], idx_map[fj]]
                        corr_matrix = mapped
                    break
                except Exception as exc:
                    logger.debug("Priors correlation load skipped: %s", exc)

        resolved_priors = priors
        if resolved_priors is None and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
            wild_priors_path = Path(__file__).resolve().parents[2] / "data" / "priors_wild_video_en.json"
            if wild_priors_path.is_file():
                resolved_priors = wild_priors_path

        baseline_cfg = BaselineConfig(
            decoupled_shrinkage=True,
            scale_shrinkage_k=6.0,
            l2_location_shrinkage_k=float(shrinkage_k),
            acoustic_location_shrinkage_k=3.0,
        )
        baseline_store = BaselineStore(config=baseline_cfg, priors=resolved_priors)
        # Expand feature_order & corr_matrix if resolved_priors includes eGeMAPS / AU / gaze features
        extra_feats = [k for k in baseline_store.priors if k not in feature_order]
        if extra_feats:
            old_n = len(feature_order)
            feature_order = feature_order + extra_feats
            new_n = len(feature_order)
            expanded_corr = np.eye(new_n, dtype=np.float64)
            expanded_corr[:old_n, :old_n] = corr_matrix
            corr_matrix = expanded_corr
        maha_cfg = MahalanobisConfig()
        prosody_cfg = ProsodyConfig()

        # Optional vision summary across sampled 15 FPS video frames (fresh RunningMode.VIDEO per file)
        def _aggregate_vision_slice(seq_slice: Sequence[VisionFeatures]) -> VisionFeatures:
            valid_vis = [v for v in seq_slice if v.face_detected and v.quality_flag == "ok"]
            blink_rates = [v.blink_rate_per_min for v in valid_vis if v.blink_rate_per_min is not None]
            lip_apts = [v.mean_lip_aperture for v in valid_vis if v.mean_lip_aperture is not None]
            yaws = [v.head_pose_yaw_deg for v in valid_vis if v.head_pose_yaw_deg is not None]
            pitches = [v.head_pose_pitch_deg for v in valid_vis if v.head_pose_pitch_deg is not None]
            rolls = [v.head_pose_roll_deg for v in valid_vis if v.head_pose_roll_deg is not None]
            gaze_hs = [
                v.eye_gaze_horizontal
                for v in valid_vis
                if v.eye_gaze_horizontal is not None
                and (v.head_pose_yaw_deg is None or abs(v.head_pose_yaw_deg) <= 20.0)
            ]
            gaze_vs = [
                v.eye_gaze_vertical
                for v in valid_vis
                if v.eye_gaze_vertical is not None
                and (v.head_pose_yaw_deg is None or abs(v.head_pose_yaw_deg) <= 20.0)
            ]
            frontal_vis = [
                v
                for v in valid_vis
                if v.head_pose_yaw_deg is None or abs(v.head_pose_yaw_deg) <= 20.0
            ]
            agg_aus: Dict[str, float] = {}
            for au_key in (
                "AU01_inner_brow_raise",
                "AU04_brow_lowerer",
                "AU12_lip_corner_puller",
                "AU25_lips_part",
            ):
                au_vals = [v.action_units[au_key] for v in frontal_vis if au_key in v.action_units]
                if au_vals:
                    agg_aus[au_key] = float(np.mean(au_vals))
            if yaws or lip_apts or blink_rates or agg_aus or gaze_hs:
                return VisionFeatures(
                    face_detected=True,
                    blink_rate_per_min=float(np.mean(blink_rates)) if blink_rates else None,
                    mean_lip_aperture=float(np.mean(lip_apts)) if lip_apts else None,
                    head_pose_yaw_deg=float(np.mean(yaws)) if yaws else None,
                    head_pose_pitch_deg=float(np.mean(pitches)) if pitches else None,
                    head_pose_roll_deg=float(np.mean(rolls)) if rolls else None,
                    eye_gaze_horizontal=float(np.mean(gaze_hs)) if gaze_hs else None,
                    eye_gaze_vertical=float(np.mean(gaze_vs)) if gaze_vs else None,
                    action_units=agg_aus,
                    quality_flag="ok",
                )
            if any(v.quality_flag == "extreme_head_pose" for v in seq_slice):
                return VisionFeatures(face_detected=True, quality_flag="extreme_head_pose")
            return VisionFeatures(face_detected=False, quality_flag="no_face")

        seq_results: List[VisionFeatures] = []
        vision_feats = VisionFeatures(face_detected=False, quality_flag="no_video")
        if video_frames:
            seq_results = extract_vision_sequence(video_frames, fps=15.0)
            vision_feats = _aggregate_vision_slice(seq_results)

        # Build turn boundaries
        turn_specs: List[Dict[str, Any]] = []
        if transcript_override:
            n_items = len(transcript_override)
            total_dur_sec = len(audio) / float(sr) if len(audio) > 0 else 1.0
            for idx, item in enumerate(transcript_override):
                tid = int(item.get("turn_id", idx + 1))
                s_sec = float(item.get("start_sec", (idx / max(1, n_items)) * total_dur_sec))
                e_sec = float(item.get("end_sec", ((idx + 1) / max(1, n_items)) * total_dur_sec))
                s_idx = max(0, min(len(audio), int(round(s_sec * sr))))
                e_idx = max(s_idx, min(len(audio), int(round(e_sec * sr))))
                turn_specs.append(
                    {
                        "turn_id": tid,
                        "speaker_id": str(item.get("speaker_id", "speaker_0")),
                        "start_sample": s_idx,
                        "end_sample": e_idx,
                        "forced_split": bool(item.get("forced_split", False)),
                        "overlap_flag": bool(item.get("overlap_flag", False)),
                        "override": item,
                    }
                )
        else:
            raw_segs = _segment_audio_energy(audio, sample_rate=sr)
            if not raw_segs:
                raw_segs = [(0, len(audio), False)]
            pre_specs: List[Dict[str, Any]] = []
            for s_idx, e_idx, f_split in raw_segs:
                seg_audio = audio[s_idx:e_idx].copy() if e_idx > s_idx else np.zeros(0, dtype=np.float32)
                if use_asr and len(seg_audio) > 0:
                    asr_res = transcribe_audio(
                        seg_audio,
                        sample_rate=sr,
                        model_size=asr_model,
                        device=asr_device,
                    )
                    seg_text = asr_res.text.strip()
                    seg_words = list(asr_res.words)
                else:
                    seg_text = ""
                    seg_words = []

                # Split Q&A boundaries at '?' when followed by additional words
                split_done = False
                if seg_words and "?" in seg_text:
                    q_word_indices = [
                        wi for wi, wobj in enumerate(seg_words)
                        if "?" in str(wobj.word) and wi < len(seg_words) - 1
                    ]
                    if q_word_indices:
                        split_done = True
                        prev_wi = 0
                        prev_sample = s_idx
                        for qi in q_word_indices:
                            q_end_rel = float(seg_words[qi].end)
                            next_start_rel = float(seg_words[qi + 1].start)
                            sub_words = [
                                WordTimestamp(
                                    word=w.word,
                                    start=max(0.0, float(w.start) - (prev_sample - s_idx) / float(sr)),
                                    end=max(0.0, float(w.end) - (prev_sample - s_idx) / float(sr)),
                                    probability=w.probability,
                                )
                                for w in seg_words[prev_wi : qi + 1]
                            ]
                            q_end_sample = min(e_idx, max(prev_sample, s_idx + int(round(q_end_rel * sr))))
                            pre_specs.append(
                                {
                                    "start_sample": prev_sample,
                                    "end_sample": q_end_sample,
                                    "forced_split": False,
                                    "overlap_flag": False,
                                    "pre_text": " ".join(w.word.strip() for w in sub_words).strip(),
                                    "pre_words": sub_words,
                                }
                            )
                            prev_wi = qi + 1
                            prev_sample = min(e_idx, max(q_end_sample, s_idx + int(round(next_start_rel * sr))))
                        # Remainder after last '?'
                        rem_words = [
                            WordTimestamp(
                                word=w.word,
                                start=max(0.0, float(w.start) - (prev_sample - s_idx) / float(sr)),
                                end=max(0.0, float(w.end) - (prev_sample - s_idx) / float(sr)),
                                probability=w.probability,
                            )
                            for w in seg_words[prev_wi:]
                        ]
                        if rem_words and e_idx > prev_sample:
                            pre_specs.append(
                                {
                                    "start_sample": prev_sample,
                                    "end_sample": e_idx,
                                    "forced_split": f_split,
                                    "overlap_flag": False,
                                    "pre_text": " ".join(w.word.strip() for w in rem_words).strip(),
                                    "pre_words": rem_words,
                                }
                            )

                if not split_done:
                    pre_specs.append(
                        {
                            "start_sample": s_idx,
                            "end_sample": e_idx,
                            "forced_split": f_split,
                            "overlap_flag": False,
                            "pre_text": seg_text,
                            "pre_words": seg_words,
                        }
                    )

            # Lightweight Q&A / Pitch Speaker Diarization
            turn_f0s: List[Optional[float]] = []
            for ps in pre_specs:
                sub_a = audio[ps["start_sample"] : ps["end_sample"]]
                if len(sub_a) >= int(0.3 * sr):
                    p_quick = extract_prosody(sub_a, sample_rate=sr, config=prosody_cfg)
                    turn_f0s.append(p_quick.f0_median_hz)
                else:
                    turn_f0s.append(None)

            # Determine witness reference pitch from non-question turns (or all turns)
            ans_f0s = [
                f0 for ps, f0 in zip(pre_specs, turn_f0s)
                if f0 is not None and not ps["pre_text"].strip().endswith("?")
            ]
            ref_f0 = float(np.median(ans_f0s)) if ans_f0s else None

            for idx, ps in enumerate(pre_specs):
                t_txt = ps["pre_text"].strip()
                is_q = t_txt.endswith("?")
                has_next_ans = (idx + 1 < len(pre_specs)) and not pre_specs[idx + 1]["pre_text"].strip().endswith("?")
                f0_val = turn_f0s[idx]
                pitch_diff_st = (
                    abs(12.0 * math.log2(f0_val / ref_f0))
                    if (f0_val is not None and ref_f0 is not None and ref_f0 > 0.0 and f0_val > 0.0)
                    else 0.0
                )
                if (is_q and has_next_ans) or (is_q and pitch_diff_st >= 3.5):
                    spk_id = "Q_ASKER"
                elif not is_q and pitch_diff_st >= 4.5 and len(pre_specs) >= 3:
                    spk_id = "speaker_1"
                else:
                    spk_id = "speaker_0"
                turn_specs.append(
                    {
                        "turn_id": idx + 1,
                        "speaker_id": spk_id,
                        "start_sample": ps["start_sample"],
                        "end_sample": ps["end_sample"],
                        "forced_split": ps["forced_split"],
                        "overlap_flag": ps["overlap_flag"],
                        "override": None,
                        "pre_text": t_txt,
                        "pre_words": ps["pre_words"],
                    }
                )

        stored_turns: Dict[int, _StoredTurn] = {}
        turn_order: List[int] = []
        firewall_turns_out: List[Dict[str, Any]] = []
        _closed_q_prefixes = {
            "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
            "can", "could", "will", "would", "should", "may", "might", "shall",
        }
        prev_q_end_sec: Optional[float] = None
        prev_q_is_closed: bool = False

        for spec in turn_specs:
            tid = spec["turn_id"]
            speaker_id = spec["speaker_id"]
            s_idx = spec["start_sample"]
            e_idx = spec["end_sample"]
            start_ns = int(round((s_idx / float(sr)) * 1e9))
            end_ns = int(round((e_idx / float(sr)) * 1e9))
            turn_audio = audio[s_idx:e_idx].copy() if e_idx > s_idx else np.zeros(0, dtype=np.float32)
            override = spec["override"]

            word_objs: List[WordTimestamp] = []
            word_alts: Dict[str, str] = {}
            if override is not None:
                raw_text = str(override.get("text", "")).strip()
                for w in override.get("words", []):
                    w_text = str(w.get("word", "")).strip()
                    if not w_text:
                        continue
                    word_objs.append(
                        WordTimestamp(
                            word=w_text,
                            start=float(w.get("start", 0.0)),
                            end=float(w.get("end", 0.0)),
                            probability=(
                                float(w["probability"]) if w.get("probability") is not None else None
                            ),
                        )
                    )
                    if w.get("alternative"):
                        word_alts[w_text.lower()] = str(w["alternative"])
            elif "pre_text" in spec:
                raw_text = str(spec["pre_text"]).strip()
                word_objs = list(spec.get("pre_words") or [])
            elif use_asr and len(turn_audio) > 0:
                asr_res = transcribe_audio(
                    turn_audio,
                    sample_rate=sr,
                    model_size=asr_model,
                    device=asr_device,
                )
                raw_text = asr_res.text.strip()
                word_objs = list(asr_res.words)
            else:
                raw_text = ""

            annotated_text, low_conf_words = _annotate_transcript(
                raw_text=raw_text,
                words=word_objs,
                low_conf_threshold=low_conf_threshold,
                word_alternatives=word_alts,
            )
            incomplete_flag = (
                bool(override["incomplete_sentence"])
                if (override is not None and "incomplete_sentence" in override)
                else _is_incomplete_sentence(raw_text, spec["forced_split"])
            )

            # Extract acoustic prosody + spectral features (locked in backend)
            word_count = len(re.findall(r"[a-zA-Z0-9']+", raw_text))
            prosody_feats = extract_prosody(
                turn_audio,
                sample_rate=sr,
                config=prosody_cfg,
            )
            egemaps = extract_egemaps_spectral_features(turn_audio, sample_rate=sr)
            if prosody_feats.acoustic_snr_db is None:
                prosody_feats.acoustic_snr_db = egemaps.get("acoustic_snr_db")

            # Compute pre-word hesitation pauses (> 400 ms) from WordTimestamp gaps
            word_gaps_ms = [
                max(0.0, (float(word_objs[wi].start) - float(word_objs[wi - 1].end)) * 1000.0)
                for wi in range(1, len(word_objs))
            ]
            hesitation_gaps_ms = [g for g in word_gaps_ms if g > 400.0]
            max_pre_word_pause_ms = max(word_gaps_ms) if word_gaps_ms else None
            hesitation_pause_count = len(hesitation_gaps_ms)

            # Compute Q&A response latency if this turn answers a preceding question
            resp_lat_open: Optional[float] = None
            resp_lat_closed: Optional[float] = None
            curr_turn_start_sec = s_idx / float(sr)
            curr_turn_end_sec = e_idx / float(sr)
            if prev_q_end_sec is not None and speaker_id != "Q_ASKER" and not raw_text.endswith("?"):
                ans_onset_sec = curr_turn_start_sec + (float(word_objs[0].start) if word_objs else 0.0)
                gap_sec = max(0.0, ans_onset_sec - prev_q_end_sec)
                if prev_q_is_closed:
                    resp_lat_closed = gap_sec
                else:
                    resp_lat_open = gap_sec
                prev_q_end_sec = None

            if raw_text.endswith("?") or speaker_id == "Q_ASKER":
                prev_q_end_sec = curr_turn_start_sec + (float(word_objs[-1].end) if word_objs else (curr_turn_end_sec - curr_turn_start_sec))
                first_tok = re.findall(r"[a-zA-Z']+", raw_text.lower())
                prev_q_is_closed = bool(first_tok and first_tok[0] in _closed_q_prefixes)

            turn_dur_ms = (end_ns - start_ns) / 1e6
            total_hes_ms = float(sum(hesitation_gaps_ms))
            timing_feats = TimingFeatures(
                response_latency_open=resp_lat_open,
                response_latency_closed=resp_lat_closed,
                turn_duration_ms=turn_dur_ms,
                speech_duration_ms=max(0.0, turn_dur_ms - total_hes_ms),
                total_pause_ms=total_hes_ms,
                pause_count=hesitation_pause_count,
                max_pre_word_pause_ms=max_pre_word_pause_ms,
                hesitation_pause_count=hesitation_pause_count,
            )
            turn_vision_feats = vision_feats
            if seq_results:
                f_start = max(0, int(round((s_idx / float(sr)) * 15.0)))
                f_end = min(len(seq_results), max(f_start + 1, int(round((e_idx / float(sr)) * 15.0))))
                turn_slice = seq_results[f_start:f_end]
                if sum(1 for v in turn_slice if v.face_detected and v.quality_flag == "ok") >= 3:
                    turn_vision_feats = _aggregate_vision_slice(turn_slice)

            turn_features = TurnFeatures(
                turn_id=tid,
                speaker_id=speaker_id,
                start_ns=start_ns,
                end_ns=end_ns,
                timing=timing_feats,
                prosody=prosody_feats,
                vision=turn_vision_feats,
            )

            # Score & update BaselineStore (k_mu = shrinkage_k, default 0.0)
            feature_scores = baseline_store.score_and_update_turn(
                speaker_id=speaker_id,
                turn_features=turn_features,
                word_count=word_count if word_count > 0 else None,
            )
            # One-sided cognitive load clamping:
            # - f0_semitone_sd (stress_polarity == -1): emotional expressiveness (z > 0) does not inflate d_sq.
            # - response_latency_open/closed (stress_polarity == +1): immediate answers (z < 0) do not inflate d_sq.
            maha_scores = dict(feature_scores)
            f0_prior = baseline_store.priors.get("f0_semitone_sd")
            f0_fs = maha_scores.get("f0_semitone_sd")
            if (
                f0_prior is not None
                and f0_prior.stress_polarity == -1
                and f0_fs is not None
                and f0_fs.z_score is not None
                and f0_fs.z_score > 0.0
            ):
                maha_scores["f0_semitone_sd"] = f0_fs.model_copy(update={"z_score": 0.0})
            for lat_feat in ("response_latency_open", "response_latency_closed"):
                lat_prior = baseline_store.priors.get(lat_feat)
                lat_fs = maha_scores.get(lat_feat)
                if (
                    lat_prior is not None
                    and lat_prior.stress_polarity == 1
                    and lat_fs is not None
                    and lat_fs.z_score is not None
                    and lat_fs.z_score < 0.0
                ):
                    maha_scores[lat_feat] = lat_fs.model_copy(update={"z_score": 0.0})

            maha_res = compute_mahalanobis(
                feature_scores=maha_scores,
                correlation_matrix=corr_matrix,
                feature_order=feature_order,
                priors=baseline_store.priors,
                reg_lambda=maha_cfg.reg_lambda,
                eig_floor=maha_cfg.eig_floor,
                cond_limit=maha_cfg.cond_limit,
                tail_score_threshold=maha_cfg.tail_score_threshold,
                tension_z_threshold=maha_cfg.tension_z_threshold,
                hnr_tension_z_threshold=maha_cfg.hnr_tension_z_threshold,
                calm_z_threshold=maha_cfg.calm_z_threshold,
            )

            phys_expl = generate_physical_explanation(
                direction_label=maha_res.direction_label,
                evaluated_features=maha_res.evaluated_features,
                feature_z_scores=maha_res.feature_z_scores,
                vocal_tension_flag=maha_res.vocal_tension_flag,
            )

            transcript_obj = TurnTranscript(
                turn_id=tid,
                speaker_id=speaker_id,
                text=raw_text,
                start_ns=start_ns,
                end_ns=end_ns,
                is_question=raw_text.endswith("?"),
                words=word_objs,
            )
            speaker_assign = SpeakerAssignment(speaker_id=speaker_id, similarity=1.0)

            stored_turns[tid] = _StoredTurn(
                turn_id=tid,
                speaker_id=speaker_id,
                start_ns=start_ns,
                end_ns=end_ns,
                audio_pcm16k=turn_audio,
                raw_text=raw_text,
                annotated_text=annotated_text,
                low_confidence_words=low_conf_words,
                incomplete_sentence=incomplete_flag,
                forced_split=spec["forced_split"],
                overlap_flag=spec["overlap_flag"],
                transcript_obj=transcript_obj,
                features_obj=turn_features,
                mahalanobis_obj=maha_res,
                physical_explanation=phys_expl,
                speaker_assignment_obj=speaker_assign,
            )
            turn_order.append(tid)

            firewall_turns_out.append(
                {
                    "turn_id": tid,
                    "speaker_id": speaker_id,
                    "start_sec": round(s_idx / float(sr), 3),
                    "end_sec": round(e_idx / float(sr), 3),
                    "annotated_text": annotated_text,
                    "raw_text": raw_text,
                    "incomplete_sentence": incomplete_flag,
                    "forced_split": spec["forced_split"],
                    "overlap_flag": spec["overlap_flag"],
                    "low_confidence_words": low_conf_words,
                }
            )

        with self._lock:
            self._sessions[sid] = _DCESession(
                session_id=sid,
                state="INGESTED",
                media_path=str(path),
                sample_rate=sr,
                shrinkage_k=float(shrinkage_k),
                turns=stored_turns,
                turn_order=turn_order,
            )

        response = {
            "session_id": sid,
            "status": "INGESTED",
            "turn_count": len(turn_order),
            "turns": firewall_turns_out,
            "next_step": (
                "Analyze the annotated transcripts for factual claims and contradictions, "
                "then call dce_submit_verdicts."
            ),
        }
        # Hard Neuro-Symbolic Firewall invariant check on response keys and metadata
        # (exclude spoken transcript text so natural speech containing words like
        # "stress", "blink", or "anomaly" does not trigger a false firewall block).
        sanitized_for_firewall = {
            **response,
            "turns": [
                {
                    k: v
                    for k, v in t.items()
                    if k not in ("annotated_text", "raw_text", "low_confidence_words")
                }
                for t in firewall_turns_out
            ],
        }
        assert_firewall_compliance(json.dumps(sanitized_for_firewall, ensure_ascii=False))
        return response

    def submit_verdicts(
        self,
        session_id: str,
        verdicts: List[Dict[str, Any]],
        claims: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Step 2: Validate turn_ids, downgrade phonetic-fragile conflicts, lock verdicts."""
        from dce.reasoning.claims_ledger import (
            _PHONETIC_FRAGILE_SET,
            is_phonetic_fragile_difference,
        )
        from dce.schemas import (
            ClaimItem,
            ConsistencyVerificationResponse,
            VerdictStatus,
        )

        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                raise KeyError(f"Unknown session_id: {session_id}")

            parsed_claims: List[ClaimItem] = []
            claims_by_id: Dict[str, ClaimItem] = {}
            for idx, c in enumerate(claims or []):
                c_dict = dict(c)
                c_dict.setdefault("claim_id", f"c_{idx + 1}")
                c_dict.setdefault("speaker_id", "speaker_0")
                c_dict.setdefault("raw_subject", "speaker")
                c_dict.setdefault("canonical_entity", "entity")
                c_dict.setdefault("predicate", "stated")
                c_dict.setdefault("object_value", "")
                ci = ClaimItem.model_validate(c_dict)
                parsed_claims.append(ci)
                claims_by_id[ci.claim_id] = ci

            validated_out: List[Dict[str, Any]] = []
            validated_objs: List[ConsistencyVerificationResponse] = []
            downgraded_count = 0
            rejected_count = 0

            for v in verdicts:
                raw_verdict = str(v.get("verdict", "UNVERIFIABLE"))
                src_tid = int(v["source_turn_id"])
                prior_tid = v.get("prior_turn_id")
                prior_tid_int = int(prior_tid) if prior_tid is not None else None
                src_cid = v.get("source_claim_id")
                prior_cid = v.get("prior_claim_id")
                explanation = str(v.get("explanation", ""))
                confidence = float(v.get("confidence", 1.0))

                # Validate turn_id references against real session turns
                src_valid = src_tid in sess.turns
                prior_valid = prior_tid_int is None or (prior_tid_int in sess.turns)
                if not src_valid or not prior_valid:
                    rejected_count += 1
                    resp_obj = ConsistencyVerificationResponse(
                        verdict=VerdictStatus.REJECTED_INVALID_REF,
                        source_turn_id=src_tid,
                        prior_turn_id=prior_tid_int,
                        source_claim_id=src_cid,
                        prior_claim_id=prior_cid,
                        explanation=f"Invalid turn_id reference (source={src_tid}, prior={prior_tid_int}). {explanation}".strip(),
                        confidence=0.0,
                    )
                    validated_objs.append(resp_obj)
                    validated_out.append(
                        {
                            **resp_obj.model_dump(mode="json"),
                            "valid_ref": False,
                            "review_status": "REJECTED_INVALID_REF",
                        }
                    )
                    continue

                # Phonetic-fragile / low-confidence ASR collision check for contradictions
                is_conflict = raw_verdict in {
                    VerdictStatus.DIRECT_CONTRADICTION.value,
                    VerdictStatus.TIMELINE_SHIFT.value,
                }
                downgrade_reason: Optional[str] = None
                if is_conflict:
                    src_claim = claims_by_id.get(src_cid) if src_cid else None
                    prior_claim = claims_by_id.get(prior_cid) if prior_cid else None
                    if src_claim is not None and prior_claim is not None:
                        if is_phonetic_fragile_difference(src_claim, prior_claim):
                            downgrade_reason = "phonetic_fragile_claim_pair"

                    src_turn = sess.turns[src_tid]
                    prior_turn = sess.turns[prior_tid_int] if prior_tid_int is not None else None

                    if downgrade_reason is None and prior_turn is not None:
                        toks_a = set(_extract_turn_tokens(src_turn.raw_text))
                        toks_b = set(_extract_turn_tokens(prior_turn.raw_text))
                        only_a = toks_a - toks_b
                        only_b = toks_b - toks_a
                        for ta in only_a:
                            for tb in only_b:
                                if frozenset((ta, tb)) in _PHONETIC_FRAGILE_SET:
                                    downgrade_reason = f"phonetic_fragile_turn_tokens:{ta}/{tb}"
                                    break
                            if downgrade_reason:
                                break

                    # Also downgrade if any low-confidence word in involved turns has a phonetic alt
                    if downgrade_reason is None:
                        involved_low_words = list(src_turn.low_confidence_words)
                        if prior_turn is not None:
                            involved_low_words.extend(prior_turn.low_confidence_words)
                        for lw in involved_low_words:
                            if lw.get("alternative"):
                                downgrade_reason = f"low_confidence_asr_word:{lw.get('word')}"
                                break

                if downgrade_reason is not None:
                    downgraded_count += 1
                    resp_obj = ConsistencyVerificationResponse(
                        verdict=VerdictStatus.NEEDS_AUDIO_REVIEW,
                        source_turn_id=src_tid,
                        prior_turn_id=prior_tid_int,
                        source_claim_id=src_cid,
                        prior_claim_id=prior_cid,
                        explanation=f"[Downgraded from {raw_verdict}: {downgrade_reason}] {explanation}".strip(),
                        confidence=min(confidence, 0.5),
                    )
                    validated_objs.append(resp_obj)
                    validated_out.append(
                        {
                            **resp_obj.model_dump(mode="json"),
                            "valid_ref": True,
                            "review_status": "NEEDS_HUMAN_REVIEW",
                            "downgraded_from": raw_verdict,
                            "downgrade_reason": downgrade_reason,
                        }
                    )
                else:
                    normalized_verdict = (
                        VerdictStatus.NEEDS_AUDIO_REVIEW.value
                        if raw_verdict == "NEEDS_HUMAN_REVIEW"
                        else raw_verdict
                    )
                    verdict_enum = VerdictStatus(normalized_verdict)
                    resp_obj = ConsistencyVerificationResponse(
                        verdict=verdict_enum,
                        source_turn_id=src_tid,
                        prior_turn_id=prior_tid_int,
                        source_claim_id=src_cid,
                        prior_claim_id=prior_cid,
                        explanation=explanation,
                        confidence=confidence,
                    )
                    validated_objs.append(resp_obj)
                    validated_out.append(
                        {
                            **resp_obj.model_dump(mode="json"),
                            "valid_ref": True,
                            "review_status": verdict_enum.value,
                        }
                    )

            sess.submitted_claims = parsed_claims
            sess.locked_verdicts = validated_out
            sess.validated_verdict_objs = validated_objs
            sess.state = "VERDICTS_LOCKED"

        return {
            "session_id": session_id,
            "status": "VERDICTS_LOCKED",
            "verdicts_locked": True,
            "validated_verdicts": validated_out,
            "downgraded_count": downgraded_count,
            "rejected_invalid_ref_count": rejected_count,
            "next_step": "Call dce_unlock_signals to unlock mathematical/acoustic signals and save replay .jsonl.",
        }

    def unlock_signals(
        self,
        session_id: str,
        output_jsonl_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step 3: Unlock Mahalanobis/BaselineStore/templates.py signals & write .jsonl replay."""
        from dce.recorder import FeatureRecorder
        from dce.schemas import RecorderTurnRecord

        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            if sess.state == "INGESTED":
                raise PermissionError(
                    "FIREWALL_LOCKED: Text verdicts must be submitted and locked via "
                    "dce_submit_verdicts before acoustic/facial signals can be unlocked."
                )

            if output_jsonl_path:
                jsonl_path = Path(output_jsonl_path)
            else:
                media_p = Path(sess.media_path)
                jsonl_path = media_p.parent / f"{media_p.stem}_{sess.session_id}_replay.jsonl"

            recorder = FeatureRecorder(jsonl_path)
            unlocked_turns: List[Dict[str, Any]] = []
            try:
                for tid in sess.turn_order:
                    turn = sess.turns[tid]
                    turn_claims = [c for c in sess.submitted_claims if c.turn_id == tid]
                    turn_verdicts = [
                        v for v in sess.validated_verdict_objs if v.source_turn_id == tid
                    ]
                    record = RecorderTurnRecord(
                        turn_id=tid,
                        speaker_assignment=turn.speaker_assignment_obj,
                        transcript=turn.transcript_obj,
                        features=turn.features_obj,
                        mahalanobis=turn.mahalanobis_obj,
                        claims=turn_claims,
                        verifications=turn_verdicts,
                        verification=turn_verdicts[0] if turn_verdicts else None,
                    )
                    recorder.write_record(record)
                    unlocked_turns.append(
                        {
                            "turn_id": tid,
                            "speaker_id": turn.speaker_id,
                            "annotated_text": turn.annotated_text,
                            "mahalanobis": turn.mahalanobis_obj.model_dump(mode="json"),
                            "physical_explanation": turn.physical_explanation,
                            "prosody": turn.features_obj.prosody.model_dump(mode="json"),
                            "vision": turn.features_obj.vision.model_dump(mode="json"),
                            "timing": turn.features_obj.timing.model_dump(mode="json"),
                            "verifications": [v.model_dump(mode="json") for v in turn_verdicts],
                        }
                    )
            finally:
                recorder.close()

            sess.replay_jsonl_path = str(jsonl_path)
            sess.state = "SIGNALS_UNLOCKED"

            return {
                "session_id": session_id,
                "status": "SIGNALS_UNLOCKED",
                "shrinkage_k_used": sess.shrinkage_k,
                "replay_jsonl_path": str(jsonl_path),
                "turns": unlocked_turns,
                "locked_verdicts": sess.locked_verdicts,
            }

    def slice_turn_audio(
        self,
        session_id: str,
        turn_id: int,
        duration_sec: float = 3.0,
        start_offset_sec: float = 0.0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step 4: Slice a short WAV clip (default 3.0s) for a target turn_id."""
        import numpy as np

        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            turn = sess.turns.get(int(turn_id))
            if turn is None:
                raise KeyError(f"Unknown turn_id {turn_id} in session {session_id}")

            sr = sess.sample_rate
            pcm = turn.audio_pcm16k
            start_idx = max(0, int(round(max(0.0, float(start_offset_sec)) * sr)))
            max_samples = max(1, int(round(max(0.1, float(duration_sec)) * sr)))
            sliced = pcm[start_idx : start_idx + max_samples]

            target_dir = (
                Path(output_dir)
                if output_dir
                else (Path(sess.media_path).parent / "dce_audio_slices" / sess.session_id)
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            clip_path = target_dir / f"turn_{int(turn_id):04d}.wav"

            pcm_i16 = (np.clip(sliced, -1.0, 1.0) * 32767.0).astype(np.int16)
            with wave.open(str(clip_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sr))
                wf.writeframes(pcm_i16.tobytes())

            actual_dur = round(len(sliced) / float(sr), 3) if sr > 0 else 0.0
            resolved = clip_path.resolve()
            return {
                "session_id": session_id,
                "turn_id": int(turn_id),
                "wav_path": str(resolved),
                "file_uri": resolved.as_uri(),
                "duration_sec": actual_dur,
                "sample_rate": sr,
            }


_DEFAULT_STORE = DCESessionStore()


def handle_jsonrpc_request(
    msg: Dict[str, Any],
    store: Optional[DCESessionStore] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 MCP request dict and return a response dict (or None for notifications)."""
    active_store = store or _DEFAULT_STORE
    req_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    # Notifications have no id and require no response
    if req_id is None:
        return None

    if method == "initialize":
        client_proto = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_proto,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments") or {}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if tool_name == "dce_ingest_media":
                    payload = active_store.ingest_media(**args)
                elif tool_name == "dce_submit_verdicts":
                    payload = active_store.submit_verdicts(**args)
                elif tool_name == "dce_unlock_signals":
                    payload = active_store.unlock_signals(**args)
                elif tool_name == "dce_slice_turn_audio":
                    payload = active_store.slice_turn_audio(**args)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:
            logger.exception("Tool %s failed: %s", tool_name, exc)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(exc)}, ensure_ascii=False),
                        }
                    ],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _configure_stderr_logging() -> None:
    """Route all Python logging exclusively to sys.stderr."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def run_stdio_server(store: Optional[DCESessionStore] = None) -> None:
    """Run the Windows-safe JSON-RPC 2.0 MCP stdio loop."""
    _configure_stderr_logging()

    # Reserve fd 1 / original stdout exclusively for MCP JSON-RPC frames,
    # and redirect both Python sys.stdout AND C-level fd 1 to sys.stderr.
    mcp_binary_out: Any = None
    mcp_text_out: Any = None
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        mcp_fd = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
        mcp_binary_out = os.fdopen(mcp_fd, "wb", buffering=0)
    except Exception:
        # Fallback for in-memory StringIO tests without fileno()
        mcp_text_out = sys.stdout

    sys.stdout = sys.stderr

    def _write_frame(payload: Dict[str, Any], use_content_length: bool) -> None:
        raw_json = json.dumps(payload, ensure_ascii=False)
        if use_content_length:
            body_bytes = raw_json.encode("utf-8")
            header = f"Content-Length: {len(body_bytes)}\r\n\r\n".encode("ascii")
            frame_bytes = header + body_bytes
        else:
            frame_bytes = (raw_json + "\n").encode("utf-8")

        if mcp_binary_out is not None:
            mcp_binary_out.write(frame_bytes)
            mcp_binary_out.flush()
        elif mcp_text_out is not None:
            mcp_text_out.write(frame_bytes.decode("utf-8"))
            mcp_text_out.flush()

    stdin_stream = getattr(sys.stdin, "buffer", None)
    try:
        while True:
            if stdin_stream is not None:
                line_bytes = stdin_stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
            else:
                line = sys.stdin.readline()
                if not line:
                    break

            stripped = line.strip()
            if not stripped:
                continue

            use_cl = False
            if stripped.lower().startswith("content-length:"):
                use_cl = True
                try:
                    content_len = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    continue
                # Consume remaining header lines until blank line
                while True:
                    hdr = (
                        stdin_stream.readline().decode("utf-8", errors="replace")
                        if stdin_stream is not None
                        else sys.stdin.readline()
                    )
                    if not hdr or hdr.strip() == "":
                        break
                if stdin_stream is not None:
                    body = stdin_stream.read(content_len).decode("utf-8", errors="replace")
                else:
                    body = sys.stdin.read(content_len)
                raw_msg = body
            else:
                raw_msg = stripped

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError as exc:
                _write_frame(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc}"},
                    },
                    use_content_length=use_cl,
                )
                continue

            if isinstance(msg, dict):
                resp = handle_jsonrpc_request(msg, store=store)
                if resp is not None:
                    _write_frame(resp, use_content_length=use_cl)
    finally:
        if mcp_binary_out is not None:
            with contextlib.suppress(Exception):
                mcp_binary_out.close()


def main() -> None:
    run_stdio_server()


if __name__ == "__main__":
    main()
