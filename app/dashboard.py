"""Discourse Consistency Engine - Interactive Web Dashboard.

Provides visualization for conversational audio prosody, facial kinematics,
Mahalanobis anomaly metrics, Neuro-Symbolic claims consistency ledger, and
3-second timestamp-sliced audio playback from JSONL replay recordings.
Enforces lazy imports for streamlit and plotly inside render_dashboard().
"""

from __future__ import annotations

import io
import logging
import re
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from dce.recorder import ReplayLoader
from dce.schemas import RecorderTurnRecord, VerdictStatus

logger = logging.getLogger(__name__)

VERDICT_BADGES: Dict[str, str] = {
    VerdictStatus.CONSISTENT.value: "🟢 CONSISTENT",
    VerdictStatus.DIRECT_CONTRADICTION.value: "🔴 DIRECT_CONTRADICTION",
    VerdictStatus.TIMELINE_SHIFT.value: "🟠 TIMELINE_SHIFT",
    VerdictStatus.MODIFIER_EROSION.value: "🟡 MODIFIER_EROSION",
    VerdictStatus.EVASION.value: "🟡 EVASION",
    VerdictStatus.NEEDS_AUDIO_REVIEW.value: "⚪ NEEDS_AUDIO_REVIEW",
    VerdictStatus.UNVERIFIABLE.value: "⚪ UNVERIFIABLE",
    VerdictStatus.REJECTED_INVALID_REF.value: "⛔ REJECTED_INVALID_REF",
    VerdictStatus.LLM_ERROR.value: "⚠️ LLM_ERROR",
}


def load_replay_records(replay_path: Union[str, Path]) -> List[RecorderTurnRecord]:
    """Load RecorderTurnRecord entries from a JSONL replay file if it exists."""
    path = Path(replay_path)
    if not path.is_file():
        return []
    return ReplayLoader(path).load_all()


def slice_wav_clip_bytes(
    wav_path: Union[str, Path],
    start_sec: float = 0.0,
    duration_sec: float = 3.0,
) -> Optional[bytes]:
    """Read a WAV file and return an in-memory WAV clip sliced by timestamp."""
    path = Path(wav_path)
    if not path.is_file():
        return None

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        total_frames = wf.getnframes()

        if framerate <= 0 or total_frames <= 0:
            return None

        start_frame = max(0, int(round(max(0.0, start_sec) * framerate)))
        if start_frame >= total_frames:
            start_frame = 0
        n_frames = max(1, int(round(max(0.1, duration_sec) * framerate)))
        n_frames = min(n_frames, total_frames - start_frame)

        wf.setpos(start_frame)
        raw_frames = wf.readframes(n_frames)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out_wf:
        out_wf.setnchannels(n_channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(framerate)
        out_wf.writeframes(raw_frames)
    return buf.getvalue()


def find_target_word_offset_sec(
    record: RecorderTurnRecord,
    target_terms: Optional[Sequence[str]] = None,
) -> float:
    """Find a 3-second playback start offset (in seconds) centered near a key claim word."""
    if record.transcript is None or not record.transcript.words:
        return 0.0

    search_tokens = set()
    if target_terms:
        for term in target_terms:
            search_tokens.update(re.findall(r"[a-z0-9']+", term.lower()))
    else:
        for claim in record.claims:
            for field in (claim.object_value, claim.time_ref_raw, claim.canonical_entity):
                if field:
                    search_tokens.update(re.findall(r"[a-z0-9']+", field.lower()))

    if not search_tokens:
        return 0.0

    for w in record.transcript.words:
        w_toks = re.findall(r"[a-z0-9']+", w.word.lower())
        if any(tok in search_tokens for tok in w_toks):
            return max(0.0, float(w.start) - 1.0)
    return 0.0


def assess_modality_reliability(record: RecorderTurnRecord) -> Dict[str, Dict[str, Any]]:
    """Assess independent reliability status and fusion weight for Audio, Vision, and Text."""
    pros = record.features.prosody
    vis = record.features.vision
    transcript = record.transcript

    # 1. Audio modality
    if pros.quality_flag in ("no_audio_backend", "unvoiced", "insufficient_data", "extraction_error"):
        audio_info = {
            "status": f"Skipped ({pros.quality_flag})",
            "reliability": "Low",
            "weight": 0.0,
        }
    elif pros.acoustic_snr_db is not None and pros.acoustic_snr_db < 10.0:
        audio_info = {
            "status": f"Degraded (SNR {pros.acoustic_snr_db:.1f} dB)",
            "reliability": "Medium",
            "weight": 0.5,
        }
    else:
        audio_info = {"status": "OK", "reliability": "High", "weight": 1.0}

    # 2. Vision modality
    if vis.quality_flag == "low_light" or vis.low_light_flag:
        vision_info = {"status": "Skipped/Low-light", "reliability": "Low", "weight": 0.0}
    elif vis.quality_flag == "extreme_head_pose":
        vision_info = {"status": "Skipped (Head Pose > 30°)", "reliability": "Low", "weight": 0.0}
    elif not vis.face_detected or vis.quality_flag != "ok":
        vision_info = {
            "status": f"Skipped ({vis.quality_flag})",
            "reliability": "Low",
            "weight": 0.0,
        }
    else:
        vision_info = {"status": "OK", "reliability": "High", "weight": 1.0}

    # 3. Text / Ledger modality
    verifications = record.verifications or ([record.verification] if record.verification else [])
    has_llm_err = any(
        (v.verdict if hasattr(v, "verdict") else None)
        in (VerdictStatus.LLM_ERROR, VerdictStatus.REJECTED_INVALID_REF)
        for v in verifications
    )
    if transcript is None or not transcript.text.strip():
        text_info = {"status": "Empty", "reliability": "Low", "weight": 0.0}
    elif has_llm_err:
        text_info = {"status": "Degraded (LLM/Ref Error)", "reliability": "Low", "weight": 0.2}
    elif transcript.confidence is not None and transcript.confidence < 0.6:
        text_info = {
            "status": f"Low ASR Conf ({transcript.confidence:.2f})",
            "reliability": "Medium",
            "weight": 0.5,
        }
    else:
        text_info = {"status": "High", "reliability": "High", "weight": 1.0}

    return {"audio": audio_info, "vision": vision_info, "text": text_info}


def compute_review_priority(record: RecorderTurnRecord) -> Dict[str, Any]:
    """Compute max-dominant Review Priority without diluting text contradictions with calm audio/vision."""
    rel = assess_modality_reliability(record)
    mah = record.mahalanobis

    # Audio risk score from voice z-scores or Mahalanobis tail score
    audio_score: Optional[float] = None
    if rel["audio"]["weight"] > 0.0 and mah is not None:
        voice_z = [
            abs(mah.feature_z_scores[k])
            for k in ("f0_semitone_sd", "jitter_local_pct", "shimmer_local_pct", "hnr_db")
            if k in mah.feature_z_scores
        ]
        if voice_z:
            audio_score = min(1.0, max(voice_z) / 3.0)
        else:
            audio_score = min(1.0, max(0.0, 1.0 - mah.chi2_tail_score))

    # Vision risk score from blink_rate_per_min z-score
    vision_score: Optional[float] = None
    if rel["vision"]["weight"] > 0.0 and mah is not None:
        if "blink_rate_per_min" in mah.feature_z_scores:
            vision_score = min(1.0, abs(mah.feature_z_scores["blink_rate_per_min"]) / 3.0)

    # Text risk score from ClaimsLedger verdicts
    text_score: Optional[float] = None
    if rel["text"]["weight"] > 0.0:
        verifications = record.verifications or ([record.verification] if record.verification else [])
        if any(
            v.verdict in (
                VerdictStatus.DIRECT_CONTRADICTION,
                VerdictStatus.TIMELINE_SHIFT,
                VerdictStatus.MODIFIER_EROSION,
                VerdictStatus.EVASION,
            )
            for v in verifications
        ):
            text_score = 1.0
        elif any(v.verdict == VerdictStatus.NEEDS_AUDIO_REVIEW for v in verifications):
            text_score = 0.5
        else:
            text_score = 0.0

    active_modalities: List[str] = []
    skipped_modalities: List[str] = []
    for mod_name, mod_score in (("audio", audio_score), ("vision", vision_score), ("text", text_score)):
        if float(rel[mod_name]["weight"]) > 0.0 and mod_score is not None:
            active_modalities.append(mod_name)
        else:
            skipped_modalities.append(mod_name)

    multimodal_anomaly = max(
        [s for s in (audio_score, vision_score) if s is not None],
        default=0.0,
    )
    review_priority = max(text_score if text_score is not None else 0.0, multimodal_anomaly)
    return {
        "review_priority": round(review_priority, 4),
        "fused_risk_score": round(review_priority, 4),
        "multimodal_anomaly": round(multimodal_anomaly, 4),
        "active_modalities": active_modalities,
        "skipped_modalities": skipped_modalities,
        "modality_scores": {
            "audio": audio_score,
            "vision": vision_score,
            "text": text_score,
        },
        "reliability": rel,
    }


def compute_late_fusion_score(record: RecorderTurnRecord) -> Dict[str, Any]:
    """Backwards-compatible alias for compute_review_priority()."""
    return compute_review_priority(record)


def build_ledger_rows(records: List[RecorderTurnRecord]) -> List[Dict[str, Any]]:
    """Build flat display rows for the Discourse Consistency Ledger table."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        verifications = rec.verifications or ([rec.verification] if rec.verification else [])
        claim_text = "; ".join(
            f"{c.canonical_entity} {c.predicate} {c.object_value}".strip()
            for c in rec.claims
        )
        transcript_text = rec.transcript.text if rec.transcript is not None else ""

        if not verifications:
            rows.append(
                {
                    "Turn": rec.turn_id,
                    "Speaker": rec.speaker_assignment.speaker_id,
                    "Start (s)": round(rec.features.start_ns / 1_000_000_000.0, 2),
                    "Transcript": transcript_text,
                    "Claims": claim_text or "—",
                    "Verdict": "—",
                    "Explanation": "",
                }
            )
            continue

        for ver in verifications:
            v_str = ver.verdict.value if hasattr(ver.verdict, "value") else str(ver.verdict)
            rows.append(
                {
                    "Turn": rec.turn_id,
                    "Speaker": rec.speaker_assignment.speaker_id,
                    "Start (s)": round(rec.features.start_ns / 1_000_000_000.0, 2),
                    "Transcript": transcript_text,
                    "Claims": claim_text or "—",
                    "Verdict": VERDICT_BADGES.get(v_str, v_str),
                    "Explanation": ver.explanation,
                }
            )
    return rows


def render_dashboard(replay_file: Optional[str] = None) -> None:
    """Render Streamlit web dashboard from a real JSONL replay file.

    Imports streamlit and plotly lazily within the function scope.

    Args:
        replay_file: Path to a recorded JSONL replay file.
    """
    # LAZY IMPORT of streamlit and plotly
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError:
        print("streamlit or plotly not installed. Install with 'pip install .[ui]'")
        return

    st.set_page_config(
        page_title="Discourse Consistency Engine (DCE)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Discourse Consistency Engine (`dce`)")
    st.markdown(
        "**Neuro-Symbolic Conversational Cognitive Load Analysis & Consistency Verification**\n\n"
        "> *Notice: This system is NOT a lie detector. Physiological deviations reflect situational "
        "cognitive load, anxiety, or topic complexity.*"
    )

    st.sidebar.header("Controls & Session Replay")
    file_path = st.sidebar.text_input("JSONL Replay File", value=replay_file or "data/session.jsonl")
    session_wav_path = st.sidebar.text_input(
        "Optional Full-Session WAV (fallback for timestamp slicing)",
        value="",
    )

    records = load_replay_records(file_path)
    if not records:
        st.info(
            f"No replay records found at `{file_path}`. Run a pipeline session or provide a valid JSONL path."
        )
        return

    turn_ids = [r.turn_id for r in records]
    selected_turn_id = st.sidebar.selectbox("Inspect Turn", options=turn_ids, index=len(turn_ids) - 1)
    selected_record = next((r for r in records if r.turn_id == selected_turn_id), records[-1])

    st.subheader(f"Modality Reliability & Score-Level Late Fusion (Turn {selected_record.turn_id})")
    fusion = compute_late_fusion_score(selected_record)
    rel = fusion["reliability"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Audio", rel["audio"]["status"], delta=f"w={rel['audio']['weight']:.1f}")
    m2.metric("Vision", rel["vision"]["status"], delta=f"w={rel['vision']['weight']:.1f}")
    m3.metric("Text", rel["text"]["status"], delta=f"w={rel['text']['weight']:.1f}")
    m4.metric(
        "Review Priority (Max-Signal)",
        f"{fusion['review_priority']:.2f}",
        delta=f"active={','.join(fusion['active_modalities']) or 'none'}",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Multimodal Prosodic & Kinematic Stream")
        f0_turns = [r.turn_id for r in records if r.features.prosody.f0_semitone_sd is not None]
        f0_vals = [r.features.prosody.f0_semitone_sd for r in records if r.features.prosody.f0_semitone_sd is not None]
        lip_turns = [r.turn_id for r in records if r.features.vision.mean_lip_aperture is not None]
        lip_vals = [r.features.vision.mean_lip_aperture for r in records if r.features.vision.mean_lip_aperture is not None]

        fig = go.Figure()
        if f0_turns:
            fig.add_trace(
                go.Scatter(x=f0_turns, y=f0_vals, mode="lines+markers", name="F0 Semitone SD")
            )
        if lip_turns:
            fig.add_trace(
                go.Scatter(x=lip_turns, y=lip_vals, mode="lines", name="Lip Aperture", yaxis="y2")
            )
        fig.update_layout(
            xaxis=dict(title="Turn ID"),
            yaxis=dict(title="F0 Semitone SD"),
            yaxis2=dict(title="Lip Aperture", overlaying="y", side="right"),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Mahalanobis Cognitive Load Deviation (Turn {selected_record.turn_id})")
        mah = selected_record.mahalanobis
        if mah is not None and mah.feature_z_scores:
            feats = list(mah.feature_z_scores.keys())
            z_vals = list(mah.feature_z_scores.values())
            colors = ["red" if abs(z) >= 1.5 else "blue" for z in z_vals]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=feats, y=z_vals, marker_color=colors))
            fig_bar.update_layout(yaxis=dict(title="Baseline Z-Score"), height=280)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                f"d²={mah.d_sq:.2f} | tail_p={mah.chi2_tail_score:.4f} | "
                f"direction={mah.direction_label.value} | vocal_tension={mah.vocal_tension_flag}"
            )
        else:
            st.info("No Mahalanobis z-scores recorded for this turn.")

    st.subheader("3-Second Audio Evidence Playback")
    clip_bytes: Optional[bytes] = None
    word_offset_sec = find_target_word_offset_sec(selected_record)
    if selected_record.audio_clip_path:
        clip_bytes = slice_wav_clip_bytes(
            selected_record.audio_clip_path,
            start_sec=word_offset_sec,
            duration_sec=3.0,
        )
    elif session_wav_path.strip():
        start_sec = max(0.0, selected_record.features.start_ns / 1_000_000_000.0 + word_offset_sec)
        clip_bytes = slice_wav_clip_bytes(session_wav_path.strip(), start_sec=start_sec, duration_sec=3.0)

    if clip_bytes is not None:
        st.audio(clip_bytes, format="audio/wav")
    else:
        st.caption("No WAV audio clip linked to this turn (set `audio_dir` in FeatureRecorder or provide a session WAV).")

    st.subheader("Verifiable Claims & Discourse Consistency Ledger")
    rows = build_ledger_rows(records)
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
