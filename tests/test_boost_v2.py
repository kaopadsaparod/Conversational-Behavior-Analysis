"""Unit tests for /boost v2 improvements across DECODE text and multimodal video."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import List

import numpy as np
import pytest

from benchmarks.run_ablation import load_decode_dataset
from dce.baseline_store import BaselineStore
from dce.mcp_server import DCESessionStore
from dce.reasoning.claims_ledger import NLIPreFilter
from dce.schemas import (
    ASRResult,
    BaselineConfig,
    DirectionLabel,
    ProsodyFeatures,
    TimingFeatures,
    TurnFeatures,
    VisionFeatures,
    WordTimestamp,
)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_nli_split_clauses_and_multi_sentence_contradiction() -> None:
    nli = NLIPreFilter(use_hf_model=False)
    clauses = nli.split_clauses("Great! I hope you get the job. I'll be rooting for you!")
    assert len(clauses) == 3
    assert clauses[1] == "I hope you get the job."

    # Single-clause contradiction buried in multi-sentence turn
    score = nli.predict_contradiction_score(
        "Great! I went to Paris yesterday. The weather was nice.",
        "I am tired today. I never went to Paris yesterday.",
    )
    assert score >= 0.80


def test_load_decode_dataset_loads_persona_and_context_as_turn0(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "decode_sample.jsonl"
    record = {
        "id": "decode_ctx_test",
        "agents": {
            "0": {"is_human": True, "persona_lines": ["i have two dogs."]},
            "1": {"is_human": True, "persona_lines": ["i work as a chef."]},
        },
        "conversation_contexts": "anxious ## My son is starting kindergarten next week.",
        "turns": [
            {"turn_id": 0, "agent_id": 0, "text": "I have no pets at all."},
            {"turn_id": 1, "agent_id": 1, "text": "Oh really?"},
        ],
        "is_contradiction": True,
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    samples = load_decode_dataset(jsonl_path)
    assert len(samples) == 1
    turns = samples[0].turns
    assert len(turns) == 5
    assert turns[0] == {"speaker": "SPK_0", "text": "i have two dogs."}
    assert turns[1] == {"speaker": "SPK_1", "text": "i work as a chef."}
    assert turns[2] == {"speaker": "SPK_0", "text": "My son is starting kindergarten next week."}
    assert turns[3] == {"speaker": "SPK_0", "text": "I have no pets at all."}


def test_baseline_store_scores_aus_gaze_and_egemaps_with_wild_priors() -> None:
    priors_path = Path(__file__).resolve().parents[1] / "data" / "priors_wild_video_en.json"
    assert priors_path.is_file()
    store = BaselineStore(config=BaselineConfig(), priors=priors_path)

    assert store.priors["f0_semitone_sd"].stress_polarity == -1
    for key in (
        "spectral_slope_db",
        "alpha_ratio_db",
        "loudness_db",
        "au01_inner_brow_raise",
        "au04_brow_lowerer",
        "au12_lip_corner_puller",
        "au25_lips_part",
        "eye_gaze_horizontal",
    ):
        assert key in store.priors

    tf = TurnFeatures(
        turn_id=1,
        speaker_id="speaker_0",
        start_ns=0,
        end_ns=2_000_000_000,
        timing=TimingFeatures(articulation_rate=4.3) if hasattr(TimingFeatures, "articulation_rate") else TimingFeatures(),
        prosody=ProsodyFeatures(
            f0_semitone_sd=4.5,
            spectral_slope_db=-5.0,
            alpha_ratio_db=11.0,
            loudness_db=-22.0,
            quality_flag="ok",
        ),
        vision=VisionFeatures(
            face_detected=True,
            eye_gaze_horizontal=0.25,
            action_units={
                "AU01_inner_brow_raise": 0.42,
                "AU04_brow_lowerer": 0.60,
                "AU12_lip_corner_puller": 0.50,
                "AU25_lips_part": 0.12,
            },
            quality_flag="ok",
        ),
    )
    scores = store.score_and_update_turn("speaker_0", tf, word_count=10)
    assert "au04_brow_lowerer" in scores
    assert "eye_gaze_horizontal" in scores
    assert "spectral_slope_db" in scores
    assert scores["au04_brow_lowerer"].z_score is not None


def test_mcp_ingest_media_qa_split_latency_and_preword_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 16000
    t = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 160.0 * t).astype(np.float32)
    wav_path = tmp_path / "qa_sample.wav"
    _write_wav(wav_path, sig, sr=sr)

    def _fake_transcribe(audio_pcm16k, sample_rate=16000, model_size="small.en", device="cpu"):
        return ASRResult(
            text="Did you go? Not at all.",
            words=[
                WordTimestamp(word="Did", start=0.10, end=0.30, probability=0.95),
                WordTimestamp(word="you", start=0.32, end=0.50, probability=0.95),
                WordTimestamp(word="go?", start=0.52, end=0.80, probability=0.95),
                WordTimestamp(word="Not", start=1.40, end=1.60, probability=0.95),
                WordTimestamp(word="at", start=1.62, end=1.80, probability=0.95),
                WordTimestamp(word="all.", start=2.30, end=2.50, probability=0.95),
            ],
        )

    import dce.extractors.audio_dsp as audio_dsp
    monkeypatch.setattr(audio_dsp, "transcribe_audio", _fake_transcribe)

    store = DCESessionStore()
    res = store.ingest_media(str(wav_path), use_asr=True)
    assert res["turn_count"] == 2
    assert res["turns"][0]["speaker_id"] == "Q_ASKER"
    assert res["turns"][1]["speaker_id"] == "speaker_0"

    sess = store._sessions[res["session_id"]]
    ans_turn = sess.turns[2]
    assert ans_turn.features_obj.timing.response_latency_closed is not None
    assert ans_turn.features_obj.timing.response_latency_closed == pytest.approx(0.60, abs=0.05)
    assert ans_turn.features_obj.timing.max_pre_word_pause_ms == pytest.approx(500.0, abs=5.0)
    assert ans_turn.features_obj.timing.hesitation_pause_count == 1


def test_mcp_expressive_f0_not_penalized_in_wild_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 16000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 160.0 * t).astype(np.float32)
    mp4_dummy = tmp_path / "expressive.wav"
    _write_wav(mp4_dummy, sig, sr=sr)

    priors_path = Path(__file__).resolve().parents[1] / "data" / "priors_wild_video_en.json"

    import dce.extractors.audio_dsp as audio_dsp
    monkeypatch.setattr(
        audio_dsp,
        "extract_prosody",
        lambda *a, **kw: ProsodyFeatures(f0_median_hz=160.0, f0_semitone_sd=5.8, quality_flag="ok"),
    )

    store = DCESessionStore()
    res = store.ingest_media(
        str(mp4_dummy),
        use_asr=False,
        priors=str(priors_path),
        transcript_override=[{"turn_id": 1, "speaker_id": "speaker_0", "text": "I am telling the truth."}],
    )
    sess = store._sessions[res["session_id"]]
    maha = sess.turns[1].mahalanobis_obj
    # High f0_semitone_sd (5.8 vs 2.95) must NOT trigger cognitive_load_anomaly
    assert maha.direction_label != DirectionLabel.COGNITIVE_LOAD_ANOMALY
    assert maha.d_sq == pytest.approx(0.0, abs=1e-6)


def test_mcp_immediate_response_latency_not_penalized_and_unlock_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 16000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 160.0 * t).astype(np.float32)
    wav_path = tmp_path / "fast_qa.wav"
    _write_wav(wav_path, sig, sr=sr)

    def _fake_transcribe(audio_pcm16k, sample_rate=16000, model_size="small.en", device="cpu"):
        return ASRResult(
            text="Did you go? Yes.",
            words=[
                WordTimestamp(word="Did", start=0.10, end=0.30, probability=0.95),
                WordTimestamp(word="you", start=0.32, end=0.50, probability=0.95),
                WordTimestamp(word="go?", start=0.52, end=0.80, probability=0.95),
                WordTimestamp(word="Yes.", start=0.80, end=1.00, probability=0.95),
            ],
        )

    import dce.extractors.audio_dsp as audio_dsp
    monkeypatch.setattr(audio_dsp, "transcribe_audio", _fake_transcribe)
    monkeypatch.setattr(
        audio_dsp,
        "extract_prosody",
        lambda *a, **kw: ProsodyFeatures(f0_median_hz=160.0, f0_semitone_sd=3.1, quality_flag="ok"),
    )
    monkeypatch.setattr(
        audio_dsp,
        "extract_egemaps_spectral_features",
        lambda *a, **kw: {"acoustic_snr_db": 25.0},
    )

    priors_path = Path(__file__).resolve().parents[1] / "data" / "priors_wild_video_en.json"
    store = DCESessionStore()
    res = store.ingest_media(str(wav_path), use_asr=True, priors=str(priors_path))
    store.submit_verdicts(res["session_id"], verdicts=[])
    unlocked = store.unlock_signals(res["session_id"], output_jsonl_path=str(tmp_path / "replay.jsonl"))
    ans_u = unlocked["turns"][1]
    assert ans_u["timing"]["response_latency_closed"] == pytest.approx(0.0, abs=0.02)
    assert ans_u["mahalanobis"]["d_sq"] == pytest.approx(0.0, abs=1e-6)
    assert ans_u["mahalanobis"]["direction_label"] == "normal"
    assert "degrees_of_freedom" in ans_u["mahalanobis"]


def test_audit_fixes_verdict_enums_replay_overwrite_and_session_cap(tmp_path: Path) -> None:
    from dce.schemas import VerdictStatus

    assert VerdictStatus.MODIFIER_EROSION.value == "MODIFIER_EROSION"
    assert VerdictStatus.EVASION.value == "EVASION"

    sr = 16000
    t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False)
    sig = 0.2 * np.sin(2.0 * np.pi * 160.0 * t).astype(np.float32)
    wav_path = tmp_path / "cap.wav"
    _write_wav(wav_path, sig, sr=sr)

    store = DCESessionStore()
    replay_path = tmp_path / "replay_overwrite.jsonl"
    res = store.ingest_media(
        str(wav_path),
        use_asr=False,
        transcript_override=[{"turn_id": 1, "speaker_id": "speaker_0", "text": "Hello."}],
    )
    sid = res["session_id"]
    store.submit_verdicts(
        sid,
        verdicts=[{"verdict": "EVASION", "source_turn_id": 1, "explanation": "Refused to answer."}],
    )
    store.unlock_signals(sid, output_jsonl_path=str(replay_path))
    store.unlock_signals(sid, output_jsonl_path=str(replay_path))
    lines = [ln for ln in replay_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1

    # Verify DCESessionStore caps active sessions at MAX_SESSIONS (16)
    for _ in range(18):
        store.ingest_media(
            str(wav_path),
            use_asr=False,
            transcript_override=[{"turn_id": 1, "speaker_id": "speaker_0", "text": "Hello."}],
        )
    assert len(store._sessions) == store.MAX_SESSIONS

