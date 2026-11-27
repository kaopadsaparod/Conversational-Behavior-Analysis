"""Dedicated unit tests for the 6 production-hardening upgrades."""

from __future__ import annotations

import json
import multiprocessing
import time
import wave
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.run_ablation import DecodeSample, evaluate_decode_benchmark
from benchmarks.statistical_significance import mcnemar_test, stratified_bootstrap_ci
from dce.extractors.audio_dsp import extract_voice_embedding
from dce.extractors.vision_face import extract_vision_sequence
from dce.mcp_server import (
    DCESessionStore,
    _should_split_question_boundary,
    handle_jsonrpc_request,
)
from dce.pipeline import DiscourseConsistencyPipeline, audio_worker_process
from dce.reasoning.claims_ledger import ClaimsLedger, extract_clause_spo_claims
from dce.schemas import (
    AppConfig,
    ClaimExtractionResponse,
    ClaimItem,
    ClaimPolarity,
    HardTurnCommitEvent,
    TurnTranscript,
    VisionFeatures,
)


def _synth_voice(
    f0_hz: float,
    formant_boost_hz: float,
    duration_sec: float = 1.5,
    sr: int = 16000,
) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, int(round(duration_sec * sr)), endpoint=False)
    sig = (
        0.30 * np.sin(2.0 * np.pi * f0_hz * t)
        + 0.18 * np.sin(2.0 * np.pi * (2.0 * f0_hz) * t)
        + 0.15 * np.sin(2.0 * np.pi * formant_boost_hz * t)
        + 0.10 * np.sin(2.0 * np.pi * (formant_boost_hz + f0_hz) * t)
    )
    return sig.astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> Path:
    pcm_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())
    return path


# ---------------------------------------------------------------------------
# 1. Real Acoustic Voice Embedding & Unified SpeakerRegistry
# ---------------------------------------------------------------------------


def test_extract_voice_embedding_deterministic_and_discriminative() -> None:
    spk_a1 = _synth_voice(f0_hz=125.0, formant_boost_hz=700.0)
    spk_a2 = _synth_voice(f0_hz=128.0, formant_boost_hz=710.0)
    spk_b = _synth_voice(f0_hz=126.0, formant_boost_hz=2600.0)  # same F0, very different timbre

    emb_a1 = extract_voice_embedding(spk_a1, sample_rate=16000)
    emb_a1_repeat = extract_voice_embedding(spk_a1, sample_rate=16000)
    emb_a2 = extract_voice_embedding(spk_a2, sample_rate=16000)
    emb_b = extract_voice_embedding(spk_b, sample_rate=16000)

    assert emb_a1.shape == (64,)
    assert np.isclose(np.linalg.norm(emb_a1), 1.0, atol=1e-6)
    assert np.allclose(emb_a1, emb_a1_repeat, atol=1e-9)

    sim_same = float(np.dot(emb_a1, emb_a2))
    sim_diff = float(np.dot(emb_a1, emb_b))
    assert sim_same > sim_diff
    assert sim_diff < 0.85

    # Empty / silent / non-finite -> 64-D zero vector
    assert np.allclose(extract_voice_embedding(np.zeros(0), 16000), np.zeros(64))
    assert np.allclose(extract_voice_embedding(np.zeros(16000), 16000), np.zeros(64))


def test_pipeline_handle_turn_commit_uses_real_embedding_not_dummy_192() -> None:
    pipeline = DiscourseConsistencyPipeline(config=AppConfig())
    audio_a = _synth_voice(f0_hz=120.0, formant_boost_hz=650.0, duration_sec=2.5)
    ev = HardTurnCommitEvent(
        turn_id=1,
        epoch_id=1,
        valid_epoch_id=1,
        start_ns=0,
        end_ns=2_500_000_000,
        audio_pcm16k=audio_a.tolist(),
    )
    rec = pipeline._handle_turn_commit(ev, transcript_text="I arrived at nine.")
    assert rec.speaker_assignment.speaker_id == "SPEAKER_00"
    spk_profile = pipeline._speaker_registry.speakers["SPEAKER_00"]
    assert spk_profile.centroid.shape == (64,)


def test_audio_worker_process_computes_prosody_and_embedding_when_requested() -> None:
    in_q: multiprocessing.Queue = multiprocessing.Queue()
    out_q: multiprocessing.Queue = multiprocessing.Queue()
    sig = _synth_voice(f0_hz=140.0, formant_boost_hz=900.0, duration_sec=1.0)
    in_q.put((1_000_000_000, sig))
    in_q.put("STOP")

    audio_worker_process(
        in_q,
        out_q,
        config_dict={"sample_rate": 16000, "extract_prosody": True, "extract_embedding": True},
    )
    res1 = out_q.get(timeout=5)
    res2 = out_q.get(timeout=5)
    assert res1["status"] == "OK"
    assert "prosody" in res1
    assert "voice_embedding" in res1
    assert len(res1["voice_embedding"]) == 64
    assert res2["status"] == "STOPPED"


def test_mcp_routes_non_asker_turns_through_speaker_registry(tmp_path: Path) -> None:
    # Two non-asker turns with identical F0 (130 Hz) but distinct spectral timbre
    spk_a = _synth_voice(f0_hz=130.0, formant_boost_hz=650.0, duration_sec=2.2)
    silence = np.zeros(int(1.0 * 16000), dtype=np.float32)
    spk_b = _synth_voice(f0_hz=130.0, formant_boost_hz=2800.0, duration_sec=2.2)
    full_audio = np.concatenate([spk_a, silence, spk_b])
    wav_path = _write_wav(tmp_path / "two_speakers_same_f0.wav", full_audio)

    store = DCESessionStore()
    with patch("dce.extractors.audio_dsp.transcribe_audio") as mock_asr:
        mock_asr.side_effect = [
            type("ASR", (), {"text": "I stayed home all evening.", "words": []})(),
            type("ASR", (), {"text": "No, I saw him at the station.", "words": []})(),
        ]
        res = store.ingest_media(media_path=str(wav_path), use_asr=True)

    turns = res["turns"]
    assert len(turns) == 2
    assert turns[0]["speaker_id"] != turns[1]["speaker_id"]


# ---------------------------------------------------------------------------
# 2. Fix Rhetorical Question Diarization Trap
# ---------------------------------------------------------------------------


def test_rhetorical_question_not_split_on_short_pause_or_tag_or_same_speaker() -> None:
    a1 = _synth_voice(f0_hz=135.0, formant_boost_hz=800.0, duration_sec=1.0)
    a2_same = _synth_voice(f0_hz=138.0, formant_boost_hz=810.0, duration_sec=1.0)
    a2_diff = _synth_voice(f0_hz=220.0, formant_boost_hz=2400.0, duration_sec=1.0)

    # (a) Short pause (<250ms) even with different speaker -> False
    assert _should_split_question_boundary(
        "Why would I steal it?", 0.12, a1, a2_diff, 16000
    ) is False

    # (b) Short tag question ("right?", "huh?", "you know?", "okay?") -> False
    for tag in ("right?", "huh?", "you know?", "okay?"):
        assert _should_split_question_boundary(tag, 0.40, a1, a2_diff, 16000) is False

    # (c) Long pause (>=250ms), non-tag, but SAME speaker -> False
    assert _should_split_question_boundary(
        "Why would I steal it?", 0.40, a1, a2_same, 16000
    ) is False

    # All 3 conditions met (pause >= 250ms, real question, different speaker) -> True
    assert _should_split_question_boundary(
        "Why would I steal it?", 0.40, a1, a2_diff, 16000
    ) is True


# ---------------------------------------------------------------------------
# 3. Multi-Face Tracking & Active-Speaker Lip-Sync Gate
# ---------------------------------------------------------------------------


def test_extract_vision_sequence_max_faces_empty_and_multiface_gate(tmp_path: Path) -> None:
    assert extract_vision_sequence([], fps=15.0, max_faces=3) == {}

    # Build a 2-second voiced audio with strong amplitude modulation at 3 Hz
    sr = 16000
    dur = 2.0
    t_aud = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 3.0 * t_aud))
    audio = (env * 0.3 * np.sin(2.0 * np.pi * 150.0 * t_aud)).astype(np.float32)
    wav_path = _write_wav(tmp_path / "av_gate.wav", audio, sr=sr)

    # 30 frames at 15 FPS: face_0 is a static listener; face_1 lip aperture tracks env
    n_frames = 30
    t_vis = np.arange(n_frames) / 15.0
    lip_speaking = 0.15 + 0.10 * np.sin(2.0 * np.pi * 3.0 * t_vis)
    face_listener = [
        VisionFeatures(
            face_detected=True,
            face_id_hint="face_0",
            mean_lip_aperture=0.05,
            blink_rate_per_min=18.0,
            quality_flag="ok",
        )
        for _ in range(n_frames)
    ]
    face_speaker = [
        VisionFeatures(
            face_detected=True,
            face_id_hint="face_1",
            mean_lip_aperture=float(lip_speaking[i]),
            blink_rate_per_min=14.0,
            quality_flag="ok",
        )
        for i in range(n_frames)
    ]

    store = DCESessionStore()
    with (
        patch("dce.mcp_server._load_media_audio_and_vision", return_value=(audio, sr, [np.zeros((8, 8, 3), dtype=np.uint8)] * n_frames)),
        patch("dce.extractors.vision_face.extract_vision_sequence", return_value={"face_0": face_listener, "face_1": face_speaker}),
    ):
        res_ok = store.ingest_media(
            media_path=str(wav_path),
            use_asr=False,
            transcript_override=[{"turn_id": 1, "start_sec": 0.0, "end_sec": 2.0, "text": "I spoke clearly."}],
        )
    sess_ok = store._sessions[res_ok["session_id"]]
    vis_ok = sess_ok.turns[1].features_obj.vision
    assert vis_ok.face_id_hint == "face_1"
    assert vis_ok.quality_flag == "ok"

    # Now test Active-Speaker Gate: both faces are static listeners (corr < 0.12 / std < 1e-4)
    face_listener_2 = [v.model_copy(update={"face_id_hint": "face_1"}) for v in face_listener]
    with (
        patch("dce.mcp_server._load_media_audio_and_vision", return_value=(audio, sr, [np.zeros((8, 8, 3), dtype=np.uint8)] * n_frames)),
        patch("dce.extractors.vision_face.extract_vision_sequence", return_value={"face_0": face_listener, "face_1": face_listener_2}),
    ):
        res_gated = store.ingest_media(
            media_path=str(wav_path),
            use_asr=False,
            transcript_override=[{"turn_id": 1, "start_sec": 0.0, "end_sec": 2.0, "text": "Offscreen speaker."}],
        )
    sess_gated = store._sessions[res_gated["session_id"]]
    vis_gated = sess_gated.turns[1].features_obj.vision
    assert vis_gated.quality_flag == "listener_or_offscreen_face"


# ---------------------------------------------------------------------------
# 4. MCP Long-Video Timeout Protection
# ---------------------------------------------------------------------------


def test_mcp_progress_notifications_and_async_ingest_job_status(tmp_path: Path) -> None:
    wav_path = _write_wav(tmp_path / "prog.wav", _synth_voice(150.0, 900.0, 2.0))
    store = DCESessionStore()
    notifications: List[Dict[str, Any]] = []

    turns_override = [
        {"turn_id": i, "start_sec": (i - 1) * 0.25, "end_sec": i * 0.25, "text": f"Turn {i}."}
        for i in range(1, 8)
    ]
    resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "dce_ingest_media",
                "arguments": {
                    "media_path": str(wav_path),
                    "use_asr": False,
                    "transcript_override": turns_override,
                },
                "_meta": {"progressToken": "tok-42"},
            },
        },
        store=store,
        progress_callback=notifications.append,
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    stages = [n["params"]["message"] for n in notifications]
    assert all(n["method"] == "notifications/progress" for n in notifications)
    assert all(n["params"]["progressToken"] == "tok-42" for n in notifications)
    assert "media_decoded" in stages
    assert "vision_completed" in stages
    assert "processed_turn_5" in stages

    # Test async_ingest=True + dce_get_job_status polling
    async_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": {
                "name": "dce_ingest_media",
                "arguments": {
                    "media_path": str(wav_path),
                    "use_asr": False,
                    "async_ingest": True,
                    "transcript_override": turns_override[:2],
                },
            },
        },
        store=store,
    )
    assert async_resp is not None
    async_payload = json.loads(async_resp["result"]["content"][0]["text"])
    assert async_payload["status"] == "INGESTING_ASYNC"
    job_id = async_payload["job_id"]

    final_status = None
    for _ in range(50):
        st_resp = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 103,
                "method": "tools/call",
                "params": {"name": "dce_get_job_status", "arguments": {"job_id": job_id}},
            },
            store=store,
        )
        assert st_resp is not None
        final_status = json.loads(st_resp["result"]["content"][0]["text"])
        if final_status["status"] == "COMPLETED":
            break
        time.sleep(0.05)

    assert final_status is not None
    assert final_status["status"] == "COMPLETED"
    assert final_status["result"]["status"] == "INGESTED"
    assert final_status["result"]["turn_count"] == 2


# ---------------------------------------------------------------------------
# 5. Explicit Step-A Decomposition Mode in Ablation
# ---------------------------------------------------------------------------


def test_ablation_extraction_modes_clause_spo_and_llm_step_a() -> None:
    tx = TurnTranscript(
        turn_id=1,
        speaker_id="SPK_0",
        text="I arrived at 9pm, and I did not see anyone there.",
        start_ns=0,
        end_ns=1_000_000_000,
    )
    spo_claims = extract_clause_spo_claims(tx)
    assert len(spo_claims) == 2
    assert spo_claims[0].polarity == ClaimPolarity.AFFIRMATIVE
    assert spo_claims[1].polarity == ClaimPolarity.NEGATIVE

    samples = [
        DecodeSample(
            sample_id="s1",
            turns=[
                {"speaker": "SPK_0", "text": "I own a red car, and I live in Boston."},
                {"speaker": "SPK_0", "text": "I do not own any car."},
            ],
            is_contradiction=True,
        )
    ]
    m_utt = evaluate_decode_benchmark(samples, backend="nli", extraction_mode="utterance")
    m_spo = evaluate_decode_benchmark(samples, backend="nli", extraction_mode="clause_spo")
    assert m_utt.total_samples == 1
    assert m_spo.total_samples == 1

    class _MockLLM:
        def generate_structured(self, prompt: str, response_model: Any, system_prompt: str = "") -> Any:
            if response_model is ClaimExtractionResponse:
                return ClaimExtractionResponse(
                    claims=[
                        ClaimItem(
                            claim_id="mock_1",
                            turn_id=1,
                            speaker_id="SPK_0",
                            raw_subject="I",
                            canonical_entity="spk_0",
                            predicate="own",
                            object_value="red car",
                            polarity=ClaimPolarity.NEGATIVE if "not" in prompt.lower() else ClaimPolarity.AFFIRMATIVE,
                        )
                    ]
                )
            raise RuntimeError("unused")

    m_llm = evaluate_decode_benchmark(
        samples,
        ledger_factory=lambda: ClaimsLedger(llm_adapter=_MockLLM()),
        backend="nli",
        extraction_mode="llm_step_a",
    )
    assert m_llm.total_samples == 1


# ---------------------------------------------------------------------------
# 6. Statistical Significance (Bootstrap CIs, Binomial & McNemar)
# ---------------------------------------------------------------------------


def test_statistical_significance_bootstrap_and_mcnemar() -> None:
    y_true = [True] * 25 + [False] * 25
    pred_a = [True] * 22 + [False] * 3 + [False] * 22 + [True] * 3  # 88% acc
    pred_b = [True] * 14 + [False] * 11 + [False] * 14 + [True] * 11  # 56% acc

    ci_a = stratified_bootstrap_ci(y_true, pred_a, n_boot=1000, seed=42)
    assert ci_a["accuracy_ci95"][0] <= ci_a["accuracy"] <= ci_a["accuracy_ci95"][1]
    assert ci_a["f1_ci95"][0] <= ci_a["f1"] <= ci_a["f1_ci95"][1]
    assert ci_a["p_value_vs_chance"] < 1e-6

    mc = mcnemar_test(y_true, pred_a, pred_b)
    assert mc["n"] == 50
    assert mc["p_value_exact"] < 0.01


def test_mcp_provisional_speaker_promotion_merges_baseline_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Two short turns (< 2.0s each) from the same speaker: Turn 1 -> PROV_01, Turn 2 -> SPEAKER_00
    spk_1 = _synth_voice(f0_hz=135.0, formant_boost_hz=750.0, duration_sec=1.2)
    silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
    spk_2 = _synth_voice(f0_hz=136.0, formant_boost_hz=755.0, duration_sec=1.2)
    wav_path = _write_wav(tmp_path / "promo_merge.wav", np.concatenate([spk_1, silence, spk_2]))

    store = DCESessionStore()
    with caplog.at_level("INFO", logger="dce.baseline_store"):
        with patch("dce.extractors.audio_dsp.transcribe_audio") as mock_asr:
            mock_asr.side_effect = [
                type("ASR", (), {"text": "First short turn.", "words": []})(),
                type("ASR", (), {"text": "Second short turn.", "words": []})(),
            ]
            res = store.ingest_media(media_path=str(wav_path), use_asr=True)

    turns = res["turns"]
    assert len(turns) == 2
    assert turns[0]["speaker_id"] == "PROV_01"
    assert turns[1]["speaker_id"] == "SPEAKER_00"
    assert "Successfully merged speaker profile PROV_01 into SPEAKER_00" in caplog.text

    sess = store._sessions[res["session_id"]]
    assert sess.turns[2].speaker_assignment_obj.promoted_from == "PROV_01"
    store.submit_verdicts(res["session_id"], verdicts=[])
    unlocked = store.unlock_signals(res["session_id"])
    assert len(unlocked["turns"][1]["mahalanobis"]["evaluated_features"]) >= 1

