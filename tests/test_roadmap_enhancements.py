"""Tests for ASRResult caching/hotwords, Warm-up Trap guard, L2/Phonetic-Fragile ledger, and Dashboard/WAV slicing."""

from __future__ import annotations

import io
import sys
import types
import wave
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from app.dashboard import (
    build_ledger_rows,
    find_target_word_offset_sec,
    load_replay_records,
    slice_wav_clip_bytes,
)
from dce.baseline_store import BaselineStore
from dce.extractors.audio_dsp import clear_whisper_model_cache, transcribe_audio
from dce.pipeline import DiscourseConsistencyPipeline
from dce.reasoning.claims_ledger import ClaimsLedger, is_phonetic_fragile_difference
from dce.recorder import FeatureRecorder
from dce.schemas import (
    ASRResult,
    BaselineConfig,
    ClaimExtractionResponse,
    ClaimItem,
    ClaimPolarity,
    ConsistencyVerificationResponse,
    ProsodyFeatures,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    TurnTranscript,
    VerdictStatus,
    VisionFeatures,
    WordTimestamp,
)


def test_transcribe_audio_caches_model_returns_words_and_forwards_hotwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify WhisperModel singleton caching, WordTimestamp extraction, and hotwords forwarding."""
    clear_whisper_model_cache()
    init_calls = 0
    captured_kwargs: List[dict] = []

    class FakeWord:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class FakeSegment:
        def __init__(self, text: str, words: List[FakeWord]) -> None:
            self.text = text
            self.no_speech_prob = 0.05
            self.avg_logprob = -0.15
            self.compression_ratio = 1.1
            self.words = words

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal init_calls
            init_calls += 1

        def transcribe(self, audio: Any, **kwargs: Any):
            captured_kwargs.append(dict(kwargs))
            words = [
                FakeWord(" Bangkok", 0.1, 0.5, 0.94),
                FakeWord(" tomorrow", 0.55, 1.0, 0.91),
            ]
            return [FakeSegment("Bangkok tomorrow", words)], types.SimpleNamespace(language="en")

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    res1 = transcribe_audio([0.1] * 1600, hotwords="Bangkok, Sukhumvit")
    res2 = transcribe_audio([0.2] * 1600, hotwords="Chiang Mai")

    assert isinstance(res1, ASRResult)
    assert res1.text == "Bangkok tomorrow"
    assert res1.language == "en"
    assert len(res1.words) == 2
    assert res1.words[0].word == " Bangkok"
    assert res1.words[0].start == pytest.approx(0.1)
    assert res1.words[0].end == pytest.approx(0.5)
    assert res1.words[0].probability == pytest.approx(0.94)

    # Cached across both calls -> WhisperModel instantiated only once
    assert init_calls == 1
    assert captured_kwargs[0]["hotwords"] == "Bangkok, Sukhumvit"
    assert captured_kwargs[1]["hotwords"] == "Chiang Mai"
    assert isinstance(res2, ASRResult)


def test_baseline_store_warmup_trap_ignores_short_turns_for_anchor_lock() -> None:
    """Short greeting turns (< min_anchor_words) must not lock the anchor baseline."""
    cfg = BaselineConfig(anchor_lock_turns=3, min_anchor_words=8)
    store = BaselineStore(config=cfg, allow_builtin_priors=True)

    # 5 short turns (2 words each) should NOT lock anchor
    for _ in range(5):
        store.update_feature(
            speaker_id="SPK_0",
            feature_name="f0_semitone_sd",
            raw_value=2.5,
            quality_flag="ok",
            word_count=2,
        )
    assert store._anchor_locked.get("SPK_0", {}).get("f0_semitone_sd", False) is False

    # 3 substantive turns (>= 8 words) SHOULD lock anchor
    for _ in range(3):
        store.update_feature(
            speaker_id="SPK_0",
            feature_name="f0_semitone_sd",
            raw_value=3.0,
            quality_flag="ok",
            word_count=10,
        )
    assert store._anchor_locked["SPK_0"]["f0_semitone_sd"] is True


def test_claims_ledger_l2_prompt_firewall_and_phonetic_fragile_downgrade() -> None:
    """Verify L2 negative-question context passes firewall and phonetic-fragile conflicts become NEEDS_AUDIO_REVIEW."""
    class MockLLM:
        def __init__(self) -> None:
            self.prompts: List[str] = []

        def generate_structured(self, prompt: str, response_model: Any, system_prompt: str = "") -> Any:
            self.prompts.append(prompt)
            if response_model is ClaimExtractionResponse:
                return ClaimExtractionResponse(
                    claims=[
                        ClaimItem(
                            claim_id="TMP",
                            turn_id=2,
                            speaker_id="SPK_B",
                            raw_subject="I",
                            canonical_entity="speaker_b",
                            predicate="went_to",
                            object_value="London",
                            polarity=ClaimPolarity.NEGATIVE,
                        )
                    ]
                )
            return ConsistencyVerificationResponse(
                verdict=VerdictStatus.DIRECT_CONTRADICTION,
                source_turn_id=2,
                prior_turn_id=1,
                source_claim_id="CLM_0002_01",
                prior_claim_id="CLM_0001_01",
                explanation="Count differs: fifty vs fifteen.",
                confidence=0.9,
            )

    llm = MockLLM()
    ledger = ClaimsLedger(llm_adapter=llm)  # type: ignore[arg-type]

    q_turn = TurnTranscript(
        turn_id=1,
        speaker_id="SPK_A",
        text="Didn't you go to London yesterday?",
        start_ns=0,
        end_ns=1_000_000_000,
        is_question=True,
    )
    a_turn = TurnTranscript(
        turn_id=2,
        speaker_id="SPK_B",
        text="Yes, I didn't go.",
        start_ns=1_500_000_000,
        end_ns=2_500_000_000,
        is_question=False,
    )

    extracted = ledger.extract_claims(a_turn, context_turns=[q_turn])
    assert len(extracted) == 1
    assert "Didn't you go to London yesterday?" in llm.prompts[0]

    # Phonetic-fragile pair: "fifteen" vs "fifty" -> NEEDS_AUDIO_REVIEW
    prior_claim = ClaimItem(
        claim_id="CLM_0001_01",
        turn_id=1,
        speaker_id="SPK_B",
        raw_subject="tickets",
        canonical_entity="tickets",
        predicate="count",
        object_value="fifteen dollars",
    )
    curr_claim = ClaimItem(
        claim_id="CLM_0002_01",
        turn_id=2,
        speaker_id="SPK_B",
        raw_subject="tickets",
        canonical_entity="tickets",
        predicate="count",
        object_value="fifty dollars",
    )
    ledger.add_claim(prior_claim)
    ledger.add_claim(curr_claim)

    assert is_phonetic_fragile_difference(curr_claim, prior_claim) is True
    res = ledger.verify_consistency(curr_claim, candidate_prior_claims=[prior_claim])
    assert res.response.verdict == VerdictStatus.NEEDS_AUDIO_REVIEW
    assert "Audio review recommended" in res.response.explanation

    # Non-fragile difference: "Paris" vs "London" -> stays DIRECT_CONTRADICTION
    non_fragile = ClaimItem(
        claim_id="CLM_0002_02",
        turn_id=2,
        speaker_id="SPK_B",
        raw_subject="city",
        canonical_entity="city",
        predicate="visited",
        object_value="Paris",
    )
    prior_london = ClaimItem(
        claim_id="CLM_0001_02",
        turn_id=1,
        speaker_id="SPK_B",
        raw_subject="city",
        canonical_entity="city",
        predicate="visited",
        object_value="London",
    )
    ledger.add_claim(prior_london)
    assert is_phonetic_fragile_difference(non_fragile, prior_london) is False

    # Normalized numeric & contraction pairs: "15" vs "fifty", "cannot" vs "can"
    c_15 = ClaimItem(
        claim_id="C15",
        turn_id=1,
        speaker_id="SPK_B",
        raw_subject="tickets",
        canonical_entity="tickets",
        predicate="count",
        object_value="15 dollars",
    )
    c_fifty = ClaimItem(
        claim_id="C50",
        turn_id=2,
        speaker_id="SPK_B",
        raw_subject="tickets",
        canonical_entity="tickets",
        predicate="count",
        object_value="fifty dollars",
    )
    assert is_phonetic_fragile_difference(c_15, c_fifty) is True

    c_cannot = ClaimItem(
        claim_id="CC1",
        turn_id=1,
        speaker_id="SPK_B",
        raw_subject="trip",
        canonical_entity="trip",
        predicate="status",
        object_value="cannot attend",
    )
    c_can = ClaimItem(
        claim_id="CC2",
        turn_id=2,
        speaker_id="SPK_B",
        raw_subject="trip",
        canonical_entity="trip",
        predicate="status",
        object_value="can attend",
    )
    assert is_phonetic_fragile_difference(c_cannot, c_can) is True


def test_recorder_wav_clip_and_dashboard_helpers(tmp_path: Path) -> None:
    """Verify FeatureRecorder saves full-turn WAVs and dashboard slices 3s around WordTimestamp."""
    jsonl_path = tmp_path / "session.jsonl"
    audio_dir = tmp_path / "clips"

    # 5 seconds of 16kHz sine wave; default save_turn_audio saves full 5.0s (80000 samples)
    sr = 16000
    samples = np.sin(np.linspace(0, 100.0, sr * 5)).astype(np.float32)

    with FeatureRecorder(jsonl_path, audio_dir=audio_dir) as rec:
        clip_path = rec.save_turn_audio(turn_id=1, audio_pcm16k=samples, sample_rate=sr)
        turn_rec = RecorderTurnRecord(
            turn_id=1,
            speaker_assignment=SpeakerAssignment(speaker_id="SPK_00", similarity=0.95),
            transcript=TurnTranscript(
                turn_id=1,
                speaker_id="SPK_00",
                text="Actually I paid fifty dollars.",
                start_ns=0,
                end_ns=5_000_000_000,
                words=[
                    WordTimestamp(word="Actually", start=0.2, end=0.7, probability=0.95),
                    WordTimestamp(word="fifty", start=4.2, end=4.6, probability=0.88),
                ],
            ),
            features=TurnFeatures(
                turn_id=1,
                speaker_id="SPK_00",
                start_ns=0,
                end_ns=5_000_000_000,
                timing=TimingFeatures(),
                prosody=ProsodyFeatures(f0_semitone_sd=2.4, quality_flag="ok"),
                vision=VisionFeatures(quality_flag="unavailable"),
            ),
            claims=[
                ClaimItem(
                    claim_id="CLM_0001_01",
                    turn_id=1,
                    speaker_id="SPK_00",
                    raw_subject="I",
                    canonical_entity="speaker_00",
                    predicate="paid",
                    object_value="fifty dollars",
                )
            ],
            verifications=[
                ConsistencyVerificationResponse(
                    verdict=VerdictStatus.NEEDS_AUDIO_REVIEW,
                    source_turn_id=1,
                    prior_turn_id=0,
                    explanation="Possible fifteen/fifty ASR confusion.",
                )
            ],
            audio_clip_path=clip_path,
        )
        rec.write_record(turn_rec)

    assert Path(clip_path).is_file()
    with wave.open(clip_path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 80000

    loaded = load_replay_records(jsonl_path)
    assert len(loaded) == 1
    assert loaded[0].audio_clip_path == clip_path

    offset_sec = find_target_word_offset_sec(loaded[0])
    assert offset_sec == pytest.approx(3.2)

    sliced_bytes = slice_wav_clip_bytes(clip_path, start_sec=offset_sec, duration_sec=1.5)
    assert sliced_bytes is not None
    with wave.open(io.BytesIO(sliced_bytes), "rb") as swf:
        assert swf.getframerate() == 16000
        assert swf.getnframes() == 24000

    rows = build_ledger_rows(loaded)
    assert len(rows) == 1
    assert rows[0]["Verdict"] == "⚪ NEEDS_AUDIO_REVIEW"
    assert rows[0]["Transcript"] == "Actually I paid fifty dollars."


def test_pipeline_rolling_hotwords_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pipeline collects entities into a bounded rolling hotwords buffer and forwards them to ASR."""
    pipeline = DiscourseConsistencyPipeline()
    claims = [
        ClaimItem(
            claim_id=f"C{i}",
            turn_id=i,
            speaker_id="SPEAKER_00",
            raw_subject="I",
            canonical_entity=f"Entity{i}",
            predicate="visited",
            object_value="place",
            location_ref=f"City{i}" if i == 1 else None,
        )
        for i in range(1, 25)
    ]
    pipeline._update_hotwords_from_claims(claims)
    hotwords = pipeline.get_hotwords()
    assert hotwords is not None
    parts = [p.strip() for p in hotwords.split(",")]
    assert len(parts) == 20
    assert "I" not in parts
    assert "Entity24" in parts

    captured = {}

    def fake_transcribe(audio_pcm: Any, **kwargs: Any) -> ASRResult:
        captured.update(kwargs)
        return ASRResult(text="test")

    monkeypatch.setattr("dce.pipeline.transcribe_audio", fake_transcribe)
    res = pipeline.transcribe_turn_audio([0.1] * 1600)
    assert res.text == "test"
    assert captured["hotwords"] == hotwords
    pipeline.close()


def test_zero_phase_highpass_and_voip_compression_gate() -> None:
    """Verify 65 Hz zero-phase high-pass removes 50 Hz hum and VoIP detector disables fine wave metrics."""
    from dce.extractors.audio_dsp import apply_zero_phase_highpass, detect_voip_or_denoised_audio

    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    hum_50hz = 0.5 * np.sin(2.0 * np.pi * 50.0 * t)
    voice_150hz = 0.5 * np.sin(2.0 * np.pi * 150.0 * t)
    filtered = apply_zero_phase_highpass(hum_50hz + voice_150hz, sample_rate=sr, cutoff_hz=65.0)

    # 50 Hz hum attenuated strongly while 150 Hz voice remains intact
    hum_only_filtered = apply_zero_phase_highpass(hum_50hz, sample_rate=sr, cutoff_hz=65.0)
    assert np.sqrt(np.mean(hum_only_filtered ** 2)) < 0.15 * np.sqrt(np.mean(hum_50hz ** 2))
    assert np.sqrt(np.mean(filtered ** 2)) > 0.45 * np.sqrt(np.mean(voice_150hz ** 2))

    # Construct VoIP/AI-denoised signal: 30% digital zero silence + speech band (500+1200 Hz) with zero >4kHz energy
    voip_sig = np.zeros(sr, dtype=np.float64)
    active = 0.3 * np.sin(2.0 * np.pi * 500.0 * t[: int(0.7 * sr)]) + 0.15 * np.sin(2.0 * np.pi * 1200.0 * t[: int(0.7 * sr)])
    voip_sig[: int(0.7 * sr)] = active
    is_voip, diag = detect_voip_or_denoised_audio(voip_sig, sample_rate=sr)
    assert is_voip is True
    assert diag["gated_ratio"] >= 0.15
    assert diag["highband_rolloff_db"] <= -30.0

    # BaselineStore must exclude jitter/shimmer/hnr when voip_or_denoised_flag=True, while keeping F0 SD & rate
    store = BaselineStore(allow_builtin_priors=True)
    tf_voip = TurnFeatures(
        turn_id=1,
        speaker_id="SPK_VOIP",
        start_ns=0,
        end_ns=1_000_000_000,
        prosody=ProsodyFeatures(
            f0_semitone_sd=3.1,
            articulation_rate=4.5,
            jitter_local_pct=2.5,
            shimmer_local_pct=9.0,
            hnr_db=10.0,
            voip_or_denoised_flag=True,
            quality_flag="ok",
        ),
    )
    scores = store.score_and_update_turn("SPK_VOIP", tf_voip)
    assert scores["f0_semitone_sd"].z_score is not None
    assert scores["articulation_rate"].z_score is not None
    assert scores["jitter_local_pct"].z_score is None
    assert scores["shimmer_local_pct"].z_score is None
    assert scores["hnr_db"].z_score is None
