"""Tests for the Windows-safe stdio MCP server (src/dce/mcp_server.py)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from dce.mcp_server import (
    DCESessionStore,
    MCP_TOOLS,
    handle_jsonrpc_request,
    run_stdio_server,
)
from dce.reasoning.claims_ledger import assert_firewall_compliance


def _write_test_wav(path: Path, duration_sec: float = 5.0, sample_rate: int = 16000) -> Path:
    """Create a synthetic voiced-like 16kHz mono WAV file."""
    t = np.linspace(0.0, duration_sec, int(round(duration_sec * sample_rate)), endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 150.0 * t) + 0.1 * np.sin(2.0 * np.pi * 300.0 * t)
    pcm_i16 = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
    return path


def test_subprocess_stdio_handshake_and_stdout_cleanliness() -> None:
    """Spawn python -m dce.mcp_server and verify fast initialize + tools/list + clean stdout."""
    repo_root = Path(__file__).resolve().parent.parent
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "dce.mcp_server"],
        input=stdin_payload,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=10,
    )
    elapsed = time.perf_counter() - t0

    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    # Every non-empty line on stdout MUST be valid JSON-RPC 2.0
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 3, f"Expected 3 JSON-RPC responses, got {len(lines)}: {lines}"

    resp_init = json.loads(lines[0])
    assert resp_init["id"] == 1
    assert resp_init["result"]["serverInfo"]["name"] == "dce-analyzer"

    resp_tools = json.loads(lines[1])
    assert resp_tools["id"] == 2
    tool_names = [t["name"] for t in resp_tools["result"]["tools"]]
    assert tool_names == [
        "dce_ingest_media",
        "dce_get_job_status",
        "dce_submit_verdicts",
        "dce_unlock_signals",
        "dce_slice_turn_audio",
    ]

    resp_ping = json.loads(lines[2])
    assert resp_ping["id"] == 3
    assert resp_ping["result"] == {}
    # Subprocess startup + 3 round-trips must stay fast (<0.5s even under load; typically ~0.12s)
    assert elapsed < 0.5


def test_subprocess_fd1_and_print_isolation_during_tool_call(tmp_path: Path) -> None:
    """Verify Python print() and C-level fd-1 writes inside tools/call never pollute stdout."""
    repo_root = Path(__file__).resolve().parent.parent
    wav_path = _write_test_wav(tmp_path / "fd1_test.wav", duration_sec=1.5)
    runner = (
        "import os\n"
        "from dce import mcp_server\n"
        "orig = mcp_server.DCESessionStore.ingest_media\n"
        "def noisy(self, *a, **kw):\n"
        "    print('PY_STDOUT_NOISE')\n"
        "    os.write(1, b'FD1_C_NOISE\\n')\n"
        "    return orig(self, *a, **kw)\n"
        "mcp_server.DCESessionStore.ingest_media = noisy\n"
        "mcp_server.run_stdio_server()\n"
    )
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "dce_ingest_media",
            "arguments": {
                "media_path": str(wav_path),
                "use_asr": False,
                "transcript_override": [{"turn_id": 1, "start_sec": 0.0, "end_sec": 1.5, "text": "Hello world."}],
            },
        },
    }
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        input=json.dumps(req) + "\n",
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=10,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(out_lines) == 1, f"Expected 1 clean JSON-RPC line on stdout, got: {out_lines}"
    parsed = json.loads(out_lines[0])
    assert parsed["id"] == 7
    assert parsed["result"]["isError"] is False
    assert "PY_STDOUT_NOISE" in proc.stderr
    assert "FD1_C_NOISE" in proc.stderr


def test_stdio_content_length_framing_and_stdout_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Content-Length framing works and print()/logging inside tool stays off stdout."""
    body = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}}).encode("utf-8")
    frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body

    fake_stdin = io.TextIOWrapper(io.BytesIO(frame), encoding="utf-8")
    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()

    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    run_stdio_server(store=DCESessionStore())

    out_text = fake_stdout.getvalue()
    assert out_text.startswith("Content-Length:")
    _, json_part = out_text.split("\r\n\r\n", 1)
    parsed = json.loads(json_part)
    assert parsed["id"] == 10
    assert len(parsed["result"]["tools"]) == 5


def test_firewall_state_machine_and_phonetic_downgrade(tmp_path: Path) -> None:
    """Verify full 4-tool workflow: ingest -> firewall block -> verdicts -> unlock -> slice."""
    wav_path = _write_test_wav(tmp_path / "sample.wav", duration_sec=6.0)
    store = DCESessionStore()

    # 1. Ingest media with low-confidence phonetic words and incomplete sentence
    ingest_res = store.ingest_media(
        media_path=str(wav_path),
        session_id="sess-test-1",
        shrinkage_k=0.0,
        use_asr=False,
        transcript_override=[
            {
                "turn_id": 1,
                "start_sec": 0.0,
                "end_sec": 4.5,
                "text": "I can meet at 15 and",
                "words": [
                    {"word": "I", "start": 0.0, "end": 0.2, "probability": 0.99},
                    {"word": "can", "start": 0.2, "end": 0.5, "probability": 0.52},
                    {"word": "meet", "start": 0.5, "end": 0.8, "probability": 0.95},
                    {"word": "at", "start": 0.8, "end": 1.0, "probability": 0.96},
                    {"word": "15", "start": 1.0, "end": 1.4, "probability": 0.61},
                    {"word": "and", "start": 1.4, "end": 1.7, "probability": 0.91},
                ],
            },
            {
                "turn_id": 2,
                "start_sec": 4.5,
                "end_sec": 6.0,
                "text": "No I didn't go to Tokyo.",
                "words": [
                    {"word": "No", "start": 4.5, "end": 4.7, "probability": 0.98},
                    {"word": "I", "start": 4.7, "end": 4.8, "probability": 0.98},
                    {"word": "didn't", "start": 4.8, "end": 5.1, "probability": 0.68},
                    {"word": "go", "start": 5.1, "end": 5.3, "probability": 0.97},
                    {"word": "to", "start": 5.3, "end": 5.5, "probability": 0.97},
                    {"word": "Tokyo.", "start": 5.5, "end": 6.0, "probability": 0.99},
                ],
            },
        ],
    )

    assert ingest_res["status"] == "INGESTED"
    assert ingest_res["turn_count"] == 2
    t1, t2 = ingest_res["turns"]
    assert "[can? p=0.52 | alt: can't]" in t1["annotated_text"]
    assert "[15? p=0.61 | alt: 50]" in t1["annotated_text"]
    assert t1["incomplete_sentence"] is True
    assert "[didn't? p=0.68 | alt: did]" in t2["annotated_text"]
    assert t2["incomplete_sentence"] is False

    # Firewall check: zero acoustic/Mahalanobis terms in ingest output
    assert_firewall_compliance(json.dumps(ingest_res))

    # 2. Unlock MUST be refused before dce_submit_verdicts
    with pytest.raises(PermissionError, match="FIREWALL_LOCKED"):
        store.unlock_signals(session_id="sess-test-1")

    # Also verify via JSON-RPC handle_jsonrpc_request returns isError=True
    err_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "dce_unlock_signals", "arguments": {"session_id": "sess-test-1"}},
        },
        store=store,
    )
    assert err_resp is not None
    assert err_resp["result"]["isError"] is True
    assert "FIREWALL_LOCKED" in err_resp["result"]["content"][0]["text"]

    # 3. Submit verdicts: invalid turn_id, phonetic-fragile conflict (did/didn't & 15/50), and clean consistent
    verdict_res = store.submit_verdicts(
        session_id="sess-test-1",
        claims=[
            {
                "claim_id": "c1",
                "turn_id": 1,
                "speaker_id": "speaker_0",
                "raw_subject": "I",
                "canonical_entity": "meeting_time",
                "predicate": "meet_at",
                "object_value": "15",
            },
            {
                "claim_id": "c2",
                "turn_id": 2,
                "speaker_id": "speaker_0",
                "raw_subject": "I",
                "canonical_entity": "meeting_time",
                "predicate": "meet_at",
                "object_value": "50",
            },
        ],
        verdicts=[
            {
                "verdict": "DIRECT_CONTRADICTION",
                "source_turn_id": 2,
                "prior_turn_id": 1,
                "source_claim_id": "c2",
                "prior_claim_id": "c1",
                "explanation": "15 vs 50 conflict",
            },
            {
                "verdict": "DIRECT_CONTRADICTION",
                "source_turn_id": 2,
                "prior_turn_id": 999,
                "explanation": "Hallucinated turn_id",
            },
        ],
    )

    assert verdict_res["status"] == "VERDICTS_LOCKED"
    assert verdict_res["verdicts_locked"] is True
    assert verdict_res["downgraded_count"] == 1
    assert verdict_res["rejected_invalid_ref_count"] == 1

    v0, v1 = verdict_res["validated_verdicts"]
    assert v0["verdict"] == "NEEDS_AUDIO_REVIEW"
    assert v0["review_status"] == "NEEDS_HUMAN_REVIEW"
    assert v0["downgraded_from"] == "DIRECT_CONTRADICTION"
    assert v1["verdict"] == "REJECTED_INVALID_REF"

    # 4. Unlock signals now succeeds and writes .jsonl replay
    jsonl_out = tmp_path / "replay.jsonl"
    unlock_res = store.unlock_signals(
        session_id="sess-test-1",
        output_jsonl_path=str(jsonl_out),
    )
    assert unlock_res["status"] == "SIGNALS_UNLOCKED"
    assert unlock_res["shrinkage_k_used"] == 0.0
    assert jsonl_out.is_file()
    jsonl_lines = [json.loads(ln) for ln in jsonl_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(jsonl_lines) == 2
    assert jsonl_lines[0]["turn_id"] == 1
    assert "mahalanobis" in unlock_res["turns"][0]
    assert "physical_explanation" in unlock_res["turns"][0]

    # 5. Slice 3-second WAV clip for turn 1 (which is 4.5s long)
    slice_res = store.slice_turn_audio(
        session_id="sess-test-1",
        turn_id=1,
        duration_sec=3.0,
        output_dir=str(tmp_path / "clips"),
    )
    wav_clip = Path(slice_res["wav_path"])
    assert wav_clip.is_file()
    assert slice_res["duration_sec"] == 3.0
    with wave.open(str(wav_clip), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 48000


def test_did_didnt_phonetic_fragile_downgrade(tmp_path: Path) -> None:
    """Verify did vs didn't contradiction is downgraded to NEEDS_AUDIO_REVIEW / NEEDS_HUMAN_REVIEW."""
    wav_path = _write_test_wav(tmp_path / "did.wav", duration_sec=2.0)
    store = DCESessionStore()
    store.ingest_media(
        media_path=str(wav_path),
        session_id="sess-did",
        use_asr=False,
        transcript_override=[
            {"turn_id": 1, "start_sec": 0.0, "end_sec": 1.0, "text": "I did sign the form."},
            {"turn_id": 2, "start_sec": 1.0, "end_sec": 2.0, "text": "I didn't sign the form."},
        ],
    )
    res = store.submit_verdicts(
        session_id="sess-did",
        verdicts=[
            {
                "verdict": "DIRECT_CONTRADICTION",
                "source_turn_id": 2,
                "prior_turn_id": 1,
                "explanation": "Speaker said did vs didn't.",
            }
        ],
    )
    assert res["downgraded_count"] == 1
    assert res["validated_verdicts"][0]["verdict"] == "NEEDS_AUDIO_REVIEW"
    assert res["validated_verdicts"][0]["review_status"] == "NEEDS_HUMAN_REVIEW"


def test_missing_media_and_unknown_tool_errors(tmp_path: Path) -> None:
    """Verify clean error handling for missing media files and unknown tools."""
    store = DCESessionStore()
    missing_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 50,
            "method": "tools/call",
            "params": {
                "name": "dce_ingest_media",
                "arguments": {"media_path": str(tmp_path / "nonexistent.wav")},
            },
        },
        store=store,
    )
    assert missing_resp is not None
    assert missing_resp["result"]["isError"] is True
    assert "not found" in missing_resp["result"]["content"][0]["text"].lower()

    unknown_resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 51,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        },
        store=store,
    )
    assert unknown_resp is not None
    assert unknown_resp["error"]["code"] == -32601
    assert len(MCP_TOOLS) == 5


def test_single_turn_low_conf_downgrade_and_human_review_alias(tmp_path: Path) -> None:
    """Verify prior_turn_id=None still checks low_confidence_words and NEEDS_HUMAN_REVIEW maps cleanly."""
    wav_path = _write_test_wav(tmp_path / "single.wav", duration_sec=2.0)
    store = DCESessionStore()
    store.ingest_media(
        media_path=str(wav_path),
        session_id="sess-single",
        use_asr=False,
        transcript_override=[
            {
                "turn_id": 1,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "text": "I can go tomorrow.",
                "words": [
                    {"word": "I", "start": 0.0, "end": 0.2, "probability": 0.99},
                    {"word": "can", "start": 0.2, "end": 0.5, "probability": 0.51},
                    {"word": "go", "start": 0.5, "end": 0.8, "probability": 0.96},
                    {"word": "tomorrow.", "start": 0.8, "end": 1.5, "probability": 0.98},
                ],
            }
        ],
    )
    res = store.submit_verdicts(
        session_id="sess-single",
        verdicts=[
            {
                "verdict": "DIRECT_CONTRADICTION",
                "source_turn_id": 1,
                "prior_turn_id": None,
                "explanation": "Self-contradiction without prior_turn_id.",
            },
            {
                "verdict": "NEEDS_HUMAN_REVIEW",
                "source_turn_id": 1,
                "prior_turn_id": None,
                "explanation": "Explicit human review request.",
            },
        ],
    )
    assert res["downgraded_count"] == 1
    assert res["validated_verdicts"][0]["verdict"] == "NEEDS_AUDIO_REVIEW"
    assert res["validated_verdicts"][0]["review_status"] == "NEEDS_HUMAN_REVIEW"
    assert res["validated_verdicts"][1]["verdict"] == "NEEDS_AUDIO_REVIEW"
