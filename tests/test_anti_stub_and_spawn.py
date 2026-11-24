"""Anti-stub verification, lazy import enforcement, Windows spawn IPC, and JSON safety tests.

Audits all 10 core DCE modules to guarantee zero stubbing (no pass, no Ellipsis, no TODO,
no NotImplementedError outside of Protocol/abstractmethod/empty Exception), verifies that
heavy dependencies are only imported inside functions, executes true Windows spawn IPC,
and validates that FeatureRecorder rejects NaN and Inf via allow_nan=False.
"""

from __future__ import annotations

import ast
import math
import multiprocessing
from pathlib import Path
from typing import List

import pytest

from dce.pipeline import audio_worker_process
from dce.recorder import FeatureRecorder
from dce.schemas import (
    ProsodyFeatures,
    RecorderTurnRecord,
    SpeakerAssignment,
    TimingFeatures,
    TurnFeatures,
    VisionFeatures,
)

CORE_FILES: List[str] = [
    "schemas.py",
    "clock.py",
    "math_core.py",
    "baseline_store.py",
    "speaker_registry.py",
    "vad_endpointing.py",
    "recorder.py",
    "pipeline.py",
    "extractors/audio_dsp.py",
    "extractors/vision_face.py",
    "reasoning/templates.py",
    "reasoning/llm_adapter.py",
    "reasoning/claims_ledger.py",
]

HEAVY_PACKAGES = {
    "faster_whisper",
    "parselmouth",
    "onnxruntime",
    "mediapipe",
    "cv2",
    "httpx",
    "ollama",
    "google.genai",
    "google",
    "streamlit",
    "plotly",
}


def test_anti_stub_ast_audit_all_10_core_modules() -> None:
    """AST audit checking all 10 core files for zero pass, ..., TODO, or NotImplementedError."""
    project_root = Path(__file__).resolve().parent.parent
    src_dce = project_root / "src" / "dce"

    for rel_path in CORE_FILES:
        file_path = src_dce / rel_path
        assert file_path.is_file(), f"Core file missing: {rel_path}"

        source_code = file_path.read_text(encoding="utf-8")
        assert "TODO" not in source_code, f"Core file {rel_path} contains 'TODO'"

        tree = ast.parse(source_code, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine if function has @abstractmethod
                is_abstract = any(
                    (isinstance(d, ast.Name) and d.id == "abstractmethod")
                    or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                    for d in node.decorator_list
                )

                for child in node.body:
                    if isinstance(child, ast.Pass):
                        assert is_abstract, (
                            f"Illegal 'pass' found in non-abstract function {node.name} at {rel_path}:{child.lineno}"
                        )

                    if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                        if child.value.value is Ellipsis:
                            assert is_abstract, (
                                f"Illegal '...' (Ellipsis) found in non-abstract function {node.name} at {rel_path}:{child.lineno}"
                            )

                    if isinstance(child, ast.Raise):
                        if isinstance(child.exc, ast.Name) and child.exc.id == "NotImplementedError":
                            assert is_abstract, (
                                f"Illegal NotImplementedError found in non-abstract function {node.name} at {rel_path}:{child.lineno}"
                            )
                        elif (
                            isinstance(child.exc, ast.Call)
                            and isinstance(child.exc.func, ast.Name)
                            and child.exc.func.id == "NotImplementedError"
                        ):
                            assert is_abstract, (
                                f"Illegal NotImplementedError call found in non-abstract function {node.name} at {rel_path}:{child.lineno}"
                            )


def test_lazy_imports_enforcement() -> None:
    """Verify that heavy libraries are never imported at module level in src/."""
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src" / "dce"

    violations: List[str] = []
    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for stmt in tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in HEAVY_PACKAGES or alias.name in HEAVY_PACKAGES:
                        violations.append(f"{py_file.name}:{stmt.lineno} imports {alias.name}")
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    root_pkg = stmt.module.split(".")[0]
                    if root_pkg in HEAVY_PACKAGES or stmt.module in HEAVY_PACKAGES:
                        violations.append(f"{py_file.name}:{stmt.lineno} imports from {stmt.module}")

    assert not violations, f"Heavy packages must be lazy-imported inside functions: {violations}"


def test_windows_spawn_multiprocessing_ipc() -> None:
    """Verify audio_worker_process operates under multiprocessing spawn context."""
    ctx = multiprocessing.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()

    proc = ctx.Process(target=audio_worker_process, args=(in_q, out_q))
    proc.start()

    try:
        # Send synthetic audio packet: (pts_ns, pcm_f32_samples)
        sample_audio = [0.1, -0.2, 0.3, -0.4, 0.5, 0.0]
        pts_test = 1_000_000_000
        in_q.put((pts_test, sample_audio))

        # Retrieve processed result with 5.0s timeout
        res = out_q.get(timeout=5.0)
        assert res is not None
        assert res.get("status") == "OK"
        assert res.get("pts_ns") == pts_test
        assert res.get("samples_count") == len(sample_audio)
        assert "rms" in res and res["rms"] > 0.0

        # Terminate cleanly with STOP
        in_q.put("STOP")
        stop_res = out_q.get(timeout=5.0)
        assert stop_res.get("status") == "STOPPED"

    finally:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
        assert not proc.is_alive()


def test_feature_recorder_rejects_nan_and_inf(tmp_path: Path) -> None:
    """Verify FeatureRecorder strictly enforces allow_nan=False when serializing."""
    record_file = tmp_path / "test_safety.jsonl"
    recorder = FeatureRecorder(record_file)

    # 1. Valid record without NaN/Inf should succeed
    clean_record = RecorderTurnRecord(
        turn_id=1,
        speaker_assignment=SpeakerAssignment(
            speaker_id="SPEAKER_00",
            similarity=1.0,
            is_provisional=False,
            low_confidence=False,
        ),
        features=TurnFeatures(
            turn_id=1,
            speaker_id="SPEAKER_00",
            start_ns=100_000,
            end_ns=900_000,
            timing=TimingFeatures(response_latency_open=0.45),
            prosody=ProsodyFeatures(articulation_rate=4.2),
            vision=VisionFeatures(blink_rate_per_min=15.0),
        ),
    )
    recorder.write_record(clean_record)
    recorder.flush()
    recorder.close()

    assert record_file.is_file()
    lines = record_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    # 2. Raw serialization with float('nan') or float('inf') into FeatureRecorder
    # By contract, if an un-sanitized dict containing NaN is provided to json.dumps(..., allow_nan=False),
    # it must raise ValueError
    class MockInvalidRecord:
        def model_dump(self, mode: str = "json"):
            return {
                "turn_id": 2,
                "illegal_nan": float("nan"),
            }

    recorder_bad = FeatureRecorder(tmp_path / "bad.jsonl")
    with pytest.raises(ValueError, match="Out of range float values"):
        recorder_bad.write_record(MockInvalidRecord())  # type: ignore

    class MockInvalidInfRecord:
        def model_dump(self, mode: str = "json"):
            return {
                "turn_id": 3,
                "illegal_inf": float("inf"),
            }

    with pytest.raises(ValueError, match="Out of range float values"):
        recorder_bad.write_record(MockInvalidInfRecord())  # type: ignore

    recorder_bad.close()
