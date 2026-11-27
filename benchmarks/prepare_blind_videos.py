"""Part 3 Prep: Anonymize 12 Test/ + 18 Train/ (9T+9L) courtroom videos.

Creates shuffled anonymized filenames (anon_clip_01..12, anon_ext_01..18) in
benchmarks/results/_anon_videos/ and writes the ground-truth mapping ONLY to
benchmarks/results/_hidden_video_key.json for post-lock scoring.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "dataset" / "archive" / "Real-life Deception Detection Dataset With Train Test" / "Test"
TRAIN_DIR = REPO_ROOT / "dataset" / "archive" / "Real-life Deception Detection Dataset With Train Test" / "Train"
ANON_DIR = REPO_ROOT / "benchmarks" / "results" / "_anon_videos"
HIDDEN_KEY_PATH = REPO_ROOT / "benchmarks" / "results" / "_hidden_video_key.json"
BLIND_LIST_PATH = REPO_ROOT / "benchmarks" / "results" / "video_blind_manifest.json"


def _label_from_name(name: str) -> str:
    low = name.lower()
    if "lie" in low:
        return "deceptive"
    if "truth" in low:
        return "truthful"
    raise ValueError(f"Cannot infer label from {name}")


def main() -> None:
    rng = random.Random(42)
    ANON_DIR.mkdir(parents=True, exist_ok=True)

    # All 12 Test/ clips
    test_mp4s = sorted(TEST_DIR.glob("*.mp4"))
    assert len(test_mp4s) == 12, f"Expected 12 Test mp4s, got {len(test_mp4s)}"
    rng.shuffle(test_mp4s)

    # 18 longest Train/ clips (9 Truthful + 9 Deceptive)
    train_truth = sorted(
        TRAIN_DIR.glob("trial_truth_*.mp4"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )[:9]
    train_lie = sorted(
        TRAIN_DIR.glob("trial_lie_*.mp4"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )[:9]
    assert len(train_truth) == 9 and len(train_lie) == 9
    train_18 = train_truth + train_lie
    rng.shuffle(train_18)

    hidden_key: Dict[str, Dict[str, Any]] = {}
    blind_manifest: List[Dict[str, str]] = []

    def _stage(src: Path, anon_id: str, split: str) -> None:
        dst = ANON_DIR / f"{anon_id}.mp4"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        hidden_key[anon_id] = {
            "anon_id": anon_id,
            "split": split,
            "original_filename": src.name,
            "ground_truth": _label_from_name(src.name),
            "size_bytes": src.stat().st_size,
        }
        blind_manifest.append(
            {
                "anon_id": anon_id,
                "split": split,
                "anon_path": str(dst),
            }
        )

    for idx, p in enumerate(test_mp4s, start=1):
        _stage(p, f"anon_clip_{idx:02d}", "test_12")

    for idx, p in enumerate(train_18, start=1):
        _stage(p, f"anon_ext_{idx:02d}", "train_ext_18")

    HIDDEN_KEY_PATH.write_text(json.dumps(hidden_key, indent=2), encoding="utf-8")
    BLIND_LIST_PATH.write_text(json.dumps(blind_manifest, indent=2), encoding="utf-8")
    print(f"Staged {len(blind_manifest)} anonymized videos in {ANON_DIR}")
    print(f"Blind manifest -> {BLIND_LIST_PATH}")
    print(f"Hidden key -> {HIDDEN_KEY_PATH} (DO NOT READ UNTIL VERDICTS LOCKED)")


if __name__ == "__main__":
    main()
