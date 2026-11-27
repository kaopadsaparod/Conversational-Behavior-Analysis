"""Part 3 & 4: Full 4-Tool Stateful MCP Video Pipeline + Hardware Profiling.

Phases:
  --phase ingest : Runs dce_ingest_media (use_asr=True, small.en, 15 FPS, MediaPipe,
                   Praat+eGeMAPS, priors_wild_video_en.json) on all 30 anonymized videos.
                   Saves firewall-safe transcripts to video_blind_transcripts.json,
                   hardware/runtime profiling to video_profiling.json, and session
                   state to _session_store_state.pkl.
  --phase unlock : Loads _session_store_state.pkl and video_blind_verdicts.json,
                   calls dce_submit_verdicts -> dce_unlock_signals -> dce_slice_turn_audio
                   (where applicable), saving unlocked signals to video_unlocked_signals.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

from dce.extractors.audio_dsp import decode_media_audio_and_frames, extract_prosody, transcribe_audio
from dce.extractors.vision_face import extract_vision_sequence
from dce.mcp_server import DCESessionStore

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
MANIFEST_PATH = RESULTS_DIR / "video_blind_manifest.json"
BLIND_TRANSCRIPTS_PATH = RESULTS_DIR / "video_blind_transcripts.json"
PROFILING_PATH = RESULTS_DIR / "video_profiling.json"
SESSION_STATE_PATH = RESULTS_DIR / "_session_store_state.pkl"
BLIND_VERDICTS_PATH = RESULTS_DIR / "video_blind_verdicts.json"
UNLOCKED_SIGNALS_PATH = RESULTS_DIR / "video_unlocked_signals.json"
REPLAY_DIR = RESULTS_DIR / "replays"
SLICE_DIR = RESULTS_DIR / "audio_slices"


def _profile_component_breakdown(sample_path: Path) -> Dict[str, Any]:
    """Measure PyAV decode, MediaPipe 15 FPS, Praat+eGeMAPS, and Whisper small.en breakdown."""
    t0 = time.perf_counter()
    audio, sr, frames = decode_media_audio_and_frames(sample_path, target_sr=16000, video_fps=15.0)
    t_decode = time.perf_counter() - t0
    dur_sec = len(audio) / float(max(sr, 1))

    t1 = time.perf_counter()
    if frames:
        _ = extract_vision_sequence(frames, fps=15.0)
    t_vision = time.perf_counter() - t1

    t2 = time.perf_counter()
    _ = extract_prosody(audio[: min(len(audio), 5 * sr)], sample_rate=sr)
    t_praat = time.perf_counter() - t2

    t3 = time.perf_counter()
    _ = transcribe_audio(audio, sample_rate=sr, model_size="small.en", device="cpu")
    t_whisper = time.perf_counter() - t3

    return {
        "sample_file": sample_path.name,
        "media_duration_sec": round(dur_sec, 3),
        "n_frames_15fps": len(frames),
        "decode_av_sec": round(t_decode, 3),
        "mediapipe_15fps_sec": round(t_vision, 3),
        "praat_egemaps_5s_sec": round(t_praat, 3),
        "whisper_small_en_cpu_sec": round(t_whisper, 3),
    }


def run_ingest() -> None:
    manifest: List[Dict[str, str]] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    store = DCESessionStore()
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # prime cpu_percent

    # Component breakdown on first clip (includes model warmup) and second clip (warm cache)
    breakdown_cold = _profile_component_breakdown(Path(manifest[0]["anon_path"]))
    breakdown_warm = _profile_component_breakdown(Path(manifest[1]["anon_path"]))

    blind_transcripts: List[Dict[str, Any]] = []
    per_clip_prof: List[Dict[str, Any]] = []
    total_video_sec = 0.0
    total_wall_sec = 0.0
    peak_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    t_all_start = time.perf_counter()
    for item in manifest:
        anon_id = item["anon_id"]
        split = item["split"]
        anon_path = Path(item["anon_path"])

        t0 = time.perf_counter()
        res = store.ingest_media(
            media_path=str(anon_path),
            session_id=anon_id,
            shrinkage_k=0.0,
            low_conf_threshold=0.70,
            asr_device="cpu",
            asr_model="small.en",
            use_asr=True,
        )
        wall_sec = time.perf_counter() - t0
        rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        peak_rss_mb = max(peak_rss_mb, rss_mb)

        turns = res.get("turns", [])
        clip_dur_sec = max((t.get("end_sec", 0.0) for t in turns), default=0.0)
        total_video_sec += clip_dur_sec
        total_wall_sec += wall_sec

        blind_transcripts.append(
            {
                "anon_id": anon_id,
                "split": split,
                "session_id": res["session_id"],
                "turn_count": res["turn_count"],
                "turns": turns,
            }
        )
        per_clip_prof.append(
            {
                "anon_id": anon_id,
                "split": split,
                "turn_count": res["turn_count"],
                "clip_duration_sec": round(clip_dur_sec, 3),
                "wall_sec": round(wall_sec, 3),
                "rtf": round(wall_sec / max(clip_dur_sec, 0.1), 3),
                "rss_mb": round(rss_mb, 1),
            }
        )
        print(
            f"  Ingested {anon_id} ({split}) | turns={res['turn_count']} | "
            f"dur={clip_dur_sec:.1f}s | wall={wall_sec:.2f}s | rss={rss_mb:.0f}MB"
        )

    total_elapsed = time.perf_counter() - t_all_start
    cpu_pct = proc.cpu_percent(interval=None)

    BLIND_TRANSCRIPTS_PATH.write_text(
        json.dumps(blind_transcripts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    profiling = {
        "total_clips": len(manifest),
        "total_video_duration_sec": round(total_video_sec, 2),
        "total_ingest_wall_sec": round(total_wall_sec, 2),
        "overall_wall_sec": round(total_elapsed, 2),
        "overall_rtf": round(total_wall_sec / max(total_video_sec, 0.1), 3),
        "sec_per_min_video": round((total_wall_sec / max(total_video_sec, 0.1)) * 60.0, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "process_cpu_percent_last": round(cpu_pct, 1),
        "component_breakdown_cold": breakdown_cold,
        "component_breakdown_warm": breakdown_warm,
        "per_clip": per_clip_prof,
    }
    PROFILING_PATH.write_text(json.dumps(profiling, indent=2), encoding="utf-8")

    with SESSION_STATE_PATH.open("wb") as f:
        pickle.dump(store._sessions, f)

    print(
        f"Ingest complete: {len(manifest)} clips, {total_video_sec:.1f}s video in "
        f"{total_wall_sec:.1f}s (RTF={profiling['overall_rtf']}, peak_rss={peak_rss_mb:.0f}MB)"
    )


def run_unlock() -> None:
    if not SESSION_STATE_PATH.is_file():
        raise FileNotFoundError(f"Missing {SESSION_STATE_PATH}. Run --phase ingest first.")
    if not BLIND_VERDICTS_PATH.is_file():
        raise FileNotFoundError(f"Missing {BLIND_VERDICTS_PATH}. Submit blind verdicts first.")

    store = DCESessionStore()
    with SESSION_STATE_PATH.open("rb") as f:
        store._sessions = pickle.load(f)

    verdicts_map: Dict[str, Dict[str, Any]] = json.loads(
        BLIND_VERDICTS_PATH.read_text(encoding="utf-8")
    )

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    unlocked_all: Dict[str, Any] = {}
    for anon_id, payload in verdicts_map.items():
        sid = payload["session_id"]
        verdicts = payload.get("verdicts", [])
        claims = payload.get("claims", [])

        sub_res = store.submit_verdicts(session_id=sid, verdicts=verdicts, claims=claims)
        replay_path = REPLAY_DIR / f"{anon_id}_replay.jsonl"
        unlock_res = store.unlock_signals(session_id=sid, output_jsonl_path=str(replay_path))

        # Step 4: slice audio for any turn with NEEDS_AUDIO_REVIEW / NEEDS_HUMAN_REVIEW
        # or low-confidence phonetic words, plus first turn of first 3 clips as smoke check
        sliced_turns: List[Dict[str, Any]] = []
        needs_review_tids = {
            v["source_turn_id"]
            for v in sub_res.get("validated_verdicts", [])
            if v.get("verdict") in ("NEEDS_AUDIO_REVIEW", "NEEDS_HUMAN_REVIEW")
            or v.get("review_status") == "NEEDS_HUMAN_REVIEW"
        }
        for tid in sorted(needs_review_tids):
            try:
                sl = store.slice_turn_audio(
                    session_id=sid, turn_id=int(tid), duration_sec=3.0, output_dir=str(SLICE_DIR)
                )
                sliced_turns.append(sl)
            except Exception as exc:
                sliced_turns.append({"turn_id": tid, "error": str(exc)})

        unlocked_all[anon_id] = {
            "session_id": sid,
            "submit_result": sub_res,
            "unlock_result": unlock_res,
            "sliced_audio": sliced_turns,
        }

    UNLOCKED_SIGNALS_PATH.write_text(
        json.dumps(unlocked_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Unlocked {len(unlocked_all)} sessions -> {UNLOCKED_SIGNALS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["ingest", "unlock"], required=True)
    args = parser.parse_args()
    if args.phase == "ingest":
        run_ingest()
    else:
        run_unlock()


if __name__ == "__main__":
    main()
