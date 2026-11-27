"""Zero-Cheat Blind Mixed-Dataset Video Benchmark (RLDD Test + Werewolf-Among-Us).

Combines:
  - 12 held-out RLDD Test/ clips (6 deceptive, 6 truthful)
  - 12 Werewolf-Among-Us Youtube/split/test.json clips (6 Werewolf/deceptive, 6 Villager-team/truthful)

Phases:
  --phase prepare_and_ingest : Slices/copies 24 clips into anonymized mixed_01..24.mp4,
                               writes _hidden_mixed_video_key.json, runs dce_ingest_media,
                               and saves firewall-safe mixed_video_blind_transcripts.json.
  --phase unlock_and_score   : Reads mixed_video_blind_verdicts.json, runs dce_submit_verdicts
                               + dce_unlock_signals, and scores Mode A, Mode B, Mode C overall
                               and per dataset (rldd_test vs werewolf_youtube).
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np

from dce.mcp_server import DCESessionStore

REPO_ROOT = Path(__file__).resolve().parent.parent
RLDD_TEST_DIR = REPO_ROOT / "dataset" / "archive" / "Real-life Deception Detection Dataset With Train Test" / "Test"
WEREWOLF_YT_DIR = REPO_ROOT / "dataset" / "Werewolf-Among-Us" / "Youtube"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
BLIND_CLIPS_DIR = RESULTS_DIR / "_mixed_blind_clips"
MANIFEST_PATH = RESULTS_DIR / "mixed_video_manifest.json"
HIDDEN_KEY_PATH = RESULTS_DIR / "_hidden_mixed_video_key.json"
TRANSCRIPTS_PATH = RESULTS_DIR / "mixed_video_blind_transcripts.json"
SESSION_STATE_PATH = RESULTS_DIR / "_mixed_session_store_state.pkl"
VERDICTS_PATH = RESULTS_DIR / "mixed_video_blind_verdicts.json"
SCORES_OUT_PATH = RESULTS_DIR / "mixed_video_benchmark_scores.json"

WEREWOLF_EVIL_ROLES = {"Werewolf", "Minion"}
WEREWOLF_GOOD_ROLES = {"Villager", "Seer", "Robber", "Troublemaker", "Insomniac", "Mason", "Drunk", "Hunter", "Tanner"}


def _parse_mmss(ts: str) -> float:
    parts = [p.strip() for p in ts.split(":")]
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    return 0.0


def _slice_mp4_segment(src_path: Path, dst_path: Path, start_sec: float, duration_sec: float = 25.0) -> None:
    """Slice a clean [start_sec, start_sec + duration_sec] MP4 (video+audio) using PyAV."""
    in_container = av.open(str(src_path))
    out_container = av.open(str(dst_path), mode="w")

    in_v = in_container.streams.video[0] if in_container.streams.video else None
    in_a = in_container.streams.audio[0] if in_container.streams.audio else None

    out_v = None
    if in_v is not None:
        out_v = out_container.add_stream("libx264", rate=15)
        out_v.width = min(in_v.width, 640)
        out_v.height = int(round(in_v.height * (out_v.width / max(in_v.width, 1)))) // 2 * 2
        out_v.pix_fmt = "yuv420p"
        out_v.time_base = Fraction(1, 15)

    out_a = None
    if in_a is not None:
        out_a = out_container.add_stream("aac", rate=16000)
        out_a.layout = "mono"
        out_a.time_base = Fraction(1, 16000)

    end_sec = start_sec + duration_sec
    in_container.seek(int(max(0.0, start_sec - 1.0) * av.time_base))

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000) if out_a is not None else None
    v_pts = 0
    a_pts = 0

    for packet in in_container.demux([s for s in (in_v, in_a) if s is not None]):
        for frame in packet.decode():
            if frame.pts is None or frame.time_base is None:
                continue
            t_sec = float(frame.pts * frame.time_base)
            if t_sec < start_sec:
                continue
            if t_sec > end_sec:
                break
            if isinstance(frame, av.VideoFrame) and out_v is not None:
                v_resized = frame.reformat(width=out_v.width, height=out_v.height, format="yuv420p")
                v_resized.pts = v_pts
                v_resized.time_base = Fraction(1, 15)
                v_pts += 1
                for enc_pkt in out_v.encode(v_resized):
                    out_container.mux(enc_pkt)
            elif isinstance(frame, av.AudioFrame) and out_a is not None and resampler is not None:
                for res_f in resampler.resample(frame):
                    res_f.pts = a_pts
                    res_f.time_base = Fraction(1, 16000)
                    a_pts += res_f.samples
                    for enc_pkt in out_a.encode(res_f):
                        out_container.mux(enc_pkt)
        # Early stop check
        if packet.pts is not None and packet.time_base is not None:
            if float(packet.pts * packet.time_base) > end_sec + 1.0:
                break

    if out_v is not None:
        for enc_pkt in out_v.encode(None):
            out_container.mux(enc_pkt)
    if out_a is not None:
        for enc_pkt in out_a.encode(None):
            out_container.mux(enc_pkt)

    out_container.close()
    in_container.close()


def _select_werewolf_clips(n_per_class: int = 6) -> List[Dict[str, Any]]:
    test_items = json.loads((WEREWOLF_YT_DIR / "split" / "test.json").read_text(encoding="utf-8"))
    deceptive_cands: List[Dict[str, Any]] = []
    truthful_cands: List[Dict[str, Any]] = []
    seen_games = set()

    for item in test_items:
        vname = item["video_name"]
        gid = item["Game_ID"]
        mp4_path = WEREWOLF_YT_DIR / "videos" / f"{vname}_{gid}.mp4"
        vote_path = WEREWOLF_YT_DIR / "vote_outcome_youtube_released" / f"{vname}.json"
        if not mp4_path.is_file() or not vote_path.is_file():
            continue

        vote_data = json.loads(vote_path.read_text(encoding="utf-8")).get(gid, {})
        players = vote_data.get("playerNames", [])
        roles = vote_data.get("endRoles", [])
        if not players or len(players) != len(roles):
            continue
        role_map = {p: r for p, r in zip(players, roles)}

        for d in item.get("Dialogue", []):
            spk = d.get("speaker", "")
            anns = set(d.get("annotation", []))
            if spk not in role_map:
                continue
            if not (anns & {"Defense", "Identity Declaration", "Evidence"}):
                continue
            role = role_map[spk]
            t_sec = _parse_mmss(d.get("timestamp", "00:00"))
            key = (vname, gid, spk)
            if key in seen_games:
                continue

            entry = {
                "dataset": "werewolf_youtube",
                "src_path": mp4_path,
                "start_sec": max(0.0, t_sec - 2.0),
                "duration_sec": 25.0,
                "speaker": spk,
                "role": role,
                "orig_id": f"{vname}_{gid}_{spk}",
            }
            if role in WEREWOLF_EVIL_ROLES and len(deceptive_cands) < n_per_class:
                entry["ground_truth"] = "deceptive"
                deceptive_cands.append(entry)
                seen_games.add(key)
            elif role in WEREWOLF_GOOD_ROLES and len(truthful_cands) < n_per_class:
                entry["ground_truth"] = "truthful"
                truthful_cands.append(entry)
                seen_games.add(key)

        if len(deceptive_cands) >= n_per_class and len(truthful_cands) >= n_per_class:
            break

    return deceptive_cands + truthful_cands


def run_prepare_and_ingest(skip_prepare: bool = False) -> None:
    if not skip_prepare:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if BLIND_CLIPS_DIR.exists():
            shutil.rmtree(BLIND_CLIPS_DIR)
        BLIND_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

        all_entries: List[Dict[str, Any]] = []
        # 1. RLDD Test/ (12 clips)
        for mp4 in sorted(RLDD_TEST_DIR.glob("*.mp4")):
            gt = "deceptive" if "lie" in mp4.stem.lower() else "truthful"
            all_entries.append(
                {
                    "dataset": "rldd_test",
                    "src_path": mp4,
                    "start_sec": None,
                    "ground_truth": gt,
                    "orig_id": mp4.stem,
                }
            )

        # 2. Werewolf Youtube Test (12 clips: 6 deceptive, 6 truthful)
        ww_entries = _select_werewolf_clips(n_per_class=6)
        all_entries.extend(ww_entries)

        rng = random.Random(42)
        rng.shuffle(all_entries)

        manifest: List[Dict[str, Any]] = []
        hidden_key: Dict[str, Dict[str, Any]] = {}

        for idx, entry in enumerate(all_entries, start=1):
            anon_id = f"mixed_{idx:02d}"
            dst_mp4 = BLIND_CLIPS_DIR / f"{anon_id}.mp4"
            if entry["start_sec"] is None:
                shutil.copy2(entry["src_path"], dst_mp4)
            else:
                _slice_mp4_segment(entry["src_path"], dst_mp4, entry["start_sec"], entry["duration_sec"])

            manifest.append({"anon_id": anon_id, "anon_path": str(dst_mp4)})
            hidden_key[anon_id] = {
                "ground_truth": entry["ground_truth"],
                "dataset": entry["dataset"],
                "orig_id": entry["orig_id"],
            }

        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        HIDDEN_KEY_PATH.write_text(json.dumps(hidden_key, indent=2), encoding="utf-8")
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # Ingest all 24 clips through DCESessionStore
    store = DCESessionStore()
    all_sessions: Dict[str, Any] = {}
    blind_transcripts: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for item in manifest:
        anon_id = item["anon_id"]
        res = store.ingest_media(
            media_path=item["anon_path"],
            session_id=f"sess_{anon_id}",
            shrinkage_k=0.0,
            low_conf_threshold=0.70,
            asr_device="cpu",
            asr_model="small.en",
            use_asr=True,
        )
        all_sessions[res["session_id"]] = store._sessions[res["session_id"]]
        blind_transcripts.append(
            {
                "anon_id": anon_id,
                "session_id": res["session_id"],
                "turn_count": res["turn_count"],
                "turns": res["turns"],
            }
        )
        print(f"Ingested {anon_id}: turns={res['turn_count']}", flush=True)

    elapsed = round(time.perf_counter() - t0, 2)
    TRANSCRIPTS_PATH.write_text(json.dumps(blind_transcripts, indent=2, ensure_ascii=False), encoding="utf-8")
    with SESSION_STATE_PATH.open("wb") as f:
        pickle.dump(all_sessions, f)
    print(f"Prepared & ingested {len(manifest)} mixed clips in {elapsed}s")


def _compute_metrics(preds: List[bool], gts: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for p, g in zip(preds, gts) if p and g)
    fp = sum(1 for p, g in zip(preds, gts) if p and not g)
    tn = sum(1 for p, g in zip(preds, gts) if not p and not g)
    fn = sum(1 for p, g in zip(preds, gts) if not p and g)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    acc = (tp + tn) / max(len(gts), 1)
    return {
        "n": len(gts),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def run_unlock_and_score() -> None:
    hidden_key: Dict[str, Dict[str, Any]] = json.loads(HIDDEN_KEY_PATH.read_text(encoding="utf-8"))
    verdicts_map: Dict[str, Dict[str, Any]] = json.loads(VERDICTS_PATH.read_text(encoding="utf-8"))
    with SESSION_STATE_PATH.open("rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, DCESessionStore):
        store = loaded
    else:
        store = DCESessionStore()
        store._sessions = loaded

    rows: List[Dict[str, Any]] = []
    for anon_id in sorted(hidden_key.keys()):
        gt_info = hidden_key[anon_id]
        is_dec = gt_info["ground_truth"] == "deceptive"
        sid = f"sess_{anon_id}"
        v_entry = verdicts_map.get(anon_id, {})
        raw_v = v_entry.get("verdict", "CONSISTENT")
        submit_list = v_entry.get("verdicts", [])
        if not submit_list and raw_v in ("DIRECT_CONTRADICTION", "TIMELINE_SHIFT"):
            submit_list = [{"source_turn_id": 1, "verdict": raw_v, "explanation": "Blind Mode B verdict."}]
        store.submit_verdicts(sid, verdicts=submit_list)
        unlock_res = store.unlock_signals(sid)

        turns_u = unlock_res["turns"]
        locked_v = unlock_res["locked_verdicts"]

        # Mode A: Sustained physical signals only (never trigger from a single noisy short turn)
        w_sum = 0.0
        d_sum = 0.0
        high_turns = 0
        valid_4f_turns = 0
        for t in turns_u:
            if t.get("speaker_id") == "Q_ASKER":
                continue
            maha = t.get("mahalanobis") or {}
            df = int(maha.get("degrees_of_freedom", 0) or 0)
            d_sq = maha.get("d_sq")
            dur = max(float((t.get("timing") or {}).get("turn_duration_ms") or 0.0), 100.0)
            if df >= 4 and d_sq is not None:
                valid_4f_turns += 1
                ratio = float(d_sq) / float(df)
                w_sum += dur * ratio
                d_sum += dur
                if ratio >= 2.8:
                    high_turns += 1
        mean_d_sq_df = (w_sum / d_sum) if d_sum > 0.0 else 0.0
        mode_a = (mean_d_sq_df >= 1.8) or (high_turns >= 2)

        # Mode B: Semantic reasoning verdicts
        mode_b = (raw_v in ("DIRECT_CONTRADICTION", "TIMELINE_SHIFT", "MODIFIER_EROSION", "EVASION", "DECEPTIVE_EVASION")) or any(
            v.get("verdict") in ("DIRECT_CONTRADICTION", "TIMELINE_SHIFT", "MODIFIER_EROSION", "EVASION", "DECEPTIVE_EVASION")
            for v in locked_v
        )

        # Mode C: Neuro-Symbolic Fusion (Mode B always triggers; Mode A only when sustained across >= 2 valid turns with >= 4 features)
        sustained_phys = (mean_d_sq_df >= 1.8) and (valid_4f_turns >= 2)
        mode_c = mode_b or sustained_phys

        rows.append(
            {
                "anon_id": anon_id,
                "dataset": gt_info["dataset"],
                "ground_truth": gt_info["ground_truth"],
                "is_deceptive": is_dec,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "mode_c": mode_c,
            }
        )

    summary: Dict[str, Any] = {
        "overall": {
            "mode_a": _compute_metrics([r["mode_a"] for r in rows], [r["is_deceptive"] for r in rows]),
            "mode_b": _compute_metrics([r["mode_b"] for r in rows], [r["is_deceptive"] for r in rows]),
            "mode_c": _compute_metrics([r["mode_c"] for r in rows], [r["is_deceptive"] for r in rows]),
        },
        "by_dataset": {},
        "per_clip": rows,
    }
    for ds in ("rldd_test", "werewolf_youtube"):
        sub = [r for r in rows if r["dataset"] == ds]
        summary["by_dataset"][ds] = {
            "mode_a": _compute_metrics([r["mode_a"] for r in sub], [r["is_deceptive"] for r in sub]),
            "mode_b": _compute_metrics([r["mode_b"] for r in sub], [r["is_deceptive"] for r in sub]),
            "mode_c": _compute_metrics([r["mode_c"] for r in sub], [r["is_deceptive"] for r in sub]),
        }

    SCORES_OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(json.dumps(summary["by_dataset"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare_and_ingest", "ingest_only", "unlock_and_score"], required=True)
    args = parser.parse_args()
    if args.phase == "prepare_and_ingest":
        run_prepare_and_ingest(skip_prepare=False)
    elif args.phase == "ingest_only":
        run_prepare_and_ingest(skip_prepare=True)
    else:
        run_unlock_and_score()


if __name__ == "__main__":
    main()
