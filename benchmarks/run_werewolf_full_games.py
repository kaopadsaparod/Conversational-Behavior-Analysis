"""Longitudinal Multi-Turn Evaluation on 6 Complete Werewolf-Among-Us Games.

Evaluates each complete game chronologically from Turn 1 to the end of the game:
  - Per-player BaselineStore self-adaptation across early turns (n_eff >= 3 in-game baseline)
    when real audio/video exists, OR longitudinal dialogue turn-latency/hedging + NLI tracking.
  - Full-game ClaimsLedger + DeBERTa-v3-xsmall tracking role claims and cross-turn
    self-contradictions / role-claim conflicts across the entire game.
  - Scores player-level deception (Werewolf/Minion vs Villager-team) against hidden roles.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dce.baseline_store import BaselineStore
from dce.extractors.audio_dsp import decode_media_audio_and_frames, extract_prosody
from dce.math_core import compute_mahalanobis
from dce.reasoning.claims_ledger import NLIPreFilter
from dce.schemas import (
    BaselineConfig,
    ClaimItem,
    ClaimPolarity,
    FEATURE_GROUPS,
    ProsodyFeatures,
    TimingFeatures,
    TurnFeatures,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_JSON = REPO_ROOT / "dataset" / "Werewolf-Among-Us" / "Youtube" / "split" / "test.json"
VIDEOS_DIR = REPO_ROOT / "dataset" / "Werewolf-Among-Us" / "Youtube" / "videos"
OUT_PATH = REPO_ROOT / "benchmarks" / "results" / "werewolf_full_games_scores.json"
BLIND_GAMES_PATH = REPO_ROOT / "benchmarks" / "results" / "werewolf_blind_games.json"
BLIND_VERDICTS_PATH = REPO_ROOT / "benchmarks" / "results" / "werewolf_blind_verdicts.json"

EVIL_ROLES = {"Werewolf", "Minion"}
GOOD_ROLES = {
    "Villager", "Seer", "Robber", "Troublemaker",
    "Insomniac", "Mason", "Drunk", "Hunter", "Tanner",
}

ROLE_CLAIM_RE = re.compile(
    r"\b(?:i\s+am|i'm|im)\s+(?:the\s+|a\s+)?"
    r"(villager|seer|robber|troublemaker|insomniac|mason|drunk|hunter|tanner|werewolf|minion)\b",
    re.IGNORECASE,
)


def _parse_ts(ts: str) -> float:
    parts = [p.strip() for p in ts.split(":")]
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    return 0.0


def _find_video(yt_id: str, video_name: str) -> Optional[Path]:
    if not VIDEOS_DIR.is_dir():
        return None
    for p in VIDEOS_DIR.iterdir():
        if p.suffix.lower() in {".mp4", ".mkv", ".webm"} and (yt_id in p.name or video_name[:20].lower() in p.name.lower()):
            return p
    return None


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
        "n": len(gts), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "accuracy": round(acc, 4),
    }


def evaluate_full_games(n_games: int = 6) -> Dict[str, Any]:
    games = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    selected_games = [
        g for g in games
        if any(r in EVIL_ROLES for r in g.get("startRoles", []))
        and len(g.get("Dialogue", [])) >= 40
    ][:n_games]

    nli = NLIPreFilter(model_name="cross-encoder/nli-deberta-v3-xsmall", use_hf_model=True)
    nli._ensure_hf_model()

    feature_order = list(FEATURE_GROUPS.keys())
    corr_matrix = np.eye(len(feature_order), dtype=np.float64)

    player_records: List[Dict[str, Any]] = []
    game_summaries: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for g_idx, game in enumerate(selected_games, start=1):
        players = game.get("playerNames", [])
        roles = game.get("startRoles", [])
        role_map = {p: r for p, r in zip(players, roles)}
        dialogue = game.get("Dialogue", [])
        game_start_sec = _parse_ts(game.get("startTime", "00:00"))
        video_path = _find_video(game.get("YT_ID", ""), game.get("video_name", ""))

        audio = None
        sr = 16000
        if video_path is not None:
            try:
                audio, sr, _ = decode_media_audio_and_frames(video_path, target_sr=16000, video_fps=1.0, max_frames=1)
            except Exception:
                audio = None

        baseline = BaselineStore(
            config=BaselineConfig(
                decoupled_shrinkage=True,
                scale_shrinkage_k=6.0,
                l2_location_shrinkage_k=0.0,
                acoustic_location_shrinkage_k=3.0,
            )
        )

        history_by_player: Dict[str, List[str]] = {p: [] for p in players}
        role_claims_by_player: Dict[str, List[str]] = {p: [] for p in players}
        all_role_claimants: Dict[str, List[str]] = {}  # role -> [players]
        phys_anomalies_by_player: Dict[str, int] = {p: 0 for p in players}
        valid_turns_by_player: Dict[str, int] = {p: 0 for p in players}
        nli_self_conflicts: Dict[str, int] = {p: 0 for p in players}
        prev_end_sec: Optional[float] = None

        for t_idx, utt in enumerate(dialogue):
            spk = utt.get("speaker", "")
            txt = (utt.get("utterance") or "").strip()
            if spk not in role_map or not txt:
                continue

            ts_sec = _parse_ts(utt.get("timestamp", "00:00"))
            next_ts = _parse_ts(dialogue[t_idx + 1]["timestamp"]) if t_idx + 1 < len(dialogue) else ts_sec + 3.0
            dur_sec = max(1.0, min(8.0, next_ts - ts_sec))

            # Track role claims longitudinally across the game
            for m_role in ROLE_CLAIM_RE.findall(txt):
                r_norm = m_role.lower()
                if r_norm not in role_claims_by_player[spk]:
                    role_claims_by_player[spk].append(r_norm)
                all_role_claimants.setdefault(r_norm, [])
                if spk not in all_role_claimants[r_norm]:
                    all_role_claimants[r_norm].append(spk)

            # Check same-speaker cross-turn NLI self-contradiction across the whole game
            for prev_txt in history_by_player[spk][-8:]:
                if len(prev_txt.split()) >= 4 and len(txt.split()) >= 4:
                    c_prob, _ = nli.predict_scores(prev_txt, txt)
                    if c_prob >= 0.85:
                        nli_self_conflicts[spk] += 1
            history_by_player[spk].append(txt)

            # Longitudinal BaselineStore per-player acoustic adaptation (only if video audio present)
            pros = ProsodyFeatures(quality_flag="no_audio_backend")
            if audio is not None and len(audio) > sr:
                s_idx = max(0, min(len(audio), int((game_start_sec + ts_sec) * sr)))
                e_idx = max(s_idx, min(len(audio), int((game_start_sec + ts_sec + dur_sec) * sr)))
                if (e_idx - s_idx) >= int(1.0 * sr):
                    pros = extract_prosody(audio[s_idx:e_idx], sample_rate=sr)

            gap = max(0.0, ts_sec - prev_end_sec) if prev_end_sec is not None else None
            prev_end_sec = ts_sec + dur_sec

            tf = TurnFeatures(
                turn_id=t_idx + 1,
                speaker_id=spk,
                start_ns=int(ts_sec * 1e9),
                end_ns=int((ts_sec + dur_sec) * 1e9),
                timing=TimingFeatures(
                    response_latency_open=gap if gap is not None and gap <= 5.0 else None,
                    turn_duration_ms=dur_sec * 1000.0,
                ),
                prosody=pros,
            )
            scores = baseline.score_and_update_turn(spk, tf, word_count=len(txt.split()))
            # Count anomalies strictly AFTER player builds in-game baseline (n_eff >= 3)
            valid_turns_by_player[spk] += 1
            if valid_turns_by_player[spk] >= 4:
                maha = compute_mahalanobis(
                    feature_scores=scores,
                    correlation_matrix=corr_matrix,
                    feature_order=feature_order,
                    priors=baseline.priors,
                )
                if maha.degrees_of_freedom >= 4 and (maha.d_sq / maha.degrees_of_freedom) >= 2.5:
                    phys_anomalies_by_player[spk] += 1

        # Evaluate per-player predictions at end of full game
        for p in players:
            role = role_map[p]
            if role not in EVIL_ROLES and role not in GOOD_ROLES:
                continue
            is_evil = role in EVIL_ROLES
            # Cross-player unique role conflict (e.g., two people claiming Seer/Robber/Troublemaker, or player shifting own role)
            multi_role_shift = len(role_claims_by_player[p]) >= 2
            unique_role_clash = any(
                r in ("seer", "robber", "troublemaker", "insomniac", "drunk", "hunter", "tanner")
                and len(all_role_claimants.get(r, [])) >= 2
                for r in role_claims_by_player[p]
            )
            mode_a = phys_anomalies_by_player[p] >= 2
            mode_b = multi_role_shift or unique_role_clash or nli_self_conflicts[p] >= 1
            mode_c = mode_b or (mode_a and (len(role_claims_by_player[p]) > 0 or valid_turns_by_player[p] >= 8))

            player_records.append({
                "game_id": game.get("Game_ID"),
                "player": p,
                "role": role,
                "is_deceptive": is_evil,
                "turns": valid_turns_by_player[p],
                "role_claims": role_claims_by_player[p],
                "nli_self_conflicts": nli_self_conflicts[p],
                "unique_role_clash": unique_role_clash,
                "phys_anomalies": phys_anomalies_by_player[p],
                "mode_a": mode_a,
                "mode_b": mode_b,
                "mode_c": mode_c,
            })

        game_summaries.append({
            "game_num": g_idx,
            "game_id": game.get("Game_ID"),
            "video_found": video_path is not None,
            "turns": len(dialogue),
            "players": len(players),
        })

    gts = [r["is_deceptive"] for r in player_records]
    out = {
        "n_games": len(selected_games),
        "n_players": len(player_records),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "mode_a": _compute_metrics([r["mode_a"] for r in player_records], gts),
        "mode_b": _compute_metrics([r["mode_b"] for r in player_records], gts),
        "mode_c": _compute_metrics([r["mode_c"] for r in player_records], gts),
        "games": game_summaries,
        "players": player_records,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_games", "n_players", "elapsed_sec", "mode_a", "mode_b", "mode_c")}, indent=2))
    return out


def export_blind_games(n_games: int = 6) -> Path:
    """Export 6 complete games with ALL ground-truth labels (startRoles/endRoles/votingResults) stripped."""
    games = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    selected_games = [
        g for g in games
        if any(r in EVIL_ROLES for r in g.get("startRoles", []))
        and len(g.get("Dialogue", [])) >= 40
    ][:n_games]
    blind_payload = []
    for g_idx, g in enumerate(selected_games, start=1):
        blind_payload.append({
            "game_num": g_idx,
            "players": g.get("playerNames", []),
            "dialogue": [
                {
                    "turn": i + 1,
                    "timestamp": u.get("timestamp", ""),
                    "speaker": u.get("speaker", ""),
                    "utterance": (u.get("utterance") or "").strip(),
                }
                for i, u in enumerate(g.get("Dialogue", []))
                if (u.get("utterance") or "").strip()
            ],
        })
    BLIND_GAMES_PATH.write_text(json.dumps(blind_payload, indent=2), encoding="utf-8")
    return BLIND_GAMES_PATH


def score_blind_verdicts() -> Dict[str, Any]:
    """Score blind Stage-2 per-player verdicts combined with Mode A from werewolf_full_games_scores.json."""
    base = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    verdicts_raw = json.loads(BLIND_VERDICTS_PATH.read_text(encoding="utf-8"))
    verdict_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for g_entry in verdicts_raw:
        g_num = int(g_entry["game_num"])
        for p_entry in g_entry.get("players", []):
            verdict_map[(g_num, p_entry["player"])] = p_entry

    # Map each player_record back to its game_num sequentially
    game_player_counts = [g["players"] for g in base["games"]]
    g_num_seq: List[int] = []
    for idx, cnt in enumerate(game_player_counts, start=1):
        g_num_seq.extend([idx] * cnt)

    updated_players: List[Dict[str, Any]] = []
    for rec, g_num in zip(base["players"], g_num_seq):
        v = verdict_map.get((g_num, rec["player"]), {})
        mode_b = bool(v.get("mode_b_deceptive", False))
        mode_a = bool(rec["mode_a"])
        # Mode C fusion: Mode B high-precision semantic signal always triggers;
        # Mode A only triggers if sustained (phys_anomalies >= 4 AND >= 15 turns) to avoid drowning Mode B.
        sustained_a = mode_a and (rec.get("phys_anomalies", 0) >= 4 and rec.get("turns", 0) >= 15)
        mode_c = mode_b or sustained_a
        updated = dict(rec)
        updated["game_num"] = g_num
        updated["mode_b"] = mode_b
        updated["mode_b_rationale"] = v.get("rationale", "")
        updated["mode_c"] = mode_c
        updated_players.append(updated)

    gts = [r["is_deceptive"] for r in updated_players]
    base["mode_a"] = _compute_metrics([r["mode_a"] for r in updated_players], gts)
    base["mode_b"] = _compute_metrics([r["mode_b"] for r in updated_players], gts)
    base["mode_c"] = _compute_metrics([r["mode_c"] for r in updated_players], gts)
    base["players"] = updated_players
    OUT_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(json.dumps({k: base[k] for k in ("n_games", "n_players", "mode_a", "mode_b", "mode_c")}, indent=2))
    return base


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["full", "export_blind", "score_verdicts"], default="full")
    args = parser.parse_args()
    if args.phase == "export_blind":
        p = export_blind_games(n_games=6)
        print(f"Exported blind games to {p}")
    elif args.phase == "score_verdicts":
        score_blind_verdicts()
    else:
        evaluate_full_games(n_games=6)
