# Discourse Consistency Engine (`dce`)

`dce` tracks cross-turn semantic consistency and speaker-relative vocal/facial shifts in multi-turn English conversations. It pairs a 2-stage NLI + LLM claims ledger with per-speaker Bayesian-shrinkage baselines and a strict boundary that keeps physiological metrics out of LLM prompts until text verdicts are locked.

> **Scope & Non-Diagnostic Notice:** `dce` is a research tool for discourse coherence and cognitive-load telemetry, **not** a lie detector. Vocal jitter, shimmer, pitch shifts, and blink rates reflect cognitive load, emotional arousal, or microphone acoustics just as strongly as deception, and isolated short clips of unfamiliar speakers cannot reliably separate truthful stress from rehearsed falsehood.

## How It Works

1. **Audio / Video Extraction (`audio_dsp.py`, `vision_face.py`)**: Decodes 16 kHz PCM + 15 FPS RGB via PyAV, transcribes with `faster-whisper` (`small.en`), extracts Praat/eGeMAPS prosody (`f0_semitone_sd`, `jitter_local_pct`, `shimmer_local_pct`, `hnr_db`, `spectral_slope_db`, `alpha_ratio_db`, `loudness_db`), and measures MediaPipe `FaceLandmarker` blink rate, AUs, and gaze (gated to frontal faces `|yaw| <= 20°`).
2. **Audio Quality Gates**: Detects reverberation/clipping and VoIP/Zoom noise-gate spoofing (`acoustic_snr_db` artificially ~60 dB), disabling unreliable perturbation features (`jitter`/`shimmer`/`hnr`) before baseline updates.
3. **Per-Speaker Baselines (`BaselineStore`)**: Decouples location shrinkage (`k_mu = 3.0` for acoustic/visual, `0.0` for timing after warm-up) and scale shrinkage (`k_sigma = 6.0`), bypasses z-score outlier rejection during warm-up (`n_eff < 3`) so room mics adapt, and scores turns via block-diagonal regularized Mahalanobis distance (`math_core.py`).
4. **State-Locked 4-Tool MCP Workflow (`dce.mcp_server`)**:
   - `dce_ingest_media`: Extracts audio/video features into locked backend memory and returns only firewall-safe transcripts with low-probability word markers (`[word? p=0.52 | alt: ...]`).
   - `dce_submit_verdicts`: Validates `turn_id` references, downgrades phonetic-fragile collisions to `NEEDS_AUDIO_REVIEW`, and locks text verdicts.
   - `dce_unlock_signals`: Refuses (`FIREWALL_LOCKED`) until verdicts are submitted; then unlocks `BaselineStore` + Mahalanobis scores and writes JSONL replay logs.
   - `dce_slice_turn_audio`: Exports a 3-second 16 kHz WAV clip for human review.

## Quick Start

```bash
git clone https://github.com/kaopadsaparod/Conversational-Behavior-Analysis.git
cd Conversational-Behavior-Analysis

# Core + test runner only
pip install -e ".[dev]"

# Full audio/video extraction (Whisper, Parselmouth, MediaPipe, OpenCV)
pip install -e ".[dev,audio,vision]"

# Optional LLM / UI extras
pip install -e ".[llm,ui]"
```

Run the test suite:

```bash
python -m pytest -q
```

Start the stdio MCP server:

```bash
python -m dce.mcp_server
```

## Verified Benchmarks

All benchmarks were run with real extractors (`faster-whisper 1.2.1`, `praat-parselmouth 0.4.7`, `mediapipe 1.2.3`, `av 19.0.2`, `cross-encoder/nli-deberta-v3-xsmall`) and blinded inputs (filenames and ground-truth labels stripped before reasoning). Detailed tables and failure analyses are in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

### Unit & Integration Tests
- `python -m pytest -q`: **154 passed in 10.21s**

### DECODE Contradiction Pipeline (`dev.jsonl`, 100 balanced dialogues: 50 Pos / 50 Neg)
Same-speaker raw utterance pairs with clause splitting, Turn-0 context, and `nli_threshold = 0.72`:

| Stage | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy | LLM Calls Saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 1 (`DeBERTa-v3-xsmall >= 0.72`) | 48 | 35 | 15 | 2 | 0.5783 | 0.9600 | 0.7218 | 0.6300 | 1,787 / 3,135 (57.0%) |
| **Full 2-Stage (Stage 1 + Blind Stage 2)** | **47** | **0** | **50** | **3** | **1.0000** | **0.9400** | **0.9691** | **0.9700** | **1,787 / 3,135 (57.0%)** |

### 24 Mixed Wild Videos (`12 RLDD Test/` + `12 Werewolf-YouTube` 20s sub-clips)
Recalibrated on 54 `RLDD Train/` Truthful clips (`priors_wild_video_en.json`, causal 10s rolling blink rate, block-diagonal voice correlations), frontal-gated AUs/gaze (`|yaw| <= 20°`), neutral fillers excluded from `df`, and 5-fold `Train/` CV threshold (`mean(d_sq/df) >= 1.8` across `>= 2` turns with `df >= 4`):

| Split | Mode | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall (`n=24`) | Mode A (Physio) | 1 | 1 | 11 | 11 | 0.5000 | 0.0833 | 0.1429 | 0.5000 |
| Overall (`n=24`) | **Mode B (Semantic)** | **3** | **1** | **11** | **9** | **0.7500** | **0.2500** | **0.3750** | **0.5833** |
| Overall (`n=24`) | Mode C (Fusion) | 3 | 2 | 10 | 9 | 0.6000 | 0.2500 | 0.3529 | 0.5417 |
| `RLDD Test/` (`n=12`) | Mode A (Physio) | 1 | 1 | 5 | 5 | 0.5000 | 0.1667 | 0.2500 | 0.5000 |
| `RLDD Test/` (`n=12`) | **Mode B (Semantic)** | **2** | **0** | **6** | **4** | **1.0000** | **0.3333** | **0.5000** | **0.6667** |
| `RLDD Test/` (`n=12`) | Mode C (Fusion) | 2 | 1 | 5 | 4 | 0.6667 | 0.3333 | 0.4444 | 0.5833 |
| `Werewolf` 20s (`n=12`) | Mode A (Physio) | 0 | 0 | 6 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| `Werewolf` 20s (`n=12`) | Mode B & Mode C | 1 | 1 | 5 | 5 | 0.5000 | 0.1667 | 0.2500 | 0.5000 |

### 6 Complete Multi-Turn `Werewolf-Among-Us` Games (`n=6 games, 26 players`)
Evaluated chronologically from Turn 1 to end of game with per-player `BaselineStore` adaptation (`n_eff >= 3`) and blind Stage-2 role/claim reasoning (`Werewolf/Minion` vs Villager-team):

| Mode | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mode A (In-Game Baseline) | 4 | 9 | 9 | 4 | 0.3077 | 0.5000 | 0.3810 | 0.5000 |
| Mode B (Blind Stage-2 Reasoning) | 4 | 5 | 13 | 4 | 0.4444 | 0.5000 | 0.4706 | 0.6538 |
| **Mode C (Neuro-Symbolic Fusion)** | **6** | 7 | 11 | **2** | **0.4615** | **0.7500** | **0.5714** | **0.6538** |

## Repository Layout

```
├── BENCHMARK_REPORT.md              # Full benchmark report & failure analysis (Thai)
├── config/
│   ├── default.yaml                 # Engine configuration
│   └── mcp_config.example.json      # Example stdio MCP configuration
├── data/
│   ├── priors_en.json               # Literature population priors
│   └── priors_wild_video_en.json    # 54-clip RLDD Train-Truthful priors
├── src/dce/
│   ├── mcp_server.py                # 4-tool state-locked stdio MCP server
│   ├── schemas.py                   # Pydantic v2 models & NaN/Inf sanitization
│   ├── math_core.py                 # Semitone, weighted median/MAD, Mahalanobis, ZNCC
│   ├── baseline_store.py            # Shrinkage baselines, anchor locking, warm-up gate
│   ├── speaker_registry.py          # Speaker centroid tracking
│   ├── vad_endpointing.py           # Dual-level VAD & speculative ASR mailbox
│   ├── recorder.py                  # Deterministic JSONL replay recorder
│   ├── pipeline.py                  # Windows spawn multiprocessing pipeline
│   ├── extractors/                  # Audio (Parselmouth/Whisper) & Vision (MediaPipe)
│   └── reasoning/                   # ClaimsLedger, NLIPreFilter, templates, LLM adapter
├── benchmarks/                      # Reproducible DECODE, audio stress, RLDD & Werewolf scripts
└── tests/                           # 154 unit & integration tests
```

## License

GNU General Public License v3.0 (GPL-3.0-or-later). See [LICENSE](LICENSE) for the full license text (including copyleft obligations for derivative works, source distribution, and patent/liability terms). External datasets (e.g., RAVDESS CC BY-NC-SA 4.0) remain subject to their respective licenses.
