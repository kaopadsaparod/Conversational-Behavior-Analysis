# Discourse Consistency Engine (`dce`) — Operational Plan & Checklist

## Executive Summary & Engineering Architecture
Discourse Consistency Engine (`dce`) is an open-source, neuro-symbolic conversational cognitive load analysis and consistency verification engine. The system integrates real-time multimodal streaming digital signal processing (DSP), dynamic Bayesian-inspired baseline profiling with shrinkage estimation and regularized Block-Diagonal Mahalanobis distance, and a two-stage Large Language Model (LLM) symbolic reasoning claims ledger protected by an impenetrable Neuro-Symbolic Firewall.

This document serves as the master 6-phase operational roadmap, detailing completed foundation milestones, active core module implementations, and upcoming experimental validation, licensing, and benchmarking phases.

---

## Phase 1: Foundation & Core Engine Architecture (Milestones 1–3)

### 1.1 Foundation & Configuration Contracts
- [x] **Lightweight Package Specification (`pyproject.toml`)**:
  - [x] Core dependencies restricted to lightweight numerical libraries (`numpy>=1.26`, `scipy>=1.11`, `pydantic>=2.5`, `pyyaml>=6.0`).
  - [x] Dev dependencies configured with `pytest>=8.0`.
  - [x] Optional heavy dependencies segregated into extras: `[audio]` (`faster-whisper`, `praat-parselmouth`, `onnxruntime`), `[vision]` (`mediapipe`, `opencv-python-headless`), `[llm]` (`httpx`, `ollama`, `google-genai`), and `[ui]` (`streamlit`, `plotly`).
- [x] **System Configuration (`config/default.yaml`)**:
  - [x] Hardware profile tailored for Intel Core i5-14400F + NVIDIA GeForce RTX 5060 (8GB VRAM) + 24GB RAM.
  - [x] Subsystem sections declared: `mode`, `clock`, `queues`, `hardware`, `vad`, `prosody`, `baseline`, `mahalanobis`, `av_sync`, `speaker_registry`, `reasoning`.
- [x] **Master Clock & Thread-Safe Policy Queues (`src/dce/clock.py`)**:
  - [x] `MasterClock` supporting offline mode (`pts_ns` stepping) and live mode (`monotonic_ns + av_offset_ns`).
  - [x] `PolicyQueue` implementing `block`, `drop_oldest`, and `never_drop` overflow policies with thread condition variables and drop counter metrics.
- [x] **Pydantic v2 Data Contracts & Schemas (`src/dce/schemas.py`)**:
  - [x] Universal `DCEBaseModel` with `allow_inf_nan=False` and recursive float sanitization (`NaN`/`Inf` -> `None`).
  - [x] Configuration models: `ClockConfig`, `QueuesConfig`, `HardwareConfig`, `VadConfig`, `ProsodyConfig`, `BaselineConfig`, `MahalanobisConfig`, `AvSyncConfig`, `SpeakerRegistryConfig`, `ReasoningConfig`, `AppConfig`.
  - [x] VAD event models: `PauseRecord`, `SpeculativeTriggerEvent`, `CancelSpeculativeEvent`, `HardTurnCommitEvent`, `VADChunkResult`, `TurnTranscript`.
  - [x] Feature extraction models: `ProsodyFeatures`, `VisionFeatures`, `TimingFeatures`, `TurnFeatures`.
  - [x] Statistical baseline models: `PriorParameter`, `PriorsData`, `FeatureScore`, `MahalanobisResult`, `DirectionLabel` (4-state).
  - [x] Speaker registry models: `SpeakerAssignment`.
  - [x] Symbolic reasoning models: `ClaimPolarity`, `ClaimItem`, `ClaimExtractionResponse`, `VerdictStatus`, `ConsistencyVerificationResponse`.
  - [x] Replay & recorder models: `RecorderTurnRecord`.
- [x] **Stage 1 Verification Suite (`tests/test_stage1_foundation.py`)**:
  - [x] 100% test pass on default YAML loading, MasterClock offline/live modes, PolicyQueue policies, and Pydantic schema contracts.

### 1.2 Documentation, Plan & Prior Distributions
- [x] **Operational Plan & Phase Checklist (`PLAN_CHECKLIST.md`)**:
  - [x] Comprehensive 6-phase engineering checklist with explicit completion markers.
  - [x] License compliance gating, re-clustering, and LOSO cross-validation specifications.
- [x] **High-Density Technical Engineering Documentation (`README.md`)**:
  - [x] Mathematical equations: Semitone pitch conversion, exact tie-breaking weighted median/MAD, shrinkage estimation, Block-Diagonal Mahalanobis distance with eigen-safe regularized fallback, 4-state direction classification, 50 Hz zero-normalized cross-correlation.
  - [x] Multimodal architectural dataflow diagram.
  - [x] Scientific explanation of the ~20% physiological signal variance ceiling and cognitive load ambiguity.
  - [x] Prominent ethical and scientific disclaimer: *"This system is NOT a lie detector"*.
- [x] **Literature Population Priors Database (`data/priors_en.json`)**:
  - [x] Validated against `PriorsData` schema with `"calibrated": false` and `"source": "literature_initial_estimates"`.
  - [x] 8 multimodal features: `response_latency_open`, `response_latency_closed`, `articulation_rate`, `f0_semitone_sd`, `jitter_local_pct`, `shimmer_local_pct`, `hnr_db`, `blink_rate_per_min`.
  - [x] Block-diagonal 8x8 correlation matrix with zero cross-group correlations and index (0, 1) set to 0.0.

### 1.3 Core Mathematical Foundations (`src/dce/math_core.py`)
- [x] **Acoustic Frequency Transformations**:
  - [x] Logarithmic semitone pitch normalization: $S(t) = 12 \log_2(F_0(t) / \tilde{F}_0)$ with unvoiced frame handling (`np.nan` -> `None`).
- [x] **Robust Non-Parametric Statistics**:
  - [x] Deterministic `weighted_median` with exact 0.5-tie linear interpolation: $0.5 \cdot (x_k + x_{k+1})$ when cumulative weight $C_k = 0.5 W$.
  - [x] Standardized `weighted_mad` utilizing identical weighted median kernel: $\text{MAD}_w = \text{weighted\_median}(|x_k - \tilde{x}_w|, w_k)$.
- [x] **Block-Diagonal Regularized Mahalanobis Distance**:
  - [x] Cross-group zeroing across `timing` (0..2), `voice` (3..6), and `vision` (7).
  - [x] Matrix symmetrization $(\mathbf{M} + \mathbf{M}^\top)/2$ and shrinkage regularization $\hat{\boldsymbol{\Sigma}} = (1 - \lambda)\boldsymbol{\Sigma} + \lambda\mathbf{I}$ ($\lambda = 0.30$).
  - [x] Condition number validation ($\le 10^{12}$) and eigen-decomposition fallback (`np.linalg.eigh`) clamping eigenvalues to $\epsilon_{\text{floor}} = 10^{-6}$.
  - [x] Non-negative squared distance enforcement: $d_{\text{sq}} = \max(\mathbf{z}_V^\top \hat{\boldsymbol{\Sigma}}_V^{-1} \mathbf{z}_V, 0.0)$, $D_M = \sqrt{d_{\text{sq}}}$.
  - [x] Chi-square upper survival function: $\text{chi2\_tail\_score} = \chi^2_{\text{sf}}(d_{\text{sq}}, \text{df}=|V|)$.
- [x] **Four-State Direction Classifier**:
  - [x] Stress-direction polarity alignment: $z_{\text{stress}, i} = \text{polarity}_i \cdot z_i$.
  - [x] Vocal tension indicator: $(z_{\text{jitter}} > 1.5) \lor (z_{\text{shimmer}} > 1.5) \lor (z_{\text{hnr}} < -1.5)$.
  - [x] Exhaustive state labeling: `normal`, `cognitive_load_anomaly`, `calm_deviation`, `mixed_subthreshold_deviation`.
- [x] **Audio-Visual Cross-Correlation Synchronization (`normalized_av_xcorr`)**:
  - [x] Strict temporal sorting and finite-coordinate validation.
  - [x] Resampling onto uniform 50.0 Hz temporal grid.
  - [x] Discrete lag search $\tau \in [-200, +200]\text{ ms}$ with minimum per-lag overlap constraint ($\ge 0.5\text{ s}$ / 25 samples).
  - [x] Per-lag local mean subtraction and L2-norm scaling on overlapping segments.
  - [x] Safe fallback returning `(None, None)` on degenerate variance ($\text{std} < 10^{-6}$) or insufficient overlap.

### 1.4 Dynamic Baseline Profiling & Shrinkage Store (`src/dce/baseline_store.py`)
- [x] **Strict Cold-Start Protection**:
  - [x] Effective sample weight guard: immediate fallback to population prior if $n_{\text{eff}} \le 0.0$ without invoking median routines.
- [x] **Score-Before-Update Sequential Processing**:
  - [x] Prior shrinkage formula: $\hat{\mu} = \frac{n_{\text{eff}}\tilde{x}_w + k\mu_{\text{pop}}}{n_{\text{eff}} + k}$, $\hat{\sigma} = \max\left(\frac{n_{\text{eff}}(1.4826 \cdot \text{MAD}_w) + k\sigma_{\text{pop}}}{n_{\text{eff}} + k}, \epsilon_f\right)$.
  - [x] Confidence weight metric: $c = \frac{n_{\text{eff}}}{n_{\text{eff}} + k}$.
- [x] **Outlier Gating & Per-Feature Anchor Locking**:
  - [x] Three-tier weight gating: $|z| \le 3.0 \to w=1.0$; $3.0 < |z| \le 5.0 \to w=0.2$; $|z| > 5.0$ or corrupted quality $\to w=0.0$.
  - [x] Independent per-feature anchor lock after `anchor_lock_turns = 8` verified updates.
  - [x] Drift restriction: bounding sample median within $[\text{anchor} - \Delta_{\max}, \text{anchor} + \Delta_{\max}]$.
  - [x] `reset_anchor(speaker_id, feature_name=None)` interface.
- [x] **Idempotent Speaker Merging (`merge_speaker_id`)**:
  - [x] No-op on identical or missing IDs.
  - [x] History concatenation with rolling window retention (`maxlen = 40`).
  - [x] Counter aggregation and automatic anchor locking if combined ok count $\ge 8$.

### 1.5 Real-Time VAD & Dual-Level Endpointing (`src/dce/vad_endpointing.py`)
- [x] **Sample-Accurate PCM Buffering**:
  - [x] Raw audio buffer indexed with `last_voiced_chunk_idx` to guarantee zero silence clipping latency.
- [x] **Hysteresis State Machine & Micro-Pause Tracking**:
  - [x] Speaking activation at `vad_prob >= speech_prob_on` (0.55).
  - [x] Ubiquitous micro-pause recording ($\ge 200\text{ ms}$) in both `SPEAKING` and `SOFT_PAUSE` states.
  - [x] Soft pause transition at 400 ms silence; speculative ASR trigger event for turns $\ge 600\text{ ms}$.
  - [x] Speech resumption requiring `vad_prob >= 0.55` with speculative cancellation and hesitation tracking.
  - [x] Hard turn commitment at 800 ms silence or 25,000 ms ceiling with speech duration validation ($\ge 160\text{ ms}$).
  - [x] End-Of-File (EOF) `flush()` method committing trailing conversational turns.
- [x] **Single-Slot Speculative Mailbox (`LatestWinsSpeculativeMailbox`)**:
  - [x] Non-blocking atomic overwrite (`put_latest`), epoch matching discard (`discard_if_matches`), and retrieval (`pop_latest`).

### 1.6 Speaker Identity Registry (`src/dce/speaker_registry.py`)
- [x] **Unit Centroid Vector Space**:
  - [x] Zero-norm, NaN, and Inf guard rejecting corrupt embeddings (`np.linalg.norm < 1e-8`).
  - [x] Duration-weighted online centroid accumulation: $\mathbf{s}_{\text{new}} = D_{\text{old}}\mathbf{c}_{\text{old}} + d_{\text{turn}}\hat{\mathbf{e}}$, $\mathbf{c}_{\text{new}} = \mathbf{s}_{\text{new}} / \|\mathbf{s}_{\text{new}}\|_2$.
  - [x] Strict 1-to-1 `face_hint` to `speaker_id` binding.
- [x] **Provisional Speaker Promotion & Bounded Capacity**:
  - [x] Overlap guard: freeze profile updates during acoustic overlap (`overlap_flag == True`).
  - [x] Long utterance ($\ge 2.0\text{ s}$) logic: instant adoption of matching provisional buffers (`sim >= 0.45`) or new speaker assignment up to `max_speakers = 4`.
  - [x] Short utterance ($< 2.0\text{ s}$) provisional buffer accumulation (`max_provisional_buffers = 6`, TTL = 60.0s).
  - [x] Zombie buffer elimination: no new provisional buffers created when max speakers limit reached; automatic fallback to nearest speaker with `low_confidence = True`.

### 1.7 Feature Recorder & Bit-Exact Replay (`src/dce/recorder.py`)
- [x] **Standardized JSONL Recording**:
  - [x] `FeatureRecorder` enforcing `json.dumps(..., allow_nan=False)` for bit-exact serialization.
  - [x] Synchronous and thread-safe turn recording (`RecorderTurnRecord`).
- [x] **Deterministic Replay Engine**:
  - [x] `ReplayLoader` providing reproducible offline stream replay without numerical divergence.

### 1.8 Neuro-Symbolic Reasoning & Firewall (`src/dce/reasoning/`)
- [x] **Neuro-Symbolic Firewall Isolation**:
  - [x] Absolute prohibition of raw physiological signal names and numeric metrics in LLM prompts.
  - [x] Rule-based template generation in `reasoning/templates.py` translating statistical anomalies into descriptive behavioral context.
  - [x] Complete suppression of stress warnings under `calm_deviation` state.
- [x] **Context-Aware Conversational Question Gate**:
  - [x] Automatic bypass of LLM for brief affirmative/acknowledgment tokens ($< 4$ words) unless directly preceded by a question.
  - [x] Mandatory pairing of preceding interrogatives with terse replies (e.g. *"Did you enter the server room?"* $\to$ *"No."*).
- [x] **Two-Step Structured Reasoning Engine**:
  - [x] Step A (`ClaimExtractionResponse`): Proposition extraction with entity alias normalization in `ClaimsLedger`.
  - [x] Step B (`ConsistencyVerificationResponse`): Multi-speaker claim verification, cross-turn contradiction detection, and `REJECTED_INVALID_REF` guard against hallucinatory claim citations.
- [x] **Gemini Schema Sanitizer (`sanitize_gemini_schema`)**:
  - [x] Step 1: Deep recursive dereferencing and inlining of all `{"$ref": "#/$defs/..."}` schemas.
  - [x] Step 2: Flattening of `anyOf` null unions into primary types with `"nullable": True`.
  - [x] Step 3: Pruning of unsupported Gemini keywords (`$defs`, `default`, `additionalProperties`, `title`) while strictly preserving user field names matching property keys.
- [x] **Ollama & Gemini Structured Provider Adapters**:
  - [x] Pydantic schema serialization into JSON format specs with 1x automated retry mechanism.

### 1.9 Multimodal Connectors & Spawning Pipeline
- [ ] **Lazy-Import Extractor Modules** *(live backend execution unverified)*:
  - [ ] `extractors/audio_dsp.py`: Dynamic import of `praat-parselmouth` and `faster-whisper`; vocal creak trimming (200 ms tail); `boundary_trim_ms` (15 ms) Praat execution and `transcribe_audio` live execution pending live backend verification.
  - [ ] `extractors/vision_face.py`: Dynamic import of `mediapipe` and `cv2`; lip aperture and head-pose `±30°` Euler angle gate pending live backend verification.
- [ ] **Multiprocessing Spawning Safety (`src/dce/pipeline.py`)**:
  - [ ] Real audio/vision worker loops connected to live extractor backends (pending live backend verification).
  - [x] Windows `spawn` compatibility: top-level module worker functions accepting `multiprocessing.Queue` instances, zero global queue allocations, and clean process termination.

### 1.10 Phase 1 Analytical QA Verification Suite (`tests/`)
- [x] `tests/test_anti_stub_and_spawn.py`: AST scan confirming zero `pass`/`...`/`TODO` in 10 core files; Windows `spawn` multiprocessing IPC execution.
- [x] `tests/test_math_core.py`: Hand-calculated numerical assertions for semitones, exact-0.5 tie-breaking, eigen-fallback, 4-state direction classifier, and 50 Hz ZNCC.
- [x] `tests/test_baseline_store.py`: Mock interception ensuring zero median calculation under $n_{\text{eff}} = 0$; per-feature anchor locking; idempotent merging.
- [x] `tests/test_speaker_registry.py`: Zero-norm input rejection, duration-weighted unit centroids, provisional promotions, and bounded capacity enforcement.
- [x] `tests/test_vad_endpointing.py`: 200 ms micro-pause capture, soft pause speculative trigger, 0.55 hysteresis, 25s forced split, EOF `flush()`, and mailbox atomic overwrite.
- [x] `tests/test_claims_ledger.py`: Neuro-Symbolic Firewall integrity test (zero metric leakage); short answer question gating; 3-step schema sanitization; invalid reference rejection.

---

## Phase 2: Dataset Acquisition, NC/SA Licensing Compliance & Acoustic Prior Calibration

- [ ] **Dataset Licensing & Legal Compliance Review**:
  - [ ] Perform legal audit on candidate acoustic datasets: RAVDESS (CC BY-NC-SA 4.0), CREMA-D (Open Database), and IEMOCAP (Academic Research License).
  - [ ] Verify non-commercial (NC) and share-alike (SA) downstream distribution constraints.
  - [ ] Establish strict data boundary ensuring proprietary enterprise deployments isolate NC/SA training weights from open-source runtime artifacts.
- [ ] **Acoustic Data Ingestion & Quality Filtering**:
  - [ ] Implement automated audio sanity pipeline: signal-to-noise ratio (SNR) filtering ($\ge 20\text{ dB}$).
  - [ ] Waveform clipping rejection ($>-0.1\text{ dBFS}$) and reverberation boundary identification.
  - [ ] Dynamic silence stripping using `DualLevelEndpointTracker`.
- [ ] **Demographic Stratification & Subgroup Clustering**:
  - [ ] Stratify speech samples across gender cohorts (masculine, feminine, non-binary fundamental frequency distributions).
  - [ ] Calculate empirical variance distributions across age brackets (18–30, 31–50, 51+).
- [ ] **Optional Offline Re-Clustering**:
  - [ ] Implement optional offline k-means / Gaussian Mixture Model (GMM) re-clustering over voice embeddings to detect latent acoustic clusters.
  - [ ] Evaluate cluster separation via silhouette score and Calinski-Harabasz index.
  - [ ] Generate cluster-conditional prior parameter sets for specialized acoustic environments.
- [ ] **Prior Calibration Execution (`scripts/build_priors_ravdess.py`)**:
  - [ ] Extract 8 multimodal metrics across speech corpora: latency distributions, articulation rates, F0 variance, jitter, shimmer, HNR, and blink intervals.
  - [ ] Derive empirical empirical medians ($\mu_{\text{pop}}$) and median absolute deviations ($\sigma_{\text{pop}}$).
  - [ ] Compute empirical 8x8 covariance and correlation matrices; enforce positive semi-definiteness via eigenvalue thresholding.
  - [ ] Export calibrated database to `data/priors_en_calibrated.json` with `"calibrated": true`.

---

## Phase 3: Real-Time Multimodal Streaming & Hardware Acceleration Integration

- [ ] **Silero VAD ONNX Runtime Acceleration**:
  - [ ] Integrate quantized Silero VAD v5 ONNX model running on CPU (`int8`) or DirectML/CUDA.
  - [ ] Implement continuous circular ring buffer streaming 512-sample frames at 16,000 Hz.
- [ ] **ASR Inference Optimization on RTX 5060**:
  - [ ] Benchmark `faster-whisper` (`small.en`) latency and VRAM usage on NVIDIA GeForce RTX 5060 (8GB VRAM).
  - [ ] Enforce CPU execution default with zero VRAM competition; enable opt-in CUDA fp16 execution with bounded memory allocations.
  - [ ] Implement beam search speculative decoding utilizing `LatestWinsSpeculativeMailbox`.
- [ ] **Visual Feature Extraction via MediaPipe Face Mesh**:
  - [ ] Stream video frames through MediaPipe Face Mesh running on dedicated thread.
  - [ ] Extract canonical lip aperture: Euclidean distance between upper and lower inner lip landmarks normalized by outer inter-commissure distance.
  - [ ] Compute Eye Aspect Ratio (EAR) across left and right eye landmarks for blink event detection.
  - [ ] Apply head-pose Euler angle thresholding to gate facial metric validity during extreme yaw/pitch deviations ($> 30^\circ$).
- [ ] **Real-Time 50 Hz Audio-Visual Synchronization**:
  - [ ] Feed extracted lip aperture and acoustic RMS envelope into `normalized_av_xcorr`.
  - [ ] Track real-time temporal lag $\tau$ and correlation coefficients on sliding 5-second observation windows.
  - [ ] Alert on desynchronization anomalies indicative of synthetic audio injection or deepfake artifacts.
- [ ] **Inter-Process Communication (IPC) Throughput Benchmarking**:
  - [ ] Benchmark multiprocessing queue throughput vs shared memory (`multiprocessing.shared_memory`) on Windows 11.
  - [ ] Verify zero memory leaks over 24-hour continuous streaming stress tests.

---

## Phase 4: Advanced Neuro-Symbolic Knowledge Graphs & Multi-Party Claims Ledger

- [ ] **Temporal Knowledge Graph Extension**:
  - [ ] Expand `ClaimsLedger` from flat proposition list into an incremental temporal graph.
  - [ ] Represent claims as directed edges: `(SubjectEntity) -[Predicate {time, location, polarity}]-> (ObjectEntity)`.
  - [ ] Support discourse horizons exceeding 500 conversational turns with index-based sub-graph retrieval.
- [ ] **Multi-Party Conversational Dynamics**:
  - [ ] Track turn-taking latency across alternating speaker pairs (`SPEAKER_00` $\leftrightarrow$ `SPEAKER_01`).
  - [ ] Monitor floor interruptions, simultaneous speech ratios, and backchannel token frequencies.
  - [ ] Isolate acoustic baseline adaptations conditioned on conversational partner identity (interactive alignment / phonetic convergence).
- [ ] **Cross-Speaker Transitive Contradiction Reasoning**:
  - [ ] Detect third-party inconsistencies (e.g. Speaker A affirms proposition $P$ while Speaker B negates $P$).
  - [ ] Model epistemic modality (claims stated as certainty vs hearsay vs conjecture).
- [ ] **Local LLM Fine-Tuning & Quantization Benchmarks**:
  - [ ] Evaluate quantized local models (Qwen 2.5 7B Q4_K_M, Llama 3.1 8B Instruct) on Claim Extraction (Step A) and Consistency Verification (Step B).
  - [ ] Measure token generation throughput and structured JSON formatting compliance rate ($>99.5\%$).
- [ ] **Adversarial Red-Teaming of the Neuro-Symbolic Firewall**:
  - [ ] Probe LLM prompt templates against prompt injection attacks attempting to smuggle acoustic metrics into LLM reasoning contexts.
  - [ ] Enforce strict schema validators stripping extraneous keys and text escaping.

---

## Phase 5: Benchmarking, Ablation Studies & Leave-One-Subject-Out (LOSO) Validation

- [ ] **Leave-One-Subject-Out (LOSO) Cross-Validation**:
  - [ ] Structure experimental partitions by subject identifier: train baselines and priors on $N-1$ subjects, evaluate exclusively on the withheld subject.
  - [ ] Guarantee zero speaker identity or acoustic profile leakage between train and test splits.
- [ ] **Three-Way Ablation Matrix**:
  - [ ] **System 1 (Acoustic Only)**: Cognitive load anomaly detection using Mahalanobis distance exclusively.
  - [ ] **System 2 (Symbolic Only)**: Consistency verification using LLM Claims Ledger exclusively.
  - [ ] **System 3 (Hybrid Neuro-Symbolic)**: Full `dce` engine integrating acoustic cognitive load cues with symbolic contradiction tracking.
- [ ] **Empirical Metrics & Statistical Significance**:
  - [ ] Evaluate Area Under the ROC Curve (AUC-ROC), Precision-Recall AUC (PR-AUC), and Macro-F1 across annotated deception corpora (e.g., Columbia-SRI-Colorado Deception Corpus).
  - [ ] Compute statistical significance using McNemar's test on paired classification outcomes and Wilcoxon signed-rank tests across LOSO folds ($p < 0.01$).
- [ ] **False Positive Rate Analysis under Cognitive Stress**:
  - [ ] Subject truth-telling participants to elevated cognitive stress paradigms (Stroop task, mental arithmetic, complex temporal sequencing).
  - [ ] Document physiological false-positive behavior; demonstrate how symbolic claims ledger prevents incorrect deception verdicts on truthful subjects under stress.
- [ ] **Automated Ablation Benchmark Script (`benchmarks/run_ablation.py`)**:
  - [ ] Scriptable headless benchmark runner reading recorded JSONL corpora.
  - [ ] Output LaTeX and Markdown comparative evaluation tables.

---

## Phase 6: Interactive Dashboard, Audit Logging & Production Hardening

- [ ] **Streamlit Multimodal Analytics Dashboard (`app/dashboard.py`)**:
  - [ ] Real-time time-series visualization of pitch semitones, jitter, shimmer, HNR, and blink rates.
  - [ ] Dynamic 2D/3D scatter projection of Mahalanobis distance and baseline drift.
  - [ ] Interactive Claims Ledger table highlighting detected contradictions, timeline shifts, and unverified propositions.
  - [ ] Individual channel inspection tabs: Audio Prosody, Visual Kinematics, Symbolic Claims, and System Health.
- [ ] **Cryptographic Audit Log Chaining**:
  - [ ] Implement SHA-256 hash chaining across consecutive records in `FeatureRecorder`.
  - [ ] Verify tamper-evident integrity for forensic legal review and post-interview auditability.
- [ ] **Ethical Disclosure Banner & Non-Diagnostic Advisory**:
  - [ ] Unmissable UI banner: *"This system is an automated discourse consistency research instrument, NOT a lie detector."*
  - [ ] Mandatory confirmation prompt acknowledging physiological ambiguity and prohibiting unassisted adverse employment, judicial, or law enforcement decisions.
- [ ] **Production Packaging & Containerization**:
  - [ ] Multi-stage Dockerfile containing Python 3.13, CUDA 12.8 runtime, and headless OpenCV/MediaPipe bindings.
  - [ ] Automated health checks and graceful degradation fallback when GPU or webcam is unavailable.
- [x] **Real-Media Adaptation, Wild Calibration & MCP Server (`dce-analyzer`)**:
  - [x] `src/dce/mcp_server.py` (4 state-locked stdio MCP tools, `<0.2s` lazy startup, `stdout`/`fd 1` -> `stderr` isolation, `config/mcp_config.example.json`).
  - [x] Acoustic & Vision Real-Media Hardening: 65 Hz zero-phase Butterworth high-pass, physiological reverb ceiling + `fine_wave_quality_flag` isolation, VoIP/noise-gate detector, Voiced Intensity Envelope `articulation_rate`, `FaceLandmarker` (`RunningMode.IMAGE` / `RunningMode.VIDEO` @ 15 FPS), and PyAV (`av 19.0.2`) MP4 decoder.
  - [x] Decoupled `scale_shrinkage_k=6.0` + warm-up starvation guard + calibrated `data/priors_wild_video_en.json` (from `RLDD Train/` Truthful clips).
  - [x] `cross-encoder/nli-deberta-v3-xsmall` NLI pre-filter calibrated on DECODE (`same_speaker_only`, raw utterance text, `thr=0.90` -> `F1=0.715`, `Recall=0.880`, `1,055 LLM calls saved` on 100 balanced `dev.jsonl` samples).
