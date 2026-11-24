# Discourse Consistency Engine (`dce`)

> **CRITICAL SCIENTIFIC & ETHICAL DISCLAIMER**  
> **THIS SYSTEM IS NOT A LIE DETECTOR.**  
>
> Automated analysis of acoustic prosody, speech latency, facial kinematics, and autonomic physiological markers **CANNOT** determine truthfulness, deception, or honesty. Peer-reviewed scientific consensus (National Research Council, 2003; American Psychological Association; Vrij et al., 2019) conclusively demonstrates that somatic and vocal stress signals account for at most **15% to 20%** of the variance in human communication behavior.
>
> Elevated vocal tension, speech latency, fundamental frequency fluctuations, or altered blink rates reflect **cognitive workload, affective arousal, situational anxiety, neurodiversity, or conversational hesitation**—phenomena frequently exhibited by completely truthful individuals under social pressure or investigative scrutiny.
>
> Under the **European Union Artificial Intelligence Act (EU AI Act, Articles 5 & 14)** and foundational jurisprudence (e.g., US *Employee Polygraph Protection Act*, 29 U.S.C. § 2001), using biometric or emotion recognition systems to infer truthfulness, culpability, or deceit in law enforcement, employment, educational, or judicial contexts is strictly prohibited or categorized as unacceptable/high risk.
>
> The Discourse Consistency Engine (`dce`) is designed strictly as a **neuro-symbolic research instrument** for studying discourse coherence, conversational turn-taking dynamics, and communicative cognitive load. It must **NEVER** be deployed as an autonomous or semi-autonomous truth verification tool or used to make adverse decisions against any human being.

---

## 1. System Overview & Architecture

The **Discourse Consistency Engine (`dce`)** is a high-performance neuro-symbolic framework for analyzing conversational dynamics, measuring speaker cognitive workload, and verifying discourse consistency across multi-turn English conversations.

The engine unifies:
1. **Streaming Multimodal DSP**: Real-time extraction of fundamental frequency ($F_0$), micro-prosody (Jitter, Shimmer, HNR), speech latency, and facial kinematics (lip aperture, blink intervals) with millisecond-level precision.
2. **Dynamic Bayesian-Inspired Baseline Profiling**: A non-parametric rolling baseline store employing sample-weight shrinkage towards literature population priors, per-feature anchor locking, and outlier gating.
3. **Block-Diagonal Mahalanobis Distance**: Robust multidimensional anomaly detection over regularized covariance matrices with spectral condition monitoring and eigen-safe inversion.
4. **Neuro-Symbolic Firewall**: A rigid architectural isolation boundary ensuring that no raw physiological metrics or numerical arousal scores ever contaminate LLM reasoning prompts.
5. **Two-Stage Symbolic Claims Ledger**: LLM-driven structured proposition extraction (Step A) and cross-turn contradiction verification (Step B) with citation-integrity validation and context-aware interrogative gating.

```
+----------------------------------------------------------------------------------------------------+
|                                    DISCOURSE CONSISTENCY ENGINE                                    |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 Audio Stream (16 kHz PCM)        │        Video Stream (Frames)
                             │                    │                    │
                             ▼                    │                    ▼
                  [ MasterClock (PTS/Live) ]      │         [ PolicyQueue (drop_oldest) ]
                             │                    │                    │
                             ▼                    │                    ▼
              [ DualLevelEndpointTracker ]        │         [ MediaPipe Face Mesh ]
              - Speech Trigger (prob >= 0.55)     │         - Lip Aperture (Normalized)
              - Micro-Pauses (200-399 ms)         │         - Eye Aspect Ratio (Blinks)
              - Soft Pause (400 ms) -> Spec ASR   │                    │
              - Hard Commit (800 ms / 25 s)       │                    │
                             │                    │                    │
               ┌─────────────┴─────────────┐      │                    │
               ▼                           ▼      │                    ▼
       [ Audio DSP ]                [ Whisper ASR ]  ───► [ 50 Hz A/V Cross-Correlation ]
       - Parselmouth F0 (Semitone)   - Fast greedy/spec    - Per-Lag Overlap ZNCC ([-200, +200] ms)
       - Creak Trim (200 ms)         - Transcript & Q-Gate
       - Jitter, Shimmer, HNR
               │                           │
               └─────────────┬─────────────┘
                             ▼
                 [ SpeakerRegistry (Unit Centroid) ]
                 - Duration-Weighted Accumulation
                 - Short/Long Provisional Buffer Promotion
                 - Max Speaker Bounding (<= 4 Speakers)
                             │
                             ▼
                 [ BaselineStore & Shrinkage ]
                 - Strict Cold-Start Guard (n_eff > 0)
                 - Shrinkage Bayesian Prior Fusion
                 - 3-Tier Outlier Gating (|z| <= 3.0 / 5.0)
                 - Per-Feature Anchor Locking (8 turns)
                             │
                             ▼
         [ Block-Diagonal Mahalanobis Anomaly Engine ]
         - Zeroed Cross-Group Blocks (Timing / Voice / Vision)
         - Regularization: Sigma_hat = 0.70 * Sigma + 0.30 * I
         - Eigen-Safe Spectral Fallback (eig_floor = 1e-6)
         - 4-State Direction Classifier:
           * normal
           * cognitive_load_anomaly
           * calm_deviation (suppresses stress alert)
           * mixed_subthreshold_deviation
                             │
                             ▼
         ===========================================================
         ====            NEURO-SYMBOLIC FIREWALL                ====
         ====  (NO physical metrics / z-scores / signal names   ====
         ====   permitted across the reasoning boundary)       ====
         ===========================================================
                             │
                   Clean Transcript Only
                             │
                             ▼
             [ Two-Stage LLM Claims Ledger ]
             - Context-Aware Question Gate (pair short replies)
             - Step A: Proposition Extraction (ClaimsLedger)
             - Step B: Consistency Verification
             - Invalid Reference Rejection (VerdictStatus.REJECTED_INVALID_REF)
             - Structured Output Schema Sanitizer (Gemini / Ollama)
                             │
                             ▼
                 [ FeatureRecorder & Replay ]
                 - Bit-Exact Deterministic JSONL Replay
                 - allow_nan=False Strict Validation
```

---

## 2. Mathematical Formulations

### 2.1 Semitone Frequency Conversion
Human pitch perception is logarithmic. Raw fundamental frequency values $F_0(t)$ (in Hz) are mapped onto a relative semitone scale $S(t)$ normalized by the speaker's individualized median pitch baseline $\tilde{F}_0$:

$$S(t) = 12 \log_2\left(\frac{F_0(t)}{\tilde{F}_0}\right) = \frac{12}{\ln 2} \ln\left(\frac{F_0(t)}{\tilde{F}_0}\right)$$

- **Domain constraints**: $F_0(t) > 0$ and $\tilde{F}_0 > 0$.
- **Unvoiced frames**: For unvoiced segments ($F_0(t) \le 0$), $S(t) = \text{NaN}$ in NumPy arrays and is serialized as `None` in exported Pydantic structures.
- **Dispersion metric**: The acoustic pitch variability feature `f0_semitone_sd` represents the sample standard deviation of $S(t)$ evaluated over all voiced frames of the turn.

### 2.2 Exact Tie-Breaking Weighted Median and Weighted MAD
To ensure deterministic, bit-exact reproducibility matching manual reference solutions across platforms, `weighted_median` and `weighted_mad` are defined with exact half-weight tie handling.

Given pairs of observations and positive weights $(x_k, w_k)$ for $k = 1, \dots, n$, sort the pairs in ascending order of $x_k$ such that $x_1 \le x_2 \le \dots \le x_n$. Let the cumulative weight sequence be:

$$C_k = \sum_{j=1}^k w_j, \quad \text{with total weight } W = C_n$$

The weighted median $\tilde{x}_w$ is evaluated as:
1. If there exists an index $k < n$ such that $C_k = 0.5 \cdot W$ (evaluated within floating-point tolerance `math.isclose(C_k, 0.5 * W, rel_tol=1e-12, abs_tol=1e-12)`):

   $$\tilde{x}_w = \frac{x_k + x_{k+1}}{2}$$

2. Otherwise, $\tilde{x}_w = x_k$ for the smallest index $k$ satisfying $C_k > 0.5 \cdot W$.

The **Weighted Median Absolute Deviation ($\text{MAD}_w$)** is computed by passing absolute centered deviations through the identical median kernel:

$$\text{MAD}_w = \text{weighted\_median}\left(\left|x_k - \tilde{x}_w\right|, w_k\right)$$

### 2.3 Bayesian Baseline Shrinkage & Anchor Locking

Individual speakers exhibit unique baseline characteristics. To prevent small-sample volatility while tracking genuine behavioral shifts, `BaselineStore` combines empirical observations with population priors using Bayesian-inspired shrinkage.

#### Log-Transformed Features
Features with skewed log-normal distributions (e.g., `response_latency_open`, `response_latency_closed`) are transformed before baseline estimation:

$$x' = \ln\left(\max(x, 0) + \delta\right), \quad \text{where } \delta = 0.05\text{ s}$$

#### Cold-Start Guard
Before invoking median routines, the effective sample weight is verified:

$$n_{\text{eff}} = \sum_{i=1}^M w_i$$

If $M = 0$ or $n_{\text{eff}} \le 0.0$, the baseline immediately returns the literature prior without calculation:

$$\hat{\mu} = \mu_{\text{pop}}, \quad \hat{\sigma} = \max(\sigma_{\text{pop}}, \epsilon_f), \quad c = 0.0$$

#### Shrinkage Estimation
Given rolling window samples with effective weight $n_{\text{eff}}$, sample weighted median $\tilde{x}_w$, sample $\text{MAD}_w$, shrinkage hyperparameter $k = 6.0$, population median $\mu_{\text{pop}}$, and population scale $\sigma_{\text{pop}}$:

$$\hat{\mu} = \frac{n_{\text{eff}}\tilde{x}_w + k\mu_{\text{pop}}}{n_{\text{eff}} + k}$$

$$\hat{\sigma} = \max\left(\frac{n_{\text{eff}}(1.4826 \cdot \text{MAD}_w) + k\sigma_{\text{pop}}}{n_{\text{eff}} + k}, \, \epsilon_f\right)$$

$$z = \frac{x' - \hat{\mu}}{\hat{\sigma}}, \qquad c = \frac{n_{\text{eff}}}{n_{\text{eff}} + k}$$

where $\epsilon_f$ is the per-feature standard deviation floor preventing division by zero.

#### Outlier Gating & Per-Feature Anchor Lock
To prevent corrupt utterances or acute stress spikes from pulling the baseline, updates follow a **Score-Before-Update** policy with gated weighting:
- If $|z| \le 3.0$ and quality is `"ok"`: observation is appended with weight $w = 1.0$.
- If $3.0 < |z| \le 5.0$ and quality is `"ok"`: observation is appended with downweighted weight $w = 0.2$.
- If $|z| > 5.0$ or quality is not `"ok"`: observation is rejected ($w = 0.0$).

When a feature accumulates `anchor_lock_turns = 8` successful updates for a speaker, its **anchor median** is permanently locked. Subsequent sample medians are clamped to prevent drift:

$$\tilde{x}_w \in \left[\text{anchor\_median} - \Delta_{\max}, \, \text{anchor\_median} + \Delta_{\max}\right]$$

### 2.4 Block-Diagonal Regularized Mahalanobis Distance

Features are partitioned into three orthogonal behavioral modalities:
1. **Timing Group ($V_{\text{timing}}$)**: `response_latency_open`, `response_latency_closed`, `articulation_rate`
2. **Voice Group ($V_{\text{voice}}$)**: `f0_semitone_sd`, `jitter_local_pct`, `shimmer_local_pct`, `hnr_db`
3. **Vision Group ($V_{\text{vision}}$)**: `blink_rate_per_min`

#### Cross-Group Zeroing and Regularization
Cross-group correlation entries between orthogonal modalities are explicitly zeroed out to prevent spurious inter-modal cross-talk. The empirical correlation matrix $\boldsymbol{\Sigma}_{\text{corr}}$ is symmetrized and regularized with parameter $\lambda = 0.30$:

$$\mathbf{M}_{\text{sym}} = \frac{\boldsymbol{\Sigma}_{\text{corr}} + \boldsymbol{\Sigma}_{\text{corr}}^\top}{2}, \qquad \hat{\boldsymbol{\Sigma}} = (1 - \lambda)\mathbf{M}_{\text{sym}} + \lambda\mathbf{I}$$

#### Submatrix Inversion & Eigen-Safe Spectral Fallback
For the set of valid features $V$ present in the current turn:
1. Extract submatrix $\hat{\boldsymbol{\Sigma}}_V = \hat{\boldsymbol{\Sigma}}[V, V]$ and normalized score vector $\mathbf{z}_V$.
2. Check matrix condition number: $\kappa = \text{cond}(\hat{\boldsymbol{\Sigma}}_V)$.
3. If $\kappa \le 10^{12}$ and $\hat{\boldsymbol{\Sigma}}_V$ is non-singular, solve $\hat{\boldsymbol{\Sigma}}_V \mathbf{y} = \mathbf{z}_V$ directly via Cholesky/LU factorization (`np.linalg.solve`).
4. If singular or near-singular ($\kappa > 10^{12}$ or `LinAlgError`), execute spectral fallback:
   - Perform symmetric eigen-decomposition: $\hat{\boldsymbol{\Sigma}}_V = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$.
   - Clamp eigenvalues to safety floor $\epsilon_{\text{eig}} = 10^{-6}$: $\tilde{\lambda}_i = \max(\lambda_i, \epsilon_{\text{eig}})$.
   - Reconstruct regularized inverse: $\hat{\boldsymbol{\Sigma}}_V^{-1} = \mathbf{Q} \operatorname{diag}\left(\tilde{\lambda}_1^{-1}, \dots, \tilde{\lambda}_{|V|}^{-1}\right) \mathbf{Q}^\top$.

#### Anomaly Scoring
The squared Mahalanobis distance and upper tail chi-square score are computed as:

$$d_{\text{sq}} = \max\left(\mathbf{z}_V^\top \hat{\boldsymbol{\Sigma}}_V^{-1} \mathbf{z}_V, \, 0.0\right), \qquad D_M = \sqrt{d_{\text{sq}}}$$

$$\text{chi2\_tail\_score} = \chi^2_{\text{sf}}\left(d_{\text{sq}}, \, \text{df} = |V|\right) = \int_{d_{\text{sq}}}^\infty \frac{t^{\frac{|V|}{2} - 1} e^{-\frac{t}{2}}}{2^{\frac{|V|}{2}} \Gamma\left(\frac{|V|}{2}\right)} dt$$

*(Note: Named `chi2_tail_score` rather than `p_value` to maintain strict non-inferential clarity).*

### 2.5 Four-State Direction Classification Logic

To distinguish elevated cognitive load from relaxation or calm states, features are partitioned into **directional** ($\text{polarity} \in \{+1, -1\}$) and **neutral** ($\text{polarity} = 0$) sets:

$$z_{\text{stress}, i} = \text{polarity}_i \cdot z_i, \qquad z_{\text{neutral}, j} = |z_j|$$

- **Vocal Tension Indicator**:
  
  $$\text{vocal\_tension\_flag} = (z_{\text{jitter}} > 1.5) \lor (z_{\text{shimmer}} > 1.5) \lor (z_{\text{hnr}} < -1.5)$$

- **Direction Label Classification Rules**:
  1. **`normal`**:  
     $$\text{chi2\_tail\_score} \ge 0.05$$
  2. **`cognitive_load_anomaly`**:  
     $$\text{chi2\_tail\_score} < 0.05 \quad \land \quad \left(\max_{i \in \text{dir}}(z_{\text{stress}, i}) \ge 1.5 \;\lor\; \max_{j \in \text{neu}}(|z_j|) \ge 1.5\right)$$
  3. **`calm_deviation`**:  
     $$\text{chi2\_tail\_score} < 0.05 \quad \land \quad |\text{dir}| > 0 \quad \land \quad \max_{i \in \text{dir}}(z_{\text{stress}, i}) < 0.0 \quad \land \quad \min_{i \in \text{dir}}(z_{\text{stress}, i}) \le -1.5 \quad \land \quad \forall j \in \text{neu}: |z_j| < 1.5$$  
     *(Under `calm_deviation`, all physiological stress warnings in template reasoning are completely suppressed).*
  4. **`mixed_subthreshold_deviation`**:  
     All remaining turns where $\text{chi2\_tail\_score} < 0.05$.

### 2.6 Audio-Visual Cross-Correlation Synchronization

Audio-visual temporal coherence validates whether speech acoustics correspond directly to visible articulatory kinematics (lip aperture), detecting synthetic voice substitution or transmission desynchronization.

Given visual timestamps $t_{\text{vis}}$ with lip aperture $A(t)$ and audio timestamps $t_{\text{aud}}$ with RMS energy envelope $E(t)$:
1. Verify monotonic ascending timestamps and finite values.
2. Define overlapping temporal span: $[t_{\text{start}}, t_{\text{end}}] = [\max(t_{\text{vis}}[0], t_{\text{aud}}[0]), \, \min(t_{\text{vis}}[-1], t_{\text{aud}}[-1])]$.
3. Construct a uniform temporal grid with frequency $f_s = 50.0\text{ Hz}$ ($\Delta t = 20\text{ ms}$). Resample both signals onto the grid via linear interpolation.
4. For discrete lags $\tau \in [-200\text{ ms}, +200\text{ ms}]$ (step size $\Delta \tau = 20\text{ ms}$):
   - Isolate the mutually overlapping segment $[a(\tau), b(\tau)]$.
   - Require segment duration $b(\tau) - a(\tau) \ge \text{min\_overlap\_sec} = 0.5\text{ s}$ (25 grid points).
   - Compute Zero-Normalized Cross-Correlation (**ZNCC**) **strictly evaluated over the overlapping segment for that specific lag**:
     
     $$\rho(\tau) = \frac{\sum_{n} (A[n] - \bar{A}_\tau)(E[n + \tau] - \bar{E}_\tau)}{\sqrt{\sum_n (A[n] - \bar{A}_\tau)^2 \cdot \sum_n (E[n + \tau] - \bar{E}_\tau)^2}}$$
     
   - If signal variance $\sigma_\tau < 10^{-6}$ or valid overlap is unsatisfied for all lags, return `(None, None)` without throwing exceptions or emitting `NaN`.

---

## 3. The 20% Variance Ceiling & Cognitive Load Ambiguity

A fundamental principle of conversational neuroscience is the **physiological signal ceiling**:

```
+-------------------------------------------------------------------------+
|                  TOTAL VARIANCE IN CONVERSATIONAL BEHAVIOR              |
+-------------------------------------------------------------------------+
| [ Autonomic / Vocal Stress ] | [ Context, Demographics, Personality,   ] |
| [     MAXIMUM 15% - 20%    ] | [ Syntax, Epistemics, Language, Memory  ] |
|                              | [               80% - 85%              ] |
+-------------------------------------------------------------------------+
```

### 3.1 Why Physiological Cues Cannot Prove Deception
1. **Identical Autonomic Arousal Pathways**: Sympathetic nervous system activation (elevated heart rate, vocal cord micro-tremors, dry mouth, irregular breathing) is activated by:
   - Fear of being disbelieved (truth-teller anxiety, *Othello Error*).
   - High cognitive difficulty (mental arithmetic, reconstructing non-linear chronologies, technical jargon recall).
   - Social evaluation apprehension, introversion, or neurodiversity (e.g., Autism spectrum traits).
   - Deceptive fabrication.
2. **Individual Baselines & Diurnal Variation**: Acoustic parameters like Jitter, Shimmer, and $F_0$ drift significantly based on physical fatigue, ambient room temperature, hydration, time of day, and emotional valence.
3. **Absence of a "Pinocchio Response"**: Decades of empirical psychophysiological research have established that no unique physiological indicator or combination of indicators correlates exclusively with falsehood.

### 3.2 The Role of the Neuro-Symbolic Firewall
Because acoustic and kinematic signals are inherently ambiguous, `dce` strictly separates the **physiological layer** from the **symbolic reasoning layer**:
- **Firewall Guarantee**: LLM prompts never receive raw metrics (e.g. `jitter=2.4%`, `chi2_tail_score=0.001`, `z_score=3.8`).
- **Preventing Algorithmic Bias**: If an LLM were informed that a speaker exhibited "high vocal tension," it would be prone to confirmation bias, prematurely hallucinating contradictions in truthful utterances.
- **Independent Verification**: The LLM analyzes propositional semantics (`ClaimsLedger`) purely on textual evidence. The statistical engine provides independent telemetry on cognitive effort without making truth claims.

---

## 4. Hardware Optimization & Deployment Profile

The engine is engineered for low-latency edge deployment on personal workstation hardware:

| Component | Target Specification | DCE Optimization Strategy |
|---|---|---|
| **CPU** | Intel Core i5-14400F (10C / 16T) | Multiprocess pipeline (`spawn`), 16 kHz VAD chunking, Parselmouth DSP |
| **GPU** | NVIDIA GeForce RTX 5060 (8GB VRAM) | Reserved for local LLM inference (Ollama Qwen 2.5 7B Q4_K_M); ASR defaults to CPU `int8` to prevent VRAM oversubscription |
| **RAM** | 24 GB DDR4/DDR5 | Rolling 40-turn window, bounded queues (`max_size: 128`), memory-safe JSONL streaming |
| **OS** | Windows 11 Pro 64-bit | Clean Windows `multiprocessing` `spawn` compliance, non-blocking queue policies |

---

## 5. Repository Structure

```
discourse-consistency-engine/
├── PLAN_CHECKLIST.md               # 6-phase operational plan and progress checklist
├── README.md                       # Comprehensive engineering documentation & formulas
├── pyproject.toml                  # Lightweight core dependencies + optional extras
├── config/
│   └── default.yaml                # Production engine configuration
├── data/
│   └── priors_en.json              # 8-feature literature population priors (calibrated: false)
├── src/
│   └── dce/
│       ├── __init__.py             # Package root
│       ├── schemas.py              # Pydantic v2 data contracts & NaN/Inf sanitization
│       ├── clock.py                # MasterClock & PolicyQueue (block, drop_oldest, never_drop)
│       ├── math_core.py            # Semitone, weighted median/MAD, Mahalanobis, 4-state, 50Hz ZNCC
│       ├── baseline_store.py       # Shrinkage estimation, anchor locking, idempotent merging
│       ├── speaker_registry.py     # Unit centroid clustering, provisional promotion, capacity limits
│       ├── vad_endpointing.py      # Dual-level VAD, micro-pauses, soft pause, speculative mailbox
│       ├── recorder.py             # FeatureRecorder (allow_nan=False) & Bit-Exact Replay Loader
│       ├── pipeline.py             # Windows spawn multiprocessing workers & IPC
│       ├── extractors/
│       │   ├── __init__.py
│       │   ├── audio_dsp.py        # Lazy-imported Parselmouth & Whisper acoustic extraction
│       │   └── vision_face.py      # Lazy-imported MediaPipe facial kinematics
│       └── reasoning/
│           ├── __init__.py
│           ├── templates.py        # Rule-based behavioral descriptions (calm_deviation suppression)
│           ├── llm_adapter.py      # Ollama / Gemini structured output & 3-step schema sanitizer
│           └── claims_ledger.py    # Neuro-Symbolic Firewall, question gate, contradiction detection
├── scripts/
│   └── build_priors_ravdess.py     # Offline acoustic prior calibration script
├── benchmarks/
│   └── run_ablation.py             # Headless ablation benchmark runner (LOSO validation)
├── app/
│   └── dashboard.py                # Streamlit multi-channel analytics dashboard
└── tests/
    ├── test_stage1_foundation.py   # Stage 1 verification suite (passed)
    ├── test_anti_stub_and_spawn.py # AST zero-stub check & Windows spawn IPC test
    ├── test_math_core.py           # Hand-calculated assertions for mathematical core
    ├── test_baseline_store.py      # Cold-start guard, anchor locking, and profile merging tests
    ├── test_speaker_registry.py    # Zero-norm guard, provisional promotion, and TTL tests
    ├── test_vad_endpointing.py     # Dual-level endpointing, 200ms pauses, and mailbox tests
    └── test_claims_ledger.py       # Firewall integrity, question gating, and schema tests
```

---

## 6. Quick Start & Testing

### Installation
```bash
# Clone the repository
git clone https://github.com/example/discourse-consistency-engine.git
cd discourse-consistency-engine

# Install lightweight core and development dependencies
pip install -e ".[dev]"
```

### Running Test Suites
```bash
# Execute Stage 1 Foundation Tests
pytest -v tests/test_stage1_foundation.py

# Execute Full Verification Suite
pytest -v
```

---

## 7. License & Ethical Compliance

- **Software License**: Apache License 2.0. See [LICENSE](LICENSE) for details.
- **Dataset Compliance**: Any empirical calibration scripts utilizing external acoustic datasets (e.g. RAVDESS CC BY-NC-SA 4.0) must strictly comply with Non-Commercial and Share-Alike provisions.
- **Non-Diagnostic Policy**: This software must not be used for polygraph examination, lie detection, employment screening, credibility assessment in interrogations, or any autonomous evaluation of human integrity.
