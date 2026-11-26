# Audit Round 1 Fix Log (`fix/audit-round1`)

Baseline tag: `phase1-audited` on `main` (`594aa7d`).

---

## Step 0 — W0: Schemas (`src/dce/schemas.py`)

### Changes
- `HardTurnCommitEvent`: added required `valid_epoch_id: int`.
- `VerdictStatus`: added `LLM_ERROR = "LLM_ERROR"`.
- `VerificationResult(DCEBaseModel)`: added wrapper with `response: ConsistencyVerificationResponse` and `conflict_scope: Optional[Literal["self", "cross_speaker"]] = None`.
- `FeatureScore.z_score`: verified `Optional[float] = None`.
- `FEATURE_GROUPS`: added `Dict[str, Literal["timing", "voice", "vision"]]` mapping all 8 features by name.
- `MahalanobisConfig`: added `stress_z_threshold: float = 1.5`, `z_clip: float = 50.0`, updated `calm_z_threshold: float = 1.5`.
- `ReasoningConfig`: added `retrieval_top_k: int = 12`.

### Verification
- `python -m pytest tests/test_stage1_foundation.py -q` after Step 0 (before `tests/test_stage1_foundation.py` was updated by W5):
  `8 passed, 1 failed` (`test_schemas_enums_and_contracts` failed at `HardTurnCommitEvent` missing `valid_epoch_id`, and `test_default_config_loading` expected `calm_z_threshold == -1.5`).
- Updated `tests/test_stage1_foundation.py` in W5 to assert `calm_z_threshold == 1.5` and pass `valid_epoch_id=1` to `HardTurnCommitEvent`.

---

## W1: Math Core (`src/dce/math_core.py`, `tests/test_math_core.py`)

### Changes
- Imported `FEATURE_GROUPS` from `dce.schemas` (removed local `FEATURE_GROUP_MAP`).
- `compute_mahalanobis`: raises `ValueError` on `correlation_matrix` shape mismatch or non-finite values; added `stress_z_threshold=1.5`, `calm_z_threshold=1.5`, `z_clip=50.0`; clips `z_v` to `[-|z_clip|, |z_clip|]` before `solve`/`d_sq`; Cholesky PD check before `solve` with eigen-clamp fallback + `eigen_fallback_used=True` + `logger.warning`.
- `classify_direction`: uses `stress_z_threshold` and `-abs(calm_z_threshold)` with zero hardcoded `1.5`/`-1.5` literals.
- `weighted_median`: verified `rel_tol=1e-12, abs_tol=0.0`.

### Red-Before-Fix (`python -m pytest tests/test_math_core.py`)
```text
3 failed, 19 passed in 1.13s
- test_compute_mahalanobis_invalid_correlation_matrix_raises: Failed: DID NOT RAISE ValueError
- test_compute_mahalanobis_z_clip_prevents_overflow: TypeError: compute_mahalanobis() got an unexpected keyword argument 'z_clip'
- test_classify_direction_uses_positive_calm_and_stress_thresholds: AssertionError: assert <DirectionLabel.CALM_DEVIATION> == <DirectionLabel.MIXED_SUBTHRESHOLD_DEVIATION>
```

### Green-After-Fix (`python -m pytest tests/test_math_core.py`)
```text
22 passed in 1.09s
```

---

## W2: Priors & Baseline Store (`src/dce/baseline_store.py`, `data/priors_en.json`, `config/default.yaml`, `tests/test_baseline_store.py`, `tests/test_priors_spec.py`)

### Changes
- Verified `score_feature` returns `FeatureScore` with `z_score=None, stress_z=None` when `quality_flag != "ok"` and `score_and_update_turn` excludes it from Mahalanobis `evaluated_features`.
- Updated `data/priors_en.json` and `get_default_priors()` (`max_anchor_drift` for all 8 features + `stress_polarity=1` for `f0_semitone_sd` and `blink_rate_per_min`), and updated `config/default.yaml` (`mahalanobis.stress_z_threshold: 1.5`, `mahalanobis.calm_z_threshold: 1.5`, `mahalanobis.z_clip: 50.0`, `reasoning.retrieval_top_k: 12`).
- Updated `BaselineStore.__init__` (`allow_builtin_priors: bool = False`) and `_resolve_priors` (`explicit priors` -> `DCE_PRIORS_PATH` -> `Path(__file__).resolve().parents[2] / "data" / "priors_en.json"`, raising `FileNotFoundError` when missing unless `allow_builtin_priors=True`).

### Red-Before-Fix (`python -m pytest tests/test_baseline_store.py tests/test_priors_spec.py -q`)
```text
5 failed, 7 passed in 1.14s
- test_priors_resolution_independent_of_cwd (AssertionError: 0.85 != 0.6)
- test_priors_resolution_env_and_missing_file (AssertionError: 4.4 != 5.25)
- test_priors_json_matches_spec (AssertionError: response_latency_open max_anchor_drift 0.85 != 0.6)
- test_get_default_priors_matches_spec (AssertionError: response_latency_open max_anchor_drift 0.85 != 0.6)
- test_default_yaml_has_round1_config_keys (AssertionError: calm_z_threshold -1.5 != 1.5)
```

### Green-After-Fix (`python -m pytest tests/test_baseline_store.py tests/test_priors_spec.py -q`)
```text
12 passed in 1.01s
```

---

## W3: Claims Ledger & Firewall (`src/dce/reasoning/claims_ledger.py`, `tests/test_claims_ledger.py`)

### Changes
- Separated `prompt_framing` vs transcript-derived content; `assert_firewall_compliance` runs ONLY on `prompt_framing` and `system_instruction`. Tokenizes on non-alphanumeric (incl. `_`) and camelCase, matching stems + plurals + phrases + all 8 `FEATURE_GROUPS` names.
- Hardened reference validation in `verify_consistency` (`REJECTED_INVALID_REF` on wrong `source_turn_id`, missing `prior_turn_id`, `prior_turn_id >= current_claim.turn_id`, unknown `prior_claim_id`, missing turn, or `prior_turn_id != candidate.turn_id`).
- `verify_consistency` returns `VerificationResult` with dynamic `conflict_scope` (`"self"` vs `"cross_speaker"`).
- Retrieval caps candidates at `retrieval_top_k` (default 12, newest first, equal across speakers), matches `time_ref_normalized`, normalizes whitespace in `register_alias` and `add_claim`.
- Step B exception returns `VerdictStatus.LLM_ERROR`; Step A & B exceptions increment `ledger.stats["llm_errors"]`.
- `should_skip_extraction` uses strict acknowledgment lexicon (single + multi-word `"i see"`, `"got it"`, `"thank you"`) AND `not prior_transcript.is_question`.

### Red-Before-Fix (`python -m pytest tests/test_claims_ledger.py -q`)
```text
9 failed, 7 passed in 1.40s
Failed: test_firewall_tokenizes_snake_camel_plurals_and_allows_spoken_text,
test_question_gate_skips_short_acknowledgments_without_prior_question,
test_verify_consistency_invalid_turn_reference_converted_to_rejected,
test_verify_consistency_valid_turn_reference_accepted,
test_verify_consistency_all_invalid_ref_cases,
test_verify_consistency_backend_exception_fallback,
test_extract_claims_exception_increments_llm_errors_stat,
test_conflict_scope_dynamic_before_and_after_speaker_merge,
test_retrieval_top_k_newest_first_and_time_ref_normalized.
```

### Green-After-Fix (`python -m pytest tests/test_claims_ledger.py -q`)
```text
16 passed in 0.95s
```

---

## W4: VAD & Endpointing (`src/dce/vad_endpointing.py`, `tests/test_vad_endpointing.py`)

### Changes
- Added `is_speculative_result_valid(result_turn_id: int, result_epoch_id: int, commit: HardTurnCommitEvent) -> bool`.
- Extracted `_build_commit_event(forced_split=...)` so all `HardTurnCommitEvent` creations set `valid_epoch_id=self._epoch_id`.
- Fixed `max_turn_ms` forced split to preserve the triggering speech chunk (`vad_prob >= speech_prob_on`) as the first chunk of the next turn via `_start_turn(...)`.
- Switched internal PCM buffering to `np.ndarray` while returning `List[float]` for schema compatibility.
- **Design Note**: `SOFT_PAUSE` resume hysteresis threshold (`speech_prob_on=0.55` vs `SPEAKING` `speech_prob_off=0.35`) was intentionally left as-is.

### Red-Before-Fix (`python -m pytest tests/test_vad_endpointing.py -q`)
```text
3 failed, 2 passed in 1.32s (baseline before test additions: ValidationError valid_epoch_id Field required on test_dual_level_endpointing_lifecycle, test_forced_split_at_max_turn_duration, test_flush_commits_trailing_turn_at_eof)
1 error in 1.24s (after adding new tests, before fix: ImportError: cannot import name 'is_speculative_result_valid' from 'dce.vad_endpointing')
```

### Green-After-Fix (`python -m pytest tests/test_vad_endpointing.py -q`)
```text
11 passed in 1.14s
```

---

## W5: Pipeline, Extractors, Docs & P2 Checks

### Changes
- `src/dce/pipeline.py`:
  - Added top-level `apply_promotion(assignment, ledger, baseline)` and wired it in `_handle_turn_commit`.
  - Reconciled pending speculative triggers against `HardTurnCommitEvent.valid_epoch_id` using `is_speculative_result_valid`.
  - Loaded correlation matrix from `DCE_PRIORS_PATH` or `data/priors_en.json` mapped by feature name into `FEATURE_GROUPS` order.
  - Replaced fake constants in `vision_worker_process` (`0.25`, `face_detected=True`) and `reasoning_worker_process` (`"CONSISTENT"`) with `extract_vision_frame` and `no_llm_backend` / `ClaimsLedger` execution.
  - Preserved all `VerificationResult`s per turn (`vr.response` in `record.verifications`, `VerificationResult` list in `self._last_verification_results`).
  - Fixed response latency: computed `(current_start_ns - prior_end_ns) / 1e9` only when prior turn was a question; classified open (wh-word) vs closed (leading aux/modal); `None` when prior turn was not a question.
- `src/dce/extractors/audio_dsp.py`: removed fake fallback `f0_semitone_sd=1.0` / `glottal_periods_count=30` when `parselmouth` is missing; returns `None` fields + `quality_flag="no_audio_backend"`.
- `src/dce/extractors/vision_face.py`: removed fake `blink_rate_per_min=20.0/40.0` and `face_id_hint="FACE_0"`; returns `None`.
- `benchmarks/run_ablation.py`: updated `evaluate_loso` to handle both `VerificationResult` and `ConsistencyVerificationResponse`.
- `tests/test_stage1_foundation.py`, `tests/test_workers_and_scripts.py`, `tests/test_audit_fixes.py`, `tests/test_pipeline_integration.py`: updated for Step 0 schema contracts, W1/W3/W5 fixes, Thai UTF-8 recorder test, `apply_promotion` test, `is_speculative_result_valid` test, and metamorphic latency/vision/`"blink"` tests.
- `PLAN_CHECKLIST.md`: verified `ReplayLoader` naming and marked unverified live backend executions (`head-pose ±30°`, `boundary_trim_ms` 15ms Praat execution, real audio/vision worker loops, `transcribe_audio` live execution) as `[ ]`.
- `README.md`: updated `git clone` URL to `https://github.com/kaopadsaparod/Conversational-Behavior-Analysis.git`, removed `"15% to 20%"` variance claim and `"Variance Ceiling"` ASCII diagram, generalized legal citation beyond `"EU AI Act, Articles 5 & 14"`, and removed `"detecting synthetic voice substitution"` claim.

### Red-Before-Fix (`python -m pytest tests/test_audit_fixes.py tests/test_pipeline_integration.py -q`)
```text
5 failed, 16 passed in 29.42s
- FAILED tests/test_audit_fixes.py::test_claims_ledger_alias_time_matching_and_skip_edge_cases
- FAILED tests/test_audit_fixes.py::test_pipeline_cancel_speculative_priors_context_and_verifications
- FAILED tests/test_audit_fixes.py::test_apply_promotion_calls_merge_on_both_stores
- FAILED tests/test_audit_fixes.py::test_no_fake_constants_in_extractors_and_workers
- FAILED tests/test_pipeline_integration.py::test_pipeline_metamorphic_response_latency_and_question_classification
```

### Green-After-Fix (`python -m pytest tests/test_audit_fixes.py tests/test_pipeline_integration.py -q`)
```text
22 passed in 0.98s
```

---

## Final Full-Suite Verification

### Full Test Suite (`python -m pytest -q`)
```text
........................................................................ [ 62%]
............................................                             [100%]
116 passed in 5.06s
```

### AST Stub Scan (`src/dce/**/*.py`)
```text
AST_SCAN files=16 stubs=[] todos=[]
```
