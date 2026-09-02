# Independent CEAK successor re-audit

**Audit date:** 2026-08-28  
**Scope:** the current local snapshot under `F:\PRQ4\02_experiment\code` only.  This audit used CPU synthetic tensors and static source/config inspection.  It did not connect to the cloud host, read real imagery/labels/checkpoints or pixels, probe a GPU, modify source files or `CODE_REPORT.json`, or access the sealed test split.

## Decision

**BLOCK — the operator implementation is structurally visible and the two formerly reported implementation defects are closed, but the snapshot is not yet eligible for code sync or cloud screening.** Two manifest-level contract holes remain:

1. the resolved CEAK rank/M/K contract is present and consumed by the factory, but `build_run_manifest()` accepts a tampered resolved contract and does not bind those values into the emitted run manifest;
2. successor direction IDs are correctly checked for successor mechanisms, but a baseline mechanism can still be paired with an arbitrary successor direction ID (for example `always_fuse + CEAK-01`).

These are pre-experiment identity/reproducibility blockers, not scientific results.

## Executed checks

| Check | Result | Evidence |
|---|---:|---|
| Full CPU synthetic suite | **PASS** | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest 02_experiment/code/tests -q`: **224 passed, 0 failed** (one existing scalar-conversion warning) |
| Python compilation | **PASS** | `compileall -q 02_experiment/code/src 02_experiment/code/tests` |
| ResearchPilot code validator | **PASS** | `validate_code_project.py --project-root F:\PRQ4`: 77 executable/config files scanned, 0 problems, 0 violations; `local_gpu_probe=forbidden_not_run` |
| Synthetic CROMA VFM parity | **PASS** | For CEAK-01, SUBPACK-02 and CFEDGE-03: identical state-dict key order, identical `requires_grad` masks, exact zero-start logits after matched state loading, dense output shape `(1,6,32,32)` |
| Candidate zero-start gradients | **PASS** | At the actual zero-start state, all tested internal mechanism parameters had finite non-`None` gradients; see below |
| CFEDGE allocation | **PASS** | Four coalition utilities and two individual Shapley allocations were recomputed; conservation error `max |phi_O+phi_S-(v(OS)-v(empty))| = 2.98e-8` on CPU synthetic data |
| SUBPACK budget | **PASS** | At `N=225`, resolved/factory path reported `M=64`, `K=32`, exactly 32 selected packets, finite output |

## Closed findings

### Zero-start no-op plus gradient-live

The residual uses the straight-through zero-start construction in `GeoToken3PathFusion._zero_start_residual`: the forward residual is exactly zero at a zero scale while the mechanism path retains a live Jacobian.  After loading a matched `always_fuse` state dict, all three successor mechanisms were bitwise equal to the baseline on CPU synthetic inputs and still produced finite internal gradients:

| Mechanism | Scale gradient norm | Representative internal gradient norm |
|---|---:|---:|
| CEAK-01 | 0.1363 | `ceak_query.weight`: 0.1162 |
| SUBPACK-02 | 0.3311 | `ceak_query.weight`: 0.0358 |
| CFEDGE-03 | 0.4570 | `cfedge_utility.0.weight`: 0.0137 |

The same zero-start and gradient reachability behavior was observed through the synthetic CROMA tap bridge.  This closes the earlier scalar-only gate defect.

### CFEDGE four-coalition/Shapley operation

The current branch evaluates `paired`, `optical_only`, `sar_only` and `null` coalition utilities and computes the individual two-player Shapley allocations

`phi_O = 1/2[(v(O)-v(empty)) + (v(OS)-v(S))]`

and

`phi_S = 1/2[(v(S)-v(empty)) + (v(OS)-v(O))]`.

The implementation also uses a separate signed interaction term for routing.  The code therefore no longer overclaims the interaction finite difference as the individual allocation.  Causal language remains prohibited by the code/design contract.

### State-dict and trainability parity

All three candidate branches allocate the same shared operator surface, so the baseline and each candidate have identical state-dict keys and trainability masks.  Candidate-only branches are selected only by `mechanism_set`; no mechanism-specific optimizer or backbone policy was introduced.  Synthetic CROMA bridge checks showed the same tap paths and parity surface.

### Candidate mechanism registration

The three mechanism names are registered in `GeoToken3PathFusion.VALID_MECHANISMS`, fusion dispatch, the factory, the formal runner, the training/evaluation CLI choices, and the approved route/config.  The approved route remains `R-EO-CEAK-01`, primary `CEAK-01`, with `SUBPACK-02` and `CFEDGE-03` as the two controls.  Task, dataset, CROMA initialization, baseline, 24-epoch budget, RTX 3090 target, and sealed-test status were not changed.

## Blocking findings

### CEAK-R2 — resolved rank/M/K is not fail-closed at manifest construction (P1)

The YAML resolver now checks and emits:

```text
resolved.model.successor_edge_contract = {
  evidence_rank: 8,
  subpack_candidate_limit: 64,
  subpack_edge_budget: 32
}
```

`build_model(resolved)` consumes those values, and YAML mutations to rank 9, M 65 or K 33 are rejected by `resolve_approved_config()`.  However, `_validated_resolved_snapshot()` in `src/geotoken3path/utils/run_manifest.py` does not validate `resolved.model.successor_edge_contract`, and `build_run_manifest()` does not emit the contract as a manifest field.

Independent mutation test: starting from a valid resolved CEAK snapshot, changing each of `evidence_rank`, `subpack_candidate_limit` and `subpack_edge_budget` to an alternative value was **accepted** by `build_run_manifest()` for a screening row.  The emitted run-contract hash was unchanged because the contract was not part of the manifest payload and the pre-existing protocol hash was trusted.  Thus a model can be constructed with one M/K/rank contract while its run manifest records no corresponding binding.

**Required closure:** validate the successor contract in `_validated_resolved_snapshot()` with the same exact `rank=8`, `M=64`, `K=32` constraints and either include it in the manifest payload or recompute/verify the common protocol digest from the validated snapshot.  Add a regression test that mutates each value after resolution and requires rejection.

### CEAK-R3 — baseline rows accept successor direction IDs (P1)

For a successor mechanism, the new exact mapping works: a missing ID and mismatches such as `CEAK mechanism + SUBPACK-02` are rejected by `build_run_manifest()`; the formal runner has the same mapping.  The remaining hole is the reverse direction: an `always_fuse` resolved row accepted each of `CEAK-01`, `SUBPACK-02`, `CFEDGE-03` and even `PCTA-01` as `candidate_direction_id` during a screening manifest build.

That permits a baseline run to be labeled as a candidate direction.  It is especially undesirable now that the approved portfolio has one primary mechanism and two independent controls.

**Required closure:** for baseline/control rows, require no candidate direction ID unless a canonical control mapping is explicitly declared; at minimum reject the three CEAK successor IDs whenever `mechanism_set=always_fuse`, and keep the same rule in the formal runner and CLI path.  Add both positive and reverse-mismatch tests.

## Current code hashes

These hashes identify the reviewed snapshot; they are not a release manifest:

```text
fusion.py                 6CCD3372629D5E6FAB44F8BDBDD813DCAFAAC572B5B90DA572B8F6B87FA97E68
config.py                 2CCE5FF6676BADCB6A5765C9351B5CFB7916041E3E756362E7E9CF94D5C5A57F
run_manifest.py           BA48FF1681F577EDACBC217AE1DDD9950A8DE6062564BBF1909CCA1FE44A30C7
formal_runner.py          456AC3906C9F2BDA0457B9C505E992B9E75B69F207647FC720DCFA77FB70C39B
factory.py                B1357A30EA521D7AFAF0EDDC23D1F47B61F4DB1BE40CFE0FB59A915BFE80D8FE
test_ceak_successor.py   744B8228486BE88E94D7E4A4DC88F08480D3BA4BBDB1B9A92D7463AC04B5350E
```

## Re-review gate

Until CEAK-R2 and CEAK-R3 are repaired and the full synthetic/static checks are rerun, this audit is **BLOCK**.  No cloud sync, data access, pretrained-weight access, GPU execution, training, metric generation or sealed-test action is authorized by this report.
