# Independent CEAK-R1~R4 closure review

**Review date:** 2026-08-28  
**Scope:** current local snapshot of `F:\PRQ4\02_experiment\code`  
**Decision:** **PASS — CEAK-R1, CEAK-R2, CEAK-R3, and CEAK-R4 are closed in the current code snapshot.**

This is a code-contract closure result only. It is not evidence for a baseline,
candidate, cloud run, RTX-3090 performance, real-data compatibility, pretrained
checkpoint compatibility, scientific gain, or sealed-test access.

## Guardrails and review boundary

- Read-only inspection of source, configuration, tests, and prior local review
  artifacts.
- CPU-only synthetic pytest and local static validation were run.
- No cloud connection, real imagery/labels, real weights/checkpoint binary,
  GPU probe, training, or sealed-test access was performed.
- No source, configuration, or test file was changed by this review. The only
  requested write is this review receipt.
- The prior blocker report
  `F:\PRQ4\02_experiment\code\review\independent_ceak_code_review.md`
  is historical for the pre-successor snapshot; its four findings were
  rechecked against the current source rather than treated as current status.

## Executed checks

| Check | Command / evidence | Result |
|---|---|---|
| CPU synthetic regression | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest -q`, working directory `F:\PRQ4\02_experiment\code` | **224 passed**, 1 warning, 7.79 s |
| Static ResearchPilot validator | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4` | **PASS**, 77 executable/config files, 0 problems, 0 violations; `local_real_data_allowed=false`; `local_gpu_probe=forbidden_not_run` |
| Source-level closure probe | CPU-only in-memory tensors; no file/data/checkpoint I/O | **PASS** for R1–R4 probes below |

The pytest warning is from the existing assertion converting a grad-bearing
telemetry tensor to `float` in
`tests/unit/test_ceak_successor.py`; it does not affect pass/fail or the
closure checks.

## Closure matrix

### CEAK-R1 — zero-start no-op and gradient-live initialization: CLOSED

Current implementation in
`src/geotoken3path/models/fusion.py:806-818` uses
`scaled + (value - value.detach())`. At scale zero the forward residual is
exactly zero, while the straight-through term gives the mechanism path a unit
initial derivative. The branch is used by CEAK, SUBPACK, and CFEDGE at
`fusion.py:871`, `943`, and `997`.

Independent CPU probe results after loading the same baseline state dict into
each candidate:

| Direction | Bitwise output identity | Scale gradient max | Minimum mechanism gradient max |
|---|---:|---:|---:|
| CEAK-01 | `True` | 0.4988232553 | 0.0000554810 |
| SUBPACK-02 | `True` | 0.6103379726 | 0.0014582730 |
| CFEDGE-03 | `True` | 0.6496550441 | 0.0168620273 |

This closes the previous scalar-only gate defect. The targeted regression is
also present in `tests/unit/test_ceak_successor.py:31-45` and passed in the
full suite.

### CEAK-R2 — rank/M/K controls are resolved, hashed, and consumed: CLOSED

The current constructor validates and stores `evidence_rank`,
`subpack_candidate_limit`, and `subpack_edge_budget` at
`src/geotoken3path/models/fusion.py:102-125,182-207`; the execution path uses
the stored limits at `fusion.py:927-938`. `factory.py:33-57` passes the
resolved `successor_edge_contract` into the shared model, and
`fusion.py:1236-1278` passes it to every stage.

`config.py:195-207` fail-closes the approved CEAK values, while
`config.py:324-345` records them in the immutable resolved snapshot and hence
the matched common protocol hash. `tests/unit/test_resolved_config.py:55-72`
passed the invalid-value cases.

Independent binding probe:

- All three resolved CEAK rows carried
  `evidence_rank=8`, `subpack_candidate_limit=64`, and
  `subpack_edge_budget=32`.
- Both `mid` and `late` fusion stages consumed `(8, 64, 32)`.
- A synthetic constructor override `(4, 16, 8)` produced both stages with
  `(ceak_rank, candidate_limit, edge_budget)=(4, 16, 8)` and evidence-head
  output width 4, demonstrating that the controls are execution inputs rather
  than unused YAML declarations.
- Invalid direct controls (`rank=0`, `M=0`, and `K=33` with `M=32`) were
  rejected before model construction.

### CEAK-R3 — candidate-direction ID is bound to mechanism set: CLOSED

The exact mapping is defined in
`src/geotoken3path/utils/run_manifest.py:97-101` and enforced for non-smoke
candidate rows at `run_manifest.py:338-348`. The formal runner carries the same
mapping and rejects missing or mismatched IDs before data-loader/model work at
`src/geotoken3path/engine/formal_runner.py:27-31,238-246`.

Independent manifest probe:

- `CEAK-01` → `evidential_conflict_null_sink_cross_attention_kernel` passed.
- `SUBPACK-02` → `conflict_weighted_logdet_submodular_edge_packet_selection`
  passed.
- `CFEDGE-03` → `counterfactual_shapley_edge_credit_with_private_bypass`
  passed.
- Missing IDs were rejected for all three formal candidate rows.
- Wrong IDs were rejected by both `build_run_manifest` and the formal-runner
  preflight for all three rows with
  `candidate_direction_id does not match the selected CEAK mechanism`.

The omission of a direction ID remains allowed only for local `smoke` manifest
construction; formal candidate rows require it. This is consistent with the
local synthetic smoke boundary and does not weaken cloud-row identity binding.

### CEAK-R4 — CFEDGE Shapley naming/operation boundary: CLOSED

`src/geotoken3path/models/fusion.py:970-987` evaluates the four coalitions
`v(OS)`, `v(O)`, `v(S)`, and `v(empty)`, then computes both individual two-player
Shapley allocations:

```text
phi_O = 0.5 * ((v(O)-v(empty)) + (v(OS)-v(S)))
phi_S = 0.5 * ((v(S)-v(empty)) + (v(OS)-v(O)))
```

The signed interaction finite difference remains a separate routing quantity,
and the telemetry is explicitly exposed as
`cfedge_optical_shapley_mean` and `cfedge_sar_shapley_mean`. The private SAR
path and null routing remain in the same branch at `fusion.py:988-1010`.

Independent CPU reconstruction of the four coalition values matched both
telemetry values (`torch.allclose=True`); the maximum Shapley efficiency error
`|phi_O + phi_S - (v(OS)-v(empty))|` was `1.49e-08`, with finite logits and
positive/negative interaction fractions summing to 1. The targeted test
`tests/unit/test_ceak_successor.py:93-112` also passed.

## Source hashes captured for this review

| File | SHA-256 |
|---|---|
| `src/geotoken3path/models/fusion.py` | `6CCD3372629D5E6FAB44F8BDBDD813DCAFAAC572B5B90DA572B8F6B87FA97E68` |
| `src/geotoken3path/utils/config.py` | `2CCE5FF6676BADCB6A5765C9351B5CFB7916041E3E756362E7E9CF94D5C5A57F` |
| `src/geotoken3path/utils/run_manifest.py` | `BA48FF1681F577EDACBC217AE1DDD9950A8DE6062564BBF1909CCA1FE44A30C7` |
| `src/geotoken3path/engine/formal_runner.py` | `456AC3906C9F2BDA0457B9C505E992B9E75B69F207647FC720DCFA77FB70C39B` |
| `src/geotoken3path/models/factory.py` | `B1357A30EA521D7AFAF0EDDC23D1F47B61F4DB1BE40CFE0FB59A915BFE80D8FE` |
| `tests/unit/test_ceak_successor.py` | `744B8228486BE88E94D7E4A4DC88F08480D3BA4BBDB1B9A92D7463AC04B5350E` |
| `tests/unit/test_resolved_config.py` | `14085D47B64F9C219B2D4D9C9EF281082A82EA252253086FBB70448CC9C20960` |
| `configs/model/geotoken3path.yaml` | `596F41C3B4F6E072DB2CA6F97C99AFA149833DA661A931D7B149B68E362E5527` |

## Handoff boundary

The four historical CEAK code-contract blockers are closed locally. This
receipt does not reopen or advance any ResearchPilot experiment gate. Any next
cloud action still requires the separately authorized cloud controls, audited
initialization/data artifacts, parity checks, and sealed-test policy.
