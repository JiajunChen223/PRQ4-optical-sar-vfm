# D0 architecture and protocol review

## Scope

This read-only review covers the new checkpoint-level diagnostic layer for
`PLAN_REVISION_PRQ4_V1`: `src/geotoken3path/diagnostics.py`,
`scripts/diagnose_d0.py`, the D0 telemetry additions in
`src/geotoken3path/models/fusion.py`, and the associated unit tests. It does
not treat a diagnostic receipt as a scientific result and does not authorize
training, composition, confirmation, or sealed-test access.

## Checks performed

- `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest tests -q --disable-warnings`
  → 234 passed.
- `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m compileall -q src tests scripts`
  → pass.
- ResearchPilot code validator against `F:\PRQ4` → pass, zero violations,
  `local_gpu_probe=forbidden_not_run`.
- Read-only line inspection of the D0 runner and CROMA bridge/model interfaces.

## Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| D0-A01 | note | A0 uses the validation loader's ordered paired rows; A1 applies one deterministic global derangement; A2 applies one deterministic per-batch derangement. Unique parent IDs are required before construction, and permutation hashes are recorded. | fixed/pass |
| D0-A02 | note | D0-B shifts only post-encoder SAR stage mappings and the four-way SAR depth groups, while optical mappings and image preprocessing remain unchanged. The shift grid is exactly the seven plan-declared displacements. | fixed/pass |
| D0-A03 | note | D0 telemetry is detached and emitted only through `return_aux=True`; the formal training path does not request auxiliary outputs. CEAK and CFEDGE pairwise variables are required and missing variables fail closed. | fixed/pass |
| D0-A04 | note | The runner requires validation-only loading, 180 records, sealed test status, CUDA execution, distinct output/input locations, and one common resolved protocol hash across baseline and candidates. | fixed/pass |
| D0-A05 | note | The receipt explicitly sets `scientific_result=false`, forbids candidate ranking/composition/confirmation/final-test actions, and records 24 epochs as the frozen ranking horizon. | fixed/pass |
| D0-A06 | note | The cloud runner computes and records actual input checkpoint hashes; the exact expected hashes remain bound by the execution control/command and must be checked in the cloud receipt before interpretation. | accepted risk |

## Decision

`CONDITIONAL_PASS_FOR_LOCAL_D0_CODE_REVIEW`.

The local D0 implementation is structurally consistent with the attached plan
and is eligible for a reviewed code-only synchronization. It is not evidence
that any mechanism is supported, and it cannot be used to open the sealed test.
Cloud execution remains gated by the canonical Experiment state and the
one-use code-sync control.
