# Independent CROMA bridge code review — 2026-08-22 r1

## Scope and boundary

This is a local code-service review of the keyword/dtype bridge patch only. It
uses synthetic fixtures and static inspection. No real data, checkpoint binary,
GPU probe, training, evaluation, metrics, or sealed-test access occurred.

Reviewed files:

- `02_experiment/code/src/geotoken3path/models/croma_bridge.py`
- `02_experiment/code/tests/integration/test_croma_bridge.py`
- `02_experiment/code/manifests/clean_sync_manifest.json`

Source SHA256: `00f21f4a1fbbf3b65ffa2a9d3aef2f8a3e0c6e7d3890be1debc847b22ac19af1`  
Test SHA256: `dbc58442af247b37c5c132669a4ddb3b9faa66d6d389de686c8a4224f6b512e8`  
Clean-manifest SHA256 after the patch: `77b3a0aab8977d5c6fae5f5158a297bc263d6463bcc9e1105388513fa5f2e97d`

## Role findings

| Role | Verdict | Evidence | Finding |
|---|---|---|---|
| Architecture/modularity | PASS | `CromaBackboneBridge` remains dependency-injected; the model factory and common train/evaluate path are unchanged | Keyword binding and dtype validation are located at the VFM boundary; no new external router or hidden trainable wrapper was introduced. |
| Data/leakage | PASS (local) | final local-data audit `croma_bridge_local_data_audit_20260822_r3.json`; validator scan | The patch contains no local data root, sample bytes, checkpoint, or test-split access. Real SEN12TS/CROMA compatibility remains a cloud/data-gate issue, not local evidence. |
| Reproducibility/run tracking | PASS | 110 synthetic tests; immutable manifest and test-seal tests; `validate_code_project.py` | The bridge change is hash-bound in the clean manifest and tested through the same injected-backbone path used by the model factory. |
| Public-release/dependency hygiene | PASS | package allowlist and path/secret scan | No credential, absolute local path, binary weight, cache, link, or forbidden archive member is introduced by the changed files. |
| Adversarial failure/path/secret scan | PASS | negative channel, feature-key, and dtype tests; validator reports zero violations | Wrong channels and non-float32 inputs fail closed. No bypass of the sealed-test guard or formal runner was found. |

## Tests and static checks

- Full local synthetic suite: **110 passed**.
- `validate_code_project.py`: **PASS**, 39 executable/config files, 0
  violations, `local_gpu_probe=forbidden_not_run`.
- Local real-data policy: **PASS**, 0 suspects.
- Targeted bridge tests: 3 passed in the preceding bridge audit.

## Residual scientific/cloud issue

The official CROMA loader's published output is final encodings/GAP, while the
GeoToken-3Path contract requires explicit mid/late optical and SAR tokens plus
`sar_fine[B,N,4,D]`. The patch closes only the positional keyword-swap risk and
enforces float32 at the interface. It does **not** prove or invent the missing
feature adapter. That adapter must be resolved by a cloud-only official-source /
checkpoint audit and, if code changes are needed, returned through this local
review and a new guarded sync package.

## Decision

`CONDITIONAL_PASS_CODE_ONLY_SYNC_PREPARATION`.

The local code patch has no unresolved code-service blocker and is eligible for
clean-package preparation. This decision is not a CROMA compatibility pass,
data-gate pass, baseline-training approval, or scientific result.
