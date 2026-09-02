# Independent V6 CC-SCBC code review

**Review date:** 2026-08-29  
**Scope:** `F:\PRQ4\02_experiment\code` current V6 snapshot  
**Route:** `R-EO-CCSCBC-01` / `CC-SCBC-01`  
**Review mode:** independent, read-only source/contract audit

## Boundary

This review did not open SEN12TS pixels, labels, caches, real CROMA weights,
checkpoints, or the sealed-test split. It did not open SSH, inspect CUDA/GPU
state, download anything, or run training/evaluation. The CPU checks below are
synthetic implementation checks only. The report writes no scientific result
and cannot authorize a cloud run.

## Executive finding

The V6 mechanism implementation itself is **CONDITIONAL_PASS_LOCAL_HARD_CONTRACT**:
the custom operator is forward-identical, its backward recomputation retains a
non-diagonal token-set Jacobian, class anchors/responsibilities are detached,
and the candidate is matched to `always_fuse` in the local model surface.

The current snapshot is **BLOCKED_FOR_CLOUD_C1_ENTRY_BINDING**. The formal cloud
runner has not been updated for `CC-SCBC-01`, and therefore a C1 invocation with
the approved candidate direction is rejected before any data or model access.
The clean-sync manifests and canonical `CODE_REPORT.json` are also stale
(D3/V5); they cannot be used as the V6 package binding.

## Checks and results

### 1. CC-SCBC operator contract — PASS

- `src/geotoken3path/mechanisms/cc_scbc.py:64-79` returns `fused_tokens`
  directly from the custom autograd forward. The three modes are bitwise
  identical to the input in the current tests, including nonzero beta.
- `:82-115` sends the ordinary downstream gradient to `fused_tokens` and a
  separately recomputed surrogate gradient only to `sar_tokens`; classifier
  anchors, responsibilities and mode metadata receive no gradient.
- `:41-57` computes class-specific token-axis softmax scores and retains the
  softmax derivative in `uniform + raw - raw.detach()` for C1/C3. A CPU
  adversarial check produced finite off-token credit; the synthetic liveness
  receipt reports `cross_token_gradient_ratio=0.7469740547`.
- `:203-244` performs score/probability calculations in float32 and bounds the
  stage scale with `tanh(beta)`. CPU BF16 autocast-like forward/backward checks
  were finite.
- `:277-301` returns the untouched fused tensor and marks the scaffold
  inactive in evaluation mode.

### 2. Matched dispatch and model surface — PASS

- `models/fusion.py:1530-1532` assigns CC-SCBC to the same all-current hard
  route as `always_fuse`; `:1609-1616` fixes the local window to one token.
  Both therefore enter the ordinary `_dispatch` same-index exchange at
  `:1728-1735`.
- `:1925-1955` applies the scaffold only after the matched local fusion and
  calls the explicit eval bypass when `self.training` is false. No new forward
  fusion module or external router is introduced.
- A two-stage, batch-1/2 synthetic comparison after state transfer gave exact
  train/eval output equality (`torch.equal=True`) against `always_fuse`; hard
  routes were all current.
- Full resolved V6 model checks show equal state-dict key order (`180` keys)
  and equal trainable-parameter names (`168` names) for baseline and C1.

### 3. Configuration and manifest binding — PASS locally, incomplete formally

- `utils/config.py:232-252` enforces the 11-class, token-axis, beta-zero,
  bounded, detached, forward-identity, inference-removable and non-diagonal
  contract. `:487-501` resolves the V6 model/route pair without rewriting V5.
- `utils/run_manifest.py:103-123` and `:370-382` register
  `CC-SCBC-01 -> cc_scbc_class_conditioned_set_credit`. Local V6 baseline/C1
  resolution differs only at `model.mechanism_set` and shares the common
  protocol SHA.
- `scripts/train.py:31-35,179-180` recognizes all three V6 modes and selects
  the V6 resolver.

### 4. Blocking formal-entry defect — BLOCK

`src/geotoken3path/engine/formal_runner.py` is still the pre-V6 entry surface:

- `:27-32` `_SUCCESSOR_DIRECTION_TO_MECHANISM` contains CEAK, SUBPACK, CFEDGE
  and IF-SGC but omits `CC-SCBC-01`.
- `:241-242` repeats an inline candidate-direction allowlist that also omits
  `CC-SCBC-01`.
- `:243-250` enforces that map and requires a direction for successor
  mechanisms. Consequently, `run_formal_cloud(... mechanism_set=
  "cc_scbc_class_conditioned_set_credit", candidate_direction_id=
  "CC-SCBC-01")` cannot pass the pre-training contract. This is an
  `invalid_protocol` preflight blocker, not a scientific result.

`scripts/evaluate.py` is a secondary V6 entry gap: its `--mechanism-set`
choices do not include the three CC-SCBC modes and its local resolver dispatch
only recognizes V5 IF-SGC. This should be repaired before any V6 evaluation
artifact is requested, even though the formal runner performs the actual
cloud validation loop.

### 5. Code/package freshness — BLOCK_FOR_PACKAGE

The current tree has no V6 clean-sync manifest. Existing manifests are
`clean_sync_manifest_d3_20260829.json` and
`clean_sync_manifest_v5_if_sgc_20260829.json`; `review/CODE_REPORT.json` is
also bound to the D3 manifest. A fresh V6 clean export, source hash, and
guarded code-only sync are required after repairing the formal-entry binding.
No cloud data/weights/training action should be attempted from those stale
bindings.

## Verification receipts

- V6 targeted unit suite: **14 passed**.
- Full local suite after the latest eval-removal test: **291 passed**, one
  pre-existing warning in `test_ceak_successor.py`.
- Independent AST/YAML parse: **79 Python files + 12 YAML files, 0 errors**.
- Existing ResearchPilot validator receipt:
  `independent_v6_cc_scbc_validator_20260829.json` — `status=pass`, 106
  executable/config files, 0 violations, local GPU probe
  `forbidden_not_run`.
- Synthetic Jacobian liveness:
  `reports/v6_cc_scbc_synthetic_jacobian_liveness_20260829.json` —
  `overall_pass=true`, CPU-only, no data/weights/GPU/training/evaluation,
  embedded receipt SHA
  `90468afeb13e8f4bf9ec2b0ce760c352276248bd2c9811545708f2fff21f9be4`.

## Current disposition and required repair

**Disposition: BLOCKED_FOR_CLOUD_C1_ENTRY_BINDING.**

The local hard-contract evidence is sufficient to continue code repair and
fresh package review. It is not sufficient to launch C1. Before cloud
preflight, repair both formal-entry allowlists/maps, add a regression test that
the C1 formal request reaches the normal preflight rather than rejecting its
direction, regenerate a V6 clean-sync manifest/CODE_REPORT, and pass the
guarded code-only sync. Preserve the sealed test and all V5/D3 artifacts.
After those gates, C1 alone may run for the approved 24-epoch seed-0 screen;
C2/C3 remain forbidden unless C1 reaches `50.0075%` mIoU.

## Current relevant SHA256

| File | SHA256 |
|---|---|
| `src/geotoken3path/mechanisms/cc_scbc.py` | `b3f3e8852e3c05733527c3d4f1035b4d62c3f682dcc041079edf14005835da70` |
| `src/geotoken3path/models/fusion.py` | `bd940fb2961e248fb173664371b085ff9f8c6b24c92eee3f9420aefa0c33668e` |
| `src/geotoken3path/utils/config.py` | `1d4558acb23b70a02f5aa757c8092bf5e6ba16a5f6e3c95fae51a70451f93e67` |
| `src/geotoken3path/utils/run_manifest.py` | `8283658fefe4e849d5281e222c8039bea1f211ea2aa5ca4a2c0afb6bb03649d3` |
| `src/geotoken3path/engine/formal_runner.py` | `d4ce7918588d9569f1e1b3f10cc95fdde8ead2d5488d1d869be06eb98b036c25` |
| `scripts/train.py` | `2b9cf08ffdba430b119626d9226c6f2c5b7a16c56e8a46eba7b711db730d181e` |
| `scripts/evaluate.py` | `a89e49ad7d05f1ab8a0c15b0d8e41ec218e4fd57951c91f06291efe0a0c4cc27b` |
| `tests/unit/test_cc_scbc.py` | `ea5411dd853c21b5471ff319be7778807ed3870fe37fa85272dff967f86e23a8` |
| `configs/model/v6_cc_scbc.yaml` | `6cd88ff793ba5b27c440ee779a1e2eb4f8c67460b51b9bea2ed0466f74ae290c` |
| `configs/experiment/v6_cc_scbc_route.yaml` | `80fd2d0d926ef51caf41fb25ffc35a597f277a96f493e9a6b82d57e044c76e1b` |

All hashes are lower-case SHA256 values emitted from the current files. No
source file was modified by this review.
