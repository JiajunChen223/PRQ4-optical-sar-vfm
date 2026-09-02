# Independent V6 CC-SCBC test audit

**Audit date:** 2026-08-29  
**Scope:** `F:\PRQ4\02_experiment\code` frozen V6 CC-SCBC implementation  
**Route:** `R-EO-CCSCBC-01` / `CC-SCBC-01`  
**Audit type:** read-only local contract and reproducibility audit

## Boundary

This audit did not open or inspect SEN12TS pixels, labels, caches, real
pretrained weights, or sealed-test artifacts. It did not open an SSH session,
invoke CUDA/GPU discovery, or run training/evaluation. The synthetic liveness
run used CPU tensors only. Therefore the result below is evidence for the
implementation contract, not a scientific result, a cloud preflight, or a
promotion of C1.

## Executed checks

All Python commands used `F:\anaconda3\envs\dl_env\python.exe` with
`PYTHONDONTWRITEBYTECODE=1`; pytest used `-p no:cacheprovider`.

| Check | Result |
|---|---|
| V6 targeted unit suite, `tests\\unit\\test_cc_scbc.py` | **14 passed** in 2.11 s |
| Full local suite, all configured tests | **291 passed**, 1 pre-existing warning, 8.21 s |
| AST parse of project Python files | **79 files, 0 errors** |
| ResearchPilot code validator | **PASS**, 106 executable/config files, 0 violations |
| Synthetic Jacobian liveness | **PASS**, CPU-only; no data/weights/GPU/training/evaluation |

The sole pytest warning is an existing scalar-conversion warning in
`tests/unit/test_ceak_successor.py`; it is outside the V6 path and did not
fail the suite.

## Adversarial contract results

The independent ephemeral checks covered 45 assertions; all **45 passed**.

- Exact value and storage identity held for `class_set`,
  `uniform_score_backward`, and `class_agnostic` modes, including nonzero
  beta.
- A one-token upstream signal produced finite off-diagonal SAR-token credit;
  the non-diagonal Jacobian witness was nonzero (`off-token norm` 0.6396 in
  the adversarial fixture), and beta received a finite nonzero gradient.
- Permuting detached class anchors preserved the forward output but changed
  SAR-token credit, supporting the class-conditioning witness.
- C2 removed the score derivative while retaining its declared control path;
  C3 exposed the class-agnostic collapse. C1 gradients differed from both
  controls in the matched fixture.
- Classifier weight/bias gradients were absent through the auxiliary
  scaffold, and the fused-feature gradient matched the direct upstream
  gradient exactly.
- Evaluation bypass returned the exact fused tensor and marked the
  auxiliary path inactive.
- CPU BF16 autocast-like execution produced finite outputs, telemetry, and
  gradients.
- Two-stage model checks in both train and eval mode matched `always_fuse`
  outputs exactly after state transfer; hard routes, state-dict keys, and
  trainable-parameter names matched.
- All three declared V6 modes retained the same state-dict and trainability
  surface as the baseline model.
- V6 baseline/candidate config resolution differed only at
  `model.mechanism_set`, shared the matched protocol hash, and resolved to
  `R-EO-CCSCBC-01` / `CC-SCBC-01`.

## Validator and seal boundary

The validator returned `status=pass`, `problems=[]`, `violations=[]`,
`local_real_data_allowed=false`, and `local_gpu_probe=forbidden_not_run`.
The runtime test-seal guard is present and both entry points call it before
validation/split execution. Cloud-path strings found by static scanning are
declared remote paths or test fixtures; no local data or checkpoint binary was
found in the code-sync scope.

## Snapshot and receipts

The auditable code/config/test snapshot contains 109 files and 980,088 bytes.
Its canonical path/content aggregate SHA256 is:

`1186016cf78f721089e2b1d439d0d6af6daf792093d4ca97ddb3471da36cb3b3`

Key file hashes:

| File | SHA256 |
|---|---|
| `src/geotoken3path/mechanisms/cc_scbc.py` | `b3f3e8852e3c05733527c3d4f1035b4d62c3f682dcc041079edf14005835da70` |
| `src/geotoken3path/models/fusion.py` | `bd940fb2961e248fb173664371b085ff9f8c6b24c92eee3f9420aefa0c33668e` |
| `tests/unit/test_cc_scbc.py` | `ea5411dd853c21b5471ff319be7778807ed3870fe37fa85272dff967f86e23a8` |
| `configs/model/v6_cc_scbc.yaml` | `6cd88ff793ba5b27c440ee779a1e2eb4f8c67460b51b9bea2ed0466f74ae290c` |
| `configs/experiment/v6_cc_scbc_route.yaml` | `80fd2d0d926ef51caf41fb25ffc35a597f277a96f493e9a6b82d57e044c76e1b` |

Validator receipt:

`F:\PRQ4\02_experiment\code\review\independent_v6_cc_scbc_validator_20260829.json`  
file SHA256: `ad3e851a515a0b0258c4d5bcaa77fce566e4763b41b847e158482ce22ce9a91e`

Synthetic liveness receipt:

`F:\PRQ4\02_experiment\reports\v6_cc_scbc_synthetic_jacobian_liveness_20260829.json`  
file SHA256: `74f6cd418aadc8c3d653441e552f9d4e6ea96835892606eaa20556368be0f778`  
embedded receipt SHA256: `90468afeb13e8f4bf9ec2b0ce760c352276248bd2c9811545708f2fff21f9be4`

## Independent disposition

**LOCAL_CONTRACT_PASS.** No blocking implementation or test failure was
found in this read-only audit. The evidence is sufficient to hand the code
back to the Experiment workflow for guarded cloud reattachment and the
approved V6 C1 seed-0 24-epoch launch procedure. It does **not** authorize
that launch by itself and does not alter the rule that C2/C3 may run only if
C1 reaches `50.0075%` mIoU.
