# Independent V6 CC-SCBC test re-audit (current snapshot)

**Date:** 2026-08-29  
**Scope:** `F:\PRQ4\02_experiment\code`  
**Route:** `R-EO-CCSCBC-01` / `CC-SCBC-01`  
**Purpose:** re-audit after the formal-runner/evaluate entry whitelist and
successor mapping repair

## Boundary

This was a read-only local re-audit. No SEN12TS pixels, labels, caches, real
pretrained weights, SSH endpoints, CUDA/GPU probing, training, or sealed-test
artifacts were accessed. All synthetic checks ran on CPU. The results are
implementation-contract evidence only; they are not C1 scientific metrics and
do not authorize a cloud run or C2/C3.

## Current snapshot

The auditable source/config/test snapshot contains 109 files and 980,626
bytes. The canonical path/content aggregate SHA256 is:

`81ebd6341147fef6fbb6a8038a10bb1500a858f70c8eac82a7f2933c3c6023f3`

Current key-file hashes:

| File | SHA256 |
|---|---|
| `src/geotoken3path/engine/formal_runner.py` | `324ebdfb4c155bb5d17acd3a36c8368a72433367502ef901eecd90c4b0bd6ed3` |
| `scripts/evaluate.py` | `cb78303b1e56b441a0923e561d7f5c93a064709a5aea61f9934833040c6b1a78` |
| `src/geotoken3path/mechanisms/cc_scbc.py` | `b3f3e8852e3c05733527c3d4f1035b4d62c3f682dcc041079edf14005835da70` |
| `src/geotoken3path/models/fusion.py` | `bd940fb2961e248fb173664371b085ff9f8c6b24c92eee3f9420aefa0c33668e` |
| `tests/unit/test_cc_scbc.py` | `b0c78fd97b0e9c459cb1d3e5216475eedfaec220408ea68fe7ab81e2b7a3fa9e` |

The source change relevant to this re-audit is confined to formal successor
entry validation/mapping and its coverage; the CC-SCBC mechanism implementation
hash is unchanged.

## Re-executed checks

All Python commands used `F:\anaconda3\envs\dl_env\python.exe` with
`PYTHONDONTWRITEBYTECODE=1`; pytest used `-p no:cacheprovider`.

| Check | Result |
|---|---|
| V6 targeted suite, `tests\\unit\\test_cc_scbc.py` | **14 passed** in 4.02 s |
| Full local suite | **291 passed**, 1 pre-existing warning, 7.51 s |
| AST parse | **79 files, 0 errors** |
| ResearchPilot validator | **PASS**, 106 executable/config files, 0 problems, 0 violations |
| Synthetic Jacobian liveness | **PASS**, CPU-only, no data/weights/GPU/training/evaluation |

The one pytest warning remains the existing scalar-conversion warning in
`tests/unit/test_ceak_successor.py`; it is outside the V6 path and is
non-blocking.

## Entry-point smoke checks

Both entry points were run in their local synthetic lane with
`cc_scbc_class_conditioned_set_credit` and the V6 route binding.

- `scripts/train.py --mechanism-set cc_scbc_class_conditioned_set_credit
  --route-variant v6_cc_scbc`: exit 0; synthetic segmentation contract pass;
  logits `[16, 11, 16, 16]`; finite loss/gradients; AdamW, cosine warmup,
  gradient accumulation 2; no scientific result.
- `scripts/evaluate.py --mechanism-set cc_scbc_class_conditioned_set_credit`:
  exit 0; synthetic validation contract pass; logits `[1, 11, 16, 16]`;
  finite metric; no checkpoint/data/device access.

Both emitted the same protocol hash
`a084bec329c19e407334a5b381f7e060a6baa1175af3b7aac9a83cd73373eb3b` and run
contract hash
`41af14b3447a6a19649016cc14f491b4ed484730962839b3cafbd54be182fd63`.

Additional direct entry-contract checks all passed: approved remote artifact
paths are accepted; a local Windows path is rejected; `final_test` is rejected
by the non-final evaluation preflight; `CC-SCBC-01` resolves to
`cc_scbc_class_conditioned_set_credit`; and the mismatched successor direction
is rejected.

## Independent adversarial re-check

The ephemeral CPU-only adversarial re-audit contained **39 assertions; 39
passed**. It reconfirmed:

- exact forward identity and inactive evaluation bypass;
- non-diagonal SAR-token Jacobian and live beta gradient;
- class-anchor conditioning witness;
- C2 score-backward and C3 class-agnostic controls;
- detached classifier anchors/bias and exact direct fused gradient;
- finite CPU BF16 autocast-like behavior;
- two-stage train/eval model parity, state-dict parity, and trainability parity;
- V6 config single-mechanism diff and protocol hash equality;
- run-manifest candidate binding and formal successor direction mapping.

## Disposition

**LOCAL_CONTRACT_PASS — re-audited.** The formal-runner/evaluate successor
whitelist and mapping repair are now covered by the current source and tests,
and no blocking local contract failure remains. This re-audit hands control back
to the guarded Experiment workflow for cloud reattachment and the approved C1
seed-0 24-epoch procedure. It does not report or imply any mIoU improvement,
does not open the sealed test, and does not permit C2/C3 before C1 reaches
`50.0075%` mIoU.

## Receipts

- Validator JSON:
  `F:\PRQ4\02_experiment\code\review\independent_v6_cc_scbc_validator_20260829.json`
  (SHA256 `ad3e851a515a0b0258c4d5bcaa77fce566e4763b41b847e158482ce22ce9a91e`).
- Synthetic liveness JSON:
  `F:\PRQ4\02_experiment\reports\v6_cc_scbc_synthetic_jacobian_liveness_20260829.json`
  (file SHA256 `74f6cd418aadc8c3d653441e552f9d4e6ea96835892606eaa20556368be0f778`,
  embedded receipt SHA256
  `90468afeb13e8f4bf9ec2b0ce760c352276248bd2c9811545708f2fff21f9be4`).
