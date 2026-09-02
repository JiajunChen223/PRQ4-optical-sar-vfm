# Independent frozen-snapshot synthetic re-audit

- Date: 2026-08-20 (Asia/Shanghai)
- Scope: `F:\PRQ4\02_experiment\code`
- Verdict: **CONDITIONAL_PASS** for the local synthetic code contract
- Formal experiment status: **not approved and not executed**
- Safety: CPU/in-memory synthetic checks only; no real data, pretrained binary, download, local GPU/CUDA probe, or scientific training.

## Decision

The frozen snapshot clears the previous operation-level blockers. GeoToken-3Path now performs exact-capacity hard one-hot routing, dispatches only selected tokens to the selected operator, preserves bypass tokens bitwise at the fusion boundary, requires explicit fine-scale blocks in formal mode, executes the configured `mid` and `late` stages, and exposes all six approved mechanism/control rows through a one-field configuration delta.

The result is `CONDITIONAL_PASS`, not unconditional `PASS`, because the cloud-only pretrained audit remains pending, no clean synchronization manifest exists, the formal cloud entry point is intentionally locked, and no real CROMA features, dataset samples, RTX 3090 measurements, or scientific results have been examined. The current canonical `CODE_REPORT.json` should therefore remain `BLOCKED` until the local handoff records and required cloud gates are completed.

## Frozen source identity

| File | SHA256 |
|---|---|
| `src/geotoken3path/models/fusion.py` | `A1C2885C4410F28A54C87B008448945580E5EE344F69582D7E91D01DCCA29A71` |
| `src/geotoken3path/models/factory.py` | `A0FE450AFDC87AE0DAF80A420064FC1A5BDA7E4A8CA2B43264DE54504AD88D31` |
| `src/geotoken3path/models/initialization.py` | `967471A368600DDE05376F4A6D2F51B230CC8C113D8B436BFB20D8A362CAB09E` |
| `src/geotoken3path/utils/config.py` | `A8F40B6B3AB5B74D7CEC68EBD25E7CA492DE70BF94F88E29FE7EB304894E82E3` |
| `src/geotoken3path/utils/run_manifest.py` | `B75D6BE89BDB549A518EFBB60E357C49CA41B88254F07EBB98209C06342A2BB0` |

## Required contract results

### 1. Hard one-hot routing and exact branch non-invocation — PASS

For all six approved mechanism sets, every route row summed exactly to one and every entry was exactly zero or one. Forced candidate routing produced:

| Forced state | Selected active tokens | Current operator calls | Escalation operator calls | Bypass identity |
|---|---:|---:|---:|---|
| bypass | 0 | 0 | 0 | bitwise exact for all tokens |
| current | 16 of 32 | 1 | 0 | bitwise exact on 16 bypass tokens |
| escalation | 16 of 32 | 0 | 1 | bitwise exact on 16 bypass tokens |

This verifies actual branch non-invocation, not merely zero multiplication after dense branch evaluation. The straight-through route tensor retained exact one-hot forward values.

### 2. Identity residual — PASS

- `unimodal_optical` returned `torch.equal(output, optical) == True`.
- Mixed static, random, and learned routes returned bitwise-equal outputs on every bypass index.
- Normalization is applied to selected cross-modal deltas before residual scatter, not to the bypass stream.

### 3. Native fine input and fail-closed behavior — PASS

With escalation forced, all malformed cases were rejected:

- missing fine input: `ValueError: finer-scale escalation requires explicit fine_sar tokens`;
- wrong block count `[B,N,3,D]`;
- wrong flattened length `[B,N*3,D]`;
- wrong rank;
- formal model with `allow_synthetic_fine_fallback=False` and no fine input;
- stage mapping missing `fine_sar['late']`.

Valid native input `[B,N,4,D]` reached only selected escalation tokens. The interpolation fallback remains explicitly limited to synthetic smoke configurations.

### 4. Two-stage execution — PASS

A mapping with distinct `mid` and `late` optical, SAR, and fine-SAR tensors invoked each stage exactly once. Returned stage auxiliary keys were exactly `{mid, late}`, both stages reported 16 active tokens over a two-sample 16-token batch, `fine_input_provided=True`, and dense logits had shape `[2,5,20,20]`.

### 5. Six mechanism/control budgets — PASS

At `B=2`, `N=16`, budget `0.5`:

| Mechanism | Active per sample | Active fraction | Aggregate branch counts `[bypass,current,escalation]` |
|---|---:|---:|---|
| `unimodal_optical` | `[0,0]` | 0.0 | `[32,0,0]` |
| `always_fuse` | `[16,16]` | 1.0 | `[0,32,0]` |
| `static_sparse` | `[8,8]` | 0.5 | `[16,16,0]` |
| `random_budget` | `[8,8]` | 0.5 | `[16,8,8]` |
| `local_exchange_without_state_machine` | `[16,16]` | 1.0 | `[0,32,0]` |
| `geotoken_3path` | `[8,8]` | 0.5 | `[16,10,6]` for this fixture |

The two dense controls intentionally use full activation; the three budgeted mechanisms use exact per-sample capacity; the optical-only control uses none. Random-budget routes were repeatable under the declared seed.

### 6. YAML resolution and single structural delta — PASS

The four frozen YAML inputs parsed without fallback. Relative to `always_fuse`, every other approved row differed only at `model.mechanism_set`; every row shared protocol hash:

`96c81d110126b62edae4283b0864415d70bd351bcc472df526dca776da566eba`

An unapproved mechanism is rejected. Baseline and candidate share factory, state-dict key surface, trainable parameter names, dimensions, stages, optimizer-facing parameters, data/runtime policy, and common protocol hash.

### 7. Segmentation step, checkpoint, run manifest, and test seal — PASS

- Synthetic dense logits/loss: finite; optimizer step changed classifier parameters.
- In-memory checkpoint contained model and optimizer state; post-update restored logits were bitwise identical (`max_abs_diff=0.0`). No checkpoint file was written.
- Validation run manifest: `test_accessed=False`, 64-hex run-contract hash.
- Test split rejected for smoke, baseline, screening, strengthening, confirmation, acceptance, and extension execution scales.
- A separate in-memory manifest with both `execution_scale=final_test` and `test_seal_status=final_test` admitted the test split and emitted a 64-hex hash. This exercised the guard only; it did not access test data.
- Actual synthetic entry points passed: train logits `[2,19,16,16]`, finite loss, active fraction `0.5`; validation logits `[1,19,16,16]`, finite metric contract.
- `--execution-scale cloud` was refused with exit code 1 and `RuntimeError: Cloud execution requires the approved remote control card.`

### 8. Complete initialization schema and negative cases — PASS locally

A complete in-memory cloud-audit record passed. Independent negative probes all failed closed for:

- malformed SHA256;
- non-HTTP(S) source URL, unresolved license, or unresolved commit;
- incompatible architecture;
- SAR band-order mismatch;
- invalid normalization;
- invalid GSD or patch shape;
- unrecorded head replacement;
- state-dict shape mismatch;
- invalid position/resolution adaptation state;
- comparison-policy mismatch or target-test leakage;
- minimal `compatibility: {status: pass}` record;
- unexplained loaded state-dict key difference.

The unexplained-key probe was rejected before model mutation. The loader accepts only an already-loaded tensor mapping and performs no local checkpoint/network I/O. The parametrized initialization suite covers all required nested-field omissions and invalid values.

## Whole-suite and policy validation

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
F:\anaconda3\envs\dl_env\python.exe -m pytest -q -p no:cacheprovider
# 88 passed in 2.83s
```

```powershell
F:\anaconda3\envs\dl_env\python.exe C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4
# status=pass; 33 executable/config files; problems=[]; violations=[]
# local_real_data_allowed=false; local_gpu_probe=forbidden_not_run
```

Hygiene scan found zero checkpoint/data binaries and zero executable/config hits for local GPU probing or credential patterns.

## Items that can only be completed on the authorized cloud host

The following are not local-test omissions and must not be fabricated locally:

1. acquire the approved public CROMA checkpoint and verify its bytes, 64-hex SHA256, source URL, license, and source commit;
2. populate the new complete pretrained-audit schema with actual architecture, band order, normalization, GSD/patch, position/resolution adaptation, head replacement, and observed state-dict compatibility;
3. verify pretraining-geography/target-test exclusion using authoritative metadata;
4. acquire and hash the core dataset, confirm license inheritance, split manifest, storage footprint below the 45 GB hard stop, and native mid/late fine-feature provenance;
5. detect the authorized cloud hardware, then validate AMP, micro/effective batch, peak VRAM, throughput, wall time, and any conditional-compute benefit on the RTX 3090 profile;
6. run real baseline reproduction, candidate screening, robustness/generalization evaluation, multi-seed confirmation, and the separately approved final test.

No local result in this report supports accuracy, robustness, generalization, latency, memory, or publication claims.

## Remaining non-cloud handoff conditions

Before code-service `PASS`, the coordinator still needs to:

- replace the stale bootstrap-only `CODE_REPORT.json` with a report referencing this frozen snapshot and full test evidence;
- align the pending `pretrained_weight_audit.json` template with the new fail-closed schema before cloud population;
- produce a clean code-only sync manifest/commit and confirm no data, checkpoint, cache, credential, or local absolute path enters it;
- retain `BASELINE_TRAINING_APPROVAL` and remote-control enforcement before any cloud data/weight acquisition or formal execution.

Subject to those handoff conditions, no unresolved local blocker was found in the requested hard-routing, identity, fine-scale, staged-fusion, matched-control, configuration, segmentation/checkpoint, run-manifest/test-seal, or initializer contracts.
