# Final hardened independent local test audit

- Date: 2026-08-20 (Asia/Shanghai)
- Frozen scope: `F:\PRQ4\02_experiment\code`
- **Local synthetic-contract verdict: PASS**
- **Formal/cloud experiment status: BLOCKED by approval and cloud-only evidence gates**
- Boundary: CPU/in-memory synthetic validation only. No data or checkpoint download, no local GPU/CUDA probe, and no real or formal training.

## Final decision

The latest frozen snapshot closes all five local adversarial findings from the previous audit. The full suite reports **109 passed in 3.88 seconds**; the ResearchPilot code validator reports **PASS over 39 executable/config files with zero problems and zero violations**. Independent probes reproduced the fixes rather than relying only on the new unit tests.

Accordingly, the code now earns `PASS` for the local synthetic contract. This does not authorize baseline training and is not a scientific result. Formal execution remains blocked until the required ResearchPilot approval and cloud-only CROMA, weight, data, hardware, and protocol evidence are complete.

## Frozen hardened-source identity

| File | SHA256 |
|---|---|
| `src/geotoken3path/data/contracts.py` | `D0113674F481C3C68F6B0DA51B7082B3485D72F0C8550179253925C506788581` |
| `src/geotoken3path/utils/config.py` | `06E08A769A31A6BC52329C2F8CDB937AC9549E0A456E00C2749C45939C970137` |
| `src/geotoken3path/utils/run_manifest.py` | `48BC4722C988FED268081D187F7AF2128B5CBAF30CB5BBBA3F2BE6442CCC8C9C` |
| `scripts/train.py` | `25765EB98F24AE236EBC5050D9FD3F0DC56B67D6BCA49DFDA1DBE33C415CCF9B` |
| `tests/unit/test_dataset_manifest.py` | `699305802801DB45A1B9302BBFDE42022DF64D91E358A8D7EAA0EB78826F3963` |
| `tests/unit/test_resolved_config.py` | `6552C139BB77CF4BD4E738CBE537699E3FC9D42C7F4F3FAD5C40EBDE5357251B` |

## Re-audit of the five former findings

### 1. Run-manifest immutability and hash verification — PASS

- `build_run_manifest` now validates the resolved contract and JSON-clones it before extracting nested fields.
- Mutating the source resolved learning rate from `0.0001` to `9.9` after manifest creation did not change the returned manifest.
- The untouched manifest passed `verify_run_manifest`.
- Mutating the returned manifest learning rate caused deterministic `run manifest hash mismatch` rejection.
- JSON serialization uses `allow_nan=False`, preventing non-finite values from entering a hashed snapshot.

### 2. Empty/partial resolved configurations — PASS, fail-closed

All independently repeated malformed inputs were rejected:

- `{}`;
- `{"model": {}}`;
- `{"model": {"mechanism_set": "always_fuse"}, "runtime": {}}`.

The builder now requires nonempty model/runtime/input/storage/trainability mappings; approved mechanism; positive dimensions and batch fields; exact channel/patch contract; consistent effective batch; valid seed, optimizer, scheduler, clipping and trainability; nonempty identifiers; and a 64-hex common protocol hash.

### 3. Optimizer/scheduler/clip/accumulation configuration validation — PASS, fail-closed

In-memory YAML-loader mutations were independently rejected for all twelve probes:

- unsupported optimizer `sgd`;
- negative or zero learning rate;
- one-element beta list;
- beta equal to one;
- unsupported scheduler `step`;
- warmup fraction equal to one;
- negative gradient clip;
- zero accumulation;
- effective-batch mismatch;
- non-approved precision `fp32`;
- boolean micro-batch.

The frozen six mechanism rows still share identical optimizer, scheduler, clip, effective-batch, trainability, and matched-common protocol fields; only `model.mechanism_set` differs.

### 4. Actual smoke runtime behavior — PASS

The hardened smoke lane now executes the declared semantics rather than merely recording them:

- AdamW with `lr=0.0001`, `weight_decay=0.05`, `betas=[0.9,0.999]`;
- two micro-batches accumulated into effective batch four in the focused fixture;
- loss divided by the accumulation count;
- CPU bfloat16 autocast enabled for the AMP smoke surrogate;
- one finite gradient-norm calculation and clipping call at max norm 1.0;
- one optimizer step;
- explicit cosine-warmup scheduler construction and one post-optimizer scheduler step.

Instrumentation observed `clip_grad_norm_=1`, `AdamW.step=1`, and `LambdaLR.step=2`. The scheduler count includes PyTorch `LambdaLR`'s initialization step plus the explicit post-optimizer step; the returned semantic `scheduler_steps` was 1 (`last_epoch=1`).

The actual frozen 3090-plan smoke entry point also passed for both baseline and candidate:

| Row | Logits | Active fraction | Accumulation | Micro/effective batch | Result |
|---|---|---:|---:|---|---|
| `always_fuse` | `[16,19,16,16]` | 1.0 | 2 | 16 / 32 | finite |
| `geotoken_3path` | `[16,19,16,16]` | 0.5 | 2 | 16 / 32 | finite |

Both emitted the same matched-common protocol hash and distinct mechanism-specific run hashes. These remain synthetic checks, not accuracy or efficiency evidence.

### 5. Dataset exact root, resolved path, normalization, boolean and ledger validation — PASS locally

The complete approved synthetic manifest passed. Every repeated negative case was rejected:

- wrong dataset child, scratch root itself, and traversal root;
- sample outside the exact root, sample containing traversal, or unresolved-realpath flag;
- boolean payload, component or total byte count;
- empty normalization, wrong vector length, zero standard deviation, NaN, or boolean normalization entry;
- ledger sum mismatch and 45.8 GB overflow.

The contract now requires exact configured root `/root/autodl-tmp/copernicus_bench`, at least one normalized resolved sample path strictly below that root, exact optical/SAR band order, finite normalization means, positive standard deviations, integer-only reconciled storage components, and total active footprint below 45 GB.

The local checker cannot itself call cloud `realpath`; it correctly requires `sample_realpaths_resolved=True` and validates the provided canonical paths. Actual symlink resolution and containment must be performed and evidenced on the cloud host before setting this field.

## Full validation evidence

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
F:\anaconda3\envs\dl_env\python.exe -m pytest -q -p no:cacheprovider
# 109 passed in 3.88s
```

```powershell
F:\anaconda3\envs\dl_env\python.exe C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4
# status=pass
# scanned_executable_or_config_files=39
# problems=[]; violations=[]
# local_real_data_allowed=false
# local_gpu_probe=forbidden_not_run
```

Hygiene scan found zero checkpoint/data binaries and zero executable/config matches for local GPU probing or credential patterns.

The clean-sync manifest also passed an independent content audit: declared count 42 equals listed count 42; no file is missing; every listed byte count and SHA256 matches; all data/weight/credential/cache flags are false; local GPU probing is marked forbidden and not run.

## Remaining local boundary

No unresolved source or test blocker remains in the requested hardened local synthetic-contract scope.

One local handoff artifact is stale: `review/CODE_REPORT.json` still describes the old bootstrap-only state and references `not_created_bootstrap_only`, even though a valid 42-file clean-sync manifest now exists. The coordinator must update the canonical CODE_REPORT to reference the frozen hashes, 109-test result, validator result, independent audits, and clean-sync manifest. This is a handoff-record correction, not a source-code failure.

The pretrained-audit template is structurally aligned with the hardened schema but intentionally remains `status=pending`; its empty/pending factual fields must not be filled locally.

## Cloud-only and approval blockers

The following cannot be cleared by this local PASS:

1. explicit `BASELINE_TRAINING_APPROVAL` and experiment-guard validation;
2. cloud synchronization verification against the clean-sync manifest;
3. official CROMA source/license/commit and checkpoint download on authorized storage;
4. actual checkpoint SHA256, architecture/input/head/state-dict/positional compatibility, normalization and geography-overlap audit;
5. real dataset acquisition, file hashes, license inheritance, split manifest, true `realpath`/symlink containment, and measured active storage ledger below 45 GB;
6. authorized RTX 3090 hardware/data-pipeline preflight, including AMP implementation, actual micro/effective batch, VRAM, throughput and wall time;
7. real baseline reproduction, structural screening, multi-seed confirmation, robustness/generalization evaluation, and separately approved final-test access.

No statement about accuracy, robustness, generalization, latency, memory, or publication merit is supported by this report. The local `PASS` means only that the frozen synthetic code contract is ready for the next governed handoff.
