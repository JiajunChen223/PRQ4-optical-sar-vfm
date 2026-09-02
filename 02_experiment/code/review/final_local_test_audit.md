# Final incremental local test audit

- Date: 2026-08-20 (Asia/Shanghai)
- Frozen scope: `F:\PRQ4\02_experiment\code`
- Local synthetic-contract verdict: **CONDITIONAL_PASS**
- Code-service/formal experiment verdict: **BLOCKED pending listed local hardening and cloud gates**
- Audit boundary: CPU and in-memory synthetic checks only. No real data, pretrained weight, download, local GPU/CUDA inspection, or scientific training was performed.

## Executive decision

The frozen happy path is internally coherent: the dependency-injected CROMA bridge enforces the two-stage optical/SAR/fine-SAR interface; approved dataset metadata and a reconciled 34.8 GB ledger pass; baseline, candidate, and four controls resolve to the same optimizer/scheduler/clip/trainability protocol; train/evaluate entry points now emit run-contract hashes; and the complete 92-test suite passes.

The local result is nevertheless `CONDITIONAL_PASS`, not `PASS`, because adversarial probes found reproducibility and fail-closed gaps outside the happy path:

1. `build_run_manifest` aliases nested resolved-config objects. Mutating the resolved optimizer after manifest creation changed the manifest learning rate while leaving its stored hash unchanged.
2. `build_run_manifest({"model": {}}...)` succeeds and emits a hash over mostly null fields instead of rejecting an incomplete resolved configuration.
3. The resolver accepts unsupported/invalid optimizer and runtime declarations, including `name=sgd`, negative learning rate, malformed/out-of-range betas, warmup above one, and negative gradient clipping.
4. The smoke training entry point records scheduler, clipping, gradient accumulation, AMP/effective-batch semantics in the protocol but does not execute scheduler stepping, gradient clipping, or accumulation. This is acceptable only as an explicitly synthetic one-step smoke lane; it cannot be treated as the formal runner.
5. Dataset validation accepts booleans as integer byte counts, accepts empty normalization bodies, and does not bind `cloud_root` to the exact configured dataset directory. Lexical traversal is rejected, but actual symlink/realpath containment remains cloud-only.

These findings do not invalidate the current synthetic outputs, but they prevent a clean code-service `PASS` or formal training authorization.

## Frozen incremental source hashes

| File | SHA256 |
|---|---|
| `src/geotoken3path/models/croma_bridge.py` | `3F70D8B414625E641E992C1A4C8BD456889CC885E5094E4B52EC23649F9FB216` |
| `src/geotoken3path/models/factory.py` | `B670F8E3F63C3D315F9F0B6923BE851DB5DB2F2EEC6A6E9B047C5F6F28C33189` |
| `src/geotoken3path/data/contracts.py` | `C45994900930251332BD17C0DE1B2AAC6A3F207973B56FB42AE6E1DE2DD62D38` |
| `src/geotoken3path/utils/config.py` | `A4AAF9355A2A3360F9A0577B3813CA21003F78D60BEB44B8E0172E8B30970DEF` |
| `src/geotoken3path/utils/run_manifest.py` | `687A8DBBD2BF6A7AD1F545EF3B85A4A6D58D62065F7EBF2B51E72B58B7FB749C` |
| `scripts/train.py` | `50EAA7AB2454980351B2DC79ECA29B83BF77F9D8E820B0506C8D2E9ED078907B` |
| `scripts/evaluate.py` | `C9E74DC3D0C2A76A7B27243F1D5BDCB1869B96FD620939C266E86A911DC13EDE` |
| `configs/runtime/3090_plan.yaml` | `274AD487623986C272471CA311D9F87F3C13CC8084E9F660D79870B6303B78AF` |

## CROMA bridge

### Passing evidence

- A synthetic dependency-injected backbone produced raw-image segmentation logits `[1,4,24,24]` through both `mid` and `late` stages.
- The injected backbone was frozen; the GeoToken router remained trainable and received gradient.
- Baseline and candidate wrapper models exposed identical parameter names and identical trainable-parameter names.
- The bridge rejected every tested malformed case: wrong optical/SAR channel count, mismatched batch or spatial size, non-mapping output, extra output key, missing stage, token-dimension mismatch, and wrong fine-block shape.
- Formal bridge output requires exact keys `{optical,sar,sar_fine}`, exact configured stages, `[B,N,D]` optical/SAR parity, and `[B,N,4,D]` native fine-SAR blocks.

### Cloud-only limitation

The synthetic backbone proves interface behavior only. It cannot prove that the official CROMA implementation exposes the required stage names, feature dimensions, native fine-SAR provenance, normalization, positional treatment, or checkpoint compatibility. Those require the audited cloud dependency and real checkpoint.

## Dataset ledger and path contract

### Passing evidence

The approved fixture passed with:

- exact dataset ID, 12-band Sentinel-2 order, and `VV,VH` SAR order;
- matching pretrained target band order/normalization;
- reconciled components: 20.0 GB raw + 10.0 GB extracted + 2.0 GB cache + 0.8 GB weights + 2.0 GB checkpoints = 34.8 GB;
- verified license marker, sealed test flag, and valid SHA256 fields.

Adversarial validation rejected traversal (`/root/autodl-tmp/../escape`), prefix confusion (`/root/autodl-tmpx/...`), Windows paths, double-slash normalization changes, negative ledger values, component/total mismatch, raw-payload mismatch, and a 45.8 GB overflow.

### Local hardening findings

1. **Boolean byte counts accepted.** Python booleans satisfy `isinstance(value, int)`. A manifest with `storage_bytes=True`, `raw_payload=True`, and total `1` passed. Numeric fields must explicitly reject `bool`.
2. **Empty normalization bodies accepted.** `{"optical": {}, "sar": {}}` passes the dataset validator. The current cross-binding can reject it only if compared against a separately valid pretrained target; the dataset contract itself does not validate vector lengths, finiteness, or positive standard deviations.
3. **Exact root not bound.** Both `/root/autodl-tmp/` and `/root/autodl-tmp/other_dataset` passed. The manifest should be bound to the resolved `cloud_data_root=/root/autodl-tmp/copernicus_bench`, unless a governed relocation field is explicitly introduced.
4. **Symlink containment is unverified.** `PurePosixPath` correctly checks lexical traversal but cannot prove that a cloud child is not a symlink outside approved scratch storage. Resolve-and-containment verification is necessarily cloud-only.

## Run-manifest wiring

### Passing evidence

- Train and evaluation entry points both call `build_run_manifest` and print its 64-hex contract hash.
- The valid manifest carries optimizer, scheduler, gradient clip, effective batch, input/storage/trainability records, data-manifest reference, pretrained-audit reference, code-sync reference, seed, split, execution scale, and test-seal status.
- Frozen manifest optimizer and seed match the resolved runtime.
- Test access was rejected for smoke, baseline, screening, strengthening, confirmation, acceptance, and extension scales.
- Baseline and candidate generated different run hashes because their declared mechanism sets differ, while retaining the same matched-common protocol hash.

### Blocking local findings

1. **Manifest is not immutable.** Nested dictionaries are assigned by reference. After building a manifest at learning rate `0.0001`, changing the source resolved dictionary to `9.9` changed `manifest['optimizer']['learning_rate']` to `9.9`; recomputation no longer matched the stored hash. Deep-copy/canonicalize before hashing and returning.
2. **Incomplete resolved configs do not fail closed.** A minimal `{"model": {}}` generated a manifest with null route/dataset/protocol/optimizer fields and a valid-looking hash. All required identifiers, hashes, refs, runtime fields, and protocol objects must be type/value validated before emission.
3. **References are paths, not verified identities.** The manifest records data/pretrained/code-sync reference strings but no verified artifact hash. The cloud orchestration must bind these references to existing, hash-validated artifacts before a run can start.

## Optimizer configuration parity

### Passing evidence

Across all six approved mechanism rows:

- optimizer dictionaries were equal;
- scheduler dictionaries were equal;
- gradient-clip values were equal;
- matched-common protocol hashes were equal;
- the only resolved leaf difference from `always_fuse` was `model.mechanism_set`.

The frozen runtime resolved to `AdamW(lr=0.0001, weight_decay=0.05, betas=[0.9,0.999])`; a constructed optimizer reported those exact values.

### Fail-closed/runtime gaps

In-memory YAML-loader mutation probes showed the resolver accepted all of the following:

- optimizer name `sgd`, although `train.py` always constructs AdamW;
- learning rate `-1.0`;
- one-element beta list;
- beta value `1.2`;
- warmup fraction `1.5`;
- gradient clip `-1.0`.

Thus a malformed or altered YAML can produce a manifest that disagrees with actual runtime behavior. The resolver must validate supported names, positivity/ranges, beta arity, warmup interval, and clipping range.

The smoke entry point does not instantiate/step the declared cosine-warmup scheduler, call gradient clipping, or implement accumulation. That is acceptable for a clearly labeled one-step tensor smoke test only. Formal parity requires these operations in the approved cloud runner and must be verified against the same resolved config and manifest.

## Whole-suite and entry-point results

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
F:\anaconda3\envs\dl_env\python.exe -m pytest -q -p no:cacheprovider
# 92 passed in 3.01s
```

```powershell
F:\anaconda3\envs\dl_env\python.exe C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4
# PASS; 37 executable/config files scanned; problems=[]; violations=[]
# local_real_data_allowed=false; local_gpu_probe=forbidden_not_run
```

Synthetic entry points passed for `always_fuse` and `geotoken_3path` training and candidate evaluation. They emitted finite losses/metric contracts, correct dense shapes, the common protocol hash, and per-run hashes. These remain non-scientific smoke outputs.

## Required next actions

### Local before clean code-service PASS

1. Deep-copy and strictly validate every required run-manifest field before hashing/returning; add mutation-stability and minimal-config rejection tests.
2. Validate optimizer/scheduler/clip domains and supported names; ensure the formal runner constructs exactly the declared operations.
3. Reject boolean storage values, validate normalization vectors, and bind the dataset root to the resolved approved path.
4. Update the canonical `CODE_REPORT.json`, rerun independent tests, and create a hash-bound clean code-only sync manifest.

### Cloud-only blockers

1. Audit and inject the official CROMA code/checkpoint; verify source, license, commit, SHA256, architecture, stage outputs, bands, normalization, GSD/patch, head and positional adaptation, and target-test/geography exclusion.
2. Resolve the real dataset path, including symlinks, against approved scratch storage; verify download/extraction hashes, split manifest, license, normalization, and the complete active-storage ledger below 45 GB.
3. Verify the formal runner uses the declared AdamW, scheduler, warmup, clipping, accumulation, AMP, seed, effective batch, initializer, and immutable artifact hashes.
4. Perform authorized RTX 3090 preflight and only then run baseline/candidate experiments. VRAM, throughput, wall time, accuracy, robustness, and generalization remain completely untested.

No local evidence supports a scientific or efficiency claim. The current checkpoint must not be interpreted as `BASELINE_TRAINING_APPROVAL`.
