# Independent CROMA Source-Only Random-Initialization Audit

## Scope and evidence boundary

- Project root: `F:\PRQ4`
- Active Experiment gate observed read-only: `BASELINE_REPRODUCTION / PENDING`
- Test seal: `sealed`
- Audit scope: existing project metadata, journaled cloud output, pinned official CROMA source identity, and current local constructor/bridge contract.
- This audit performed no SSH/cloud command, checkpoint read, data-pixel read, GPU access, training, evaluation, metric generation, gate transition, or code/configuration change.
- This report is an engineering diagnosis, not scientific evidence and not a baseline result.

## Direct evidence

### Pinned official source identity

- Cloud source path recorded by the prior environment audit: `/root/autodl-tmp/audits/prq4-croma-loader-compat-20260822-r1/use_croma.py`
- Frozen source SHA256: `a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b`
- Existing journaled cloud stdout: `F:\PRQ4\02_experiment\cloud\commands\outputs\CROMA-ENVIRONMENT-R7B-20260823T110528166842-37104.out`
- Existing journaled cloud stderr: `F:\PRQ4\02_experiment\cloud\commands\outputs\CROMA-ENVIRONMENT-R7B-20260823T110528166842-37104.err`
- Structured result: `F:\PRQ4\02_experiment\reports\croma_environment_feature_audit_result_20260823_r7b.json`
- The same 14,556-byte source currently served by the official CROMA repository at `https://raw.githubusercontent.com/antofuller/CROMA/main/use_croma.py` independently hashes to the same SHA256. This network cross-check does not replace the project-bound cloud hash evidence.

### Constructor contract

The pinned source defines:

```text
PretrainedCROMA(pretrained_path='CROMA_base.pt', size='base', modality='both', image_resolution=120)
```

It exposes neither `pretrained=False` nor `weights=None`. In `modality='both'`, its constructor contains five unconditional checkpoint-load operations:

1. `torch.load(pretrained_path)['s1_encoder']`
2. `torch.load(pretrained_path)['s1_GAP_FFN']`
3. `torch.load(pretrained_path)['s2_encoder']`
4. `torch.load(pretrained_path)['s2_GAP_FFN']`
5. `torch.load(pretrained_path)['joint_encoder']`

Each result is immediately applied through `load_state_dict`. Therefore the official `PretrainedCROMA` helper is not a valid no-checkpoint random-initialization constructor. Passing a dummy path, `None`, or reinitializing after loading would either fail or violate the leakage-driven no-weight-read policy.

### Source-only architecture primitives

The same pinned source contains the architecture primitives required to reproduce the CROMA-base/both/120 topology without reading a checkpoint:

- `ViT`
- `BaseTransformer`
- `BaseTransformerCrossAttn`
- `Attention`
- `CrossAttention`
- `FFN`
- `get_2dalibi`

The official topology is structurally explicit: 768-dimensional base encoders, 12 optical layers, 6 SAR layers, 6 joint cross-attention layers, 16 heads, patch size 8, 12 optical channels, 2 SAR channels, and modality-specific GAP feed-forward heads. A project-owned wrapper can therefore assemble the same information-flow graph with ordinary PyTorch initialization while omitting all five checkpoint loads. Such a wrapper is an engineering compatibility layer, not a scientific innovation, and must retain MIT attribution.

## Current failure diagnosis

- Failed run stderr: `F:\PRQ4\02_experiment\cloud\commands\outputs\PRQ4-BASELINE-SEED0-RANDOM-INIT-R13-20260823T145602124464-28336.err`
- Failure: `ModuleNotFoundError: No module named 'croma'`, surfaced as `cannot import audited CROMA constructor croma.model:CROMA`.
- The command exited after approximately 2.281 seconds according to `02_experiment/cloud/commands/command_log.jsonl`.
- This is a constructor/import contract defect. It is not a training result, GPU result, baseline metric, or scientific failure.

## Decision

1. `official_random_constructor_available = false` for `PretrainedCROMA`.
2. `source_only_same_topology_wrapper_feasible = true`, conditional on local review, synthetic contract tests, license attribution, clean code packaging, and guarded code synchronization.
3. The leakage-blocked checkpoint must remain unread for the formal route.
4. Formal training must not restart until the constructor is proven to instantiate from the pinned source topology with `checkpoint_loaded=false` and the bridge/depth-tap contract passes.

## High-risk scientific issue: frozen random trunk

`F:\PRQ4\02_experiment\code\src\geotoken3path\models\factory.py` calls `freeze_backbone_for_peft`, which freezes all parameters under `model.bridge.backbone`. Under the approved random-initialization exception, this would freeze a randomly initialized CROMA trunk and train only the downstream routing/adaptation/decoder path.

That condition is mechanically fair only if baseline and candidates share it, but it is scientifically high-risk:

- it no longer represents ordinary VFM adaptation from a learned foundation representation;
- fixed random optical/SAR tokens may make the acceptable same-type baseline unattainable;
- any later gain could be confounded by learning around random features rather than adapting a VFM;
- reviewers may reject the VFM/PEFT narrative even if training-object parity is preserved.

Before formal baseline relaunch, the project must explicitly audit whether the frozen-trunk policy remains scientifically defensible under random initialization. Unfreezing the trunk is not a cosmetic repair: it changes trainability, memory, optimization, and the locked protocol, so it requires an ordinary protocol/code repair and matched baseline/candidate policy rather than a silent implementation change.

## Minimal guarded remote source audit recommended

If a cloud-side confirmation is still needed, use the canonical executor only:

```text
classification=evaluation
guard_action=baseline_metric_audit
```

The command should read only the pinned `use_croma.py`, verify the frozen SHA256, parse its AST, and emit one JSON object containing the constructor signature, five `torch.load` sites, five `load_state_dict` sites, explicit weight-disable parameters (empty), required primitive presence, and safety flags. It must not receive a checkpoint path, import/instantiate `PretrainedCROMA`, read data, use CUDA, write the remote code tree, train, evaluate, or generate metrics. The journaled stdout is diagnostic evidence only.

## Final recommendation

Proceed with a reviewed, source-hash-pinned random-topology wrapper only after the read-only AST conclusion is bound to the code report. In parallel, treat the frozen-random-trunk policy as the next scientific blocker. Do not relaunch the formal baseline merely because constructor import succeeds.
