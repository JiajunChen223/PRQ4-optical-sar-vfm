# Independent depth-tapped CROMA bridge review — 2026-08-22

## Verdict

`CONDITIONAL_PASS_FOR_LOCAL_CODE_REVIEW; BLOCKED_FOR_CLOUD_ENVIRONMENT_AND_FORMAL_EXECUTION`.

The current local implementation has removed the **native spatial fine-SAR
assumption at the CROMA bridge boundary**.  It now obtains modality-specific
tokens from named transformer-depth taps and constructs each `[B,N,4,D]`
object by stacking four SAR **depth** outputs.  It does not interpolate the
official CROMA final tokens, and the adapter fails closed if a configured tap
does not run or has the wrong rank/dimension.

This is not yet a compatibility pass for the real CROMA release.  The fixed
module paths, preprocessing contract, and exact upstream source revision have
not been verified against a pinned official implementation.  Those omissions
block `CLOUD_ENVIRONMENT` and formal initialization/training, but they do not
invalidate the local structural revision or its code-only review handoff.

## Review scope and immutable inputs

Read-only review of the local depth-tapped revision.  No SSH, download,
real-data access, GPU probing, training, metric generation, test-split access,
source-code modification, or gate change was performed.

| Item | SHA256 |
| --- | --- |
| `src/geotoken3path/models/croma_bridge.py` | `df123b250ee4844c6c61498cb7f877e27b001c0e664b3c16e1c02d59bd72accb` |
| `src/geotoken3path/models/factory.py` | `117f56326aaa272dcd20398e4737c1bd298a63828a24585e09dbb006103bc8b0` |
| `src/geotoken3path/models/fusion.py` | `9bd20766e053a277fe8771d0131308ee00df46bb8b87e0967dcf9134c8bba193` |
| `src/geotoken3path/utils/config.py` | `22a96b169907aa3f7e6c43141e9759dfb9c24da4c5e3bb2a12e1bfef8268a678` |
| `configs/model/geotoken3path.yaml` | `d0fac802b75a78e0b01dcd439af57931e1878918902a2a18fa7bc09d3495fe9f` |
| `configs/experiment/approved_route.yaml` | `15ad68cea44063d825a3fef7ca058df1da70577327d70603768528b3aa815d0b` |
| `tests/integration/test_croma_bridge.py` | `da670db5d2a9da1b84f50048f17f1b4ad990451808140c489dd8bdb8b70a4b06` |

## Findings

| ID | Severity | Status | Evidence and conclusion |
| --- | --- | --- | --- |
| DTR-01 | blocking for cloud/formal only | OPEN | The official cloud audit reports an unpinned `main` source, undeclared input normalization, and no official-forward/native `[B,N,4,D]` feature. The new adapter intentionally no longer needs the last item, but the YAML module paths (`s1_encoder/s2_encoder.transformer.layers.*.1`) are still synthetic-fixture evidence only. Before formal use, pin and hash the upstream source, verify every named tap executes on that exact source with `[B,225,768]`, and bind preprocessing/band order to the CROMA and dataset contracts. |
| DTR-02 | major claim-hygiene/test gap | OPEN | The raw-image bridge correctly names its output `sar_depth_group` and documents it as non-spatial. However, the downstream public interface still calls it `fine_sar`; `fusion.py` retains messages such as “finer-scale escalation” and the local smoke fallback `_interpolated_fine_blocks`. This does not create a native feature in `CromaGeoTokenSegmentation` because it always supplies the adapter's depth group, but it can misstate provenance and permit a generic synthetic path outside the formal bridge. Rename or isolate the legacy public aliases before manuscript claims, and add a regression test proving the formal VFM factory cannot silently take the interpolation fallback. |
| DTR-03 | medium parity/test gap | OPEN | Resolved baseline/candidate configs differ only in `model.mechanism_set`; token-only factory state keys and trainable parameter names match. Both formal VFM builds call the same `build_vfm_segmentation_model` and `freeze_backbone_for_peft`, so the intended frozen-trunk parity is structurally shared. There is not yet a direct test building both VFM rows from resolved configs, comparing their full `requires_grad` maps, and checking that both receive identical tap topology/backbone initialization. Add that test before a new code-only release. |
| DTR-04 | medium lifecycle risk | OPEN | `CromaDepthTapAdapter` owns persistent forward hooks and offers `close()`, but the formal model does not expose or call a teardown method. This is harmless for one long-lived model instance, yet repeated construction in audits/notebooks can retain hooks until garbage collection. Add an explicit model teardown/context manager or an idempotent bridge close path before repeated cloud audit sessions. |

## Positive structural checks

- `CromaDepthTapAdapter` resolves all configured module paths at construction,
  captures only current-forward tensor outputs, validates `[B,N,D]`, and stacks
  exactly four depth outputs along axis 2.  It returns only `optical`, `sar`,
  and `sar_depth_group`.
- `CromaBackboneBridge` rejects any non-exact output-key set, token-stage
  mismatch, non-float32 raw input, incorrect 12/2 channel counts, and any
  non-`[B,N,4,D]` group.  No final-token interpolation is present in this
  bridge.
- The mechanism/config route now declares
  `croma_depth_tapped_token_groups` and `depth_group_escalation`, not native
  spatial fine feature provenance.  The resolver and run-manifest validator
  require both modality stage taps and four SAR taps for every configured
  stage.
- `build_vfm_segmentation_model` is the common raw-image construction path,
  injects the adapter into the bridge, then freezes all CROMA-backbone
  parameters.  The synthetic integration test confirms a dense output,
  frozen trunk, and gradient reaching the router head.

## Executed local checks

```text
F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest -q \
  tests/integration/test_croma_bridge.py \
  tests/unit/test_model_factory.py \
  tests/unit/test_hard_routing_contract.py \
  tests/unit/test_resolved_config.py \
  tests/integration/test_training_object_contract.py
```

Result: `30 passed in 5.01s`.

```text
validate_code_project.py --project-root F:\PRQ4
```

Result: `pass`; 39 executable/config files scanned; zero violations;
`local_real_data_allowed=false`; `local_gpu_probe=forbidden_not_run`.

## Required successor order

1. Address DTR-02 and DTR-03 locally; then rerun the affected synthetic tests,
   independent review, clean-manifest/package audit, and guarded code sync.
2. Under a fresh cloud-environment control, verify the pinned official CROMA
   source's named module paths and observed tap shapes using synthetic inputs
   only; also complete its preprocessing contract.
3. Only after that cloud evidence passes may `CLOUD_ENVIRONMENT` be reopened.
   Data acquisition, GPU adaptation, baseline training, and sealed-test access
   remain outside this review and unauthorized here.
