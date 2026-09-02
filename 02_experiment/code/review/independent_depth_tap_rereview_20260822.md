# Independent depth-tapped CROMA bridge re-review — 2026-08-22

## Verdict

`CONDITIONAL_PASS_FOR_LOCAL_CODE_RELEASE; BLOCKED_FOR_CLOUD_ENVIRONMENT_AND_FORMAL_EXECUTION`.

The DTR-02, DTR-03, and DTR-04 local code findings are closed in the reviewed
snapshot.  The remaining conditional boundary is DTR-01: local synthetic
evidence cannot establish compatibility with the unpinned official CROMA
source, its actual named-module paths, or its missing preprocessing contract.
That boundary continues to block cloud-environment advancement and any formal
initialization/training; it does not block the next local code-review/sync
handoff.

## Scope

Read-only local re-review of `bridge`, `fusion`, `factory`, approved config,
and associated synthetic tests.  No SSH, download, GPU probe, real data,
training, evaluation, metric generation, sealed-test access, source mutation,
or gate mutation occurred.

## Latest reviewed hashes

| Item | SHA256 |
| --- | --- |
| `src/geotoken3path/models/croma_bridge.py` | `2fc950d08cfa7efdcf80744e998144f2bcb041197b23886e526b381a8786422e` |
| `src/geotoken3path/models/fusion.py` | `c40f49c953aa63860d2e9984cb4c4520f83f3f6cafecd730204a00f9a51013ca` |
| `src/geotoken3path/models/factory.py` | `2ec5a4531ab25ee70c6e6fd841741a5a23b399e2a3bdf01464f8b43dd5773e86` |
| `src/geotoken3path/utils/config.py` | `22a96b169907aa3f7e6c43141e9759dfb9c24da4c5e3bb2a12e1bfef8268a678` |
| `configs/model/geotoken3path.yaml` | `d0fac802b75a78e0b01dcd439af57931e1878918902a2a18fa7bc09d3495fe9f` |
| `tests/integration/test_croma_bridge.py` | `c50d28f32a6a9bdf5f29ade433bcb850c3ef9d101e3ebd20ae044161d5a5fa7b` |
| `tests/unit/test_model_factory.py` | `7eedd8fd8e764fe5430a816eadb71fa99bf101662d62467e931ab64a9369cefc` |
| `tests/unit/test_hard_routing_contract.py` | `5d4c7a259668851796dbc6cfd06f0ea9c7f6303fd04f50e127647b7bae2c918d` |
| `tests/integration/test_training_object_contract.py` | `2926ac355167bd636c91ca46c023c1525b74cdfade164db5a2a1ecfa0a11dc81` |

## Finding disposition

| Prior ID | Result | Re-review evidence |
| --- | --- | --- |
| DTR-02: legacy fine-feature terminology and fallback | CLOSED | `fusion.py` uses `depth_group` consistently at the public operator/model boundary; no `fine_sar`, `finer-scale`, interpolation, or legacy fallback token remains in executable source/config/tests. The synthetic-only fallback is now explicitly named `_synthetic_depth_group` and repeats SAR token-depth rows rather than fabricating spatial resolution. `build_vfm_segmentation_model` forcibly sets `allow_synthetic_depth_group_fallback=False`; the new integration test begins with that flag true and verifies it becomes false in the formal VFM lane. |
| DTR-03: formal VFM baseline/candidate parity | CLOSED | `test_baseline_candidate_vfm_parity_shares_taps_and_trainability` constructs both formal VFM rows, verifies identical state-dict key surface, full `requires_grad` mapping, depth-tap topology, depth-group topology, and frozen CROMA wrapper parameters. Existing resolved-config/training-object tests retain the single `model.mechanism_set` delta and matched protocol hash. |
| DTR-04: forward-hook lifecycle | CLOSED | The adapter, bridge, and raw-image segmentation model each expose idempotent `close()` forwarding; adapter and model add context-manager teardown. The integration test calls `model.close()` twice without error. |
| DTR-01: official source/tap/preprocessing compatibility | OPEN, cloud blocker | The cloud audit still records `main_unpinned`, no declared input normalization, and no verified real-CROMA execution for the new configured module paths. The local adapter correctly fails closed for missing/nonexecuted/wrong-shape taps, but only a new bounded cloud environment control can verify this against the exact official source/checkpoint. |

## Structural conclusion

The method path now carries a non-spatial SAR depth group throughout:

```text
CROMA SAR transformer depth taps -> sar_depth_group [B,N,4,D]
    -> depth-group escalation operator -> residual routed output
```

No local bridge operation turns official final tokens into a spatially finer
feature. The smoke-only token-model fallback is explicitly isolated, while the
formal raw-image VFM path is fail-closed and always requires the adapter's
actual depth-group output.

## Executed checks

Targeted re-review suite:

```text
tests/integration/test_croma_bridge.py
tests/unit/test_model_factory.py
tests/unit/test_hard_routing_contract.py
tests/unit/test_resolved_config.py
tests/integration/test_training_object_contract.py
```

Result: `32 passed in 4.20s`.

Full synthetic suite: `114 passed in 4.76s`.

`validate_code_project.py --project-root F:\PRQ4` result: `pass`; 39
executable/config files scanned, zero violations,
`local_real_data_allowed=false`, `local_gpu_probe=forbidden_not_run`.

## Required next boundary

Prepare the new reviewed clean manifest/package and use the guarded code-only
sync path. Thereafter, a fresh cloud-only environment control must pin/hash the
official CROMA source, verify the configured depth taps and `[B,225,768]`
outputs on synthetic tensors, and resolve preprocessing/band-order evidence.
No data acquisition or training follows from this local re-review.
