# D0-C matched decomposition code review

## Scope

This review covers the C0–C6 decomposition implementation added after the
D0-A/B diagnosis. It checks that each row remains inside the same detector,
shares the same state-dict and resolved common protocol, and changes only the
declared dense/conflict/null/private information path. It does not treat any
future row metric as proof of a paper claim.

## Row contract

- C0 reuses the frozen `always_fuse` baseline reference.
- C1 enables dense cross-attention only.
- C2 enables dense cross-attention plus conflict modulation.
- C3 enables dense cross-attention plus a learned null sink.
- C4 enables dense cross-attention plus a SAR-derived private residual.
- C5 is the current full CEAK path.
- C6 is C5 with a deterministic token-axis conflict-field shuffle.

All six enabled rows use the same `build_model`/`run_formal_cloud` entry point,
same initialization, decoder, optimizer, scheduler, split, seed 0, 24 epochs,
and sealed test. New rows are controls for causal attribution, not new bank
candidates.

## Evidence

- `tests/unit/test_ceak_decomposition.py`: 8 tests pass, covering registration,
  common protocol hash, run-manifest acceptance, state-surface parity,
  zero-start identity, branch flags, finite output, and live gradients.
- Full local synthetic suite: 243 passed.
- `compileall` and ResearchPilot code validator: pass; zero violations and
  `local_gpu_probe=forbidden_not_run`.
- The implementation keeps the baseline current-scale dispatch as the common
  prefix and applies one zero-start residual at the declared component path.
  C6 scrambles only the conflict field after it is computed; it does not alter
  the data, target, or optical/SAR encoder inputs.

## Findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| D0C-01 | note | The decomposition variants are now registered in route controls and the run-manifest approved mechanism set; formal command choices include all five new names. | fixed/pass |
| D0C-02 | note | `ceak_scale` remains the shared zero-start residual parameter, so the enabled rows begin as exact baseline identities and retain the common trainability surface. | fixed/pass |
| D0C-03 | note | C1/C3/C4 do not compute conflict evidence; C2/C5/C6 do. C3 alone activates the null sink, and C4 alone activates the private residual among reduced rows. | fixed/pass |
| D0C-04 | note | C6 uses a deterministic token-axis roll as the spatial conflict control; the cloud receipt must record its fixed row ID and all matched protocol fields. | accepted risk |

## Decision

`CONDITIONAL_PASS_FOR_D0C_CODE_ONLY`.

The local decomposition code is ready for a guarded code-only synchronization
and matched 24-epoch cloud rows. The rows remain conditional scientific
evidence; no candidate promotion, composition, confirmation, or sealed-test
access is authorized until the C0–C6 results are reviewed.
