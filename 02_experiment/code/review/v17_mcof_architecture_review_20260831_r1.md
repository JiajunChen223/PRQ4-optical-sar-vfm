# V17 MCOF architecture review (read-only)

- Review date: 2026-08-31
- Project root: `F:\PRQ4`
- Route: `R-EO-MCOF-V17-01` / `MCOF-01`
- Scope: architecture, mathematical equivalence, numerical stability, RTX 3090/24 GB resource implications, and test coverage
- Boundary: read-only review; no source modification, real-data access, pretrained-weight access, GPU probe, or training
- Review status: `CONDITIONAL_PASS`
- Blocker count: `0`
- High count: `1`
- Medium count: `3`
- Low count: `0`

## Executive assessment

The MCOF path is reachable and structurally matches the intended operator-field
idea. The raw-image branch in `CromaGeoTokenSegmentation` calls the MCOF decoder
after the common token model, so the implementation is not a dead module. The
correction is an implicit low-rank operator and does not allocate a dense
per-pixel `[D, C]` classifier tensor. At `alpha=0`, the decoder returns the
upsampled coarse classifier output exactly for finite inputs.

The main issue before a formal cloud run is that the implementation describes a
bounded operator but does not bound the learnable scalar or the two low-rank
factors. This is a real numerical-contract gap, even though the current
synthetic tests pass. The full raw-image MCOF training path also lacks a
training-mode integration test; the current smoke lane exercises the token-only
model, while the MCOF-specific bridge test is evaluation-mode only.

## Findings

### HIGH-01 — Operator amplitude is not actually bounded

- Severity: `HIGH`
- Status: `OPEN_BEFORE_FORMAL_TRAINING`
- Files: `02_experiment/code/src/geotoken3path/mechanisms/mcof.py:77-90, 182-194`
- Evidence:
  - `alpha` is declared as an unconstrained scalar parameter with
    `nn.Parameter(torch.zeros(()))`.
  - `semantic_projection` and `class_basis` are ordinary unconstrained linear
    layers.
  - The forward path multiplies the correction by `self.alpha` directly;
    only the controller output is bounded by `tanh`.
  - The contract and route documentation call the field bounded/low-rank, but
    `validate_mcof_contract` has no amplitude bound, spectral bound, or factor
    normalization field.
- Consequence: `condition` is bounded, but the actual correction magnitude
  `alpha * U(a * V^T z)` is unbounded. Under AdamW and AMP, a large alpha or
  factor norm can dominate the frozen classifier, produce overflow/NaN, and
  invalidate the claimed bounded-operator interpretation. Gradient clipping in
  `formal_runner.py` limits an update norm, not the resulting operator norm.
- Recommended fix:
  - Replace the direct scalar with an explicit bounded parameterization such as
    `alpha = alpha_max * tanh(alpha_raw)`, with `alpha_max` frozen in the
    contract and manifest; and
  - either spectral-normalize/bound the two low-rank factors or document that
    only alpha is bounded and weaken the route wording accordingly.
  - Add tests for the bound, finite output under large finite inputs, and
    identity at `alpha_raw=0`.
- Gate implication: do not call this a scientific failure; however, formal
  training should remain closed until the contract is either repaired or the
  bounded claim is explicitly amended and revalidated.

### MEDIUM-01 — Full raw-image MCOF training path is not covered by the smoke lane

- Severity: `MEDIUM`
- Status: `OPEN`
- Files: `02_experiment/code/scripts/train.py:84-175`,
  `02_experiment/code/src/geotoken3path/models/croma_bridge.py:783-816`,
  `02_experiment/code/tests/integration/test_croma_bridge.py:245-292`
- Evidence:
  - `run_synthetic_smoke` constructs `OpticalSarTokenModel` through
    `build_model` and calls it with token tensors; it does not construct
    `CromaGeoTokenSegmentation` or execute the MCOF raw-image branch.
  - The MCOF integration test uses `.eval()` and checks identity/effect after
    manually changing alpha, but does not run a full raw-image training
    forward/backward step.
  - `run_v17_mcof_hard_contract.py` validates the standalone decoder, not the
    CROMA bridge, token fusion, classifier, and decoder as one training graph.
- Consequence: a bridge-only wiring, dtype, or gradient error could survive the
  local smoke gate even though the standalone decoder contract passes.
- Recommended fix: add one synthetic `model.train()` integration test with a
  15x15 synthetic CROMA fixture and 120x120 images that runs CE backward and
  asserts nonzero gradients for `alpha`, `condition_stem`,
  `semantic_projection`, and `class_basis`; also assert baseline/candidate
  trainability parity in training mode. This remains CPU/synthetic-only.

### MEDIUM-02 — AMP intermediate/output finiteness is not fail-closed

- Severity: `MEDIUM`
- Status: `OPEN`
- File: `02_experiment/code/src/geotoken3path/mechanisms/mcof.py:122-129, 144-194`
- Evidence:
  - The module checks finiteness only for its four inputs.
  - There is no explicit finite check for `condition`, `projected`, `gated`,
    `correction`, or `logits` after convolution/linear/interpolation.
  - At zero start, `0 * correction` is safe only when `correction` is finite;
    `0 * inf` can become NaN.
  - The hard contract uses default CPU float tensors and does not exercise
    CUDA float16/bfloat16 autocast or large-magnitude finite inputs.
- Consequence: overflow can enter the training loss before the contract emits a
  useful MCOF-specific error. This is especially relevant because the formal
  runtime is AMP and the amplitude bound is currently absent.
- Recommended fix: retain strict input checks, add finite checks for the
  controller output and final logits (at least under `audit`/debug and in the
  liveness harness), and add a synthetic autocast stress test on the available
  cloud preflight path. Do not silently clamp or replace NaNs.

### MEDIUM-03 — Pixel/token alignment is assumed rather than witnessed

- Severity: `MEDIUM`
- Status: `OPEN`
- Files: `02_experiment/code/src/geotoken3path/mechanisms/mcof.py:111-121, 145-150`,
  `02_experiment/code/src/geotoken3path/models/croma_bridge.py:703-714`
- Evidence:
  - The decoder enforces a 225-token count and equal image shapes, then
    interpolates the controller to `(120, 120)` if needed; it does not enforce
    that input images are exactly the declared 120x120 patch.
  - The bridge validates `[B, N, D]` with `N=225` indirectly through the
    downstream decoder, but carries no explicit spatial coordinate or
    patch-order witness connecting the CROMA token order to the raw image
    lattice.
  - `align_corners=False` is consistently selected, which is a coherent
    convention, but the convention is not tested with an impulse/coordinate
    fixture.
- Consequence: a future tap, crop, or token-order change could preserve all
  tensor shapes while silently misaligning `a(p)` and the semantic coordinate.
  This would undermine the local-operator interpretation without necessarily
  raising an exception.
- Recommended fix: for the formal route, enforce `H=W=output_size=120` (or
  make the accepted geometry explicit in the contract), and add a synthetic
  coordinate/impulse alignment test that verifies the 15x15-to-120x120 mapping
  and token ordering.

## Mathematical equivalence assessment

The implemented correction is mathematically consistent with an implicit
low-rank local classifier field, with one important wording precision:

```text
z       = fused token feature
q       = upsample(V^T z)                  [B, r, H, W]
a(p)    = tanh(h(optical(p), SAR(p)))      [B, r, H, W]
c(p)    = U (a(p) elementwise q(p))       [B, C, H, W]
L(p)    = upsample(W0 z + b0)(p) + alpha*c(p)
```

`semantic_projection` implements `V^T`, `class_basis` implements `U`, and the
same `alpha` is used at every pixel. The code computes the projected semantic
coordinate on the coarse grid before interpolation and never materializes a
`[B, H, W, D, C]` classifier field. Because the controller is spatially
varying, the exact implementation should be described as a correction applied
to the interpolated semantic coordinate; it is not literally an interpolation
of a fully materialized per-pixel classifier matrix.

The zero-start identity is sound for finite tensors: `alpha` is exactly zero,
the coarse logits are interpolated with the same bilinear convention, and the
correction is added after that base. The existing tests confirm this with
`torch.equal`.

## Resource assessment

- No resource blocker was found by static inspection.
- The decoder has 15,913 parameters in the synthetic hard-contract run, well
  below the 500,000-parameter contract.
- Its largest rank-dependent activations are `[B, 16, 120, 120]` for projected,
  conditioned, and gated tensors; this is small relative to a 24 GB card and
  does not create a dense 768-channel 120x120 activation in the MCOF branch.
- The full CROMA forward/tap bridge remains the dominant cost and is common to
  baseline/candidate. A cloud preflight is still required for actual peak VRAM,
  throughput, and AMP behavior; this review did not probe hardware.
- The repeated `torch.isfinite(...).all()` input scans in every decoder forward
  can introduce synchronization/throughput overhead on CUDA. This is not a
  capacity blocker, but should be measured or restricted to audit mode after
  the finite-output contract is added.

## Tests executed in this review

All commands were CPU/synthetic-only and did not access real data or weights.

1. `python -m pytest tests/unit/test_mcof.py tests/unit/test_v17_mcof_config.py tests/integration/test_croma_bridge.py -q --disable-warnings --maxfail=20`
   - Result: `32 passed in 3.97s`
2. `python scripts/run_v17_mcof_hard_contract.py`
   - Result: `11/11` checks passed
   - Reported decoder parameter count: `15,913`
   - `scientific_result=false`, `real_data_read=false`, `weights_read=false`,
     `gpu_used=false`, `test_accessed=false`

These results establish software/synthetic contract behavior only. They are not
scientific performance evidence and do not authorize cloud training or sealed
test access.

## Validated / no-finding items

- No blocker was found in the reviewed MCOF call path.
- MCOF is reachable from `CromaGeoTokenSegmentation` for all six declared MCOF
  mechanism modes.
- Baseline and candidate use the same model factory and state-dict surface in
  the existing integration test.
- The decoder uses one shared coarse classifier output as its base, and the
  initial candidate output is exactly identical to that base for finite inputs.
- Static, sample-level, shuffled, optical-only, and SAR-only modes are exposed
  and are distinct under the existing synthetic hard-contract fixture.
- The reviewed code did not read local real data, local pretrained binaries, or
  query the local GPU.

## Recommended disposition

Keep the route in code-hardening/liveness preparation. Before any formal
24-epoch cloud run, repair or explicitly amend HIGH-01, then add the full
raw-image training integration test and AMP finite-output/alignment witnesses.
No scientific result, promotion, multi-seed confirmation, or sealed-test access
is supported by this review.
