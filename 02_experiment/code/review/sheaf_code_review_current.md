# CSLA local code review — current snapshot

## Verdict

`PASS_FOR_LOCAL_CODE_REVIEW_CONDITIONAL_ON_CLOUD_SYNTHETIC_PREFLIGHT`

The approved route is `R-EO-SHEAF-01 / CSLA-01`. The implementation changes
one internal detector mechanism and keeps the baseline factory, CROMA loader,
data interface, optimizer and test seal unchanged. No real data, checkpoint,
GPU or cloud command was used in this review.

## Closed findings

- Fixed 15x15 incidence has exactly 420 horizontal/vertical edges.
- Zero-start readout gives exact baseline-equivalent optical/joint outputs.
- Sheaf residual, edge flow and node signal shapes are explicit and checked.
- Zero-flow sheaf energy uses a finite epsilon; NaN gradients are rejected by
  the test contract.
- Edge-scale, optical-fiber and SAR-fiber gradients are finite in synthetic
  backward tests.
- Wrong token-grid size fails closed.
- The same `build_vfm_segmentation_model` entry point exposes CSLA to the
  decoder-visible depth-tap path.
- HODGE and BWG remain registered as plan candidates only; they are not
  silently claimed as supported. Their synthetic mode branches are now
  distinct: HODGE removes the grid harmonic component; BWG applies a bounded
  diagonal-covariance Bures gate.

## Residual cloud-only checks

- Official CROMA forward/backward compatibility and activation-memory audit
  must be rerun after code synchronization.
- Real checkpoint loading, trainability mask and cloud package parity remain
  pending the ordinary guarded sync path.
- No scientific claim is made from synthetic tests.

## Evidence

- Full tests: 176 passed.
- Validator: 58 executable/config files, 0 violations.
- Local real data: forbidden and absent.
- Local GPU probe: forbidden and not run.
- Test seal: sealed.
