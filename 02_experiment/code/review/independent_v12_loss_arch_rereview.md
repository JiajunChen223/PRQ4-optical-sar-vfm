# Independent V12-D0 loss / runner architecture rereview

**Scope:** repaired V12-D0 objective/loss implementation and its formal-runner
integration.  This was a read-only engineering review: no source,
configuration, real data, checkpoint, cloud host, GPU, or sealed-test object
was accessed or changed.

**Decision: PASS (loss/runner scope)**

The five previously blocking loss/runner findings are repaired and the
current synthetic test suite is green.  This PASS is only a code-contract
receipt.  It is not a scientific result, does not authorize CMCD, and does
not itself advance the cloud/protocol gate.

## Evidence

- Targeted V12/formal-runner tests: **16 passed** with `--cache-clear`.
- Full local synthetic suite: **310 passed, 1 existing warning** with
  `--cache-clear`.
- Independent closed-form check: MacroCE per-class logit-gradient norms now
  match the analytic derivative and `autograd.grad` for imbalanced present
  classes (absolute difference 0 in the checked tensors).
- Integer-label enforcement rejects floating targets before conversion.
- Direct `run_formal_cloud` invocation rejects V12-D0 `pixel_ce` R0
  retraining.
- Resolved V12 rows carry an objective block and the run manifest carries the
  same objective/policy; the matched common protocol hash remains identical
  across MacroCE and CE+Lovasz rows.
- Static search found no CMCD implementation or candidate module in
  `src/`, `scripts/`, or tests; `CMCD-01` remains route metadata only.
- No local real-data/weight access and no local GPU probe were performed.

## Repaired contract checks

### MacroCE and gradient telemetry

`src/geotoken3path/losses/segmentation.py:122-138` now applies the class
pixel-count denominator and present-class denominator.  For a class `c`, the
reported L2 contribution is consistent with the derivative of the batch
MacroCE term:

```text
||softmax - one_hot||_2 / (n_c * |C_present|)
```

Ignored pixels are excluded and all-ignore batches fail closed.  The
telemetry is detached and remains diagnostic-only; it does not create a
second backward or optimizer step.

### Lovasz and 1:1 composition

The present-class, flat-batch Lovasz-Softmax implementation remains finite and
differentiable on the synthetic matrix.  The combined objectives use the
explicit `CE + Lovasz` sum with fixed weights `1.0/1.0`.  No class-weight,
focal, temperature, sampler, or optimizer change was introduced.

### Objective binding and parity

`configs/experiment/v12_objective_route.yaml` now declares the allowed
objective IDs, `ignore_index=255`, present-class MacroCE policy, fixed CE and
Lovasz weights, and the conditional R3 trigger.  The resolved configuration
and run manifest expose the selected `objective.id`, baseline-only status,
and objective policy.  `run_formal_cloud` checks that an explicit objective
argument agrees with the resolved V12 objective and enforces the fixed
24-epoch horizon.

### R0 and aggregation safeguards

The V12 formal API refuses to retrain the inherited `pixel_ce` R0 row.  Train
class-frequency reporting now averages counts over completed epochs, while
CE and gradient diagnostics retain their declared batch/pixel weighting.  The
new adversarial tests cover the R0 rejection and frequency aggregation path.

### Label integrity

The new objective path accepts only integer label dtypes and rejects floating
targets rather than silently truncating them.  Negative and out-of-range
integer labels fail through the cross-entropy path; ignore index `255` is
excluded from loss/statistics.

## Non-blocking observations

1. The general-purpose `run_formal_cloud` API still accepts an explicitly
   supplied non-pixel objective for non-V12 routes.  The V12 CLI never sends
   such an override, and this rereview found no V12 protocol violation.  If
   the project later requires a global “objective override only on V12” rule,
   add that restriction and a direct API test before reusing the runner for
   historical routes.
2. `single_mechanism_diff` reports derived objective metadata leaves in
   addition to `objective.id` (for example `objective.lovasz_enabled` and
   `v12_d0.row_objective`).  These are deterministic consequences/receipts of
   the selected objective, not independent training factors; the common
   protocol hash is unchanged.  A future parity report should explicitly
   classify them as allowed objective metadata.
3. The existing test warning is a test-side scalar conversion in the TASR
   integration test; it is unrelated to V12 loss/runner behavior.

## Handoff

The loss/runner implementation is ready for the next ResearchPilot code
release review and guarded code-only synchronization.  Only after the normal
release manifest/CODE_REPORT is refreshed and the experiment owner completes
the protected cloud preparation may V12-D0 R1/R2 validation-only rows run.
R3 remains conditional on the approved `+0.5 pp` trigger, CMCD remains
unauthorized before the D0 decision, and sealed-test access remains closed.
