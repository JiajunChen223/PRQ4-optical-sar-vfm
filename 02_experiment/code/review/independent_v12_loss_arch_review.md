# Independent V12-D0 loss / runner architecture review

**Review scope:** V12-D0 standard-objective stress implementation only.  The
review was read-only with respect to source/configuration: no source, data,
checkpoint, cloud host, or GPU was accessed or changed.  This report is the
only artifact written by this review.

**Decision: BLOCKED**

The objective paths are mostly present and the current tests pass, but the
analytic MacroCE gradient telemetry is mathematically incorrect for every
class with more than one pixel.  Because this telemetry is a required D0
observation and HF-06 is a hard falsifier, V12-D0 must not proceed to a cloud
run until the blocking items below are repaired and retested.

## Evidence executed

- `tests/unit/test_v12_objectives.py`: **4 passed** (`--cache-clear`).
- Full local synthetic suite: **307 passed, 1 warning** (`--cache-clear`).
- No local real-data/weight access and no local GPU probe were performed.
- Static search of `src/`, `scripts/`, `configs/`, and tests found no CMCD
  implementation or candidate registration; CMCD occurs only as the
  conditional V12 route/metadata identifier.
- Independent synthetic check against `autograd.grad` on logits found the
  MacroCE per-class gradient norm reported by
  `src/geotoken3path/losses/segmentation.py:121-128` is exactly `n_c` times
  the correct value (`3.0x` for a class with three valid pixels).

## Findings

### [BLOCKER] HF-06: MacroCE analytic gradient misses the per-class pixel mean

`_analytic_class_gradient_contribution` uses `1 / |C_B|` for the MacroCE
branch, but the derivative of the class-mean CE term is

```text
(softmax - one_hot) / (n_c * |C_B|)
```

The implementation computes the L2 norm of the unnormalised per-pixel vector
and divides only by `|C_B|`.  Its `macro_ce` telemetry therefore depends on
class pixel count in the wrong direction and is not the closed-form
per-class contribution promised by the plan.  The failure is reproducible
without real data and is independent of AMP or model architecture.

**Required repair:** divide the class contribution by `n_c * |C_B|` (or
explicitly implement and document another exact norm/mass definition) and add
a closed-form-vs-autograd unit test for both balanced and imbalanced present
classes.  Keep the telemetry detached and diagnostic-only.

### [BLOCKER] HF-09/HF-10: objective conventions and matched diff are not fully
resolved/config-bound

The V12 YAML files do not carry the pinned objective contract (Lovasz variant,
flat-batch reduction, present-class policy, ignore index, and CE/Lovasz
coefficients).  These semantics and the fixed `1:1` coefficient are instead
hard-coded in `segmentation.py` and `resolve_v12_d0_config`.  The resolved
snapshot exposes only `runtime.objective_name`; it does not emit a structured
objective contract in the run manifest.

Comparing the two resolved rows with the existing helper returns

```text
['runtime.objective_name', 'v12_d0.row_objective']
```

not a declared one-factor `objective.id` delta.  The second field is diagnostic
metadata rather than a training change, but the contract currently does not
state that exception.  This leaves the row semantics and parity audit less
reproducible than the approved plan requires.

**Required repair:** bind a structured objective block in YAML/resolved
configuration and manifest, including `id`, `ignore_index=255`,
`lovasz_variant=multiclass_lovasz_softmax`, `lovasz_reduction=flat_batch`,
`lovasz_classes=present`, and `coefficients={ce:1.0, lovasz:1.0}`.  Make the
parity audit explicitly allow only the objective identifier plus declared
diagnostic metadata, and test that all model/data/optimizer/scheduler/seed/
loader/evaluator fields remain identical.

### [BLOCKER] D0 R0 exclusion can be bypassed at the runner API

The CLI rejects `v12_d0 --objective pixel_ce`, but `run_formal_cloud` accepts
`route_id=R-EO-CMCD-01` with `objective_name=pixel_ce` and will construct an
optimizer and train.  It also permits a non-V12 resolved route to receive a
non-pixel objective override.  The approved intent says the frozen P1 result
is inherited and no new R0 retraining is allowed; that invariant must be
enforced at the protected runner boundary, not only in one CLI branch.

**Required repair:** fail closed in `run_formal_cloud` unless the V12-D0
resolved marker, route, objective contract, and non-pixel row are mutually
consistent; reject pixel-CE R0 retraining and reject objective overrides on
non-V12 routes.  Add direct-runner tests.

### [MAJOR] Required frequency evidence is accumulated once per epoch

`run_formal_cloud` accumulates `train_class_pixel_total`, CE totals, and
gradient totals inside every training epoch (`formal_runner.py:472-505`) but
reports them as `train_pixel_frequency` and train per-class statistics.  A
24-epoch run consequently repeats each training pixel up to 24 times.  The
relative proportions may remain unchanged, but the reported denominator is not
the train dataset denominator required by the plan, and early stopping makes
the multiplier run-dependent.  Validation is accumulated separately in one
pass.

**Required repair:** compute train frequency/CE/telemetry over one declared
dataset pass (or record and normalize the epoch multiplier explicitly), and
emit exact valid-pixel denominators.  Add a two-epoch synthetic aggregation
test proving the reported frequency is not multiplied by horizon.

### [MAJOR] Fractional target labels are silently truncated

`_validate_inputs` calls `target.long()` before checking label integrity.  A
target value such as `0.9` is silently converted to class `0` and accepted.
The approved contract requires valid contiguous labels plus `ignore_index=255`;
fractional, NaN, and other non-integral target values should fail closed.

**Required repair:** validate the original target dtype/values (or require an
integer target dtype) before conversion, then enforce `0 <= target < C` or
`target == 255`.  Add tests for fractional, NaN, negative, out-of-range, and
all-ignore targets.

## Positive checks

- Present-class MacroCE reduction matches the independent per-class reference
  on ordinary tensors and excludes ignored pixels.
- All-ignore batches in the new objective path raise instead of returning a
  finite-looking zero/NaN; negative and out-of-range integer labels fail via
  the CE path.
- Lovasz-Softmax follows the pinned flat, present-class construction and the
  combined objectives use an explicit `ce + lovasz` sum with no hidden weight
  sweep.  Finite/differentiable synthetic tests pass.
- Validation confusion matrix, mIoU, OA, per-class IoU, and class CE
  aggregation continue to use the existing evaluator path; no metric code was
  altered by this review.
- The objective branch does not add an optimizer step, model mechanism, CMCD
  module, or sealed-test access.  `formal_runner._step` carries the selected
  objective into the same training graph and retains the existing optimizer,
  scheduler, AMP, clipping, loader, model, and 24-epoch interfaces.

## Handoff

Do not sync or launch V12-D0 cloud rows on this review alone.  Repair the three
blockers first, add the missing contract/aggregation tests, rerun the complete
synthetic suite and independent review, then regenerate the clean-sync
manifest.  This is an engineering contract decision only; it makes no
scientific statement about MacroCE, Lovasz, or CMCD performance.
