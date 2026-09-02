# V12-D0 loss/runner contract scout

## Scope and decision boundary

This is a read-only engineering audit for `V12-D0-STANDARD-OBJECTIVE-STRESS-TEST`.
No source code, configuration, data, checkpoint, cloud host, GPU, or training
state was changed or opened.  The only file written by this subtask is this
review note.

The authoritative local plan is
`F:\PRQ4\02_experiment\reports\v12_objective_metric_alignment_plan_20260829.json`
and the execution intent is
`F:\PRQ4\02_experiment\reports\v12_d0_standard_objective_stress_intent_20260829.json`.
They authorize validation-only baseline strengthening: R1 MacroCE, R2
CE+Lovasz, and R3 MacroCE+Lovasz only when R1 or R2 reaches +0.5 percentage
points over the frozen P1 reference.  They do not authorize CMCD code,
CMCD training, or sealed-test access.

The current code snapshot is the V11 code tree.  The relevant files are:

- `02_experiment/code/src/geotoken3path/losses/segmentation.py`
- `02_experiment/code/src/geotoken3path/losses/__init__.py`
- `02_experiment/code/src/geotoken3path/engine/formal_runner.py`
- `02_experiment/code/scripts/train.py`
- `02_experiment/code/src/geotoken3path/metrics/segmentation.py`
- `02_experiment/code/src/geotoken3path/data/sen12ts.py`

The current loss surface exposes only `segmentation_cross_entropy`.  The
runner's `_step` calls that loss directly and returns only a detached loss,
logits, and target.  Validation currently records scalar loss, mIoU, and OA;
it does not yet expose per-class IoU, class frequencies, per-class CE, or
class-wise gradient telemetry.  Therefore V12 requires a small objective
registry plus telemetry plumbing, not a CMCD implementation.

## Recommended minimal objective contract

Use one explicit objective identifier in the resolved run snapshot, rather
than branching on a free-form loss name inside the training loop:

```text
pixel_cross_entropy                 # existing P1/R0 semantics
macro_class_cross_entropy           # V12-D0-R1
cross_entropy_plus_lovasz            # V12-D0-R2
macro_class_cross_entropy_plus_lovasz # V12-D0-R3, conditional
```

The loss module should expose a backward-compatible
`segmentation_cross_entropy` and a new pure entry point with a stable return
contract, for example:

```text
loss, stats = compute_segmentation_objective(
    logits, target, objective_id=..., ignore_index=255,
    telemetry_mode="analytic_logit_gradient",
)
```

`stats` must be detached JSON-ready data or tensors that are detached before
serialization.  It must never contain a graph-connected scalar that can
accidentally receive a second optimizer update.  A dataclass or a small
mapping is acceptable; the fields should be fixed by the resolved config.

All reductions must be performed in float32 (or float64 for final CPU
aggregation) even when the forward runner is under CUDA AMP.  The logits can
remain in the autocast dtype for the model forward, but the loss path should
use `logits.float()` before `log_softmax`, `softmax`, sorting, and reduction.
The target must be converted to `long` and validated against the 11-class
contract, with only `ignore_index=255` admitted as an ignored value.

## R1: MacroCE implementation recommendation

For a batch (B), let (V_B=\{i:y_i\ne255\}),
(C_B=\{y_i:i\in V_B\}), and

```text
pixel_ce_i = cross_entropy(logits_i, y_i, reduction="none")
L_c = mean(pixel_ce_i for y_i == c)
L_macro = mean(L_c for c in C_B)
```

This is exactly the plan's batch-present class macro objective.  It must not
be replaced by inverse-frequency pixel weights, a fixed class-weight vector,
focal gamma, a temperature, or a global class prior.  A reference implementation
for contract tests should be deliberately simple and independent of the
production helper:

```python
valid = target != 255
if not bool(valid.any()):
    raise ValueError("MacroCE received an all-ignore batch")
per_pixel = F.cross_entropy(
    logits.float(), target.long(), reduction="none", ignore_index=255
)
present = torch.unique(target[valid], sorted=True)
terms = [per_pixel[target == cls].mean() for cls in present]
loss = torch.stack(terms).mean()
```

The production version can use vectorized `scatter_add`/`bincount`, but the
reference and production outputs must match to a tight float32 tolerance.
The implementation should return, for every model class, `count`,
`pixel_fraction`, and `mean_ce`; absent classes have `count=0` and a null
mean, not an imputed zero.  The batch loss averages only over `count > 0`.

Important edge cases:

1. The padded last validation batch contains repeated pixels whose target is
   255.  They must contribute to neither MacroCE nor any frequency/statistic.
2. An all-ignore batch is a contract error.  Returning NaN or silently
   returning zero would hide a loader/padding bug.
3. A target outside `[0, 10]` and `255` must fail closed before the reduction.
4. `C_B` is batch-local.  Do not average over all 11 classes when a class is
   absent, and do not let the presence of a rare class alter other classes'
   within-class pixel means.
5. `MacroCE` is not the same as a dataset-level macro of pixel CE.  This
   distinction must be recorded in the resolved config and result receipt.
   The batch sampler and micro-batch remain frozen; no class-balanced sampler
   may be introduced to compensate for this choice.

The most useful analytic gradient telemetry is the exact logit-space
derivative of each class term, not a claim about the entire network's
parameter gradient.  For class (c), with (n_c>0),

```text
dL_c/dlogit_i,k = (softmax(logits_i)_k - 1[k=c]) / n_c,
```

for pixels with `y_i=c`, and zero elsewhere.  Record a documented norm/mass,
such as `logit_grad_l1` and `logit_grad_l2`, per class, and optionally the
weighted contribution `1/|C_B|` used in `L_macro`.  If the project requires
parameter-gradient evidence, compute it only in a bounded diagnostic lane
with `torch.autograd.grad` on one declared shared parameter and
`retain_graph=True`; do not run eight extra backwards in the formal training
step.  The plan explicitly asks for *analytic per-class logit-gradient
contribution*, so the analytic telemetry is the preferred D0 contract.

## R2: CE+Lovasz implementation recommendation

Use a standard multiclass Lovasz-Softmax surrogate as a strong mainstream
control, not as a V12 innovation.  Freeze all conventions in configuration:

```text
lovasz_variant: multiclass_lovasz_softmax
lovasz_reduction: flat_batch
lovasz_classes: present
lovasz_ignore_index: 255
objective_coefficients: CE=1.0, Lovasz=1.0
```

The recommended reduction is:

1. Remove `target == 255` before flattening the batch.
2. Compute `probas = softmax(logits.float(), dim=1)`.
3. For every class present in the valid flattened target, form
   `errors = abs(foreground - probas[:, class])`.
4. Sort errors descending and take the dot product with the standard Lovasz
   Jaccard gradient for the correspondingly permuted foreground vector.
5. Average only over present classes.
6. Return `CE + Lovasz` with fixed 1:1 weighting.

Do not silently switch to per-image Lovasz, all-class averaging, Dice,
weighted CE, focal loss, or a coefficient sweep.  These choices are not
numerically interchangeable and would create an uncontrolled R2.  Ties in
the error sort should use the framework's deterministic stable behavior if
available; the selected behavior must be pinned in the result manifest.

The Lovasz helper needs its own empty-input policy.  An all-ignore batch must
raise the same contract error as MacroCE.  A class absent from a valid batch
is skipped under `classes=present`; its value must not be treated as a zero
loss.  The output must be finite under both ordinary float32 and the formal
AMP context.  The Lovasz path should never inspect validation labels outside
the loss call or alter inference/evaluation logits.

For R3, compose the already-tested functions as
`MacroCE + Lovasz`, still with 1:1 coefficients and still only over valid
pixels/classes.  R3 remains conditional and must not be encoded as a formal
innovation candidate before the R1/R2 threshold is observed.

## Runner integration recommendation

The smallest safe runner change is:

1. Add `objective_id` to the resolved run configuration and pass it from
   `run_formal_cloud` to `_step`.
2. Replace the direct loss call in `_step` with the objective registry.
3. Extend `_step` to return a detached telemetry mapping, while preserving a
   compatibility wrapper for existing tests expecting `(loss, logits,
   target)`.
4. Keep the optimizer, scheduler, gradient accumulation, clipping, AMP,
   model factory, loaders, and early stopping untouched.
5. In validation, compute the selected objective loss for reporting, but
   always compute mIoU/OA from the same frozen evaluator.  Add a separate
   unambiguous `pixel_ce` field so objective-loss values are not compared as
   if they were the same quantity.
6. Accumulate confusion matrix, valid pixel counts, per-class CE sums/counts,
   and analytic logit-gradient telemetry in float64 on the host after each
   batch.  Never accumulate a GPU graph or a padded target count.
7. Add `per_class_iou`, `train_pixel_frequency`, `validation_pixel_frequency`,
   `per_class_mean_ce`, `per_class_logit_gradient`, and
   `frequent_rare_macro_iou` to the result schema.  Preserve `mIoU` and `OA`
   fields for existing consumers.
8. Record `best` and exact `epoch24` views.  The objective used for a run is
   part of the run result, but it is the only allowed objective delta between
   matched R0/R1/R2/R3 rows.

The existing `mean_iou` helper returns only one scalar.  Add a pure
`per_class_iou(confusion)` helper and derive the scalar from the valid union
entries, rather than reconstructing per-class IoU from rounded logs.  Class
names should be bound to the explicit WorldCover raw-code mapping in
`configs/benchmarks/sen12ts_worldcover.yaml`; model IDs are contiguous 0..10.
In particular, the word `Water` must not be matched by a guessed index.  The
current contract maps raw code 80 to model ID 7 (`permanent_water` in the
existing diagnostic vocabulary).  If the plan's “classes 4/5/8” means model
IDs, report those IDs and the raw-code/name mapping side by side.

The frozen V12 protocol must be represented as a one-factor diff.  A suitable
audit object is:

```text
changed_fields: ["objective.id"]
unchanged_fields: [model, initialization, data, split, preprocessing,
                   augmentation, sampler, micro_batch, effective_batch,
                   optimizer, scheduler, seed, evaluator, trainability,
                   epochs, test_seal_status]
```

Do not encode the loss as a mechanism-set change.  `R1/R2/R3` are standard
objective baseline-strengthening rows and are not innovation candidates.
Do not use a newly trained R0: the frozen P1 reference remains the existing
48.007495-percent validation result, per the V12 plan.

## Data/metric audit boundaries

The active data contract has 11 WorldCover labels, ignore index 255, 840 train
parents, 180 validation parents, and 180 sealed-test parents.  The loader
uses fixed micro-batch 16, drops the final training batch, and pads the final
validation batch with ignore targets.  Frequency and CE statistics must be
computed only over valid mapped labels.  No source test row, test cache, or
sealed object may be opened for this D0.

The reported 96.12%/3.88% dominant/low-frequency split is a motivating
hypothesis recorded in the approved plan, not a permission to trust stale
counts.  R1/R2 must recompute train and validation frequencies from the
approved non-test rows and record the exact denominators.  If the observed
frequency split differs, report the observation; do not edit the plan or
retrofit class groups to rescue a result.

## Hard falsifiers required before any D0 cloud run

| ID | Required falsifier | Fail condition / consequence |
|---|---|---|
| HF-01 | Objective registry identity | `pixel_cross_entropy` no longer matches the existing CE output on a hand-crafted tensor; stop and repair parity. |
| HF-02 | MacroCE hand reference | Production MacroCE differs from the independent per-class loop beyond a fixed tolerance; R1 is invalid. |
| HF-03 | Present-class semantics | A batch containing only classes 0 and 7 produces exactly the mean of those two class means; any all-11 or pixel-weighted reduction fails. |
| HF-04 | Ignore/padding semantics | Ignore 255 changes any loss, class count, frequency, IoU denominator, or gradient statistic; reject the implementation. |
| HF-05 | Empty and invalid labels | All-ignore, negative, or non-255 out-of-range targets return NaN/zero or are silently accepted; fail closed. |
| HF-06 | MacroCE analytic gradient | Recorded per-class logit-gradient mass/norm disagrees with the closed-form derivative or contains NaN/Inf; D0 cannot claim gradient telemetry. |
| HF-07 | Lovasz reference parity | A known small multiclass tensor disagrees with the pinned standard Lovasz-Softmax reference, or absent classes are imputed as zero. |
| HF-08 | AMP numerical stability | R1/R2/R3 loss or gradients become non-finite under the formal autocast path; no hardware run proceeds. |
| HF-09 | 1:1 composition | R2/R3 contains any hidden coefficient, class weight, focal factor, temperature, sampler change, or reduction change; row is protocol-invalid. |
| HF-10 | Objective-only matched diff | Any resolved diff outside `objective.id` (including model, CROMA, init, trainability, optimizer, scheduler, seed, loader, augmentation, or evaluator) invalidates the row. |
| HF-11 | Optimizer/update parity | Objective telemetry triggers an extra optimizer step, changes gradient accumulation/clipping, or retains a graph into aggregation; invalidate the run. |
| HF-12 | Metric parity | Confusion matrix, mIoU, OA, or per-class IoU differs from the frozen evaluator on the same logits/targets; stop before training. |
| HF-13 | Frequency denominator | Train/validation frequencies include padding, ignored labels, test rows, or guessed class names; invalidate all frequency claims. |
| HF-14 | R3 condition gate | R3 is launched when neither R1 nor R2 has reached +0.5 pp over frozen P1; do not accept its result. |
| HF-15 | Baseline/innovation boundary | CMCD code, candidate mechanism registration, or CMCD training appears before the D0 decision/new plan approval; stop and preserve the gate. |
| HF-16 | Test seal | Any D0 command reads a sealed-test manifest row, payload, label, or cache; classify as protocol failure and do not use its metrics. |
| HF-17 | Local-data/GPU policy | Local real data/weights or a local GPU probe is used to validate the loss; use synthetic fixtures only and reject scientific interpretation. |
| HF-18 | Result completeness | A valid run omits per-class IoU, train/validation frequencies, per-class CE, logit-gradient telemetry, best, or epoch-24 records; it is incomplete even if mIoU exists. |

## Suggested synthetic contract matrix

Before code-only synchronization, run a synthetic-only matrix with fixed seed:

- CE parity against the current function;
- MacroCE with all 11 classes, with one rare class, and with a class absent;
- ignore-only padded rows and all-ignore rejection;
- closed-form MacroCE logit-gradient check against `autograd.grad` on logits;
- Lovasz known-vector value and finite-gradient check;
- CE+Lovasz and MacroCE+Lovasz finite checks under CPU float32 and a mocked
  autocast-compatible path;
- metric/frequency aggregation against a hand-computed confusion matrix;
- resolved-config one-factor diff and run-manifest hash stability;
- no test-seal access and no local real-data/GPU path.

The synthetic matrix is a code contract only.  A pass does not support an
objective or innovation claim.  After it passes, the experiment owner still
has to perform the guarded cloud preparation and then the authorized R1/R2
validation runs, with R3 conditional exactly as specified in the plan.

## Bottom-line recommendation

The current loss/runner is not V12-D0-ready because it has only plain CE and
does not carry the required class-wise evidence.  The minimal safe change is
an explicit, float32-stable objective registry, MacroCE with batch-present
class means, a pinned flat-batch present-class Lovasz-Softmax control, analytic
logit-gradient telemetry, and an objective-only matched-diff/result schema.
The main scientific risk is not GPU memory; it is silently changing the
meaning of MacroCE/Lovasz or confusing logit-space diagnostics with network
parameter gradients.  Lock those conventions before any cloud execution.

**Read-only scout status:** implementation guidance complete; source changes,
cloud preparation, data access, GPU use, and D0 training were not performed by
this subtask.
