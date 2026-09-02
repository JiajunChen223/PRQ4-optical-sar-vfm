# Independent D3 code review

## Scope

This review covers the D3-A checkpoint-only inference intervention path and
the D3-B training-path control registrations. It does not approve a new
scientific mechanism or open the sealed test.

## Checks

- D3 branch-off uses a zero residual scale without mutating checkpoint
  parameters; the local SAR path remains intact.
- P4/P5 state interventions reuse the existing state-override contract, with
  per-stage finite tensor and batch-shape validation.
- T3 detaches the learned pooling weights in the forward pooling operation;
  T4 uses uniform forward weights with a straight-through learned backward
  path; T5 detaches SAR tokens only for the auxiliary global branch.
- P0/P1/P2/P3/P4/P5 and T0–T5 remain selectable through the same factory and
  entry point; no external model or router is introduced.
- All D3 command paths require the cloud data manifest, audited CROMA,
  checkpoint and code-manifest bindings. The test-seal guard remains active.

## Verification

- Targeted D2/D3 and historical CEAK regression tests pass.
- Full local test count is recorded in the current `CODE_REPORT.json` after the
  final snapshot; local data and GPU probes remain forbidden.
- Code validator reports zero violations after the D3 path is added.

## Finding

`CONDITIONAL_PASS_FOR_D3_A_THEN_D3_B`. The first D3-A attempt exposed a legacy
telemetry compatibility defect: P0/P1/P2 predate D3 and cannot be required to
emit D3 residual fields. The repaired runner requires that telemetry only for
P3/P4/P5 and evaluates P0/P1/P2 metrics without it. D3-A may run again through
the guarded cloud-only path; D3-B remains blocked until the D3-A receipt passes
and its decision is recorded. No candidate promotion, composition, confirmation
or sealed-test access is permitted in this plan.

The subsequent P1 provenance inspection found that the immutable baseline
checkpoint uses `clean_sync_manifest.json`; this historical reference is now
explicitly allow-listed alongside the D1/D2 manifests. No checkpoint, data,
metric or protocol content was changed.
