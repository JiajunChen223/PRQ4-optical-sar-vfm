# Independent D2 set-state code review

## Scope

This review covers the D2 checkpoint-only intervention path added to the
existing G3 query-independent SAR pooling control. It does not approve a new
scientific mechanism, change the dataset/split, or unlock P0–P5 training.

## Checks

- `GeoToken3PathFusion` exposes the G3 pooling weights, value tokens and pooled
  state only through `return_aux`.
- `uniformize` replaces only the learned pooling weights with exactly `1/N`.
- `decouple` permutes weights within a sample and leaves the value tokens
  untouched; the permutation is validated as device-matched int64 `[B,N]`.
- `state_override` replaces only the pooled state and validates finite
  `[B,D]`/`[B,1,D]` input. `state_override_by_stage` must define every stage.
- No D2 intervention is accepted for a non-G3 mechanism through the new path;
  existing mechanisms retain their previous forward semantics.
- The test-seal guard and cloud-only data/weight contracts remain unchanged.

## Verification

- Targeted D2 unit tests: 4 passed.
- Full local test suite before the cloud attempt: 258 passed, 1 warning; after
  adding the P4/P5 D2-C controls and repairing the legacy CEAK dispatch, the
  complete suite passes 260 tests, with the targeted D2 suite at 6/6.
- ResearchPilot code validator after the repair: 93 executable/config files,
  0 violations.
- Local GPU probe: forbidden and not run.
- No local real data, labels, weights or checkpoints were read.

## Finding

`CONDITIONAL_PASS_FOR_D2_DIAGNOSTICS_ONLY`. The repaired intervention path is
suitable for a new guarded cloud-only checkpoint audit. The first cloud attempt
is retained as `failed_code_defect`; its output directory must not be reused.
This is not a candidate approval and does not support a paper claim until the D2
gate evidence is complete.

## Remaining cloud checks

The cloud runner must bind the V3 plan hash, G3 checkpoint SHA, data-manifest
SHA, common-protocol SHA, CROMA audit and code manifest before reading any
artifact; it must report `test_accessed=false`, retain no raw maps locally and
write a new immutable output directory.
