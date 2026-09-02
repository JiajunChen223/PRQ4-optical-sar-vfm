# D0 adversarial and test review

## Scope

This independent-process audit targets failure modes that could make the D0
receipt look valid while measuring the wrong condition: accidental test access,
non-deranged pairing, wrapped token shifts, padded samples entering metrics,
missing mechanism telemetry, path overwrite, or protocol drift.

## Adversarial checks

- `fixed_derangement` is deterministic, permutation-valued, and has no fixed
  points for every batch with at least two records; the singleton case is
  explicitly covered as the only impossible derangement.
- A0/A1/A2 construction is tested on 32 synthetic unique-parent rows. The
  global permutation mismatch count is 32, and the within-batch permutation
  metadata is hashed per batch.
- Token-grid shifting is tested on a 3×3 grid with a non-wrapping clamped
  border. Both `[B,N,D]` and `[B,N,G,D]` contracts are covered; non-square
  token counts fail closed.
- Local Windows paths and paths outside the declared cloud roots are rejected;
  a diagnostic output directory equal to or nested under an input artifact is
  rejected.
- Final padded validation rows are ignored through `valid_count` and the
  existing ignore-index metric contract. Empty valid-pixel metrics fail closed.
- CEAK/CFEDGE pairwise separation requires every declared telemetry variable;
  missing variables cannot silently produce a passing but empty mechanism test.
- The runner rejects non-CUDA execution, rejects a missing/incorrect model
  state, closes CROMA hooks, and clears cloud CUDA cache between models.

## Results

- Targeted D0 tests: 9 passed.
- Full project synthetic tests: 234 passed, one pre-existing warning only.
- `py_compile`/`compileall`: pass.
- ResearchPilot validator: pass, zero violations; no local data or GPU probe.

## Residual risks

1. The cloud execution control must bind the three exact checkpoint paths and
   their expected SHA256 values before the D0 command is dispatched.
2. A successful process is not a scientific conclusion; the returned JSON must
   be independently checked for all 180 validation rows, seven shifts, all
   three models, `test_accessed=false`, and `scientific_result=false`.
3. D0-C matched C0–C6 training is deliberately not implemented or launched
   in this step; it remains conditional on review of the D0 receipt.

## Decision

`CONDITIONAL_PASS_FOR_LOCAL_D0_TEST_CONTRACT`.

No adversarial test found a local contract violation. The cloud and scientific
gates remain unchanged; no mechanism ranking or paper claim is permitted from
this review.
