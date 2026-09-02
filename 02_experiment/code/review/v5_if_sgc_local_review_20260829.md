# Independent local review — V5 IF-SGC

- **Scope**: local code/contract review only; no SSH, data, weights, GPU or training.
- **Route**: `R-EO-IFSGC-01 / IF-SGC-01`.
- **Review status**: `CONDITIONAL_PASS_FOR_LOCAL_HARD_CONTRACT`.

## Closed checks

1. **Forward identity**: custom autograd operator returns the exact token tensor in forward; targeted test confirms bitwise equality and analytic backward scaling.
2. **Backward contract**: `dL/ds_i = gamma_i dL/ds_tilde_i`; salience consumes `s.detach()` and receives only the declared surrogate gamma gradient.
3. **Gain constraints**: finite, positive, per-sample mean-one token gain for C1; scalar C2 is token-constant and batch-normalized; C3 preserves the C1 per-sample gain multiset while shifting token assignment.
4. **Inference removal**: `self.training=False` bypasses the salience projection and returns uniform-one telemetry; targeted test confirms zero salience-head forward calls.
5. **Parity surface**: IF-SGC adds no state-dict parameters; baseline and candidate state-dict keys and trainable parameter names match in the synthetic factory contract.
6. **Protected boundary**: no local data/weight I/O, no GPU probe, no test-split access, no change to D3 historical artifacts.
7. **Formal entry binding**: `IF-SGC-01` is now registered in the formal runner direction map and `train.py`/`evaluate.py` route selectors; the old CEAK path remains unchanged.

## Fresh verification

- `tests/unit/test_if_sgc.py`: 10 passed.
- Full local suite: 275 passed, one pre-existing warning, no failures.
- `validate_code_project.py`: 100 executable/config files, 0 problems, 0 violations; local GPU probe remains `forbidden_not_run`.
- After formal-entry binding repair, the code snapshot requires a fresh package hash and guarded sync; no cloud training is started from the stale package.

## Post-C1 repair

- The first C1 launch `PRQ4-V5-IFSGC-C1-FORMAL24-SEED0-20260829` is classified `invalid_protocol`: it failed on the first batch because the float32 mean-one assertion was too strict for AMP salience values. No complete epoch, metric, checkpoint or scientific result exists.
- The repair computes the normalization denominator in float64 and adds a 225-token half-precision regression test. A fresh 276-test snapshot is required before retry; the failed run remains immutable and cannot count as a candidate rejection.
- The linked retry `...-R2` exposed a second precision layer: CUDA autocast converted the salience projection itself to float16. It is also classified `invalid_protocol` (first batch, no complete epoch/metric/checkpoint). The successor repair computes and validates gamma in float32 before the final token-dtype cast and adds an autocast-like bfloat16 test.
- Synthetic liveness: gamma mean 1.0, variance 0.0010043, gradient-scale Pearson 1.0, operator scale error 1.79e-7, top/bottom effective scale ratio 1.09385, eval auxiliary active=false.

## Residual blockers before a formal cloud C1 run

- Active CEAK YAML is preserved for historical replay; the V5 route is in explicit successor configs `configs/model/v5_if_sgc.yaml` and `configs/experiment/v5_if_sgc_route.yaml`. A fresh clean-sync manifest must include them and the IF-SGC source/tests.
- C1 cloud execution still requires the ordinary guarded code-sync package, cloud CROMA/data audit and a current run control; no cloud command is authorized by this local review alone.
- SSRN 5887466 remains a high-risk prior-art record with public metadata/abstract but HTTP-403 full text; do not convert it into a novelty-positive claim.
- No seed-0 mIoU, baseline result, C2/C3 result or scientific support is present in this report.

## Decision

Local IF-SGC hard-contract implementation is suitable to proceed to clean packaging and cloud preflight preparation. It is **not** baseline-training-ready and it does not advance `INNOVATION_REVIEW` or unseal the test.
