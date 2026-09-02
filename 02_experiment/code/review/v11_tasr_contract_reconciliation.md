# V11 TASR contract reconciliation

The V11 plan contains two related but distinct equations. The intermediate
spatial redistribution projection enforces the raw token anchor
`mean(P_i(anchor_conserved)) = L_i`. The final candidate contract then defines
the learnable residual `D = Pi(Diffuse(B) - B)` with
`mean(P_i(D)) = 0` and `Z = B + alpha D`; this latter clause is the explicit
zero-start/identity hard contract and is what the approved hard-contract list
tests. The implementation exposes both quantities instead of conflating them:

* `tasr_token_anchor_conservation_max_abs` audits the intermediate raw-token
  anchor and is below `1e-6` in the formal 225-token synthetic fixture.
* `tasr_residual_conservation_max_abs` audits the final residual and is below
  `1e-6` for every class and token patch.
* `alpha=0` returns the P1 bilinear output bitwise, while nonzero alpha uses the
  same zero-mean residual and never receives a label or boundary target.

This reconciliation deliberately avoids an unapproved discontinuous hard gate:
the final TASR candidate remains the continuous zero-start residual specified
in the V11 plan. The raw-token anchor is a checked semantic witness for the
diffusion projection, not a second classifier or decoder.

Evidence:

* `tests/unit/test_tasr.py::test_formal_225_token_geometry_has_exact_anchor_witness`
* `tests/unit/test_tasr.py::test_zero_start_is_bitwise_p1_identity`
* `02_experiment/reports/v11_tasr_synthetic_liveness_20260829.json`

No scientific performance or candidate-support claim is made by this document.
