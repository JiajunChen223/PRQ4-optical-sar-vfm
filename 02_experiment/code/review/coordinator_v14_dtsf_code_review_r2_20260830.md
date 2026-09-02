# V14-DTSF code review r2

- Review mode: coordinator single-thread adversarial review, user-authorized; no independent code reviewer was available in this turn.
- Scope: r2 delta after initial DTSF code surface: training/evaluation CLI registration and synthetic hard-contract witness correction.
- Failure repaired: the first hard-contract script used a squared residual loss at a zero forward residual, which mathematically gives zero alpha gradient at initialization. This was a test-witness defect, not a model-path defect.
- Repair: the synthetic liveness witness now uses a deterministic non-symmetric linear weighting of the zero-start residual, exposing the declared alpha Jacobian while preserving exact forward identity.
- V14 mechanism unchanged: fixed H4 over late SAR taps [2,3,4,5], C1:C3 rank-8 readout, alpha=0 straight-through residual, late pre-fusion insertion.
- CLI parity: `train.py` and `evaluate.py` now expose DTSF mechanism choices and resolve `v14_dtsf` through the dedicated resolver; formal runner direction IDs DTSF-01/C2/C3/C4 are registered.
- Safety: no local real data, local weight, checkpoint, GPU probe or test split; synthetic contract uses CPU tensors only.
- Regression evidence: 328 tests passed with one pre-existing warning; code validator scanned 135 executable/config files with zero violations.
- Packaging: r4 clean-sync manifest and release package contain code/config/tests only; package audit must be rerun before synchronization.
- Findings: no unresolved blocking engineering finding in the reviewed delta.
- Decision: PASS for guarded r4 code-only synchronization and subsequent DTSF cloud run preparation.
