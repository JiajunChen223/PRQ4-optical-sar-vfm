# V13-D0 device-contract repair review

- Review mode: coordinator single-thread adversarial review; not independent review.
- Scope: `scripts/v13_d0_counterfactual_audit.py` and the new unit test only.
- Failure evidence: V13-D0 r2 stopped before result because CUDA batch statistics were added to CPU class accumulators.
- Repair: `_accumulator_value` now detaches every batch statistic and explicitly aligns it to the persistent accumulator's device and dtype before arithmetic.
- Information-flow contract: unchanged. Each validation batch still calls `model.bridge` once; ON uses `always_fuse`; OFF uses `unimodal_optical` on the same CROMA outputs.
- Scientific protocol: unchanged. Same SEN12TS validation split, epoch18 best-validation checkpoint, CROMA initialization, metrics and class-oracle gate.
- Safety: no training path was added; test access remains rejected; real data and weights remain cloud-only.
- Regression evidence: 319 tests passed with one pre-existing warning; code validator scanned 126 executable/config files with zero violations.
- Packaging: 109 reviewed code/config/test files; no data, weights, credentials or cache artifacts.
- Findings: no unresolved blocking engineering finding in the reviewed delta.
- Decision: PASS for guarded code-only synchronization and identical V13-D0 validation rerun.

This review cannot approve a scientific hypothesis. CMCD remains conditional on the completed V13-D0 Gate A result.
