# V14-DTSF code review

- Review mode: coordinator single-thread adversarial review, user-authorized; no independent code reviewer was available in this turn.
- Scope: `mechanisms/dtsf.py`, shared `OpticalSarTokenModel` integration, resolver/factory/run-manifest/formal-runner registration, V14 configs and synthetic tests.
- Mechanism boundary: the four explicit late SAR CROMA depth taps are normalized, transformed by fixed H4, and only C1:C3 are passed through a rank-8 readout. The residual is zero-start straight-through and is inserted before late local optical-SAR fusion.
- Training-object parity: baseline and all DTSF controls use the same model factory, state-dict surface, trainability surface, optimizer, objective, split and entry point; only `model.mechanism_set` selects the internal branch.
- Safety: the implementation has no real-data path, no local GPU probe, no test-split access, no new checkpoint/weight binary and no external router/refiner. Formal raw-image execution still requires the existing cloud-only CROMA audit and test-seal guard.
- Numerical safeguards: `[B,N,4,D]` validation, finite checks, deterministic H4/random control basis, fixed rank=8, no N×N or D×D allocation, and explicit telemetry for consensus/innovation energy, orthogonality and residual norm.
- Zero-start evidence: synthetic model test confirms strict shared-token output identity at alpha=0; adapter test confirms finite modes and nonzero alpha gradient.
- Regression evidence: 328 synthetic tests passed with one pre-existing warning; code validator scanned 133 executable/config files with zero violations.
- Packaging: r3 clean-sync manifest contains 114 code/config/test files; release archive contains no data, weights, checkpoints, credentials or caches.
- Scientific boundary: no mIoU/OA or candidate-support claim is made by this code review. The formal DTSF-01 result remains pending cloud execution.
- Findings: no unresolved blocking engineering finding in the reviewed delta.
- Decision: PASS for guarded code-only synchronization and subsequent cloud environment/data/protocol reattachment.
