# Pre-baseline local scope

- Plan approval: recorded in the Router for `PLAN_APPROVAL`.
- Approved route: `R-EO-TRI-FUSE-01` / `CAND-01`.
- Local implementation: shared `OpticalSarTokenModel` factory with `always_fuse`, `static_sparse`, and `geotoken_3path` mechanism sets; baseline and candidate expose the same trainable parameter surface.
- Structural smoke checks: six unit tests pass on synthetic tensors; the candidate exposes three per-token route states and changes information flow under matched initialization.
- Real data: forbidden locally; no dataset was downloaded.
- Real weights: forbidden locally; no checkpoint binary was downloaded.
- GPU probing: forbidden locally; no `nvidia-smi`, CUDA probe or device inspection was run.
- Test split: sealed; synthetic validation only.
- Code release status: blocked pending the cloud-only pretrained compatibility audit and protocol lock; this is a software-contract status, not an experiment result.
- Scientific status: no baseline, innovation, efficiency or generalization result exists.
- Next protected checkpoint: `BASELINE_TRAINING_APPROVAL`.
