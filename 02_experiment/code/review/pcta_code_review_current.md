# PCTA local code-service review

**Route:** `R-EO-PCTA-01 / PCTA-01`  
**Scope:** local code contract only; no real data, checkpoint binary, GPU, cloud execution or test access.  
**Verdict:** `PASS_FOR_LOCAL_CODE_ONLY; CLOUD_SYNC_AND_FORMAL_EXECUTION_PENDING`

## Checks completed

- `pcta_adapter.py` implements one bounded information-flow change: compact
  rank-limited carriers, normalized cross-power phase correlation, local
  5x5 soft-argmax and Fourier query transport with raw-query preservation.
- Zero-start `transport_strength=0` gives bitwise baseline parity in the
  synthetic adapter contract.
- PCTA, CPC and SCAT are distinct carrier operations; they are registered as
  separate mechanism sets and candidate IDs, not aliases or optimizer/loss
  changes.
- PCTA residual is injected only at the late CROMA tap; no invented fine-scale
  feature or extra token is introduced.
- FP16/bfloat16 compatibility is guarded by FP32 frequency-domain accumulation;
  a half-token regression test is included because arbitrary 15x15 FP16 FFT is
  unsupported on common CUDA paths.
- Config, train/evaluate entry points, run-manifest allowlist and baseline/
  candidate direction binding resolve to `PCTA-01`.
- Local data/GPU/weight/network scans pass; GPU probing remains forbidden.

## Quantitative evidence

- Full local synthetic test suite: **183 passed**.
- PCTA targeted tests: **5 passed**.
- Code validator: **PASS**, 60 executable/config files, 0 violations,
  `local_gpu_probe=forbidden_not_run`.
- Synthetic train smoke: passed with `scientific_result=false`.
- Synthetic evaluate: passed with `scientific_result=false`.
- Code-only manifest: 63 files, SHA256
  `13899f32729d3d3e26762409b120f3509f71600c82b074240ff0c541d4425a17`.
- Code-only package: `geotoken3path_code_runmanifest_r4.tar.gz`, SHA256
  `3d2b1d7ded23a0ce5b4273310a9a6f2659eb459d5ec0d00a77062c31fd0274b4`,
  93,512 bytes.

## Findings

1. **Closed:** previous odd-grid FP16 FFT hazard; fixed by FP32 accumulator and
   half-token regression.
2. **Closed:** route/config/report identifiers rebound from sheaf to PCTA.
3. **Closed locally:** runtime data-manifest and pretrained-audit references are
   now passed into the run manifest instead of using legacy hard-coded paths;
   the binding is covered by a regression test.
4. **Pending cloud-only:** actual official CROMA constructor/tap/output parity,
   checkpoint load receipt, 3090 VRAM/throughput and real SEN12TS loader audit.
5. **Pending scientific evidence:** PCTA has no metric claim; controls and
   controlled registration perturbation must be evaluated only after the
   Experiment gate is open.

No local finding authorizes training or turns synthetic results into a method
result. The old sheaf reports remain historical and are not consumed as PCTA
support.
