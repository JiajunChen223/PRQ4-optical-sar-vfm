# GRCA local code review

Status: `CONDITIONAL_PASS_FOR_LOCAL_CODE_SERVICE`.

- CPU synthetic tests: **170 passed**;
- validator: **PASS**, 56 executable/config files, 0 violations;
- GRCA/MRHT/DAGB joint-attention modes are selectable through the same factory;
- the wrapper changes the relative-bias input passed to the audited CROMA
  `cross_encoder` and records a zero-start `joint_readout` into the existing
  late-token path;
- synthetic tests verify `[B,H,N,N]` bias shape, unchanged Q/K/V outputs at
  zero-start, exact joint-output no-op parity, and decoder-visible readout
  shape;
- local data/GPU probing remain forbidden; test remains sealed.

Open cloud-only checks: exact official Q/K/V backward memory, one/few-layer
trainability mask, strict checkpoint loading with the wrapper, and 3090
activation/throughput preflight. No scientific claim is made by this review.
