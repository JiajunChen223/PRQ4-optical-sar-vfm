# Independent CEAK-R1~R4 final closure review

**Date:** 2026-08-28  
**Scope:** current local snapshot of `F:\PRQ4\02_experiment\code`  
**Decision:** **PASS — CEAK-R1, CEAK-R2, CEAK-R3, and CEAK-R4 are closed locally.**

This is a software-contract result only. It does not approve cloud execution,
real-data/weight compatibility, GPU performance, scientific gains, or any
sealed-test access.

## Review boundary

This review read local source/config/test files and ran only CPU synthetic
checks. It did not connect to cloud, read real imagery/labels/checkpoints,
probe GPU, train, or access the sealed test split. No source file or
`CODE_REPORT.json` was modified; this receipt is the only requested write.

## Re-executed checks

| Check | Command | Result |
|---|---|---|
| Targeted CEAK/manifest/config tests | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest -q tests/unit/test_ceak_successor.py tests/unit/test_run_manifest_hardening.py tests/unit/test_resolved_config.py` from `F:\PRQ4\02_experiment\code` | **31 passed**, 1 existing warning, 2.09 s |
| Full CPU synthetic suite | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest -q` | **225 passed**, 1 existing warning, 5.82 s |
| Static ResearchPilot validator | `F:\anaconda3\envs\dl_env\python.exe -X utf8 -B C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4` | **PASS**, 77 executable/config files, 0 problems, 0 violations; `local_gpu_probe=forbidden_not_run` |

The sole warning is the existing test assertion converting a grad-bearing
telemetry tensor to `float` (`tests/unit/test_ceak_successor.py:50`); it is not
a failure or an R1–R4 closure defect.

## Closure evidence

### CEAK-R1 — zero-start value no-op and gradient-live path: CLOSED

`src/geotoken3path/models/fusion.py:806-818` implements
`scaled + (value - value.detach())`, and the CEAK/SUBPACK/CFEDGE branches use it
at lines 871, 943, and 997. The targeted zero-start test at
`tests/unit/test_ceak_successor.py:59-71` passed. An independent CPU probe on
the current snapshot also showed bitwise baseline identity for all three rows
and nonzero mechanism gradients at scale zero:

| Direction | Bitwise identity | Scale-grad max | Minimum mechanism-grad max |
|---|---:|---:|---:|
| CEAK-01 | `True` | 0.7604598999 | 0.0000510305 |
| SUBPACK-02 | `True` | 0.7978363037 | 0.0007396119 |
| CFEDGE-03 | `True` | 0.8571330309 | 0.0174142793 |

### CEAK-R2 — rank/M/K controls resolved, consumed, and manifest-bound: CLOSED

The constructor validates and stores the controls at
`src/geotoken3path/models/fusion.py:102-125,182-207`; SUBPACK consumes the
stored limits at `fusion.py:927-938`. The factory/model path passes the
resolved contract into every stage (`models/factory.py:33-57` and
`models/fusion.py:1236-1278`). The resolver records the controls in the
resolved snapshot (`utils/config.py:195-207,324-345`), and the manifest now
validates them at `utils/run_manifest.py:182-190` and carries them at
`run_manifest.py:313-315`.

Targeted tests cover invalid YAML controls and tampered resolved contracts
(`tests/unit/test_resolved_config.py:55-72` and
`tests/unit/test_ceak_successor.py:137-158`), and passed. The independent
probe confirmed all current stages consume `(rank=8, M=64, K=32)` and that a
synthetic `(4,16,8)` constructor contract changes both stage attributes and
evidence-head width.

### CEAK-R3 — direction/mechanism identity and reverse annotations: CLOSED

The exact mapping is declared at
`src/geotoken3path/utils/run_manifest.py:97-101`. `build_run_manifest` now
rejects mismatches both when an ID is supplied and when a formal successor row
omits it (`run_manifest.py:348-360`). It also rejects baseline rows carrying a
successor direction, closing the reverse-label hole. The manifest includes the
selected mechanism and successor edge contract in the hashed payload, and
`verify_run_manifest` validates the digest (`run_manifest.py:273-282,308-315,
361-366`).

The formal runner applies the same exact mapping before any loader/model work
(`engine/formal_runner.py:27-31,238-246`). Targeted tests passed for correct
CEAK-01 labeling, missing/wrong labels, baseline reverse annotation, manifest
contract field, and tampered contract rejection
(`tests/unit/test_ceak_successor.py:116-158`). Independent CPU probes also
rejected missing and wrong IDs for CEAK-01, SUBPACK-02, and CFEDGE-03 in both
manifest construction and formal-runner preflight.

### CEAK-R4 — CFEDGE Shapley operation/name boundary: CLOSED

`src/geotoken3path/models/fusion.py:970-987` evaluates the four coalition
utilities and computes the two individual two-player Shapley allocations,
while retaining the signed interaction finite difference as a distinct routing
quantity. The explicit telemetry names are
`cfedge_optical_shapley_mean` and `cfedge_sar_shapley_mean`; the private SAR
path and null routing remain in `fusion.py:988-1010`.

The targeted CFEDGE test passed. An independent CPU reconstruction matched both
telemetry values and measured maximum Shapley efficiency error
`1.49e-08`; logits were finite.

## Current source hashes

| File | SHA-256 |
|---|---|
| `src/geotoken3path/utils/run_manifest.py` | `292D62001475209C61F767B5A59D61203C15E9C3F938209D6F66BD657DDF9952` |
| `tests/unit/test_ceak_successor.py` | `E7EFBD734CEB7395B595C15AFCC56E62705F4431095E73F894E68F4DBAC64716` |
| `tests/unit/test_run_manifest_hardening.py` | `85F8414C2548255A71780A3A363BC08F0E84A7AF77E80ADFD9E5DBCFEED5FA97` |
| `tests/unit/test_resolved_config.py` | `14085D47B64F9C219B2D4D9C9EF281082A82EA252253086FBB70448CC9C20960` |
| `src/geotoken3path/models/fusion.py` | `6CCD3372629D5E6FAB44F8BDBDD813DCAFAAC572B5B90DA572B8F6B87FA97E68` |
| `src/geotoken3path/utils/config.py` | `2CCE5FF6676BADCB6A5765C9351B5CFB7916041E3E756362E7E9CF94D5C5A57F` |
| `src/geotoken3path/engine/formal_runner.py` | `456AC3906C9F2BDA0457B9C505E992B9E75B69F207647FC720DCFA77FB70C39B` |
| `src/geotoken3path/models/factory.py` | `B1357A30EA521D7AFAF0EDDC23D1F47B61F4DB1BE40CFE0FB59A915BFE80D8FE` |
| `configs/model/geotoken3path.yaml` | `596F41C3B4F6E072DB2CA6F97C99AFA149833DA661A931D7B149B68E362E5527` |

**Handoff:** R1–R4 are locally closed. Keep the separate ResearchPilot
approval, cloud-only data/weights, and sealed-test gates unchanged.
