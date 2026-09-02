# Independent V6 CC-SCBC code re-review

**Review date:** 2026-08-29  
**Scope:** current `F:\PRQ4\02_experiment\code` snapshot after the
formal-entry repair  
**Route:** `R-EO-CCSCBC-01` / `CC-SCBC-01`  
**Review mode:** independent, read-only

## Boundary

No SEN12TS pixels/labels/caches, real CROMA weights, checkpoints, sealed-test
data, SSH, CUDA/GPU discovery, cloud access, download, or training was used.
The smoke command and synthetic liveness are implementation checks only and
are not scientific results.

## Re-review conclusion

**PASS_FOR_V6_LOCAL_REVIEW; FRESH_PACKAGE_GATE_PENDING.**

The original V6 cloud-entry blocker is closed: `formal_runner.py` now contains
the `CC-SCBC-01` direction mapping and allowlist, and the generic successor
error wording is current. `evaluate.py` now exposes all V6 CC-SCBC modes and
selects the V6 resolver. The local hard contract remains passing. The next
required step is still a fresh V6 clean-sync manifest/CODE_REPORT and guarded
code-only sync; no cloud run may use the stale D3/V5 manifests.

## Closed checks

### Formal runner binding — PASS

- `src/geotoken3path/engine/formal_runner.py:27-33` binds
  `CC-SCBC-01 -> cc_scbc_class_conditioned_set_credit`.
- The inline formal-direction allowlist at `:242` includes `CC-SCBC-01`.
- The successor validation at `:244-251` now accepts a matched C1 direction,
  requires it for the candidate mechanism, and rejects mismatched directions.
- The current error text is mechanism-family neutral (`successor candidate
  rows`), so no stale CEAK-only wording remains in this path.
- The V6 unit regression checks the direction map and manifest binding; a
  static current-source check also confirmed the map, allowlist token, and
  generic error string.

### Evaluation entry binding — PASS

- `scripts/evaluate.py:18-22` imports `resolve_v6_cc_scbc_config`.
- Its choices at `:30-32` include the C1, C2 and C3 CC-SCBC mechanism names.
- Its resolver branch at `:72-78` selects the V6 config for `cc_scbc_*`.
- Local smoke invocation of C1 completed with
  `scientific_result=false`, finite synthetic logits/metric, and no real
  artifact access.

### CC-SCBC mechanism and parity — PASS (unchanged)

- Exact forward identity and non-diagonal token-set Jacobian remain covered by
  the targeted tests and the CPU synthetic liveness receipt.
- Class-anchor/responsibility detachment and eval scaffold removal remain
  covered.
- Two-stage matched `always_fuse`/C1 output equality, state-dict key parity,
  and trainable-parameter parity remain passing.
- V6 resolver parity remains one leaf difference:
  `model.mechanism_set`; the baseline and C1 share the matched common
  protocol digest for a given execution scale.

## Fresh verification

- Targeted V6/formal-entry/integration checks: **23 passed**.
- Full local suite after the repair: **291 passed**, one pre-existing warning
  in `tests/unit/test_ceak_successor.py`.
- Independent AST/YAML parse: **79 Python files + 12 YAML files, 0 errors**.
- V6 synthetic Jacobian liveness: **PASS**, CPU-only; embedded receipt SHA
  `90468afeb13e8f4bf9ec2b0ce760c352276248bd2c9811545708f2fff21f9be4`.
- Existing ResearchPilot validator receipt remains `status=pass`, 106
  executable/config files, 0 violations, with local GPU probe
  `forbidden_not_run`. The two repaired entry scripts contain only the
  approved routing/wording changes and require a fresh validator/package
  binding before sync.

## Remaining package condition

The current `02_experiment/code/manifests` directory still has no V6
clean-sync manifest, and the historical `review/CODE_REPORT.json` is bound to
the D3 manifest. This is no longer a source-entry defect, but it remains a
hard release condition. Generate a new V6 manifest and CODE_REPORT from the
post-repair tree, then perform only the guarded code-only sync. Preserve V5/D3
artifacts and the sealed-test state. After cloud reattachment, the approved
next experiment is C1 seed-0 at 24 epochs; C2/C3 remain conditional on
`mIoU >= 50.0075%`.

## Current relevant SHA256

| File | SHA256 |
|---|---|
| `src/geotoken3path/mechanisms/cc_scbc.py` | `b3f3e8852e3c05733527c3d4f1035b4d62c3f682dcc041079edf14005835da70` |
| `src/geotoken3path/models/fusion.py` | `bd940fb2961e248fb173664371b085ff9f8c6b24c92eee3f9420aefa0c33668e` |
| `src/geotoken3path/utils/config.py` | `1d4558acb23b70a02f5aa757c8092bf5e6ba16a5f6e3c95fae51a70451f93e67` |
| `src/geotoken3path/utils/run_manifest.py` | `8283658fefe4e849d5281e222c8039bea1f211ea2aa5ca4a2c0afb6bb03649d3` |
| `src/geotoken3path/engine/formal_runner.py` | `324ebdfb4c155bb5d17acd3a36c8368a72433367502ef901eecd90c4b0bd6ed3` |
| `scripts/train.py` | `2b9cf08ffdba430b119626d9226c6f2c5b7a16c56e8a46eba7b711db730d181e` |
| `scripts/evaluate.py` | `cb78303b1e56b441a0923e561d7f5c93a064709a5aea61f9934833040c6b1a78` |
| `tests/unit/test_cc_scbc.py` | `b0c78fd97b0e9c459cb1d3e5216475eedfaec220408ea68fe7ab81e2b7a3fa9e` |
| `configs/model/v6_cc_scbc.yaml` | `6cd88ff793ba5b27c440ee779a1e2eb4f8c67460b51b9bea2ed0466f74ae290c` |
| `configs/experiment/v6_cc_scbc_route.yaml` | `80fd2d0d926ef51caf41fb25ffc35a597f277a96f493e9a6b82d57e044c76e1b` |

No source file was modified by this re-review.
