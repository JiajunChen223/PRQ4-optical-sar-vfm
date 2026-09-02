# Independent SEN12TS successor code-contract review

Review scope: `F:\PRQ4\02_experiment\code` only.  This is a read-only
independent review of the active resolver, benchmark configuration, dataset
contracts, CROMA preprocessing implementation, fixtures, `CODE_REPORT.json`,
and the code clean-sync manifest.  No project-level protocol, experiment
manifest, Router state, approval/binding, gate, data, weight, GPU, SSH, or
cloud artifact was changed.

## Decision

`CODE_CONTRACT_STATUS=PASS` for the active local SEN12TS resolver and synthetic
contract suite.  `PROJECT_BINDING_STATUS=BLOCKED_OUT_OF_SCOPE` because the
project-level `02_experiment/protocol/experiment_protocol.yaml` and
`02_experiment/experiment_manifest.json` still point to the historical
`01_literature/synthesis/dataset_registry.json`.  The project-level successor
registry also does not yet contain the approved SEN12TS entry.  Those bindings
must be generated and validated by the Plan/Router service before the
Experiment service updates them.

## Active code evidence

- `configs/benchmarks/sen12ts_worldcover.yaml` resolves to
  `sen12ts_worldcover_3region_1200` and
  `/root/autodl-tmp/sen12ts_worldcover_3region_1200`.
- Raw channel contract is SAR=19 and optical=14.  Selectors are SAR `[1,0]`
  from raw `[VH,VV]` to canonical `[VV,VH]`, and optical `[0..11]` from the
  14-channel source.
- Parent/crop contract is `[256,256] -> [120,120]`; the active model has 11
  WorldCover classes and `ignore_index=255`.
- `croma_official_dynamic_v1` is per-micro-batch/per-channel over axes
  `[0,2,3]`, with micro-batch 16, train `drop_last`, validation/inference
  deterministic repeat-padding plus output trimming, per-rank statistics,
  explicit zero-standard/nodata/float32 policies, pinned source revision and
  README/loader hashes, and `normalization_locked=false` pending value-level
  parity.
- `resolve_approved_config(..., "geotoken_3path")` resolves the SEN12TS ID,
  active cloud root, 12+2 model channels, selectors, 11 classes, and ignore
  255 without a BigEarthNet fallback.
- The old `configs/benchmarks/copernicus_bench.yaml` is retained as an explicit
  legacy backup.  Its SHA256 remains
  `2aa5d3b6d59e194795bfe9448d24daacae3170a78adebad2de5fd81141c7249e`.
- The active smoke fixture now uses the SEN12TS dataset ID.  BigEarthNet
  strings remaining in code are confined to the explicit legacy backup,
  legacy compatibility constants, or historical review/license notes; no
  active resolver/test entry selects them.

## Independent checks

- No-cache synthetic pytest:
  `123 passed`.
- Global scope-aware `validate_code_project.py --project-root F:\PRQ4`:
  `PASS`, 43 executable/config files scanned, 0 violations.
- `CODE_REPORT.json` still says `123 passed` but records the previous 42-file
  validator count and the previous clean-manifest SHA; it is therefore a
  stale report artifact, not current evidence.
- `manifests/clean_sync_manifest.json` still has 45 entries and does not list
  the newly added `configs/benchmarks/dataset_registry.yaml`; its package
  control is consequently stale until the code-only manifest update is
  authorized in the next write step.

## Out-of-scope project-level stale references

The following were observed but intentionally not modified:

- `02_experiment/protocol/experiment_protocol.yaml`:
  `dataset_summary_ref=01_literature/synthesis/dataset_registry.json`.
- `02_experiment/experiment_manifest.json`:
  `approved_plan.dataset_summary_ref=01_literature/synthesis/dataset_registry.json`.
- `01_literature/synthesis/dataset_registry_successor_20260821.json` contains
  DFC/C2Seg/legacy candidates but no SEN12TS active entry.

These are binding/handoff changes, not local code-contract edits, and remain
blocked pending a Plan/Router-generated successor registry/handoff/binding.

## Boundary

This review proves only local configuration/contract consistency and synthetic
software behavior.  It is not evidence of real-data shape/hash/rights closure,
CROMA checkpoint compatibility, GPU feasibility, training, or any scientific
result.
