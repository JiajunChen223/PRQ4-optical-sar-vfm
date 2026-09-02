# Independent V12-D0 release rereview

- Scope: latest V12-D0 code snapshot, clean-sync manifest, r2 release package,
  canonical `CODE_REPORT.json`, README, plan/intent binding, test seal, and
  local validator.
- Mode: read-only. No source code, real data, pretrained weight binary, cloud
  host, GPU probe, training, or sealed-test object was accessed.
- Date: 2026-08-29 (Asia/Shanghai).
- Decision: **PASS** for V12-D0 baseline-stress code-only handoff.

This is an engineering/release decision only. It is not a scientific result,
does not decide the V12-D0 objective gate, and does not authorize CMCD or
sealed-test access.

## 1. Current canonical report and release identity

`F:\PRQ4\02_experiment\code\review\CODE_REPORT.json` now binds the current
snapshot rather than the previously stale V11 snapshot:

- status: `PASS`
- route/candidate: `R-EO-CMCD-01` / `CMCD-01`
- manifest:
  `F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v12_d0_20260829.json`
- manifest SHA256:
  `a26ec691d7681004d6e31e654bca70016879cffa6f5eeb5b472668ade0ac4be6`
- package:
  `F:\PRQ4\02_experiment\artifacts\geotoken3path_code_v12_d0_20260829_r2.tar.gz`
- package SHA256:
  `6723b0665e53a21e9256024715effb0826bd41c22014a5ee004d4cd187efceb7`
- package size: 188,254 bytes
- tests: `310 passed`
- validator: `117 executable/config files; 0 violations`
- local data: `clean`
- local GPU probe: `forbidden_not_run`
- scientific result: `false`
- V12-D0 CMCD implementation: explicitly absent/unauthorized

The previous `independent_v12_release_review.md` was a review of the earlier
stale snapshot and reported BLOCKED. Its findings are resolved by the current
manifest/package/README/CODE_REPORT refresh; this rereview is the current
release decision. The historical report should remain append-only, not be
rewritten.

## 2. Manifest and package audit

The V12 manifest declares 106 files. I recomputed every listed file's byte
count and SHA256 against the current code tree: 106/106 exist and match, with
zero missing or changed entries. The manifest's current SHA256 is
`a26ec691d7681004d6e31e654bca70016879cffa6f5eeb5b472668ade0ac4be6`.

The r2 tarball contains 107 members: 106 payload files plus the embedded
`researchpilot_code_release_manifest.json`. The payload set exactly equals the
external manifest set; there are no missing or extra payload paths. The
embedded manifest declares 106 files and points to the same external manifest
SHA256. The package SHA256 and size are:

```text
6723b0665e53a21e9256024715effb0826bd41c22014a5ee004d4cd187efceb7
188254 bytes
```

The V12 additions and shared implementation changes are all included:

- `configs/model/v12_objective.yaml`
- `configs/experiment/v12_objective_route.yaml`
- `src/geotoken3path/losses/segmentation.py`
- `src/geotoken3path/losses/__init__.py`
- `src/geotoken3path/engine/formal_runner.py`
- `src/geotoken3path/utils/config.py`
- `src/geotoken3path/utils/run_manifest.py`
- `scripts/train.py`
- `tests/unit/test_v12_objectives.py`

No package member has an absolute archive path or a data/checkpoint/cache
suffix. In particular, no `.pt`, `.pth`, `.ckpt`, `.safetensors`, raster,
array, archive, or CSV payload is present.

## 3. README and route documentation

`F:\PRQ4\02_experiment\code\README.md` now correctly describes V12-D0 as an
objective/metric-alignment baseline stress test, identifies
`R-EO-CMCD-01` / conditional `CMCD-01`, states the MacroCE/CE+Lovasz rows and
conditional R3 trigger, keeps CMCD implementation/training prohibited before
the D0 decision, preserves V11 as a closed negative result, and keeps the
sealed test closed. It no longer presents V11 TASR as the current route.

## 4. Validator, tests, and seal checks

Fresh independent executions used the configured interpreter:

```text
F:\anaconda3\envs\dl_env\python.exe -B -m pytest F:\PRQ4\02_experiment\code\tests -q --disable-warnings --cache-clear
310 passed, 1 warning in 7.14s

F:\anaconda3\envs\dl_env\python.exe -B C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4 --output F:\PRQ4\02_experiment\code\review\validate_code_project_v12_d0_rereview_20260829.json
status=pass; scanned_executable_or_config_files=117; problems=[]; violations=[]; local_gpu_probe=forbidden_not_run
```

The validator receipt is at
`F:\PRQ4\02_experiment\code\review\validate_code_project_v12_d0_rereview_20260829.json`.
The runtime guard exists at
`src/geotoken3path/utils/test_seal.py`; train/evaluate/formal paths invoke
`assert_test_access_allowed`, and the guard permits `test` only for the
explicit `final_test` execution scale and seal status. No local GPU probe was
run.

## 5. V12-D0 binding and CMCD boundary

The plan, intent, and YAML route files agree on:

- route `R-EO-CMCD-01`, candidate metadata `CMCD-01`;
- diagnostic `V12-D0-STANDARD-OBJECTIVE-STRESS-TEST`;
- baseline-strengthening only, not innovation-claim eligible;
- validation-only rows with 840 train / 180 validation records;
- sealed test declared but not accessed;
- seed 0, 24 epochs, unchanged CROMA, optimizer, scheduler, augmentation,
  trainability, and RTX 3090 target;
- R1 MacroCE and R2 CE+Lovasz, with R3 conditional on the fixed +0.5 pp
  trigger.

The V12 model route keeps both baseline and candidate mechanism sets at
`always_fuse`; only the explicitly selected objective changes. Static source
inspection found no CMCD counterfactual branch, CMCD loss, or CMCD candidate
module in `src/`, `scripts/`, or tests. The `CMCD-01` identifier is route
metadata and a conditional future plan label, not an executed innovation.

## 6. Previous-route preservation

Compared with the V11 TASR manifest, V12 adds the three V12 config/test files
and changes only shared objective/runner/manifest plumbing plus README. No
V11-specific route/config file was changed, no file was removed, and no prior
V11/V10/V9 gate or result was rewritten by the V12 release payload.

## 7. Final decision

**PASS — V12-D0 baseline-stress code-only handoff is release-ready.** The
earlier stale-report and stale-README blockers are resolved. The experiment
owner may proceed to the protected cloud preparation and validation-only R1/R2
execution under the approved V12-D0 intent. R3 remains conditional on its
pre-registered trigger; CMCD implementation/training, any scientific claim,
and sealed-test access remain prohibited until the D0 decision and a later
approved plan.

