# Independent V12-D0 release final r5 review

- Scope: frozen V12-D0 r5 code-only release, its clean-sync manifest, and canonical `CODE_REPORT.json`.
- Mode: read-only. No code, data, weights, cloud execution, GPU probe, training, evaluation, or sealed-test access was performed.
- Date: 2026-08-29 (Asia/Shanghai).
- Decision: **PASS for V12-D0 code-only release handoff.**

This is an engineering/reproducibility decision only. It is not a scientific result and does not authorize CMCD or sealed-test access.

## 1. Manifest and package consistency

| Check | Result |
|---|---|
| Manifest | `F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v12_d0_20260829.json` |
| Manifest | `status=pass`; declared/observed entries `106/106` |
| Manifest SHA256 | `9667880bfc87214e8a5bff48399046a708bc35f56e941e8378178c9b8c7c5ae6` |
| Local manifest verification | 106/106 present; byte/SHA mismatches `0` |
| r5 package | `F:\PRQ4\02_experiment\artifacts\geotoken3path_code_v12_d0_20260829_r5.tar.gz` |
| r5 package | `188492` bytes; SHA256 `fa32fe81ba3f827d0d04ce3dd2cf2d5de812d89ef6cf8ade46c7dbd07d8c2522` |
| Archive structure | 107 regular members = 106 payload files + embedded release manifest; no unsafe paths/special members/forbidden data-weight binaries |
| Embedded manifest | 106 files; payload paths exactly match the external manifest; all payload byte/SHA checks pass; embedded source manifest SHA equals `9667880b...c7c5ae6` |

## 2. Canonical report and runtime checks

`F:\PRQ4\02_experiment\code\review\CODE_REPORT.json` is now bound to the current manifest and r5 package. Its fields agree with the recomputed values above, and all seven referenced review-report hashes match their files. It records `status=PASS`, `310 passed`, validator `117 executable/config files; 0 violations`, `local_data_status=clean`, `local_gpu_probe=forbidden_not_run`, `scientific_result=false`, and `test_accessed=false`.

Fresh current-snapshot checks (stdout-only, without overwriting any evidence file) also passed:

```text
F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest F:\PRQ4\02_experiment\code\tests -q -p no:cacheprovider --disable-warnings
310 passed, 1 warning in 7.56s

validate_code_project.py --project-root F:\PRQ4
status=pass; scanned_executable_or_config_files=117; problems=[]; violations=[]; local_gpu_probe=forbidden_not_run
```

The manifest excludes local real data, pretrained-weight binaries, and cache artifacts; no forbidden data/weight/archive binary was found under `02_experiment\code`. The test seal remains closed. Current gate state is `LOCAL_REVIEW` / `PENDING`, milestone `CODE_READY`, with no cloud execution requested by this review.

## Final decision

**PASS — r5 package, 106-entry manifest, canonical CODE_REPORT, full synthetic tests, validator, and local-data/GPU safeguards are consistent.** Proceed only within the existing guarded ResearchPilot workflow; retain real data/weights on cloud and the sealed test closed.
