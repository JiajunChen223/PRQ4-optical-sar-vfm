# Independent V12-D0 release and reproducibility review

- Review scope: `F:\PRQ4\02_experiment\code` and the V12-D0 code-only package.
- Review mode: read-only; no source, real data, pretrained weights, cloud host, GPU, training run, or sealed-test object was accessed.
- Review date: 2026-08-29 (Asia/Shanghai).
- Verdict: **BLOCKED** for code-service handoff until the canonical `CODE_REPORT.json` and the package README are refreshed.

This is an engineering/release verdict only. It is not a scientific result and does not advance the V12-D0 objective gate.

## Evidence checked

### V12 plan, intent, and configuration binding

The plan and execution intent are present and agree on the V12-D0 binding:

- plan: `F:\PRQ4\02_experiment\reports\v12_objective_metric_alignment_plan_20260829.json`
- intent: `F:\PRQ4\02_experiment\reports\v12_d0_standard_objective_stress_intent_20260829.json`
- route: `R-EO-CMCD-01`
- candidate label: `CMCD-01` (conditional only; not authorized as a D0 training innovation)
- diagnostic: `V12-D0-STANDARD-OBJECTIVE-STRESS-TEST`
- rows: R1 MacroCE, R2 CE+Lovasz, R3 MacroCE+Lovasz conditional on the declared +0.5 pp trigger
- baseline-only / innovation-claim-eligible: `true` / `false`
- split and horizon: 840 train records, 180 validation records, 24 epochs, seed 0
- sealed test: 180 records declared, `test_accessed=false`, `test_sealed=true`
- unchanged protocol declarations: CROMA-base audited checkpoint, AdamW, cosine-with-warmup, paired D4 augmentation, RTX 3090 24 GB

`configs/model/v12_objective.yaml` and `configs/experiment/v12_objective_route.yaml` carry the same route/candidate identifiers. The model route resolves to `always_fuse`; the V12-D0 change is the explicitly selected objective, not a CMCD model mechanism.

### Manifest/package completeness

The V12 clean-sync manifest is internally consistent:

- manifest: `F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v12_d0_20260829.json`
- manifest status: `pass`
- payload files: 106
- manifest SHA256: `1b3bafbc1cfd2e793db661fcbdd3531f3dafa093c349e8e6d51805037c7dc9c4`
- package: `F:\PRQ4\02_experiment\artifacts\geotoken3path_code_v12_d0_20260829.tar.gz`
- package size: 186,881 bytes
- package SHA256: `6c266280873b179bbf7eead31cc6281cc2be339a02ab6cc23636060aa80687c0`

Every one of the 106 manifest entries exists with the recorded byte count and SHA256. The package contains 107 members: the 106 manifest payload files plus the embedded release manifest. The payload set matches the external manifest exactly; the embedded manifest points to the external manifest SHA256 above.

The V12 additions/changes are present:

- added: `configs/model/v12_objective.yaml`
- added: `configs/experiment/v12_objective_route.yaml`
- added: `tests/unit/test_v12_objectives.py`
- changed shared path: `src/geotoken3path/losses/__init__.py`
- changed shared path: `src/geotoken3path/losses/segmentation.py`
- changed shared path: `src/geotoken3path/engine/formal_runner.py`
- changed shared path: `src/geotoken3path/utils/config.py`
- changed shared path: `src/geotoken3path/utils/run_manifest.py`
- changed shared path: `scripts/train.py`

The loss, formal runner, V12 configs, and V12 tests are therefore included in the package. No package member has an absolute archive path or a data/checkpoint suffix (`.pt`, `.pth`, `.ckpt`, `.safetensors`, raster/archive/cache formats).

### Validator, tests, and test seal

Fresh local checks were run with the project interpreter:

```text
F:\anaconda3\envs\dl_env\python.exe -B -m pytest F:\PRQ4\02_experiment\code\tests -q --disable-warnings --cache-clear
307 passed, 1 warning in 7.88s

F:\anaconda3\envs\dl_env\python.exe -B C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4 --output F:\PRQ4\02_experiment\code\review\validate_code_project_v12_d0_independent_20260829.json
status=pass; scanned_executable_or_config_files=117; problems=[]; violations=[]; local_gpu_probe=forbidden_not_run
```

Fresh validator receipt:
`F:\PRQ4\02_experiment\code\review\validate_code_project_v12_d0_independent_20260829.json`.

`src/geotoken3path/utils/test_seal.py` is present. `scripts/train.py`, `scripts/evaluate.py`, and the formal runner invoke `assert_test_access_allowed`; the guard only permits the test split for `execution_scale=final_test` and `test_seal_status=final_test`. No local GPU probe was run.

The V12 objective implementation exposes only the declared `pixel_ce`, `macro_ce`, `ce_lovasz`, and `macro_ce_lovasz` objectives. No CMCD counterfactual forward/loss implementation is present before the D0 decision. The `CMCD-01` string in the V12 route metadata is conditional plan identity, not evidence of an implemented or executed CMCD candidate.

### Previous-route preservation

Comparing the V11 TASR manifest with the V12-D0 manifest found three additions and six shared-path changes, with no removals. The V11-specific route/config files and TASR mechanism source are unchanged by their recorded V11 hashes. The shared changes add V12 objective/config/manifest plumbing; no old route gate, sealed-test state, or V11 scientific result was rewritten by the V12 package manifest.

## Blocking findings

### BLOCK-01 (P0): canonical CODE_REPORT is stale

`F:\PRQ4\02_experiment\code\review\CODE_REPORT.json` still reports:

- route `R-EO-TASR-01`, candidate `TASR-01`
- V11 manifest `clean_sync_manifest_v11_tasr_20260829.json`
- `303 passed`
- `113` validator files
- generated at `2026-08-29T13:16:10.437895+00:00`

The current V12-D0 snapshot is route `R-EO-CMCD-01` / D0 baseline-strengthening, uses the V12 manifest above, has 307 passing tests, and the fresh validator scans 117 files. The ResearchPilot code skill requires the canonical `CODE_REPORT.json` to describe the reviewed snapshot before handoff. This mismatch blocks a PASS even though the fresh tests and validator pass.

Required repair: regenerate `CODE_REPORT.json` for V12-D0, bind it to the V12 manifest SHA256, include this independent review and the fresh validator receipt, and state explicitly that V12-D0 is baseline strengthening only and that CMCD remains unauthorized before the D0 decision.

### BLOCK-02 (P1): package README is stale and misbinds the route

`F:\PRQ4\02_experiment\code\README.md` is still the V11 TASR README. It states `R-EO-TASR-01` / `TASR-01` and says the TASR C1 run has not been executed. Because this README is included in the V12 package, a recipient unpacking the package would receive a route description inconsistent with the V12 manifest/configuration. It must be refreshed to identify V12-D0 as the current baseline-objective stress snapshot, preserve V11 as closed history, and state that no CMCD code/training is authorized before the D0 decision.

## Conclusion

**BLOCKED for release handoff, not blocked for the scientific V12-D0 plan.** The actual code/package payload is complete, hash-consistent, cloud-only, test-sealed, and passes 307 tests plus the fresh 117-file validator. The release cannot be marked PASS until `CODE_REPORT.json` and the included README are updated and the corresponding hashes are rechecked. No cloud D0 execution should be launched from this review receipt until that repair is complete.

