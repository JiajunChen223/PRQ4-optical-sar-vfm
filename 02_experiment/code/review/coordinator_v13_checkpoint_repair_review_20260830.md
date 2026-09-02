# V13 checkpoint repair coordinator single-thread adversarial review

- Review mode: coordinator single-thread review, explicitly authorized by user amendment `AM-20260829-162711-d5a5cf4b` after Luna-Max subagent quota exhaustion.
- Independence status: **not independent**; this report must not be described as a subagent review.
- Scope: checkpoint persistence/identity repair only. No data, weights, cloud command, GPU probe, training, evaluation, CMCD implementation, or sealed-test access.
- Decision: **PASS under the user-authorized single-thread review amendment**.

## Contract findings

1. In full-horizon screening, `early_enabled=false`; therefore `best_was_restored=false`, the ordinary final checkpoint records `epoch=stopped_epoch` and `checkpoint_role=final`. It cannot be mislabeled as the best epoch.
2. The best-validation state is CPU-cloned when the validation score improves and is saved separately as `<mechanism>_seed<seed>_best_epoch<E>.pt`, with `epoch=E` and `checkpoint_role=best_validation`.
3. In a genuinely early-stopped run with restore-best enabled, the restored final file records `checkpoint_role=best_restored_final` and its true restored epoch.
4. The best-validation checkpoint intentionally excludes optimizer state because it is an evaluation/reference artifact; final and best-restored-final checkpoints retain optimizer state.
5. Both best and final files are content-hashed after atomic persistence; `path`, `bytes`, and SHA256 are emitted in `run_result.json`.
6. The run manifest remains embedded in both files and retains validation split plus `test_accessed=false`.
7. No training variables, loss implementation, dataset, initialization, optimizer, scheduler, seed, epoch budget, augmentation, trainability, or model graph changed.

## Adversarial tests

- Validated best role/epoch with no optimizer.
- Validated final role/epoch with optimizer.
- Rejected zero/bool epoch and unknown role.
- Verified content SHA256/bytes identity.
- Atomic save/load roundtrip proved best epoch17 and final epoch24 remain distinct, including model-state values and optimizer presence.

Results:

- checkpoint-role tests: `7 passed`.
- full test suite: `317 passed, 1 warning`.
- ResearchPilot code validator: `PASS`, `121` executable/config files, `0` violations.
- local GPU probe: `forbidden_not_run`.

## Release binding

- Clean manifest: `F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v13_checkpoint_repair_20260830_r4.json`
- Manifest SHA256: `ba9d9f4f67d96a53f22085694a7c7652ee7a26484fd9bcc63444ff87c2160ef3`
- Manifest entries: `107/107`, mismatches `0`.
- Package: `F:\PRQ4\02_experiment\artifacts\geotoken3path_code_v13_checkpoint_repair_20260830_r4.tar.gz`
- Package SHA256: `b2a8fa0beb0dc9b1ff6926dab1d8c3418e348ff5f44ddd884af9109fed920b64`
- Package bytes: `189947`; members: `108` = 107 payload + embedded release manifest.
- Real data, pretrained-weight binaries, credentials, caches, and sealed-test material included: `false`.

## Residual boundary

The repaired runner has not yet been exercised on the cloud. A guarded code-only sync and one unchanged deterministic R2 replay are still required. The replay is baseline recovery, not innovation evidence. CMCD remains unauthorized until a verified best checkpoint exists and V13-D0 passes its opportunity gate.
