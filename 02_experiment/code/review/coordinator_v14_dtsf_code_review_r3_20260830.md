# V14-DTSF code review r3

- Review mode: coordinator adversarial repair review; no independent reviewer was available in this turn.
- Scope: repair after the DTSF-01 seed-0 run stopped during run-manifest finalization.
- Root cause verified: `run_manifest.py` omitted `DTSF-01` and `DTSF-C2/C3/C4` from the approved formal-direction allowlist while the DTSF model factory and formal runner already registered those directions.
- Repair: the reviewed local allowlist now includes `DTSF-01`, `DTSF-C2`, `DTSF-C3` and `DTSF-C4`; no model, data, optimizer, trainability, epoch, or test-seal variable changed.
- Regression protection: a synthetic unit test builds a screening manifest for `DTSF-01`; the full local suite passes 329 tests with one pre-existing warning and the code validator reports zero violations across 136 executable/config files.
- Scientific boundary: the failed cloud run produced no final metrics, best/final checkpoint, or scientific support; its artifacts remain append-only. The repaired package must be synchronized before creating a new run ID.
- Packaging: r5 clean-sync/release manifests are route-bound to `R-EO-DTSF-V14-01 / DTSF-01`, contain code/config/tests only, and exclude data, weights, checkpoints, caches and credentials.
- Decision: PASS for guarded r5 code-only synchronization and a fresh DTSF-01 seed-0 full-horizon launch after reattachment of the unchanged environment/data/protocol evidence.
