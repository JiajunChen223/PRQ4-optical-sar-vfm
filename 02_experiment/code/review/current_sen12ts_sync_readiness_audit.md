# Current SEN12TS code-sync readiness audit

Date: 2026-08-22 (Asia/Shanghai)

Scope: read-only audit of the local code handoff under
`F:\PRQ4\02_experiment\code`. No Skill, source, gate, cloud state, SSH,
upload/download, data/pixel, weight, GPU, or training state was changed.

## Decision

`LOCAL_CODE_SYNC_READINESS=PASS_FOR_REPACKAGING`

`CURRENT_RELEASE_PACKAGE=STALE_BLOCKED_FOR_SYNC`

The source tree is ready for a new clean code-only package, but the existing
r5b package and its one-use control are not bound to the current source. Do not
replay r5b or use any older r3/r2 control.

## Read-only evidence

| Item | Result |
|---|---|
| `review/CODE_REPORT.json` | Current local report says PASS; manifest SHA `8611489c20f80c0ceb02f1272433546c11ca4aed0ed061ef53810c24a7845fde`; 46 files; 123 tests; validator 43 files/0 violations. |
| Current `manifests/clean_sync_manifest.json` | PASS; 46 listed entries; every path exists and every byte count/SHA256 matches. Flags for real data, pretrained binaries, credentials, caches are false; GPU probe is `forbidden_not_run`. |
| Fresh validator rerun | PASS; 43 executable/config files, 0 problems, 0 violations; local real-data false; GPU probe not run. |
| Fresh no-cache pytest rerun | `123 passed in 3.89s`; `PYTHONDONTWRITEBYTECODE=1`, no cache provider. |
| r5b archive | SHA `b12d8003a9c4c11a73a97e85ed7a6b0533924c92360c5b6d9a21cf41a3b8d448`, 42 payload files. Its internal source manifest SHA is `b55f21364189558cdd1162351db704552883288594771d2d83e4a1f1ed15bbd7`, not the current SHA. |
| r5b vs current source | 4 current files absent from r5b: `configs/benchmarks/dataset_registry.yaml`, `configs/benchmarks/sen12ts_worldcover.yaml`, `src/geotoken3path/data/preprocessing.py`, `tests/unit/test_croma_preprocessing.py`; 8 common files have changed SHA256. |
| r5b control | `cloud_sync_package_control_PRQ4-CLOUDSYNC-20260822-R5B.json` binds the old r5b package and old `b55f...` manifest; it is therefore stale for the current tree. |
| Canonical gate | `CLOUD_DATA_DOWNLOAD/BLOCKED`; code sync requires a fresh `CLOUD_SYNC/PENDING` binding and one-use control. |

## Exact coordinator actions

1. Freeze the current 46-entry manifest and its SHA
   `8611489c20f80c0ceb02f1272433546c11ca4aed0ed061ef53810c24a7845fde`.
2. Rebuild a new code-only archive from that manifest. Include only the
   manifest's 46 files plus the archive's internal release manifest; exclude
   `review/`, `outputs/`, `__pycache__/`, `.pytest_cache/`, data, weights,
   checkpoints, credentials, and logs.
3. Recompute archive SHA256 and write a new release report and independent
   package audit. The new report must bind the current manifest SHA, 46-file
   count, archive SHA, and the fresh 123-test/43-file-validator evidence.
4. Create a new one-use `cloud_sync` control card binding the new archive path,
   archive SHA, current manifest path/SHA, operation ID, command hash,
   `/root/autodl-workspace` target, backup/staging paths, and the code-only
   safety flags. Retire/leave r5b controls unused; do not mutate them.
5. Through the Router/gate repair path, reopen only the canonical
   `CLOUD_SYNC` gate as `PENDING`; then run the guard and exactly one
   `code_sync` operation. Do not bypass the gate, use ad-hoc SSH/SCP, or infer
   data/training readiness from a code-sync PASS.

Until actions 1–5 are completed, handoff remains
`BLOCKED_FOR_CLOUD_SYNC_CONTROL_CLOSURE`; local code itself is not the
blocker. Scientific/data gates remain unchanged and blocked.

