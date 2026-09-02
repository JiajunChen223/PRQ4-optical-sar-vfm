# Final clean code package audit — 2026-08-22 r1

## Scope and boundary

This is an independent, read-only local audit of the proposed R4 code-only
sync package. It did not use SSH, inspect a remote host, probe a GPU, read real
data, access weights, train, evaluate, or modify any project artifact other
than this report.

Audited artifacts:

- `logs/release_packages/geotoken3path_code_r3.tar.gz`
- `02_experiment/code/manifests/clean_sync_manifest.json`
- `02_experiment/reports/cloud_sync_package_control_PRQ4-CLOUDSYNC-20260822-R4.json`
- `02_experiment/code/review/CODE_REPORT.json`

## Results

| Check | Result | Evidence |
|---|---|---|
| JSON parse and required artifact presence | PASS | All four audited JSON/files were present and parseable where applicable. |
| Clean manifest flags | PASS | `status=pass`; real data, pretrained binaries, credentials, and cache artifacts are all `false`; GPU probe is `forbidden_not_run`. |
| Manifest-to-local source hashes and byte counts | PASS | 42/42 entries matched local files exactly; no missing, byte-count, or SHA-256 mismatch. |
| Package SHA-256 and size | PASS | 34,319 bytes; SHA-256 `e7ea80daa0f24d67eb6511d02463cd783add607b05504feb4ba82ffdf6d4947e`. |
| Manifest SHA-256 | PASS | `77b3a0aab8977d5c6fae5f5158a297bc263d6463bcc9e1105388513fa5f2e97d`. |
| Archive allowlist and traversal check | PASS | 43 unique regular members (42 payload files plus the internal release manifest); no absolute paths, `..` traversal, symlinks, artifact-root data/weights/checkpoint/cache/output members, or binary data/weight suffixes. The `src/geotoken3path/data/` entries are source code, not dataset payloads. |
| Internal release manifest | PASS | Its 42 file entries exactly equal the clean manifest entries and its source manifest hash equals `77b3a0aab8977d5c6fae5f5158a297bc263d6463bcc9e1105388513fa5f2e97d`. |
| Credential/data-string scan over manifest files | PASS | No private-key marker, common secret marker, credential assignment, Hugging Face token, or SEN12TS cloud-root string was found. |
| Code validator | PASS | `validate_code_project.py --project-root F:\PRQ4`; exit 0; 39 executable/config files scanned; 0 problems and 0 violations; local real-data allowed `false`; GPU probe `forbidden_not_run`. |
| CODE_REPORT consistency | PASS / conditional handoff | `status=PASS`; local synthetic code service is reported PASS, while its cloud handoff is explicitly `BLOCKED_CURRENT_GATE_CLOUD_DATA_DOWNLOAD`. |
| R4 Guard | BLOCKED as expected | `researchpilot_guard.py --action cloud_sync` exited 3 with `current gate must be CLOUD_SYNC` and `generic action requires a pending canonical gate`; observed canonical state is `current_gate=CLOUD_DATA_DOWNLOAD`, `status=BLOCKED`, test seal `sealed`. No remote action was attempted. |
| R4 command-log evidence | PASS for no execution | `02_experiment/cloud/commands/command_log.jsonl` contains zero occurrences of `PRQ4-CLOUDSYNC-20260822-R4`, `geotoken3path_code_r3`, or `CLOUDSYNC-20260822-R4`. |

## Control and hash binding

The R4 control declares one-use code-only synchronization, no data transfer,
no training/evaluation, and remote target `/root/autodl-workspace`. Its control
file SHA-256 is
`ee727e266ebcc9d91dacb5dcd0f77d034a4b2ad1cd342a230ca2d43dc5f707c8`.
The control's package and manifest hashes match the independently computed
hashes above. The control itself requires `CLOUD_SYNC/PENDING`; the canonical
gate is instead `CLOUD_DATA_DOWNLOAD/BLOCKED`, so the package is locally
integrity-valid but not currently executable under the Guard.

`CODE_REPORT.json` SHA-256 at audit time:
`f5aba9904598ccc3c0f1b4efcf6f5eb21836da5debeb2d3ef031ddfc2ce5f1b3`.

## Independent conclusion

The R3 archive is a clean, internally consistent, code-only package and the
local validator passes. There is no evidence of an R4 cloud synchronization
attempt. The only observed release blocker in this audit is the canonical
ResearchPilot gate mismatch/block (`CLOUD_DATA_DOWNLOAD/BLOCKED` versus the
control's required `CLOUD_SYNC/PENDING`). This report does not establish any
remote filesystem state; remote verification remains pending a separately
Guard-authorized sync.

