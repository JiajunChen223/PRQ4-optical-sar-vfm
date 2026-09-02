# Final package audit — R5b — 2026-08-22

## Scope

Independent read-only audit of:

- `02_experiment/artifacts/geotoken3path_code_r5b.tar.gz`
- `02_experiment/code/manifests/clean_sync_manifest.json`
- existing cloud-sync package-control inventory

No SSH, remote access, download, GPU probe, data read, training, evaluation,
or gate mutation was performed. Only this report was written.

## Integrity and allowlist checks

| Check | Result |
|---|---|
| Clean manifest status | PASS (`status=pass`, `file_count=42`) |
| Current manifest SHA-256 | PASS — `b55f21364189558cdd1162351db704552883288594771d2d83e4a1f1ed15bbd7` |
| Manifest flags | PASS — local real data, pretrained binaries, credentials, and cache artifacts all `false`; GPU probe `forbidden_not_run` |
| Local manifest entries | PASS — 42/42 paths present; byte counts and SHA-256 values all match |
| R5b archive SHA-256 | PASS — `b12d8003a9c4c11a73a97e85ed7a6b0533924c92360c5b6d9a21cf41a3b8d448` |
| R5b archive size | PASS — 37,319 bytes |
| Archive members | PASS — 43 unique regular members: 42 payload files plus `researchpilot_code_release_manifest.json` |
| Archive allowlist | PASS — payload member set exactly equals the 42 manifest paths; no absolute path, `..` traversal, symlink, non-regular member, artifact-root data/weight/checkpoint/cache/output/log member, or data/weight binary suffix |
| Internal release manifest | PASS — 42 entries exactly equal the clean manifest; source manifest SHA is `b55f21364189558cdd1162351db704552883288594771d2d83e4a1f1ed15bbd7` |
| Credential/secret markers | PASS — no private-key, common secret, credential assignment, Hugging Face token, or SEN12TS cloud-root marker in archive contents |

## Validator

`validate_code_project.py --project-root F:\PRQ4` returned exit code 0:

- status `pass`
- 39 executable/config files scanned
- 0 problems and 0 violations
- local real data allowed `false`
- local GPU probe `forbidden_not_run`

## Package-control readiness

Overall release readiness is **BLOCK** despite the package itself being
integrity-valid.

No package-control JSON currently binds the R5b package path or its package
SHA. The latest visible controls still bind the older R3 package:

- `cloud_sync_package_control_PRQ4-CLOUDSYNC-20260822-R4.json`
  - package: `logs/release_packages/geotoken3path_code_r3.tar.gz`
  - package SHA: `e7ea80daa0f24d67eb6511d02463cd783add607b05504feb4ba82ffdf6d4947e`
  - manifest SHA: `77b3a0aab8977d5c6fae5f5158a297bc263d6463bcc9e1105388513fa5f2e97d`
  - control SHA at audit time: `10aead60e0a53bc525e62869cb21a8cde2624883a1fe86e0ca0071e5bf8b0c59`

Therefore R5b is not yet an executable one-use controlled sync artifact. A
new control must bind the R5b package path, package hash, manifest hash,
operation id, command hash, and the applicable pending gate before any sync.

The read-only Guard check also returned exit code 3 with:
`current gate must be CLOUD_SYNC`, `gate-rebind requires the canonical gate to
remain PENDING`, and `generic action requires a pending canonical gate`.
Observed state was `CLOUD_ENVIRONMENT/BLOCKED`, with the test seal `sealed`.

## Final decision

**BLOCK for cloud synchronization; PASS for local package integrity.** The
R5b package is clean and reproducibly hash-bound to the current manifest, but
there is no matching package-control and the current canonical gate does not
authorize `cloud_sync`. This audit establishes no remote filesystem state.

