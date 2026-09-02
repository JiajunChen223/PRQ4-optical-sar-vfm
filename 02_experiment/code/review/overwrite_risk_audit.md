# Cloud code overwrite risk audit

## Scope and evidence boundary

This is a read-only local audit for the user request to overwrite the cloud
code workspace with the reviewed code package. No SSH probe, upload, remote
shell, or remote file mutation was executed. The audit is bound to the
current project state and the files listed below:

- Project root: `F:\\PRQ4`
- Endpoint binding: `F:\\PRQ4\\02_experiment\\cloud\\cloud_connection.json`
- Gate state: `F:\\PRQ4\\02_experiment\\gate_status.json`
- Code report: `F:\\PRQ4\\02_experiment\\code\\review\\CODE_REPORT.json`
- Clean-sync manifest: `F:\\PRQ4\\02_experiment\\code\\manifests\\clean_sync_manifest.json`
- Release report: `F:\\PRQ4\\02_experiment\\code\\review\\code_release_report.json`
- Release package: `F:\\PRQ4\\logs\\release_packages\\geotoken3path_code_r2.tar.gz`
- Current cloud blocker: `F:\\PRQ4\\02_experiment\\cloud\\cloud_sync_blocker.json`

## User-amendment classification

The user has explicitly authorized direct replacement of the remote code
workspace. That is a compatible project-level intent for the *scope* of the
sync, but it does not waive protected ResearchPilot invariants. The formal
cloud-code mutation still has to be produced by the local reviewed code path
and delivered through `cloud_exec.py --classification code_sync` (or another
currently validated, package-bound code-sync operation). An arbitrary SSH
`scp`, remote editor, `cat >`, `rsync`, or shell overwrite would be an
unreviewed mutation and remains out of scope even with user authorization.

The user request also does not constitute `BASELINE_TRAINING_APPROVAL`.
Therefore, a successful code replacement must not launch training, download
data, download weights, inspect the GPU, open the test split, or change the
approval checkpoint.

## Endpoint and remote path boundary

The bound endpoint is `root@connect.nmb2.seetacloud.com:28974`. The project
binding declares:

- `remote_root=/root/autodl-workspace`: remote code/shell workspace only;
- `cloud_data_root=/root/autodl-tmp`: separate real-data, cache, checkpoint,
  and pretrained-weight location.

The path contract forbids placing real data, labels, caches, raw archives,
pretrained binaries, checkpoints, or experiment outputs in
`/root/autodl-workspace`. A safe overwrite therefore means replacing only the
reviewed code tree beneath the declared `remote_root`, with an explicit
preflight that records the target and verifies containment. It must not use
`rm -rf /root`, `/root/autodl-tmp`, or any broad parent; it must not follow a
symlink outside the code workspace; and it must not delete or move existing
data/weights/checkpoints in the data root.

## Local package contents and contamination check

The clean-sync manifest is `status=pass` with 42 files and declares:

- `local_real_data_included=false`
- `pretrained_weight_binaries_included=false`
- `credentials_included=false`
- `cache_artifacts_included=false`
- `local_gpu_probe=forbidden_not_run`

The release report further records `payload_scope=reviewed_code_configs_tests_manifests_only`,
`scientific_results_included=false`, and the same zero-data/zero-weights/
zero-credentials claims. The package is outside `02_experiment` because the
local policy scanner treats archives in that tree as data-like payloads; this
relocation does not add any data to the package. SHA256 recorded locally:

`2b12cfd39d52cb3f51e9b1240609cd3c902bf830f3a9fabdad04bf3df507fc48`

The package is suitable as a code-only transfer candidate, subject to the
remote preflight and an exact package hash check after transfer.

## Guard and overwrite risks

The local `cloud_exec.py` implementation has a dedicated package-sync path
that stages a base64-encoded package through SSH stdin and then runs an exact,
control-bound finalize command. It journals the preflight, upload, finalize
result, remote root, operation ID, package hash, and command hash. It does not
provide a generic upload or arbitrary overwrite API. The current project has
no active release/code-sync control artifact and its `CLOUD_SYNC` gate is
`BLOCKED`; therefore the package-sync operation is not currently authorized
by a PASS control card.

The highest risks if an unbound overwrite were attempted are:

1. **Data destruction:** a broad extraction or cleanup command could delete
   `/root/autodl-tmp` data, labels, caches, or checkpoints.
2. **Evidence destruction:** replacing an active workspace could remove prior
   run manifests, logs, and accepted evidence, violating the no-overwrite
   rule for accepted runs.
3. **Path escape:** archive entries or symlinks could write outside
   `/root/autodl-workspace`.
4. **Provenance loss:** without the clean-sync manifest and package SHA256,
   the remote tree would not be traceable to the reviewed local code.
5. **Approval bypass:** code upload followed by a shell command could
   accidentally become data download, GPU probe, or training before the
   corresponding guard/checkpoint is passed.

## Required safe operation contract

Before any remote write, the Experiment service must create or validate a
current package-bound code-sync control card and obtain `researchpilot_guard.py
--action cloud_sync` with exit code 0. The control card must bind:

- the exact clean-sync manifest path and SHA256;
- the exact release package path, byte count, and SHA256;
- the endpoint and `remote_root`;
- a non-data staging path outside `cloud_data_root`;
- a preflight command that is read-only and verifies target containment,
  existing data-root separation, and available disk space;
- a finalize command that atomically installs only the package's reviewed code
  paths, preserves data/weights/checkpoints/logs, rejects unsafe archive
  members, and verifies the resulting manifest/hash.

The operation must be one-shot and journaled. If preflight, upload, hash
verification, or finalization fails, stop and preserve the remote state; do
not retry with a different command or broaden the target. After successful
sync, run only the normal cloud environment probe through `cloud_exec.py`.
The probe may inspect cloud hardware, but it must not download data/weights or
train until the later gates and approvals are satisfied.

## Verdict

`CONDITIONAL_PASS_FOR_CODE_ONLY_SYNC_PREPARATION`.

The reviewed package is clean and the declared remote code/data roots are
separable. Direct overwrite is safe only when implemented as a guarded,
manifest-bound `code_sync` operation that preserves `/root/autodl-tmp` and
all existing cloud evidence. The current local state is **not yet authorized
for execution** because `CLOUD_SYNC` has no PASS control card and no cloud
probe has been performed. This audit neither authorizes SSH nor claims that
the remote workspace has been changed.

