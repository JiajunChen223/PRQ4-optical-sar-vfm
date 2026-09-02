# CLOUD_SYNC clean-export package lane design

## Scope and decision

本记录是只读设计审查，不是同步授权，也不表示已连接或已写入云端。

结论：可以在现有 `CLOUD_SYNC` 中增加 `--release-package` 支持，但只能把它实现为 `cloud_exec.py --classification code_sync` 的一个一次性、当前回合绑定、代码包专用子路径；不能把 `release_maintenance`、`baseline_repair_code_sync` 或 `baseline_route_code_sync` 直接复用为普通云同步入口，也不能通过 SSH/SCP/远程编辑旁路覆盖。

## Existing implementation audit

已检查的本地实现：

- `C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\cloud_exec.py`
- `C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\researchpilot_guard.py`
- `C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\cloud_connection.py`
- `C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\references\tool-chain.md`

当前行为：

1. 普通 `classification=code_sync` 只执行经过 guard 的远端命令；`--release-package` 默认被拒绝。
2. `--release-package` 当前只允许三类受限 package lane：
   - `release_maintenance` / `release_binding_maintenance`：要求当前 `INNOVATION_SCREENING_SEED0`，只允许写入非活动 release，明确禁止覆盖已有目标；
   - `code_sync` / `baseline_repair_code_sync`：要求当前 `BASELINE_REPRODUCTION` 的 one-use repair control；
   - `code_sync` / `baseline_route_code_sync`：要求 BR08 当前回合审批和非活动 release 路径。
3. 这些 lanes 都不是当前 `CLOUD_SYNC`（项目当前 gate 为 `CLOUD_SYNC`、status 为 `BLOCKED`）的通用入口，因此不能无条件解决本项目的 clean-export 同步阻断。
4. `cloud_exec.py` 已有可复用的受控执行原语：预检 → 通过 SSH stdin/base64 传输包 → 远端 finalize；结果写入 `command_log.jsonl`，且 package/control 哈希可绑定。该原语不能被直接命令或 ad-hoc SCP 替代。

## Minimal compatible extension

建议新增一个普通 `cloud_sync` 的 package 子路径，保持 protected formal code-sync invariant：

```text
cloud_exec.py
  --classification code_sync
  --guard-action cloud_sync
  --release-package <verified-code-only-package>
  --command-file <exact-operation-command-snapshot>
```

其中 `--release-package` 只表示本地已审查的代码包；真正的远端写入仍必须由 `cloud_exec.py` 执行，并写入命令账本。

### Current-turn control artifact

在项目内新增一次性 control（建议路径 `02_experiment/reports/cloud_sync_package_control.json`），至少包含：

```json
{
  "artifact_type": "researchpilot_cloud_sync_package_control",
  "schema_version": "researchpilot.cloud_sync_package.v1",
  "status": "approved_for_one_code_sync_only",
  "guard_action": "cloud_sync",
  "required_current_gate": "CLOUD_SYNC",
  "required_gate_status": "PENDING",
  "required_router_checkpoint": "PLAN_APPROVAL",
  "current_turn_approval": {
    "approved_by": "user",
    "scope": "cloud_sync_code_only",
    "approval_text_sha256": "<hash-of-current-turn-authorization>"
  },
  "package": {
    "path": "<project-relative-code-only-package>",
    "sha256": "<sha256>",
    "manifest": "02_experiment/code/manifests/clean_sync_manifest.json",
    "manifest_sha256": "<sha256>"
  },
  "operation": {
    "classification": "code_sync",
    "guard_action": "cloud_sync",
    "operation_type": "one_use_clean_export_target_replace",
    "remote_effect": "code_workspace_only",
    "remote_target_path": "/root/autodl-workspace",
    "remote_staging_path": "/root/autodl-workspace.previous/.researchpilot_code_sync/<operation-id>.staging",
    "remote_backup_path": "/root/autodl-workspace.previous/<operation-id>",
    "overwrite_existing_target": true,
    "active_pointer_switch": false,
    "test_accessed": false,
    "training_started": false,
    "data_transfer": false,
    "retry_allowed": false,
    "max_executions": 1,
    "preflight_command_sha256": "<hash>",
    "command_sha256": "<hash>"
  }
}
```

The control must be hash-checked before any SSH process is constructed. The package must be resolved inside the local project, match the clean-sync manifest, and pass the existing code-only/data/credential/cache checks.

## Exact-target overwrite semantics

“覆盖云端”应解释为一次可回滚的原子替换，而不是 `rm -rf` 或无界 `rsync --delete`：

1. Preflight verifies the endpoint, exact target path, target is not a symlink, staging/backup paths are distinct, and the workspace is quiescent (no active ResearchPilot run lock or declared active pointer).
2. Upload only the package bytes to the dedicated code-workspace staging path under `/root/autodl-workspace.previous/.researchpilot_code_sync/<operation-id>.staging`; this is a transient code-only location, never the data or experiment directory `/root/autodl-tmp`.
3. Verify the uploaded archive SHA256 and allow-listed archive members.
4. Extract to a fresh staging directory and verify the extracted manifest.
5. If `/root/autodl-workspace` exists, first require a remote allow-list/manifest proving that it is code-only and contains no data, weights, caches, logs, checkpoints, or active-run outputs. Only then atomically move it to the control-bound sibling `remote_backup_path` (outside `/root/autodl-tmp`); atomically move the verified staged code tree into the exact target. If the proof is absent or any unknown/data-like entry is present, fail closed rather than moving the target. The previous tree is retained for rollback and is never synchronized back locally.
6. No active pointer, dataset root (`/root/autodl-tmp`), weights root, logs, checkpoints, or experiment output path may be switched or overwritten. The only permitted remote mutation is the code-only target replacement after the allow-list check.

The target replacement is therefore explicit and reversible. A missing target may be created; an unexpected target type, symlink, active lock, package mismatch, or non-empty staging/backup collision must fail closed.

## Required code/guard changes (design only)

No source files were changed in this review. If implemented, the change must be limited to the following:

- `researchpilot_guard.py`: recognize the new control artifact for `cloud_sync`, require current gate `CLOUD_SYNC` with `PENDING` status, current-turn approval scope `cloud_sync_code_only`, exact package/manifest hashes, one operation, `retry_allowed=false`, and no data/training/test/gate mutation.
- `cloud_exec.py`: allow `--release-package` only when `classification=code_sync` and the new `cloud_sync` control validates; reuse the existing preflight/upload/finalize pattern, but bind target, staging, backup, package, and command hashes from the control. Journal control ID, operation ID, package SHA256, manifest SHA256, target paths, and execution phase.
- Tests: add blocked tests for ordinary unbound `--release-package`, stale current-turn hash, package/manifest mismatch, wrong gate/status, symlink target, target outside `/root/autodl-workspace`, data/weights paths, retry, second execution, and active-run lock; add a synthetic command/journal test that never opens SSH.
- Gate handling: reopen the currently blocked `CLOUD_SYNC` as `PENDING` only through the canonical `advance_gate.py --reopen` path with this control as evidence. A successful sync may advance `CLOUD_SYNC` to `CLOUD_ENVIRONMENT`; it must not approve `BASELINE_TRAINING_APPROVAL` or start any experiment.

## Explicit non-goals and preserved invariants

- No direct SSH file-writing command, remote editor, or ad-hoc SCP.
- No Git remote is fabricated and no endpoint/port/key is changed.
- No real data, labels, caches, weights, credentials, logs, checkpoints, or scientific results enter the package.
- No baseline approval is inferred from the user’s sync authorization.
- No training, GPU probe, data download, environment installation, protocol lock, or innovation screening is triggered by code sync.
- The current clean package remains the only payload candidate:
  `F:\PRQ4\logs\release_packages\geotoken3path_code_r2.tar.gz`
  (SHA256 `2b12cfd39d52cb3f51e9b1240609cd3c902bf830f3a9fabdad04bf3df507fc48`).

## Final design verdict

`--release-package` in the ordinary `CLOUD_SYNC` lane is compatible with ResearchPilot only as the new one-use, current-turn-bound, exact-target atomic replacement described above. The existing special package lanes cannot be repurposed for this project’s current gate. Until the control artifact is created, the gate is reopened canonically, and the guard/control tests pass, no SSH or remote overwrite is authorized.
