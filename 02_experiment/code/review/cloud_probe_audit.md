# Cloud endpoint and probe audit

审计日期：2026-08-20（本地只读审查）  
项目根：`F:\PRQ4`  
审计范围：ResearchPilot `researchpilot-research-experiment` 的 `cloud-connection`、`cloud-runtime`、`tool-chain` 契约，以及用户当前提供的 SSH 端点。

## 结论

用户提供的端点

```text
ssh -p 28974 root@connect.nmb2.seetacloud.com
```

已绑定到 `02_experiment/cloud/cloud_connection.json`，且 `validate_user_cloud_endpoint.py` 只读校验通过。由于当前实验 gate 仍为 `CLOUD_SYNC/PENDING`，ResearchPilot guard 按契约拒绝了 `cloud_probe`；因此本次没有建立 SSH 会话、没有读取远端硬件、没有下载数据/权重，也没有训练。

## 端点绑定审计

当前绑定文件：`F:\PRQ4\02_experiment\cloud\cloud_connection.json`

已观察字段：

- `schema_version=2`
- `connection_mode=user_provided`
- `endpoint_source=user_supplied`
- `provider=user_provided_ssh`
- `host=connect.nmb2.seetacloud.com`
- `port=28974`
- `user=root`
- `remote_shell=posix`
- `cloud_data_root=/root/autodl-tmp`
- `remote_root=/root/autodl-workspace`
- `strict_host_key_checking=yes`
- `identity_file` 为空：按契约依赖用户 SSH agent 或正常 SSH 配置；未写入密码、token 或私钥。
- `connection_state=USER_ENDPOINT_CONFIGURED_PENDING_PROBE`

`known_hosts_file` 与 `host_key_fingerprint` 目前为空。由于严格主机密钥检查仍为 `yes`，实际 probe 时必须依赖用户默认 `known_hosts`/SSH 配置中的可信记录；若不存在，probe 应诚实记录主机密钥失败并停止，不得关闭检查或替换端点。

## 已执行的只读命令与结果

### 1. 用户端点 schema 校验

正确命令（Windows）：

```powershell
& 'C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\run_windows.ps1' `
  -Script 'C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\validate_user_cloud_endpoint.py' `
  -ProjectRoot 'F:\PRQ4' `
  -Arguments @('--project-root', 'F:\PRQ4')
```

本次直接使用同一项目 Python 运行脚本，结果为 `status=pass`、`network_probe_performed=false`、`exit=0`。校验输出确认的 endpoint 为同一 host/port/user，并明确下一步是对同一项目根运行 `cloud_exec.py --probe`。

### 2. probe guard 预检

```powershell
& 'C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\researchpilot_guard.py' `
  --project-root 'F:\PRQ4' --action cloud_probe
```

结果：`status=BLOCKED`，原因为 `action cloud_probe requires current gate CLOUD_ENVIRONMENT, not CLOUD_SYNC`，退出码 `3`。

### 3. 按用户端点执行标准 probe 命令

```powershell
& 'C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\run_windows.ps1' `
  -Script 'C:\Users\Administrator\.codex\skills\researchpilot-research-experiment\scripts\cloud_exec.py' `
  -ProjectRoot 'F:\PRQ4' `
  -Arguments @(
    '--project-root', 'F:\PRQ4',
    '--host', 'connect.nmb2.seetacloud.com',
    '--port', '28974',
    '--user', 'root',
    '--probe',
    '--guard-action', 'cloud_probe'
  )
```

结果：executor 在构造 SSH 前被 guard 拒绝，报错同上，退出码 `2`。这不是网络失败，也不是认证失败；本次没有发出远端命令，故不能生成 `CAPABILITY_AUDIT` 或宣称 `cloud_access=true`。

## 当前 gate/approval 状态

- `02_experiment/gate_status.json`: `current_gate=CLOUD_SYNC`, `status=PENDING`。
- `00_project/researchpilot_state.json`: 当前固定 checkpoint 为 `BASELINE_TRAINING_APPROVAL`，状态 `awaiting_user`；仅 `PLAN_APPROVAL` 已批准。
- 端点 schema 校验不需要新增用户批准；probe 必须等 gate 按顺序进入 `CLOUD_ENVIRONMENT`，并继续使用该同一端点。
- `BASELINE_TRAINING_APPROVAL` 仍未批准，不能把本次端点提供或 probe 预检解释为基线训练授权。

## 下一步的最小合法动作

1. 在 `CLOUD_SYNC` 通过既有 clean-sync 证据并按 guard 使用 `cloud_exec.py --classification code_sync --guard-action cloud_sync`；不传输数据、权重或缓存。
2. gate 进入 `CLOUD_ENVIRONMENT` 后，重新运行上面的标准 probe 命令；此时才允许一次实际 SSH capability probe，并由 `cloud_exec.py` 写入命令日志及 `CAPABILITY_AUDIT`。
3. 若 probe 成功，再进行远端硬件/软件预检。若失败，保留原 endpoint、记录精确失败类别并停止；不得换端口、改 host key 策略、切换代理或自动修复。
4. 基线正式训练仍需单独获得当前请求绑定的 `BASELINE_TRAINING_APPROVAL`，并且必须在云端审计通过后执行。

本报告不包含任何远端硬件、数据、权重、吞吐或科学结果结论。
