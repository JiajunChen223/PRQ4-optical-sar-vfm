# 10_CURRENT — 现代有效区（唯一工作区）

两区制重组于 2026-09-02 完成。本区只包含**唯一一份最新代码**与**全部经过验证的结论性证据**；所有被否定的方法、候选、数值与旧产物均已隔离至 `../20_HISTORY/`（只读归档，零删除，可随时反查恢复）。

## 目录导航

| 路径 | 内容 |
|---|---|
| `00_project/` | 项目元数据：`project.json`、`researchpilot_state.json`（门禁与审批状态）、`runtime.json`、`controls/` |
| `01_literature/library/` | 冻结文献库（literature.jsonl/csv/sqlite + manifest + 核心证据批量） |
| `01_literature/synthesis/` | 保留项：`dataset_registry*`、`sen12ts_*` 数据证据 8 份、`frozen_evidence_packet.json/.md`（v19 冻结证据包）、`v19_ctsp_target_venue_status`（目标期刊状态）、`targeted_evidence/` |
| `02_experiment/code/` | **唯一一份最新代码**（u2026-09-02 机制净化后：仅基线 always_fuse 一条机制，pytest 135 通过） |
| `02_experiment/protocol/` | 生效协议 `experiment_protocol.yaml` |
| `02_experiment/claims/` | 声明台账 `claim_experiment_ledger.jsonl`（v14–v19 全部路线 rejection 记录，证据链） |
| `02_experiment/reports/` | 50 份结论性收据：32 份基线证据 + 18 份 v19 门禁判定收据 |
| `02_experiment/gate_status.json` / `experiment_manifest.json` | 门禁状态 / 实验清单 |
| `02_experiment/cloud/` | `cloud_connection.json` 等状态文件（历史云命令在 20_HISTORY） |
| `03_writing/` | 论文写作区（当前为空，尚未动笔） |

## 当前科学状态（2026-09-02）

- **门禁**：`INNOVATION_REVIEW` = BLOCKED；`next_action = request_new_plan_revision_or_user_route_decision`（等待新路线决策）。
- **基线（verified R2 deterministic replay，唯一有效对照）**：best mIoU **49.7807879%**（epoch 18）、OA 77.2274%、rare macro IoU 38.7029%、epoch-24 49.7661%。
- **v5–v19 全部创新路线均被拒**（<+1pp 预注册关闭线）：最新 v19 CTSP-01 best 50.7031%（+0.9223pp）→ route closed；全部 rejection 记录见 claims 台账。
- **测试集**：自始至终 sealed 未访问；所有候选的 controls、多种子、扩展集、final test 全部锁定。

## 代码净化声明（2026-09-02）

- 唯一机制：`always_fuse`（`VALID_MECHANISMS={"always_fuse"}`）；被否机制类/模块/配置/测试全部移入 `20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/`（9 个机制模块、4 个 models adapter、19 个被否测试、11 个全量原文件副本、被否路由池 `rejected_routers_pool.py`、D0–D3 诊断 7 脚本 + 4 测试）。
- 唯一保留的机制名残迹：`GeoToken3PathFusion` 内 `ceak_*` 等**模型图容量参数**（被否机制的共享表面权重）。它们不参与任何 forward 路径（零引用），但因与已验证基线 checkpoint 的 state-dict 绑定而**不能改名或删除**（改名会破坏云端正式权重的 strict 加载）。此保留仅为状态兼容，不是方法残留。
- 工作树 sha256：净化为 `1beec648…`（deepseekpro_pr 同款历史快照关系：正式 run 的可复现锚点见 20_HISTORY 说明）。
- 历史 full_original 副本（净化前的完整源文件）全部保留在 rejected_mechanisms_20260902/，可随时恢复。

## 使用指引

- 阅读当前状态：本文档 + `02_experiment/gate_status.json`。
- 查找历史内容：`../20_HISTORY/README.md`（含全部归档索引）。
- 恢复任何隔离内容：按 `../REORGANIZATION_MANIFEST_PRQ4.json` 的源→目标对照复制回即可（零删除原则）。