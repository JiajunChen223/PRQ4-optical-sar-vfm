# 10_CURRENT — 现代区资产清单（2026-09-02 两区制重组 + 机制净化后）

按用户三原则核对：**代码唯一一份 / 全部经过验证 / 零被否方法残迹**。

## 1. 代码（唯一工作树，pytest 135 passed）

| 路径 | 内容 | 验证状态 |
|---|---|---|
| `code/src/geotoken3path/models/fusion.py`（755 行） | GeoToken3PathFusion + OpticalSarTokenModel，机制白名单 `{always_fuse}`；被否分支/常量/方法全清 | 单测 + smoke 覆盖；图内 `ceak_*` 等容量参数为 state-dict 兼容保留（零 forward 引用） |
| `code/src/geotoken3path/models/croma_bridge.py` | CROMA 基线路桥（backbone/depth tap/segmentation 入口），9 个被否 adapter 已删 | 单测覆盖 |
| `code/src/geotoken3path/models/factory.py` | build_model / build_vfm_segmentation_model（纯基线） | 单测覆盖 |
| `code/src/geotoken3path/utils/config.py`（422 行） | resolve_approved_config 基线解析 + 契约校验；v5–v19 resolver/机制校验全清 | run_manifest/config 单测覆盖 |
| `code/src/geotoken3path/engine/formal_runner.py`（723 行） | 云训练/评测 runner（基线行） | smoke + 单测覆盖 |
| `code/src/geotoken3path/utils/run_manifest.py`（311 行） | run manifest 构造/校验（`_APPROVED_MECHANISMS={"always_fuse"}`） | 单测覆盖 |
| `code/scripts/train.py` / `evaluate.py` | 训练/评测入口（重写为纯基线） | smoke 单测覆盖 |
| `code/configs/` | `model/geotoken3path.yaml`（mechanism=geotoken_3path，架构名，非路线候选）、`experiment/approved_route.yaml`（PRQ4-BASELINE-VERIFIED-R2）、`benchmarks/sen12ts_worldcover.yaml`、`runtime/3090_plan.yaml`、`model/initialization.yaml` | 全部为重建/验证过的基线契约 |

## 2. 协议与证据

- `protocol/experiment_protocol.yaml`：生效协议。
- `reports/` **50 份**：32 份基线证据（P0R1–R2 系：best/last/rapid6 审计、比较、等价性、启动/完成收据）+ 18 份 v19 门禁判定收据（route closure、formal result、screening 判定、effect evidence、claim record 等——支撑 gate_status 的 BLOCKED 判定）。
- `runs/`：v14–v19 全部 run manifest 已隔离至 HISTORY（05_runs_history）。
- `claims/claim_experiment_ledger.jsonl`：v14–v19 全部 rejected 记录（数值与依据，供论文"路线排除"章节引用）。
- `gate_status.json`（INNOVATION_REVIEW BLOCKED）+ `experiment_manifest.json`：权威状态。

## 3. 文献

- `01_literature/library/`：冻结文献库（jsonl/csv/sqlite + manifest + core evidence）。
- `01_literature/synthesis/`：dataset_registry、sen12ts 数据证据 8 份、frozen_evidence_packet（v19 冻结证据）、target_venue_status、targeted_evidence/。
- 检索过程/论文全文提取/旧版库：`20_HISTORY/06_literature_process/`。

## 4. 隔离内容索引（不在本区）

- 全部被否机制（代码/配置/测试/诊断）：`20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/`
- 全部被否路线计划与收据：`20_HISTORY/01_rejected_routes/`、`04_superseded_reports/`
- 版本管理脚本/过程文档/缓存：`20_HISTORY/09_root_scraps/`
- 正式复现锚点：被否机制的逐字节原文件 = `rejected_mechanisms_20260902/*.full_original`

## 5. 当前状态

- 门禁 INNOVATION_REVIEW BLOCKED；下一步 = 新路线计划（plan revision）或用户路线决策。
- 基线 mIoU 49.7807879%（唯一有效对照）；v5–v19 创新路线全部 <+1pp 关闭；测试集 sealed 未访问。
- 03_writing 为空（论文未动笔）。

## 6. 版本控制

- git 仓库已初始化（首个 commit `7b9f2ad`，385 文件）；HISTORY 区不在仓库内，保持只读归档。
- 全部移动有源→目标清单（`../REORGANIZATION_MANIFEST_PRQ4.json`，零删除）。