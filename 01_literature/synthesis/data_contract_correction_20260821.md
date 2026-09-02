# PRQ4 数据契约修正（PLAN successor，2026-08-21）

## 判定

用户选择了“保留光学+SAR 稠密语义分割并替换 BigEarthNet 核心数据”。该选择已被记录为项目修正，但当前不能进入下载或代码改造执行：没有候选同时完成许可、可复核 split、CROMA 输入兼容和 50GB 账本四项闭环。

## 保留与删除

- 保留：`R-EO-TRI-FUSE-01`、GeoToken-3Path、一个光学+SAR land-cover segmentation 主任务、受控空间配准误差主维度。
- 删除：BigEarthNet 19 类场景级 multilabel 作为核心数据；删除没有地理元数据支撑的 cross-region 主张。
- 条件核心：Copernicus-Bench 派生的 DFC2020 S1/S2；源数据是 S2 13 bands + S1 VV/VH、256×256 parent、8 个有效类加 ignore=255。CROMA 兼容只能通过审计后的 B10 删除与 120×120 derived crop/stitch 实现。

## 硬阻断

1. IEEE GRSS DFC2020 原始条款要求论文使用经 IADF/TUM 个案批准；镜像的 CC BY-4.0 表格不是原始权利链的替代证明。
2. 派生 benchmark 无 geolocation/time metadata；3156/986/986 不能自动解释为跨区域 holdout。
3. 父场景级 split、8 类映射、裁块不泄漏、实际解包账本和 test seal 均尚未验证。

## 可审计来源

- [IEEE GRSS DFC2020 官方页](https://www.grss-ieee.org/community/technical-committees/2020-ieee-grss-data-fusion-contest/)
- [Copernicus-Bench 数据卡](https://huggingface.co/datasets/wangyi111/Copernicus-Bench)
- [CROMA 官方仓库](https://github.com/antofuller/CROMA)
- [AI4LCC / MultiSenGE 官方元数据](https://doi.theia.data-terra.org/ai4lcc/?lang=en)（完整 S1/S2 资产超过 50GB，排除）

## 当前后续

保持 `CLOUD_DATA_DOWNLOAD=BLOCKED`。只有在书面许可或明确再许可、parent split/hash、标签/通道映射和 `<45GB` 实际 ledger 全部闭合后，才可由 Router 生成 ready successor、改代码、重做本地审查并重新走 guarded code-sync。历史 handoff 与原始阻断证据不覆盖、不删除。

独立证据包：

- `logs/agent_runs/prq4-data-correction-20260821/dense_dataset_scout/report.md`
- `logs/agent_runs/prq4-data-correction-20260821/license_split_scout/report.md`
- `logs/agent_runs/prq4-data-correction-20260821/protocol_dataset_critic/report.md`
