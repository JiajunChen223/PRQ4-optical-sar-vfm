# SEN12TS successor approval / Router binding audit

审计日期：2026-08-22  
审计范围：只读核对当前 ResearchPilot Router state、用户修正记录、Plan
successor、最新版 Plan validator 和 Experiment gate。未修改全局 Skill 或
审批脚本，未 SSH，未下载数据/权重，未探测 GPU，未训练、评估或生成指标。

## 结论

判定：**`CONDITIONAL_PASS_FOR_PROJECT_AMENDMENT_ONLY`**。

本轮用户消息已经由 Router 以项目修正记录
`AM-20260822-104751-5aacb6c3` 记录，适合作为一个有边界的
`user_project_constraint` amendment。它授权把 SEN12TS WorldCover 三地区
successor 作为当前候选核心数据范围，并只允许后续受控的元数据/shape/hash
预检；它没有产生第二个固定 `PLAN_APPROVAL`，也没有授权数据/权重下载、GPU、
训练、评估或指标。

**不需要、也不能重开 `PLAN_APPROVAL`。** 当前 Skill 的固定审批顺序只有
`PLAN_APPROVAL → FINAL_CORE_APPROVAL`；`PLAN_APPROVAL` 已经批准，Router
当前位于 `FINAL_CORE_APPROVAL / automatic`。`record_user_approval.py` 会因
expected checkpoint 不是 `PLAN_APPROVAL` 而拒绝回跳。应使用 amendment 加上
一个独立 successor binding artifact，而不是伪造或重复一个固定审批事件。

但 successor 尚未完成绑定，原因有二：

1. Router 原始 `PLAN_APPROVAL` 的 request hash 是
   `793c257572c3482968129dbb72e8e0b9413b2a035794d104fa986e792a814b0f`，对应
   保留的历史文件
   `00_project/runtime/legacy/plan_handoff_bigearthnet_20260822.json`。
   当前 `01_literature/synthesis/plan_handoff.json` 已被改写，当前 hash 为
   `5a9060898ae19b6336a09cdaee4b9f46b13b8eb35b62abe634399e269843b4c4`，两者
   不一致。不能把改写后的文件继续当作已批准 request artifact；历史批准
   必须保持原 hash，新的 SEN12TS 内容应另建 successor。
2. 对当前 canonical handoff 和 SEN12TS successor 分别运行最新版
   `validate_plan_handoff.py` 均为 `reject`。当前 canonical 文件虽标记
   `handoff_status=ready`，当前 validator 实际统计 paper-only idea pool 为 0，
   仍缺少 6–10 个 sketches，并把十候选
   fast bank 直接放入了当前 Skill 要求恰好三个的 handoff bank；
   `plan_handoff_successor_sen12ts_20260821.json` 仍是
   `blocked_pending_cloud_dataset_contract`，且缺少最新版 handoff 所需的
   路线、portfolio、compute/data budget、initialization 和 evidence 字段。

因此，用户 amendment 是有效授权，但不是对一个已通过最新版 Plan
validator 的 successor handoff 的自动认可。当前不能据此修改代码合同，也
不能据此让云端 gate 前进。

## 直接核对结果

| 项目 | 结果 | 证据 |
|---|---|---|
| Router state schema | PASS，`existing_state_mutated=false` | `init_project_state.py` |
| 当前 phase/service | `EXPERIMENT / experiment` | `00_project/researchpilot_state.json` |
| 当前 gate | `CLOUD_ENVIRONMENT / BLOCKED` | `02_experiment/gate_status.json` 最新记录 |
| 固定审批 | 仅 `PLAN_APPROVAL` 已批准 | Router `approved_checkpoints` |
| 当前固定 checkpoint | `FINAL_CORE_APPROVAL / automatic` | Router `approval_protocol` |
| 用户修正 | `ADAPTED`，record 26 | `00_project/controls/user_amendments.jsonl` |
| canonical handoff validator | REJECT：idea pool=0，fast bank=10 而非 3 | `validate_plan_handoff.py` |
| SEN12TS successor validator | REJECT：handoff blocked，缺少新版必需字段 | `validate_plan_handoff.py --handoff ...successor_sen12ts...` |
| cloud environment guard | BLOCKED：当前 gate 非 pending | `researchpilot_guard.py --action cloud_environment` |
| cloud data guard | BLOCKED：当前 gate 不是 `CLOUD_DATA_DOWNLOAD` | `researchpilot_guard.py --action cloud_data_download` |
| test seal | sealed | guard output |

## 必须生成的 project-local 证据

下一步应由 Plan/Router 在项目目录内生成并校验以下 successor 证据；不应
修改全局 Skill、审批脚本或历史批准文件：

1. **完整 successor handoff**：新文件而非覆盖历史
   `plan_handoff.json`，保留原路线、CAND-01、Pattern Recognition 方法叙事，
   将核心数据改为
   `sen12ts_worldcover_3region_1200`，并满足最新版 handoff schema：
   `handoff_status=ready`、3–5 portfolio、6–10 paper-only sketches、恰好
   3 个 fast-screening candidates、compute/data budget、initialization policy、
   冻结 packet/library 和独立多智能体完整性证据。
2. **successor binding record**：记录 amendment ID、用户原文 SHA、原批准
   request SHA、successor handoff SHA、数据集 ID、只允许的预检范围，以及
   明确的禁止项。它是项目证据，不是新的固定审批 checkpoint。
3. **Router child reference 更新记录**：仅在 successor handoff 通过当前
   validator 后，才用 Router-owned state update 将 `child_state_refs.plan`
   指向 successor；不要把一个 `blocked` 或 validator `reject` 的文件标成
   当前正式 Plan。
4. **gate repair/reopen evidence**：当前不生成、不重开。待 successor
   handoff 通过并完成 SEN12TS 代码合同的本地修复/测试/独立 review 后，若
   代码路径受影响，应以 route-replacement evidence 通过普通 failure-recovery
   路径回到最早受影响的 `CORE_CODE`，依次经过 `LOCAL_REVIEW`、`CLOUD_SYNC`、
   `CLOUD_ENVIRONMENT`；不能直接从当前 blocked environment 跳到数据 gate。
   若未来仅对现有 environment blocker 做一个已绑定的 successor 审计，才可
   用独立证据文件配合 `advance_gate.py --reopen` 将同一
   `CLOUD_ENVIRONMENT` 重开为 `PENDING`。

## 安全执行顺序

1. 保留原始 BigEarthNet handoff 及其 `793c2575...` hash，只把它视为历史
   approved request；不要以当前改写后的 canonical 文件通过 guard。
2. 完成并验证 SEN12TS successor handoff 和 successor binding record。
3. 本地修复 SEN12TS 数据合同：11 类标签、S1 `[1,0] → [VV,VH]`、S2
   `[0..11]`、parent-first split、nodata/finite 检查、CROMA 动态归一化和
   4 路 depth-tap 证据；完成 synthetic parity、代码 review 和 clean package。
4. 以 route-replacement evidence 回到最早受影响的 canonical Experiment gate，
   再走受 guard 保护的 code-only sync，并重做 CROMA 环境兼容性审计。
5. 只有 `CLOUD_ENVIRONMENT` 通过并进入 `CLOUD_DATA_DOWNLOAD/PENDING` 后，
   才能执行本用户消息授权的 SEN12TS 元数据/shape/hash 预检；预检仍须云端
   运行、无本地二进制落盘、无 sealed-test、无权重、无 GPU、无训练/评估/指标。
6. 预检结果不能自动视为数据合同或科学结果通过；license/attribution、完整
   manifest/hash、shape/dtype/CRS/nodata、parent-first split、存储账本和
   preprocessing parity 仍要分别闭合，之后才考虑数据获取和协议锁定。

## 当前不应执行的动作

- 不调用 `set_approval_checkpoint.py` 将 checkpoint 回退到 `PLAN_APPROVAL`。
- 不调用 `record_user_approval.py --checkpoint PLAN_APPROVAL` 重播用户批准。
- 不把当前 validator `reject` 的 handoff 当作已批准 successor。
- 不重放已消费的 CROMA 环境审计，不直接开 `CLOUD_DATA_DOWNLOAD`。
- 不修改全局 `C:\Users\Administrator\.codex\skills` 下的任何文件。

## Evidence hashes（审计时点）

- Router state：`539954ab4178d05b5073f39abd4b80a43c49b412b6c091a05c3cea8d657658c8`
- amendment ledger：`336dc1607cc7f9f344dc4d13d085423a3b5e722e574a455e18fbf84e33ea31d0`
- 保留历史批准 handoff：`793c257572c3482968129dbb72e8e0b9413b2a035794d104fa986e792a814b0f`
- 当前改写 handoff：`5a9060898ae19b6336a09cdaee4b9f46b13b8eb35b62abe634399e269843b4c4`
- SEN12TS successor：`401fb6c0e0a627c288645422af33ee84f85b07f488c7e6513b93db659f496d80`
- gate status：`59da910fecb62e199c6184b559a81e00ea0c279a50636db7038a746998622c6f`

审计决定：**无需重开 PLAN_APPROVAL；当前仅有 amendment 授权，successor
binding 和 Plan validator 仍待完成；Experiment gate 保持阻断。**
