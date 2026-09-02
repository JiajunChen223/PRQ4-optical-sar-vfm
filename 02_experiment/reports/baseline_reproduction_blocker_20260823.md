# 基线复现阻塞记录（2026-08-23）

## 结论

`BASELINE_REPRODUCTION` 不能安全启动，状态为 `BLOCKED_FOR_RANDOM_INIT_PROTOCOL_REBIND`。这不是基线科学结果，也不是数据或模型失败。

本轮已完成 random-init 泄漏规避接口：SEN12TS lazy loader、CROMA random constructor contract、formal train/validation skeleton、原子 checkpoint/run-result 输出和固定 16 样本 validation padding。当前本地 code-service 合同通过，但尚未云端同步。

## 直接证据

- 当前 CROMA 审计已确认 SSL4EO 中心点分别落入 Ethiopia、Uganda、Sumatra；因此 CROMA 权重被 `leakage_blocked`，不再作为初始化。
- `pretrained_alternative_search_20260823.json` 记录了 random-init 例外、候选搜索、重叠证据和资源副作用全 false 字段。
- 本地 clean-sync manifest 当前为 51 files，137 项测试通过，validator 48 files/0 violations；这些是代码合同证据，不是科学结果。
- random constructor 必须显式接受 `pretrained=False` 或 `weights=None`；未完成云端构造器兼容性审计前不得运行 baseline。

## 保护边界

本记录未下载数据或权重、未访问 sealed test、未启用 GPU、未训练，也未修改全局 Skill、guard 或 cloud_exec。当前基线文献门的通过只冻结了可接受参考带，不产生 SEN12TS 指标。

## 最小修复路径

1. 重新生成 random-init 代码包并完成独立 code review。
2. guarded code-only sync 到云端，做 random constructor 的 weights-disabled preflight，不下载权重。
3. 重新锁定 protocol 和 baseline request，验证 baseline/candidate 初始化 parity。
4. 仅在上述证据通过后生成一次性 baseline seed-0 control；输出 checkpoint、validation curve、run manifest、metrics，并在同一 `T_rapid` 保存参考值。

## 当前下一动作

保持实验 Gate 在 `BASELINE_REPRODUCTION/BLOCKED`，先完成 random-init constructor 的云端兼容性审查与同步准备；不以 synthetic smoke、云端旧进程或远端旧代码替代正式基线证据。当前本地 `validate_code_project` 已 PASS，no-cache synthetic pytest 为 137 passed，但这只证明代码合同，不解除正式基线阻塞。
