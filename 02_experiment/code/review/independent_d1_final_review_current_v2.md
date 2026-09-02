# D1-A/B current final review v2

## Scope

本复审针对当前 `PLAN_REVISION_PRQ4_V2` 快照，覆盖 D1-A/B helper、C1
checkpoint-only runner、D1 validator、D1 controls，以及 G1–G4 control
registration。旧版 D1 审查与发现保持不可变；本文只记录最新快照的复测。

## Evidence

- `pytest tests -q --disable-warnings`: 254 passed，1 个既有 warning。
- `compileall src tests scripts`: pass。
- ResearchPilot code validator: pass，90 executable/config files、0 violations，
  `local_gpu_probe=forbidden_not_run`。
- D1 helper/control targeted tests: 11 passed。
- 本地未读取真实数据/权重，未探测本地 GPU，未 SSH、未训练、未访问 sealed-test。

## Closed checks

- `valid_token_mask` 与 backward-sampling shift 的物理位移约定一致；正负方向
  impulse 测试和 15×15→120×120 conservative bilinear-support mask 已覆盖。
- `_cloud_path` 拒绝 `.`/`..`、非 `/root/autodl-tmp` 或 `/root/autodl-workspace`
  路径和父级 symlink；输出目录必须全新，结果用 fsync+atomic replace 且拒绝覆盖。
- runner 在 validation loader 前检查 CUDA、V2 plan SHA、code release source SHA；
  随后强制核对 common protocol、C1 source manifest、C1 checkpoint、CROMA
  checkpoint 和 run-manifest provenance。
- attention 统计包含 nonnegative/row-stochastic contract、row-sum bounds、
  normalized entropy、same-index/local mass；D1-B 输出 15 个 displacement
  recovery rows、per-sample means、Spearman、calibration 和 trapezoid AUC 定义。
- G1–G4 已注册到 fusion/factory/config/run-manifest/train CLI，均复用同一 state
  surface、zero-start residual、decoder-visible path 和 trainability policy；
  它们尚未产生云端科学结果。

## Decision

`PASS_FOR_LOCAL_D1_CODE_ONLY; FRESH_CLOUD_SYNC_REQUIRED`。

当前允许一次受保护 code-only 同步，之后只可执行 D1-A/B validation-only
诊断。G0–G5 的 24 轮控制训练必须等待 D1-A/B receipt 和 D1 decision；不得
提前排名、晋级候选、组合、确认或访问 sealed-test。
