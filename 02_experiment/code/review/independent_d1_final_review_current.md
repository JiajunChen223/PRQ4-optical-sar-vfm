# D1-A/B current-snapshot final review

审查对象为 V2 D1-A/B 当前工作树，包含 D1 helper、C1 checkpoint-only runner、
D1 validator、G1–G4 matched controls 和相应测试。原始独立审查报告保留在
同目录，本文只记录修复后的当前快照复核，不覆盖历史发现。

## 复核结果

- D1/D0 helper 与 runner tests：11 passed。
- D1 control tests：9 passed。
- 全部本地 synthetic suite：254 passed，1 个既有 warning。
- `compileall`：pass。
- ResearchPilot code validator：pass，90 executable/config files、0 violations，
  `local_gpu_probe=forbidden_not_run`。
- 本次没有读取本地真实数据/权重，没有本地 GPU 探测、SSH、训练或 sealed-test 访问。

## 已闭合阻断

1. shift 的正负约定已统一：`shift_token_grid` 的物理位移与
   `valid_token_mask` 同号，增加正向源内容移动测试；bilinear-support mask
   对 15→120 上采样的插值支持做保守排除。
2. D1 路径拒绝 `.`、`..`、越界 root 和父级 symlink；output 目录必须预先不存在，
   结果通过临时文件、fsync 和原子替换写入。
3. runner 在读取 validation 前检查 CUDA、V2 plan SHA、当前 release manifest SHA；
   加载后强制核对 common protocol、C1 source manifest、C1 checkpoint 和 CROMA SHA。
4. 输出包含 row-sum/min-weight attention contract、Spearman、15 个 calibration 点、
   逐样本 displacement mean、zero-shift directional N/A 和 trapezoid AUC 定义。
5. G1–G4 已注册到 route/benchmark/model/run-manifest/train CLI，并通过同 state surface、
   zero-start、finite forward、attention shape 和 live-gradient 测试；它们仍是诊断
   controls，不是新候选。

## 最终判定

`PASS_FOR_LOCAL_D1_CODE_ONLY; FRESH_CLOUD_SYNC_REQUIRED`。

当前 D1 代码可以进入一次受保护 code-only 同步；同步后只允许执行 D1-A/B
validation-only runner。D1 科学结果仍未产生，G0–G5 尚未训练，任何候选晋级、
composition、confirmation 和 sealed-test 访问继续禁止。
