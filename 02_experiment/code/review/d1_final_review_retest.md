# D1-A/B final retest review

本复核针对两份独立审查提出的阻断项进行逐项复测；原始独立审查仍保留为
历史记录，不能被覆盖。复测范围是当前工作树中的
`d1_diagnostics.py`、`diagnose_d1_ab.py`、D1 validator、D1 tests，以及
G1–G4 control registration。

## Retest evidence

- D1/D0 targeted tests: 11 passed。
- D1 helper/control tests: 9 passed。
- Full local synthetic suite: 253 passed，1 个既有 warning。
- `compileall`: pass。
- ResearchPilot code validator: pass，89 executable/config files、0 violations，
  `local_gpu_probe=forbidden_not_run`。
- 未读取本地真实数据/权重，未探测本地 GPU，未 SSH、未训练、未访问 sealed-test。

## Finding closure

| Finding | Retest result |
|---|---|
| Shift sign / valid-interior mismatch | fixed：物理位移 mask 与 source-content movement 同号，并用 exact bilinear-support mask 排除插值混合边界；正负方向和 15→120 mask 有回归测试。 |
| Output overwrite | fixed：output directory 必须不存在，result 文件拒绝覆盖，使用临时文件+fsync+atomic replace。 |
| Cloud path traversal/symlink | fixed：拒绝 `.`/`..`、非声明 cloud root 和任一父级 symlink。 |
| Provenance binding | fixed：运行前绑定 V2 plan SHA、当前 release manifest source SHA、common protocol、C1 source manifest、C1 checkpoint SHA 和 audited CROMA SHA。 |
| G1–G4 unavailable | fixed：四个 control mechanism 已进入 YAML route controls、factory/fusion/run manifest/train CLI，并通过 strict state-surface/finite/gradient tests。 |
| Missing Spearman/calibration/raw recovery | fixed：输出 nonzero shift Spearman、15 个校准点、每 shift 180 个 per-sample displacement means、RMSE 和 directional accuracy；zero shift directional accuracy 为 N/A。 |
| Negative attention | fixed：attention contract 显式拒绝负值，并记录 row-sum/min-weight bounds。 |
| Padding ambiguity | fixed：runner 显式验证 final batch full-16 shape 和 padded target=255；metrics 使用 conservative valid pixel mask。 |
| AUC semantics | fixed：robustness AUC 采用 unique displacement magnitude 聚合后的 trapezoid integration，并记录定义；同时保留 mean-grid metric。 |
| Device ordering | fixed：CUDA device、plan hash、manifest binding 在 validation loader/data read 前检查。 |

## Decision

`PASS_FOR_LOCAL_D1_CODE_ONLY; FRESH_CLOUD_SYNC_REQUIRED`。

该结论只表示 D1-A/B 工程合同已可同步；不表示 D1 科学假设成立。云端
D1 输出仍需通过 `validate_d1_ab.py`，之后才可依据 D1 结果决定是否执行
G0–G5。G0–G5 不是当前 retest 的结果，不能提前排名或晋级候选。
