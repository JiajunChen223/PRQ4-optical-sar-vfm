# V17 MCOF 修复后架构复审（只读）

- 日期：2026-08-31
- 路线：`R-EO-MCOF-V17-01 / MCOF-01`
- 范围：谱范数与算子上界、120×120 对齐、有限性、raw-image 训练见证、3090 资源和测试覆盖
- 边界：未修改代码；未读取真实数据/权重；未探测 GPU；未训练
- 结论：`PASS_WITH_NONBLOCKING_MEDIUMS`
- blocker：`0`；high：`0`；medium：`2`

## 总结

上一版的 HIGH-01 已在实现层面解决。MCOF 现在把实际尺度写成
`0.25*tanh(alpha)`，并在每次前向中用 float32 计算并限制
`semantic_projection` 和 `class_basis` 的谱范数至 1。由于 controller 经过
`tanh`，局部增量 `U diag(a(p)) V^T` 的理论谱范数上界为
`0.25 × 1 × 1 = 0.25`。该上界不是依赖 optimizer 或 gradient clipping 的
经验假设，而是前向计算的显式约束。原始候选的零起点 identity 仍保持精确。

## 已解决事项

### HIGH-01：算子幅值无界 — 已关闭

- 文件：`02_experiment/code/src/geotoken3path/mechanisms/mcof.py:35-52, 212-260`
- 证据：新增 `_spectrally_bounded_weight`；两组因子均先转 float32 求
  `torch.linalg.matrix_norm(..., ord=2)`，再以 `min(bound/norm, 1)` 缩放；
  `scale = operator_scale_bound * tanh(alpha)`；contract 固定
  `operator_scale_bound=0.25`、`factor_spectral_bound=1.0`。
- 测试：`tests/unit/test_mcof.py` 的极大 alpha/因子/输入 stress case 通过，
  输出 finite，两个 bounded factor ≤1；hard contract 的谱范数和理论算子
  上界检查通过。
- 评价：原“bounded operator”表述现在有实现支撑。注意这是前向有效因子
  的上界；原始参数本身仍可变大，但不会直接进入算子。

### MEDIUM-01：raw-image 训练图未见证 — 已关闭

- 文件：`02_experiment/code/tests/integration/test_croma_bridge.py:301-345`
- 证据：新增 15×15 synthetic CROMA + 120×120 optical/SAR 的
  `model.train()`、CE backward；`alpha`、condition stem、semantic projection、
  class basis 均检查 finite 且非零梯度。
- 测试：MCOF/CROMA 相关测试从 32 项增至 34 项，`34 passed`；配置、manifest、
  factory 相关测试另有 `12 passed`。
- 评价：MCOF standalone 与 raw-image bridge 两条路径均已覆盖；仍只是
  synthetic/software evidence，不是科学性能证据。

### MEDIUM-02：中间量和输出有限性 — 基本关闭

- 文件：`02_experiment/code/src/geotoken3path/mechanisms/mcof.py:172-189, 240-247`
- 证据：condition、projected、gated、correction、logits 均显式检查 finite；
  输入也继续检查 finite；大幅值 stress 测试通过。
- 评价：正式路径中的可见张量已 fail-closed。`condition_stem` 的 tanh 前
  极端溢出会被 tanh 饱和为有限值，但不会静默产生 NaN；3090 AMP 真实行为
  仍需云端 preflight 测量，不能由本地测试替代。

## 尚存的非阻塞问题

### MEDIUM-R1：run manifest 未强制校验 factor spectral bound

- 文件：`02_experiment/code/src/geotoken3path/utils/run_manifest.py:331-359`
- 证据：MCOF manifest 校验固定了 `operator_scale_bound=0.25`，但没有固定
  `factor_spectral_bound=1.0`，尽管 resolver、contract 和运行时都已包含该字段。
- 风险：当前官方 resolver 生成的 manifest 是正确的，但独立篡改/错误构造
  的 resolved snapshot 仍可能绕过该一项 manifest 级契约。
- 建议：在 `expected_mcof` 加入 `factor_spectral_bound: 1.0`，并增加一条
  manifest rejection 测试。此问题不改变当前已运行代码的算子上界。

### MEDIUM-R2：token 顺序与像素中心的显式 witness 仍缺失

- 文件：`mcof.py:135-149, 219-225`；`croma_bridge.py:703-714`
- 已解决部分：现在严格要求 optical/SAR 为 120×120，语义坐标固定为 15×15，
  使用统一 `bilinear, align_corners=False`；因此尺寸契约已闭合。
- 尚缺部分：没有 impulse/coordinate fixture 显式证明 CROMA 的 token index
  顺序与 raw image lattice 对应。
- 建议：补充 CPU synthetic 坐标/脉冲测试，固定 15×15→120×120 的中心映射
  和 row-major token 顺序。该缺口是可审计性增强，不是当前运行 blocker。

## 资源与数学复核

MCOF formal width 的 synthetic 参数量为 15,913，远低于 500,000 上限。最大
新增空间张量约为 `[B,16,120,120]` 的 projected/gated/correction，未创建
`[B,768,120,120]` dense activation 或 `[B,H,W,D,C]` classifier tensor。谱范数
计算作用于 16×768 与 11×16 小矩阵，静态上不会形成 3090 显存 blocker；但
真实 CROMA 的峰值显存、吞吐、AMP 溢出和多 worker 行为仍只能由云端 preflight
确认。

## 本轮合成验证

1. `pytest tests/unit/test_mcof.py tests/unit/test_v17_mcof_config.py tests/integration/test_croma_bridge.py -q --disable-warnings --maxfail=20`：`34 passed`。
2. `pytest tests/unit/test_run_manifest_hardening.py tests/unit/test_config_contract.py tests/unit/test_model_factory.py -q --disable-warnings --maxfail=20`：`12 passed`。
3. `python scripts/run_v17_mcof_hard_contract.py`：`14/14 PASS`，包含谱因子、
   理论算子上界、zero-start、两步梯度和 controls；`scientific_result=false`，
   `real_data_read=false`，`weights_read=false`，`gpu_used=false`，
   `test_accessed=false`。

## 复审结论与边界

HIGH-01、raw-image training witness 和可见中间量有限性问题已解决；当前仅
保留 manifest 字段绑定与显式空间顺序 witness 两个 medium 可审计性缺口，均
不构成 blocker。可以进入正式云端 preflight，但本报告不构成科学性能支持，
不授权多种子、晋级或 sealed-test。
