# V20 Plan Revision — R2 / R1 候选合同（用户批准方向）

Timestamp: 2026-09-03
Status: plan_locked_not_implemented（本文档定稿后进入实现）
Baseline: PRQ4-BASELINE-VERIFIED-R2（best mIoU 49.7807879% @ epoch18，24-epoch CE+Lovasz，sealed test）
Gate: <+1pp 关闭 / +1~+2pp marginal / >=+2pp 支持（对照组解锁）；**判定升级为 3-seed mean±95%CI**

## 0. 冻结协议清单（不可变）

pretrained_initialization（CROMA-base 审计权重）/ data_split / preprocessing / augmentation（paired_geometric_v1 D4）/ sampler / micro_batch 16 / effective_batch 32 / optimizer AdamW / learning_rate / scheduler / evaluator（segmentation_v1, present-class mIoU）/ objective：CE+Lovasz 1:1 / 24 epochs / seed 0 / test sealed。任何候选不得改动上述字段；机制参数全部挂载于 `router.*` 前缀（trainability parity 自动排除），机制不改变 state-dict 的 common 键集合语义（新增键仅限 router.*）。

## 1. R2：r2_depth_group_inject —— SAR 深度组的确定性轻量注入

### 1.1 动机（来自项目诊断数据）
- 审计确认：基线 always_fuse 前向计算 SAR 深度组 `depth_features ∈ [B,N,4,D]`（CROMA S1 四层 tap），但 `_dispatch` 仅在 ESCALATION 状态消费它，基线恒为 CURRENT → **4 层 SAR 深度信息在基线中完全未使用**。
- v13 反事实：SAR-off → 29.17%，SAR 全局有用；深度组是"已提取但未使用"的最大信息缺口。
- v14 DTSF 失败模式：固定 Hadamard 谱坐标 + rank-8 读出 + 后融合注入 → best +0.62pp / endpoint −0.42pp。本候选**剔除**其失败部件（谱变换、读出来、后融合注入），保留"深度组有信号"这一中性证据。

### 1.2 数学定义（hard contract）
- 位置：`OpticalSarTokenModel.forward` 的 stage 循环内、`self.fusions[stage]` 调用**之前**，仅作用于最后一个 stage（late）。
- 输入：`depth_features [B,N,4,D]`（已由 stem 投影）、`sar_stage [B,N,D]`（late）。
- 参数（全部 `router.` 前缀）：
  - `layer_weights ∈ R^4`（未归一化 logits，softmax 归一化；初始化全 0 → 均匀 1/4）
  - `layer_proj ∈ R^{D×D}`，**零初始化**（残差注入零起步）
- 前向：
  ```
  a = softmax(layer_weights)                 # [4]
  h = Σ_l a_l · depth_features[:, :, l, :]    # [B,N,D] 深度组加权聚合
  r = layer_proj(h)                           # [B,N,D]
  sar_stage = sar_stage + r                   # 融合前注入（仅 late stage）
  ```
- 不变量：`layer_proj` 零初始化 → 初始注入恒为零，模型等于基线；无第二次 CROMA forward；无 label/argmax/验证调参；无谱/频率/坐标变换（与 DTSF 家族区分的核心）；推理路径与训练完全一致（无任何 training-only 分支）。
- 计算开销：B·N·D·(D+4)，对 [2,225,768] ≈ 0.35G FLOPs（可忽略，无新注意力）。

### 1.3 与 DTSF 家族的区分（overlap 论证要点）
| 轴 | DTSF-01（已关闭） | R2 |
|---|---|---|
| 变换 | 固定正交 Hadamard 谱坐标 | 无（可学习 4 标量线性加权） |
| 读出 | rank-8 低秩读出网络注入 late SAR | 单层 D→D 零起点投影 |
| 位置 | 融合后（fusion 输出处） | 融合前（sar_stage 输入处） |
| 叙述 | 深度轨迹谱融合 | 深度组信息缺口补全（信息利用，非谱域方法） |
无需声称：深度特征利用/加权聚合/残差注入/投影 为新概念。窄声明：固定下游接口上对"未使用深度组"的确定性融合前注入，且效应经 3-seed CI 判定。

## 2. R1：r1_low_energy_channel_gain —— 分类器前的劣势通道增益（无 label）

### 2.1 动机
- v19 CTSP 唯一 meaningful 观察：rare class +1.86pp（best checkpoint）；rare IoU 0.138/0.244/0.298 vs 主类 0.78 → 类别能量失衡是验证集最大可解释差距。
- 约束：objective 冻结（不能用 focal/重采样/加权 loss）；数据冻结；不做分类器几何（CTSP 家族关闭）；不做梯度改写（v5/v6 家族关闭）；避免 train/eval 不对称（v5/v6 失败教训之一）。

### 2.2 数学定义（hard contract）
- 位置：stage 循环之后、`self.classifier(fused)` 之前，作用于最终 fused carrier `[B,N,D]`。
- 参数（`router.` 前缀）：`raw_gamma ∈ R`，**初始化为 0.0**；`gamma = relu(raw_gamma)`（零起步、恒 ≥ 0 → "仅放大"不变量由参数化保证，评审修复 2026-09-03）。
- 前向（训练与推理同一路径，无 label）：
  ```
  e_d = mean_{b,n} |fused[b,n,d]|                     # 每通道平均绝对能量 [D]
  e_max = max_d e_d + 1e-6
  scale_d = 1.0 + gamma * (1.0 - e_d / e_max)         # 低能量通道被放大，高能量通道≈1
  fused = fused * scale_d[None, None, :]              # 逐通道增益
  ```
- 不变量：gamma=0 → 恒等（模型=基线）；`scale_d ∈ [1, 1+|gamma|]`（谱系化：仅放大不衰减，无负效应通道）；无 label、无 argmax、无第二次 forward；训练/推理完全一致；不触碰分类器权重（与 CTSP 家族区分）；不触碰损失（与 v12/v5/v6 区分）。
- 文档域：特征能量自举的逐通道劣势放大（low-energy channel gain），明确"非归一化"（不做除均值/方差——避免与 BN/LayerNorm 家族撞车："仅放大、不重标定、单标量整体步长"）。

### 2.3 撞车论证
- 与归一化族（BN/LayerNorm）：BN 做的是 per-channel 均值方差归一化（可减可增、逐层学习）。R1 是单标量 γ 的绝对能量放大（增益≥1、无均值对齐）。措辞明确不做 normalization claim。
- 与注意力/通道重标定（SE 类，按功能类口径）：SE 是逐通道独立可学习网络（每个通道一个学习参数/网络输出），R1 是单一全局标量 γ × 固定能量函数（无逐通道参数、无网络）；"无网络"不作为唯一辩护，功能差异为"参数复杂度从 θ^D 降到 θ^1"。
- 与 CTSP 家族（评审修复 2026-09-03，按功能类口径）：特征域对角缩放与分类器列缩放数学等价——承认该等价存在；R1 的剥离论证：① 缩放由**单一全局标量**（非逐类/逐通道/逐 token 网络）与激活统计决定，无分类器投影/伪逆/判别几何；② 缩放仅依赖载体自身能量（无 label、无分类器权重参与）；③ 判定不依赖 CTSP 家族的任何"位移上限/切空间"构造。若 plan 评审仍判 Level 2 重叠，R1 降级为消融诊断，不作为论文 claim。
- 与特征缩放家族先验（DOLG 等）：DOLG 是对局部/全局特征做正交融合；R1 是最终载波上的单标量通道增益。窄声明：固定下游载体上的、零参数规模化出道前的能量劣势增益，效应经 3-seed CI 判定。

## 3. 判定协议（plan）

- 阶段 1（本轮）：R2、R1 各 seed-0 24-epoch 全流程。
- 阶段 1 判定规则（预注册，评审修复 2026-09-03，单元格全覆盖）：
  - **通过**：best-delta ≥ +1.0pp 且 endpoint-delta ≥ +0.5pp（不允许再现 v14/v19 的"中段领先、终点反噬"模式）；
  - **边际保留**：best-delta ∈ [+0.5, +1.0)pp 且端点 ≥ 0**，或 best-delta ≥ +1.0pp 且 endpoint-delta ∈ [0, +0.5)pp** → 进入 3-seed 判定的"统计澄清"通道；
  - **关闭**：best < +0.5pp，或端点 < 0，或出现反噬模式（best ≥ +0.5 且 endpoint ≤ −0.25pp）。
- 阶段 2（3 seeds mean±95%CI，判定线按通道分别预注册）：
  - **直接通过者**（阶段 1"通过"）：CI 下界 ≥ +1.0pp 且均值 ≥ +2.0pp → 解锁对照（C1 位置/消融）与后续强化；否则家族关闭并入"失败谱"。
  - **边际澄清者**（阶段 1"边际保留"）：CI 下界 ≥ +0.5pp 且均值 ≥ +1.0pp → 解锁 R3 消融与一次强化（可加种子至 5）；否则家族关闭并入"失败谱"。两通道判定线无交集、无死路（"有效方向"在每通道都有明确后续动作）。
- 控制项：R3（解码容量：线性头→两级轻量头，零起步）作为消融件与胜者组合，不单独成立。
- 全程 sealed test 不访问；云端 /root/autodl-workspace 主线干净运行；runs 输出至 /root/autodl-tmp/runs_v20/。

## 4. 实验与资源

- 每 run：24 epoch、seed 0、RTX 3090、约 1–2 天（并行 2 run 或串行）。阶段 1 合计 ≈ 2–4 天；阶段 2（3 seeds）≈ 3–6 天。
- 与门禁体系：本文档提交 PLAN_REVISION 审批流；PLAN_APPROVAL 已由用户在对话中授权（"我认可，做吧"），以本文档 + 后续实现收据为准。

## 5. 风险与撤回点

- R2 撞车 DTSF 家族的判定风险：已在 1.3 论证，若 plan 评审判 Level 3 重叠，R2 自动降级为"消融诊断"（不作为论文 claim，仅作信息缺口证据）。
- R1 的"增益>1"特性在理论新颖性上较弱：若阶段 2 未达 CI 下界 ≥+1.0，R1 并入失败谱；计划时间盒到 2026-10-15，届时无 ≥+1.0 支持的方向则停止新方向，转为"失败谱+诊断"论文（Plan B），写入专刊叙事。