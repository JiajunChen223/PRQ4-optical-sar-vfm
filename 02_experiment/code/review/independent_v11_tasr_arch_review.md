# Independent V11 TASR architecture review

审查时间：2026-08-29（本地只读审查）  
审查范围：V11 `TASR-01 / tasr_token_anchored_spatial_redistribution` 的数学实现、训练/推理边界、CROMA bridge 集成、模型 parity、配置与入口绑定。  
禁止项遵守：没有读取真实数据或权重，没有 GPU 探测，没有云端执行，没有打开 sealed test。报告只评价代码结构，不把 synthetic/liveness 结果当作科学结果。

## 总体结论

**BLOCKED — 暂不建议进行 V11-C1 正式同步或 24-epoch 训练。**

当前代码通过了现有的结构测试，但关键的“token semantic conservation”实现并没有满足 V11 计划中写死的数学契约：代码保留的是每个 8×8 patch 的 bilinear baseline 均值，而不是对应 15×15 token logit (L_i)。对于 `15×15 → 120×120`、`align_corners=False` 的正式路径，两者一般并不相等。因此当前 synthetic liveness 中的 `token_patch_mean_conservation=true` 是对 `output - B` 的零均值检查，不能证明计划要求的 `mean_{P_i}(Z)=L_i`。

在修复该 blocker 并补充 exact-anchor 测试前，现有 31 个 TASR/CROMA targeted tests 的 PASS 只能说明代码可运行，不能作为 TASR hard-contract PASS。

## 已执行的只读验证

```text
python -m pytest -q --disable-warnings --cache-clear \
  tests/unit/test_tasr.py tests/integration/test_croma_bridge.py
31 passed, 1 warning

python -m pytest -q --disable-warnings --cache-clear
301 passed, 1 warning
```

另外做了 CPU-only 的 synthetic 数学检查、CPU bfloat16 identity 检查和 full raw-image bridge parity 检查；没有使用真实数据、权重或 GPU。

## Findings

### F1 — BLOCKER：保守量锚定到 bilinear (B)，不是 token logits (L_i)

位置：`src/geotoken3path/mechanisms/tasr.py:156-174`。

当前逻辑为：

1. `base = Bilinear(token_logits)`；
2. `base_patch = mean_P(base)`；
3. 将 diffused 图像重新居中到 `base_patch`；
4. 再把 `residual` 投影为 `mean_P(residual)=0`；
5. 输出 `Z = base + alpha * residual`。

这严格保证的是：

```text
mean_P(Z - B) = 0
```

并不保证计划中的：

```text
mean_P(Z) = L_i
```

这是数值上可复现的，不是边界上的纯理论疑虑。CPU witness 使用随机 `[1,225,11]` token logits：

```text
max |mean_P(B) - L| = 1.8555816411972046
mean |mean_P(B) - L| = 0.35894396901130676
```

使用简单的二维 ramp 时，最大偏差为 `2.0`。原因是 `align_corners=False` 下，8×8 block 内的 bilinear samples 对任意离散 token 网格并不具有“block mean 等于中心 token”性质；即使是内部 patch 也可能存在偏差。

建议修复方向（本审查不改源码）：保留 `base` 作为 `alpha=0` 的 identity reference，但 conservation correction 的目标应由 `token_logits` 重排为 `[B,C,15,15]` 的 `L_grid`，而不是 `base_patch`。随后单独验证：

```text
alpha = 0  =>  output == B  (bitwise)
alpha != 0 => max_abs(mean_P(output) - L_grid) < 1e-6
```

还应在测试中同时检查 `mean_P(output-base)=0` 与 `mean_P(output)-L_grid=0`，避免再次把 residual conservation 误命名为 token conservation。

### F2 — HIGH：声明的 per-modality guidance convolutions 是 dead parameters

位置：`src/geotoken3path/mechanisms/tasr.py:70-85` 与 `:99-105`。

`self.optical_guidance` 和 `self.sar_guidance` 被构造为浅层 depthwise convolution，但 `_affinity()` 实际直接拼接原始 `optical[:, :8]` 与 `sar[:, :2]`，只调用了 `self.affinity`。因此这四组参数在 backward 中始终为 `grad=None`；当前 liveness 只检查 raw input gradient，没有检查所有 guidance-module parameter gradient，故无法暴露该问题。

这会造成两个不一致：

* 代码表面上有独立 optical/SAR guidance path，实际 affinity path 没有使用它们；
* 这部分参数仍进入 state dict、optimizer 和 parity surface，却不参与 TASR 学习。

如果 per-modality depthwise layers 是设计的一部分，应在 `_affinity()` 中实际调用并为其增加梯度可见性测试；如果它们不是设计的一部分，应移除或明确标注为不参与训练，并同步更新参数/contract 审计。该 finding 单独不否定“class-agnostic affinity”，但目前的实现与描述不一致。

### F3 — MEDIUM：模块自身的 `.eval()` 不会 bypass

位置：`src/geotoken3path/mechanisms/tasr.py:136-192`；外层 bypass 在 `src/geotoken3path/models/croma_bridge.py:756-780`。

`CromaGeoTokenSegmentation` 的 formal path 在 `self.training == False` 时确实调用 `bypass_tasr_in_eval()`，因此当前外层 raw-image integration 的 eval identity 是 PASS。可是直接调用：

```text
module.eval(); module(token_logits, optical, sar)
```

仍会执行 affinity、diffusion 和 active auxiliary path；CPU witness 中 `alpha=0.379948...` 且输出不等于 bypass。若部署/导出路径直接持有 `TASRSpatialRedistributor`，则“inference no additional TASR path”不能仅靠外层调用约定保证。

建议在模块级别也 fail-closed：eval 时直接返回 bilinear bypass；并增加 `module.eval()` 的 exact identity、`auxiliary_active=False`、无 guidance dependency 测试。若项目明确禁止直接部署该 module，则应在导出 validator 中硬性验证只允许通过 `CromaGeoTokenSegmentation` 的 bypass path。

### F4 — MEDIUM：token grid 与 output grid 的几何契约未在 runtime 强制绑定

位置：`src/geotoken3path/mechanisms/tasr.py:32-39, 144-150`。

当前只检查 token 数是 square，以及 output spatial size 能被 patch size 整除；没有检查：

```text
sqrt(num_tokens) == output_height / patch_size == output_width / patch_size
```

所以 `16` tokens 配 `32×32` 会通过（现有测试也如此），`16` tokens 配 `120×120` 也可以进入 interpolation/conservation，只是语义上已经不是“一 token 对应一个 8×8 patch”。正式 V11 的 `225/120/8` 是一致的，但 hard contract 应拒绝其它组合，防止 future config 或错误 checkpoint 静默改变 token anchor 的含义。

### F5 — MEDIUM：eta、通道选择和 contract 与构造器的绑定仍不完整

位置：`src/geotoken3path/mechanisms/tasr.py:63-103`、`models/fusion.py:1810-1828`、`models/factory.py:46-58`、`configs/model/v11_tasr.yaml`。

当前最新快照已经加入 `validate_tasr_contract()` 和实际 `parameter_count <= 100000` 检查，这一部分 PASS；resolver/run-manifest 也会携带 TASR contract。仍有三点需要在正式 release 前明确：

* `eta=0.25` 没有进入 YAML/resolved manifest，运行契约无法从 manifest 独立恢复；
* guidance 使用的是“前 8 个 optical channels、前 2 个 SAR channels”，配置没有明确绑定这些 channel indices/order；
* direct `build_model()` 可用默认 TASR contract 构造候选，不要求 caller 传入 resolved TASR contract。formal resolver 会约束正式 cloud 配置，但 synthetic/direct factory lane 仍不是完全 fail-closed。

这些不一定需要改变方法，但应在 release validator 中解决，否则不同入口可能运行不同的数学 operator，而 manifest 仍看起来相同。

### F6 — LOW/MEDIUM：每个 forward 的 Python finite/equality 检查可能引入 CUDA synchronization

位置：`src/geotoken3path/mechanisms/tasr.py:24-29, 180-188`。

`torch.isfinite(...).all()` 被放在 Python `if` 中，且 `torch.equal(output, base)` 被转成 Python bool。CUDA formal training 中这些检查可能在每个 batch 产生 device-to-host synchronization；`return_aux=True` 的 telemetry 会让该路径默认执行。它们对 correctness audit 有用，但与 3090 throughput/step-time 目标冲突。

建议把同步性检查限制在明确的 hard-contract/liveness 模式，正式训练只保留 device-side finite telemetry 或低频检查；`torch.equal` 也应只在 contract test/periodic audit 使用。该 finding 不改变当前数学 blocker，但应在正式 benchmark 前处理并测量。

### F7 — MEDIUM（入口协议风险）：V11 baseline 可误走 legacy resolver

位置：`scripts/train.py:181-193`、`scripts/evaluate.py:72-80`。

`train.py` 只有在 `--route-variant v11_tasr` 或机制是 TASR 时才选 V11 resolver；当 V11 baseline 使用 `always_fuse` 且调用方省略 `--route-variant v11_tasr` 时，会退回旧 `resolve_approved_config()`（`R-EO-CEAK-01`/legacy route）。这会使 baseline/candidate 的 route/candidate metadata 与 matched protocol context 不再由同一个 V11 route 文件生成，尽管两者都可能成功运行。

建议 V11 formal command/validator 对 baseline 也强制 `route_variant=v11_tasr`，或根据当前 route plan 自动绑定 V11 baseline；并增加测试确保 `always_fuse` baseline 与 TASR candidate 的 resolved route、common protocol hash、data/initialization/seal contract 来自同一 V11 route。

## Positive checks (not blockers)

以下项目在本快照中通过：

* `VALID_MECHANISMS`、TASR direction id、train/evaluate choices、run-manifest mapping 均包含 TASR；
* `TASRSpatialRedistributor` 的 affinity 输出为 `[B,1,H,W]`，没有 class dimension；
* 4-neighbor boundary mask 没有发现 `torch.roll` wrap-around leakage；
* `alpha=0` 时 standalone output 与 bilinear `B` bitwise 相等；
* CPU bfloat16 下 standalone identity，以及 full raw-image bridge eval baseline/candidate identity 均 exact；
* full raw-image bridge 在 training mode、非零 `tasr_scale` 时能向 raw optical/SAR 输入和 affinity head 传播有限梯度；
* baseline/TASR token-model 及 full CROMA synthetic bridge 的 state-dict key order 和 `requires_grad` surface 一致；
* 最新 constructor 的 TASR parameter count 为 `212`，低于 `100000` 上限；
* targeted tests `31 passed`，full local suite `301 passed`。这些均是代码/fixture evidence，不是科学性能证据。

## Required closure before C1

1. 修复 F1，并用 exact `mean_P(output)-L_grid` 测试替换/补强现有 residual-only 检查。
2. 明确 F2：接通 per-modality guidance 或移除 dead modules，随后要求实际 branch parameter gradients。
3. 对正式几何尺寸增加 F4 runtime guard；把正式 `225/15/120/8` 写入测试。
4. 将 eta、guidance channel selection 和 resolved contract 绑定到 manifest；formal baseline/candidate 强制同一 V11 resolver（F5/F7）。
5. 评估 F6 的 CUDA sync 影响；如保留检查，必须单独记录其训练开销。
6. 重新运行 targeted/full tests、independent test audit、package/manifest audit；在上述 closure 完成前不要执行 TASR-01 C1 24-epoch cloud run。

