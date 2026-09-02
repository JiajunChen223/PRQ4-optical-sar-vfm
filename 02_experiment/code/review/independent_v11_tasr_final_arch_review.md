# Independent V11 TASR final architecture review

审查时间：2026-08-29（只读）  
范围：`tasr.py` 的 guidance、geometry、dual conservation，以及 raw-image CROMA bridge 集成。

## 结论

**PASS（architecture/hard-contract scope）**。

## Evidence

- 在 `F:\PRQ4\02_experiment\code` 执行：
  `pytest tests/unit/test_tasr.py tests/integration/test_croma_bridge.py -q --cache-clear`
  结果：**33 passed, 1 warning**。
- TASR guidance 明确选取 optical `[0..7]` 与 SAR `[0,1]`，经实际 depthwise guidance 后进入单通道 affinity；class-agnostic，训练梯度路径可达 guidance 参数和原始光学/SAR输入。
- geometry runtime 强制 square token grid、`output_size` 可被 patch size=8 整除，且 `sqrt(T)*8 == H == W`；4-neighbour `roll` 边界 mask 阻断 wrap-around。正式 `225 -> 120x120` witness 通过。
- dual conservation 通过：中间 `anchor_conserved` 的每个 patch/class 均值回到原始 token logits；最终 `output-base` residual 的每个 patch/class 均值为零；`tasr_scale=0` 时输出与 P1 bilinear readout bitwise 相等（FP32 synthetic witness 均 `<1e-6`）。
- CROMA bridge 只接受 raw `12+2` float32 配对输入，严格返回 `optical/sar/sar_depth_group` stage maps，并校验 `[B,N,D]` 与 `[B,N,4,D]`；depth taps 来自显式模块路径，没有 guessed/interpolated fine feature 或 joint-state decoder 偷渡。TASR raw-image 入口在无标签 eval 与训练路径均可达，且 zero-start parity 通过。

本审查未读取真实数据/权重、未探测 GPU、未访问 sealed test；PASS 仅表示当前代码结构合同通过，不构成 V11-C1 科学结果或实验放行。
