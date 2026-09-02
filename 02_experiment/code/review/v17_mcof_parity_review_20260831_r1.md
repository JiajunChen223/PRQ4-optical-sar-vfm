# V17 MCOF 训练对象与协议 parity 审查

审查日期：2026-08-31  
审查根目录：`F:\\PRQ4`  
审查范围：V17 `R-EO-MCOF-V17-01 / MCOF-01` 的 baseline `always_fuse` 与候选 `mcof_multimodal_conditional_operator`，以及已声明的 MCOF matched controls。此次审查仅读取代码、配置和合成 fixture 测试结果；没有读取真实像素、云端权重或 GPU，没有训练，没有解封 sealed-test。

## 总结

Blocker：none。Major：none。当前代码满足进入云端正式前置预检的 parity 条件，但正式执行前仍须完成 clean-sync manifest 生成并重新解析正式 execution scale；该项属于打包/绑定前置条件，不是科学结果，也不应被当作候选失败。

## 已通过项目

1. 路由和机制边界：`configs/experiment/v17_mcof_route.yaml` 明确固定任务、SEN12TS、24 epochs、单路由、baseline/candidate/control 集合和 `single_internal_mechanism_delta=true`。`configs/model/v17_mcof.yaml` 固定 rank=16、15×15 coarse grid、120 输出、12+2 输入、无 auxiliary loss、无 label guidance、无 dense classifier tensor，并声明参数预算上限 500,000。

2. 配置差分：用 `resolve_v17_mcof_config(..., execution_scale="smoke")` 分别解析两行，得到相同的 `matched_common_protocol_sha256`：`209f70d45486dc064ab0a40ae58584354ccbfdc7af9147dde615349af88cdbb6`。`single_mechanism_diff` 仅返回 `model.mechanism_set` 与 `objective.innovation_claim_eligible`；后者是候选资格元数据，不改变 objective 的 `ce_lovasz`、权重 1:1、ignore index 255 或 evaluator，属于显式且可审计的声明差异。

3. state dict 和 trainability：`build_model` 对 baseline/candidate 产生完全相同的 state-dict key 顺序和总参数数目：44,135,875；可训练参数均为 44,120,515，`requires_grad` 名称集合完全一致。MCOF decoder 在 `OpticalSarTokenModel` 内部统一分配，两行使用同一个 model factory；candidate 只在内部 MCOF 分支调用它，没有冻结 baseline 后另训外部 router/refiner/auxiliary model 的路径。

4. 训练协议：解析配置一致固定 RTX_3090_24GB、AMP、micro-batch 16、effective batch 32、gradient accumulation 2、AdamW (`lr=1e-4`, `weight_decay=0.05`, `betas=(0.9,0.999)`)、cosine-with-warmup、gradient clip 1.0、24 epoch 上限、validation mIoU early stopping（burn-in 8、patience 5、min delta 0.1pp、restore best）。D4 paired augmentation、train-only、deterministic 也一致。正式 runner 在 `formal_runner.py` 中对 V17 强制 24 epoch、validation-only 和 sealed test。

5. 初始化、数据和评估：两行共享同一 cloud-only CROMA pretrained audit 引用、构造器参数和 checkpoint policy；代码树未包含权重二进制。`croma_bridge.py` 的 MCOF 分支接收同一 audited CROMA bridge 的 token logits/fused tokens 和原始 12-channel optical、2-channel SAR 图像；没有改变 split、标签或预处理。输出仍为 11 类 dense segmentation，`ce_lovasz` 与现有 baseline evaluator 共用。

6. matched controls：`croma_bridge.py` 将 conditional、static、sample-level、shuffled、optical-only、SAR-only 映射到同一 `MultimodalConditionalOperatorField`，只改变 condition axis。`mcof.py` 对 rank/grid/channel/预算/zero-start 等契约 fail-closed。测试覆盖了 zero-start bitwise identity、激活后的结构性 logits 改变、六类 condition control 的差异、参数维度约束和两步 gradient liveness；相关 MCOF unit/integration/config/parity 测试共 41 项全部通过（`41 passed`）。这只是机制和接口证据，不是科学性能证据。

## 记录项

### Finding N-01

- severity: note
- file: `02_experiment/code/src/geotoken3path/utils/config.py`; `02_experiment/code/manifests/`
- finding: 在当前 authoring tree 中，V17 配置声明的 `clean_sync_manifest_v17_mcof_20260831_r1.json` 尚未生成，因此以 `screening` 等正式 execution scale 解析配置会 fail-closed，报 `route requires a source or embedded clean-sync release manifest binding`；smoke 解析按设计不需要该绑定。
- evidence: `resolve_v17_mcof_config(..., execution_scale="smoke")` 成功且 parity hash 一致；同函数以 screening scale 解析时触发上述明确错误。`build_v17_mcof_release_r1.py` 已提供生成、逐文件 hash、压缩包成员校验和 CODE_REPORT 的正规路径。
- proposed fix: 三份 V17 review 完成后运行 `02_experiment/code/review/build_v17_mcof_release_r1.py`，生成 clean-sync manifest 和 code-only package；随后重新运行正式 scale 配置解析、run-manifest hash 校验和 code validator。不得通过删除绑定检查或使用旧版本 manifest 绕过。
- status: open_prepackaging
- retest: 重新执行 V17 config resolution、`build_run_manifest`、`validate_code_project.py` 及 package member/hash audit。

## 最终审查裁决

V17 MCOF baseline/candidate 在共享 detector factory、state-dict surface、参数 trainability、初始化、数据接口、objective、optimizer/scheduler、batch、24-epoch horizon、evaluator 和 test seal 上均通过 parity；matched controls 具备同一入口和明确 condition-only 操作差异。没有 blocker 或 major finding，允许进入正式 code-only release finalization 和云端受控前置预检。任何正式 mIoU、OA、显存、吞吐或候选支持结论仍必须等待云端 protocol-valid run；本报告不构成科学支持。
