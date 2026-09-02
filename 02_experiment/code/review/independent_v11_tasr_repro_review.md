# Independent V11 TASR reproducibility and release review

审查时间：2026-08-29 20:01（Asia/Shanghai）  
审查对象：`R-EO-TASR-01` / `TASR-01`，当前 `F:\PRQ4\02_experiment\code` 快照  
审查类型：独立只读可复现、数据边界、训练对象、封存测试和发布审计  
执行边界：未下载或读取真实数据/权重，未访问云端，未启用 GPU，未修改源代码、配置或历史 gate。仅使用 CPU synthetic fixtures、静态文件/hash 检查和本地既有代码验证器。

## 结论

**结论：BLOCKED_FOR_CODE_HANDOFF（结构合同大部分通过，但当前快照不能进入 guarded code sync 或 V11-C1）。**

V11 的核心配置、TASR 结构合同、`TASR-01` 方向绑定、test-seal、初始化声明、3090 预算和同一模型工厂均已具备可复现基础；本地 synthetic 验证通过。但是当前发布账本在最近代码修订后已经 stale，canonical `CODE_REPORT.json` 仍然指向 D3/CEAK，且没有在默认 run manifest 中绑定 V11 clean-sync manifest。另有一个需要在 C1 前解决的核心语义问题：代码目前守恒的是 **bilinear P1 dense output 的 8×8 patch mean**，不是直接的原始 15×15 token-logit mean；测试名称和路线叙述把两者混称为 token mean。

因此本报告不支持任何科学性能结论，也不授权 V11-C1、C2/C3 或 sealed test。修复以下 blocker、重新生成 manifest/CODE_REPORT 并完成独立复核后，才可交回 Experiment Skill 做 guarded code sync。

## 已通过的检查

| 检查项 | 结果 | 证据 |
|---|---|---|
| V11 resolver 与 route | PASS | `resolve_v11_tasr_config` 成功加载 `v11_tasr.yaml` / `v11_tasr_route.yaml`；route=`R-EO-TASR-01`，candidate=`TASR-01`。 |
| baseline/candidate 配置差分 | PASS | `single_mechanism_diff(baseline,candidate) == ["model.mechanism_set"]`；两者 `matched_common_protocol_sha256` 相同。 |
| TASR contract | PASS | patch=8、optical guidance=8、SAR guidance=2、affinity=1、diffusion steps=1、budget≤100000 以及四个布尔不变量均被 resolver/run-manifest 检查。当前实现实际参数量为 212。 |
| `TASR-01` manifest binding | PASS（在显式 candidate id 下） | `build_run_manifest(... candidate_direction_id="TASR-01")` 成功，映射为 `tasr_token_anchored_spatial_redistribution`；hash 可由 `verify_run_manifest` 重算。 |
| 模型工厂/训练对象 | PASS | synthetic formal VFM fixture 中 baseline/candidate state-dict key 顺序相同（223 项）、总参数量相同（99,076），`requires_grad` 参数名完全相同（205 项）；同一 `build_vfm_segmentation_model` 入口。 |
| 初始化策略 | PASS（云端待执行审计） | `initialization.yaml` 声明同一 cloud-only CROMA checkpoint/audit、`strict_load=true`、target test data=false；本地未读取二进制。 |
| 优化器/预算声明 | PASS（声明层） | AdamW，lr=`1e-4`，weight decay=`0.05`，betas=`[0.9,0.999]`，cosine warmup=`0.05`，micro/effective batch=`16/32`，accumulation=`2`，AMP，24 epoch，3090 24GB。 |
| test-seal | PASS | `src/geotoken3path/utils/test_seal.py` 在非 `final_test` 或非 `final_test` seal 下拒绝 test；train/evaluate 入口在构造 split loader 前调用 guard。当前所有验证均为 validation/synthetic，`test_accessed=false`。 |
| local data / weights / GPU isolation | PASS | ResearchPilot validator：113 个 executable/config 文件，0 violation；无 `.pt/.pth/.ckpt/.safetensors` 或遥感数据二进制进入 code tree；`local_gpu_probe=forbidden_not_run`。 |
| synthetic tests | PASS | `F:\anaconda3\envs\dl_env\python.exe -m pytest tests -q`：**301 passed, 1 warning**。另有 compileall PASS。 |
| TASR synthetic liveness | PASS（仅机制合同） | `run_v11_tasr_liveness.py` 的 CPU receipt `6ada2f73cc6fe150af41f4b0060beac0245ae364dd4b7c810079fe735d397309`：zero-start identity、finite gradients、optical/SAR guidance gradient、class-agnostic affinity、patch conservation、eval bypass 均 PASS；明确不是科学结果。 |
| train/evaluate smoke | PASS（仅软件合同） | TASR synthetic train smoke 和 evaluate smoke 均返回 contract pass；没有真实数据、权重、GPU 或指标实验。 |

## Open findings

### V11-R1 — blocker：V11 clean-sync manifest 已被最近代码修改打破

文件：`F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v11_tasr_20260829.json`  
manifest SHA256（当前文件）：`55436ba3201ebe0a84bfc592524b1fbc25144c6ef80f43b379e9a58cccd98b98`  
manifest 声称 `generated_for=v11_tasr_hard_contract`、`file_count=103`，但逐文件 hash 重算发现 3 项不一致：

| 文件 | manifest hash | 当前 hash |
|---|---|---|
| `src/geotoken3path/mechanisms/tasr.py` | `6189854c173dc82f9ff52aab9907f806c01229e6a267522051edaca6cfe77282` | `e9bbe2d7783d4aa50709eba4ae6658adec82a8811d5c3ffe4632f81da134b492` |
| `src/geotoken3path/models/factory.py` | `b1357a30ea521d7afaf0eddc23d1f47b61f4db1be40cfe0fb59a915bfe80d8fe` | `9ea9f5086738182057fed9b2599e869bb4deec3eda2e045dc17775531c7d084f` |
| `src/geotoken3path/models/fusion.py` | `76d97eb1930b513b4308c765d38f3248df6c5846c3bd40a90655f2a4fb80cb1c` | `3d1bc0aa320414c6c1dd09aa6dcbdb84a93a8d78cfe30d187301daef519fe68e` |

旧 manifest 中 `factory.py` 的完整 hash 为 `b1357a30ea521d7afaf0eddc23d1f47b61f4db1be40cfe0fb59a915bfe80d8fe`；重建后的 manifest 必须以逐文件 SHA256 实际计算值为准。

原因是 manifest 生成于 19:49:49，而 TASR dtype/budget/parity 修订发生在其后。当前 manifest 不能作为 clean-sync source of truth。

最小修复：冻结源代码后重新运行 V11 manifest builder，逐文件重算并独立复核 `file_count`、禁止二进制、manifest SHA；随后以新的 manifest 重新生成 CODE_REPORT。不得复用旧 manifest SHA。

### V11-R2 — blocker：canonical `CODE_REPORT.json` 仍是 D3/CEAK 报告

当前 `F:\PRQ4\02_experiment\code\review\CODE_REPORT.json`：

- `status=PASS`；
- `route_id=R-EO-CEAK-01`，`primary_core_candidate_id=CEAK-01`；
- `reviewed_commit_or_sync_manifest=clean_sync_manifest_d3_20260829.json`；
- 生成时间 `2026-08-29T03:15:18.708954+00:00`。

该报告早于 V11 源代码、配置、测试和 manifest，不能作为当前 V11 的 code-service handoff。`v6_cc_scbc_CODE_REPORT.json` 同样是历史 V6 报告，不是替代品。

最小修复：在源代码最终冻结、manifest 重建、全量 pytest/validator 和本次独立 review 关闭前，重建当前 canonical `CODE_REPORT.json`；其中必须绑定 V11 route/candidate、当前 clean-sync manifest/hash、当前 tests/validator、local-data/GPU 状态、初始化 audit ref、training-object parity 和未解决 finding。当前应标为 BLOCKED，而不是沿用历史 PASS。

### V11-R3 — blocker：默认 run manifest 仍可能绑定旧 PCTA manifest

证据：`src/geotoken3path/utils/run_manifest.py:384-388` 的默认逻辑为：

```text
resolved.code_sync_manifest_ref
or GEOTOKEN3PATH_CODE_SYNC_MANIFEST_REF
or 02_experiment/code/manifests/clean_sync_manifest.json
```

V11 resolver 当前返回的 `resolved.code_sync_manifest_ref` 为 `None`。在没有显式环境变量时，刚生成的 V11 candidate manifest 实际记录：

```text
code_sync_manifest_ref=02_experiment/code/manifests/clean_sync_manifest.json
```

该文件的历史 route 是 `R-EO-PCTA-01/PCTA-01`，不是 V11。显式设置 `GEOTOKEN3PATH_CODE_SYNC_MANIFEST_REF=02_experiment/code/manifests/clean_sync_manifest_v11_tasr_20260829.json` 时才会得到正确 V11 ref；但当前尚无 V11 C1 guarded command，且 runner 不验证该 ref 是否存在、是否与当前 code manifest hash 匹配。

最小修复：将当前 V11 manifest ref 明确写入 V11 resolved snapshot，或在唯一 guarded V11 command 中显式 export 并加入 fail-closed assertion；新增测试要求无 override 时也不会落回 PCTA manifest，并验证 ref/manifest hash 与 code sync source 一致。完成前不能进行 C1。

### V11-R4 — blocker：24-epoch C1 不是 runner 的 fail-closed 约束

V11 route 声明 screening/formal horizon 为 24 epoch，但：

- `scripts/train.py:178` 的 `--epochs` 默认值为 `1`；
- `scripts/train.py:208` 将用户参数直接传给 `run_formal_cloud`；
- `formal_runner.py:60-68` 的 `_validate_formal_horizon` 只检查 `epochs` 为正数、rapid horizon 不超过它，并未检查它等于 resolved 的 `max_formal_epochs=24`；
- 因而形式上可以用 `epochs=1` 启动一个名义上使用 V11 配置但不满足 C1 预算的 cloud run。

这不是当前已发生的科学结果，而是会使未来 C1 失去预注册 horizon parity 的可执行漏洞。

最小修复：对 V11-C1 的 execution scale/route 让 runner 强制 `epochs==24`（或由一个不可覆盖的 V11 command 绑定并在 runner 中校验 resolved budget），同时增加负测：`epochs=1/5/23/25` 必须 fail，`epochs=24` 才可进入 cloud runner。若保留 rapid checkpoint，rapid horizon 必须作为单独审计字段，不能改变 C1 的 24-epoch主训练 horizon。

### V11-R5 — major/contract clarification required：当前守恒定义不是 raw token-logit mean

代码 `src/geotoken3path/mechanisms/tasr.py:159-177` 的实际逻辑是：

1. 将 15×15 token logits bilinear interpolate 到 120×120，得到 `base`；
2. 计算 `base_patch = patch_mean(base)`；
3. 使输出满足 `patch_mean(output - base)=0`。

这确实守恒了 **P1 bilinear dense output 的每个 8×8 patch mean**。但如果路线中的 “every token's class-wise mean semantic evidence / token mean conservation” 指的是原始 15×15 token logits，那么目前实现和测试并未验证该条件：在随机 15×15 logits 上，`bilinear(base)` 的 8×8 patch mean 与原 token logits 的差异可达约 2.19（均值约 0.37，具体数值依随机输入而定）。当前 `test_tasr_conserves_each_token_patch_mean_and_is_class_agnostic` 只检查 `output-base` 的 patch mean 为零，因此即使没有守恒 raw token mean 也会通过。

这需要在 C1 前明确而不是默认解释：

- 若科学定义是保留 P1 dense readout 的 patch mean，应将 config/test/plan wording 改为 `bilinear_patch_mean_conservation`，并把该定义写入 manifest；
- 若科学定义确实是保留原始 token logits，应重写 anchor/conservation 算子，使每一 token 对应 8×8 cell 的输出 mean 等于原始 token logit，并增加 raw-token mean 的 hard test。

在定义关闭前，不能把当前 PASS liveness 解读为 “token mean conservation PASS”。

### V11-R6 — major：发布 README 仍指向已封存的 PCTA 路线

`F:\PRQ4\02_experiment\code\README.md:5-16` 仍写着：

```text
Approved route: R-EO-PCTA-01 / PCTA-01
...
cross-modal phase-correlation transport
```

而 V11 manifest 把该 README 纳入 103 个 clean-sync 文件。该文件还给出旧的 `geotoken_3path` smoke 命令，未说明 TASR synthetic liveness、V11 route、24-epoch C1 约束或当前 sealed-test 状态。它不会改变运行代码，但会使上传的代码包向读者描述错误路线，破坏公开复现入口。

最小修复：更新 README 为 V11 TASR route，明确“仅 synthetic local checks；真实数据/权重/GPU 仅 cloud；C1 尚未执行；C2/C3 条件触发；test sealed”，并重新生成 manifest。

### V11-R7 — minor：`eta=0.25` 是代码常量，未进入 resolved TASR contract

`TASRSpatialRedistributor.__init__` 默认 `eta=0.25`，但 `v11_tasr.yaml`、resolved `tasr_contract` 和 run manifest 均未记录 `eta`。当前代码 hash 可以使整个实现可追溯，但配置层无法独立重建或审计该超参数；若后续调整 eta，`single_mechanism_diff` 仍可能只显示 mechanism_set 差异。

最小修复：把 eta 纳入 V11 config/resolved contract/run manifest，并在 resolver/runtime 校验范围与固定值；或者将它明确声明为版本化代码常量并在 contract 中记录。

### V11-R8 — note：eval bypass 关闭计算分支，但没有部署剥离接口

`CromaGeoTokenSegmentation` 在 eval 时调用 `bypass_tasr_in_eval`，因此 synthetic FP32 以及 CPU bfloat16 parity 均为 exact，且不会调用 affinity/diffusion。当前“零 inference cost”在执行路径上成立；不过 TASR 子模块仍保留在模型 state-dict/内存中，没有显式 `strip/export_deploy_model` 接口。若论文或发布包声称“auxiliary branch removed”，应增加部署剥离/导出验证；若只声称“eval path bypasses auxiliary computation”，当前证据足够但措辞应收窄。

## 数据、权重、GPU 与 test-seal 审计

- 本地 code tree 未发现真实遥感数组、标签、checkpoint 或压缩数据二进制；V11 clean-sync generator 的 forbidden suffix 检查和 ResearchPilot validator 均通过。
- `configs/model/initialization.yaml` 与 `pretrained_audit_successor.json` 只保存 cloud path、SHA256、构造器和兼容性元数据，未把权重复制到本地 code sync tree。
- 本审查未运行 `nvidia-smi`、`torch.cuda.is_available`、`torch.cuda.device_count` 或任何 GPU probing；synthetic liveness/train/evaluate 均在 CPU。
- 当前 `gate_status.json` 仍为 `INNOVATION_SCREENING_SEED0/PENDING`，V7–V10 历史记录保持不变；V11 D0 仍是 validation-only diagnostic，`scientific_result=false`，没有 C1/C2/C3 或 sealed-test 权限。
- V11 D0 的云端结果不在本审查中重新读取；其既有 receipt 只作为路线背景，不被当作 TASR 科学性能支持。

## 训练对象与协议 parity 结论

### 已确认

`resolve_v11_tasr_config` 的 baseline/candidate 仅变更 `model.mechanism_set`，共享 SEN12TS successor、CROMA audit ref、tap-connected policy、optimizer、scheduler、AMP、batch/accumulation、augmentation、24-epoch声明和 sealed-test。formal synthetic fixture 中 baseline/candidate 的 state-dict 结构与 trainability mask一致，没有外置冻结 baseline + 独立训练 router/refiner。

### 仍需在 cloud C1 前锁死

1. code-sync ref 必须绑定 V11 manifest，而非旧默认。
2. cloud command/runner 必须强制 24 epochs。
3. 使用同一 pretrained checkpoint SHA、同一 source constructor SHA、同一 data manifest SHA 和同一 protocol hash；这些只能在云端 preflight 后核实，不得在本地推断。
4. TASR eval bypass 的 AMP dtype parity 已在 CPU bfloat16 synthetic fixture 中重测为 exact；仍需在 cloud 3090 AMP preflight 中记录，不得把本地 fixture当硬件证据。

## 进入 guarded code sync 前的必需闭环

1. 停止修改源代码后重建 `clean_sync_manifest_v11_tasr_20260829.json`，逐文件复核无 stale hash、无二进制/缓存/secret。
2. 解决 V11-R3：在 resolved/guarded command/runner 中绑定 V11 manifest，并增加无 override 的回归测试。
3. 解决 V11-R4：让 V11 C1 的 24 epoch 成为 fail-closed contract。
4. 解决 V11-R5：明确守恒对象；若保留 bilinear patch mean 定义，统一改名/文档/测试，若采用 raw token mean 则修复实现和 hard test。
5. 更新 README 等发布入口，重新跑 full pytest、compileall、ResearchPilot validator。
6. 由 architecture、data/leakage、reproducibility、release/adversarial 角色重新审阅最新快照；只有无 blocker/unresolved major 才生成当前 V11 `CODE_REPORT.json`。
7. 之后才可进入 guarded code-only sync；sync 完成后先做 cloud preflight/data+pretrained audit，仍不能跳过 V11-C1 的 24-epoch seed-0 gate。C2/C3 继续保持“仅 C1≥50.0075% mIoU 后允许”。

## Final handoff state

```text
route: R-EO-TASR-01
candidate: TASR-01
local_code_contract: CONDITIONAL_PASS
reproducibility_release_gate: BLOCKED
blocking_findings: V11-R1, V11-R2, V11-R3, V11-R4
major_pending: V11-R5, V11-R6
minor_pending: V11-R7, V11-R8
local_real_data: not accessed
local_weights: not accessed
local_gpu_probe: forbidden_not_run
test_accessed: false
scientific_result: false
c1_authorized: false
c2_c3_authorized: false
sealed_test: sealed
```
