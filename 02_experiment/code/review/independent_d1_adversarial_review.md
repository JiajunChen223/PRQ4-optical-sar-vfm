# D1-A/B 独立对抗审查

审查日期：2026-08-28  
审查根目录：`F:\PRQ4`  
代码根目录：`F:\PRQ4\02_experiment\code`  
审查方式：只读静态审查 + 不生成缓存的 CPU synthetic tests；未读取本地真实数据/权重，未 SSH、未同步、未探测本地 GPU、未训练。

## 结论

当前 D1 runner **BLOCKED_FOR_D1_CLOUD_EXECUTION**。attention 行归一化、validation-only/test seal 和 SHA 常量的基本防线存在，但 valid-interior mask 与 feature shift 的坐标约定相反，会使所有非零 shift 的 D1-B 恢复和 masked mIoU 落在错误的边界区域。另有输出覆盖、cloud path 穿越和运行时 provenance 绑定缺口。

本审查没有发现 G0-G5 已执行的证据。当前批准计划仍是“D1-A/B 后再执行 matched G0-G5”，而代码 factory 也尚未实现 G1-G4 的 mechanism set；不能把 D1 receipt 中的 `g0_g5_training_allowed_after_review=true` 当作 G0-G5 已执行或无条件授权。

## 审查快照和边界

- D1 runner：`scripts/diagnose_d1_ab.py`，SHA256 `52c0351d3de90ca68f8309d20b48f60694f003729bcd475d719a9c0834f4f095`。
- D1 helpers：`src/geotoken3path/d1_diagnostics.py`，SHA256 `18aed2e70f06f18ff4afcab2a5b9246dbc70b7531de1b6f0992b4db08bef307c`。
- Shift helper：`src/geotoken3path/diagnostics.py`，SHA256 `207f117860e229f65ea1f047cf5664e152cc33cd5ce0a44172d11f17cecda2fc`。
- `validate_code_project_d1_20260828.json` 报告 88 个 executable/config 文件、0 个静态 violation、`local_real_data_allowed=false`、`local_gpu_probe=forbidden_not_run`；这只证明静态/本地代码约束，不证明云端运行成功。
- `representation_state_revision_plan_prq4_v2.json` 明确要求 D1-A/B 后再做 G0-G5，且规定 barycenter 还应包含 Spearman correlation 和 calibration。
- `gate_status.json` 的当前记录仍为 `CORE_CODE/PENDING`、test seal `sealed`；最近的 INNOVATION_REVIEW 记录要求先完成 D1 代码 review/sync。

## 测试收据

使用 `F:\anaconda3\envs\dl_env\python.exe -B`、`PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`：

| 检查 | 结果 |
|---|---:|
| D1/D0 diagnostics、test seal targeted tests | **20 passed** |
| 全部 `02_experiment/code/tests` synthetic suite | **250 passed**, 1 个既有 UserWarning |
| `valid_token_mask` 与 `shift_token_grid` 坐标夹具 | 发现反号，见 F1 |
| `_cloud_path('/root/autodl-tmp/../../etc/passwd', ...)` | **错误地 ACCEPT**，见 F3 |
| resolver 对 G1-G4 mechanism set | **全部 REJECT**，见 F5 |

测试没有调用 D1 `run()` 的真实云端路径，因为所需数据 manifest、C1 checkpoint 和 CROMA 权重按策略必须留在云端；因此不存在 D1 scientific output 可供消费。

## 已通过的防线

1. **Test seal：基本通过。** D1 在构造 loader 前调用 `assert_test_access_allowed`，固定为 `screening/sealed`；SEN12TS loader 只允许 train/validation，并由 loader 再次检查 seal。D1 不请求 test split，也把 `test_accessed` 固定为 false。
2. **数据/C1 checkpoint SHA：代码层面通过。** runner 对 data manifest 使用完整 SHA256 `bd1c6f...49967`，对 C1 checkpoint 使用完整 SHA256 `29befe...d8e2e9`，并对 C1 state dict 使用 `strict=True`。CROMA loader 也会核对审计中声明的 SHA 和实际 checkpoint，并严格加载五个 nested blocks。由于二进制只在云端，本次不能对实际字节重新计算。
3. **attention 行和 coordinate 基础实现：部分通过。** C1 的 `softmax` 产出理论上是非负且逐行和为 1；`_check_attention` 检查 `[B,N,N]`、有限值和行和；`token_grid_coordinates` 的输出顺序为 `(x,y)`，expected offset 为 key-query，和 helper 测试一致。
4. **本地数据/GPU政策：静态通过。** 没有执行 `nvidia-smi` 或 CUDA capability probe，也没有把真实数据/权重放入本地代码树；D1 入口只接受声明的 POSIX cloud-root 路径。但路径 containment 仍有 F3 的漏洞。

## Findings

### F1 — [P1 / BLOCKER] valid-interior mask 与 shift sign 相反

证据：

- `diagnostics.py:56-63` 定义 `shift_token_grid`：输出 `(y,x)` 读取源 `(y-dy,x-dx)`；因此源 token 被移动到输出坐标 `(x+dx,y+dy)`。
- `d1_diagnostics.py:43-48` 的 `valid_token_mask` 却选择 `x >= dx`、`x < side + min(0,dx)`，等价于要求 `(x-dx,y-dy)` 在网格内。它应当检查 `(x+dx,y+dy)` 在网格内，或调用 helper 时传 `(-dx,-dy)`。
- `d1_diagnostics.py:111-113` 的 recovery target 明确设为 `[dx,dy]`，因此 recovery 的正号与 feature 的正向移动一致，进一步确认 mask 反号。
- runner 在 `diagnose_d1_ab.py:236-239` 用该 mask 屏蔽像素，在 `:263-270` 用同一 mask 做 barycenter recovery；这会同时污染 D1-B 的两个结果。

合成夹具结果：

```text
shift +1 grid: [[0,0,1], [3,3,4], [6,6,7]]
implemented mask +1: [[F,T,T], [F,T,T], [F,T,T]]
```

在 15x15 网格中，`dx=+1` 的实现 mask 与正确 mask 都有 210 个 token，但交集只有 195、对称差 30；`(2,2)` 的交集为 121/169、对称差 96。也就是说，仅检查 count 的现有测试会通过，但边界方向错了。

建议：统一“`dx,dy` 是 SAR feature 的输出位移”定义，改为检查 `x+dx`/`y+dy` 的范围；增加 one-hot/coordinate impulse test，逐项断言正负 shift 的有效查询集合和 expected displacement 方向。修复前不得消费任何 D1 shift metric 或做 D1 决策。

### F2 — [P1] output artifact 会无条件覆盖历史收据

证据：`diagnose_d1_ab.py:312` 使用 `mkdir(..., exist_ok=True)`，`:435-437` 对固定文件名 `d1_dense_gain_representation_audit.json` 直接 `write_text`。同一 `--output-dir` 的重跑、并发运行或旧输出复用会覆盖既有审计；没有 run id、existing-output rejection、atomic temp+rename、completion marker 或输出 SHA manifest。

建议：要求全新、唯一、空的 output directory；若目标 JSON 已存在立即拒绝；写入临时文件后 fsync/atomic rename，并把输入三件套、代码 sync manifest、plan hash、实际 CROMA SHA 和输出 digest 写入不可变 receipt。此项与 F3 组合时可能导致通过路径别名覆盖不应覆盖的目录。

### F3 — [P1] `_cloud_path` 可被 `..` 和 symlink 绕过 cloud-root containment

证据：`diagnose_d1_ab.py:66-73` 仅检查原始字符串首段和 `PurePosixPath.parts[1:3]`，没有拒绝 `..`、没有 `resolve()` 后再验证，也没有拒绝输入/输出路径及其父目录中的 symlink。只读夹具中：

```text
/root/autodl-tmp/../etc/passwd   => ACCEPT
/root/autodl-tmp/../../etc/passwd => ACCEPT
/root/autodl-tmp/link/../x       => ACCEPT
```

随后 `_sha256_file`、`read_text`、`mkdir` 和 `write_text` 都会按实际 filesystem 语义使用这些路径。SEN12TS manifest 内部对象路径有较严格的 lexical root 检查，但 D1 的 manifest/audit/checkpoint/output 顶层路径仍暴露该问题。

建议：对输入做 `Path.resolve(strict=False)`，要求 resolved path 位于允许的 `/root/autodl-tmp` 或明确批准的 `/root/autodl-workspace` 下，拒绝任何 `..`、path/symlink parent escape，并对 output 做同样的 canonical containment 检查；输入 hash 和实际读取最好使用同一打开句柄或二次 hash，避免 TOCTOU。

### F4 — [P1] runner 未在运行时绑定 approved plan、common protocol 和 CROMA audit identity

证据：

- `diagnose_d1_ab.py:106-109` 只按当前代码树 resolve `ceak_dense_cross_attention`，并直接接受 CLI 的 `audit_path`；没有核对 `initialization.yaml` 中的 `audit_report`、C1 run manifest、PLAN_REVISION_PRQ4_V2 hash 或 D1 code-sync manifest。
- `:110-127` 核对 C1 checkpoint SHA，但只把 `resolved_common_protocol_sha256` 记录到 metadata，没有断言其等于 frozen `EXPECTED_COMMON_PROTOCOL_SHA256`，也没有断言 `croma_load.checkpoint_sha256` 等于 approved CROMA `0238d814...3b63`。
- 输出 `:387` 无条件写入 frozen protocol 常量。因此当前配置/审计被替换时，runner 可能先产生 `status=pass` 的非匹配 receipt；后续 validator 即使能抓到部分 mismatch，也不是运行时 fail-closed。

建议：在加载前要求 resolved common hash、plan/approval hash、D1 code manifest hash、initialization audit ref、CROMA checkpoint SHA 和 C1 checkpoint provenance 全部精确匹配；将这些实际值写入 receipt，任一缺失/不匹配立即失败。同步前必须使用包含 D1 文件的新 clean-sync manifest，不能复用 D0 的 75-file manifest。

### F5 — [P1 / G0-G5 BLOCKER] G1-G4 尚未实现，当前 factory 无法执行计划中的 G0-G5

证据：

- approved plan `representation_state_revision_plan_prq4_v2.json:32-38` 定义 G0–G5，其中 G1–G4 为 `d1_optical_dense_self_attention`、`d1_global_sar_mean_context`、`d1_query_independent_sar_pooling` 和 `d1_local_query_conditioned_cross_attention`。
- 当前 `fusion.py:28-105` 的 `VALID_MECHANISMS` 不包含上述四个名称；`resolve_approved_config` 对四者的只读调用均返回 `ConfigContractError: mechanism_set is not approved`。只有 `always_fuse` 和 `ceak_dense_cross_attention` 能通过当前 resolver。
- 代码/云 command/report 搜索没有发现 G0-G5 run output、run manifest、checkpoint 或 24-epoch curves；当前 D1 runner 自己也只做 C1 的 D1-A/B，输出 `d1_status=pending_d1c_matched_topology_controls`。

建议：保持 G0-G5 状态为 **NOT EXECUTED / PENDING**；先为 G1-G4 建立同一 entrypoint、同 CROMA/decoder/optimizer/scheduler/trainability、24 epoch/seed 0、validation-only 的配置和 mechanism implementations，生成新的 code sync/review receipt 后再执行。任何 G0-G5 结果都只能是 matched controls，不能被当作新 candidate promotion。

### F6 — [P2] approved D1 barycenter metric contract 未完整实现

计划 `representation_state_revision_plan_prq4_v2.json:25-26` 要求 `expected_displacement`、RMSE、directional accuracy、`spearman_correlation` 和 `calibration`。但 `d1_diagnostics.py:97-128` 只返回 predicted mean、RMSE、directional accuracy 和 predicted magnitude；runner 也没有 Spearman 或 calibration 字段。D1-A/B 即使修复 F1，也不能完整应用计划中的 decision contract。

建议：定义 calibration 的 reference/分箱规则和 Spearman 的样本粒度，输出完整字段及 finite/count checks；在 validator 中要求它们存在。

### F7 — [P2] attention contract 允许负权重，且 receipt 不记录归一化证据

`d1_diagnostics.py:16-25` 只检查有限值和 row sum≈1，没有 `attention >= 0` 检查。合成行 `[1.2,-0.2,0,0]` 的 row sum 为 1，当前 helper 接受它并给出 expected displacement；`:59` 又用 `clamp_min` 计算 entropy，会掩盖负值。生产 C1 路径来自 softmax，故本项是 fail-closed contract gap 而非已证实的 C1 输出错误。

建议：拒绝负权重；记录每批次/全局 row-sum min/max、最大误差和最小 attention value，使独立 validator 能审计全量归一化，而不是只看到 summary mean。

### F8 — [P2] final padded batch 的 metric 正确性依赖未显式验证的隐含约定

`sen12ts.py:322-340` 会把 validation 最后一批补到 16，并把 padded target 填为 255；runner `diagnose_d1_ab.py:212-239` 对 16 行都前向和累计 confusion matrix，只在 `:242-245` 截断 attention/statistics 到 `valid_count`。当前固定 collate 因 padded target=255 通常使 metric 不计入 padded 行，但 `_collect_validation_batches` 没有验证 batch shape、`count` 与真实样本数或 `target[count:]` 的 ignore sentinel。若 collate/loader 变化，metric 和 telemetry 会悄悄使用不同样本集合。

建议：保留 full-16 normalization/forward 语义，但在 runner 中显式断言最后 `target[count:] == 255`，并对 logits/target 的 metric 贡献按 `valid_count` 明确裁剪或 mask；增加专门的 4+12 padding synthetic test。

### F9 — [P2] `robustness_AUC_mIoU_percent` 实际是无权算术均值

`diagnose_d1_ab.py:290-298` 将 15 个未按 shift magnitude 排序、且包含多个重复 magnitude 的 mIoU 直接求均值后命名为 AUC。它不是按位移幅度积分的 AUC，也没有按有效像素数加权。若下游把该字段当 AUC 做决策，统计含义不成立。

建议：若需要 AUC，先按 magnitude 聚合方向，再按明确横轴做 trapezoid integration；否则改名为 `mean_grid_mIoU_percent` 并在决策规则中使用正确的 summary。

### F10 — [P2] device policy 检查晚于 validation data read

`diagnose_d1_ab.py:316-328` 先构造 loader 并收集 180 个 validation rows，`:329-331` 才拒绝非-CUDA device。`--device=cpu` 会先读取完整 validation 数据再失败；这不构成已发生的本地数据泄露，但违背“先确认 cloud CUDA 再读真实数据”的 fail-closed 顺序。代码也没有绑定 external cloud preflight 的实际 GPU identity。

建议：在 loader/data read 前解析并检查 `device.type == 'cuda'`，再要求外部 cloud preflight/run manifest 明确 approved GPU；本地 synthetic tests 继续保持 CPU-only。

## 最小修复与放行条件

按以下顺序完成后，才可重新做 D1 cloud-only validation：

1. 修正 F1，并加入正/负 shift 的坐标、one-hot movement、valid mask 和 recovery 单元测试。
2. 修正 F2/F3，使用 canonical cloud paths、唯一空 output、不可覆盖和 atomic receipt。
3. 修正 F4，绑定 PLAN_REVISION_PRQ4_V2、D1 clean-sync manifest、C1/CROMA/data SHA 和 common protocol hash。
4. 补全 F6-F8 的 metric/contract tests，并把 F9 字段改名或改为真正的 AUC。
5. 另行实现和审计 G1-G4 后，生成 G0-G5 的 matched-control command/manifest；在此之前禁止训练、排序、候选晋级、composition、confirmation 和 final-test。

## 最终状态矩阵

| 项目 | 状态 | 说明 |
|---|---|---|
| D1-A/B 本地代码/synthetic contract | CONDITIONAL | 250 tests pass，但 F1/F2/F3/F4 未闭合 |
| D1 cloud execution | **BLOCKED** | 未执行；真实数据/权重只允许云端 |
| data/C1 SHA constants | PASS (code-level) | 实际字节未在本地读取 |
| sealed-test guard | PASS (static/unit) | 无 test split 请求 |
| local real-data/GPU policy | PASS (static only) | 未做本地 GPU probe；F3/F10 仍需修复 |
| G0-G5 | **NOT EXECUTED / PENDING** | G1-G4 当前 resolver 不支持 |
| candidate/composition/confirmation/final test | CLOSED | 保持 D0/D1 边界，不产生科学结论 |

