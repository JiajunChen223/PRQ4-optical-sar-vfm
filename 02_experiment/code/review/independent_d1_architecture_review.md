# D1-A/B 独立架构与科学契约审查

## 结论

总体判定：**BLOCKED（禁止据此启动或消费云端 D1-A/B 结果）**。

纯张量层面的坐标、attention 统计和 token 级 shift 约定基本正确，且当前
runner 确实是 validation-only、checkpoint-only、无训练、无 sealed-test 读取。
但是，V2 要求的 D1-B 证据尚未完整实现；更重要的是，当前所谓
valid-interior 像素评价在双线性上采样边界处仍混入被排除的 token，不能把
registration shift 与 clamped-border 内容效应完全分离。当前快照也没有被现有
clean-sync manifest 绑定，common protocol hash 只被写入输出而没有被强制核验。

因此：

- d1_diagnostics.py 的 token 坐标/统计 helper：**CONDITIONAL PASS**；
- diagnose_d1_ab.py 的完整 D1-A/B 云端执行契约：**BLOCKED**；
- 在修复下列 blocker、生成新的代码同步 manifest 并重新做独立审查前，不得运行
  或解释 D1-A/B 的科学结果。

## 审查边界与证据

审查对象是 F:\PRQ4\02_experiment\code 当前快照，参照附件
PLAN_REVISION_PRQ4_V2（附件 SHA256：
b19d7a7c8c370f303acba8479c1d7c48f7dd592709e1bb320a68a88f98952284）及其项目
副本 02_experiment/reports/representation_state_revision_plan_prq4_v2.json。
覆盖：

- src/geotoken3path/d1_diagnostics.py；
- scripts/diagnose_d1_ab.py；
- src/geotoken3path/models/fusion.py 中 ceak_dense_cross_attention 和
  d1_attention_weights telemetry；
- D1 helper tests、当前代码 validator、既有 C1/D0 manifest 和 D0-C 结果中的
  checkpoint/protocol 绑定。

当前快照的 SHA256（用于本次审查定位）：

| 文件 | SHA256 | 字节数 |
|---|---|---:|
| src/geotoken3path/d1_diagnostics.py | 18aed2e70f06f18ff4afcab2a5b9246dbc70b7531de1b6f0992b4db08bef307c | 6075 |
| scripts/diagnose_d1_ab.py | 52c0351d3de90ca68f8309d20b48f60694f003729bcd475d719a9c0834f4f095 | 19892 |
| src/geotoken3path/models/fusion.py | 68bcbf2da48ae5b0d94d4e032971c01d995c948f834bfe183b5ffb93b435d059 | 84371 |
| tests/unit/test_d1_diagnostics.py | 786fe5446e3d5378030a1507c111234ec46481c51e55500644acade31a6ed9f6 | 2972 |
| tests/unit/test_d1_runner_helpers.py | aa6fa7574035739ab69aee33c306891c9600c33db84e7957a353e1e867221f08 | 2228 |

已做的只读检查：

- D1 两组 unit tests：**7 passed**；
- 三个目标 Python 文件 py_compile：**pass**；
- 02_experiment/code/review/validate_code_project_d1_20260828.json：**pass**，
  88 个 executable/config 文件、0 个静态 violation；
- 未读取本地真实数据或权重，未探测本地 GPU，未训练，未评估 sealed-test。

## 通过项

### D1-A attention geometry

1. attention_statistics() 在 d1_diagnostics.py:51-71 对 [B,N,N] 的
   row-stochastic attention 计算了 H/log(N)、A_ii 和
   dist(i,j)<=r 的 radius-1/2/3 mass；uniform_local_mass()
   (:74-85) 的 uniform 期望也按同一 row-major square grid 计算。与 V2 的
   normalized entropy、same-index mass、local-neighborhood mass 定义一致。
2. fusion.py:934-956 的 C1 分支使用
   softmax(Q_o K_s^T/sqrt(d))，diagnose_d1_ab.py:990-991 只在
   ceak_dense_cross_attention（无 conflict/null/private）分支暴露
   d1_attention_weights，且 telemetry 是 detached，不改变模型前向或训练。
3. 当前 D1 代码新增到 fusion.py 的差异仅是 d1_attention_weights 的
   detached 输出；相对于 D0 归档包的旧 fusion.py，没有发现改变 C1 前向数值
   的其它差异。这降低了“旧 checkpoint 配新计算”的即时科学风险，但不替代
   新快照的 manifest 绑定（见 D1-P02）。

### D1-B token coordinate/sign convention

1. token_grid_coordinates()（d1_diagnostics.py:29-40）返回 row-major 的
   (x,y) 坐标；attention_expected_displacement()（:88-94）的广播结果
   对 query i、key j 实际计算 coord[j]-coord[i]，与 V2 的
   delta_ij=(x_j-x_i,y_j-y_i) 一致。
2. shift_token_grid()（src/geotoken3path/diagnostics.py:56-77）的正向
   (dx,dy) 是输出位置采样 source (x-dx,y-dy)；因此未被 clamp 的输出内部
   区域中，原内容向正方向位移，对应 barycenter 的 key offset 为正方向。
   valid_token_mask()（d1_diagnostics.py:43-48）选择的是未发生边界 clamp 的
   输出 query 区域，正负方向边界对称。
3. D1_SHIFT_GRID（diagnose_d1_ab.py:43-59）与 V2 的 15 个 shift 完全一致，
   包括 +/-x、+/-y 和两组对角线；runner 只用 validation split，固定
   seed=0、micro-batch 16 和 180 个 validation parent。

### checkpoint-only / no-training / sealed-test

diagnose_d1_ab.py:103-128 对 C1 checkpoint 做 SHA256 和 strict state-dict load；
_evaluate_grid（:200-249）在 torch.no_grad() 中只做 bridge/token inference；
run()（:312-340）请求 validation loader、拒绝非 CUDA，并通过
assert_test_access_allowed(..., "validation") 保持 test seal。没有 optimizer、
backward 或训练入口，输出也明确写入 scientific_result=false、
training=false、test_accessed=false（:365-433）。这些部分符合 V2 hard-stop
边界，但不抵消下列科学/绑定问题。

## Findings

| ID | 严重性 | 具体发现 | 对 V2 的影响与处理 |
|---|---|---|---|
| D1-B01 | **blocker** | V2 要求 D1-B 报告 spearman_correlation 和 calibration plot。summarize_shift_recovery()（d1_diagnostics.py:97-128）只返回 predicted mean、RMSE、directional accuracy、magnitude；runner（diagnose_d1_ab.py:252-300）也没有 Spearman、校准分箱/曲线或 raw per-query displacement 输出。 | D1-B 的核心 recovery 证据不完整，且当前 JSON 无法事后重建这两个量。必须增加非零 shift 的逐 query/per-sample recovery 表或可重算 raw artifact、Spearman 及明确的 calibration 表/图后才可执行。 |
| D1-B02 | **blocker** | “valid interior” mask 只把 token mask 扩展成像素块（diagnose_d1_ab.py:169-175），但 OpticalSarTokenModel.forward()（fusion.py:1567-1571）在 mask 前用 bilinear, align_corners=False 从 15×15 上采样到 120×120。15→120 时第一列被标记为 valid 的像素（例如输出 x=8）仍混合相邻被排除 token（输入坐标约 0.5625）；因此 valid_interior_pixel_count_per_sample 并不保证所有参与预测的 token 都是 valid。 | 边界 duplication 与 shift effect 尚未被严格分离，D1-B 的 mIoU/OA shift curve 可能含边界伪影。应改为 token-level/conservative interpolation-support mask、对 valid 区域腐蚀一 token，或用 nearest/完全 valid 的上采样支持后再计分，并增加回归测试。 |
| D1-P01 | **major** | runner 把 EXPECTED_COMMON_PROTOCOL_SHA256 写入输出（:385-395），但 _load_c1_model() 只返回 resolved.get("matched_common_protocol_sha256")（:121-128），没有断言它等于 expected hash；也没有在 run() 中像 D0 runner 那样对实际 resolved hash 做 fail-closed 集合/精确匹配。 | 本次直接解析当前 config 得到的 hash 确实为 71d1665b...e31a8，与常量一致；但任何 config/protocol drift 都会被错误地标成冻结 hash。必须在加载后强制比较实际 resolved hash、dataset/shape/route 等绑定字段，比较失败即停止。 |
| D1-P02 | **major** | 当前 D1 文件不在现有 D0/CEAK clean-sync manifest 中：clean_sync_manifest_d0_20260828.json 未列出 d1_diagnostics.py、diagnose_d1_ab.py 或 D1 tests；当前 fusion.py hash 为 68bcbf...，旧 D0 manifest 记录为 2f8a22...。虽然两者 diff 仅为 detached telemetry，但当前 D1 runner 没有记录 code manifest/hash。 | 当前快照尚未形成可同步、可复核的正式代码释放。必须生成包含 D1 文件的新 clean-sync manifest，更新 code review/CODE_REPORT，并让云端执行控制绑定该 manifest；在此之前禁止把云端输出作为本审查快照的结果。 |
| D1-P03 | **major** | _cloud_path()（diagnose_d1_ab.py:66-73）只做词法前缀检查，未拒绝 ..、未 resolve/检查 symlink。形如 /root/autodl-tmp/../../... 可通过；输入/输出 containment（:309-313）也在未 resolve 的路径上比较。 | 可能越过声明的 cloud roots，或通过路径别名把 output 指向输入/已有 artifact，违反 cloud/output 边界。应规范化并拒绝 ../symlink，resolve 后再做 containment，且对 output 要求独立、预先不存在的目录。 |
| D1-O01 | **major** | runner 用固定文件名写结果（diagnose_d1_ab.py:434-436），mkdir(..., exist_ok=True) 且不检查已有 output；重复执行会覆盖同一个 evidence artifact，也没有 runner/run id 或 output hash。 | 违反证据不可覆盖和可追溯性要求。应使用显式唯一 run/output 目录，发现已有目标即 fail closed，并记录 code/data/audit/checkpoint/output 绑定。 |
| D1-U01 | **minor** | recovery 的 dx、dy、predicted_displacement_mean、RMSE 和 magnitude 没有 JSON unit 字段；代码实际使用的是 15×15 token-grid units，不是 Sentinel 像素或米。 | 容易把 token displacement 误读为 pixel/地理单位。输出应显式标注 coordinate_system=token_grid_xy、displacement_unit=token，并单独保留 120×120 pixel mask 计数。 |
| D1-A01 | **minor** | _check_attention()（d1_diagnostics.py:16-26）检查 finite 和 row sum，但不拒绝负 attention。负值会被 clamp_min() 后参与 entropy，可能产生负 entropy；当前 fusion softmax 保证真实 C1 telemetry 非负，因此这是 helper 的 fail-closed 缺口。 | 增加 attention >= 0 检查并测试 malformed input。否则纯 helper 不完全满足 attention probability contract。 |
| D1-B03 | **minor** | summarize_shift_recovery() 在 (dx,dy)=(0,0) 时无条件把 directional accuracy 设为 1（:115-119）；同时 directional accuracy 只是 dot-product>0，没有角度/轴向定义。 | 零 shift 方向准确率应标记 N/A，不应与非零 shift 聚合；应在输出中明确指标定义和适用范围。 |
| D1-O02 | **minor** | 输出顶层 status 固定为 "pass"（diagnose_d1_ab.py:365-372），但 decision.d1_status 明确仍是 pending_d1c_matched_topology_controls（:414-427），且 D1-B 必需量尚未实现。 | 容易把“runner 完成一次执行”误解为“D1 已通过”。建议改成 pass_d1ab_execution_only/conditional，保留 scientific_result=false 和 D1C pending。 |

## 对要求项目的逐项判定

| V2 项目 | 判定 | 说明 |
|---|---|---|
| normalized entropy | **通过（有 helper 输入校验缺口）** | 公式和 log(N) 归一化正确。 |
| same-index mass | **通过** | 使用 zero-shift A.diagonal()；输出名称明确。 |
| local mass r=1/2/3 | **通过但比较不完整** | Euclidean token-grid neighborhood 和 uniform mean 已计算，但未直接输出 observed - uniform 或 ratio。 |
| barycenter delta_ij | **token 级通过** | key-query、(x,y) 和正负 shift 符号一致。 |
| recovery RMSE/directional accuracy | **部分通过** | 已实现，但 zero-shift N/A 和方向定义需修订。 |
| Spearman | **不通过** | 缺失。 |
| calibration | **不通过** | 缺失且没有 raw 数据可重建。 |
| symmetric shift grid | **通过** | 15 个 V2 shift 均在常量中。 |
| valid-interior evaluation | **不通过** | token mask 正确，但 bilinear output boundary contamination 未处理。 |
| C1 checkpoint-only | **通过（绑定需加强）** | strict load 和预期 checkpoint hash 存在；缺 run-manifest/code-manifest 绑定。 |
| no training | **通过** | no-grad、无 optimizer/backward/training call。 |
| no sealed-test | **通过当前路径** | 只请求 validation，seal guard 存在；路径 traversal 风险仍需修复。 |
| protocol/code hash | **不通过** | 当前 hash 值一致，但没有 fail-closed assertion，且 D1 snapshot 未进 sync manifest。 |

## 最小修复顺序

1. 修复 valid-interior 的 interpolation-support 定义，并补充一个 15→120 的
   边界混合回归测试。
2. 增加 D1-B 的 per-query recovery artifact、非零 shift Spearman、校准分箱/图，
   明确 token-unit 与 zero-shift N/A 语义。
3. 对实际 resolved common hash 做强制精确比较；同时绑定 route、dataset、15×15、
   C1 checkpoint provenance 和当前 code manifest。
4. 收紧 POSIX path：拒绝 .. / symlink，resolve 后检查 roots 和 output/input
   containment；output 目录预存在时 fail closed，禁止覆盖旧 evidence。
5. 生成新的 D1 clean-sync manifest/CODE_REPORT，并在重新审查通过后才允许
   D1-A/B 云端 validation-only 执行。D1-C G0–G5 训练仍须保持在 D1-A/B 审查
   之后，不能由本 runner 自动升级。

## 最终状态

CONDITIONAL_PASS_FOR_TOKEN_HELPERS; BLOCKED_FOR_D1_AB_CLOUD_EXECUTION。

本报告没有授权训练、候选晋升、composition、multi-seed、confirmation 或
sealed-test；V2 的 D1 决策仍保持未决定。
