# GeoToken-3Path 独立代码审查

## 结论

**BLOCK**

当前代码是一个边界合规的合成原型，但尚不能进入代码同步、云端数据/权重预检或基线训练。决定性原因不是单元测试失败，而是当前信息流没有实现获批路线的核心语义：所谓三态路由仍是三个路径的连续软混合，所有路径均被无条件计算，因而既不是“每 token 选择一态”，也没有形成可测的计算路由或激活预算。正式模型配置也未接入模型工厂，当前训练/评估入口只运行硬编码的随机张量。

本结论仅评价软件与协议实现，不评价科学假设是否成立，不产生任何基线、创新、效率或泛化结论。

## 审查边界与快照

- 审查根目录：`F:\PRQ4\02_experiment\code`
- 审查方式：静态源码/配置审查 + CPU 合成张量检查；未下载数据或权重，未访问真实数据，未探测 GPU，未训练。
- 冻结快照：
  - `fusion.py` SHA256：`5C23A359A4197036BF8D02754678ECB0D7109294951501C30DD882ADE357CAE6`
  - `factory.py` SHA256：`D87F7A7CCB74B5BCD3663382D490B6EEC5C78C137E2E2ACE91D09CB8EA05FFFE`
  - `test_model_factory.py` SHA256：`DE64B98193FF207442894E48EDB2AFB080E37836FC4FC36C7A8DA92FCC549F8A`
- 独立测试命令：`F:\anaconda3\envs\dl_env\python.exe -B -m pytest -p no:cacheprovider F:\PRQ4\02_experiment\code\tests\unit -q`
- 测试结果：`9 passed in 1.82s`。这些测试是合成接口证据，不是训练或实验结果。

## 已确认的合规点

1. baseline、candidate 和 `static_sparse` 由同一个 `build_model`/`OpticalSarTokenModel` 工厂构造，参数键表面一致。
2. 路由器位于模型内部，不是冻结 baseline 外挂的单独可训练模型。
3. `test_seal.py` 对非 `final_test/final_test` 的 test 请求实施运行时拒绝，现有 3 项 seal 单测通过。
4. 初始化接口会拒绝 `pending` 审计，默认 `strict=True`，且本地没有 checkpoint 二进制。
5. 当前入口明确拒绝非 smoke 执行；因此本地原型没有越权训练。

这些合规点不足以抵消下述阻断项。

## Findings

### CR-GT3P-001 — 核心三态机制未按批准语义实现

- role：architecture
- severity：**blocker**
- path：`src/geotoken3path/models/fusion.py:42-115`
- status：open
- finding：
  - `geotoken_3path` 使用三个严格正概率的 softmax/加权和，而不是对每个空间 token 选择 `bypass/current/escalation` 中的一态。独立合成检查得到 `all_one_hot=False`、每 token 平均非零状态数 `3.0`。
  - `_paths()` 在判定机制前先计算 `current` 和 `finer`；即使 `always_fuse` 也会执行 `sar_escalation`。forward hook 证据为 `{'exchange': 1, 'escalation': 1}`。因此 `active_fraction` 只是概率统计，不控制实际算子执行、显存或延迟。
  - `_finer_scale_context()` 只是同一 token 网格上的 3x3 平均池化；没有输入/访问更细尺度特征。非平方 token 数时直接返回原 SAR token，所谓 escalation 退化为另一条同位置线性投影。
  - `current_scale_local_exchange` 实际是逐 token 的 `nn.Linear`，没有使用配置中的 `local_window_tokens: 49`，也没有局部邻域对应关系。
  - 规划要求“per-stage”，当前模型只有一个 fusion block，配置的 `stages: [mid, late]` 未被消费。
  - 所有路径最后统一经过 `LayerNorm`；当前实现没有单独可审计的恒等 residual write-back，不能把“identity residual invariant”视为已实现。
- evidence：`fusion.py:42-55,57-74,90-115`；`configs/model/geotoken3path.yaml:10-20`；独立合成检查如上。
- proposed_fix：
  1. 将路由定义为真正的每 token、每 stage 单态选择，例如训练时 straight-through top-1/容量受限分配，推理时确定性 hard route；明确 tie-break 与温度退火只属于训练实现，不改变主机制。
  2. 使用按路由索引的 gather/dispatch/scatter，仅对进入 current/escalation 的 token 执行相应算子；bypass token 不得计算跨模态分支。
  3. 在每个 stage 设置可验证容量上限，分别报告 requested/realized active tokens、分支调用量、理论 MAC 与实测 wall time/VRAM。预算必须约束实际执行图，而不只是概率均值。
  4. current 路径实现配置锁定的局部邻域交换；escalation 路径必须接入真实更细尺度特征或将路线/命名退回审批重新定义，不能用同尺度平均池化代替。
  5. 将 optical identity residual 作为 fusion block 外部的显式加法路径，并为 identity 保留写单测。
- retest：新增 one-hot/单态、branch non-invocation、每 stage 容量、局部窗口、真实跨尺度输入、非平方 token、identity-preservation 测试；使用 forward hooks 或 profiler 证明被 bypass 的分支未执行。

### CR-GT3P-002 — “静态稀疏”控制不是稀疏路由，关键硬证伪控制缺失

- role：architecture/reproducibility
- severity：**blocker**
- path：`src/geotoken3path/models/fusion.py:90-102`; `configs/model/geotoken3path.yaml:22-27`; `configs/experiment/approved_route.yaml:10-15`
- status：open
- finding：`static_sparse` 给每个 token 固定 `[1-budget, budget, 0]` 并做分数混合。独立检查的全部 route vector 均为 `[0.5, 0.5, 0.0]`；没有任何 token 被真正 bypass，也没有减少 exchange 调用。它不能作为 budget-matched 静态稀疏控制。模型工厂和 CLI 也未实现规划中的 `unimodal_optical`、`random_budget`、`local_exchange_without_state_machine` 三个控制。
- evidence：`fusion.py:96-102`；`geotoken3path.yaml:22-27`；`train.py:21`；`evaluate.py:20`。
- proposed_fix：以相同 active-token 容量实现确定性静态 mask、固定 seed 随机 mask、always-fuse、纯 optical、无状态机 local exchange；所有控制通过同一工厂、入口、初始化、优化器、预算 hash 和评估器解析。静态/随机控制必须和 candidate 使用相同 dispatch 算子，仅路由决策来源不同。
- retest：逐控制核验精确活跃 token 数、实际分支调用数、参数/初始化/协议 hash 一致；禁止用连续权重混合冒充稀疏控制。

### CR-GT3P-003 — 正式配置没有驱动模型工厂，resolved diff 不成立

- role：reproducibility
- severity：**blocker**
- path：`src/geotoken3path/models/factory.py:11-24`; `configs/model/geotoken3path.yaml:1-31`; `scripts/train.py:19-33`; `scripts/evaluate.py:18-26`
- status：open
- finding：工厂只读取 `mechanism_set/token_dim/num_classes/active_budget`，但正式 YAML 使用 `mechanism.name`、`mechanism.expected_active_fraction_budget`、`input.*`、`trainability.*` 等字段。将正式 YAML 直接传给 `build_model` 的独立检查解析为 `always_fuse, dim=32, classes=19, budget=0.5`，即 candidate 配置静默退回 baseline 默认值。训练/评估脚本也不加载任何 YAML，而是硬编码 `token_dim=32`、随机张量和 seed 0。因此 README 的“configuration-driven”陈述尚不成立，baseline/candidate resolved config diff 和 immutable snapshot 均不存在。
- evidence：独立解析输出 `{'resolved_mechanism':'always_fuse','resolved_dim':32,'resolved_classes':19,'resolved_budget':0.5}`；`factory.py:14-23`；`train.py:20-33`。
- proposed_fix：建立单一配置 schema/resolver，禁止未知字段与默认静默回退；由 approved-route composition 解析 dataset/model/initialization/runtime/mechanism set，生成不可变 resolved config、config diff 和 matched protocol/budget hash。训练与评估入口只接受配置/manifest 引用，工厂必须消费同一个 resolved model object。
- retest：加载正式 YAML 后应解析为 `geotoken_3path`，并精确消费 stages/window/budget/channels/trainability；删除任一必填字段或拼错字段必须 fail-closed；baseline/candidate diff 除机制集外为空。

### CR-GT3P-004 — 当前不是可训练的遥感 VFM 语义分割路径

- role：architecture/data_interface
- severity：**blocker**
- path：`src/geotoken3path/models/fusion.py:119-158`; `scripts/train.py:19-34`; `scripts/evaluate.py:18-27`
- status：open
- finding：`OpticalSarTokenModel` 是两个 `Linear` stem + 单 fusion + token classifier 的合成模块，未接入 CROMA 光学/SAR encoder、mid/late stage features、分割 decoder、空间输出恢复、真实 modality channel contract、mask/ignore index、loss、metrics、checkpoint 或 cloud data/split interface。`train.py` 没有 optimizer/backward/checkpoint/run manifest，且明确拒绝 cloud。当前代码只能做 tensor smoke，无法执行请求中的 baseline reproduction。
- evidence：源目录没有已实现的 `data/engine/losses/metrics` 模块；`train.py:26-33` 只做一次随机 forward；`evaluate.py:22-26` 只做零张量 forward。
- proposed_fix：在不触及本地真实数据的前提下，先完成 synthetic-fixture 可运行的最小闭环：cloud-path manifest -> split guard -> optical/SAR adapter -> 经审计 VFM encoder -> 同一模型工厂 -> segmentation decoder -> loss/metrics -> optimizer/checkpoint -> resolved run manifest。正式 baseline/candidate 必须复用同一闭环。
- retest：以纯内存合成 12-band optical、2-channel SAR 和 segmentation mask 运行一小步 forward/backward/save/reload/evaluate，验证空间输出、ignore index、split seal、缺失/错序通道和 malformed cloud manifest 均 fail-closed；这仍不是正式实验。

### CR-GT3P-005 — training-object parity 当前只检查“参数名字”，不能判定 PASS

- role：reproducibility
- severity：**major**
- path：`tests/unit/test_model_factory.py:19-50`; `review/CODE_REPORT.json:26-39`
- status：open
- finding：现有测试只比较 `state_dict` 键和 `requires_grad` 名称。它没有比较 resolved config、optimizer parameter groups、学习率/调度器、数据与增强、sampler、有效 batch、初始化 hash、损失、评估器或训练期间的实际梯度可达性。baseline 中 `route_head`/`sar_escalation` 参数名虽存在但分支没有进入最终输出；candidate 中这些参数参与梯度。该差异可能是“一个内部机制差异”的合法组成，但必须被显式审计，不能由参数名相同直接推出完整 training-object parity。`CODE_REPORT.json` 的 parity `pass` 与 coupling readiness `blocked` 并存，且报告早于当前源码快照，已经陈旧。
- evidence：`test_model_factory.py:19-25`；`CODE_REPORT.json:26-39`；工作区不是 Git repository，当前没有 reviewed commit/clean-sync manifest。
- proposed_fix：实现结构化 resolved config diff、optimizer-group/trainability audit、初始化 hash audit、gradient reachability report 和 matched protocol/budget hash；明确 candidate-only 分支参数如何在 baseline 控制中处理及其预算公平性。每次源码变更后重生成 `CODE_REPORT.json`，不得沿用旧测试摘要。
- retest：一条自动化 parity 测试比较所有冻结 common fields，只允许 `enabled_mechanism_ids` 和由其声明的内部图差异；任何其他差异返回 `invalid_protocol`。

### CR-GT3P-006 — 预训练兼容性校验过弱，不能支撑 CROMA 初始化

- role：reproducibility/adversarial
- severity：**major**
- path：`src/geotoken3path/models/initialization.py:11-55`; `tests/unit/test_initialization.py:12-45`; `configs/model/initialization.yaml:1-7`
- status：open
- finding：`validate_pretrained_audit` 只要求顶层字段和 `compatibility.status == pass`，一个仅含 `{'status':'pass'}` 的 compatibility 对象即可通过；测试中的 `sha256` 也不是 64 位十六进制。没有强制核验架构、输入通道/波段顺序、归一化、GSD/分辨率/位置编码适配、head replacement、missing/unexpected keys、shape mismatches、checkpoint source/license/commit 或实际文件 SHA256。`apply_audited_state_dict` 只接收已加载 state dict，不产生完整兼容性报告。
- evidence：`initialization.py:11-38`；`test_initialization.py:12-22`；云审计文件当前仍为 `pending`。
- proposed_fix：定义并验证完整审计 schema；在云端 loader 中先计算实际文件 SHA256，再检查 source/license/commit、CROMA 结构、S1/S2 通道和标准化、位置编码/分辨率适配、显式 head replacement；任何未解释 mismatch 立即失败。baseline/candidate resolved manifest 引用同一 checkpoint hash。
- retest：加入错误 hash、错误 band order、错误 normalization、错误 input shape、未声明 head replacement、missing/unexpected/shape mismatch 的负例；只有完整审计对象可加载。

### CR-GT3P-007 — 测试覆盖没有验证核心 hard falsifiers

- role：adversarial
- severity：**major**
- path：`tests/unit/*`
- status：open
- finding：9 项测试全部通过，但未检查离散单态、真实分支跳过、路由 collapse、静态/随机预算匹配、跨 stage、局部窗口、跨尺度、controlled registration perturbation、missing modality、split/leakage、loss/metric、checkpoint/run manifest 或协议 hash。`test_all_three_candidate_paths_receive_gradient` 甚至与当前“三路同时软混合”实现一致，无法区分获批离散路由与错误软门控。
- evidence：独立 pytest `9 passed`；现有测试仅位于 `tests/unit`，`tests/integration` 与 `tests/smoke` 无测试文件。
- proposed_fix：按 Skill 最低覆盖补齐 unit/integration/smoke；将计划中的四个 hard falsifier 直接编码为结构/统计验收项，但不得把未运行实验写成支持结论。
- retest：完整 synthetic suite + config snapshot comparison + clean process smoke；测试输出进入新版 CODE_REPORT。

### CR-GT3P-008 — 公共发布与 clean-sync 尚未就绪

- role：release
- severity：**major**
- path：`pyproject.toml:1-6`; `README.md:1-15`; `02_experiment/code/**/__pycache__`; `02_experiment/code/.pytest_cache`
- status：open
- finding：`pyproject.toml` 没有 `[project]` 元数据、Python/torch/yaml 依赖、包发现或可复现环境锁；缺少 `scripts/export_environment.py`、license、第三方归属、可执行 smoke/cloud/evaluate 命令和 run-manifest 字段说明。sync tree 当前含 `__pycache__` 与 `.pytest_cache`，且没有 Git commit 或 clean-sync manifest。
- evidence：文件清单与 `pyproject.toml`/README；`git status` 返回 `fatal: not a git repository`。
- proposed_fix：补齐最小包元数据和锁定环境、许可/归属、命令文档、export_environment、`.gitignore`/导出排除规则；由 clean-sync 工具生成文件清单与逐文件 hash，明确排除 cache、输出、数据、权重、凭据和本地绝对路径。
- retest：在全新临时目录从 clean-sync 包安装并运行 synthetic smoke；validator 和 secret/path scan 通过；sync manifest hash 可复核。

## Test-seal 评价

现有 seal 函数的核心判定是正确的：只有同时满足 `execution_scale=final_test` 与 `test_seal_status=final_test` 才允许 test。此项当前不构成 blocker。但正式入口尚未实现 split loader/run manifest，故 seal 还没有被端到端验证。修复 CR-GT3P-004 后必须增加：smoke、baseline、screening、strengthening、confirmation、acceptance、extension 全部拒绝 test，只有带有效 FINAL_TEST 状态的最终 manifest 才接受。

## 解除 BLOCK 的最小顺序

1. 先修复 CR-GT3P-001/002：锁定真正的离散三态执行图和 matched controls；否则后续训练不能检验批准的假设。
2. 修复 CR-GT3P-003：让唯一 resolved config 驱动同一模型工厂、训练入口和评估入口，并 fail-closed。
3. 修复 CR-GT3P-004/006：完成合成可运行的 VFM 分割闭环与严格云端初始化接口。
4. 补齐 CR-GT3P-005/007 的 parity、预算 hash、集成测试与 hard-falsifier 结构测试。
5. 完成 CR-GT3P-008 的 clean-sync/release hygiene，重生成 `CODE_REPORT.json`。

在以上 blocker 全部关闭、major finding 至少形成已验证修复或明确阻断状态前，`CORE_CODE` 与 `LOCAL_REVIEW` 不应标记 PASS，不应请求或执行任何 baseline training。即使用户随后批准 `BASELINE_TRAINING_APPROVAL`，该批准也不能覆盖软件阻断项。
