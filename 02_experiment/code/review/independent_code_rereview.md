# GeoToken-3Path 独立代码复审

## 最终结论

**BLOCK**

当前冻结源码已从“软门控 toy module”升级为可审计的**本地合成结构原型**，核心硬路由、multi-stage、正式 YAML resolver、稠密分割输出、损失/指标、初始化校验和训练对象协议均有实质修复。独立复跑结果为 `88 passed in 2.91s`，现有 validator 也记录为 `pass`。

但是，当前状态仍**不能标记为 local code-service PASS，也不能交付 baseline training**。阻断点是：正式 CROMA backbone/feature adapter 与云数据 loader 尚未实现；现有唯一训练入口明确拒绝 cloud；run manifest 只是孤立工具，未接入 train/evaluate、checkpoint、resolved snapshot 和 run directory；`CODE_REPORT.json` 仍是修复前的陈旧 BLOCKED 报告，且没有 clean-sync manifest 或 reviewed commit。云端权重审计保持 `pending` 是正确状态，但不能把“校验器已实现”写成“CROMA 已兼容或已加载”。

较准确的分层判断是：

- 核心机制本地结构实现：**CONDITIONAL_PASS**。
- 当前完整 code-service handoff：**BLOCK**。
- 科学状态：无 baseline、创新、效率、跨区域或稳健性结果。

## 复审边界与证据

- 只读复审范围：`F:\PRQ4\02_experiment\code` 及其关联的 gate/audit 状态。
- 未下载或读取真实数据/权重，未探测 GPU，未执行真实训练。
- 独立命令：`F:\anaconda3\envs\dl_env\python.exe -B -m pytest -p no:cacheprovider F:\PRQ4\02_experiment\code\tests -q`
- 结果：`88 passed in 2.91s`。
- validator artifact：`review/code_validation.json`，`status=pass`，扫描 33 个 executable/config 文件，0 problems、0 violations；本复审另行只读搜索未发现 `nvidia-smi`、`torch.cuda`、`.cuda(`、本地绝对数据路径或下载调用。
- 冻结快照 SHA256：
  - `fusion.py`: `A1C2885C4410F28A54C87B008448945580E5EE344F69582D7E91D01DCCA29A71`
  - `config.py`: `A8F40B6B3AB5B74D7CEC68EBD25E7CA492DE70BF94F88E29FE7EB304894E82E3`
  - `initialization.py`: `967471A368600DDE05376F4A6D2F51B230CC8C113D8B436BFB20D8A362CAB09E`
  - `test_training_object_contract.py`: `0B9647AE9D21FCC95D213F1E1F6B6220C11DC315C79D1611863EBD3A00447295`
  - `code_validation.json`: `ABCC91BE9E49D59018F429606702BD9AD251C5ECAA4BC701B5335E6B087FCA4A`

## 旧 findings 逐项复核

| ID | 状态 | 复审判断 |
|---|---|---|
| CR-GT3P-001 | **CLOSED（本地结构层）** | 路由前向已是 exact-capacity hard one-hot；straight-through 仅负责梯度；bypass 在融合边界精确等于 optical；current/fine operator 仅对被选 token dispatch；显式 `[B,N,4,D]` fine-SAR block、局部窗口和 mid/late `ModuleDict` 均已实现。正式 CROMA stage features 仍是下游阻断，不回退本项的结构修复结论。 |
| CR-GT3P-002 | **CLOSED** | `unimodal_optical`、`always_fuse`、`static_sparse`、`random_budget`、`local_exchange_without_state_machine`、`geotoken_3path` 均通过同一模型/dispatch；静态与随机控制为硬路由并有精确容量与实际分支计数。 |
| CR-GT3P-003 | **PARTIAL** | 正式四份 YAML 已由 fail-closed resolver 解析，candidate 不再静默回退，baseline/candidate diff 仅为 `model.mechanism_set`，共同协议 hash 一致；但 resolver 只加载本地 `3090_plan.yaml` 的 smoke 状态，入口没有 formal cloud config/resolved snapshot 写盘，也没有把 run manifest 接入执行链。 |
| CR-GT3P-004 | **PARTIAL** | 已有 multi-stage token bridge、dense `[B,C,H,W]` 输出、cross-entropy、confusion matrix/mIoU、synthetic optimizer step 和 checkpoint roundtrip；但模型仍是 Linear stems + token fusion/classifier，不是 CROMA radar-optical backbone，也没有 cloud dataset loader、正式 epoch loop、checkpoint selection 或真实 validation evaluator。 |
| CR-GT3P-005 | **PARTIAL** | 已增加 resolved config diff、common protocol hash、相同 state-dict surface、相同 requires-grad 名单和 integration test；但 `trainability` YAML 没有被 resolver/factory执行，optimizer/scheduler/LR/augmentation/sampler/data-manifest/checkpoint hash 不在 parity manifest 中，当前测试仍不能证明完整正式 training object matched。 |
| CR-GT3P-006 | **PARTIAL** | 初始化 validator 已显著加固：64位 SHA、source/license/commit、结构、bands/normalization/GSD/patch、head replacement、state-dict keys/shapes、position adaptation、comparison policy 均 fail-closed；但 `pretrained_weight_audit.json` 仍为 pending，且审计中的 target architecture/input 没有和实际 resolved config/model 自动交叉绑定。当前 toy model也不能加载官方 CROMA backbone checkpoint。 |
| CR-GT3P-007 | **PARTIAL** | 88项测试覆盖硬路由、预算、identity、branch hooks、两条 active path 梯度、multi-stage、native fine input、resolver、data manifest、segmentation、initializer负例、checkpoint roundtrip、test seal；但没有真实 loader/split implementation、正式 runner、entrypoint run-directory/manifest、CROMA adapter或 clean checkout smoke，故测试量不能替代完整路径。 |
| CR-GT3P-008 | **PARTIAL** | 已补 pyproject metadata/dependencies、README、THIRD_PARTY、`.gitignore`；但缺少 `scripts/export_environment.py`、LICENSE、锁定环境、可执行 cloud 命令、clean-sync manifest/commit。缓存虽被 ignore，但当前目录仍含 cache artifacts，尚无可审计导出清单。 |

## 当前仍阻断 handoff 的 findings

### RR-GT3P-001 — 正式 cloud/CROMA 路径不存在

- severity：**blocker**
- evidence：`scripts/train.py` 对 `--execution-scale cloud` 直接抛错；`resolve_approved_config` 固定读取 `configs/runtime/3090_plan.yaml`，其 `execution_scale=smoke`、`real_data_allowed=false`；`OpticalSarTokenModel` 仍以预编码 token 为输入，未包含 CROMA backbone/adapter；没有 dataset loader。
- impact：代码不能在授权云端仅通过配置切换进入 baseline reproduction。CROMA audit 即使将来通过，checkpoint 也不能直接加载进当前 token bridge。
- required fix：本地实现并复审 CROMA feature adapter/backbone factory、cloud manifest loader、正式 runtime config 和同一 train/evaluate runner；权重二进制仍只在云端取得。若云端发现接口变化，必须通过正式 `code_sync` 回到本地复审，不能 hot-edit 后直接训练。
- status：open。

### RR-GT3P-002 — run manifest 与可复现实验记录未接入入口

- severity：**blocker**
- evidence：`utils/run_manifest.py` 能在内存构造一个最小 manifest，但 `train.py`/`evaluate.py` 从未调用它；脚本不创建 run directory，不写 resolved config、data-manifest ref/hash、checkpoint SHA、commit/sync ID、optimizer/scheduler、metrics或日志。integration test 只在训练 step 后独立构造 manifest。
- impact：即便下一步补上 cloud runner，当前也无法证明某个 checkpoint/metric 来自哪套冻结配置与数据清单。
- required fix：让同一入口在任何 loader/model 构造前验证 dataset/initialization/test seal，写不可变 resolved snapshot 和完整 run manifest；补充 checkpoint/log/metric/data-ref/seed/sync-manifest 字段，并对 baseline/candidate 生成可比较的 config diff 与 trainable-parameter audit。
- status：open。

### RR-GT3P-003 — 完整 training-object parity 尚未锁定

- severity：**major，进入 baseline 前为 blocker**
- evidence：`configs/model/geotoken3path.yaml` 声明 trunk frozen/locked-last-blocks、router/adapters/decoder trainable，但 resolver不输出 trainability，factory不应用 mask；当前 integration test仅比较 requires-grad 名单和 state-dict keys。正式 optimizer、scheduler、LR、sampler、augmentation和有效 batch不在 resolved protocol hash中。
- impact：未来接入 CROMA 后，baseline/candidate可能在未被 diff 捕获的训练对象上运行。
- required fix：把所有 common protocol fields、trainability mask和 optimizer groups纳入 resolved config/hash；输出逐参数 `requires_grad`、optimizer group、初始化 SHA 审计；只允许 `enabled_mechanism_ids/model.mechanism_set` 变化。
- status：open。

### RR-GT3P-004 — 数据 manifest 校验尚未绑定真实输入契约

- severity：**major，数据下载/加载前为 blocker**
- evidence：`validate_cloud_dataset_manifest` 对 optical band order 只检查长度12，对 normalization 只检查非空 mapping；没有与初始化 audit/正式 resolved input交叉校验。`cloud_root.startswith('/root/autodl-tmp/')` 也未显式拒绝 `..` 路径段。storage 校验只覆盖单个 dataset bytes，不覆盖计划要求的总 raw+extract+cache+weights+checkpoints footprint。
- impact：错误或重复波段顺序、归一化漂移、路径逃逸字符串和总存储超限可能在 schema 层漏过。
- required fix：锁定精确 S2/S1 channel order，绑定 normalization/checkpoint input spec，规范化并验证 POSIX cloud path，使用总项目空间 ledger 执行 45GB hard stop/50GB ceiling。
- status：open。

### RR-GT3P-005 — canonical CODE_REPORT 与 clean-sync 证据陈旧/缺失

- severity：**blocker**
- evidence：当前 `review/CODE_REPORT.json` 仍声称仅测试 test-seal 的 `3 passed`，`reviewed_commit_or_sync_manifest=not_created_bootstrap_only`，coupling readiness 仍标记 pending；这与当前88项测试和修复源码不一致。工作区没有 reviewed commit 或 clean-sync manifest。
- impact：Experiment Skill不能消费一个与冻结源码/hash一致的 code-service结果。
- required fix：上述软件 blocker 关闭后，重跑完整无缓存测试和 validator，生成 clean code-only sync manifest/commit，重写 canonical CODE_REPORT；报告必须保留 CROMA audit pending、local GPU forbidden_not_run、test sealed和无科学结果边界。
- status：open。

### RR-GT3P-006 — public-release 闭环仍不完整

- severity：major
- evidence：缺少 Skill要求的 `scripts/export_environment.py`、LICENSE、环境锁/版本导出、clean checkout install+smoke证据和正式 cloud/evaluate命令；README只描述本地 smoke。
- required fix：补齐发布最低项，并从新临时目录验证安装、synthetic smoke、禁止数据/权重/secret/cache进入 sync tree。
- status：open。

## Test seal 与科学边界

- runtime seal 当前仍正确拒绝非 `final_test/final_test` 的 test 请求。
- 现有 formal resolver只产生 `sealed`，因此当前代码不能开启 test；这是正确的保守状态。
- 88项测试、validator pass、synthetic loss/metric和checkpoint roundtrip全部是软件契约证据，不是 baseline reproduction，也不能支持方法优越性、效率或泛化叙事。

## 解除 BLOCK 的最小闭环

1. 本地完成 CROMA/backbone feature adapter、cloud manifest loader和正式 config-driven runner；禁止在云端留下未回传的热修复。
2. 将完整 run manifest、resolved snapshot、data/weight hashes、trainability/optimizer audit、checkpoint/log/metric记录接入同一入口。
3. 加固 dataset-to-initialization input cross-check和总存储 ledger。
4. 生成 clean-sync manifest或reviewed commit，补齐 release hygiene。
5. 在最终冻结快照上重跑 tests/validator/independent review并重写 `CODE_REPORT.json`。

完成以上本地项后，code-service才可对 Experiment Skill返回 PASS。随后云端仍必须独立完成官方 CROMA权重的 URL/commit/license/SHA/输入兼容性审计、真实硬件 preflight和数据 license/hash/size验证；这些云端事实当前均未完成。
