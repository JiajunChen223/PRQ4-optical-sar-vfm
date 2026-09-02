# GeoToken-3Path 最终本地代码审查

## 结论

**CONDITIONAL_PASS — 允许 local code-service handoff，但不等于 baseline-training ready。**

当前冻结代码已经达到可移交给 ResearchPilot Experiment Skill 进行**代码同步、云端环境预检、官方 CROMA 依赖注入和权重/数据兼容性审计**的本地质量门槛。此前的结构性 blocker 已形成完整的本地合成闭环：硬路由、multi-stage、正式配置解析、训练对象公共协议 hash、分割输出、损失/指标、run contract、数据与预训练输入交叉绑定、CROMA feature bridge、环境导出和 fail-closed 测试。

本结论不授权真实训练。现有 `train.py` 仍主动拒绝 cloud 执行，官方 CROMA 实现与 checkpoint 未在本地树中，`pretrained_weight_audit.json` 明确为 `pending`，真实 dataset manifest、cloud hardware 和实际显存/吞吐均不存在。因此只能做条件移交；在云端兼容性事实闭合和正式 runner 被批准前，不得启动 baseline。

## 独立证据

- 独立测试命令：`F:\anaconda3\envs\dl_env\python.exe -B -m pytest -p no:cacheprovider F:\PRQ4\02_experiment\code\tests -q`
- 结果：`92 passed in 2.77s`。
- Validator：`review/code_validation.json` 为 `pass`；扫描36个 executable/config 文件，0 problems、0 violations，`local_gpu_probe=forbidden_not_run`。
- 本地数据审计：`reports/local_data_policy_audit.json` 为 `pass`，0 suspect、0 violation。
- 未下载数据或权重，未探测 GPU，未执行真实训练，test seal 保持 sealed。
- 关键快照：
  - `croma_bridge.py`: `3F70D8B414625E641E992C1A4C8BD456889CC885E5094E4B52EC23649F9FB216`
  - `config.py`: `A4AAF9355A2A3360F9A0577B3813CA21003F78D60BEB44B8E0172E8B30970DEF`
  - `run_manifest.py`: `687A8DBBD2BF6A7AD1F545EF3B85A4A6D58D62065F7EBF2B51E72B58B7FB749C`
  - `data/contracts.py`: `C45994900930251332BD17C0DE1B2AAC6A3F207973B56FB42AE6E1DE2DD62D38`
  - `code_validation.json`: `D5EB19224AAC5C63038766C5EA54CEA327277FE4DD3618A776EAC2A5082E7C12`
  - pending pretrained audit template: `78D60DDEE9494A7C3E3412A8E895079C52C0F80403805858571C34FD61BD7EDC`

## 增量 blocker 复核

| 先前 blocker | 状态 | 当前证据 |
|---|---|---|
| optimizer/trainability 未进入公共协议 | **CLOSED at resolver level** | optimizer、scheduler、gradient clipping、seed、effective batch 和 trainability 已进入 resolved config 与 matched common protocol SHA；baseline/candidate diff 仍仅为 `model.mechanism_set`。 |
| run manifest 未接入入口 | **CLOSED for smoke contract / CONDITIONAL for formal persistence** | train/evaluate 均构造 run manifest 并输出 run-contract SHA；manifest 已包含 input、storage、trainability、optimizer、scheduler、batch、data/pretrained/code-sync refs。正式云 runner 仍须把 manifest 与 resolved snapshot持久化到 run directory。 |
| dataset band/path/总空间契约过弱 | **CLOSED** | exact S2 12-band、VV/VH、POSIX path escape、component ledger、45GB hard stop 和 pretrained target input cross-bind 均已实现并测试。 |
| 无 CROMA/raw-image 接口 | **CLOSED as dependency-injected interface** | `CromaBackboneBridge` 验证12/2通道、mid/late stage mappings、`[B,N,D]` tokens 和 `[B,N,4,D]` native fine SAR；raw-image wrapper 输出 dense segmentation，默认冻结注入 backbone。 |
| 缺环境导出与许可状态 | **CLOSED for pre-baseline handoff** | `export_environment.py` 不探测设备并已生成 local environment record；`LICENSE_STATUS.md` 诚实声明尚未选择公开发布许可证，未把当前包误写为公开 release。 |
| 云端 pretrained 事实 | **OPEN/PENDING by design** | pending template 字段完整，但 URL、commit、license、checkpoint SHA、真实 input normalization、position adaptation、state-dict compatibility 和 geography overlap 均未验证。 |

## 条件移交后的强制边界

1. **移交前的本地记录闭合**：必须基于当前冻结 hash、92项测试、36-file validator、本报告和独立测试复审，重写 canonical `review/CODE_REPORT.json`；旧的“3 passed/bootstrap”报告不得继续作为当前证据。随后生成 code-only clean-sync manifest或 reviewed commit，排除 cache、outputs、数据、权重、凭据和本地绝对路径。
2. **移交只允许云端预检**：可以同步代码、验证环境、取得官方 CROMA 元数据/权重和 dataset manifest，但必须先通过数据 license/hash/空间 ledger、pretrained compatibility和 cloud hardware preflight。
3. **真实 CROMA 注入仍需验证**：`CromaBackboneBridge` 当前只由 synthetic dependency fixture 验证。Experiment Skill必须证明官方实现能稳定提供约定的 mid/late optical/SAR tokens 和每粗 token 2x2 fine-SAR block；若官方接口不满足，任何 adapter代码变更都必须通过 `code_sync` 返回本地复审。
4. **正式 runner 仍未开放**：现有 `train.py --execution-scale cloud` 会 fail-closed；在它被正式 remote control card、dataset loader、audited initializer、run-directory persistence、checkpoint/log/metric写入和validation-only selection逻辑接管前，不得训练。
5. **初始化不得以参数名替代事实**：`audited_croma_backbone` 是依赖注入边界，不是自动认证。只有 `pretrained_weight_audit.json.status=pass`、实际文件 SHA吻合、dataset/pretrained cross-bind通过且 state-dict load report无未解释 mismatch 后，才可称为 audited CROMA。
6. **Test seal继续有效**：本移交不允许 test split，也不授权 innovation screening、multi-seed或 final-test。

## Handoff 判定

- `CORE_CODE`：可在 canonical CODE_REPORT/clean-sync manifest更新后标记本地条件通过。
- `LOCAL_REVIEW`：本报告支持条件通过。
- `CLOUD_SYNC/CLOUD_ENVIRONMENT`：可在用户批准的下一受保护范围内开始。
- `CLOUD_DATA_DOWNLOAD`、`PRETRAINED_AUDIT`、`BASELINE_REPRODUCTION`：仍为 pending，不由本报告批准。

因此最终状态是 **CONDITIONAL_PASS**，不是 PASS，也不是科学验收。若 canonical CODE_REPORT/clean-sync 未生成，或云端需要未回传的代码修改，则自动回退为 BLOCK。
