# V17 MCOF 数据边界与 Test-Seal 只读审查

范围：数据 manifest/loader、test-seal、云端路径、CROMA 初始化/权重 locality、run manifest、train/evaluate 入口。未读真实像素/权重或云端文件，未用 GPU，未训练；结论仅是工程审计。

## 结论

运行时 test-seal 拒绝逻辑存在，默认入口只请求 validation；r3 validator 为 PASS。没有当前 test 泄漏或本地数据/权重进入代码树的证据。正式训练前有两项主要完整性风险。

## Findings

### M1 — 外部数据与预训练审计未运行时交叉绑定

证据：`engine/formal_runner.py:429-457` 分别建 loader、加载 CROMA；`data/contracts.py:267-287` 的 `cross_validate_dataset_and_pretrained()` 未被生产路径调用。风险是本次 cloud manifest 与实际 pretrained audit 的 bands/normalization 可能只各自通过、彼此未绑定。修复：建 loader 后、构模前读取同一 audit mapping，调用交叉校验，并将两份 artifact SHA 写入 run manifest，失败即停。

### M1 — split/hash/realpath 主要信任 manifest 声明

证据：`data/contracts.py:229-240` 只检查 `sample_realpaths_resolved` 和 SHA 格式；`data/sen12ts.py:188-207` 未检查 parent 跨 split 交集/唯一性，也未重算 hash。风险是替换或失真的 manifest 可能削弱泄漏与 50GB 账本防护。修复：重算 manifest/canonical split-row hash，检查全体 parent disjoint，并逐样本 realpath containment；test 仍不读取。

### M2 — 外层 cloud artifact 路径未强制 canonical realpath

证据：`engine/formal_runner.py:132-144` 只按 POSIX root 前缀判断，不拒绝 `..`、重复分隔符或符号链接。风险是 manifest/audit/output 别名路径绕过预期目录绑定。修复：realpath 后拒绝 symlink/未规范化输入，并分别限定 data、audit、output 子目录。

### M2 — 入口覆盖缺口

证据：`scripts/train.py:291-295` 的 smoke 不调用 `models/croma_bridge.py:783-816` 的 MCOF raw-image 分支；`scripts/evaluate.py:39-74` 仅做 smoke/cloud 预检，没有 final-test 通道。当前是安全的，但 smoke PASS 不证明 MCOF 入口可执行。修复：加入 synthetic raw-image bridge smoke；最终配置冻结后再实现双重 guard 的 final-test 入口。

## Blocker

已发生泄漏证据：none。M1 两项应在正式 V17 训练前修复，或由云端审计卡明确覆盖；否则不能宣称数据完整性 PASS。
