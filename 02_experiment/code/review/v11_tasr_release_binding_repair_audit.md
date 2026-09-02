# V11 TASR release-binding repair audit

审查范围：V11 resolver 修复后的 source tree、r5 code-only package、embedded
release-manifest fallback 和前一轮独立 architecture/release reviews。

## 结论

**PASS（针对本次 release-binding 修复；不等于 C1 科学支持）。**

这次修复只解决了一个已记录的 `invalid_protocol`：云端代码包不包含 source
clean-sync manifest 的目录，而 V11 resolver 将其作为必需文件。当前 resolver
按优先级使用本地 `manifests/clean_sync_manifest_v11_tasr_20260829.json`，或在
云端使用包内 `researchpilot_code_release_manifest.json`，并对实际文件内容计算
SHA256；两种布局都解析到同一个 `R-EO-TASR-01` 配置。临时包解压模拟证明云端
fallback 返回 `researchpilot_code_release_manifest.json`、V11 route 和完整
TASR contract。

## Evidence

- local clean-sync manifest：103/103 entries bytes/SHA match；当前 SHA256
  `9ec6eb8881473efb6af2365cdc51e5c1215a578058cd5094b323204c4787d792`。
- r5 package：`geotoken3path_code_v11_tasr_20260829_r5.tar.gz`，182793 bytes，
  SHA256 `dc260cd20e707b334e493fec0cec654072e621726d8358f9e90196e0a538a88b`；
  104 archive entries = embedded release manifest + 103 reviewed payload files。
- remote-layout simulation：解压 r5 到临时无数据目录后，
  `resolve_v11_tasr_config(..., "always_fuse")` 成功，ref=`researchpilot_code_release_manifest.json`，
  embedded-manifest SHA=`5c70efe6a06ae92c3818be790928ab62f18c9c354eea56ed4eff1c97c8061db5`，
  route=`R-EO-TASR-01`。
- full CPU synthetic suite：303 passed；ResearchPilot code validator：113
  executable/config files，0 violations，local GPU probe forbidden_not_run。
- prior independent architecture review：PASS；prior independent release
  rereview：PASS for the preceding r4 snapshot; the only intervening source change
  is the resolver's explicit local/embedded manifest fallback, covered above.

## Boundary

本审计不读取真实 SEN12TS 像素、不加载云端 CROMA 权重、不运行 GPU、不训练
TASR C1、不运行 C2/C3、不访问 sealed-test。前一次 C1 R1 仍是
`invalid_protocol`，没有 checkpoint、指标或科学结果；修复后必须使用新的
run id。该审计也不解除 V11 的 50.0075% mIoU 晋级门。
