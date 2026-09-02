# Independent V11 TASR final release review

审查时间：2026-08-29（只读；仅新增本报告）。

## 结论

**BLOCKED**。V11 r3 manifest、package、validator 与当前测试均闭合，但 canonical `CODE_REPORT.json` 仍是旧 D3/CEAK 报告，未绑定 V11 manifest/package，故不能作为当前 V11 最终发布凭据。

## 核验结果

- Manifest：`clean_sync_manifest_v11_tasr_20260829.json` 声明 103 个文件；逐文件重算 **103/103 SHA256 与 bytes 匹配**，缺失 0、不一致 0。Manifest SHA256=`434b0d54699d1832b28668041a7b71e69b5760ee70aa41ab226c73d613b2e319`，bytes=18521。
- r3 package：`geotoken3path_code_v11_tasr_20260829_r3.tar.gz` SHA256=`72b43d6a1f05354c5e03b02a62130edcd9f5a6dfcc56bb6ddd018013be3ed2ed`，bytes=182565；104 个 tar 条目=release manifest+103 payload，payload 集合、内部 manifest、逐文件 SHA/bytes 全部匹配，额外/非 regular payload=0。
- Validator：复跑 PASS，扫描 **113** executable/config files，problems=0、violations=0，`local_gpu_probe=forbidden_not_run`。
- Tests：`F:\anaconda3\envs\dl_env\python.exe -X utf8 -B -m pytest tests -q --disable-warnings -p no:cacheprovider`，exit=0，**303 passed, 1 warning**（7.00 s；synthetic-only，无真实数据/权重/GPU）。
- CODE_REPORT stale：当前文件 mtime=2026-08-29 11:55:41；其 `generated_at_utc=2026-08-29T03:15:18.708954+00:00`，绑定 `clean_sync_manifest_d3_20260829.json` / SHA=`979bd9...`、route=`R-EO-CEAK-01`、`265 passed`、validator `96`，package=`geotoken3path_code_d3_20260829.tar.gz`，与上述 V11 r3 闭合对象不一致。

阻塞解除条件：由发布 owner 重新生成并核验 V11 绑定的 canonical `CODE_REPORT.json`（保留上述 303/113 与 manifest/package SHA/bytes），再进行最终 handoff。未修改源代码、manifest、package、validator 或现有 CODE_REPORT。
