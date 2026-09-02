# Independent V12-D0 release final r4 hash review

- Scope: current V12-D0 clean-sync manifest and r4 code-only package.
- Mode: read-only; no source, data, weights, cloud host, GPU, training, or sealed-test state was changed or accessed.
- Date: 2026-08-29 (Asia/Shanghai).
- Decision: **BLOCKED for canonical release handoff until `CODE_REPORT.json` is regenerated.**

## Hash and payload checks

| Item | Result |
|---|---|
| Manifest | `F:\PRQ4\02_experiment\code\manifests\clean_sync_manifest_v12_d0_20260829.json` |
| Manifest declaration | `status=pass`, `file_count=106`, 106 entries |
| Current manifest SHA256 | `9667880bfc87214e8a5bff48399046a708bc35f56e941e8378178c9b8c7c5ae6` |
| Current manifest verification | 106/106 present, byte/SHA mismatches `0` |
| r4 package | `F:\PRQ4\02_experiment\artifacts\geotoken3path_code_v12_d0_20260829_r4.tar.gz` |
| r4 package bytes | `188379` |
| r4 package SHA256 | `b505b6e477cc8c6ea76332913a6777ec69a5f54c8936ce7812ebdd389267caea` |
| Archive payload | 107 members = 106 manifest payload files + embedded release manifest; no missing/extra paths |
| Embedded manifest binding | `source_clean_sync_manifest_sha256` equals current manifest SHA256; all 106 payload byte/SHA checks pass |

The package and current manifest are therefore mutually consistent and release-clean at the payload/hash level.

## Release blocker

`F:\PRQ4\02_experiment\code\review\CODE_REPORT.json` is stale: it still records the prior r3 package (`913b4702...`, 188311 bytes) and prior manifest SHA (`ede479d8...`), while the current r4 package and manifest are the values above. The r4 source snapshot also postdates the report. Regenerate the canonical `CODE_REPORT.json` to bind the current manifest/package (and retain prior reports append-only); do not launch cloud execution from the stale report.

This is an engineering/release decision only, not a scientific result. The current package/manifest hash checks themselves are **PASS**.
