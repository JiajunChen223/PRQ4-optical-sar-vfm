# 数学候选机制代码审查 R7

- scope: 仅新增 `CrossModalGeometryAdapter` 及三个候选开关；论文任务、数据、CROMA 初始化、baseline、指标、24 epoch、3090 约束和 test seal 未改。
- candidates: `GW-01/gw_relational_transport`, `BURES-02/bures_covariance_alignment`, `GRASS-03/grassmann_polar_transport`。
- information-flow proof: GW mode computes a bounded relational transport plan; Bures mode computes an epsilon-regularized SPD Bures map; Grassmann mode computes a batched polar/Procrustes rotation. Each is injected before the audited CROMA joint path and uses a zero-start residual.
- local tests: 189 passed; targeted geometry tests 5 passed.
- static validator: `validate_math_geometry_r7.json`, 64 executable/config files, 0 violations.
- local data/GPU: no real data, weights, or GPU probe used.
- package: `geotoken3path_code_math_geometry_r7.tar.gz`, 66 allow-listed files, no data/weights/credentials/cache.
- review status: PASS_FOR_LOCAL_CODE_ONLY; cloud synchronization and real VFM screening remain pending.
- cloud safeguards: before screening, require guarded code sync, cloud environment manifest match, current pretrained audit, identical trainability/seed/protocol, and validation-only execution. No candidate is a supported result until a fresh cloud run receipt and effect-policy comparison exist.
- known risks: relational GW is a low-rank anchor-signature approximation rather than an exact full four-index GW solver; Bures/Grassmann eig/SVD behavior under the real CROMA tap distribution requires cloud-side shape/finite-gradient audit. These are explicit falsifiers, not claimed results.
