# V15-RIFT code review r1

- Review mode: coordinator single-thread adversarial review, per the explicit user request; no independent multi-agent corroboration is claimed.
- Scope: new fixed 3x3 pure-modality affinity-difference operator, route/config/factory/CLI registration, hard-contract script and synthetic tests.
- Mechanism boundary: compute row-centered cosine relations separately from pure optical and pure SAR tokens; form a signed SAR-minus-optical field, L1-normalize it, and transport only the already fused semantic token neighborhood. The only trainable RIFT state is a zero-start stage-wise channel scale.
- Prohibited capacity: no learnable Q/K, router, temperature, class-specific relation, ground-truth relation, boundary loss, extra decoder or extra objective.
- Hard-contract evidence: C0 baseline identity, C1 row-sum zero, C2 identical-relation null, C3 constant-semantic null, C4 pure-modality relation, C5 live gradient and C6/C7 capacity/control checks pass on synthetic tensors. Synthetic evidence is not a scientific result.
- Regression evidence: 334 local tests pass with one pre-existing CROMA bridge warning; code validator reports 143 executable/config files and zero violations; RIFT smoke train/evaluate paths pass on CPU synthetic tensors.
- Packaging: V15 r1 release contains code/config/tests only; no real data, weights, checkpoints, caches, credentials or raw cloud logs.
- Decision: PASS for guarded V15 RIFT code-only synchronization. Real cloud seed-0 and all scientific claims remain pending; RIFT-C2/C3 remain locked behind the +2pp gate.
