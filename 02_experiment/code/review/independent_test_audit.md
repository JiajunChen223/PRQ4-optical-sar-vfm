# Independent synthetic-test and contract audit

- Audit role: independent test auditor
- Scope: `F:\PRQ4\02_experiment\code`
- Date: 2026-08-20 (Asia/Shanghai)
- Verdict: **BLOCK**
- Scientific boundary: this is a CPU-only synthetic and static code audit. It is not experimental evidence and does not approve baseline training, candidate screening, synchronization, or any paper claim.
- Safety boundary honored: no dataset or checkpoint download, no real-data execution, no GPU/CUDA inspection, no training, and no source/config modification.

## Executive decision

The frozen snapshot passes its nine existing unit tests, the ResearchPilot layout/policy validator, tensor-shape checks, candidate gradient-flow checks, common parameter-surface parity, the expected-activation ceiling, and the runtime test seal. It is nevertheless **blocked from code-service PASS and formal baseline execution** because the implemented operation is not the approved primary mechanism:

1. the approved per-token **discrete three-state choice** is implemented as a dense soft convex mixture;
2. the supposedly invariant optical identity residual is altered by a terminal `LayerNorm`, even under a forced 100% bypass route;
3. the “finer-scale escalation” path receives no finer-scale/stage feature and preserves the same token grid (with an exact raw-SAR fallback for non-square token counts);
4. the `static_sparse` matched control is not token-sparse and therefore cannot serve the preregistered budget-matched sparse falsifier;
5. the pretrained compatibility gate accepts a minimal `{"compatibility": {"status": "pass"}}` record without the detailed compatibility fields required by the skill, and the current entry points do not apply the audited initializer.

These are operation-contract mismatches in the primary mechanism and its strongest control, not cosmetic omissions. Existing passing tests do not detect them.

## Frozen source identity

| File | SHA256 |
|---|---|
| `src/geotoken3path/models/fusion.py` | `5C23A359A4197036BF8D02754678ECB0D7109294951501C30DD882ADE357CAE6` |
| `src/geotoken3path/models/factory.py` | `D87F7A7CCB74B5BCD3663382D490B6EEC5C78C137E2E2ACE91D09CB8EA05FFFE` |
| `src/geotoken3path/models/initialization.py` | `91460171FE5657B1A91097E49D873FF5299AA2F82E003CD1841127075F787EF0` |
| `src/geotoken3path/utils/test_seal.py` | `0E782A0CE9F9AC00FF427BC2BF4D812C59A19F1B7A1955B401E1034609DE04EA` |

## Findings

### B1 — Blocker: discrete three-state route is a soft dense mixture

- Contract: `plan_handoff.json` defines a “per-spatial-token, per-stage discrete bypass/current-scale/escalation transition”.
- Evidence: `fusion.py:104-110` applies `softmax`, then evaluates a weighted sum of all three path tensors. In a deterministic synthetic run at budgets 0.1, 0.5, and 0.9, the fraction of route entries strictly between 0 and 1 was `1.0`, and the fraction of tokens with a one-hot state was `0.0` for every budget.
- Execution evidence: with the route head forced to bypass probability exactly `1.0`, forward hooks still observed `sar_exchange=1` and `sar_escalation=1`; `fusion.py:90` constructs all paths before selecting/weighting them.
- Consequence: the snapshot implements adaptive feature mixing, not the approved discrete state transition or conditional computation. It cannot support a token-routing efficiency interpretation, and measured path-skipping latency/VRAM gains are structurally unavailable in this implementation.
- Required correction/retest: implement an explicit one-state-per-token transition with a documented differentiable estimator and genuinely masked/lazy path execution, then test route one-hotness, state occupancy, budget, gradients, and measured branch invocation. If soft mixing is intended instead, it requires a governed plan/claim amendment rather than silent reinterpretation.

### B2 — Blocker: invariant identity residual is not preserved at the fusion output

- Evidence: `fusion.py:52` defines `bypass = optical`, but `fusion.py:111` applies `output_norm` after routing.
- Adversarial result: forcing every token to bypass (`min bypass probability = 1.0`) produced `max_abs(fusion_output - optical) = 1.5726171731948853`.
- Consequence: the internal pre-normalization branch is an identity, but the externally observable fusion path is not. This violates the approved invariant unless the contract is explicitly redefined as “identity before common normalization”.
- Required correction/retest: preserve the residual at the declared contract boundary or revise the contract explicitly; add a forced-bypass equality test at that boundary.

### B3 — Major: “finer-scale escalation” is not connected to a finer scale

- Evidence: `fusion.py:42-49` performs stride-1 average pooling and returns the same `[B,N,D]` grid. For non-square `N`, it returns the SAR tensor exactly (`fusion.py:45-46`). `fusion.py:54` receives no finer-stage or higher-resolution feature input.
- Synthetic result: `[1,16,8] -> [1,16,8]`; for `[1,10,8]`, the context was bitwise identical to the input.
- Consequence: the third state is a same-scale smoothed SAR residual (or raw SAR fallback), not an escalation to a finer scale/depth as frozen in the plan.
- Required correction/retest: expose an explicit multi-stage/native-scale input and verify cross-scale provenance, resolution mapping, and gradient flow; alternatively amend the mechanism name and claim.

### B4 — Major: `static_sparse` is a dense interpolation, invalidating the matched sparse control

- Evidence: `fusion.py:96-102` assigns the same vector `[1-budget, budget, 0]` to every token and mixes bypass/current features at every position. At budget 0.5, the only route vector was `[0.5, 0.5, 0.0]`, with one-hot token fraction `0.0`.
- Consequence: the code contradicts its comment that it deterministically bypasses a token fraction and does not test the preregistered static sparse alternative at equal active-token budget.
- Required correction/retest: implement a deterministic hard token mask with exactly controlled occupancy, or rename the control as static dense interpolation and add the actual sparse control.

### B5 — Blocker before any cloud pretrained run: compatibility audit is under-specified and not wired into entry points

- Evidence: `validate_pretrained_audit` accepts a record containing only a nonempty hash, top-level pass flags, and `compatibility: {status: pass}`. An adversarial minimal record passed despite omitting architecture, input channels/band order, normalization, positional/resolution adaptation, head replacement, missing keys, unexpected keys, and shape mismatches.
- Additional evidence: `scripts/train.py` and `scripts/evaluate.py` build a randomly initialized synthetic model directly; they do not load `configs/model/initialization.yaml` or call `apply_audited_state_dict`.
- Consequence: strict state-dict loading is present as a utility, but the required auditable initialization path is not yet enforced by the execution path.
- Required correction/retest: validate all required compatibility fields, verify checkpoint SHA256 against the loaded bytes on the authorized cloud host, wire the initializer into the shared model factory/entry points, and add negative tests for each unexplained mismatch.

### M1 — Major readiness gap: passing unit tests do not cover the minimum code-service path

No implemented/tested evidence was found for cloud data interface and missing-path failure, split/leakage manifest, loss/metric definitions, optimizer/scheduler parity, immutable resolved configuration, run-manifest creation, checkpoint round-trip, matched protocol/budget hash, parent/full-stack ablation rows, or baseline-versus-candidate resolved config diff. The package is accurately characterized as a synthetic mechanism scaffold, not a complete baseline runner.

The missing-SAR probe also fails with a generic PyTorch `TypeError` rather than a declared missing-modality interface. That is not necessarily part of the primary scientific claim, but it is a required interface/coverage gap under the code skill.

## Passing checks

- `pytest`: **9 passed in 1.50 s**.
- ResearchPilot validator: `status=pass`, 19 executable/config files scanned, zero policy problems/violations, `local_real_data_allowed=false`, `local_gpu_probe=forbidden_not_run`.
- Shared factory: baseline and candidate have identical state-dict keys and identical `requires_grad=True` parameter names.
- Shapes: candidate returned `[B,N,19]` logits and `[B,N,3]` routes for `(B,N)=(1,16),(2,9),(1,10)`; maximum route-sum error was `5.96e-08`.
- Gradients: all candidate stems, route-head layers, exchange/escalation projections, normalization, and classifier parameters received finite nonzero gradients in the synthetic loss probe.
- Budget ceiling: observed active fractions were approximately `0.10`, `0.50`, and `0.7208` for ceilings `0.10`, `0.50`, and `0.90`; therefore the current expected-activation upper bound works for the tested fixtures. This is not evidence of conditional compute.
- Test seal: test access was rejected for smoke, baseline, screening, strengthening, confirmation, acceptance, and extension manifests; it was accepted only for `execution_scale=final_test` plus `test_seal_status=final_test`.
- Entry points: baseline/candidate synthetic smoke and candidate synthetic validation completed with `[2,16,19]` / `[1,16,19]`; `--execution-scale cloud` refused execution with exit code 1 and `RuntimeError: Cloud execution requires the approved remote control card.`
- Hygiene: zero `.pt/.pth/.ckpt/.bin/.safetensors` or raster/array data binaries in the code tree; zero static hits for GPU probing, credentials, secrets, or local absolute paths in executable/config files.

## Commands and observed results

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
F:\anaconda3\envs\dl_env\python.exe -m pytest -q -p no:cacheprovider
# 9 passed in 1.50s
```

```powershell
F:\anaconda3\envs\dl_env\python.exe C:\Users\Administrator\.codex\skills\researchpilot-research-code\scripts\validate_code_project.py --project-root F:\PRQ4
# status=pass; scanned=19; problems=[]; violations=[]; local GPU probe forbidden_not_run
```

```powershell
F:\anaconda3\envs\dl_env\python.exe scripts\train.py --mechanism-set always_fuse
F:\anaconda3\envs\dl_env\python.exe scripts\train.py --mechanism-set geotoken_3path
F:\anaconda3\envs\dl_env\python.exe scripts\evaluate.py --mechanism-set geotoken_3path
F:\anaconda3\envs\dl_env\python.exe scripts\train.py --mechanism-set geotoken_3path --execution-scale cloud
# three synthetic paths passed; cloud command refused with exit=1
```

The adversarial checks were run by piping an in-memory Python program to `F:\anaconda3\envs\dl_env\python.exe -` with `PYTHONPATH=F:\PRQ4\02_experiment\code\src`; no test script or fixture was written to the repository.

## Release/gate recommendation

Keep `CODE_REPORT.status = BLOCKED`. Do not promote this snapshot to clean sync or authorize formal baseline/candidate execution. Minimum unblock requires B1-B5 corrections, tests that fail on the current semantics, rerunning all synthetic checks and independent review, and then issuing a new reviewed source hash/clean-sync manifest. No scientific result can be inferred from this audit.
