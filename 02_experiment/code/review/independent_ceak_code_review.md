# Independent CEAK successor code review

**Review scope:** local `F:\PRQ4\02_experiment\code` snapshot only. This review did not read real imagery, labels, checkpoints or pixels; did not connect to the cloud host; did not probe a GPU; and did not access the sealed test split. `CODE_REPORT.json` was not modified.

**Decision: BLOCK for code-service PASS / cloud sync.** The synthetic implementation has a coherent decoder-visible structural path and the ordinary regression suite passes, but two contract defects must be repaired before a code-only release: the zero-start interface is not gradient-live for the mechanism parameters, and the configuration/manifest layer does not bind the new M/K limits and candidate IDs to the selected mechanism. A third claim-level issue is recorded for CFEDGE: the implementation computes a two-player interaction finite difference from four coalitions, not individual Shapley allocations; the name/claim must be narrowed or the implementation must be changed.

## Evidence inspected

- `src/geotoken3path/models/fusion.py` lines 173--196, 782--982, and 1123--1174.
- `src/geotoken3path/models/factory.py` lines 33--55 and 58--231.
- `src/geotoken3path/utils/config.py` lines 195--225 and 322--422.
- `src/geotoken3path/utils/run_manifest.py` lines 271--343.
- `src/geotoken3path/engine/formal_runner.py` lines 224--252.
- `scripts/train.py` and `scripts/evaluate.py` candidate choices.
- `configs/model/geotoken3path.yaml` and `configs/experiment/approved_route.yaml`.
- `tests/unit/test_ceak_successor.py` and existing unit/integration tests.

The reviewed source snapshot hashes were recorded during review. The most relevant source hash is `fusion.py` SHA-256 `AA1BB3D8F4AE474E53AE49D529F6C914138ADD5FA17B669609F46E035002794D`; the CEAK test hash is `A78F686CC7D3C2E24F91A9511052B34B547B05546D58CED7414834D7E5E74456`.

## Checks that pass

### Shared state-dict and trainability surface

For `always_fuse` versus each of `CEAK-01`, `SUBPACK-02`, and `CFEDGE-03`, local synthetic construction showed identical state-dict key order and identical `requires_grad` masks. The same result held through the injected synthetic CROMA VFM bridge for all three candidates. This is a structural parity result only; it is not evidence about the audited remote CROMA checkpoint.

### Baseline-preserving initialization

After loading the `always_fuse` state dict into each candidate, the zero-start candidate output was bitwise equal to the baseline output on CPU synthetic token inputs. The candidate residual scales are initialized to zero and the branch is decoder-visible when enabled. This confirms no-op value parity, not gradient parity (see blocker below).

### CEAK information-flow contract

The CEAK branch is not a cosmetic scalar gate. It computes optical/SAR positive rank-8 evidence using `softplus`, normalizes the evidence distributions, forms pairwise edge conflict, subtracts conflict from optical-to-SAR attention logits, appends an explicit null logit, and writes back a dense edge-attention context plus a separate SAR private residual. The code rejects mismatched shapes and non-finite token inputs. On synthetic input, conflict and null telemetry were finite and within their declared ranges, and enabling the residual changed the output.

### SUBPACK structural path

The branch forms edge logits and rank-8 packets, keeps a shortlist of at most `M=64`, performs a hard greedy Gram--Schmidt/log-det-gain surrogate selection with `K<=32`, masks unselected edges, and includes a null completion. For `N=225`, the local CPU check reported `M=64`, `K=32`, and exactly 32 selected packets, with finite output. This is a hard edge-selection path rather than a scalar confidence gate.

### CFEDGE four-coalition path

The branch evaluates paired, optical-only, SAR-only, and null utilities over compact rank-8 evidence packets, forms a signed interaction term, routes negative interaction toward the null logit, and retains a SAR private path. Synthetic output and gradients are finite after the branch scale is enabled.

### Shape/finite checks and regression

`pytest -q` from `F:\PRQ4\02_experiment\code` passed **219 tests**; `compileall -q src tests` passed. The CEAK-specific five tests passed. These tests do not authorize cloud execution and do not validate real CROMA/data compatibility.

### Registration coverage

The three mechanism names appear in `GeoToken3PathFusion.VALID_MECHANISMS`, fusion dispatch, the local config contract, the training CLI, the evaluation CLI, the formal runner allow-list, and the run-manifest allow-list. The approved route and benchmark config identify `R-EO-CEAK-01` and `CEAK-01`; the sealed-test flag remains present.

## Blocking findings

### CEAK-R1 — zero-start is value-no-op but not gradient-live (P0)

`ceak_scale`, `subpack_scale`, and `cfedge_scale` are scalar parameters initialized to zero (`fusion.py` 185--187), and each novel residual is multiplied by its corresponding scalar (`fusion.py` 844, 919, and 969). Therefore, at the actual initialization used by a formal run,

`d(output)/d(theta_mechanism) = scale * d(residual)/d(theta_mechanism) = 0`.

An independent CPU backward check at the true zero-start state measured:

| candidate | scale gradient | internal mechanism gradient |
|---|---:|---:|
| CEAK-01 | non-zero (`0.0247945897`) | `ceak_query`, `ceak_value`, and `ceak_null`: exactly `0` |
| SUBPACK-02 | non-zero (`0.153812066`) | `ceak_query` and `ceak_value`: exactly `0` |
| CFEDGE-03 | non-zero (`0.15926522`) | `cfedge_utility.0` and `cfedge_private`: exactly `0` |

This directly violates the existing R14 contract that a zero-start candidate must be no-op **and** gradient-live, and it can delay learning all internal mechanism parameters until the scalar leaves zero. The required repair is a common, parity-preserving parameterization such as a zero-initialized output projection/readout whose upstream mechanism parameters remain gradient-live, or another explicitly audited construction. Do not repair this only for CEAK; all three candidates need the same public initialization/training contract.

### CEAK-R2 — declared M/K/rank controls are not consumed by the execution path (P1)

The YAML declares `evidence_rank: 8`, `subpack_candidate_limit: 64`, and `subpack_edge_budget: 32`, but the resolved model does not carry these fields and the fusion implementation hard-codes `self.ceak_rank = 8`, `candidates = min(64, token_count)`, and `k = min(32, candidates)`. Consequently, changing or corrupting those config values does not change the resolved snapshot or fail closed. This weakens reproducibility and the configuration-driven claim. Either make these values immutable validated constructor arguments included in the resolved protocol hash, or remove the duplicated YAML declarations and validate the fixed constants in one canonical location. Add tests for invalid rank/M/K and for the resolved snapshot binding.

### CEAK-R3 — candidate-direction ID is not bound to mechanism set in formal manifests (P1)

`build_run_manifest` and `run_formal_cloud` only check whether `candidate_direction_id` belongs to a global allow-list. A local check successfully built all of the following inconsistent manifests without rejection:

`mechanism_set=evidential_conflict_null_sink_cross_attention_kernel` with `candidate_direction_id=SUBPACK-02` and with `candidate_direction_id=CFEDGE-03`.

The formal runner also permits omitting `candidate_direction_id`. This leaves a route/manifest identity hole: a run can be labeled as one candidate while executing another, or as an unlabeled candidate. Add a canonical exact mapping and require the ID for candidate rows; reject mismatches before any cloud/data side effect. Keep baseline rows explicitly separate.

### CEAK-R4 — CFEDGE name/claim mismatch (P1 if the name is retained)

The code evaluates four coalition utilities but computes only the two-player interaction finite difference

`v(OS) - v(O) - v(S) + v(empty)`.

That quantity is a synergy/interaction score; it is not the individual Shapley value for the optical player or the SAR player. The design artifact itself describes a “two-player edge interaction credit,” so the implementation is internally consistent with that narrower operation, but the mechanism name `counterfactual_shapley_edge_credit_with_private_bypass` and any “Shapley credit” claim would overstate what is implemented. Either rename/bind the mechanism as interaction credit, or implement and test the two individual Shapley allocations from the same four coalition utilities. No causal interpretation is justified by this code.

## Non-blocking scientific/engineering risks to carry into formal protocol

1. CEAK and SUBPACK compute dense full-dimensional `D x D` query/key edge scores before rank-8 evidence or packet selection. At the formal `N=225, D=768` tap, this is materially more expensive than the plan's low-rank correction description; the 3090 claim remains conditional on cloud preflight and measured VRAM/throughput.
2. SUBPACK limits the retained shortlist after a full `topk` over all source tokens; it bounds the selected packet set but does not bound the score-computation candidate set before scoring. Do not claim full sparse-compute reduction unless a separate measured/proven path is added.
3. CFEDGE performs an `N x N` compact utility tensor for four coalitions. This is bounded at the current token count but should be measured under AMP and the actual CROMA tap shape.
4. Existing regression tests use synthetic fixtures. They do not establish official CROMA constructor compatibility, checkpoint SHA compatibility, cloud data semantics, or sealed-test integrity.
5. The tests set candidate scales to `0.5` before checking internal gradients; that demonstrates enabled-branch reachability but does not catch the true zero-start gradient defect. Add a test that asserts non-zero/gradient-live internal parameters at the actual initialization.

## Required re-review closure

1. Replace the scalar-only zero-start gate with a common no-op/gradient-live parameterization and add an initial-step gradient test for all three mechanisms.
2. Bind or canonically validate rank/M/K values in the resolved configuration and protocol hash.
3. Add exact candidate-ID/mechanism mapping and fail-closed checks in CLI/formal runner/run manifest; require candidate IDs for candidate cloud rows.
4. Resolve the CFEDGE naming/operation boundary and add coalition-value/credit tests.
5. Re-run the full synthetic suite and static compilation, then regenerate the current code report, clean-sync manifest, package, and independent review. Do not reuse the previous R14 package or `CODE_REPORT.json` as evidence for this snapshot.

**Final status:** `BLOCK — local synthetic implementation is promising and structurally visible, but not yet eligible for guarded code sync or cloud screening.`
