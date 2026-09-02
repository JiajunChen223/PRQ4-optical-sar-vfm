# RCCR CROMA interface and no-op parity audit

**Audit ID:** `RCCR-CROMA-INTERFACE-AUDIT-R1`  
**Scope:** read-only local code inspection plus synthetic CPU fixture checks  
**Status:** `CONDITIONAL_INTERFACE; BLOCKED_FOR_RCCR_CODE_READINESS`  
**Scientific gate:** unchanged  
**Cloud/data/weights/GPU:** not accessed

## Executive finding

The current code has a feasible **minimal insertion boundary** for a new RCCR
operator, but RCCR is not implemented or contract-ready. The safest boundary
is a wrapper adjacent to `CCPAInputAdapter` in
`02_experiment/code/src/geotoken3path/models/croma_bridge.py`, between the
unimodal S1/S2 encoder outputs and the official CROMA `cross_encoder`. This is
**pre-CROMA cross-modal interaction**, not an intervention before the S1/S2
unimodal patch projectors. The current public raw-image interface and tapped
feature interface can remain unchanged if RCCR preserves the token shape.

There is one blocking trainability finding in the current CCPA-shaped wrapper:
under the existing `tap_connected` policy, its candidate adapter parameters are
frozen because the policy recognizes only `s1_encoder.*` and `s2_encoder.*`
paths. This gap must be resolved and explicitly tested before using the same
wrapper pattern for RCCR. It does not by itself invalidate the already-recorded
CCPA result; it means the local code package does not yet prove that the CCPA
adapter was trainable under the tap-connected formal contract.

## 1. Evidence and inspected paths

- `models/croma_bridge.py:28-89`: `CCPAInputAdapter`; it runs S1/S2 encoders,
  creates an optical-side residual, and calls `cross_encoder`.
- `models/croma_bridge.py:92-205`: `CromaDepthTapAdapter`; captures stage taps
  and, when present, adds `ccpa_residual` to every stage whose shape matches.
- `models/croma_bridge.py:214-269`: `CromaBackboneBridge`; validates raw input
  and tapped feature shapes.
- `models/croma_bridge.py:321-365`: `_tap_connected_parameter` and
  `freeze_backbone_for_peft`.
- `models/factory.py:49-85`: formal raw-image factory and CCPA wrapper
  selection.
- `models/fusion.py:764-881, 884-1043`: token-stage mechanisms and the shared
  token model; no RCCR mechanism ID exists.
- `configs/experiment/approved_route.yaml`: current route and mechanism
  registry; RCCR is not registered.
- `engine/formal_runner.py:31-44, 247-299, 316-326`: formal loader, optimizer,
  and validation-only early stopping contract.
- `configs/runtime/3090_plan.yaml`: declared 3090/24-epoch runtime contract.
- Existing local reports:
  `code/review/validate_code_project_ccpa_r1.json`,
  `code/review/code_release_report_ccpa_r1.json`, and
  `code/review/CODE_REPORT.json`.

No real data, real weight binary, cloud process, local GPU probe, or sealed-test
artifact was opened.

## 2. Minimal insertion point

### What is feasible now

`CCPAInputAdapter` currently receives final unimodal encoder tokens:

```text
raw optical [B,12,H,W] -> s2_encoder -> optical [B,N,D]
raw SAR     [B, 2,H,W] -> s1_encoder -> SAR     [B,N,D]
                                      -> adapter -> cross_encoder
```

The wrapper's `_adapt` operation is applied immediately before
`cross_encoder`, and the official cross encoder receives `x=sar` and
`context=adapted_optical`. A new RCCR wrapper can be inserted at this same
boundary and preserve the current public factory/loader path if it:

1. accepts two `[B,N,D]` tensors;
2. returns two `[B,N,D]` tensors to the cross encoder;
3. preserves CROMA's attention-bias and keyword call contract;
4. does not change the raw `[B,12,H,W]` and `[B,2,H,W]` input interface; and
5. exposes an explicit mechanism telemetry object without changing the required
   CROMA output keys.

### What is not currently exposed

There is no local hook for a true pre-unimodal-encoder or raw input-projector
RCCR operation. `CromaBackboneBridge` passes raw tensors into the injected
backbone, while `CromaDepthTapAdapter` hooks transformer-depth modules after
the unimodal encoders have already started. Therefore, “pre-CROMA” must be
defined precisely as **before CROMA's cross-modal encoder**, unless a separate
audited CROMA constructor/interface is added. A plan that claims intervention
before the S1/S2 patch projectors would exceed the current code contract.

### Stage-tap side-channel hazard

The current CCPA wrapper stores `ccpa_residual`; `CromaDepthTapAdapter` then
adds that residual to every optical stage whose shape matches. Since both
configured stages are `[B,N,D]`, the same residual can be applied to mid and
late taps. RCCR should not silently reuse this side channel. It needs one
explicit placement and one explicit write path; otherwise a symmetric RCCR
update could be applied twice or be misreported as a stage-level mechanism.

**Insertion verdict:** `feasible_at_pre_cross_encoder_wrapper; not_feasible_as_true_raw_stem_adapter_without_new_audit`.

## 3. Fixed input and feature contracts

The current bridge enforces:

| Boundary | Required contract | RCCR implication |
|---|---|---|
| Raw optical | `[B,12,H,W]`, float32 after normalization | unchanged; no band reorder or extra channel |
| Raw SAR | `[B,2,H,W]`, float32 after normalization | unchanged; model order remains audited VV/VH |
| Paired raw inputs | same batch and spatial size | RCCR cannot introduce independent crops |
| Stage optical/SAR taps | same `[B,N,D]` per stage | RCCR must preserve N and D; current formal D is 768 |
| SAR depth group | `[B,N,4,D]` | unchanged; RCCR must not reinterpret depth taps as a spatial grid |
| Bridge outputs | exact keys `optical`, `sar`, `sar_depth_group` | telemetry must be auxiliary, not a replacement |
| Token grid | CCPA requires square `N`; token decoder also requires square grid | RCCR should fail closed on incompatible N, not reshape silently |

The current synthetic fixture uses `N=16,D=32`; the formal route records
`D=768`. The local audit did not infer the cloud token count or GPU memory.

## 4. Tap-connected trainability audit

### Static policy

`_tap_connected_parameter` currently unfreezes:

- all `s1_encoder.*` parameters;
- S2 stem parameters and `s2_encoder.transformer.layers.0` through `.5`;
- no S2 layers `.6` through `.11`;
- no `cross_encoder.*`, GAP heads, or unrelated paths.

This is the intended common CROMA trainability mask for baseline and future
candidates. However, when `CCPAInputAdapter` is nested inside
`CromaDepthTapAdapter`, its parameters are named under a wrapper path such as
`backbone.query`, `backbone.key`, `backbone.value`, `backbone.null_cost`, and
`backbone.out`. After `removeprefix("backbone.")`, these names do not start
with `s1_encoder.` or `s2_encoder.` and are therefore set to
`requires_grad=False` by the current policy.

### Synthetic read-only probe

Using the existing `SyntheticCroma` fixture and
`build_vfm_segmentation_model` with `mechanism_set="ccpa_input_adapter"` and
`backbone_policy="tap_connected"` produced:

- S1 layers through the retained path: trainable;
- S2 stem/layers 0–5: trainable;
- S2 layers 6–7 in the fixture: frozen;
- cross encoder: frozen;
- CCPA `query`, `key`, `value`, `null_cost`, and zero-start `out`: **frozen**;
- tap-connected CROMA trainable-parameter count in the fixture: 28 tensors.

This is a blocking contract observation, not a cloud scientific conclusion.
The formal CCPA artifact must be checked for its actual parameter mask before
any interpretation of CCPA as evidence for a trainable pre-CROMA mechanism.

### Trainability verdict

`COMMON_TAP_MASK: PASS_LOCAL_STATIC`  
`CANDIDATE_ADAPTER_MASK: BLOCKED_UNBOUND_IN_CURRENT_POLICY`

For RCCR, candidate-only coupling parameters must be explicitly whitelisted by
role or module path while the common CROMA mask remains byte-for-byte identical
to the baseline. A broad “unfreeze the wrapper” rule is not acceptable because
it could accidentally unfreeze the cross encoder or GAP heads.

## 5. Zero-start and baseline identity

### What the current CCPA code demonstrates

`CCPAInputAdapter.__init__` zero-initializes `out.weight`. `_adapt` computes
`residual = out(context) * accepted`; with the zero-start state, the residual
is exactly zero and `adapted_optical` equals the original optical token. In a
synthetic same-backbone comparison, the raw tapped optical, SAR, and depth-group
outputs differed by maximum absolute error **0.0** between the unwrapped
backbone and the zero-start CCPA wrapper. This confirms the intended no-op
construction for the current fixture.

The existing test only asserts residual shape and finite null mass
(`tests/integration/test_croma_bridge.py::test_ccpa_input_adapter_produces_null_and_pre_cross_residual`); it does **not** assert full baseline parity or tap-connected candidate trainability.

### RCCR requirement

RCCR cannot be considered no-op compatible until a synthetic test proves all of
the following with identical audited-backbone weights and deterministic mode:

- zero-start coupling produces the same `x` and `context` tensors as baseline;
- the fixed-width packet has exactly the same shape and normalization range;
- common/private split and packer introduce no bias/normalization drift;
- the inverse coupling reconstructs its input within a declared tolerance;
- the first nonzero coupling update changes both intended modality paths and
  no frozen CROMA path;
- the output/tap side channel is applied once, not once per stage by accident.

`out.weight`-style zero initialization can make inner relation parameters have
zero gradient on the first backward pass. The test must therefore distinguish
“parameter is registered and optimizer-visible” from “all inner parameters
receive nonzero gradient at initialization.” A two-step synthetic reachability
probe or a nonzero-branch fixture is required.

## 6. 3090 and 24-epoch feasibility audit

### Confirmed from code/config

- Formal non-smoke scales use the optimized loader path: four workers, pinned
  memory, persistent workers, prefetch factor 2, nonblocking copies, and AMP
  when the cloud device is CUDA (`formal_runner.py:31-44,247-280`).
- The optimizer consumes every parameter with `requires_grad=True`
  (`formal_runner.py:298-302`), so an unbound RCCR wrapper would silently be
  omitted from AdamW rather than fail loudly.
- The runtime contract declares microbatch 16, effective batch 32, gradient
  accumulation 2, AMP, max 24 formal epochs, and validation-only early
  stopping with burn-in 8/patience 5/restore-best.
- No local GPU probe was performed; cloud preflight remains authoritative.

### Conditional feasibility

A pre-cross RCCR with two low-rank operations over existing `[B,N,768]` token
fields is structurally more bounded than CCPA's local offset window and does
not require a global token graph, native-resolution cache, teacher model, or
new weights. It is **conditionally compatible** with one RTX 3090/24GB, but
peak VRAM, throughput, and wall time are not established locally. The cloud
preflight must measure the actual CROMA route; FLOPs or a local CPU fixture
cannot substitute for that measurement.

### 24-epoch parity

The formal runner accepts the 24-epoch cap and enforces the validation-only
early-stopping schema. RCCR must use the same resolved runtime snapshot as the
accepted baseline. A new candidate-specific batch, scheduler, warmup, or
early-stopping setting would be a training-object parity failure.

**Compute verdict:** `CONDITIONALLY_3090_FEASIBLE; CLOUD_PREFLIGHT_REQUIRED`.

## 7. Mandatory code-contract tests before RCCR synchronization

These are required tests, not a request to run them now. They must be added to
the local synthetic suite and pass before any RCCR code package is synchronized.

### P0 — interface and mechanism registration

1. **Mechanism registry test:** RCCR has one stable mechanism ID, is accepted
   only through the model factory/config, and is rejected when undeclared.
2. **Insertion-point test:** RCCR is instantiated exactly once at the
   pre-cross-encoder wrapper; no post-CROMA fusion branch or external frozen
   baseline wrapper is introduced.
3. **Single-mechanism-delta test:** resolved baseline and RCCR configs differ
   only in the declared RCCR mechanism fields; dataset, split, initialization,
   augmentation, optimizer, scheduler, trainability policy, and evaluator
   hashes are identical.
4. **Candidate identity/run-manifest test:** manifest records route ID,
   candidate ID, insertion point, mechanism version, candidate parameter count,
   common protocol hash, and sealed-test status.

### P0 — shapes and CROMA compatibility

5. **Raw input contract:** reject non-float32, wrong channel counts, unequal
   batch sizes, and unequal spatial sizes; accept only `[B,12,H,W]` and
   `[B,2,H,W]` after normalization.
6. **Pre-cross token contract:** RCCR accepts and returns `[B,N,D]` optical and
   SAR tokens with identical `B,N,D`; reject rank/channel drift.
7. **Depth-tap contract:** RCCR leaves `[B,N,4,D]` SAR depth groups and all
   configured stage names unchanged; it must not treat depth as spatial.
8. **Output-key contract:** raw model still returns exactly the bridge keys
   `optical`, `sar`, and `sar_depth_group`; RCCR telemetry is auxiliary.
9. **Non-square fail-closed test:** if the selected RCCR implementation needs
   a square token grid, reject invalid `N` rather than silently reshaping.

### P0 — no-op identity, reversibility, and side effects

10. **Exact no-op test:** with zero-start RCCR and identical backbone weights,
    baseline and candidate pre-cross `x/context`, tapped features, and dense
    logits match within a declared tolerance.
11. **Triangular inverse test:** synthetic random token packets satisfy
    `inverse(forward(packet))` within a declared finite tolerance; NaN/Inf and
    overflow paths fail closed.
12. **Coverage test:** no modality packet is deleted at zero and nonzero
    coupling; both private streams remain addressable in the returned packet.
13. **Single-application test:** the RCCR correction is applied once; no hidden
    residual is added again by every mid/late tap.
14. **Non-noop flow test:** after deterministic nonzero candidate initialization
    in a synthetic fixture, both intended modality paths change while the
    baseline's frozen cross encoder and unused heads remain unchanged.

### P0 — trainability and optimizer parity

15. **Tap-connected common-mask test:** S1 all layers and S2 stem/layers 0–5
    match baseline exactly; S2 layers 6–11, GAP heads, and cross encoder remain
    frozen.
16. **RCCR candidate-mask test:** only explicitly declared RCCR parameters are
    added to the candidate trainable set; they are not silently excluded by
    `_tap_connected_parameter` and no unrelated CROMA path is unfrozen.
17. **Optimizer inclusion test:** every RCCR parameter with `requires_grad=True`
    appears exactly once in AdamW; every frozen parameter is absent.
18. **Gradient reachability test:** final coupling projection receives a finite
    gradient from a synthetic loss; a two-step probe demonstrates reachability
    of inner coupling parameters after the zero-start branch moves.
19. **Parameter-mask/hash test:** baseline and RCCR common parameter masks and
    protocol hashes match; candidate-only parameters are listed explicitly.

### P1 — runtime, seal, and resource contract

20. **24-epoch/early-stop parity:** RCCR resolves the same max epoch, burn-in,
    patience, min-delta, restore-best, and validation-only monitor as baseline.
21. **Loader parity:** formal RCCR config resolves the same four-worker,
    pinned/persistent/prefetch/nonblocking/AMP contract.
22. **Test-seal test:** smoke/baseline/screening/confirmation RCCR runs reject
    test split; only an explicitly authorized final-test scale can open it.
23. **Cloud-path/data boundary test:** no local data, weights, checkpoint, or
    native-resolution cache enters the code-sync tree.
24. **Synthetic memory/shape stress test:** CPU fixture exercises the largest
    declared token shape without global adjacency or hidden persistent cache;
    actual VRAM remains a cloud-preflight measurement.

## 8. Read-only audit verdict

| Item | Verdict | Reason |
|---|---|---|
| Minimal pre-cross insertion point | **Conditional pass** | CCPA-shaped wrapper boundary exists before `cross_encoder`; true raw-stem hook does not. |
| Fixed input shape | **Pass for unchanged interface** | Bridge enforces 12 optical/2 SAR raw channels and `[B,N,D]`/`[B,N,4,D]` tap contracts. |
| Tap-connected common mask | **Pass locally** | S1 and S2 0–5 policy is explicit and test-covered for unwrapped synthetic backbone. |
| Candidate adapter trainability | **Blocker** | Current wrapper parameters are frozen by the name-based tap policy. |
| Zero-start baseline identity | **Pass for current CCPA fixture** | Zero output projection gives max tapped difference 0.0; RCCR itself is not implemented/tested. |
| 3090/24 feasibility | **Conditional** | Runtime budget is present; cloud preflight must measure actual RCCR/CROMA VRAM and wall time. |
| RCCR code readiness | **Not ready** | No RCCR mechanism/config/factory/manifest contract exists yet. |

**Final audit verdict:** `DO_NOT_SYNCHRONIZE_RCCR_YET`.

The next safe code action, if the route survives plan/prior-art review, is a
local-only implementation and test pass that first repairs explicit candidate
parameter registration, proves exact no-op parity, and preserves the current
CROMA input/tap/test-seal contracts. This audit itself made no code or gate
changes.
