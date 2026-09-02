# Source-pin successor / Router state audit (2026-08-22)

## Scope

This is a project-local, read-only audit of the current Router state, gate
state, immutable Skill snapshot, R3 CROMA environment evidence, and the single
R1 source-pin successor. No global Skill file was modified. This audit did not
open SSH, download data or weights, probe a GPU, train, evaluate, or alter the
cloud code tree.

## Direct validation

| Item | Result | Evidence |
|---|---|---|
| Router state contract | PASS | `init_project_state.py`: `status=pass`, `existing_state_mutated=false`, no problems |
| Current Router state | PASS / automatic zone | `active_phase=EXPERIMENT`, `active_service=experiment`, `active_gate=CLOUD_ENVIRONMENT`, `plan_status=approved`, `PLAN_APPROVAL` preserved, `FINAL_CORE_APPROVAL` still pending |
| Skill snapshot | PASS | `skill_snapshot.py`: `status=PASS`; snapshot SHA256 `6a282891b03d791b5c1204bde221aab33ec151755487adc1b0d3a91f035695cd` |
| Gate at audit time | PENDING for one successor | `CLOUD_ENVIRONMENT`; gate reopen is bound to the pre-reopen gate hash and the R1 command hash |
| R3 result | CONDITIONAL / correctly blocked | Official loader/checkpoint/synthetic forward passed; input preprocessing and native depth-group contract remained open |
| R1 source-pin execution | PASS as bounded cloud audit | `cloud_exec` guard `PASS`, exit `0`, one attempt, 3.11 s, no retry |

## R1 source-pin result

The cloud output is:

`02_experiment/cloud/commands/outputs/croma_source_pin_preprocessing_audit_20260822_r1-20260822T103725937466-11272.out`

The result pins the public CROMA repository to commit
`59505a6bcadbf36ba20767270154bf9f3067c5e7`, and the fetched loader has the
same SHA256 as the previously audited current loader:

* pinned loader SHA256: `a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b`
* current loader SHA256: `a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b`
* pinned README SHA256: `c029b9a9f77d283f6290e1071480b1279feaf66885a9fa6a60e858bde77e3295`
* commit API payload SHA256: `c4aeef19a8399328451f5f7e625e1f38c54fa35bf3de6ef1b5da3b51dca10efc`

The audit observed README terms `mean`, `std`, `clip`, `255`, and `uint8`,
but observed no explicit band terms (`VV`, `VH`, `B01`--`B12`). Its own
decision is therefore correctly conditional:

* source pin: PASS;
* official README recipe: observed, but requires dataset-policy lock;
* formal input contract: pending explicit band order, normalization, and
  nodata policy;
* gate advance: false.

## Compliance decision

`CONDITIONAL_PASS_FOR_SUCCESSOR_AUDIT_ONLY`.

The successor is compliant with the current Skill as a single, guarded,
cloud-only, read-only environment/source audit inside the existing
`PLAN_APPROVED_TO_AUTOMATIC_EXPERIMENT` zone. It must not be interpreted as
formal initialization, data verification, hardware adaptation, baseline
evidence, or scientific result.

The gate must remain blocked after the local result is recorded until the
following are explicit and hash-bound:

1. the exact CROMA input band order for SEN12TS (including the SAR channel
   selection and the 12 optical channels);
2. the exact normalization formula and scope (per-tile/per-channel or fixed
   statistics), clipping and zero-standard-deviation policy;
3. the nodata handling and finite-value checks before the official forward;
4. the bridge contract for the four non-spatial depth taps and its real module
   paths, without promoting synthetic hooks to real-data evidence;
5. source/license lineage sufficient for the later pretrained-weight audit.

## Findings for parent Router

* The current Router state is valid, but after the gate reopen its derived
  `experiment_status=cloud_environment_blocked` may be stale while the child
  gate is temporarily `PENDING`; reconcile it only after the R1 result is
  recorded, using the Router update script.
* The R3 local result contains a traceability typo: its `stderr_file` uses
  `20220822` instead of the actual output directory timestamp
  `20260822T101639654146-37140`. Correct that project-local artifact before
  packaging or citing the R3 evidence; do not replay R3.
* The gate display reports the enclosing `CLOUD_READY` milestone as PASS while
  the child `CLOUD_ENVIRONMENT` gate has been blocked/pending in its history.
  Treat the child gate status and its latest evidence as authoritative; do not
  use the milestone label as environment clearance.

## Hash ledger

| Artifact | SHA256 |
|---|---|
| `00_project/researchpilot_state.json` | `ed54aecc8fc3b31213d68abcb49e32d6e5a2273b3052ffea3e82c06e8197e7d9` |
| `00_project/runtime/skill_snapshot.json` | `6a282891b03d791b5c1204bde221aab33ec151755487adc1b0d3a91f035695cd` |
| `02_experiment/gate_status.json` (at audit) | `ad2018cfb93e76170d625b377aab67dcd3ff2e40151eef7bc0f0df224e8f85cb` |
| R3 local result | `efc01076b63487f70b2a38cfc329ed12ce65bcccf397d3f696c4fa24d3f3860a` |
| R1 command file | `1062f38e01d02dbdf8226c9169deb05a5c4ecd42cb5395eada0a6ac1ca31b366` |
| R1 source-pin control | `16648d29bba84e4f3ddc6f5b05d584207bda6178121c822dceea47e1363b2ae4` |
| R1 gate-reopen evidence | `d1fd4a9e87455a87a8ae223a28b8f5ddb4a62477bfa74afab4f6923c24981778` |
| R1 stdout | `d191b7b0c0d54517c10ec33ee28d25f786d198792f91ca216b9b761d67357e8b` |
| R1 stderr (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

**Next automatic action:** write the structured R1 local result, correct the
R3 path typo, keep `CLOUD_ENVIRONMENT` blocked, and only then prepare the
dataset-contract successor. No formal data download or training is allowed
from this receipt.
