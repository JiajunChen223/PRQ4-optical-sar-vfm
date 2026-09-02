# SEN12TS successor vs active code contract audit (2026-08-22)

## Decision

`PROTECTED_BLOCK_FOR_CODE_REPAIR`.

The active code-sync tree is still hard-coded to the historical
Copernicus-Bench/BigEarthNet contract, while SEN12TS is only a conditional,
blocked successor. The mismatch is scientific data-contract state, not a
cosmetic naming issue. Under the current ResearchPilot Skill, code repair must
not start until the SEN12TS Plan handoff is canonical, validated, and bound to
the applicable Plan decision. No cloud action, data access, weight access,
training, or evaluation was performed for this audit.

## Direct findings

| Layer | Active BigEarthNet binding | SEN12TS state | Severity |
|---|---|---|---|
| Benchmark config | `dataset_id=copernicus_bench_bigearthnet_s1s2_10pct`; root `/root/autodl-tmp/copernicus_bench` | No active SEN12TS config | Blocking |
| Dataset validator | `APPROVED_DATASET_ROOT=.../copernicus_bench`; exact old dataset ID required | No SEN12TS selector/root/11-class contract | Blocking |
| Config loader | `utils/config.py` always loads `configs/benchmarks/copernicus_bench.yaml` | No successor loader path | Blocking |
| Unit/integration fixtures | BigEarthNet dataset ID and root are asserted | No SEN12TS fixture | Blocking |
| Public-release metadata | `THIRD_PARTY.md` describes Copernicus-Bench/BigEarthNet | SEN12TS license/attribution is not reflected in active code metadata | Blocking for release |
| Plan handoff | Current `plan_handoff.json` is `ready` and names BigEarthNet | SEN12TS successor handoff is `blocked_pending_cloud_dataset_contract`; validation status is `reject` | Protected scope mismatch |

The exact active-code locations are:

* `02_experiment/code/configs/benchmarks/copernicus_bench.yaml:1,15`
* `02_experiment/code/src/geotoken3path/data/contracts.py:14,81`
* `02_experiment/code/src/geotoken3path/utils/config.py:57`
* `02_experiment/code/tests/unit/test_dataset_manifest.py:14-22`
* `02_experiment/code/tests/integration/test_smoke_runtime_contract.py:19`
* `02_experiment/code/THIRD_PARTY.md:7`

The successor contract requires SEN12TS WorldCover segmentation with 11
classes, ignore index 255, raw S2 14-band selection `[0..11]`, raw S1 19-band
selection `[1,0]` reordered to canonical `[VV,VH]`, parent-first split, and a
cloud-only root/manifest. None of these selectors or class semantics are
enforced by the active validator.

## Plan and gate interpretation

The current Router state is `EXPERIMENT / experiment`, with
`PLAN_APPROVAL` preserved and the active gate `CLOUD_ENVIRONMENT/BLOCKED`.
The current approval is bound to the historical ready handoff, whose core
dataset is BigEarthNet. The SEN12TS successor files state:

* `plan_handoff_successor_sen12ts_20260821.json`: `blocked_pending_cloud_dataset_contract`;
* `plan_handoff_validation_sen12ts_20260821.json`: `status=reject`;
* `plan_approval_request_sen12ts_20260822.json`:
  `draft_not_recorded_due_fixed_checkpoint_order`;
* `sen12ts_code_adaptation_plan_20260822.md`: `not_implemented, not_synced,
  not_executed`.

Therefore the source-pin/CROMA environment successors do not authorize a
SEN12TS data download or a SEN12TS code mutation. The current
`CLOUD_ENVIRONMENT` blocker remains, and `CLOUD_DATA_DOWNLOAD` must not open.
The prior `data_task_contract_audit.json` correctly records the old
BigEarthNet-vs-segmentation mismatch; the SEN12TS successor report is not a
passing replacement.

## Minimal repair order

1. **Plan service:** complete one canonical SEN12TS successor handoff by
   preserving the approved route/mechanism fields while replacing the core
   dataset, task labels, split, storage, license, and input contract. Run the
   current Plan validators until `handoff_status=ready`; do not overwrite the
   historical handoff or reinterpret a rejected artifact as approval.
2. **Decision binding:** bind the dataset change to the appropriate explicit
   Plan decision/request artifact. The existing approval is for the old core
   dataset and cannot be silently replayed for SEN12TS.
3. **Code service repair:** only after step 2, add a SEN12TS benchmark config
   (or governed replacement), change the manifest root/ID/11-class semantics,
   implement explicit `[0..11]` S2 and `[1,0]` S1 selectors, preserve `[VV,VH]`
   canonical order, enforce ignore 255 and parent-first split metadata, and
   update tests/README/license metadata.
4. **Local review and sync:** run the full code validator and independent
   review, regenerate the clean sync manifest, and deliver one guarded
   code-only sync. Because the dataset contract changes, do not trust the
   existing R5/R3 environment evidence as a complete formal route handoff;
   re-check the earliest affected cloud gates after sync.
5. **Experiment service:** re-run the cloud input/data contract checks against
   the synchronized SEN12TS-aware code. Keep data download blocked until the
   SEN12TS manifest, selectors, labels, license, hashes, storage ledger,
   preprocessing parity, and split/leakage evidence all pass.

No step above changes the primary method route. It only restores consistency
between the approved task, selected successor dataset, code, and gate
evidence.

## Evidence hashes

| Artifact | SHA256 |
|---|---|
| Active benchmark config | `2aa5d3b6d59e194795bfe9448d24daacae3170a78adebad2de5fd81141c7249e` |
| Active dataset validator | `d0113674f481c3c68f6b0da51b7082b3485d72f0c8550179253925c506788581` |
| Active config loader | `22a96b169907aa3f7e6c43141e9759dfb9c24da4c5e3bb2a12e1bfef8268a678` |
| Active unit fixture | `699305802801db45a1b9302bbfde42022df64d91e358a8d7eaa0eb78826f3963` |
| Active integration fixture | `d77c2fccf0a18b16f36ee4ab696aef5f58d99b58250e81571e3f73dcefd522d0` |
| Current ready BigEarthNet handoff | `793c257572c3482968129dbb72e8e0b9413b2a035794d104fa986e792a814b0f` |
| SEN12TS successor handoff | `401fb6c0e0a627c288645422af33ee84f85b07f488c7e6513b93db659f496d80` |
| SEN12TS handoff validation | `2c6868706afc73a106369a2453b8919cafd36f82d6c45c74701c2016378ee916` |
| SEN12TS adaptation note | `495aa20a1789569f72c129a0275f8e284e1ee8c0dc6bb99234a8f2dd837b765d` |
| This audit report | recorded in the parent stage receipt after write |
