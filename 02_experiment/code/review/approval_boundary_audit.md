# ResearchPilot approval-boundary audit

## Scope and execution boundary

- Project root: `F:\PRQ4`
- Audit mode: read-only local inspection.
- No SSH command, cloud probe, code synchronization, data/weight download, GPU probe, or training was executed for this audit.
- The endpoint text reviewed was: `ssh -p 28974 root@connect.nmb2.seetacloud.com`.

## Evidence inspected

| Artifact | Observed fact |
|---|---|
| `00_project/researchpilot_state.json` | `active_phase=EXPERIMENT`; `active_gate=BASELINE_TRAINING_APPROVAL`; `current_status=awaiting_user`; only `PLAN_APPROVAL` is in `approved_checkpoints`; request hash is `5676b36f3c600578c46ba3afd22d964741695ec7d6d55b1b722ada4ba5e43a3a`. |
| `02_experiment/gate_status.json` | `current_gate=CLOUD_SYNC`, `status=PENDING`; `CLOUD_READY=IN_PROGRESS`; no run IDs are recorded. |
| `02_experiment/reports/formal_training_approval_request.json` | `status=request_pending_user_approval`; scope is one cloud-only baseline training/validation run; approval text explicitly says to stop at `BASELINE_RESULT_APPROVAL`; no approval decision is recorded. The file hash matches the Router request hash above. |
| `02_experiment/code/review/CODE_REPORT.json` | Local code service `PASS`, cloud code-sync preflight ready, `baseline_training_ready=false`, scientific readiness `BLOCKED`; no scientific result is claimed. |
| `02_experiment/reports/pretrained_weight_audit.json` | `status=pending`, `protocol_ready=false`; official checkpoint, compatibility, hash, license, and geography audit are not established. |
| `02_experiment/reports/pretrained_weight_policy_validation.json` | `status=blocked`; required cloud-side initialization facts are missing. |
| `02_experiment/reports/local_data_policy_audit.json` | Local policy `pass`, zero suspects; this is a local cleanliness result, not evidence of cloud data readiness. |
| `02_experiment/cloud/cloud_connection.json` | Endpoint is already recorded as user-provided with `host=connect.nmb2.seetacloud.com`, `port=28974`, `user=root`, `connection_state=USER_ENDPOINT_CONFIGURED_PENDING_PROBE`, strict host-key checking enabled, and no stored password/key. |

## Boundary decision

The SSH line is endpoint provisioning information, not a user approval event. It does **not** constitute `BASELINE_TRAINING_APPROVAL`, because the Router still records `current_status=awaiting_user`, the approval request remains `request_pending_user_approval`, and the only approved checkpoint is `PLAN_APPROVAL`.

Under the ResearchPilot approval contract, the existing `PLAN_APPROVAL` covers project/code/cloud-data preparation and baseline-frontier audit. Therefore this newly supplied endpoint can be consumed for the already-scoped `CLOUD_SYNC`/cloud-preparation lane after the endpoint schema and probe guards pass. It does not widen the scope to baseline execution. In this audit, no such probe or synchronization was performed.

The correct classification is:

```text
endpoint_status: user_provided_endpoint_pending_probe
cloud_probe_or_code_sync: permitted by existing PLAN_APPROVAL, but not executed here
BASELINE_TRAINING_APPROVAL: NOT GRANTED
baseline_training: BLOCKED
innovation_screening: NOT AUTHORIZED
```

The empty `identity_file` and blank `host_key_fingerprint` mean that a future probe must rely on the user's existing SSH agent/configuration and must first establish the expected strict host-key evidence. ResearchPilot must not bootstrap credentials, store a password, change the port, replace the endpoint, or switch proxy/VPN nodes.

## Required next checkpoint

To authorize the single seed-0 baseline, the user must explicitly approve the current request, for example:

> 批准 BASELINE_TRAINING_APPROVAL，按 formal_training_approval_request.json 执行。

Until that explicit decision is recorded against the current request hash, the next safe automatic work is limited to endpoint validation/probe, hash-bound code-only synchronization, and cloud compatibility/data-preparation audits. Real data, weights, GPU execution, and training remain cloud-only and guarded; no empirical conclusion may be emitted.

## Conclusion

The supplied SSH endpoint supports the already-approved cloud-preparation/code-sync lane; it is not, by itself, approval for baseline training. The project must remain at `BASELINE_TRAINING_APPROVAL / awaiting_user` until the user records the explicit checkpoint approval.
