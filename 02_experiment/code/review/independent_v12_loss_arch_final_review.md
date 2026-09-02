# Independent V12 resolver fallback final review

**Scope:** V12 `resolve_approved_config` clean-sync manifest binding and its
remote embedded-manifest fallback.  This was a read-only audit: no source,
data, weights, cloud host, GPU, or sealed-test object was accessed or changed.

**Decision: BLOCKED**

The fallback branch is not closed for the local/source layout.  When resolving
`v12_objective.yaml`, the resolver advertises the V12 manifest path but hashes
the V11 manifest bytes.  The resulting reference/hash pair is internally
inconsistent and will fail any consumer that verifies the referenced manifest.

## Evidence

Current resolver lines `402-423` use a fixed path:

```text
binding_path = code_root / "manifests" /
               "clean_sync_manifest_v11_tasr_20260829.json"
```

The branch then chooses the V12 filename for the reference when
`model_config_name == "v12_objective.yaml"`, but still computes
`code_sync_manifest_sha256` from `binding_path` (the V11 file).  Independent
resolution produced:

```text
ref:       02_experiment/code/manifests/clean_sync_manifest_v12_d0_20260829.json
stored:    9ec6eb8881473efb6af2365cdc51e5c1215a578058cd5094b323204c4787d792
actual ref ede479d8b036513cd17aec32466b2d0eae941703bb4e82ef5c28c4f6556f73c9
actual V11 9ec6eb8881473efb6af2365cdc51e5c1215a578058cd5094b323204c4787d792
match_ref: False
```

Thus the V12 source resolver currently binds `v12_d0_ref -> V11_hash`.

The existing configuration/manifest tests still pass (`30 passed` across the
V12, resolved-config, and run-manifest targeted suites), but they do not verify
that `code_sync_manifest_sha256` equals the bytes of the file named by
`code_sync_manifest_ref`.  The reported full-suite status of `310 passed` and
validator `117/0` therefore does not close this specific binding defect.

## Fallback assessment

- **Local/source path:** **FAIL** for V12 due to the mismatched path/hash pair.
- **Remote embedded path:** the `researchpilot_code_release_manifest.json`
  branch computes a hash of the file it opens and is structurally sound, but
  it can only be trusted after the local V12 binding is repaired and a temporary
  package-layout simulation verifies that the embedded manifest is the V12
  release manifest, not a stale V11 one.
- **V11 compatibility:** the fixed V11 source path and V11 reference remain
  consistent for `v11_tasr.yaml`; the defect is introduced by the attempted
  V12 coverage.

## Required repair

Select the binding path from the same `manifest_name` used for the reference,
for example choose `clean_sync_manifest_v12_d0_20260829.json` for
`v12_objective.yaml` and `clean_sync_manifest_v11_tasr_20260829.json` for
`v11_tasr.yaml`, then hash that selected path.  Keep the embedded fallback
fail-closed when the selected source manifest is absent, and add a regression
test for both routes asserting:

```text
sha256(ref_path.read_bytes()) == resolved.code_sync_manifest_sha256
```

Also run a temporary remote-layout simulation with only the embedded release
manifest and assert that V12 resolution returns the embedded reference and
matching digest.  Regenerate the V12 clean-sync manifest/package and refresh
`CODE_REPORT.json` after the repair; do not sync or run V12-D0 until those
receipts are updated.

This is an engineering release-binding verdict only.  It makes no claim about
V12 objective performance, CMCD, or any scientific result, and does not alter
the sealed-test boundary.
