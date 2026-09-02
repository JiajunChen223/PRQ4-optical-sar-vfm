# Independent V12-D0 loss/manifest final r4 (scoped)

- Scope: read-only verification of `src/geotoken3path/utils/config.py`
  lines 409--423 and the current V12 clean-sync manifest digest.
- No source, data, weights, cloud host, GPU, or sealed-test object was
  changed or accessed.

## Evidence

`config.py:409-417` selects the manifest name from the model config (`v11`
or `v12`), binds `binding_path` to that selected file, emits the matching
reference, and hashes `binding_path.read_bytes()` itself. This closes the
prior V12 reference/hash mismatch.

`config.py:418-423` is the fail-closed remote-package fallback: when the
selected source manifest is absent (or a symlink), it requires a non-symlink
`researchpilot_code_release_manifest.json` and hashes the exact embedded file
that it references.

For the current authoring tree:

```text
V12 ref: 02_experiment/code/manifests/clean_sync_manifest_v12_d0_20260829.json
V12 SHA: 9667880bfc87214e8a5bff48399046a708bc35f56e941e8378178c9b8c7c5ae6
```

Independent resolver output matched this SHA to
`sha256(V12_ref.read_bytes())`. The corresponding V11 source binding also
resolved to its own selected manifest and matching digest
(`9ec6eb8881473efb6af2365cdc51e5c1215a578058cd5094b323204c4787d792`).

## Decision

**PASS (scoped manifest-binding check).** The selected source path and hash
are now same-file bound, and the embedded release-manifest fallback hashes
the exact file opened. This is an engineering binding receipt only; it does
not constitute V12-D0 scientific evidence or authorize training/test access.
