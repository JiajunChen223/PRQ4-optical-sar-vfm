# V16 MCSL coordinator code review

This is a current-root-model coordinator review under `AM-20260830-102954-bf24a250`; it does not claim independent subagent corroboration.

## Verdict

`PASS_FOR_LOCAL_CODE_ONLY; CLOUD_SYNC_AND_AMP_PREFLIGHT_REQUIRED`

## Architecture

MCSL lives in `mechanisms/mcsl.py` and is called from the existing raw-image CROMA segmentation wrapper. Baseline and all MCSL modes retain the same CROMA bridge, two-stage always-fuse token model, classifier, optimizer entry point and trainability surface. The candidate adds no external model or frozen-baseline wrapper.

The operation is three dyadic child-detail stages. A fixed 4×3 basis spans the zero-sum child subspace, so full, shuffled and single-scale modes preserve parent means. The unconstrained control uses the same parameters but a nonzero-sum basis witness.

## Leakage and scope

The MCSL forward path receives only fused features, coarse logits, optical pixels and SAR pixels. It cannot receive target labels, boundary labels, a teacher, test objects or a local data path. Shared runtime test-seal guards are unchanged. No local real data, weights or GPU were inspected.

## Reproducibility

`v16_mcsl.yaml` and `v16_mcsl_route.yaml` freeze rank 32, three stages, 15→120 geometry, parameter ceiling, CE+Lovász, 24 epochs, CROMA, optimizer, D4 augmentation and test seal. Run-manifest and CLI allowlists bind `MCSL-01/C2/C3/C4` to their exact mechanism IDs. Baseline and candidate state-dict keys and `requires_grad` masks match.

## Adversarial checks

- zero-start output is bitwise equal to baseline bilinear logits;
- active full route adds nonzero detail while keeping original-cell residual mean below FP32 tolerance;
- unconstrained control breaks conservation;
- shuffled and single-scale controls are operation-distinct;
- detail heads receive gradients on step 1 and both guidance branches on step 2;
- formal-width MCSL adds 67,235 parameters, below the 2M ceiling;
- cloud AMP conservation and memory remain mandatory preflight evidence, not locally inferred.

No blocker or unresolved major finding remains. This review is software evidence only and makes no performance or paper claim.
