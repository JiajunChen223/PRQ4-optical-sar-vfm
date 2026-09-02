# GeoToken-3Path research code

This configuration-driven tree contains the V19 **Classifier-Tangent Selective
Projection (CTSP)** successor for paired optical--SAR land-cover
segmentation. Real data, checkpoint binaries and caches remain cloud-only.

Current route: `R-EO-CTSP-V19-01` / `CTSP-01`. V5--V18 failures remain
preserved as historical negative evidence; they are not revived by this route.

CTSP uses the single audited CROMA forward and the shared segmentation head.
At the final 15x15 token stage it decomposes the fused residual into the
classifier row-space (normal) and its tangent component, contracting only the
normalized class-logit-visible displacement to a frozen, train-only calibrated
bound. The tangent component and the original head are preserved; the module
has no trainable parameters, no second backbone pass, no dense 225x225
attention, no label guidance and no auxiliary loss. Matched mid-stage,
whole-residual and identity controls remain locked until CTSP-01 reaches the
predeclared +2pp gate.

The audited CROMA model, SEN12TS split, optimizer, augmentation, trainability
and 24-epoch budget remain fixed. The sealed test remains closed.

Local synthetic checks use the configured Python environment and never touch a
device or filesystem dataset:

```powershell
F:\anaconda3\envs\dl_env\python.exe -B -m pytest F:\PRQ4\02_experiment\code\tests -q
F:\anaconda3\envs\dl_env\python.exe -B F:\PRQ4\02_experiment\code\scripts\run_v19_ctsp_hard_contract.py --output F:\PRQ4\02_experiment\reports\v19_ctsp_synthetic_hard_contract_local.json
F:\anaconda3\envs\dl_env\python.exe -B F:\PRQ4\02_experiment\code\scripts\train.py --mechanism-set ctsp_classifier_tangent_selective_projection --candidate-direction-id CTSP-01 --route-variant v19_ctsp --execution-scale smoke --device cpu
```

Formal cloud execution must resolve the approved YAML set, pass the complete CROMA compatibility audit, preserve the test seal, and emit a run manifest. A local smoke pass is software-contract evidence only, never a baseline or method result.
