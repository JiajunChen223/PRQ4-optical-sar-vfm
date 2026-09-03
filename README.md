# PRQ4 — Optical–SAR VFM Adaptation under a Frozen Downstream Protocol

Code and experiment artifacts for an empirical study of **multimodal vision-foundation-model
(VFM) adaptation for dense optical–SAR land-cover segmentation** under a strictly frozen
downstream protocol.

## Setting

- Task: 11-class dense land-cover segmentation of paired Sentinel-2 optical + Sentinel-1 SAR.
- Data: SEN12TS (Radiant/MLHub) three-region WorldCover subset — Ethiopia / Uganda / Sumatra,
  1200 parent tiles, 120×120 crops, 11 classes (ESA WorldCover 2020), CC-BY-NC-4.0.
  Follow the official SEN12TS access procedure to obtain the data.
- Backbone: CROMA-base (NeurIPS 2023) audited checkpoint, tap-connected PEFT.
- Frozen protocol: 24 epochs, CE+Lovasz 1:1, AdamW, micro-batch 16 / effective 32,
  D4 paired augmentation, single RTX 3090, validation-based model selection, sealed test.

## Reference baseline (frozen comparator)

- Best validation mIoU **49.7808 %** (epoch 18), OA 77.23 %, rare-class macro IoU 38.70 %.

## What this study reports

1. **A systematic failure map**: >18 mechanism candidates across v5–v20 (token-domain fusion,
   gradient credit, classifier geometry, spectral/trajectory, sub-pixel, energy modulation,
   pixel-domain refinement) evaluated under the identical frozen protocol, each with the same
   pre-registered decision rule (<+1 pp family closure, 3–5 seed mean ± 95 % CI).
2. **A consistent positive observation**: utilization of the **discarded non-spatial SAR
   depth group** yields a small but reproducible rare-class gain (~+1.6 pp rare macro IoU,
   13/15 seeds positive across the R2/R3/R6 family; overall best mIoU effect ≈ +0.8 pp with
   CI [+0.4, +1.2]).
3. **An evaluation-methodology contribution**: 3–5 seed + CI decision protocol for detecting
   ~1 pp effects under a frozen 24-epoch protocol (n=3 power caveat documented).

## Repository layout

```
10_CURRENT/
  00_project/        project metadata and controls
  01_literature/     frozen literature library and dataset evidence
  02_experiment/
    code/            the single latest codebase (baseline + candidate mechanisms)
    protocol/        frozen experiment protocol
    reports/         decision receipts (baseline evidence + route closures)
    claims/          claim ledger (every route decision, incl. rejected ones)
  03_writing/        (paper writing)
```

Code layout (`02_experiment/code/`):
- `src/geotoken3path/` — package: `data/` (SEN12TS loader, dynamic normalization),
  `engine/` (formal train/validation runner), `losses/`, `metrics/`, `models/`
  (CROMA bridge, fusion, factory), `mechanisms/` (candidate modules), `utils/`.
- `scripts/train.py`, `scripts/evaluate.py` — cloud entry points.
- `configs/` — frozen protocol YAMLs (benchmarks / experiment / model / runtime).
- `tests/` — unit + integration suite (155 tests).

## Reproduce

```bash
# environment: Python >=3.10, torch>=2.2, PyYAML, rasterio, pytest
cd 02_experiment/code
pytest tests/ -q                       # 155 passed
python scripts/train.py --mechanism-set always_fuse --execution-scale smoke
```

Formal cloud training requires the SEN12TS data manifest + audited CROMA weights and the
frozen protocol config; commands and manifests are recorded in `02_experiment/` reports.

## Protocol invariants

- Sealed test never accessed; validation-only model selection throughout.
- Every candidate is a single internal mechanism delta (zero-start where applicable),
  common parameter surface identical to the baseline, same optimizer/evaluator.
- All route decisions follow pre-registered rules; rejected routes are documented in the
  claim ledger as part of the evidence chain.

## License / data notice

Code license: see `LICENSE_STATUS.md` / `THIRD_PARTY.md`. Data (SEN12TS) and pretrained
weights are governed by their own licenses and are not distributed here.
