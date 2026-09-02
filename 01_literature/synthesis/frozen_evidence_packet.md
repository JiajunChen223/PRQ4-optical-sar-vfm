# Frozen evidence packet

- library_version_id: lib-2026-08-20-a9987c7fcf59
- core evidence selected: 10 / 10
- prior-art evidence selected: 15

## Core evidence

- 503779fc4cf7c07e — GeoRL: Adaptive tokenization via reinforcement learning for remote sensing foundation models (Pattern Recognition, 2026-07-01)
- 5ed0ce9794a2038b — MM-OVSeg: Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing (CVPR 2026, 2026-06-01)
- 194d31371ff32cb8 — Adapter-Enhanced SAR-Optical Joint Segmentation With Primary Modality Regularization and Sparse Attention Adjustment (The Photogrammetric Record, 2026-05-19)
- c36b49e5c89af7cb — MaRS: A Multi-modality Very-high-resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning (Proceedings of the AAAI Conference on Artificial Intelligence, 2026-03-14)
- 2c2fde7b41f20580 — MAESTRO: Masked AutoEncoders for Multimodal, Multitemporal, and Multispectral Earth Observation Data (2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026-03-06)
- 37abd1c1ea8749d9 — MHFNet: Multimodal hybrid fusion framework for misaligned SAR-Optical ship detection (ISPRS Journal of Photogrammetry and Remote Sensing, 2026-01-15)
- 4586bd5cafe7048e — SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing (arXiv, 2025-07-18)
- f0e16d1ab9775b65 — MAPEX: Modality-Aware Pruning of Experts for Remote Sensing Foundation Models (arXiv, 2025-07-10)
- 98ee47b14b96a7dd — DUNIA: Pixel-Sized Embeddings via Cross-Modal Alignment for Earth Observation Applications (ICML 2025, 2025-06-12)
- 11152c79b5c7a34a — RingMoE: Mixture-of-Modality-Experts Multi-Modal Foundation Models for Universal Remote Sensing Image Interpretation (arXiv, 2025-04-04)

## Prior-art evidence

- 4586bd5cafe7048e — SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing (arXiv, 2025-07-18)
- f0e16d1ab9775b65 — MAPEX: Modality-Aware Pruning of Experts for Remote Sensing Foundation Models (arXiv, 2025-07-10)
- 11152c79b5c7a34a — RingMoE: Mixture-of-Modality-Experts Multi-Modal Foundation Models for Universal Remote Sensing Image Interpretation (arXiv, 2025-04-04)
- dcac468c6ad142b2 — STARS: Shared-specific Translation and Alignment for missing-modality Remote Sensing Semantic Segmentation (International Journal of Applied Earth Observation and Geoinformation, 2026-01-24)
- 37abd1c1ea8749d9 — MHFNet: Multimodal hybrid fusion framework for misaligned SAR-Optical ship detection (ISPRS Journal of Photogrammetry and Remote Sensing, 2026-01-15)
- c0a46640de12e44e — SMAF-net: semantics-guided modality transfer and hierarchical feature fusion for optical-SAR image registration (International Journal of Applied Earth Observation and Geoinformation, 2025-09-01)
- 194d31371ff32cb8 — Adapter-Enhanced SAR-Optical Joint Segmentation With Primary Modality Regularization and Sparse Attention Adjustment (The Photogrammetric Record, 2026-05-19)
- 503779fc4cf7c07e — GeoRL: Adaptive tokenization via reinforcement learning for remote sensing foundation models (Pattern Recognition, 2026-07-01)
- 5ed0ce9794a2038b — MM-OVSeg: Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing (CVPR 2026, 2026-06-01)
- 0c1bf032b88b8591 — SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images (CVPR 2025, 2025-06-12)
- 98ee47b14b96a7dd — DUNIA: Pixel-Sized Embeddings via Cross-Modal Alignment for Earth Observation Applications (ICML 2025, 2025-06-12)
- 83ec0d9909716c3c — How Usable Are Geospatial Foundation Models? A Systematic Evaluation of 89 Models (arXiv (Cornell University), 2026-08-04)
- 210ebc8967ff256a — Optimal Transport Adapter Tuning for Bridging Modality Gaps in Few-Shot Remote Sensing Scene Classification (arXiv, 2025-03-19)
- 79a66da7795b0f0a — MANet: Fine-Tuning Segment Anything Model for Multimodal Remote Sensing Semantic Segmentation (arXiv, 2024-10-15)
- c7f00e05411b1c8a — Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models (IEEE Robotics and Automation Letters, 2026-01-01)

## Hard falsifiers

- Native CROMA joint_encodings must be finite, aligned to the audited 15x15 grid, retained from one existing backbone forward, and consumed without an N-by-N or full-resolution dense attention tensor.
- The local kernel must be finite, nonnegative and receiver-normalized; JACK fails causally if optical-query, same-index native-J, shuffled-J or J-ablation controls are practically indistinguishable.
- Best 24-epoch seed-0 validation mIoU gain below +1.0pp closes JACK; +1.0 to below +2.0pp is marginal; controls unlock only at or above 51.7807879% mIoU.
- Any second CROMA forward, training-object mismatch, peak memory above 24GB, storage above 50GB, local real-data/weight transfer, or sealed-test access blocks the route.

## CAS verification

{}

## Resource budget

{
  "artifact_type": "researchpilot_independent_resource_data_review",
  "schema_version": "researchpilot.v18.resource_data.v1",
  "run_id": "prq4-v18-resource-data-20260831-r1",
  "role": "resource_data",
  "canonical_role": "resource_data",
  "attempt": 1,
  "status": "completed",
  "stage": "independent_resource_data_review",
  "created_at_utc": "2026-08-31T08:15:39.861928Z",
  "project_root": "F:/PRQ4",
  "plan_only": true,
  "experiments_started": false,
  "real_data_accessed": false,
  "real_pixels_read": false,
  "weights_downloaded_or_read": false,
  "gpu_used": false,
  "sealed_test_accessed": false,
  "test_seal_status": "sealed",
  "independence_contract": {
    "reviewed_independently_of": [
      "prq4-v18-architect-a-20260831-r1",
      "prq4-v18-architect-b-20260831-r1"
    ],
    "scope": "resource, tensor-shape, forward-count, parity and cloud/data feasibility only",
    "not_a_scientific_result": true,
    "not_a_novelty_clearance": true,
    "not_a_training_authorization": true
  },
  "frozen_project_context": {
    "current_project_gate": "INNOVATION_SCREENING_SEED0",
    "current_gate_status": "BLOCKED",
    "current_bound_route": "R-EO-MCOF-V17-01",
    "current_bound_candidate": "MCOF-01",
    "v17_route_status": "closed_below_meaningful_effect_gate",
    "v18_status": "plan_only_not_integrated_into_current_protocol_or_code",
    "task": "paired Sentinel-2 optical and Sentinel-1 SAR 11-class dense land-cover segmentation",
    "dataset_id": "sen12ts_worldcover_3region_1200",
    "model": "audited CROMA radar-optical EO vision foundation model",
    "baseline_best_validation_miou_percent": 49.78078791964122,
    "baseline_epoch24_validation_miou_percent": 49.76613907756286,
    "promotion_threshold_miou_percent": 51.78078791964122,
    "candidate_designs_reviewed": [
      "JACK-01",
      "CMEC-01",
      "RCPF-02"
    ]
  },
  "direct_interface_facts": {
    "input_and_token_contract": {
      "optical_input": "[B,12,120,120] float32",
      "sar_input": "[B,2,120,120] float32",
      "stage_optical": "[B,N,768]",
      "stage_sar": "[B,N,768]",
      "stage_sar_depth_group": "[B,N,4,768]",
      "token_count": 225,
      "token_grid": "15x15",
      "formal_output": "[B,11,120,120]",
      "evidence": [
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:678-715",
        "F:/PRQ4/02_experiment/code/configs/model/geotoken3path.yaml:5-27",
        "F:/PRQ4/02_experiment/code/configs/model/geotoken3path.yaml:70-76",
        "F:/PRQ4/02_experiment/code/configs/benchmarks/sen12ts_worldcover.yaml:25-30"
      ]
    },
    "native_joint_availability_and_drop": {
      "native_croma_output": "The official-style CROMA forward returns SAR_encodings, SAR_GAP, optical_encodings, optical_GAP, joint_encodings and joint_GAP in one call.",
      "one_forward_evidence": "CromaDepthTapAdapter.forward invokes self.backbone(...) exactly once and does not invoke it again.",
      "current_drop": "The return value of that call is ignored; the adapter returns only optical, sar and sar_depth_group. CromaBackboneBridge then requires exactly those three keys and CromaGeoTokenSegmentation unpacks exactly three values.",
      "jack_conclusion": "JACK can reuse native joint_encodings from the existing single CROMA forward, but the current bridge contract definitely discards it. A V18 bridge contract extension is required; a second CROMA forward is neither needed nor permitted.",
      "evidence": [
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_random.py:103-122",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:587-600",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:636-644",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:678-715",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:725-743"
      ]
    },
    "current_fusion_surface": {
      "baseline_fusion": "OpticalSarTokenModel first applies optical_stem and sar_stem to every configured stage; GeoToken3PathFusion then uses sar_exchange, output_norm and the selected local window. always_fuse is explicitly same-index/window_size=1.",
      "final_classifier": "The final fused [B,225,768] carrier is passed through the shared linear classifier and bilinearly interpolated to the requested output size.",
      "evidence": [
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:398-437",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:1430-1509",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:1662-1664",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:1847-1862",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:2051-2067",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:2151-2169"
      ]
    }
  },
  "protocol_and_resource_anchor": {
    "compute": {
      "gpu": "1x NVIDIA GeForce RTX 3090, 24GB",
      "micro_batch": 16,
      "effective_batch": 32,
      "gradient_accumulation": 2,
      "amp": true,
      "input_resolution": [
        120,
        120
      ],
      "token_grid": [
        15,
        15
      ],
      "formal_epochs": 24,
      "optimizer": "AdamW",
      "learning_rate": 0.0001,
      "scheduler": "cosine_with_warmup",
      "workers": 4,
      "pin_memory": true,
      "persistent_workers": true,
      "prefetch_factor": 2,
      "evidence": [
        "F:/PRQ4/02_experiment/protocol/experiment_protocol.yaml:13-25",
        "F:/PRQ4/02_experiment/protocol/experiment_protocol.yaml:116-125",
        "F:/PRQ4/02_experiment/protocol/experiment_protocol.yaml:147-179"
      ]
    },
    "measured_reference_boundary": {
      "reference": "V17 MCOF synthetic hardware preflight, not a V18 measurement",
      "gpu_name": "NVIDIA GeForce RTX 3090",
      "mcof_incremental_peak_mib": 219.4951171875,
      "mcof_total_fixture_peak_mib": 341.24560546875,
      "data_wait_ratio": 0.0,
      "interpretation": "This confirms the existing token-level code has substantial resource headroom, but it is not evidence for any V18 candidate and does not replace a V18 synthetic preflight.",
      "evidence": "F:/PRQ4/02_experiment/reports/v17_mcof_hardware_data_pipeline_adaptation_20260831_r2.json:36-71"
    },
    "data_and_seal": {
      "records": 1020,
      "train_records": 840,
      "validation_records": 180,
      "sealed_test_parent_count": 180,
      "sealed_test_objects_downloaded": 0,
      "active_storage_bytes": 7091439264,
      "hard_stop_bytes": 45000000000,
      "absolute_ceiling_bytes": 50000000000,
      "cloud_only": true,
      "new_data_or_extension_required": false,
      "test_seal": "sealed",
      "evidence": [
        "F:/PRQ4/02_experiment/reports/v17_mcof_dataset_audit_20260831_r1.json:19-45",
        "F:/PRQ4/02_experiment/protocol/experiment_protocol.yaml:21-25",
        "F:/PRQ4/02_experiment/protocol/experiment_protocol.yaml:50-72"
      ]
    },
    "parity_policy": {
      "required": true,
      "same_detector_training_object": true,
      "mechanism_inside_detector": true,
      "external_trainable_component_forbidden": true,
      "common_initialization_split_preprocessing_augmentation_batch_optimizer_scheduler_evaluator_seed": true,
      "candidate_modules_must_be_allocated_on_matched_surface": "For a V18 implementation, allocate the candidate mechanism in the shared model/factory surface and activate only the selected branch; do not use a separately trained wrapper.",
      "evidence": [
        "F:/PRQ4/02_experiment/reports/v17_mcof_training_object_parity_declaration_20260831_r2.json:11-39",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:905-931"
      ]
    }
  },
  "candidate_reviews": [
    {
      "candidate_id": "JACK-01",
      "route_id": "R-EO-JACK-V18-01",
      "name": "Joint-Anchor Kernel Rebinding",
      "design_source": "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:252-388",
      "final_verdict": "CONDITIONAL",
      "resource_verdict": "PASS_AT_DESIGN_LEVEL_PENDING_HARD_CONTRACT",
      "interface_fit": {
        "placement": "Final configured token stage immediately before the existing optical receiver/classifier; reuse the current sar_exchange/value path.",
        "inputs": {
          "native_joint_J": "[B,225,768]",
          "sar_values": "[B,225,768]",
          "local_sar_neighborhood": "[B,225,9,768] with a 3x3 bounded neighborhood",
          "query": "[B,225,r], r=32 or 64",
          "kernel_weights": "[B,225,9]",
          "readout": "[B,225,768]"
        },
        "output": "[B,225,768] fused carrier into the unchanged classifier/interpolation path",
        "current_code_change_needed": "Retain the mapping returned by the one existing CROMA call, expose native_joint in a bridge side channel or an explicit fourth mapping, and pass it only to the final JACK stage. The current three-key/three-return bridge cannot accept J without this contract change.",
        "second_croma_forward": {
          "required": false,
          "allowed": false,
          "reason": "joint_encodings is already returned by the same official-style forward call; a second call would duplicate the VFM and violate the intended resource/parity contract."
        }
      },
      "parameter_estimate": {
        "assumptions": [
          "Q and K are Linear(768,r) with bias.",
          "T_s reuses the already present final-stage sar_exchange/value projection; no duplicate dense 768-to-768 projection is added.",
          "LayerNorm affine parameters and one temperature scalar are counted when present."
        ],
        "qk_parameters": {
          "rank32": 49216,
          "rank64": 98432
        },
        "optional_low_rank_value_parameters": {
          "rank32": 49952,
          "rank64": 99136
        },
        "estimated_added_total": {
          "reuse_existing_value_rank32": "approximately 0.051M including LN/tau",
          "reuse_existing_value_rank64": "approximately 0.100M including LN/tau",
          "low_rank_value_rank32": "approximately 0.101M including LN/tau",
          "low_rank_value_rank64": "approximately 0.199M including LN/tau",
          "new_full_dense_value_warning": "A new full Linear(768,768) adds 590592 parameters and pushes the route above the declared <=0.30M target; it must not be introduced."
        }
      },
      "flops_estimate_per_microbatch": {
        "scope": "candidate-side final-stage work, excluding the already required CROMA forward and counting one multiply-add as two FLOPs",
        "qk_projection": {
          "rank32_mac": 176947200,
          "rank64_mac": 353894400
        },
        "local_kernel_distance_and_softmax": "approximately 0.003-0.010 GFLOPs for r=32-64, excluding framework overhead",
        "weighted_value_readout_mac": 24883200,
        "estimated_total_with_reused_value": {
          "rank32": "approximately 0.40-0.45 GFLOPs",
          "rank64": "approximately 0.76-0.82 GFLOPs"
        },
        "estimated_total_with_low_rank_value": {
          "rank32": "approximately 0.75-0.82 GFLOPs",
          "rank64": "approximately 1.45-1.50 GFLOPs"
        },
        "resource_interpretation": "Small relative to a CROMA backbone and bounded by 9 local values per receiver; no N-squared or native-resolution dense attention is needed."
      },
      "activation_memory_estimate": {
        "elements_or_shapes": {
          "J": "[16,225,768] = 2764800 elements",
          "q": "[16,225,r]",
          "local_k": "[16,225,9,r]",
          "local_v": "[16,225,9,768] = 24883200 elements",
          "weights": "[16,225,9]"
        },
        "raw_storage": {
          "J_fp16_mib": 5.2734375,
          "local_v_fp16_mib": 47.4609375,
          "local_v_fp32_mib": 94.921875,
          "local_k_fp16_mib_rank64": 3.955078125,
          "q_fp16_mib_rank64": 0.439453125
        },
        "incremental_peak_target": "below 0.8GB with gathered local tensors and normal autograd retention; this is an estimate, not a measured V18 result",
        "memory_contract": "Do not materialize [B,225,225,r], [B,120,120,768] attention, or a dense per-pixel classifier tensor. Retain J only for the final stage and avoid copying unused stage activations."
      },
      "amp_batch_and_24_epoch_assessment": {
        "microbatch16_effective32_amp": "CONDITIONAL_PASS",
        "why": "The local kernel is bounded and the current V17 hardware contract has 4-worker/AMP/microbatch16 headroom; the main unmeasured cost is autograd retention through native J and neighborhood gather.",
        "24_epoch_learnability": "PLAUSIBLE_BUT_NOT_PROVEN",
        "required_initialization": "The direct formula F=O+LayerNorm(local readout) is not automatically the same as the current same-index baseline. The hard contract must specify a center-preserving/zero-start blend or an exactly matched value initialization; otherwise the first update may be dominated by a representation discontinuity or boundary oversmoothing.",
        "frozen_cross_encoder_note": "Under tap_connected policy the cross_encoder is not trainable; J is a useful frozen condition/query, while Q/K/value adapter parameters must remain gradient-reachable."
      },
      "training_object_parity": {
        "status": "CONDITIONAL_PENDING_V18_CONTRACT",
        "must_hold": [
          "same CROMA checkpoint and input normalization",
          "same train/validation split and sealed test",
          "same optical/SAR stems, classifier, CE+Lovasz, AdamW, schedule, batch and seed",
          "JACK parameters inside the same detector graph, with no external router or second backbone",
          "same baseline and candidate trainability mask outside the single JACK operation"
        ]
      },
      "cloud_data_sealed_test": {
        "status": "PASS_IF_IMPLEMENTED_AS_SPECIFIED",
        "data_change": "none",
        "weights_change": "none; reuse the already audited cloud-only CROMA checkpoint",
        "storage_change": "no dataset/cache/extension required",
        "sealed_test": "remains sealed"
      },
      "engineering_risks": [
        "The bridge currently drops native joint_encodings and enforces an exact three-key output contract.",
        "The native J grid must be explicitly checked as [B,225,768] and finite; do not infer shape from a hook tap.",
        "A new full value projection would violate the stated parameter budget; reuse sar_exchange or use a low-rank value map.",
        "A softmax kernel may become center-collapsed or nearly uniform, and a positive local readout may oversmooth boundaries or follow SAR speckle.",
        "Using a second CROMA forward would inflate latency/memory and break training-object parity."
      ],
      "hard_contract_requirements": [
        "one and only one CROMA forward call",
        "native J finite and shape [B,225,768] on the same token grid",
        "3x3 kernel weights finite, nonnegative and normalized per receiver",
        "no dense N-by-N or full-resolution attention tensor",
        "baseline/candidate graph and trainability parity",
        "center-preserving identity or an explicitly matched baseline initialization",
        "synthetic proof of nonuniform J-conditioned readout, gradients and memory"
      ],
      "direct_evidence": [
        "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:261-278",
        "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:315-370",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_random.py:103-122",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:587-644",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/croma_bridge.py:678-715"
      ]
    },
    {
      "candidate_id": "CMEC-01",
      "route_id": "R-EO-CMEC-V18-01",
      "name": "Syndrome-Constrained Cross-Modal Error-Correcting Fusion",
      "design_source": "F:/PRQ4/01_literature/agents/v18_architect_a_20260831.json:129-250",
      "final_verdict": "REJECT",
      "resource_verdict": "PASS_RESOURCE_ONLY_BUT_NOT_A_DISTINCT_OPERATION_AS_SPECIFIED",
      "interface_fit": {
        "placement": "Existing paired optical/SAR token interface, preferably the final stage before the shared classifier; no new CROMA forward is required.",
        "main_shapes": {
          "optical_carrier_o": "[B,225,32]",
          "sar_carrier_s": "[B,225,32]",
          "stacked_x": "[B,225,64]",
          "parity_matrix_H": "[8,64]",
          "syndrome_e": "[B,225,8]",
          "corrected_x": "[B,225,64]",
          "decoded_delta": "[B,225,768]",
          "fused_output": "[B,225,768]"
        },
        "second_croma_forward": {
          "required": false,
          "allowed": false,
          "reason": "CMEC uses the already available stage taps and the existing stems/fusion surface."
        }
      },
      "parameter_estimate": {
        "assumptions": "r=32, q=8; P_o and P_s are Linear(768,32), Q is Linear(64,768), H has 8x64 free entries plus an orthonormalization parameterization if used, and alpha is one scalar.",
        "P_o_P_s": 49216,
        "decoder_Q": 49920,
        "H": 512,
        "alpha": 1,
        "nominal_total_without_extra_orthogonal_state": 99649,
        "declared_candidate_range": "approximately 0.10M-0.20M",
        "resource_conclusion": "Within the 3090 parameter budget if no duplicate 768-to-768 value/decoder layer is added."
      },
      "flops_estimate_per_microbatch": {
        "scope": "candidate-side final-stage work, one multiply-add counted as two FLOPs",
        "P_o_P_s_mac": 176947200,
        "H_x_and_Ht_e_mac": 7372800,
        "decoder_mac": 176947200,
        "estimated_total": "approximately 0.72-0.80 GFLOPs, plus negligible qxq solve/orthonormalization; a duplicate full decoder would increase this and is unnecessary"
      },
      "activation_memory_estimate": {
        "raw_storage": {
          "compact_x_fp16_mib": 0.439453125,
          "syndrome_fp16_mib": 0.054931640625,
          "corrected_x_fp16_mib": 0.439453125,
          "decoded_delta_fp16_mib": 5.2734375,
          "decoded_delta_fp32_mib": 10.546875
        },
        "incremental_peak_target": "below 0.3GB with autograd retention; design estimate only",
        "memory_contract": "No token-token or full-resolution tensor is required."
      },
      "foldability_audit": {
        "status": "FAIL_AS_A_NONLINEAR_OR_TOKEN_ADAPTIVE_MECHANISM",
        "finding": "With fixed learned global P_o, P_s, H, Q and scalar alpha, x=[P_o O;P_s S], P_H=I-H^T(HH^T+epsilon I)^(-1)H and Delta=Q(P_H-I)x are all linear and token-independent. The entire branch can be folded into two fixed linear maps from O and S (and combined with the baseline fusion map) at inference.",
        "implication": "The specified syndrome is not a per-token adaptive consistency operation. A parameter-matched dense bottleneck/no-syndrome control can represent the same linear map up to numerical/regularization details, so CMEC cannot be retained as the proposed distinct structural route without changing the mechanism definition.",
        "evidence": [
          "F:/PRQ4/01_literature/agents/v18_architect_a_20260831.json:140-149",
          "F:/PRQ4/01_literature/agents/v18_architect_a_20260831.json:201-207"
        ]
      },
      "amp_batch_and_24_epoch_assessment": {
        "microbatch16_effective32_amp": "CONDITIONAL_PASS",
        "amp_risk": "The q=8 solve/orthonormalization should run in FP32 or use an explicitly row-orthonormal Householder parameterization; relying on fp16 linalg.solve is not acceptable.",
        "24_epoch_learnability": "CONDITIONAL",
        "zero_start_issue": "At exact alpha=0 the output is identity, but gradients of P_o/P_s/H/Q are multiplied by alpha and are initially zero; only alpha receives a first-step signal. This is learnable in principle after alpha moves, but it creates a delayed/weak start that must be measured and cannot be treated as evidence of useful code learning.",
        "parity_issue": "The candidate must be allocated on the same model surface for baseline and candidate. A candidate-only alpha/module would violate the existing parity pattern."
      },
      "training_object_parity": {
        "status": "CONDITIONAL_PENDING_V18_CONTRACT",
        "requirements": [
          "same CROMA/tap/stem/classifier graph and trainability mask",
          "CMEC branch internal to the detector and matched in the baseline state-dict surface",
          "no objective, optimizer, scheduler, sampler or data change",
          "no labels/boundaries/test-derived H or precision"
        ]
      },
      "cloud_data_sealed_test": {
        "status": "PASS_RESOURCE_SCOPE",
        "data_change": "none",
        "storage_change": "none",
        "test_seal": "sealed"
      },
      "engineering_risks": [
        "The linear foldability makes the advertised parity/syndrome mechanism indistinguishable from a static bottleneck unless the definition is redesigned.",
        "The alpha-zero initialization delays gradients into all branch maps.",
        "FP16 orthonormalization/solve can be unstable or unsupported; FP32 casts may reduce the claimed simplicity.",
        "A fixed global H cannot encode token-specific cross-modal consistency, so the proposed causal telemetry may be non-diagnostic."
      ],
      "hard_contract_requirements_if_retained_only_as_control": [
        "prove exact alpha=0 baseline logit parity",
        "prove finite row-orthonormal H and projection idempotence",
        "run a parameter-matched dense no-syndrome control",
        "state explicitly that the branch is a static linear capacity control, not the unique innovation"
      ],
      "direct_evidence": [
        "F:/PRQ4/01_literature/agents/v18_architect_a_20260831.json:131-149",
        "F:/PRQ4/01_literature/agents/v18_architect_a_20260831.json:201-243",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:1847-1862",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:2051-2064"
      ]
    },
    {
      "candidate_id": "RCPF-02",
      "route_id": "R-EO-RCPF-V18-02",
      "name": "Robust Consensus Proximal Fusion",
      "design_source": "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:391-528",
      "final_verdict": "CONDITIONAL",
      "resource_verdict": "CONDITIONAL_PASS_PENDING_SOLVER_PREFLIGHT",
      "interface_fit": {
        "placement": "Final 15x15 optical/SAR fusion interface before the shared linear classifier; reuse the existing sar_exchange as T_s/value projection and output_norm where possible.",
        "main_shapes": {
          "O_and_S_prime": "[B,225,768] each",
          "baseline_anchor_B": "[B,225,768]",
          "precision": "[B,225,2] (or low-rank intermediates [B,225,16])",
          "four_neighbor_edges": "420 undirected, up to 840 directed, or 900 padded 4-neighbor slots on a 15x15 grid",
          "edge_weight": "[B,E,1]",
          "primal_F": "[B,225,768]",
          "dual_edge_variable": "[B,E,768]",
          "unrolled_steps": "K<=3"
        },
        "second_croma_forward": {
          "required": false,
          "allowed": false,
          "reason": "RCPF operates on the existing stage taps/stems and does not need any additional CROMA computation."
        },
        "baseline_alignment_note": "The current baseline is optical plus output_norm(sar_exchange(sar)) at the same index. A new independent T_s or an alternate normalization would change capacity/protocol; reuse the existing projection or declare the extra map as part of the single matched mechanism."
      },
      "parameter_estimate": {
        "assumptions": "rank-16 low-rank precision/edge heads; no duplicate full 768-to-768 value projection; fixed solver has no trainable optimizer state.",
        "representative_low_rank_precision_heads": "approximately 49218 parameters for two 1536-to-16-to-1 heads",
        "representative_edge_projections": "approximately 24608 parameters for two 768-to-16 projections",
        "representative_total": "approximately 0.075M before optional biases/scales",
        "declared_target": "<=0.25M",
        "full_dense_value_warning": "A new Linear(768,768) would add 590592 parameters and is incompatible with the intended bounded design; reuse sar_exchange."
      },
      "flops_estimate_per_microbatch": {
        "scope": "order estimate for K=3, B=16, N=225, D=768, E<=900; existing CROMA excluded",
        "precision_and_edge_heads": "approximately 0.15-0.35 GFLOPs depending on head topology",
        "edge_differences_and_dual_divergence": "approximately 0.20-0.45 GFLOPs for three streamed 4-neighbor iterations",
        "Huber_pointwise_prox_and_anchor_terms": "approximately 0.10-0.25 GFLOPs",
        "estimated_total": "approximately 0.5-1.0 GFLOPs for K=3, excluding any reused sar_exchange; implementation measurement is required",
        "K_assessment": "K<=3 is a rational bounded budget for a 15x15 grid, but three unrolled steps are not a convergence proof. The paper/plan must call it a fixed unrolled operator and lock its step-size stability condition."
      },
      "activation_memory_estimate": {
        "raw_storage": {
          "primal_F_fp16_mib": 5.2734375,
          "primal_F_fp32_mib": 10.546875,
          "dual_edges_fp16_mib_E840": 19.6875,
          "dual_edges_fp32_mib_E840": 39.375,
          "dual_edges_fp16_mib_E900": 21.09375,
          "dual_edges_fp32_mib_E900": 42.1875
        },
        "unrolled_storage": "Three primal/dual states are roughly 75-80MiB in fp16 at E<=900 before autograd metadata and other intermediates.",
        "incremental_peak_target": "below 1.0GB if edge differences are streamed or checkpointed; a full D-dimensional edge tensor retained for every step in FP32 can approach this bound",
        "memory_contract": "Never construct a full N-by-N graph or a 120x120x768 TV tensor. Keep the graph at 15x15 and use four-neighbor roll/index operations or bounded edge buffers."
      },
      "amp_batch_and_24_epoch_assessment": {
        "microbatch16_effective32_amp": "CONDITIONAL_PASS",
        "amp_requirements": [
          "compute Huber/prox arithmetic, norm reductions and edge exponent/precision normalization in FP32 or with explicit safe scaling",
          "avoid fp16 underflow that makes all edge weights zero or uniform",
          "measure peak memory with backward through all K steps, not only forward"
        ],
        "24_epoch_learnability": "CONDITIONAL",
        "learnability_risk": "F^(0)=B gives a reasonable baseline anchor, but a nonzero fixed solver step changes the output at initialization. If the step is zero-started, the same delayed-gradient problem as a residual branch appears; if it is not, exact baseline parity is lost. The choice must be locked before training.",
        "stability_requirement": "For the chosen primal-dual operator, enforce the declared tau/sigma/Lipschitz condition and record residual monotonicity; do not claim convergence solely because K=3."
      },
      "training_object_parity": {
        "status": "CONDITIONAL_PENDING_V18_CONTRACT",
        "requirements": [
          "same CROMA checkpoint, taps, stems, classifier, loss, optimizer, schedule, split, batch, AMP and seed",
          "RCPF solver internal to the same detector graph; no external optimization loop or auxiliary loss",
          "precision and edge weights use only paired token features, never labels, boundaries, teachers or test data",
          "matched baseline initialization and explicit identity/L2 control"
        ]
      },
      "cloud_data_sealed_test": {
        "status": "PASS_RESOURCE_SCOPE",
        "data_change": "none",
        "storage_change": "none",
        "test_seal": "sealed"
      },
      "engineering_risks": [
        "Full D-dimensional TV carries an edge vector of length 768 per graph edge; naive autograd storage may approach the 1GB incremental bound.",
        "Huber/TV/Chambolle-Pock primitives are standard and the resource-efficient implementation can still be a solverized regularizer; this report does not clear novelty.",
        "The baseline anchor may dominate or the solver may collapse to identity, while aggressive edge coupling may oversmooth thin boundaries.",
        "Three iterations can be numerically stable but still too shallow to realize the claimed proximal solution; residual decrease must be measured.",
        "Using a new full dense T_s silently changes parameter count and violates the intended comparison."
      ],
      "hard_contract_requirements": [
        "15x15-only four-neighbor graph with no dense adjacency",
        "finite energy, residual and gradient under AMP and backward",
        "fixed-step stability condition and K<=3 locked before training",
        "edge memory measured at microbatch16 with autograd",
        "baseline identity or explicitly matched L2 fusion contract",
        "precision/edge weights are label-free and test-free"
      ],
      "direct_evidence": [
        "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:400-418",
        "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:455-475",
        "F:/PRQ4/01_literature/agents/v18_architect_b_20260831.json:504-517",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:398-437",
        "F:/PRQ4/02_experiment/code/src/geotoken3path/models/fusion.py:1430-1509"
      ]
    }
  ],
  "cross_candidate_summary": {
    "JACK-01": {
      "resource_read": "Best fit to the real interface: native J exists in the one existing forward but is currently dropped; a bounded 3x3 readout is small and local.",
      "blocking_condition": "V18 bridge extension, exact J shape/finite contract, identity/baseline parity choice and synthetic memory/liveness preflight"
    },
    "CMEC-01": {
      "resource_read": "Fits the 3090 numerically, but the stated fixed linear algebra collapses into a static linear adapter and should not be retained as a distinct route.",
      "blocking_condition": "Reject as primary structural candidate; only retain as an explicitly named matched capacity control if the plan owner redesigns the mechanism"
    },
    "RCPF-02": {
      "resource_read": "15x15 and K<=3 are resource-rational; the full D-dimensional edge variable makes memory and AMP stability conditional.",
      "blocking_condition": "Solver/edge-memory preflight and fixed-step/identity contract"
    }
  },
  "unique_resource_recommendation": {
    "candidate_id": "JACK-01",
    "route_id": "R-EO-JACK-V18-01",
    "recommendation": "Carry only JACK-01 forward as the unique V18 resource-priority route for hard-contract and synthetic-liveness work.",
    "why": [
      "It reuses a native CROMA joint output already computed in one forward instead of adding a second backbone pass.",
      "Its extra interaction is limited to 9 SAR values per receiver on the existing 225-token grid.",
      "With the existing sar_exchange/value path, added parameters are approximately 0.05-0.10M for Q/K rank 32-64 and the candidate-side activation target is below 0.8GB.",
      "It preserves the fixed dataset, cloud-only weight policy, 3090 microbatch16/effective32/AMP contract and sealed test without an extension download.",
      "Its main resource risk is concrete and auditable: the current bridge drops J, so the one-forward retention contract can be tested directly."
    ],
    "not_a_scientific_selection": true,
    "next_allowed_stage": "V18 Plan successor hard contract and synthetic liveness only; no real-data training, metrics, controls or sealed-test access from this report."
  },
  "overall_resource_decision": {
    "JACK-01": "CONDITIONAL",
    "CMEC-01": "REJECT",
    "RCPF-02": "CONDITIONAL",
    "unique_recommendation": "JACK-01",
    "scope_change": false,
    "task_dataset_model_loss_optimizer_epochs_trainability_data_budget_and_test_seal_unchanged": true
  },
  "evidence_boundary": {
    "verified_facts": [
      "The current CROMA-style forward exposes joint_encodings in one official-style return, while CromaDepthTapAdapter drops that return and the downstream bridge accepts only three mappings.",
      "The current formal interface is 12+2 channels, 120x120 input, 15x15/225 tokens, 768 token dimension, 11 classes and a shared linear classifier.",
      "The cloud data is already materialized under the existing audited 7,091,439,264-byte footprint, with no sealed-test objects downloaded; no V18 candidate requires new data or weights.",
      "The existing cloud hardware adaptation passed RTX 3090, microbatch16, effective32 and AMP for a bounded token-level reference, but this does not constitute a V18 measurement."
    ],
    "design_estimates": [
      "All V18 FLOPs and incremental memory numbers are analytic upper estimates under stated tensor/materialization assumptions, not measured candidate results.",
      "JACK and RCPF require a new hard-contract/synthetic preflight before any resource PASS can become an execution-ready PASS."
    ],
    "negative_finding": "CMEC-01 as currently specified is algebraically foldable into a fixed linear map; its resource feasibility does not rescue its distinct-mechanism status.",
    "prohibited_interpretations": [
      "Do not treat this review as a performance claim or novelty clearance.",
      "Do not update the V17 protocol or run a V18 candidate from this artifact alone.",
      "Do not download data/weights, read pixels, enable GPU, train, open controls or unseal test."
    ]
  },
  "evidence_ids": [
    "4bffba5ad5dc6ea8",
    "7f3e2082d0f31b0e",
    "3f7d680c1b3ebd15",
    "194d31371ff32cb8",
    "98ee47b14b96a7dd"
  ],
  "completion_note": "Independent V18 resource_data review completed. JACK-01 is the only recommended resource-priority route; CMEC-01 is rejected as a distinct fixed-linear operation; RCPF-02 remains conditional pending solver and edge-memory preflight. No code, data, weight, GPU, training, metric or sealed-test action was performed."
}
