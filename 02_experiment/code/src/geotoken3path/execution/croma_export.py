# -*- coding: utf-8 -*-
"""Task-Specific Model Export (R21 queue 3).

Builds a physically compact backbone from an audited CROMA instance by
stripping every module that the verified always-fuse receiver never consumes:
  - S2 encoder layers 6-11 (only layers 0-5 feed the retained taps),
  - S1/S2 final norms (not consumed by the tap hook paths),
  - GAP heads (GAP_FFN_s1 / GAP_FFN_s2),
  - the joint cross-encoder (and its GAP mean, which is not a module).

The stripped backbone keeps the same submodule paths for everything retained,
so CromaDepthTapAdapter / CromaBackboneBridge / CromaGeoTokenSegmentation can
be reused unchanged. Its forward delegates to the certified ICE executor with
an empty-elimination plan (all retained modules are executed), so the executed
arithmetic is identical to ICE-Exact by construction and is re-certified
against Full on the whole validation split before any reduction numbers are
reported.

attn_bias stays a plain tensor attribute (never a buffer), exactly as in the
audited source, so state-dict key layout stays aligned with the full model.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .croma_executor import InterfaceCertifiedCromaExecutor
from .croma_plan import CromaExecutionPlan


@dataclass(frozen=True)
class ExportBackboneStats:
    """Reduction statistics relative to the full audited backbone."""

    full_parameter_count: int
    export_parameter_count: int
    removed_parameter_count: int
    removed_parameter_fraction: float
    removed_module_paths: tuple[str, ...]


def _stripped_backbone_copy(audited_backbone: nn.Module) -> nn.Module:
    """Deep-copy the audited backbone and physically remove dead modules."""
    backbone = copy.deepcopy(audited_backbone)
    # Clear any forward hooks the deep copy may carry from the source (the
    # source backbone may already be wrapped by a tap adapter whose hooks
    # would otherwise become ghost hooks bound to the source's capture dict).
    for module in backbone.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    # S2 encoder: keep layers 0..5 (ModuleList of 6).
    s2_layers = backbone.s2_encoder.transformer.layers
    backbone.s2_encoder.transformer.layers = nn.ModuleList(list(s2_layers)[:6])
    # Remove final norms (not consumed by retained tap paths).
    if hasattr(backbone.s1_encoder.transformer, "norm_out"):
        del backbone.s1_encoder.transformer.norm_out
    if hasattr(backbone.s2_encoder.transformer, "norm_out"):
        del backbone.s2_encoder.transformer.norm_out
    # Remove GAP heads and the joint cross-encoder.
    for name in ("GAP_FFN_s1", "GAP_FFN_s2", "cross_encoder"):
        if hasattr(backbone, name):
            delattr(backbone, name)
    return backbone


class CromaExportedBackbone(nn.Module):
    """Physically compact backbone that executes the ICE-Exact retained graph."""

    def __init__(self, audited_backbone: nn.Module) -> None:
        super().__init__()
        stripped = _stripped_backbone_copy(audited_backbone)
        # Move the stripped submodules onto this module, keeping paths identical.
        self.s1_encoder = stripped.s1_encoder
        self.s2_encoder = stripped.s2_encoder
        self.attn_bias = stripped.attn_bias
        # Empty-elimination plan: every retained module executes, mirroring the
        # ICE-Exact prefix with the same s1/s2 last layers (5/5).
        plan = CromaExecutionPlan(
            ablation_tier="exact",
            required_taps=(),
            s1_last_layer=5,
            s2_last_layer=5,
            require_s1_final_norm=False,
            require_s2_final_norm=False,
            require_joint_encoder=False,
            require_s1_gap=False,
            require_s2_gap=False,
            eliminated_nodes=(),
            plan_sha256="export-empty-elimination",
        )
        self._executor = InterfaceCertifiedCromaExecutor(plan)

    def forward(
        self,
        SAR_images: Tensor | None = None,
        optical_images: Tensor | None = None,
        *,
        imgs: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        """Official-style forward that executes only the retained prefix."""
        if imgs is not None:
            raise ValueError("CromaExportedBackbone supports only paired SAR/optical calls")
        return self._executor.execute(
            self, SAR_images=SAR_images, optical_images=optical_images
        )

    def retained_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_export_backbone(
    audited_backbone: nn.Module,
) -> tuple[CromaExportedBackbone, ExportBackboneStats]:
    """Build the exported backbone and report reduction stats vs the source."""
    full_count = sum(p.numel() for p in audited_backbone.parameters())
    export = CromaExportedBackbone(audited_backbone)
    export_count = export.retained_parameter_count()
    removed = full_count - export_count
    # Identify removed module paths by diffing submodule trees.
    def collect_paths(module: nn.Module, prefix: str = "") -> set[str]:
        paths = set()
        for name, child in module.named_children():
            child_path = f"{prefix}.{name}" if prefix else name
            paths.add(child_path)
            paths |= collect_paths(child, child_path)
        return paths

    full_paths = collect_paths(audited_backbone)
    export_paths = collect_paths(export)
    removed_paths = tuple(sorted(full_paths - export_paths))
    stats = ExportBackboneStats(
        full_parameter_count=full_count,
        export_parameter_count=export_count,
        removed_parameter_count=removed,
        removed_parameter_fraction=removed / full_count if full_count else 0.0,
        removed_module_paths=removed_paths,
    )
    return export, stats