"""R24 SkySense++ S2 ICE task-specific export.

The exported model is a physically compact deep copy of the audited
``SkySensePlusPlusViTMSLModel`` backbone plus its unchanged 1x1
segmentation head:

  - contract "a" (Full): no layer is eliminated; the export is a deep copy of
    the full backbone and executes the official full forward.
  - contract "b" (ICE):  the ``layers`` ModuleList is sliced to its first 12
    modules (layers 0..11), removing layers 12..23 (~49.9% of the backbone
    parameters); patch embed / pos embed / drop-after-pos / vocabulary token /
    vocabulary weight / mask token and the head are kept intact.

The retained submodule paths are unchanged, and ``forward`` executes the
official vendor forward on the truncated ModuleList (see the zero-drift
argument in ``models/skysensepp_executor``), so the export's arithmetic is
bitwise equal to the ICE prefix of the full model on the same weights -- an
empty-elimination plan: every retained module is executed.  State-dict keys of
the export are a strict subset of the full model's keys with identical values.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from ..models.skysensepp_executor import (
    SkySensePPExecutionError,
    SkySensePPPrefixExecutor,
)
from ..models.skysensepp_seg import SkySensePPSegmentationModel
from .skysensepp_plan import (
    SkySensePPExecutionPlan,
    compile_skysensepp_plan,
    validate_skysensepp_contract,
)


@dataclass(frozen=True)
class SkysenseppExportStats:
    """Reduction statistics relative to the full audited model."""

    full_parameter_count: int
    export_parameter_count: int
    removed_parameter_count: int
    removed_parameter_fraction: float
    removed_module_paths: tuple[str, ...]


class _BackboneSuffixRemovedError(SkySensePPExecutionError):
    """Internal marker: a removed layer escaped the exporter's fail-closed gate."""


def _slice_backbone_layers(backbone: nn.Module, keep: int) -> None:
    """Splice the backbone's ``layers`` ModuleList down to its first ``keep``."""
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise _BackboneSuffixRemovedError(
            "export backbone must expose its encoder stack as a layers ModuleList"
        )
    if keep < 1 or keep > len(layers):
        raise _BackboneSuffixRemovedError(
            f"cannot keep {keep} of {len(layers)} backbone layers"
        )
    retained = list(layers)[:keep]
    backbone.layers = nn.ModuleList(retained)
    # Removing parameters without removing children corrupts state_dict
    # bookkeeping (shape errors on strict reload).  Detach the removed modules
    # from the parameter tree instead.
    for module in layers[keep:]:
        for parameter in module.parameters():
            if parameter.data is not None:
                parameter.data = parameter.data.new_empty(0)


class SkysenseppExportedModel(nn.Module):
    """Physically compact SkySense++ segmentation model with an exact plan.

    The exported model owns its own deep copy of the backbone and head.  Its
    ``forward`` runs the official vendor forward over the retained modules and
    the segmentation head over the returned feature maps, producing the same
    output contract (``logits`` plus ``feature_maps``) as the full model.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module,
        plan: SkySensePPExecutionPlan,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch module")
        if not isinstance(head, nn.Module):
            raise TypeError("head must be a torch module")
        if not isinstance(plan, SkySensePPExecutionPlan):
            raise TypeError("plan must be a SkySensePPExecutionPlan")
        self.backbone = backbone
        self.head = head
        self.plan = plan

    def forward(
        self,
        pixel_values: Tensor,
        annotation: Tensor,
        *,
        max_layer: int | None = None,
    ) -> dict[str, Any]:
        """Run the official forward on the retained prefix and score it."""
        try:
            executor = SkySensePPPrefixExecutor(self.plan)
        except Exception as exc:  # pragma: no cover - construction is validated
            raise SkySensePPExecutionError("export plan is invalid") from exc
        result = executor.execute(
            self.backbone,
            pixel_values=pixel_values,
            annotation=annotation,
            max_layer=max_layer,
        )
        maps = result["feature_maps"]
        head_maps = list(maps)
        if self.plan.contract == "b" and len(maps) == 2:
            # Contract "b" consumes only the deepest executed map; feed the
            # head a full-width map tuple so its index contract is stable.
            head_maps = [maps[0], maps[1], maps[1], maps[1]]
        if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 4:
            raise SkySensePPExecutionError("pixel_values must be [B,C,H,W]")
        logits = self.head(
            head_maps,
            output_size=(int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
        )
        return {
            "logits": logits,
            "feature_maps": maps,
            "layer_indices": result["layer_indices"],
            "executed_layer_count": result["executed_layer_count"],
        }


def _clone_frozen_parameters(source: nn.Module, target: nn.Module) -> None:
    for name, parameter in source.named_parameters(recurse=True):
        if parameter.requires_grad:
            try:
                path, leaf = name.rsplit(".", 1)
                module = target.get_submodule(path)
            except (ValueError, AttributeError):
                module = None
            if module is not None:
                child = getattr(module, leaf, None)
                if isinstance(child, nn.Parameter):
                    setattr(
                        module,
                        leaf,
                        nn.Parameter(parameter.detach().clone(), requires_grad=True),
                    )
                elif isinstance(child, nn.Module):
                    # Nested parameter path under a module namespace: descend.
                    for sub_name, sub_parameter in child.named_parameters(recurse=True):
                        if sub_parameter is parameter:
                            nested_path, nested_leaf = sub_name.rsplit(".", 1)
                            nested = child.get_submodule(nested_path)
                            setattr(
                                nested,
                                nested_leaf,
                                nn.Parameter(parameter.detach().clone(), requires_grad=True),
                            )
                            break


def _copy_model_preserving_head_requires_grad(
    full_model: SkySensePPSegmentationModel,
) -> SkySensePPSegmentationModel:
    """Deep copy full -> export while keeping the frozen backbone frozen and
    re-marking the export head parameters as trainable like the source head."""
    copied = copy.deepcopy(full_model)
    copied.freeze_backbone()
    if isinstance(full_model.head, nn.Module):
        _clone_frozen_parameters(full_model.head, copied.head)
    return copied


def build_skysensepp_export_model(
    full_model: SkySensePPSegmentationModel,
    *,
    contract: str,
) -> tuple[SkysenseppExportedModel, SkysenseppExportStats]:
    """Build the physically compact export for one receiver contract.

    The full model's head contract must equal the requested export contract;
    a mismatch (exporting a contract "a" receiver from a contract "b" head,
    or vice versa) raises ``ValueError`` instead of silently producing an
    artifact whose head reads the wrong feature maps.  The export is an
    independent deep copy (shared state with ``full_model`` is never
    modified), with the backbone ModuleList sliced to ``plan.max_layer + 1``
    modules, every other submodule and the head kept intact.  The returned
    statistics report the parameter reduction and the exact removed submodule
    paths relative to the audited full model.
    """
    validate_skysensepp_contract(contract)
    if not isinstance(full_model, SkySensePPSegmentationModel):
        raise TypeError("full_model must be a SkySensePPSegmentationModel")
    if str(getattr(full_model, "contract", "")) != str(contract):
        raise ValueError(
            "export contract does not match the full model head contract: "
            f"requested {contract!r}, full model head is "
            f"{getattr(full_model, 'contract', None)!r}"
        )
    plan = compile_skysensepp_plan(contract)

    copied = _copy_model_preserving_head_requires_grad(full_model)
    keep = plan.max_layer + 1
    layers = getattr(copied.backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise _BackboneSuffixRemovedError(
            "full backbone must expose its encoder stack as a layers ModuleList"
        )
    if len(layers) < keep:
        raise _BackboneSuffixRemovedError(
            f"full backbone has {len(layers)} layers but contract {contract!r} "
            f"needs {keep}"
        )
    _slice_backbone_layers(copied.backbone, keep)

    # Fail closed: physically removed layers must never survive in any shape.
    if len(copied.backbone.layers) != keep:
        raise _BackboneSuffixRemovedError(
            f"export backbone retained {len(copied.backbone.layers)} layers, expected {keep}"
        )

    removed_paths: tuple[str, ...]
    if contract == "b":
        removed_paths = tuple(
            f"backbone.layers.{index}" for index in range(keep, len(layers))
        )
    else:
        removed_paths = ()

    export_model = SkysenseppExportedModel(backbone=copied.backbone, head=copied.head, plan=plan)

    full_count = sum(parameter.numel() for parameter in full_model.parameters())
    export_count = sum(parameter.numel() for parameter in export_model.parameters())
    removed = full_count - export_count
    if removed < 0:
        raise _BackboneSuffixRemovedError(
            "export parameter count exceeds the full model; export is invalid"
        )
    stats = SkysenseppExportStats(
        full_parameter_count=full_count,
        export_parameter_count=export_count,
        removed_parameter_count=removed,
        removed_parameter_fraction=(removed / full_count) if full_count else 0.0,
        removed_module_paths=removed_paths,
    )
    return export_model, stats
