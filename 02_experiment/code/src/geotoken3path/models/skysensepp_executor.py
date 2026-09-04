"""Zero-drift ICE prefix executor for the SkySense++ S2 backbone.

SkySense++ has no CROMA-style ``[attn, ffn]`` tap pair with a prefix FFN hook:
the official vendor forward
(``vendor/skysensepp_s2_full/modeling_skysensepp_vit_msl.py``,
``SkySensePlusPlusViTMSLModel.forward``) is itself an exact per-layer loop.  ICE
is therefore *prefix truncation of the official forward*, never a hand-written
re-execution of the loop.

Why the prefix is bitwise identical to a shorter official model (zero drift):

  * The official loop executes ``for i, layer in enumerate(self.layers)`` and
    collects ``x`` when ``i in self.out_indices``.  Everything before the loop
    (patch embed, annotation token, merge 0, positional embedding) and inside
    the loop body depends only on loop-local state plus module parameters --
    for the pinned R24 config (``merge_stage=4 >= any executed layer``,
    ``use_attn=False``, ``final_norm=False``, no hidden-state bookkeeping) the
    merge/attn/norm suffix branches never fire within the executed prefix.

  * An ICE run therefore executes the *unchanged official forward function* on
    a transient view of the backbone whose ``layers`` ModuleList is sliced to
    ``max_layer + 1`` entries and whose ``out_indices`` is the official grid
    restricted to layers ``<= max_layer``.  The slice reuses the exact child
    module objects (parameters are shared, not copied) and the view is restored
    in ``finally``, so state-dict keys and module identity are untouched.

  * Consequently an ICE prefix at ``max_layer`` is bitwise equal to the shared
    prefix of a full run of the same weights *by construction*, and running
    with ``max_layer = num_layers - 1`` is exactly the official full forward
    (the slice equals the original ModuleList).  The hard numerical gates in
    ``tests/unit/test_skysensepp_executor.py`` verify both properties on random
    weights.

The returned mapping mirrors the official tuple layout: ``feature_maps[k]``
corresponds to layer ``layer_indices[k]`` of the official grid, exactly like
the vendor's ``out_indices`` semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..execution.skysensepp_plan import (
    SkySensePPExecutionPlan,
    validate_skysensepp_contract,
)

# Official SkySense++ S2 grid.  Duplicated here (not imported from the plan
# module) so the executor stays importable in isolation; the constant is
# validated against the live backbone in every execute call.
_S2_OUT_GRID: tuple[int, ...] = (5, 11, 17, 23)


class SkySensePPExecutionError(RuntimeError):
    """Raised when an ICE prefix cannot be executed exactly as planned."""


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


class SkySensePPPrefixExecutor:
    """Execute the exact official SkySense++ forward truncated at a layer bound.

    ``execute`` runs the *vendor forward unchanged* on a transient truncated
    view of the backbone (see module docstring).  The default executed depth is
    ``plan.max_layer``; an explicit ``max_layer`` must stay between the deepest
    required output index and the backbone depth so a plan never silently drops
    a required feature map.
    """

    def __init__(self, plan: SkySensePPExecutionPlan) -> None:
        if not isinstance(plan, SkySensePPExecutionPlan):
            raise TypeError("plan must be a SkySensePPExecutionPlan")
        validate_skysensepp_contract(plan.contract)
        self.plan = plan

    def execute(
        self,
        backbone_model: nn.Module,
        pixel_values: Tensor,
        annotation: Tensor,
        *,
        max_layer: int | None = None,
    ) -> Mapping[str, Any]:
        """Run the official forward prefix and return its feature maps.

        Returns a mapping with ``feature_maps`` (tuple of
        ``[B, 1024, H/4, W/4]`` tensors in official grid order),
        ``layer_indices`` (official grid layer of each returned map) and
        ``executed_layer_count``.  ``backbone_model`` is left exactly as
        found, even on failure.
        """
        plan = self.plan
        if not isinstance(backbone_model, nn.Module):
            raise SkySensePPExecutionError(
                "executor requires the audited SkySensePlusPlusViTMSLModel as backbone_model"
            )
        layers = getattr(backbone_model, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise SkySensePPExecutionError(
                "executor requires the audited SkySensePlusPlusViTMSLModel form "
                "(nn.Module with a layers ModuleList)"
            )
        depth = len(layers)
        if depth < 1:
            raise SkySensePPExecutionError("audited backbone exposes an empty layers ModuleList")

        if max_layer is None:
            max_layer = plan.max_layer
        if _is_bool(max_layer) or not isinstance(max_layer, int):
            raise SkySensePPExecutionError("max_layer must be an integer or None")
        deepest_required = max(plan.required_output_indices)
        if max_layer < deepest_required:
            raise SkySensePPExecutionError(
                f"plan contract {plan.contract!r} requires layer {deepest_required}, "
                f"but max_layer={max_layer} would truncate before it"
            )
        if max_layer > plan.max_layer:
            raise SkySensePPExecutionError(
                f"max_layer={max_layer} exceeds the plan's executed depth "
                f"(contract {plan.contract!r} max_layer={plan.max_layer})"
            )
        if max_layer >= depth:
            raise SkySensePPExecutionError(
                f"max_layer={max_layer} is beyond the audited backbone depth {depth}"
            )
        grid = self._grid_for_max_layer(backbone_model, max_layer=max_layer, depth=depth)
        self._validate_plan_shape(plan, grid, depth)

        pixel_values = torch.as_tensor(pixel_values)
        annotation = torch.as_tensor(annotation)
        if pixel_values.ndim != 4 or annotation.ndim != 3:
            raise SkySensePPExecutionError(
                "pixel_values must be [B,C,H,W] and annotation [B,H,W]"
            )
        if not pixel_values.dtype.is_floating_point:
            raise SkySensePPExecutionError(
                "pixel_values must be a floating-point tensor"
            )
        if annotation.dtype not in (torch.int32, torch.int64):
            raise SkySensePPExecutionError(
                "annotation must be an int32/int64 class-id tensor "
                "(the vendor vocabulary index_select requires it)"
            )

        return self._run_prefix(backbone_model, pixel_values, annotation, max_layer=max_layer, grid=grid)

    @staticmethod
    def _grid_for_max_layer(
        backbone: nn.Module,
        *,
        max_layer: int,
        depth: int,
    ) -> tuple[int, ...]:
        """Derive the official-grid slot layers reachable at ``max_layer``.

        The live backbone's ``out_indices`` (its official config grid, already
        normalized to nonnegative indices by the vendor constructor) is the
        source of truth; the module-level constant only guards audited
        backbones that carry no attribute.
        """
        live = getattr(backbone, "out_indices", None)
        raw: tuple[int, ...]
        if live is None:
            raw = _S2_OUT_GRID
        elif isinstance(live, (tuple, list)) and all(
            isinstance(index, int) and not isinstance(index, bool) for index in live
        ):
            raw = tuple(int(index) for index in live)
        else:
            raise SkySensePPExecutionError(
                "audited backbone out_indices is not a sequence of layer indices"
            )
        if not raw:
            raise SkySensePPExecutionError("audited backbone exposes an empty out_indices grid")
        grid = tuple(sorted(index for index in raw if 0 <= index <= max_layer))
        if not grid:
            raise SkySensePPExecutionError(
                f"no official grid layer is reachable at max_layer={max_layer} "
                f"(grid {raw}, depth {depth})"
            )
        return grid

    @staticmethod
    def _validate_plan_shape(
        plan: SkySensePPExecutionPlan,
        executed_grid: tuple[int, ...],
        depth: int,
    ) -> None:
        """Fail closed when the live backbone disagrees with the compiled plan."""
        required = tuple(plan.required_output_indices)
        if required[-1] not in executed_grid:
            raise SkySensePPExecutionError(
                f"plan contract {plan.contract!r} requires layer {required[-1]} "
                f"but the executed grid {executed_grid} cannot provide it"
            )
        if plan.executed_layer_count != max(required) + 1:
            raise SkySensePPExecutionError(
                "plan executed_layer_count disagrees with its required outputs"
            )
        if plan.max_layer + 1 > depth:
            raise SkySensePPExecutionError(
                f"plan contract {plan.contract!r} needs {plan.max_layer + 1} layers "
                f"but the audited backbone has {depth}"
            )

    def _run_prefix(
        self,
        backbone: nn.Module,
        pixel_values: Tensor,
        annotation: Tensor,
        *,
        max_layer: int,
        grid: tuple[int, ...],
    ) -> dict[str, Any]:
        """Run the official forward on a transient truncated view, then restore."""
        layers = backbone.layers
        original_out_indices = getattr(backbone, "out_indices", None)
        truncated = len(layers) > max_layer + 1
        try:
            if truncated:
                backbone.layers = nn.ModuleList(list(layers)[: max_layer + 1])
            backbone.out_indices = list(grid)
            feature_maps = backbone(
                pixel_values=pixel_values,
                annotation=annotation,
                return_dict=False,
            )
        finally:
            if truncated:
                backbone.layers = layers
            if original_out_indices is None:
                # Defensive: an audited backbone always carries out_indices,
                # but never leave a stray attribute behind on exotic shapes.
                try:
                    delattr(backbone, "out_indices")
                except AttributeError:
                    pass
            else:
                backbone.out_indices = original_out_indices

        if not isinstance(feature_maps, (tuple, list)) or len(feature_maps) != len(grid):
            raise SkySensePPExecutionError(
                f"official forward returned {len(feature_maps) if isinstance(feature_maps, (tuple, list)) else 'non-sequence'} "
                f"maps, expected {len(grid)} for grid {grid}"
            )
        maps = tuple(Tensor.contiguous(m) if isinstance(m, Tensor) else m for m in feature_maps)
        return {
            "feature_maps": maps,
            "layer_indices": grid,
            "executed_layer_count": max_layer + 1,
        }
