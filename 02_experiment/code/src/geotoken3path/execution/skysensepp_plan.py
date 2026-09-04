"""Compile R24 SkySense++ S2 ICE execution contracts into exact plans.

SkySense++ (unlike CROMA) has no auditable ``[attn, ffn]`` tap pairs with a
prefix FFN hook: the official forward is itself an exact per-layer loop, so
ICE is expressed as *truncation of that loop at a maximum executed layer*.
Each plan names the layer indices whose outputs the receiver must see
(``required_output_indices``, layer indices with the official out-grid
semantics ``feature_maps[k] <-> out_indices[k]``), the maximum executed layer,
the executed layer count and the eliminated suffix layers.

The official S2 backbone grid is fixed at layers (5, 11, 17, 23) of a 24-layer
ViT-L:

  - contract "a" (Full):  executes layers 0..23, receiver reads the four grid
    maps at layers 5/11/17/23; nothing is eliminated.
  - contract "b" (ICE):   executes layers 0..11, receiver reads the deepest
    executed grid slot (layer 11); layers 12..23 are eliminated
    (12/24 layers, ~49.9% of the backbone parameters).

``plan_sha256`` is a stable digest over the canonical JSON of the core fields,
so downstream certification can fingerprint the exact plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

# Official SkySense++ S2 backbone grid (layer indices).  Mirrors the constant
# pinned by the R24 model layer (models/skysensepp_seg.SKYSENSEPP_OUT_INDICES).
SKYSENSEPP_GRID_OUT_INDICES: tuple[int, ...] = (5, 11, 17, 23)
SKYSENSEPP_DEFAULT_NUM_LAYERS = 24
# Contract "b" executes layers 0..11 (12 layers); its required grid layer is 11.
_ICE_MAX_LAYER_B = 11


@dataclass(frozen=True)
class SkySensePPExecutionPlan:
    """Exact SkySense++ S2 prefix required by one receiver contract.

    ``required_output_indices`` uses the official out-grid semantics: they are
    *layer indices* (``feature_maps[k]`` corresponds to layer
    ``required_output_indices[k]`` of the executed grid).  Contract "a" reads
    all four official grid layers; contract "b" reads the deepest executed grid
    layer (11).
    """

    contract: str
    max_layer: int
    required_output_indices: tuple[int, ...]
    executed_layer_count: int
    eliminated_layers: tuple[int, ...]
    plan_sha256: str

    def payload(self) -> dict[str, Any]:
        """Serialize the plan for downstream certification records."""
        return asdict(self)


def validate_skysensepp_contract(contract: object) -> None:
    """Fail closed unless ``contract`` is the pinned R24 contract id."""
    if contract not in ("a", "b"):
        raise ValueError(
            f"skysensepp ICE contract must be 'a' (Full) or 'b' (ICE), got {contract!r}"
        )


def _canonical_core(plan_fields: dict[str, Any]) -> str:
    """Stable JSON of the core plan fields (independent of field order)."""
    canonical = json.dumps(
        {
            "contract": plan_fields["contract"],
            "max_layer": plan_fields["max_layer"],
            "required_output_indices": list(plan_fields["required_output_indices"]),
            "executed_layer_count": plan_fields["executed_layer_count"],
            "eliminated_layers": list(plan_fields["eliminated_layers"]),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return canonical


def compile_skysensepp_plan(
    contract: str,
    num_layers: int = SKYSENSEPP_DEFAULT_NUM_LAYERS,
) -> SkySensePPExecutionPlan:
    """Compile the exact executed prefix for one R24 receiver contract.

    ``num_layers`` defaults to the pinned 24-layer S2 backbone.  Contract "a"
    executes every layer and requires the four official grid maps.  Contract
    "b" executes layers 0..11 and requires the deepest executed grid map
    (layer 11), eliminating layers 12..23 when the full 24-layer backbone is
    present.  Plans always fail closed on impossible depths.
    """
    validate_skysensepp_contract(contract)
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")

    if contract == "a":
        max_layer = num_layers - 1
        required = SKYSENSEPP_GRID_OUT_INDICES
        executed = num_layers
        eliminated: tuple[int, ...] = ()
        if max(required) >= num_layers:
            raise ValueError(
                "contract 'a' requires the four official grid layers "
                f"({SKYSENSEPP_GRID_OUT_INDICES[-1]} <= num_layers - 1); "
                f"num_layers={num_layers} is too shallow"
            )
    else:
        max_layer = _ICE_MAX_LAYER_B
        required = (_ICE_MAX_LAYER_B,)
        executed = _ICE_MAX_LAYER_B + 1
        eliminated = tuple(range(executed, num_layers))
        if num_layers <= _ICE_MAX_LAYER_B:
            raise ValueError(
                f"contract 'b' executes layers 0..{_ICE_MAX_LAYER_B}, which requires "
                f"num_layers >= {_ICE_MAX_LAYER_B + 1}; got num_layers={num_layers}"
            )

    core = {
        "contract": contract,
        "max_layer": max_layer,
        "required_output_indices": required,
        "executed_layer_count": executed,
        "eliminated_layers": eliminated,
    }
    digest = hashlib.sha256(_canonical_core(core).encode("utf-8")).hexdigest()
    return SkySensePPExecutionPlan(**core, plan_sha256=digest)
