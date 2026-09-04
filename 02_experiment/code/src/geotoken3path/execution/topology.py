"""Data-driven backbone topology facts for ICE execution/cropping decisions.

One backbone family's complete execution/cropping facts are declared once here
instead of being re-hardcoded in the plan compiler, the certified executor, and
the export stripper.  R22-P0 only declares the facts; no existing module
consumes them yet, so behaviour is unchanged.  ``CROMA_TOPOLOGY`` encodes the
current audited facts:

  - two modality encoders ``s1_encoder`` (SAR) and ``s2_encoder`` (optical),
    each exposing ``transformer.layers`` as a ModuleList of [self_attn, ffn]
    block pairs with a ``linear_input`` patch-embedding projection;
  - one shared 2D-ALiBi ``attn_bias`` plain tensor attribute (never a buffer);
  - per-encoder ``transformer.norm_out`` final norm (official forward applies
    it after the last layer; ICE skips it when nothing downstream consumes it);
  - droppable top-level nodes: the GAP heads ``GAP_FFN_s1`` / ``GAP_FFN_s2``,
    the joint ``cross_encoder``, and ``joint_GAP`` which is *not* a module but
    a tensor mean of the joint encodings computed by the official forward.
"""

from __future__ import annotations

from enum import Enum
import re
from dataclasses import dataclass
from typing import Literal

from .contracts import CromaExecutionContractError


class TapSemantic(Enum):
    """Semantic meaning of the module a legal tap path addresses."""

    FFN_PRE_RESIDUAL = "ffn_pre_residual"
    BLOCK_OUTPUT = "block_output"


@dataclass(frozen=True)
class DroppableBranch:
    """One top-level backbone node that a minimal plan may crop entirely.

    ``node_path`` is the backbone-level attribute path (a module for
    ``GAP_FFN_s1`` / ``GAP_FFN_s2`` / ``cross_encoder``; a bare name for
    ``joint_GAP``, which is not a module).  ``stateful_tensor_mean`` marks a
    node that exists only as a tensor computed by the official forward (the
    ``joint_GAP`` mean) and therefore has no module to prune or hook-check.
    """

    node_path: str
    stateful_tensor_mean: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.node_path, str) or not self.node_path.strip():
            raise CromaExecutionContractError("DroppableBranch.node_path must be a nonempty string")
        if not isinstance(self.stateful_tensor_mean, bool):
            raise CromaExecutionContractError("DroppableBranch.stateful_tensor_mean must be boolean")


@dataclass(frozen=True)
class BackboneTopology:
    """Complete execution/cropping facts for one audited backbone family.

    Field values must agree with the pinned official CROMA forward so that the
    certified executor, the plan compiler, and the export stripper can later
    derive their decisions from a single source of truth.  All container
    fields are immutable tuples (or str / bool / Literal scalars) so a topology
    instance is safe to share and hash.
    """

    name: str
    #: One entry per encoder; order is the official construction/execution order.
    encoder_names: tuple[str, ...]
    #: Format string for each encoder's depth container path; CROMA's ``{:s}``
    #: expands to the encoder name itself: ``s1_encoder.transformer.layers``.
    depth_path_template: str
    #: ``module_pair``: every depth layer is a ModuleList of exactly two
    #: modules, ``[self_attn, ffn]``, executed as attn+residual then ffn+residual.
    block_structure: Literal["module_pair"]
    #: Format string for each encoder's input projection attribute path.
    stem_path: str
    #: Attribute name holding the patch-embedding size on each encoder.
    patch_size_attr: str
    #: Names of shared plain-tensor attributes consumed by every forward
    #: (e.g. the 2D-ALiBi ``attn_bias``; never buffers, so state-dict layout is
    #: unaffected).
    extra_tensors: tuple[str, ...]
    #: Format string for the per-encoder final norm path.
    final_norm_path_template: str
    #: Top-level nodes droppable by a minimal plan.
    droppable_branches: tuple[DroppableBranch, ...]
    #: Compiled pattern for legal tap paths (FFN pre-residual outputs).
    tap_path_regex: re.Pattern[str]
    #: Execution kernel family identifier.
    kernel_family: str = "croma"


def validate_topology(topology: BackboneTopology) -> BackboneTopology:
    """Fail closed unless ``topology`` is a well-formed backbone topology.

    Checks field types and nonemptiness and that ``tap_path_regex`` is a
    compiled, anchored pattern.  Returns the validated topology unchanged;
    raising is reserved for malformed declarations (programming errors in
    future backbone facts, not for runtime inputs).
    """

    if not isinstance(topology, BackboneTopology):
        raise CromaExecutionContractError("topology must be a BackboneTopology")
    if not isinstance(topology.name, str) or not topology.name.strip():
        raise CromaExecutionContractError("BackboneTopology.name must be a nonempty string")
    if (
        not isinstance(topology.encoder_names, tuple)
        or not topology.encoder_names
        or any(not isinstance(name, str) or not name.strip() for name in topology.encoder_names)
    ):
        raise CromaExecutionContractError("encoder_names must be a nonempty tuple of names")
    if len(set(topology.encoder_names)) != len(topology.encoder_names):
        raise CromaExecutionContractError("encoder_names must not contain duplicates")
    for attr in (
        "depth_path_template",
        "stem_path",
        "patch_size_attr",
        "final_norm_path_template",
        "kernel_family",
    ):
        value = getattr(topology, attr)
        if not isinstance(value, str) or not value.strip():
            raise CromaExecutionContractError(f"{attr} must be a nonempty string")
    if topology.block_structure != "module_pair":
        raise CromaExecutionContractError("block_structure must be 'module_pair'")
    if not isinstance(topology.extra_tensors, tuple) or any(
        not isinstance(name, str) or not name.strip() for name in topology.extra_tensors
    ):
        raise CromaExecutionContractError("extra_tensors must be a tuple of nonempty names")
    if not isinstance(topology.droppable_branches, tuple) or not topology.droppable_branches:
        raise CromaExecutionContractError("droppable_branches must be a nonempty tuple")
    for branch in topology.droppable_branches:
        if not isinstance(branch, DroppableBranch):
            raise CromaExecutionContractError("droppable_branches entries must be DroppableBranch")
    if not isinstance(topology.tap_path_regex, re.Pattern):
        raise CromaExecutionContractError("tap_path_regex must be a compiled regex pattern")
    try:
        topology.tap_path_regex.fullmatch("s1_encoder.transformer.layers.0.1")
    except re.error as exc:
        raise CromaExecutionContractError("tap_path_regex is not a compilable regex") from exc
    return topology


#: Single source of truth for the audited CROMA execution/cropping facts.
CROMA_TOPOLOGY = validate_topology(
    BackboneTopology(
        name="croma",
        encoder_names=("s1_encoder", "s2_encoder"),
        depth_path_template="{encoder}.transformer.layers",
        block_structure="module_pair",
        stem_path="{encoder}.linear_input",
        patch_size_attr="patch_size",
        extra_tensors=("attn_bias",),
        final_norm_path_template="{encoder}.transformer.norm_out",
        droppable_branches=(
            DroppableBranch(node_path="GAP_FFN_s1"),
            DroppableBranch(node_path="GAP_FFN_s2"),
            DroppableBranch(node_path="cross_encoder"),
            DroppableBranch(node_path="joint_GAP", stateful_tensor_mean=True),
        ),
        tap_path_regex=re.compile(
            r"^(s1_encoder|s2_encoder)\.transformer\.layers\.(\d+)\.1$"
        ),
        kernel_family="croma",
    )
)
