from __future__ import annotations

import re

import pytest

from geotoken3path.execution.contracts import CromaExecutionContractError
from geotoken3path.execution.topology import (
    BackboneTopology,
    CROMA_TOPOLOGY,
    DroppableBranch,
    TapSemantic,
    validate_topology,
)


# ---------- Declared facts match the hardcoded ICE/CROMA facts ----------

def test_croma_topology_declares_dual_encoder_layout() -> None:
    assert CROMA_TOPOLOGY.name == "croma"
    assert CROMA_TOPOLOGY.kernel_family == "croma"
    assert CROMA_TOPOLOGY.encoder_names == ("s1_encoder", "s2_encoder")
    assert CROMA_TOPOLOGY.depth_path_template == "{encoder}.transformer.layers"
    assert CROMA_TOPOLOGY.block_structure == "module_pair"
    assert CROMA_TOPOLOGY.stem_path == "{encoder}.linear_input"
    assert CROMA_TOPOLOGY.patch_size_attr == "patch_size"
    assert CROMA_TOPOLOGY.extra_tensors == ("attn_bias",)
    assert CROMA_TOPOLOGY.final_norm_path_template == "{encoder}.transformer.norm_out"
    # Templates expand to the exact paths the plan/executor/export hardcode.
    assert CROMA_TOPOLOGY.depth_path_template.format(
        encoder="s1_encoder"
    ) == "s1_encoder.transformer.layers"
    assert CROMA_TOPOLOGY.depth_path_template.format(
        encoder="s2_encoder"
    ) == "s2_encoder.transformer.layers"
    assert CROMA_TOPOLOGY.stem_path.format(encoder="s2_encoder") == "s2_encoder.linear_input"


def test_croma_topology_droppable_branches_match_hardcoded_nodes() -> None:
    by_path = {branch.node_path: branch for branch in CROMA_TOPOLOGY.droppable_branches}
    assert set(by_path) == {"GAP_FFN_s1", "GAP_FFN_s2", "cross_encoder", "joint_GAP"}
    # joint_GAP is a tensor mean in the official forward, not a module.
    assert by_path["GAP_FFN_s1"].stateful_tensor_mean is False
    assert by_path["GAP_FFN_s2"].stateful_tensor_mean is False
    assert by_path["cross_encoder"].stateful_tensor_mean is False
    assert by_path["joint_GAP"].stateful_tensor_mean is True


def test_tap_semantic_members() -> None:
    assert TapSemantic.FFN_PRE_RESIDUAL.value == "ffn_pre_residual"
    assert TapSemantic.BLOCK_OUTPUT.value == "block_output"
    assert set(TapSemantic) == {
        TapSemantic.FFN_PRE_RESIDUAL,
        TapSemantic.BLOCK_OUTPUT,
    }


def test_tap_path_regex_accepts_only_audited_ffn_taps() -> None:
    pattern = CROMA_TOPOLOGY.tap_path_regex
    # Legal taps: s1/s2 encoder FFN (``.1``) at any layer index.
    assert pattern.fullmatch("s1_encoder.transformer.layers.0.1") is not None
    assert pattern.fullmatch("s2_encoder.transformer.layers.5.1") is not None
    assert pattern.fullmatch("s1_encoder.transformer.layers.11.1") is not None
    # Rejected: wrong suffix, self-attn block, encoder/module names, malformed.
    for path in (
        "s1_encoder.transformer.layers.0.0",
        "s3_encoder.transformer.layers.0.1",
        "s1_encoder.transformer.layers.0",
        "s1_encoder.transformer.layers.5.1.extra",
        "encoder.transformer.layers.5.1",
        "joint_GAP",
        "",
    ):
        assert pattern.fullmatch(path) is None, path
    # Same shape as the plan compiler's hardcoded tap regex.
    assert pattern.pattern == r"^(s1_encoder|s2_encoder)\.transformer\.layers\.(\d+)\.1$"


# ---------- validate_topology ----------

def _valid_topology() -> BackboneTopology:
    return BackboneTopology(
        name="valid",
        encoder_names=("s1_encoder", "s2_encoder"),
        depth_path_template="{encoder}.transformer.layers",
        block_structure="module_pair",
        stem_path="{encoder}.linear_input",
        patch_size_attr="patch_size",
        extra_tensors=("attn_bias",),
        final_norm_path_template="{encoder}.transformer.norm_out",
        droppable_branches=(DroppableBranch("GAP_FFN_s1"),),
        tap_path_regex=re.compile(r"^(s1_encoder|s2_encoder)\.transformer\.layers\.(\d+)\.1$"),
        kernel_family="croma",
    )


def test_validate_topology_accepts_well_formed_topology() -> None:
    topology = _valid_topology()
    assert validate_topology(topology) is topology


def test_validate_topology_rejects_empty_encoder_names() -> None:
    topology = _valid_topology()
    with pytest.raises(CromaExecutionContractError):
        validate_topology(
            topology=BackboneTopology(
                name=topology.name,
                encoder_names=(),
                depth_path_template=topology.depth_path_template,
                block_structure=topology.block_structure,
                stem_path=topology.stem_path,
                patch_size_attr=topology.patch_size_attr,
                extra_tensors=topology.extra_tensors,
                final_norm_path_template=topology.final_norm_path_template,
                droppable_branches=topology.droppable_branches,
                tap_path_regex=topology.tap_path_regex,
                kernel_family=topology.kernel_family,
            )
        )


def test_validate_topology_rejects_non_topology_object() -> None:
    with pytest.raises(CromaExecutionContractError):
        validate_topology("not a topology")  # type: ignore[arg-type]


def test_validate_topology_rejects_invalid_block_structure() -> None:
    topology = _valid_topology()
    with pytest.raises(CromaExecutionContractError):
        validate_topology(
            topology=BackboneTopology(
                name=topology.name,
                encoder_names=topology.encoder_names,
                depth_path_template=topology.depth_path_template,
                block_structure="not_module_pair",  # type: ignore[arg-type]
                stem_path=topology.stem_path,
                patch_size_attr=topology.patch_size_attr,
                extra_tensors=topology.extra_tensors,
                final_norm_path_template=topology.final_norm_path_template,
                droppable_branches=topology.droppable_branches,
                tap_path_regex=topology.tap_path_regex,
                kernel_family=topology.kernel_family,
            )
        )
