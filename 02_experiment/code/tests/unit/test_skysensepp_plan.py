"""Unit tests for R24 SkySense++ ICE execution plans (contracts a/b).

The plan compiler is pure logic over the pinned 24-layer S2 grid
(5, 11, 17, 23); it never touches the vendor model, so these tests run
instantly.
"""

from __future__ import annotations

import pytest

from geotoken3path.execution.skysensepp_plan import (
    SKYSENSEPP_GRID_OUT_INDICES,
    compile_skysensepp_plan,
    validate_skysensepp_contract,
)


def test_compile_contract_a_full_plan_fields() -> None:
    plan = compile_skysensepp_plan("a")
    assert plan.contract == "a"
    assert plan.max_layer == 23
    assert plan.required_output_indices == (5, 11, 17, 23)
    assert plan.executed_layer_count == 24
    assert plan.eliminated_layers == ()


def test_compile_contract_b_ice_plan_fields() -> None:
    plan = compile_skysensepp_plan("b")
    assert plan.contract == "b"
    assert plan.max_layer == 11
    assert plan.required_output_indices == (11,)
    assert plan.executed_layer_count == 12
    assert plan.eliminated_layers == tuple(range(12, 24))
    assert plan.eliminated_layers == (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)


def test_required_output_indices_are_layer_indices() -> None:
    """feature_maps[k] corresponds to layer required_output_indices[k] of the grid."""
    for contract in ("a", "b"):
        plan = compile_skysensepp_plan(contract)
        for required in plan.required_output_indices:
            assert required in SKYSENSEPP_GRID_OUT_INDICES
        assert max(plan.required_output_indices) <= plan.max_layer


def test_contract_b_executes_exactly_half_the_layers() -> None:
    plan_b = compile_skysensepp_plan("b")
    plan_a = compile_skysensepp_plan("a")
    assert plan_b.executed_layer_count == 12
    assert plan_a.executed_layer_count == 24
    # ICE eliminates the 12-layer suffix of Full.
    assert len(plan_b.eliminated_layers) == 12
    assert plan_b.executed_layer_count + len(plan_b.eliminated_layers) == plan_a.executed_layer_count


def test_plan_sha256_is_stable_and_deterministic() -> None:
    first = compile_skysensepp_plan("b")
    second = compile_skysensepp_plan("b")
    assert first.plan_sha256 == second.plan_sha256
    assert len(first.plan_sha256) == 64
    assert int(first.plan_sha256, 16) >= 0  # hex digest


def test_plan_a_and_b_hashes_differ() -> None:
    assert compile_skysensepp_plan("a").plan_sha256 != compile_skysensepp_plan("b").plan_sha256


def test_plan_hash_changes_with_contract_fields() -> None:
    """The digest covers the core fields (not just the contract id)."""
    # Same contract on a deeper stack must change max_layer/count and the hash.
    deep = compile_skysensepp_plan("a", num_layers=48)
    assert deep.max_layer == 47
    assert deep.executed_layer_count == 48
    assert compile_skysensepp_plan("a").plan_sha256 != deep.plan_sha256


def test_payload_round_trips_core_fields() -> None:
    payload = compile_skysensepp_plan("b").payload()
    assert isinstance(payload, dict)
    assert payload["contract"] == "b"
    assert payload["max_layer"] == 11
    assert payload["required_output_indices"] == (11,)
    assert payload["executed_layer_count"] == 12
    assert payload["eliminated_layers"] == tuple(range(12, 24))
    assert payload["plan_sha256"] == compile_skysensepp_plan("b").plan_sha256


def test_plan_dataclass_is_frozen() -> None:
    plan = compile_skysensepp_plan("a")
    with pytest.raises(Exception):
        plan.max_layer = 0  # type: ignore[misc]


def test_validate_contract_rejects_illegal_contracts() -> None:
    with pytest.raises(ValueError):
        validate_skysensepp_contract("c")
    with pytest.raises(ValueError):
        validate_skysensepp_contract("A")
    with pytest.raises(ValueError):
        validate_skysensepp_contract("")
    with pytest.raises(ValueError):
        validate_skysensepp_contract(None)


def test_compile_rejects_invalid_num_layers() -> None:
    with pytest.raises(ValueError):
        compile_skysensepp_plan("a", num_layers=0)
    with pytest.raises(ValueError):
        compile_skysensepp_plan("b", num_layers=11)  # needs >= 12 to reach layer 11
    with pytest.raises(ValueError):
        compile_skysensepp_plan("a", num_layers=1)  # grid layer 23 unreachable
    with pytest.raises(ValueError):
        compile_skysensepp_plan("b", num_layers=True)


def test_compile_contract_b_on_shallow_24_grid_matches_default() -> None:
    assert compile_skysensepp_plan("b", num_layers=24) == compile_skysensepp_plan("b")
