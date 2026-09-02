from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geotoken3path.diagnostics import (
    DiagnosticContractError,
    auroc,
    cohens_d,
    fixed_derangement,
    permutation_sha256,
    separation,
    shift_token_grid,
    spearman_correlation,
)

def test_fixed_derangement_is_reproducible_and_has_no_fixed_points() -> None:
    left = fixed_derangement(16, 401)
    right = fixed_derangement(16, 401)
    assert torch.equal(left, right)
    assert not bool((left == torch.arange(16)).any())
    assert sorted(left.tolist()) == list(range(16))
    assert permutation_sha256(left) == permutation_sha256(right)


def test_small_derangement_is_identity_only_when_impossible() -> None:
    assert torch.equal(fixed_derangement(0, 1), torch.empty(0, dtype=torch.long))
    assert torch.equal(fixed_derangement(1, 1), torch.zeros(1, dtype=torch.long))


def test_shift_token_grid_uses_clamped_nonwrapping_sampling() -> None:
    tokens = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    shifted = shift_token_grid(tokens, 1, 0)
    assert shifted.reshape(3, 3).tolist() == [[0.0, 0.0, 1.0], [3.0, 3.0, 4.0], [6.0, 6.0, 7.0]]
    assert torch.equal(shift_token_grid(tokens, 0, 0), tokens)


def test_shift_token_grid_supports_depth_group_tail_dimension() -> None:
    tokens = torch.arange(9 * 4 * 2, dtype=torch.float32).reshape(1, 9, 4, 2)
    shifted = shift_token_grid(tokens, 0, 1)
    assert shifted.shape == tokens.shape
    assert torch.equal(shifted[:, 0], tokens[:, 0])
    assert torch.equal(shifted[:, 3], tokens[:, 0])


def test_separation_reports_shuffled_as_positive() -> None:
    correct = torch.tensor([0.1, 0.2, 0.3, 0.2])
    shuffled = torch.tensor([0.7, 0.8, 0.9, 0.8])
    result = separation(correct, shuffled, label="conflict")
    assert result["mean_difference_shuffled_minus_correct"] > 0
    assert result["auroc_shuffled_positive"] == pytest.approx(1.0)
    assert result["cohens_d"] > 0


def test_statistics_are_tie_and_constant_safe() -> None:
    assert auroc(torch.ones(3), torch.ones(3)) == pytest.approx(0.5)
    assert cohens_d(torch.ones(3), torch.ones(3)) == pytest.approx(0.0)
    assert spearman_correlation(torch.arange(3, dtype=torch.float64), torch.arange(3, dtype=torch.float64)) == pytest.approx(1.0)


def test_invalid_diagnostic_inputs_fail_closed() -> None:
    with pytest.raises(DiagnosticContractError):
        shift_token_grid(torch.zeros(1, 8, 2), 0, 0)
    with pytest.raises(DiagnosticContractError):
        fixed_derangement(2, -1)


