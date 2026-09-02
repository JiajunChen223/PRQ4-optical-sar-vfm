from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch

from geotoken3path.data import normalize_croma_dynamic


def test_croma_recipe_matches_pinned_batch_spatial_formula() -> None:
    x = torch.arange(16 * 2 * 2 * 2, dtype=torch.float32).reshape(16, 2, 2, 2)
    output, trace = normalize_croma_dynamic(x, split="training")
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3), unbiased=True)
    expected = torch.clamp(
        (x - (mean.view(1, -1, 1, 1) - 2 * std.view(1, -1, 1, 1)))
        / (4 * std.view(1, -1, 1, 1)),
        0,
        1,
    )
    assert torch.allclose(output, expected)
    assert trace["statistics_axes"] == [0, 2, 3]
    assert trace["effective_batch"] == 16
    assert trace["padding_count"] == 0
    assert trace["distributed_statistics"] == "per_rank_micro_batch"


def test_validation_and_inference_padding_is_deterministic_and_changes_stats() -> None:
    base = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) + 1
    x = torch.cat((base, base + 10, base + 100), dim=0)
    first, trace = normalize_croma_dynamic(x, split="validation")
    second, second_trace = normalize_croma_dynamic(x, split="inference")
    assert first.shape == (3, 1, 2, 4)
    assert torch.equal(first, second)
    assert trace["padding_count"] == 13
    assert trace["trimmed_outputs"] is True
    assert trace["padding_changes_statistics"] is True
    assert torch.equal(trace["mean"], second_trace["mean"])
    unpadded_mean = x.mean(dim=(0, 2, 3))
    assert not torch.equal(trace["mean"], unpadded_mean)


def test_training_remainder_must_be_dropped_before_preprocessing() -> None:
    with pytest.raises(ValueError, match="exactly 16"):
        normalize_croma_dynamic(torch.arange(3 * 1 * 2 * 2, dtype=torch.float32).reshape(3, 1, 2, 2), split="training")


def test_masked_nodata_is_filled_and_zero_std_is_rejected() -> None:
    x = torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(16, 1, 2, 2)
    x[0, 0, 0, 0] = float("nan")
    mask = torch.ones((16, 2, 2), dtype=torch.bool)
    mask[0, 0, 0] = False
    output, _ = normalize_croma_dynamic(x, split="training", valid_mask=mask)
    assert torch.isfinite(output).all()
    assert output[0, 0, 0, 0].item() == 0.0

    constant = torch.ones((16, 1, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="standard deviation"):
        normalize_croma_dynamic(constant, split="training")
