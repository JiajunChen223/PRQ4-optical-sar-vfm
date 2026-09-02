from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch

from geotoken3path.losses import segmentation_cross_entropy
from geotoken3path.metrics import confusion_matrix, mean_iou
from geotoken3path.models.decoder import TokenSegmentationDecoder


def test_decoder_loss_metric_synthetic_contract() -> None:
    decoder = TokenSegmentationDecoder(dim=16, num_classes=4)
    tokens = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(0), requires_grad=True)
    target = torch.randint(0, 4, (2, 12, 12), generator=torch.Generator().manual_seed(1))
    target[:, 0, 0] = 255
    logits = decoder(tokens, (12, 12))
    loss = segmentation_cross_entropy(logits, target)
    loss.backward()
    matrix = confusion_matrix(logits.detach(), target, num_classes=4)
    assert logits.shape == (2, 4, 12, 12)
    assert tokens.grad is not None
    assert matrix.shape == (4, 4)
    assert torch.isfinite(mean_iou(matrix))


def test_decoder_rejects_non_square_token_grid() -> None:
    decoder = TokenSegmentationDecoder(dim=8, num_classes=3)
    with pytest.raises(ValueError):
        decoder(torch.zeros(1, 10, 8), (8, 8))
