"""Pure-tensor semantic segmentation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(logits: Tensor, target: Tensor, num_classes: int, ignore_index: int = 255) -> Tensor:
    prediction = logits.argmax(dim=1).reshape(-1)
    truth = target.reshape(-1)
    keep = truth != ignore_index
    truth = truth[keep]
    prediction = prediction[keep]
    valid = (truth >= 0) & (truth < num_classes)
    indices = truth[valid] * num_classes + prediction[valid]
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def mean_iou(matrix: Tensor) -> Tensor:
    matrix = matrix.to(torch.float64)
    intersection = matrix.diag()
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - intersection
    valid = union > 0
    if not bool(valid.any()):
        return matrix.new_tensor(float("nan"))
    return (intersection[valid] / union[valid]).mean()
