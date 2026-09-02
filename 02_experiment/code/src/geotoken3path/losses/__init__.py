"""Declared segmentation losses."""

from .segmentation import (
    segmentation_cross_entropy,
    per_class_cross_entropy,
    macro_class_cross_entropy,
    lovasz_softmax_loss,
    segmentation_objective,
)

__all__ = [
    "segmentation_cross_entropy",
    "per_class_cross_entropy",
    "macro_class_cross_entropy",
    "lovasz_softmax_loss",
    "segmentation_objective",
]
