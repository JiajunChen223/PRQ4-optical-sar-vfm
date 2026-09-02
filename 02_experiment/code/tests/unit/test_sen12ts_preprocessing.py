from __future__ import annotations

import torch

from geotoken3path.data.sen12ts import _center_crop


def test_center_crop_derives_frozen_croma_window() -> None:
    value = torch.arange(1 * 256 * 256, dtype=torch.float32).reshape(1, 256, 256)
    cropped = _center_crop(value, size=(120, 120), name="optical")
    assert tuple(cropped.shape) == (1, 120, 120)
    assert cropped[0, 0, 0] == value[0, 68, 68]
