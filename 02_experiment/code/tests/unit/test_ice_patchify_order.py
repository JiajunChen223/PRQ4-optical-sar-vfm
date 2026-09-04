from __future__ import annotations

import torch
from torch import nn

from geotoken3path.execution.croma_executor import InterfaceCertifiedCromaExecutor


class _Encoder(nn.Module):
    patch_size = 2


def test_patchify_matches_official_channel_inner_pixel_order() -> None:
    image = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    actual = InterfaceCertifiedCromaExecutor._patchify_like_official_croma(
        _Encoder(), image
    )

    reference_patches: list[list[float]] = []
    for patch_row in range(0, 4, 2):
        for patch_col in range(0, 4, 2):
            values: list[float] = []
            # Official einops contract flattens (c, i, j) in this order.
            for channel in range(2):
                for inner_row in range(2):
                    for inner_col in range(2):
                        values.append(
                            float(
                                image[
                                    0,
                                    channel,
                                    patch_row + inner_row,
                                    patch_col + inner_col,
                                ]
                            )
                        )
            reference_patches.append(values)
    expected = torch.tensor([reference_patches], dtype=image.dtype)
    assert torch.equal(actual, expected)
