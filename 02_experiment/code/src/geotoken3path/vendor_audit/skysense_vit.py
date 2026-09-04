"""Pure-PyTorch equivalent of the official SkySense Vision Transformer (S2/S1).

Audit-only reimplementation (R23 structural audit) of the official encoder
architecture. The official class (vendor/skysense_repo/models/vision_transformer.py,
CVPR 2024 SkySense) is built on mmcv/mmseg bricks that are not installable on
torch 2.12/Windows. The official README states the weights "exhibit
compatibility with ... torchvision and TIMM, necessitating only rudimentary
conversions", which is what makes this module-by-module rewrite possible.

Structural equivalence (all verified against the official source):
- PatchEmbed: strided Conv2d(kernel=patch_size, stride=patch_size) that maps
  [B, C, H, W] -> [B, num_patches, embed_dims]. The official mmseg PatchEmbed
  uses padding='corner' (pad only on the bottom/right so H % patch == 0); for
  the fixed 64x64 audit input with patch 4 that padding is 0, so a plain
  convolution is exact. Official key names: patch_embed.projection.weight /
  patch_embed.projection.bias (nn.Parameter requires_grad_() state under the
  module name patch_embed).
- Class token prepended: [B, 257, 1024]; pos_embed [1, 257, 1024] added
  element-wise (official drop_after_pos is nn.Dropout(drop_rate), identity at
  drop_rate=0). Official keys: cls_token, pos_embed.
- num_layers pre-LN Transformer blocks (the official TransformerEncoderLayer):
  x = x + attn(norm1(x))  with torch-style residual (official mmcv
  MultiheadAttention forwards with identity=x and returns identity + attn_out;
  its attn projection is torch.nn.MultiheadAttention itself, so self-attn is
  bit-identical to the official mmcv build when need_weights=False), then
  x = x + ffn(norm2(x)) where ffn = 2-layer MLP with GELU (official mmcv FFN
  with num_fcs=2 is Linear -> GELU -> Dropout(ffn_drop=drop_rate) -> Linear ->
  Dropout; both dropouts are identity at drop_rate=0, so the plain
  Linear-GELU-Linear MLP is exact).
- State_dict layout deliberately mirrors the official keys
  (layers.N.norm1.*, layers.N.attn.in_proj_weight/in_proj_bias/
  out_proj.weight/out_proj.bias, layers.N.norm2.*, layers.N.ffn.layers.0.*,
  layers.N.ffn.layers.1.*) so an official checkpoint can be loaded later via
  load_state_dict(strict=False) after the standard mmcv-vs-torch renames:
  mmcv stores its MHA under layers.N.attn.attn.* and marks GELU activations
  with requires_grad_(False), and official patch_embed keys sit under
  patch_embed.projection.*. The audit does not depend on any of that.

Forward contract: one output token-sequence tensor per entry of out_indices,
i.e. for each requested layer. Unlike official mmseg ViT heads, the audit
module returns the transformer token sequence [B, 1 + (H/patch)^2,
embed_dims] (class token included), exactly the tensor the official code
feeds into its decoder neck at out_indices. It does not do the mmseg
reshape(B, hw_h, hw_w, C) -> permute(0, 3, 1, 2) decoder layout, which is
head plumbing, not encoder structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


class _FFN(nn.Module):
    """2-layer MLP with GELU, analogue of the official mmcv FFN(num_fcs=2).

    The official mmcv FFN stores its stack under the attribute name
    ``layers`` (``self.layers = nn.Sequential(Linear, act, Dropout, Linear,
    Dropout)``), so parameters of the two Linear layers live at the official
    key path ``layers.N.ffn.layers.0.*`` and ``layers.N.ffn.layers.1.*``
    (here .3, the second Linear) -- kept by nesting the same way. Both
    Dropouts are identity at p=0.
    """

    def __init__(
        self, embed_dims: int, feedforward_channels: int, ffn_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.GELU(),
            nn.Dropout(p=ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(p=ffn_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SkySenseBlock(nn.Module):
    """One pre-LN ViT encoder layer, exact analogue of the official layer.

    Official (mmseg TransformerEncoderLayer) order per layer:
        x = attn(norm1(x)) with identity shortcut
        x = ffn(norm2(x))  with identity shortcut

    torch.nn.MultiheadAttention needs input in [B, N, C] when batch_first=True;
    this torch build always returns the (attn_output, attn_output_weights)
    pair from the module call even with need_weights=False (the weights value
    is None then), so forward indexes [0] on the attention result.
    The class-token self-attention pattern of the official code already feeds
    the full sequence (class token included) into every layer, so this block
    simply transforms the whole token sequence.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        # Keep official attribute names: layers.N.norm1 / layers.N.attn /
        # layers.N.norm2 / layers.N.ffn.
        self.norm1 = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        # Official mmcv FFN(num_fcs=2, act_cfg=GELU, ffn_drop=proj_drop):
        # layers.N.ffn.layers.* key path, see _FFN.
        self.ffn = _FFN(embed_dims, feedforward_channels, ffn_drop=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C] token sequence; returns [B, N, C]."""
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SkySenseVitEncoder(nn.Module):
    """Pure-torch Vision Transformer-Large encoder used by SkySense S2/S1.

    Mirrors the default audit configuration of the official VisionTransformer
    (img_size=64, patch_size=4, in_channels=10 or 2, embed_dims=1024,
    num_layers=24, num_heads=16, mlp_ratio=4, with_cls_token=True,
    patch_norm=False, final_norm=False) with stochastic depth disabled
    (drop_path_rate=0.0) so the module is fully deterministic in eval and
    train mode.

    Official position-embedding resizing is omitted deliberately: the audit
    only feeds the fixed 64x64 input the encoder was trained at, where no
    resize can ever trigger.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 10,
        embed_dims: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        out_indices: int | Iterable[int] | None = None,
        *,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if drop_path_rate:
            raise NotImplementedError(
                "stochastic depth is not implemented in the audit encoder; "
                "pass drop_path_rate=0.0"
            )
        if drop_rate or attn_drop_rate:
            raise NotImplementedError(
                "audit encoder is deterministic-only; drop_rate and "
                "attn_drop_rate must be 0.0"
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} must be divisible by patch_size {patch_size}"
            )

        # Official PatchEmbed (padding='corner' == 0 for a divisible img_size),
        # stored under the module name patch_embed so the official keys
        # patch_embed.weight / patch_embed.bias load straight onto a Conv2d.
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_size = patch_size

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.with_cls_token = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dims)
        )

        # Resolve requested output indices exactly like the official code:
        # out_indices=-1 means the final layer; everything else is kept as-is.
        if out_indices is None:
            indices: Sequence[int] = (num_layers - 1,)
        elif isinstance(out_indices, int):
            indices = (num_layers - 1,) if out_indices == -1 else (out_indices,)
        else:
            indices = tuple(out_indices)
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < num_layers:
                raise ValueError(
                    f"out_indices entries must satisfy 0 <= i < num_layers "
                    f"({num_layers}), got {indices}"
                )
        self.out_indices = sorted(set(indices))

        self.layers = nn.ModuleList(
            _SkySenseBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
            )
            for _ in range(num_layers)
        )

    @property
    def output_cls_token(self) -> bool:
        """Compatibility accessor: audit encoder always keeps the cls token."""
        return False

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Map [B, C, H, W] inputs to per-layer token-sequence outputs.

        Returns one tensor per entry of out_indices, in ascending layer order;
        each output has shape [B, 1 + num_patches, embed_dims] and is the
        encoder token sequence (class token included) right after that layer.
        """
        if inputs.ndim != 4:
            raise ValueError(f"inputs must be [B, C, H, W], got shape {tuple(inputs.shape)}")
        batch_size = inputs.shape[0]

        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed

        outputs: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens)
            if index in self.out_indices:
                outputs.append(tokens)
        return outputs


def build_skysense_vit(
    in_channels: int = 10,
    img_size: int = 64,
    num_layers: int = 24,
    *,
    patch_size: int = 4,
    embed_dims: int = 1024,
    num_heads: int = 16,
    mlp_ratio: int = 4,
    out_indices: int | Iterable[int] | None = None,
    drop_path_rate: float = 0.0,
    seed: int | None = None,
) -> SkySenseVitEncoder:
    """Build a (optionally seed-stable) random-init SkySense ViT encoder."""
    if seed is not None:
        torch.manual_seed(seed)
    return SkySenseVitEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dims=embed_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        out_indices=out_indices,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )
