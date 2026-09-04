"""Unit tests for the pure-torch SkySense ViT audit encoder."""

from __future__ import annotations

import torch

from geotoken3path.vendor_audit.skysense_vit import (
    SkySenseVitEncoder,
    build_skysense_vit,
)


def _seed() -> torch.Generator:
    generator = torch.Generator().manual_seed(0)
    return generator


def _inputs(batch: int = 2, channels: int = 10, size: int = 64) -> torch.Tensor:
    return torch.randn(batch, channels, size, size, generator=_seed())


def test_default_s2_encoder_output_shape() -> None:
    """S2 config: in_channels=10 -> [B, 257, 1024] at out_indices=[23]."""
    encoder = build_skysense_vit(in_channels=10, seed=0)
    assert isinstance(encoder, SkySenseVitEncoder)
    assert encoder.num_layers == 24
    assert encoder.out_indices == [23]
    outputs = encoder(_inputs(channels=10))
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (2, 257, 1024)


def test_any_intermediate_layer_can_be_collected() -> None:
    """out_indices=[0, 5, 11, 17, 23] returns one tensor per layer."""
    encoder = build_skysense_vit(
        in_channels=10, out_indices=[0, 5, 11, 17, 23], seed=0
    )
    assert encoder.out_indices == [0, 5, 11, 17, 23]
    outputs = encoder(_inputs(channels=10))
    assert len(outputs) == 5
    for output in outputs:
        assert tuple(output.shape) == (2, 257, 1024)


def test_out_indices_collected_layers_are_distinct() -> None:
    """The five collected layers must actually hold different activations."""
    encoder = build_skysense_vit(
        in_channels=10, out_indices=[0, 5, 11, 17, 23], seed=0
    )
    outputs = encoder(_inputs(channels=10))
    for later, earlier in zip(outputs[1:], outputs[:-1]):
        assert not torch.equal(later, earlier)


def test_every_single_layer_is_reachable() -> None:
    """All 24 layers can be emitted at once (per-layer audit hooking)."""
    encoder = build_skysense_vit(
        in_channels=10, out_indices=list(range(24)), seed=0
    )
    outputs = encoder(_inputs(channels=10))
    assert len(outputs) == 24
    for output in outputs:
        assert tuple(output.shape) == (2, 257, 1024)


def test_s1_two_channel_config_forward() -> None:
    """S1 config: in_channels=2 must build and run."""
    encoder = build_skysense_vit(in_channels=2, seed=0)
    outputs = encoder(_inputs(channels=2))
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (2, 257, 1024)


def test_eval_and_train_agree_without_randomness() -> None:
    """drop_path_rate=0.0: deterministic in each mode, modes agree to fp noise.

    torch.nn.MultiheadAttention picks slightly different internal kernels for
    training vs eval (self.training flag), so eval and train outputs differ at
    ~1e-6 float32 noise even with zero dropout. What must hold is: repeated
    calls in the same mode are bit-identical (nothing random is active), and
    eval vs train agree within float precision.
    """
    encoder = build_skysense_vit(in_channels=10, seed=0)
    inputs = _inputs(channels=10)

    encoder.eval()
    with torch.no_grad():
        eval_output_a = encoder(inputs)[0]
        eval_output_b = encoder(inputs)[0]

    encoder.train()
    train_output_a = encoder(inputs)[0]
    train_output_b = encoder(inputs)[0]

    assert torch.equal(eval_output_a, eval_output_b)
    assert torch.equal(train_output_a, train_output_b)
    assert torch.allclose(eval_output_a, train_output_a, atol=1e-5)

    # Mode toggling must not change eval determinism either.
    encoder.eval()
    with torch.no_grad():
        eval_output_c = encoder(inputs)[0]
    assert torch.allclose(eval_output_c, train_output_a, atol=1e-5)


def test_state_dict_key_shape_layout() -> None:
    """State dict keys must follow the official architecture naming."""
    encoder = build_skysense_vit(in_channels=10, seed=0)
    state = encoder.state_dict()
    assert "patch_embed.weight" in state and "patch_embed.bias" in state
    assert state["patch_embed.weight"].shape == (1024, 10, 4, 4)
    assert "cls_token" in state and state["cls_token"].shape == (1, 1, 1024)
    assert "pos_embed" in state and state["pos_embed"].shape == (1, 257, 1024)
    for block_key in (
        "layers.0.norm1.weight",
        "layers.0.norm1.bias",
        "layers.0.attn.in_proj_weight",
        "layers.0.attn.in_proj_bias",
        "layers.0.attn.out_proj.weight",
        "layers.0.attn.out_proj.bias",
        "layers.0.norm2.weight",
        "layers.0.norm2.bias",
        "layers.0.ffn.layers.0.weight",
        "layers.0.ffn.layers.0.bias",
        "layers.0.ffn.layers.3.weight",
        "layers.0.ffn.layers.3.bias",
    ):
        assert block_key in state, f"missing official-style key {block_key}"
    # 24 blocks x 15 params each: LayerNorm x2 (2 each = 4), MHA packed
    # projection (in_proj_weight/in_proj_bias + out_proj.weight/bias = 4),
    # FFN 2 x Linear (2 each = 4), GELU/Dropout stateless, totalling 12 per
    # block, so 2 (patch_embed) + 2 (cls_token, pos_embed) + 24 * 12 = 292.
    assert len(state) == 2 + 2 + 24 * 12


def test_factory_seed_reproducibility() -> None:
    """Same seed gives identical parameters, different seeds do not."""
    encoder_a = build_skysense_vit(in_channels=10, seed=7)
    encoder_b = build_skysense_vit(in_channels=10, seed=7)
    encoder_c = build_skysense_vit(in_channels=10, seed=8)
    assert torch.equal(encoder_a.pos_embed, encoder_b.pos_embed)
    # Different seeds produce different sampled parameters (both pos_embed
    # tensors are still fully zero: init is random-init throughout, with
    # cls_token/pos_embed kept as official zeros(1, 1, C)).
    assert torch.equal(encoder_a.patch_embed.weight, encoder_b.patch_embed.weight)
    assert not torch.equal(encoder_a.patch_embed.weight, encoder_c.patch_embed.weight)
    assert not torch.equal(
        encoder_a.layers[23].ffn.layers[0].weight, encoder_c.layers[23].ffn.layers[0].weight
    )
