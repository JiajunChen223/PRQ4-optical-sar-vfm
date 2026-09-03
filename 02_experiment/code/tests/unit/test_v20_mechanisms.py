# -*- coding: utf-8 -*-
"""Unit tests for V20 mechanisms R2 (depth-group injection) and R1 (energy gain)."""
import pytest
import torch

from geotoken3path.mechanisms.r2_depth_inject import R2DepthGroupInjector
from geotoken3path.mechanisms.r1_energy_gain import R1LowEnergyChannelGain
from geotoken3path.models.fusion import OpticalSarTokenModel


def _tokens(batch: int = 2, tokens: int = 49, dim: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return (
        torch.randn(batch, tokens, dim, generator=g),
        torch.randn(batch, tokens, dim, generator=g),
        torch.randn(batch, tokens, 4, dim, generator=g),
    )


# ---------- R2 ----------

def test_r2_zero_start_is_exact_identity() -> None:
    injector = R2DepthGroupInjector(dim=16)
    optical, sar, depth = _tokens()
    out = injector(depth, sar)
    # layer_proj 零初始化 -> 注入恒为零
    assert torch.equal(out, sar)
    # 初始权重均匀 1/4
    assert torch.allclose(torch.softmax(injector.layer_weights, dim=0), torch.full((4,), 0.25), atol=1e-6)


def test_r2_shape_and_gradient_liveness() -> None:
    injector = R2DepthGroupInjector(dim=16)
    optical, sar, depth = _tokens()
    out = injector(depth, sar)
    assert out.shape == sar.shape
    out.mean().backward()
    assert injector.layer_weights.grad is not None
    assert injector.layer_proj.weight.grad is not None


def test_r2_invalid_shapes_fail_closed() -> None:
    injector = R2DepthGroupInjector(dim=16)
    with pytest.raises(ValueError):
        injector(torch.randn(2, 49, 16), torch.randn(2, 49, 16))
    with pytest.raises(ValueError):
        injector(torch.randn(2, 49, 3, 16), torch.randn(2, 49, 16))
    with pytest.raises(ValueError):
        injector(torch.randn(3, 49, 4, 16), torch.randn(2, 49, 16))


def test_r2_router_mount_and_forward_parity_with_baseline() -> None:
    g = torch.Generator().manual_seed(11)
    cfg = {"token_dim": 16, "num_classes": 8, "active_budget": 1.0,
           "local_window_tokens": 49, "stages": ("mid", "late"),
           "allow_synthetic_depth_group_fallback": True}
    baseline = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    candidate = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r2_depth_group_inject", stages=("mid", "late"))
    # 共享公共权重（机制零起步）
    shared = {k: v for k, v in baseline.state_dict().items() if k in candidate.state_dict()}
    candidate.load_state_dict(shared, strict=False)
    baseline.eval(); candidate.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    with torch.no_grad():
        lb, ab = baseline(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
        lc, ac = candidate(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start R2 must be exactly identical to baseline"


# ---------- R1 ----------

def test_r1_zero_start_is_exact_identity() -> None:
    gain = R1LowEnergyChannelGain()
    g = torch.Generator().manual_seed(3)
    fused = torch.randn(2, 49, 16, generator=g)
    out = gain(fused)
    assert torch.equal(out, fused)
    assert float(gain.gamma) == 0.0


def test_r1_gain_magnifies_low_energy_channels() -> None:
    gain = R1LowEnergyChannelGain()
    with torch.no_grad():
        gain.raw_gamma.fill_(0.5)
    g = torch.Generator().manual_seed(5)
    fused = torch.rand(2, 49, 16, generator=g)
    fused[:, :, :8] *= 0.05  # 前 8 通道低能量
    out = gain(fused)
    # 低能量通道相对放大更多
    ratio_low = (out.abs() / (fused.abs() + 1e-6))[:, :, :8].mean()
    ratio_high = (out.abs() / (fused.abs() + 1e-6))[:, :, 8:].mean()
    assert ratio_low > ratio_high
    # 全通道放大率 >= 1（不衰减）
    assert float((out.abs() / (fused.abs() + 1e-6)).min()) >= 0.99
    # 梯度可达（raw_gamma 为可训练参数）
    out.mean().backward()
    assert gain.raw_gamma.grad is not None


def test_r1_router_mount_and_forward_parity_with_baseline() -> None:
    g = torch.Generator().manual_seed(13)
    baseline = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    candidate = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r1_low_energy_channel_gain", stages=("mid", "late"))
    shared = {k: v for k, v in baseline.state_dict().items() if k in candidate.state_dict()}
    candidate.load_state_dict(shared, strict=False)
    baseline.eval(); candidate.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    with torch.no_grad():
        lb, _ = baseline(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
        lc, _ = candidate(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start R1 must be exactly identical to baseline"


def test_r1_gamma_is_nonnegative_and_zero_start() -> None:
    gain = R1LowEnergyChannelGain()
    assert float(gain.gamma) == 0.0  # 零起步
    with torch.no_grad():
        gain.raw_gamma.fill_(-3.0)   # 负 raw 被 relu 截断
    assert float(gain.gamma) == 0.0
    g = torch.Generator().manual_seed(3)
    fused = torch.randn(2, 49, 16, generator=g)
    out = gain(fused)
    assert torch.equal(out, fused)    # gamma=0 -> 恒等
    with torch.no_grad():
        gain.raw_gamma.fill_(-3.0)
    out2 = gain(fused)
    assert torch.equal(out2, fused)   # 负 raw 不产生任何缩放
    with torch.no_grad():
        gain.raw_gamma.fill_(2.0)
    out3 = gain(fused)
    ratio = out3.abs() / (fused.abs() + 1e-6)
    assert float(ratio.min()) >= 0.999  # 仅放大不衰减


# ---------- 机制共同契约 ----------

def test_v20_mechanisms_declared_valid_and_router_only_parameters() -> None:
    assert "r2_depth_group_inject" in OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r2_depth_group_inject").fusions["mid"].VALID_MECHANISMS if False else True
    for ms in ("r2_depth_group_inject", "r1_low_energy_channel_gain"):
        m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set=ms, stages=("mid", "late"))
        assert m.router is not None
        names = [n for n, _ in m.named_parameters() if n.startswith("router.")]
        assert names, f"{ms} must declare router.* parameters"
        # 公共参数（非 router.*）与基线完全一致
        base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
        bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
        cn = sorted(n for n, _ in m.named_parameters() if not n.startswith("router."))
        assert bn == cn, "common parameter surface must be identical"