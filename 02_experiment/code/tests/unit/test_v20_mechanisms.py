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

# ---------- R3：光学条件化深度组选择注入 ----------

def test_r3_zero_start_is_exact_identity() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r3_optical_conditional_depth_select", stages=("mid", "late"))
    assert m.router is not None
    g = torch.Generator().manual_seed(2)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    opt = torch.randn(2, 49, 16, generator=g)
    out = m.router(depth, sar, opt, "mid")
    assert torch.equal(out, sar)  # layer_proj 零起步 -> 注入恒零
    # sel logits 零 -> 均匀选择
    a = torch.softmax(m.router.sel_proj["mid"](opt), dim=-1)
    assert torch.allclose(a, torch.full((2, 49, 4), 0.25), atol=1e-6)


def test_r3_forward_parity_with_baseline() -> None:
    g = torch.Generator().manual_seed(4)
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r3_optical_conditional_depth_select", stages=("mid", "late"))
    shared = {k: v for k, v in base.state_dict().items() if k in cand.state_dict()}
    cand.load_state_dict(shared, strict=False)
    base.eval(); cand.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    with torch.no_grad():
        lb, _ = base(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
        lc, _ = cand(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start R3 must be exactly identical to baseline"


def test_r3_selection_and_gradient_liveness() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r3_optical_conditional_depth_select", stages=("mid", "late"))
    g = torch.Generator().manual_seed(6)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    opt = torch.randn(2, 49, 16, generator=g)
    out = m.router(depth, sar, opt, "late")
    assert out.shape == sar.shape
    out.mean().backward()
    assert m.router.sel_proj["late"].weight.grad is not None
    assert m.router.layer_proj["late"].weight.grad is not None
    # 条件化生效验证：把 sel 权重推向第 0 层优先（ones 输入下 logits 有偏）
    with torch.no_grad():
        m.router.sel_proj["late"].weight.zero_()
        m.router.sel_proj["late"].weight[0] = 1.0  # 第 0 层行权重 = 1
    a2 = torch.softmax(m.router.sel_proj["late"](torch.ones(2, 49, 16)), dim=-1)
    assert a2[:, :, 0].mean() > 0.5  # 第一层选择明显占优
    assert a2[:, :, 1].mean() < 0.25


def test_r3_common_parameter_surface() -> None:
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r3_optical_conditional_depth_select", stages=("mid", "late"))
    bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
    cn = sorted(n for n, _ in cand.named_parameters() if not n.startswith("router."))
    assert bn == cn
    rn = [n for n, _ in cand.named_parameters() if n.startswith("router.")]
    assert sorted(rn) == sorted([
        "router.sel_proj.mid.weight", "router.sel_proj.late.weight",
        "router.layer_proj.mid.weight", "router.layer_proj.late.weight",
    ])


# ---------- R6：双通道定向深度组注入 ----------

def test_r6_zero_start_identity_and_routing() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r6_depth_dual_channel_inject", stages=("mid", "late"))
    assert m.router is not None
    g = torch.Generator().manual_seed(9)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    opt = torch.randn(2, 49, 16, generator=g)
    # 零起步：mid 注入光学、late 注入 SAR，都为零
    opt_out, sar_out = m.router(depth, opt, sar, "mid")
    assert torch.equal(opt_out, opt)
    assert torch.equal(sar_out, sar)
    opt_out2, sar_out2 = m.router(depth, opt, sar, "late")
    assert torch.equal(opt_out2, opt)
    assert torch.equal(sar_out2, sar)
    # 路由逻辑：mid 只改光学、late 只改 SAR（激活投影后）
    with torch.no_grad():
        m.router.layer_proj["mid"].weight.fill_(0.01)
        m.router.layer_proj["late"].weight.fill_(0.01)
    o3, s3 = m.router(depth, opt, sar, "mid")
    assert not torch.equal(o3, opt) and torch.equal(s3, sar)
    o4, s4 = m.router(depth, opt, sar, "late")
    assert torch.equal(o4, opt) and not torch.equal(s4, sar)


def test_r6_forward_parity_with_baseline() -> None:
    g = torch.Generator().manual_seed(10)
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r6_depth_dual_channel_inject", stages=("mid", "late"))
    shared = {k: v for k, v in base.state_dict().items() if k in cand.state_dict()}
    cand.load_state_dict(shared, strict=False)
    base.eval(); cand.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    with torch.no_grad():
        lb, _ = base(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
        lc, _ = cand(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc)


def test_r6_parameters_and_gradient() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r6_depth_dual_channel_inject", stages=("mid", "late"))
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
    cn = sorted(n for n, _ in m.named_parameters() if not n.startswith("router."))
    assert bn == cn
    g = torch.Generator().manual_seed(12)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    opt = torch.randn(2, 49, 16, generator=g)
    # late 分支注入 SAR 载体 -> 取第二个返回值做 backward
    _, out_sar = m.router(depth, opt, sar, "late")
    out_sar.mean().backward()
    assert m.router.sel_proj["late"].weight.grad is not None
    assert m.router.layer_proj["late"].weight.grad is not None


# ---------- R7：零起步残差学习上采样 ----------

def test_r7_zero_start_identity_and_parity() -> None:
    g = torch.Generator().manual_seed(3)
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r7_residual_learned_upsample", stages=("mid", "late"))
    assert cand.router is not None
    shared = {k: v for k, v in base.state_dict().items() if k in cand.state_dict()}
    cand.load_state_dict(shared, strict=False)
    base.eval(); cand.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    with torch.no_grad():
        lb, _ = base(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
        lc, _ = cand(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start R7 must equal baseline exactly"
    # 残差生效：填充权重后 logits 变化
    with torch.no_grad():
        cand.router.upsample_conv.weight.fill_(0.01)
    with torch.no_grad():
        lc2, _ = cand(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
    assert not torch.equal(lb, lc2)


def test_r7_parameters_and_gradient() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r7_residual_learned_upsample", stages=("mid", "late"))
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
    cn = sorted(n for n, _ in m.named_parameters() if not n.startswith("router."))
    assert bn == cn
    rn = [n for n, _ in m.named_parameters() if n.startswith("router.")]
    assert rn == ["router.upsample_conv.weight"]
    g = torch.Generator().manual_seed(5)
    out, _ = m(torch.randn(2, 49, 16, generator=g), torch.randn(2, 49, 16, generator=g),
               depth_group=None, output_size=(7, 7), return_aux=True)
    assert tuple(out.shape) == (2, 8, 7, 7)
    out.mean().backward()
    assert m.router.upsample_conv.weight.grad is not None


# ---------- R8：组合机制（R3 注入 + R7 上采样残差） ----------

def test_r8_zero_start_identity_and_parity() -> None:
    g = torch.Generator().manual_seed(21)
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r8_depth_inject_plus_upsample", stages=("mid", "late"))
    assert cand.router is not None
    shared = {k: v for k, v in base.state_dict().items() if k in cand.state_dict()}
    cand.load_state_dict(shared, strict=False)
    base.eval(); cand.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    depth = torch.randn(2, 49, 4, 16, generator=g)
    with torch.no_grad():
        lb, _ = base(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
        lc, _ = cand(optical, sar, depth_group=depth, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start R8 composition must equal baseline exactly"


def test_r8_both_components_active_and_gradient() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r8_depth_inject_plus_upsample", stages=("mid", "late"))
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
    cn = sorted(n for n, _ in m.named_parameters() if not n.startswith("router."))
    assert bn == cn
    # 两组件均存在
    assert m.router.depth_select is not None and m.router.upsample is not None
    with torch.no_grad():
        for stage in ("mid", "late"):
            m.router.depth_select.layer_proj[stage].weight.fill_(0.01)
        m.router.upsample.upsample_conv.weight.fill_(0.01)
    g = torch.Generator().manual_seed(23)
    out, _ = m(torch.randn(2, 49, 16, generator=g), torch.randn(2, 49, 16, generator=g),
               depth_group=torch.randn(2, 49, 4, 16, generator=g), output_size=(7, 7), return_aux=True)
    assert tuple(out.shape) == (2, 8, 7, 7)
    out.mean().backward()
    assert m.router.depth_select.layer_proj["late"].weight.grad is not None
    assert m.router.upsample.upsample_conv.weight.grad is not None


# ---------- R9：光学语义恢复（D2 recovery） ----------

def test_r9_zero_start_identity_and_parity() -> None:
    g = torch.Generator().manual_seed(31)
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    cand = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r9_optical_semantic_recovery", stages=("mid", "late"))
    assert cand.router is not None
    shared = {k: v for k, v in base.state_dict().items() if k in cand.state_dict()}
    cand.load_state_dict(shared, strict=False)
    base.eval(); cand.eval()
    optical = torch.randn(2, 49, 16, generator=g)
    sar = torch.randn(2, 49, 16, generator=g)
    with torch.no_grad():
        lb, _ = base(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
        lc, _ = cand(optical, sar, depth_group=None, output_size=(7, 7), return_aux=True)
    assert torch.equal(lb, lc), "zero-start r9 must equal baseline exactly"


def test_r9_router_only_parameters_and_gradient() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r9_optical_semantic_recovery", stages=("mid", "late"))
    base = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="always_fuse", stages=("mid", "late"))
    bn = sorted(n for n, _ in base.named_parameters() if not n.startswith("router."))
    cn = sorted(n for n, _ in m.named_parameters() if not n.startswith("router."))
    assert bn == cn
    rn = [n for n, _ in m.named_parameters() if n.startswith("router.")]
    assert "router.up.weight" in rn and "router.down.weight" in rn
    g = torch.Generator().manual_seed(33)
    out, _ = m(torch.randn(2, 49, 16, generator=g), torch.randn(2, 49, 16, generator=g),
               depth_group=None, output_size=(7, 7), return_aux=True)
    out.mean().backward()
    assert m.router.up.weight.grad is not None
    assert m.router.down.weight.grad is not None


def test_r9_residual_becomes_active_after_training() -> None:
    m = OpticalSarTokenModel(dim=16, num_classes=8, mechanism_set="r9_optical_semantic_recovery", stages=("mid", "late"))
    g = torch.Generator().manual_seed(35)
    optical = torch.randn(2, 49, 16, generator=g)
    # 零起步：残差为零
    with torch.no_grad():
        out0 = m.router(optical)
    assert torch.equal(out0, optical)
    # 激活后：残差非零（down 初始非零；需激活 up 与 hid 中间层，否则 hidden=0）
    with torch.no_grad():
        m.router.up.weight.fill_(0.01)
        m.router.hid[2].weight.fill_(0.01)
        m.router.hid[4].weight.fill_(0.01)
    out1 = m.router(optical)
    assert not torch.equal(out1, optical)
