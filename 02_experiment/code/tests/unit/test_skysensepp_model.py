"""Unit tests for the R24 SkySense++ S2 segmentation model family.

The backbone is 302.6M parameters, so every test builds it with random
weights only and runs a single small forward.  The real 1.2GB safetensors
load check is marked ``slow`` and skipped unless ``RUN_SKYSENSEPP_REAL=1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "vendor"))

from geotoken3path.models.skysensepp_seg import (
    Conv1x1SegmentationHead,
    SkySensePPImportError,
    SkySensePPSegmentationModel,
    _HF_CONFIG_BLACKLIST,
    build_backbone_config,
    build_skysensepp_model,
    default_safetensors_path,
    default_vendor_dir,
    load_skysensepp_weights,
    load_vendor_config,
)
from geotoken3path.data.skysensepp import (
    annotation_from_target,
    croma_dynamic_normalize_batch_r24,
    to_skysensepp_optical,
)


def _optical(batch: int, size: int, channels: int = 10) -> Tensor:
    """Backbone-ready pixel batch (default: the 10 SkySense++ S2 bands)."""
    return torch.randn(batch, channels, size, size, dtype=torch.float32)


def _optical12(batch: int, size: int) -> Tensor:
    """Raw SEN12TS loader batch with all 12 bands."""
    return _optical(batch, size, channels=12)


def _annotation(batch: int, size: int) -> Tensor:
    return torch.randint(0, 65, (batch, size, size), dtype=torch.long)


def _batch(batch: int = 2, size: int = 64) -> dict[str, Tensor]:
    return {
        "optical": _optical12(batch, size),
        "target": torch.randint(0, 11, (batch, size, size), dtype=torch.long),
        "valid_count": torch.tensor(batch, dtype=torch.int64),
    }


def test_head_contract_a_structure() -> None:
    """Contract 'a': 4 branches of 256 channels fused to num_classes."""
    torch.manual_seed(0)
    head = Conv1x1SegmentationHead(contract="a", num_classes=11, seed=0)
    assert len(head.branches) == 4
    for branch in head.branches:
        assert branch[0].out_channels == 256
        assert branch[0].in_channels == 1024
        assert branch[0].kernel_size == (1, 1)
        assert isinstance(branch[1], nn.ReLU)
    assert head.fuse.in_channels == 4 * 256
    assert head.fuse.out_channels == 11
    maps = [torch.randn(2, 1024, 16, 16) for _ in range(4)]
    out = head(maps)
    assert tuple(out.shape) == (2, 11, 16, 16)


def test_head_contract_b_structure() -> None:
    """Contract 'b': single reducer to the deepest executed map."""
    head = Conv1x1SegmentationHead(contract="b", num_classes=11, seed=0)
    assert isinstance(head.reducer, nn.Sequential)
    assert head.reducer[0].in_channels == 1024
    assert head.reducer[0].out_channels == 256
    assert isinstance(head.reducer[1], nn.ReLU)
    assert head.fuse.in_channels == 256
    assert head.fuse.out_channels == 11
    maps = [torch.randn(2, 1024, 16, 16) for _ in range(4)]
    out = head(maps)
    assert tuple(out.shape) == (2, 11, 16, 16)


def test_head_rejects_unknown_contract() -> None:
    with pytest.raises(ValueError):
        Conv1x1SegmentationHead(contract="c")


def test_head_seed_reproducibility() -> None:
    a = Conv1x1SegmentationHead(contract="a", seed=7)
    b = Conv1x1SegmentationHead(contract="a", seed=7)
    c = Conv1x1SegmentationHead(contract="a", seed=8)
    assert torch.equal(a.fuse.weight, b.fuse.weight)
    assert not torch.equal(a.fuse.weight, c.fuse.weight)
    for module_a, module_b in zip(a.branches, b.branches):
        assert torch.equal(module_a[0].weight, module_b[0].weight)
    assert torch.equal(a.fuse.bias, torch.zeros_like(a.fuse.bias))


def test_head_zero_bias_after_init() -> None:
    for contract in ("a", "b"):
        head = Conv1x1SegmentationHead(contract=contract, seed=0)
        for module in head.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                assert torch.equal(module.bias, torch.zeros_like(module.bias))


def test_head_forward_requires_enough_maps() -> None:
    with pytest.raises(ValueError):
        Conv1x1SegmentationHead(contract="a", seed=0)([torch.randn(2, 1024, 16, 16)] * 2)
    with pytest.raises(ValueError):
        Conv1x1SegmentationHead(contract="b", seed=0)([torch.randn(2, 1024, 16, 16)])


def _vendor_cfg() -> dict:
    return load_vendor_config()


def test_backbone_config_pins_out_indices_and_drop_path() -> None:
    cfg = _vendor_cfg()
    config = build_backbone_config(cfg, drop_path_rate=0.0)
    assert config.out_indices == [5, 11, 17, 23]
    assert config.merge_stage == 4
    assert config.use_attn is False
    assert config.img_size == 16
    assert config.patch_size == 4
    assert config.in_channels == 10
    assert config.embed_dims == 1024
    assert config.num_layers == 24
    assert config.num_heads == 16
    assert config.drop_path_rate == 0.0
    assert config.vocabulary_size == 64
    assert config.num_vocabulary_tokens == 65
    assert config.with_cls_token is False


def test_vendor_config_blacklist_covers_hf_bookkeeping() -> None:
    cfg = _vendor_cfg()
    for key in _HF_CONFIG_BLACKLIST:
        assert key in cfg or key in {"model_type", "output_attentions"}


def test_skysensepp_model_holds_backbone_and_head() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    assert isinstance(model, SkySensePPSegmentationModel)
    assert isinstance(model.backbone, nn.Module)
    assert isinstance(model.head, Conv1x1SegmentationHead)
    assert model.contract == "a"
    assert model.num_classes == 11
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    assert backbone_params > 300_000_000  # SkySense++ S2 ViT-L ~302.6M


def test_build_skysensepp_model_freezes_backbone_keeps_head_trainable() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.head.parameters())
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.startswith("head.") for name in trainable)


def test_forward_contract_a_shapes_at_64_pixels() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=_optical(2, 64), annotation=_annotation(2, 64))
    # Logits follow the input resolution (head emits H/4, model upsamples to H).
    assert tuple(out["logits"].shape) == (2, 11, 64, 64)
    assert len(out["feature_maps"]) == 4
    for feature in out["feature_maps"]:
        assert tuple(feature.shape) == (2, 1024, 16, 16)


def test_forward_contract_b_shapes_at_64_pixels() -> None:
    model = build_skysensepp_model(contract="b", safetensors_path="__missing__.safetensors")
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=_optical(2, 64), annotation=_annotation(2, 64))
    assert tuple(out["logits"].shape) == (2, 11, 64, 64)
    assert len(out["feature_maps"]) == 4


def test_forward_contract_b_equals_head_on_deepest_map() -> None:
    model = build_skysensepp_model(contract="b", safetensors_path="__missing__.safetensors")
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=_optical(2, 64), annotation=_annotation(2, 64))
    with torch.no_grad():
        expected = model.head(
            [out["feature_maps"][0], out["feature_maps"][1], out["feature_maps"][1], out["feature_maps"][1]],
            output_size=(64, 64),
        )
    assert torch.allclose(out["logits"], expected)


def test_forward_accepts_max_layer_kwarg() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=_optical(2, 64), annotation=_annotation(2, 64), max_layer=12)
    assert tuple(out["logits"].shape) == (2, 11, 64, 64)


def test_backbone_gradients_are_never_created_when_frozen() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    model.train()
    logits = model(pixel_values=_optical(2, 64), annotation=_annotation(2, 64))["logits"]
    logits.sum().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.head.parameters())


def test_backbone_is_deterministic_with_zero_drop_path() -> None:
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    model.eval()
    pixels, annotation = _optical(1, 64), _annotation(1, 64)
    with torch.no_grad():
        first = model(pixel_values=pixels, annotation=annotation)["logits"]
        second = model(pixel_values=pixels, annotation=annotation)["logits"]
    assert torch.equal(first, second)


def test_end_to_end_batch_through_normalization_and_model() -> None:
    """The R24 batch contract drives the backbone without index drift."""
    batch = _batch(16, size=64)
    normalized = croma_dynamic_normalize_batch_r24(batch)
    assert tuple(normalized["optical10"].shape) == (16, 10, 64, 64)
    annotation = annotation_from_target(normalized["target"])
    assert tuple(annotation.shape) == (16, 64, 64)

    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=normalized["optical10"], annotation=annotation)
    assert tuple(out["logits"].shape) == (16, 11, 64, 64)


def test_to_skysensepp_optical_feeds_patch_embed_directly() -> None:
    """The data adapter's 10-band output must feed the backbone as-is."""
    model = build_skysensepp_model(contract="a", safetensors_path="__missing__.safetensors")
    raw = _batch(2, 64)
    ten_band = to_skysensepp_optical(raw)
    assert tuple(ten_band.shape) == (2, 10, 64, 64)
    assert ten_band.shape[1] == model.backbone.config.in_channels
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=ten_band, annotation=_annotation(2, 64))
    assert tuple(out["logits"].shape) == (2, 11, 64, 64)


def test_vendor_dir_is_configurable() -> None:
    model = build_skysensepp_model(
        vendor_dir=default_vendor_dir(), safetensors_path="__missing__.safetensors",
    )
    assert model.backbone.config.in_channels == 10


def test_load_missing_checkpoint_raises() -> None:
    model = build_skysensepp_model(safetensors_path="__missing__.safetensors")
    with pytest.raises(SkySensePPImportError):
        load_skysensepp_weights(model, str(CODE_ROOT / "does_not_exist.safetensors"))


def test_dynamic_import_fails_on_missing_vendor_dir() -> None:
    from geotoken3path.models.skysensepp_seg import load_vendor_module

    with pytest.raises(SkySensePPImportError):
        load_vendor_module(str(CODE_ROOT / "vendor" / "no_such_dir"))


@pytest.mark.slow
def test_real_pretrained_weights_load_strictly() -> None:
    """Local-only: strict load of the 1.2GB checkpoint (0 missing / 0 unexpected)."""
    if os.environ.get("RUN_SKYSENSEPP_REAL") != "1":
        pytest.skip("RUN_SKYSENSEPP_REAL=1 required to touch the 1.2GB checkpoint")
    checkpoint = Path(default_safetensors_path())
    assert checkpoint.is_file(), f"real checkpoint missing at {checkpoint}"
    model = SkySensePPSegmentationModel(
        config_dict=load_vendor_config(), contract="a", num_classes=11, head_seed=0,
    )
    report = load_skysensepp_weights(model, str(checkpoint))
    assert report["missing"] == [], f"unexpected missing keys: {report['missing']}"
    assert report["unexpected"] == [], f"unexpected keys: {report['unexpected']}"
    model.eval()
    with torch.no_grad():
        logits = model(pixel_values=_optical(1, 64), annotation=_annotation(1, 64))["logits"]
    assert torch.isfinite(logits).all()
