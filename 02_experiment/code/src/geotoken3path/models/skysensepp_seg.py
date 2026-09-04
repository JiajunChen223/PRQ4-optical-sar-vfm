"""R24 SkySense++ S2 backbone segmentation model family.

The model holds the pretrained SkySense++ semantic-enhanced ViT-L backbone
(S2 modality, 24 pre-LN layers, embed 1024, 16 heads, patch 4, 10 input
bands) directly and stacks a small 1x1 segmentation head on its hierarchical
feature maps.  There is no CROMA-style three-hop bridge: the R24 ICE contract
"a" reads all four backbone output levels, while contract "b" reads only the
deepest executed level (``feature_maps[1]`` of the official out_indices
[5,11,17,23] grid).

The vendor implementation is HuggingFace-style (``import transformers``) and
lives in ``code/vendor/skysensepp_s2_full``, which must never be imported at
module scope on a machine whose ``sys.path`` does not already expose it.  All
vendor access therefore happens lazily through ``importlib``: the vendor
directory is inserted on ``sys.path`` as the parent of the package and the
module is imported under its real package name, which keeps the vendor's
relative imports working.  Every vendor path is overridable through an
environment variable for cloud hosts that relocate the artifact tree.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Official pretrained S2 backbone output indices: one feature map per stage
# boundary.  All stages always run; the head contract selects which maps to read.
SKYSENSEPP_OUT_INDICES = (5, 11, 17, 23)
# HuggingFace-only bookkeeping keys that must never reach the vendor config
# constructor (the vendor config class forwards everything else to its own
# signature and stores the remainder in config attributes).
_HF_CONFIG_BLACKLIST = frozenset(
    {
        "_name_or_path",
        "architectures",
        "auto_map",
        "chunk_size_feed_forward",
        "custom_pipelines",
        "dtype",
        "id2label",
        "is_encoder_decoder",
        "label2id",
        "model_type",
        "output_attentions",
        "output_hidden_states",
        "problem_type",
        "return_dict",
        "transformers_version",
    }
)

_ENV_VENDOR_DIR = "GEOTOKEN3PATH_SKYSENSEPP_VENDOR_DIR"
_ENV_SAFETENSORS = "GEOTOKEN3PATH_SKYSENSEPP_SAFETENSORS"
_ENV_CONFIG_FILENAME = "GEOTOKEN3PATH_SKYSENSEPP_CONFIG_FILENAME"

DEFAULT_SKYSENSEPP_VENDOR_DIR = str(
    Path(__file__).resolve().parents[3] / "vendor" / "skysensepp_s2_full"
)
DEFAULT_SKYSENSEPP_SAFETENSORS = str(
    Path(__file__).resolve().parents[3] / "vendor" / "skysensepp_s2_full" / "model.safetensors"
)


def default_vendor_dir() -> str:
    """Resolve the vendor directory from the environment or the local default."""
    return str(Path(os.environ.get(_ENV_VENDOR_DIR, DEFAULT_SKYSENSEPP_VENDOR_DIR)).resolve())


def default_safetensors_path() -> str:
    """Resolve the checkpoint path from the environment or the local default."""
    return str(Path(os.environ.get(_ENV_SAFETENSORS, DEFAULT_SKYSENSEPP_SAFETENSORS)).resolve())


def _config_filename() -> str:
    return os.environ.get(_ENV_CONFIG_FILENAME, "config.json")


class SkySensePPImportError(RuntimeError):
    """Raised when the SkySense++ vendor modeling package cannot be loaded."""


def load_vendor_module(vendor_dir: str | None = None) -> Any:
    """Dynamically import the vendor modeling module as a package member.

    The directory containing ``skysensepp_s2_full`` is inserted on
    ``sys.path`` and the module is imported by its package-qualified name
    (``<package>.modeling_skysensepp_vit_msl``), preserving the relative
    imports used inside the vendor sources.  Python caches module identity,
    so repeated calls share a single vendor import.
    """
    vendor_path = Path(vendor_dir or default_vendor_dir()).resolve()
    if not vendor_path.is_dir():
        raise SkySensePPImportError(f"skysensepp vendor directory not found: {vendor_path}")
    import sys

    parent = str(vendor_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    package = vendor_path.name
    full_name = f"{package}.modeling_skysensepp_vit_msl"
    try:
        return importlib.import_module(full_name)
    except Exception as exc:  # pragma: no cover - vendor import failure surface
        raise SkySensePPImportError(
            f"cannot import skysensepp vendor module {full_name!r} from {vendor_path}"
        ) from exc


def load_vendor_config(vendor_dir: str | None = None) -> dict[str, Any]:
    """Read the vendor ``config.json`` as a plain mapping."""
    vendor_path = Path(vendor_dir or default_vendor_dir()).resolve()
    config_file = vendor_path / _config_filename()
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SkySensePPImportError(f"cannot read skysensepp vendor config {config_file}") from exc
    if not isinstance(loaded, dict):
        raise SkySensePPImportError(f"skysensepp vendor config {config_file} is not a mapping")
    return loaded


def _vendor_constructors(vendor_dir: str | None = None) -> tuple[Any, Any]:
    """Return (config class, model class) from the vendor package."""
    modeling_module = load_vendor_module(vendor_dir)
    package = Path(vendor_dir or default_vendor_dir()).resolve().name
    config_module = importlib.import_module(f"{package}.configuration_skysensepp")
    config_type = getattr(config_module, "SkySensePlusPlusViTMSLConfig", None)
    model_type = getattr(modeling_module, "SkySensePlusPlusViTMSLModel", None)
    if config_type is None or model_type is None:
        raise SkySensePPImportError("skysensepp vendor package misses SkySensePlusPlusViTMSL classes")
    return config_type, model_type


def build_backbone_config(
    config_dict: dict[str, Any],
    *,
    drop_path_rate: float | None = None,
    vendor_dir: str | None = None,
) -> Any:
    """Build a vendor config object from a plain mapping.

    HuggingFace bookkeeping keys are stripped, ``out_indices`` is pinned to
    the official S2 stage boundaries so the backbone always computes the four
    hierarchical feature maps, and ``drop_path_rate`` can be forced to zero
    (R5 gradient certificates need determinism).
    """
    filtered = {key: value for key, value in config_dict.items() if key not in _HF_CONFIG_BLACKLIST}
    filtered["out_indices"] = list(SKYSENSEPP_OUT_INDICES)
    if drop_path_rate is not None:
        filtered["drop_path_rate"] = float(drop_path_rate)
    config_type, _ = _vendor_constructors(vendor_dir)
    return config_type(**filtered)


class Conv1x1SegmentationHead(nn.Module):
    """1x1-convolution decoder over the SkySense++ S2 feature maps.

    contract "a": one ``Conv2d(1024, 256, 1) + ReLU`` branch per backbone
    output level; the four maps are concatenated along the channel axis and a
    final ``Conv2d(4 * 256, num_classes, 1)`` emits logits.

    contract "b": ``Conv2d(1024, 256, 1) + ReLU`` then
    ``Conv2d(256, num_classes, 1)`` over the deepest executed feature map.

    The head never upsamples internally: on the official 16x16 pretraining
    grid it emits logits at 16x16.  ``forward`` accepts an optional
    ``output_size`` and bilinearly upsizes the raw head output to that grid
    (the R24 model layer passes the input pixel resolution, so SEN12TS
    120x120 crops produce 120x120 logits that the segmentation losses and
    metrics can consume directly).  Convolutions use kaiming init with zero
    biases, reproducible through ``seed``.
    """

    def __init__(self, contract: str = "a", num_classes: int = 11, seed: int = 0) -> None:
        super().__init__()
        if contract not in {"a", "b"}:
            raise ValueError("skysensepp head contract must be 'a' or 'b'")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("head seed must be a nonnegative integer")
        self.contract = contract
        self.num_classes = int(num_classes)
        if contract == "a":
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(inplace=True))
                    for _ in SKYSENSEPP_OUT_INDICES
                ]
            )
            self.fuse = nn.Conv2d(256 * len(SKYSENSEPP_OUT_INDICES), self.num_classes, kernel_size=1)
        else:
            self.reducer = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.fuse = nn.Conv2d(256, self.num_classes, kernel_size=1)

        generator = torch.Generator().manual_seed(int(seed))

        def _reinit(module: nn.Module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_reinit)

    def forward(
        self,
        feature_maps: list[Tensor] | tuple[Tensor, ...],
        *,
        output_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Score the backbone feature maps and upsample logits to the input grid.

        The head itself never upsamples: on the official 16x16 pretraining
        grid it emits logits at 16x16, and the segmentation losses and metrics
        require logits at the input resolution (SEN12TS delivers 120x120, the
        backbone executes at 30x30).  The bilinear ``F.interpolate`` follows
        the audited CROMA decoder precedent (``models/decoder.py``) and is
        resolution-agnostic: an input of HxW always produces logits of HxW, so
        any input size is safe.  The optional ``output_size`` override lets
        callers score the raw feature resolution explicitly.
        """
        logits = self._forward_lowres(feature_maps)
        if output_size is None:
            return logits
        if (
            isinstance(output_size, bool)
            or not isinstance(output_size, (tuple, list))
            or len(output_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in output_size)
        ):
            raise ValueError("output_size must be None or a pair of positive integers")
        return F.interpolate(
            logits,
            size=(int(output_size[0]), int(output_size[1])),
            mode="bilinear",
            align_corners=False,
        )

    def _forward_lowres(self, feature_maps: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        maps = list(feature_maps)
        if self.contract == "a":
            if len(maps) < len(SKYSENSEPP_OUT_INDICES):
                raise ValueError("contract 'a' head requires all four backbone feature maps")
            branches = [branch(feature) for branch, feature in zip(self.branches, maps)]
            return self.fuse(torch.cat(branches, dim=1))
        if len(maps) < 2:
            raise ValueError("contract 'b' head requires the deepest backbone feature map")
        deepest = maps[1]
        if deepest.ndim != 4 or deepest.shape[1] != 1024:
            raise ValueError("skysensepp feature map must be [B,1024,H,W]")
        return self.fuse(self.reducer(deepest))


class SkySensePPSegmentationModel(nn.Module):
    """Segmentation model that owns the SkySense++ S2 backbone directly.

    The backbone is a plain child module (no CROMA bridge, no token
    mechanism).  Pretrained weights are loaded strictly and frozen; only the
    1x1 head stays trainable.  ``forward`` accepts the SEN12TS pixel /
    annotation pair and returns dict-style outputs.
    """

    def __init__(
        self,
        *,
        vendor_dir: str | None = None,
        config_dict: dict[str, Any] | None = None,
        contract: str = "a",
        num_classes: int = 11,
        head_seed: int = 0,
    ) -> None:
        super().__init__()
        if config_dict is None:
            config_dict = load_vendor_config(vendor_dir)
        if not isinstance(config_dict, dict):
            raise TypeError("skysensepp config_dict must be a mapping")
        _, model_type = _vendor_constructors(vendor_dir)
        config = build_backbone_config(config_dict, vendor_dir=vendor_dir)
        self.backbone = model_type(config)
        self.contract = str(contract)
        self.num_classes = int(num_classes)
        self.head = Conv1x1SegmentationHead(
            contract=self.contract, num_classes=self.num_classes, seed=int(head_seed),
        )

    def freeze_backbone(self) -> None:
        """Freeze every backbone parameter, including embed/pos/vocabulary."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        pixel_values: Tensor,
        annotation: Tensor,
        *,
        max_layer: int | None = None,
    ) -> dict[str, Any]:
        """Run the backbone on SEN12TS optical [B,10,H,W] plus annotation.

        The backbone always executes all 24 layers (``max_layer`` is accepted
        for interface compatibility with the ICE execution layer and is not
        applied here).  Returns logits [B,num_classes,H,W] at the input pixel
        resolution (the head's 1x1 convolution emits H/4 x W/4, then the
        logits are bilinearly upsized -- see ``Conv1x1SegmentationHead``) and
        the four hierarchical feature maps at H/4 x W/4.
        """
        del max_layer  # reserved: this module always runs the full backbone
        feature_maps = self.backbone(
            pixel_values=pixel_values,
            annotation=annotation,
            return_dict=False,
        )
        if isinstance(feature_maps, tuple) and len(feature_maps) == 1 and isinstance(feature_maps[0], tuple):
            feature_maps = feature_maps[0]
        if feature_maps is None or (isinstance(feature_maps, tuple) and len(feature_maps) == 0):
            raise ValueError("skysensepp backbone returned no feature maps")
        first_map = feature_maps[0]
        if not isinstance(first_map, Tensor) or first_map.ndim != 4:
            raise ValueError("skysensepp backbone feature maps must be [B,1024,H,W] tensors")
        if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B,C,H,W]")
        logits = self.head(
            list(feature_maps),
            output_size=(int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
        )
        return {
            "logits": logits,
            "feature_maps": tuple(feature_maps),
        }


def load_skysensepp_weights(model: SkySensePPSegmentationModel, safetensors_path: str) -> dict[str, Any]:
    """Strictly load a safetensors checkpoint into the backbone parameters.

    Only the backbone is loaded (a pretrained SkySense++ checkpoint never
    contains segmentation-head keys).  The report returns the exact
    ``missing`` and ``unexpected`` key lists so callers can enforce a strict
    0/0 expectation without losing mismatch diagnostics.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - environment surface
        raise SkySensePPImportError("safetensors is required to load skysensepp weights") from exc
    checkpoint = Path(safetensors_path)
    if not checkpoint.is_file():
        raise SkySensePPImportError(f"skysensepp checkpoint not found: {checkpoint}")
    state = load_file(str(checkpoint))
    if not isinstance(state, dict) or any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in state.items()):
        raise SkySensePPImportError("skysensepp checkpoint does not map string keys to tensors")
    incompatible = model.backbone.load_state_dict(state, strict=False)
    return {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


def build_skysensepp_model(
    *,
    vendor_dir: str | None = None,
    safetensors_path: str | None = None,
    contract: str = "a",
    num_classes: int = 11,
    seed: int = 0,
    drop_path_rate: float = 0.0,
) -> SkySensePPSegmentationModel:
    """Build the full R24 model: vendor config + weights + frozen backbone.

    ``drop_path_rate`` is overridden to 0.0 so ICE gradient certificates are
    deterministic.  When ``safetensors_path`` names an existing checkpoint the
    weights are loaded strictly (0 missing / 0 unexpected required); otherwise
    the backbone stays randomly initialized for unit tests and smoke runs.
    The backbone is frozen in either case; only the head remains trainable.
    """
    model = SkySensePPSegmentationModel(
        vendor_dir=vendor_dir,
        config_dict=load_vendor_config(vendor_dir),
        contract=contract,
        num_classes=int(num_classes),
        head_seed=int(seed),
    )
    checkpoint = Path(safetensors_path) if safetensors_path is not None else Path(default_safetensors_path())
    if checkpoint.is_file():
        report = load_skysensepp_weights(model, str(checkpoint))
        if report["missing"] or report["unexpected"]:
            raise SkySensePPImportError(
                "skysensepp checkpoint did not load strictly: "
                f"{len(report['missing'])} missing, {len(report['unexpected'])} unexpected"
            )
    model.freeze_backbone()
    return model
