"""Dependency-injected bridge for an audited CROMA radar--optical backbone.

The official CROMA forward exposes final tokens, not a separate spatially finer
SAR tensor.  ``CromaDepthTapAdapter`` therefore captures reproducible
modality-specific transformer-depth taps and stacks four SAR depth outputs as
an explicitly non-spatial depth group.  The bridge rejects guessed or
interpolated ``fine`` features before GeoToken-3Path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fusion import OpticalSarTokenModel


class CromaFeatureContractError(ValueError):
    """Raised when injected backbone features violate the frozen interface."""


class CromaDepthTapAdapter(nn.Module):
    """Expose official CROMA transformer-depth taps as stage token mappings.

    ``stage_taps`` maps ``optical`` and ``sar`` modalities to one module path
    per stage.  ``depth_group_taps`` maps each stage to four SAR module paths;
    stacking those outputs along axis 2 yields ``[B,N,4,D]`` without claiming
    a spatially finer grid.  Paths are resolved once and hooks only retain
    tensor references for the current forward.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        stages: Sequence[str],
        dim: int,
        stage_taps: Mapping[str, Mapping[str, str]],
        depth_group_taps: Mapping[str, Sequence[str]],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.stages = tuple(str(stage) for stage in stages)
        self.dim = int(dim)
        if not self.stages:
            raise ValueError("at least one CROMA feature stage is required")
        if set(stage_taps) != {"optical", "sar"}:
            raise ValueError("stage_taps must define optical and sar modalities")
        if set(depth_group_taps) != set(self.stages):
            raise ValueError("depth_group_taps must define every configured stage")
        self.stage_taps = {
            modality: {stage: str(paths[stage]) for stage in self.stages}
            for modality, paths in stage_taps.items()
        }
        self.depth_group_taps = {
            stage: tuple(str(path) for path in depth_group_taps[stage])
            for stage in self.stages
        }
        if any(len(paths) != 4 for paths in self.depth_group_taps.values()):
            raise ValueError("every SAR depth group must contain exactly four taps")
        self._captures: dict[str, Tensor] = {}
        self._joint_encodings: Tensor | None = None
        self._handles = []
        paths = {
            *self.stage_taps["optical"].values(),
            *self.stage_taps["sar"].values(),
            *(path for paths in self.depth_group_taps.values() for path in paths),
        }
        for path in sorted(paths):
            module = self._resolve_module(path)
            self._handles.append(module.register_forward_hook(self._make_hook(path)))

    def _resolve_module(self, path: str) -> nn.Module:
        try:
            return self.backbone.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            current: Any = self.backbone
            for part in path.split("."):
                if not hasattr(current, part):
                    raise CromaFeatureContractError(f"CROMA tap path not found: {path}") from exc
                current = getattr(current, part)
            if not isinstance(current, nn.Module):
                raise CromaFeatureContractError(f"CROMA tap path is not a module: {path}")
            return current

    def _make_hook(self, path: str):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise CromaFeatureContractError(f"CROMA tap {path} must return a tensor")
            self._captures[path] = output

        return capture

    def _tensor(self, path: str) -> Tensor:
        value = self._captures.get(path)
        if value is None:
            raise CromaFeatureContractError(f"CROMA tap was not executed: {path}")
        if value.ndim != 3 or value.shape[-1] != self.dim:
            raise CromaFeatureContractError(
                f"CROMA tap {path} must have shape [B,N,{self.dim}]"
            )
        return value

    def forward(
        self,
        *,
        SAR_images: Tensor | None = None,
        optical_images: Tensor | None = None,
    ) -> dict[str, dict[str, Tensor]]:
        self._captures.clear()
        self._joint_encodings = None
        backbone_outputs = self.backbone(
            SAR_images=SAR_images,
            optical_images=optical_images,
        )
        if isinstance(backbone_outputs, Mapping):
            joint = backbone_outputs.get("joint_encodings")
            if joint is not None:
                if not isinstance(joint, Tensor):
                    raise CromaFeatureContractError(
                        "CROMA joint_encodings must be a tensor when present"
                    )
                if joint.ndim != 3 or joint.shape[-1] != self.dim:
                    raise CromaFeatureContractError(
                        f"CROMA joint_encodings must have shape [B,N,{self.dim}]"
                    )
                # V18 treats the native joint state as a frozen VFM anchor.
                # Detach at capture so no later caller can accidentally open a
                # second trainable path into the CROMA cross encoder.
                self._joint_encodings = joint.detach()
        optical: dict[str, Tensor] = {}
        sar: dict[str, Tensor] = {}
        depth_group: dict[str, Tensor] = {}
        for stage in self.stages:
            optical[stage] = self._tensor(self.stage_taps["optical"][stage])
            sar[stage] = self._tensor(self.stage_taps["sar"][stage])
            ccpa_residual = getattr(self.backbone, "ccpa_residual", None)
            if isinstance(ccpa_residual, Tensor) and ccpa_residual.shape == optical[stage].shape:
                optical[stage] = optical[stage] + ccpa_residual
            rccr_optical_residual = getattr(self.backbone, "rccr_optical_residual", None)
            rccr_sar_residual = getattr(self.backbone, "rccr_sar_residual", None)
            if isinstance(rccr_optical_residual, Tensor) and rccr_optical_residual.shape == optical[stage].shape:
                optical[stage] = optical[stage] + rccr_optical_residual
            if isinstance(rccr_sar_residual, Tensor) and rccr_sar_residual.shape == sar[stage].shape:
                sar[stage] = sar[stage] + rccr_sar_residual
            for prefix in ("ocap", "dcp", "mrta", "ccg", "fcp"):
                optical_residual = getattr(self.backbone, f"{prefix}_optical_residual", None)
                sar_residual = getattr(self.backbone, f"{prefix}_sar_residual", None)
                if isinstance(optical_residual, Tensor) and optical_residual.shape == optical[stage].shape:
                    optical[stage] = optical[stage] + optical_residual
                if isinstance(sar_residual, Tensor) and sar_residual.shape == sar[stage].shape:
                    sar[stage] = sar[stage] + sar_residual
            joint_readout = getattr(self.backbone, "gcra_joint_readout_residual", None)
            if stage == self.stages[-1] and isinstance(joint_readout, Tensor) and joint_readout.shape == optical[stage].shape:
                optical[stage] = optical[stage] + joint_readout
            operator_residual = getattr(self.backbone, "operator_sar_residual", None)
            if operator_residual is not None:
                if not isinstance(operator_residual, Tensor):
                    raise CromaFeatureContractError("operator SAR residual must be a tensor")
                if operator_residual.shape != sar[stage].shape:
                    raise CromaFeatureContractError("operator SAR residual shape does not match the CROMA SAR tap")
                if not torch.isfinite(operator_residual).all():
                    raise CromaFeatureContractError("operator SAR residual must be finite")
                if stage == self.stages[-1]:
                    sar[stage] = sar[stage] + operator_residual
            group = torch.stack(
                [self._tensor(path) for path in self.depth_group_taps[stage]], dim=2
            )
            if group.shape[:2] != sar[stage].shape[:2] or group.shape[-1] != self.dim:
                raise CromaFeatureContractError(
                    f"CROMA SAR depth group {stage} is not aligned with SAR tokens"
                )
            depth_group[stage] = group
        return {"optical": optical, "sar": sar, "sar_depth_group": depth_group}

    def take_joint_encodings(self) -> Tensor | None:
        """Return the current-forward native joint state exactly once.

        Clearing the retained reference prevents an autograd graph from one
        batch surviving into the next.  Existing three-output callers remain
        unchanged; JACK explicitly consumes this side-channel immediately.
        """

        value = self._joint_encodings
        self._joint_encodings = None
        return value

    def close(self) -> None:
        """Remove forward hooks when a short-lived audit wrapper is discarded."""

        self._joint_encodings = None
        self._captures.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "CromaDepthTapAdapter":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


class CromaBackboneBridge(nn.Module):
    def __init__(self, backbone: nn.Module, *, stages: Sequence[str], dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.stages = tuple(stages)
        self.dim = int(dim)
        self._joint_encodings: Tensor | None = None
        if not self.stages:
            raise ValueError("at least one CROMA feature stage is required")

    def _stage_mapping(self, value: Any, name: str) -> Mapping[str, Tensor]:
        if not isinstance(value, Mapping):
            raise CromaFeatureContractError(f"backbone {name} output must be a stage mapping")
        if set(value) != set(self.stages):
            raise CromaFeatureContractError(f"backbone {name} stages must exactly match {self.stages}")
        if any(not isinstance(value[stage], Tensor) for stage in self.stages):
            raise CromaFeatureContractError(f"backbone {name} stage values must be tensors")
        return value

    def forward(
        self,
        optical_image: Tensor,
        sar_image: Tensor,
    ) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor], Mapping[str, Tensor]]:
        self._joint_encodings = None
        if optical_image.ndim != 4 or optical_image.shape[1] != 12:
            raise CromaFeatureContractError("optical input must have shape [B,12,H,W]")
        if sar_image.ndim != 4 or sar_image.shape[1] != 2:
            raise CromaFeatureContractError("SAR input must have shape [B,2,H,W]")
        if optical_image.dtype != torch.float32 or sar_image.dtype != torch.float32:
            raise CromaFeatureContractError("CROMA inputs must be float32 after normalization")
        if optical_image.shape[0] != sar_image.shape[0] or optical_image.shape[-2:] != sar_image.shape[-2:]:
            raise CromaFeatureContractError("optical and SAR image batches/spatial sizes must match")
        outputs = self.backbone(SAR_images=sar_image, optical_images=optical_image)
        if not isinstance(outputs, Mapping):
            raise CromaFeatureContractError("backbone must return a mapping")
        required = {"optical", "sar", "sar_depth_group"}
        if set(outputs) != required:
            raise CromaFeatureContractError(
                "backbone output keys must be optical,sar,sar_depth_group"
            )
        optical = self._stage_mapping(outputs["optical"], "optical")
        sar = self._stage_mapping(outputs["sar"], "sar")
        depth_group = self._stage_mapping(outputs["sar_depth_group"], "sar_depth_group")
        take_joint = getattr(self.backbone, "take_joint_encodings", None)
        joint = take_joint() if callable(take_joint) else None
        batch = optical_image.shape[0]
        for stage in self.stages:
            optical_stage = optical[stage]
            sar_stage = sar[stage]
            depth_stage = depth_group[stage]
            if optical_stage.ndim != 3 or optical_stage.shape != sar_stage.shape:
                raise CromaFeatureContractError(f"{stage} optical/SAR tokens must share [B,N,D]")
            if optical_stage.shape[0] != batch or optical_stage.shape[-1] != self.dim:
                raise CromaFeatureContractError(f"{stage} token batch/dimension mismatch")
            if depth_stage.shape != (*optical_stage.shape[:2], 4, self.dim):
                raise CromaFeatureContractError(
                    f"{stage} SAR depth group must have shape [B,N,4,D]"
                )
        if joint is not None:
            reference = optical[self.stages[-1]]
            if not isinstance(joint, Tensor) or joint.shape != reference.shape:
                raise CromaFeatureContractError(
                    "native CROMA joint encodings must align with the final optical/SAR token grid"
                )
            if not torch.isfinite(joint).all():
                raise CromaFeatureContractError("native CROMA joint encodings must be finite")
            if not joint.is_floating_point():
                raise CromaFeatureContractError("native CROMA joint encodings must be floating point")
            if joint.device != reference.device:
                raise CromaFeatureContractError(
                    "native CROMA joint encodings must share device with the final taps"
                )
            # CROMA's native cross encoder can remain FP32 while retained taps
            # are autocast to FP16. Both are valid floating representations;
            # align J to the receiver precision explicitly at this boundary.
            self._joint_encodings = joint.to(dtype=reference.dtype)
        return optical, sar, depth_group

    def take_joint_encodings(self) -> Tensor | None:
        """Consume and clear the native joint state retained for this batch."""

        value = self._joint_encodings
        self._joint_encodings = None
        return value

    def close(self) -> None:
        """Forward an idempotent hook teardown to a tapped CROMA adapter."""

        self._joint_encodings = None
        close = getattr(self.backbone, "close", None)
        if callable(close):
            close()


class CromaGeoTokenSegmentation(nn.Module):
    """Raw-image VFM segmentation path shared across mechanism rows."""

    def __init__(self, bridge: CromaBackboneBridge, token_model: OpticalSarTokenModel) -> None:
        super().__init__()
        self.bridge = bridge
        self.token_model = token_model

    def forward(
        self,
        optical_image: Tensor,
        sar_image: Tensor,
        *,
        return_aux: bool = False,
        d2_intervention: Mapping[str, Any] | None = None,
        d3_intervention: Mapping[str, Any] | None = None,
    ):
        optical, sar, depth_group = self.bridge(optical_image, sar_image)
        # Consume the joint encodings immediately so no batch graph can remain
        # held by the bridge; the native joint field is not used downstream.
        joint = self.bridge.take_joint_encodings()
        physical_groups = None
        return self.token_model(
            optical,
            sar,
            joint=joint,
            depth_group=depth_group,
            physical_groups=physical_groups,
            output_size=tuple(optical_image.shape[-2:]),
            return_aux=return_aux,
        )

    def close(self) -> None:
        """Release CROMA forward hooks owned by the raw-image model."""

        self.bridge.close()

    def __enter__(self) -> "CromaGeoTokenSegmentation":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


_LAYER_RE = re.compile(r"^(?:s1_encoder|s2_encoder)\.transformer\.layers\.(\d+)(?:\.|$)")


def _tap_connected_parameter(raw_name: str) -> bool:
    """Return whether a raw CROMA parameter can affect a retained feature tap.

    The bridge consumes every S1 layer through layer 5 for the SAR depth group,
    and S2 layers 0--5 for the optical mid/late taps.  S2 layers 6--11,
    GAP-FFNs and the cross encoder are not consumed by the bridge output and
    remain frozen to avoid training dead computation.
    """

    if raw_name.startswith("s1_encoder."):
        return True
    if not raw_name.startswith("s2_encoder."):
        return False
    match = _LAYER_RE.match(raw_name)
    if match is None:
        # Patch embedding and other S2 stem parameters feed all retained taps.
        return "transformer.layers." not in raw_name
    return int(match.group(1)) <= 5


def freeze_backbone_for_peft(
    model: CromaGeoTokenSegmentation,
    *,
    policy: str = "frozen",
) -> dict[str, bool]:
    """Apply an explicit CROMA trainability policy and return the parameter mask.

    ``frozen`` preserves the historical PEFT-only baseline.  ``tap_connected``
    unfreezes only the S1 path and the S2 stem/layers feeding retained taps;
    it never unfreezes the unused cross encoder or post-encoder GAP heads.
    """

    selected = str(policy).strip().casefold()
    if selected not in {"frozen", "tap_connected"}:
        raise ValueError("unsupported CROMA trainability policy")
    wrapped_backbone = model.bridge.backbone.backbone
    wrapped_trunk_parameter_ids = {
        id(parameter)
        for child_name in ("s1_encoder", "s2_encoder", "GAP_FFN_s1", "GAP_FFN_s2", "cross_encoder")
        for parameter in getattr(wrapped_backbone, child_name, nn.Identity()).parameters()
    }
    adapter_parameter_ids = {
        id(parameter)
        for parameter in wrapped_backbone.parameters()
        if id(parameter) not in wrapped_trunk_parameter_ids
    }
    for name, parameter in model.bridge.backbone.named_parameters():
        parameter.requires_grad_(False)
        if selected == "tap_connected":
            raw_name = name.removeprefix("backbone.")
            # Two-zone cleanup 2026-09-02: rejected pre-CROMA adapter key
            # exemptions were removed; only the tap-connected policy remains.
            parameter.requires_grad_(_tap_connected_parameter(raw_name))
    if selected == "tap_connected" and not any(
        parameter.requires_grad for parameter in model.bridge.backbone.parameters()
    ):
        raise ValueError("tap_connected policy produced no trainable CROMA parameters")
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}
