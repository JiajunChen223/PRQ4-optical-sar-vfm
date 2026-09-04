"""Interface-certified exact execution backend for audited CROMA models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from einops import rearrange
import torch
from torch import Tensor, nn

from .contracts import CromaExecutionContractError
from .croma_plan import CromaExecutionPlan


_UNSAFE_DYNAMIC_ATTRS = (
    "ccpa_residual",
    "rccr_optical_residual",
    "rccr_sar_residual",
    "ocap_optical_residual",
    "ocap_sar_residual",
    "dcp_optical_residual",
    "dcp_sar_residual",
    "mrta_optical_residual",
    "mrta_sar_residual",
    "ccg_optical_residual",
    "ccg_sar_residual",
    "fcp_optical_residual",
    "fcp_sar_residual",
    "gcra_joint_readout_residual",
    "operator_sar_residual",
)


class InterfaceCertifiedCromaExecutor:
    """Execute only CROMA nodes certified necessary by a compiled tap plan.

    The executor intentionally owns no modules or parameters.  It invokes the
    exact module objects from the audited backbone so existing FFN forward hooks
    retain their original semantics and the model state-dict is unchanged.
    """

    def __init__(self, plan: CromaExecutionPlan) -> None:
        if not isinstance(plan, CromaExecutionPlan):
            raise TypeError("plan must be a CromaExecutionPlan")
        self.plan = plan

    @staticmethod
    def _get_submodule(backbone: nn.Module, path: str) -> nn.Module:
        try:
            module = backbone.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise CromaExecutionContractError(
                f"audited backbone is missing required module {path}"
            ) from exc
        if not isinstance(module, nn.Module):
            raise CromaExecutionContractError(f"{path} is not a torch module")
        return module

    def _validate_no_active_side_channels(self, backbone: nn.Module) -> None:
        for name in _UNSAFE_DYNAMIC_ATTRS:
            value = getattr(backbone, name, None)
            if isinstance(value, Tensor):
                raise CromaExecutionContractError(
                    f"ICE exact refuses active backbone side-channel tensor: {name}"
                )

    def _validate_eliminated_modules(self, backbone: nn.Module) -> None:
        """Reject pruned modules with stateful/stochastic or hooked behavior."""

        for path in self.plan.eliminated_nodes:
            if path == "joint_GAP":
                # Official CROMA computes joint_GAP as a tensor mean, not a module.
                continue
            module = self._get_submodule(backbone, path)
            if module._forward_hooks or module._forward_pre_hooks:
                raise CromaExecutionContractError(
                    f"ICE exact refuses forward hooks on eliminated module {path}"
                )
            for child in module.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.track_running_stats:
                    raise CromaExecutionContractError(
                        f"ICE exact refuses running-stat BatchNorm under eliminated module {path}"
                    )
                if isinstance(child, nn.Dropout) and float(child.p) != 0.0:
                    raise CromaExecutionContractError(
                        f"ICE exact refuses stochastic Dropout under eliminated module {path}"
                    )

    def validate(self, backbone: nn.Module) -> None:
        """Fail closed unless the audited backbone matches the exact plan contract."""

        if not isinstance(backbone, nn.Module):
            raise CromaExecutionContractError("audited backbone must be a torch module")
        for encoder_name in ("s1_encoder", "s2_encoder"):
            encoder = getattr(backbone, encoder_name, None)
            transformer = getattr(encoder, "transformer", None)
            layers = getattr(transformer, "layers", None)
            linear_input = getattr(encoder, "linear_input", None)
            if not isinstance(encoder, nn.Module) or not isinstance(layers, nn.ModuleList):
                raise CromaExecutionContractError(
                    f"audited backbone does not expose {encoder_name} in the pinned CROMA form"
                )
            if not isinstance(linear_input, nn.Module):
                raise CromaExecutionContractError(f"{encoder_name}.linear_input is missing")
            for index, block in enumerate(layers):
                if not isinstance(block, nn.ModuleList) or len(block) != 2:
                    raise CromaExecutionContractError(
                        f"{encoder_name} layer {index} is not [self_attn, ffn]"
                    )
        if not isinstance(getattr(backbone, "attn_bias", None), Tensor):
            raise CromaExecutionContractError("audited backbone attn_bias tensor is missing")
        self._validate_no_active_side_channels(backbone)
        self._validate_eliminated_modules(backbone)

    @staticmethod
    def _run_encoder_prefix(
        encoder: nn.Module,
        images: Tensor,
        attn_bias: Tensor,
        *,
        last_layer: int | None,
        final_norm: bool,
    ) -> Tensor | None:
        if last_layer is None:
            return None
        patch_size = getattr(encoder, "patch_size", None)
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise CromaExecutionContractError("CROMA encoder patch_size is invalid")
        x = rearrange(
            images,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=patch_size,
            j=patch_size,
        )
        x = encoder.linear_input(x)
        layers = encoder.transformer.layers
        if last_layer < 0 or last_layer >= len(layers):
            raise CromaExecutionContractError("execution plan exceeds encoder depth")
        for index in range(last_layer + 1):
            self_attn, ffn = layers[index]
            x = self_attn(x, attn_bias) + x
            # Call the exact FFN module object.  Existing hooks on ``.1`` fire
            # here and therefore capture the same pre-residual FFN output as
            # the verified full CROMA forward.
            ffn_output = ffn(x)
            x = ffn_output + x
        if final_norm:
            norm_out = getattr(encoder.transformer, "norm_out", None)
            if not isinstance(norm_out, nn.Module):
                raise CromaExecutionContractError("requested CROMA final norm is missing")
            x = norm_out(x)
        return x

    def execute(
        self,
        backbone: nn.Module,
        *,
        SAR_images: Tensor | None,
        optical_images: Tensor | None,
    ) -> Mapping[str, Tensor]:
        """Execute the certified exact graph and return official-style side outputs."""

        self.validate(backbone)
        if SAR_images is None or optical_images is None:
            raise CromaExecutionContractError("ICE exact requires paired SAR and optical images")
        if SAR_images.ndim != 4 or optical_images.ndim != 4:
            raise CromaExecutionContractError("ICE exact inputs must be BCHW tensors")
        if SAR_images.shape[0] != optical_images.shape[0] or SAR_images.shape[-2:] != optical_images.shape[-2:]:
            raise CromaExecutionContractError("ICE exact paired inputs must share batch/spatial shape")

        attn_bias = backbone.attn_bias.to(SAR_images.device)
        sar_encodings = self._run_encoder_prefix(
            backbone.s1_encoder,
            SAR_images,
            attn_bias,
            last_layer=self.plan.s1_last_layer,
            final_norm=self.plan.require_s1_final_norm,
        )
        optical_encodings = self._run_encoder_prefix(
            backbone.s2_encoder,
            optical_images,
            backbone.attn_bias.to(optical_images.device),
            last_layer=self.plan.s2_last_layer,
            final_norm=self.plan.require_s2_final_norm,
        )

        outputs: dict[str, Tensor] = {}
        if self.plan.require_s1_gap:
            if sar_encodings is None:
                raise CromaExecutionContractError("S1 GAP requires S1 encodings")
            outputs["SAR_encodings"] = sar_encodings
            outputs["SAR_GAP"] = backbone.GAP_FFN_s1(sar_encodings.mean(dim=1))
        if self.plan.require_s2_gap:
            if optical_encodings is None:
                raise CromaExecutionContractError("S2 GAP requires S2 encodings")
            outputs["optical_encodings"] = optical_encodings
            outputs["optical_GAP"] = backbone.GAP_FFN_s2(optical_encodings.mean(dim=1))
        if self.plan.require_joint_encoder:
            if sar_encodings is None or optical_encodings is None:
                raise CromaExecutionContractError("joint encoder requires both final modality encodings")
            joint = backbone.cross_encoder(
                x=sar_encodings,
                context=optical_encodings,
                relative_position_bias=backbone.attn_bias.to(optical_images.device),
            )
            outputs["joint_encodings"] = joint
            outputs["joint_GAP"] = joint.mean(dim=1)
        return outputs
