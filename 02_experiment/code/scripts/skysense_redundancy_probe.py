# -*- coding: utf-8 -*-
"""R23 structural audit probe: dependency-necessary layers of the SkySense S2/S1 encoder.

Answers the R23-P0 question for the S2 (and S1) VisionTransformer-Large branch:
under a chosen downstream receiver contract, which of the 24 transformer
layers are on the dependency path of the consumed output, i.e. what a
dependency compiler could in principle eliminate?

Because SkySense official weights are currently unavailable (Notion link dead,
no mirror), this audit runs on the pure-PyTorch equivalent with random
initialization. Dependency structure is a property of the graph, not of the
weights, so the conclusion transfers to the official checkpoint.

Receiver contract (mirrors how an intermediate-feature consumer would read a
deep encoder, analogous to the CROMA receiver reading shallow taps):
  contract_A (deep/endpoint): downstream reads only the final layer output
      (out_indices=[23]). No layer can be eliminated a priori; expected ~0%.
  contract_B (intermediate-tap): downstream reads layer L_out and discards the
      rest (like CROMA reading layer 5 of 12). Layers after L_out are
      structurally removable -> the audit quantifies this.
The probe reports per-layer sensitivity of the contract output to zeroing that
layer's output, for both contracts, and states the removable suffix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geotoken3path.vendor_audit.skysense_vit import build_skysense_vit  # noqa: E402

NUM_LAYERS = 24
IN_CHANNELS = 10  # S2 (10 bands); S1 is 2 and shares the same structure.


def _layer_outputs(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Collect every intermediate layer output via hooks (model returns only out_indices)."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _hook(layer_idx: int):
        def _fn(_mod, _inp, out):
            captured[layer_idx] = out.detach().clone()
        return _fn

    for i in range(NUM_LAYERS):
        handles.append(model.layers[i].register_forward_hook(_hook(i)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return [captured[i] for i in range(NUM_LAYERS)]


def _sensitivity(model: torch.nn.Module, x: torch.Tensor, layer_idx: int,
                 reference: torch.Tensor) -> float:
    """Max |delta| of the model's final-layer output when layer_idx's output is zeroed."""
    orig = model.layers[layer_idx].forward

    def _zeroed(t):
        out = orig(t)
        return torch.zeros_like(out)

    model.layers[layer_idx].forward = _zeroed
    try:
        with torch.no_grad():
            outs = model(x)
    finally:
        model.layers[layer_idx].forward = orig
    return float((reference - outs[-1]).abs().max().item())


def main() -> None:
    torch.manual_seed(0)
    print("== R23 SkySense S2/S1 ViT-L structural dependency probe ==")
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | layers={NUM_LAYERS} in_ch={IN_CHANNELS}")

    model = build_skysense_vit(in_channels=IN_CHANNELS, img_size=64, num_layers=NUM_LAYERS,
                               drop_path_rate=0.0, seed=0)
    model.eval()
    x = torch.randn(2, IN_CHANNELS, 64, 64)

    # Reference final-layer output (the model default out_indices=[23]).
    with torch.no_grad():
        ref_final = model(x)[-1]

    print("\n== contract A: downstream reads only the final layer output ==")
    sens = [_sensitivity(model, x, i, ref_final) for i in range(NUM_LAYERS)]
    necessary = [i for i, s in enumerate(sens) if s > 1e-6]
    removable = [i for i, s in enumerate(sens) if s <= 1e-6]
    print(f"layers necessary to the final-layer output: {necessary}")
    print(f"layers removable (zeroing does not change final output): {removable}")
    print(f"-> removable {len(removable)}/{NUM_LAYERS} under contract A (expected ~0: final layer depends on all prefixes)")

    print("\n== contract B: downstream reads an intermediate tap (e.g. layer L) ==")
    for L in (5, 11, 17):  # three representative tap depths
        # Collect every layer's output of the reference forward once, then
        # measure how zeroing layer i (i <= L) perturbs layer L's output.
        with torch.no_grad():
            ref_all = _layer_outputs(model, x)
        ref_L = ref_all[L]
        sens_L = []
        for i in range(L + 1):
            orig = model.layers[i].forward

            def _zeroed(t, _orig=orig):
                out = _orig(t)
                return torch.zeros_like(out)

            model.layers[i].forward = _zeroed
            try:
                with torch.no_grad():
                    pert_all = _layer_outputs(model, x)
            finally:
                model.layers[i].forward = orig
            sens_L.append(float((ref_L - pert_all[L]).abs().max().item()))
        nec = [i for i, s in enumerate(sens_L) if s > 1e-6]
        rem = [i for i, s in enumerate(sens_L) if s <= 1e-6]
        suffix_removable = list(range(L + 1, NUM_LAYERS))
        print(f"  tap layer {L}: necessary among 0..{L}: {nec}")
        print(f"    removable among 0..{L}: {rem}")
        print(f"    suffix layers {L + 1}..{NUM_LAYERS - 1} structurally removable (never read): "
              f"{len(suffix_removable)}/{NUM_LAYERS} -> {len(suffix_removable)/NUM_LAYERS*100:.1f}% of layers")

    print("\nPROBE DONE")


if __name__ == "__main__":
    main()
