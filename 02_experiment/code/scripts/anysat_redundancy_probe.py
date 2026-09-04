# -*- coding: utf-8 -*-
"""R22 probe: measure the dependency-minimal execution surface of AnySat.

This is a pure measurement over the official AnySat release graph. It does not
train, mutate weights, or change any module semantics. For the official dense
receiver contract (final patch token + per-modality subpatch), it reports which
executed tensors are actually consumed by the downstream dense feature, i.e.
what a dependency compiler could in principle eliminate.

The probe answers one question: is there structurally removable compute under
the *official* dense contract, before any ICE machinery is ported?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1] / "vendor" / "anysat_repo"
WEIGHTS = Path(__file__).resolve().parents[1] / "vendor" / "anysat" / "AnySat.pth"
# hubconf imports "from src.models..." relative to the repo root, and the
# encoder modules import "from models.networks..." relative to repo/src.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def describe(name: str, module: torch.nn.Module, depth: int = 0, max_depth: int = 2) -> None:
    """Print the submodule tree of a module (structural audit)."""
    if depth > max_depth:
        return
    for child_name, child in module.named_children():
        params = sum(p.numel() for p in child.parameters())
        print("  " * depth + f"{child_name}: {type(child).__name__} ({params/1e6:.2f}M params)")
        describe(name, child, depth + 1, max_depth)


def main() -> None:
    torch.manual_seed(0)
    from hubconf import AnySat

    print(f"== AnySat redundancy probe ==")
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    model = AnySat(model_size="base", flash_attn=False, release=True)
    sd = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)["state_dict"]
    model.model.load_state_dict(sd)
    model.eval()
    print(f"loaded weights from {WEIGHTS.name}")

    # ---- Inputs: SEN12TS-like s1(3ch incl ratio) + s2(10ch), T=1, 64px @10m
    B, T, H, W = 2, 1, 64, 64
    x = {
        "s1": torch.randn(B, T, 3, H, W),
        "s1_dates": torch.zeros(B, T),
        "s2": torch.randn(B, T, 10, H, W),
        "s2_dates": torch.zeros(B, T),
    }

    # ---- Trace which submodules execute and capture every tensor that is
    # consumed by the dense output. We do this by instrumenting forward hooks
    # and then perturbing each captured tensor in a *separate* forward pass:
    # if the dense output changes, the tensor is on the dependency path; if
    # not, a dependency compiler could have skipped its computation.
    exec_counts: dict[str, int] = {}

    def _make_hook(name: str):
        def _hook(_mod, _inp, out):
            exec_counts[name] = exec_counts.get(name, 0) + 1
            return None  # do not modify output
        return _hook

    handles = []
    # Instrument per-modality predictor blocks and global blocks.
    for i, blk in enumerate(model.model.spatial_encoder.predictor_blocks):
        h = blk.register_forward_hook(_make_hook(f"spatial_encoder.predictor_blocks.{i}"))
        handles.append(h)
    for i, blk in enumerate(model.model.blocks):
        h = blk.register_forward_hook(_make_hook(f"blocks.{i}"))
        handles.append(h)
    for name, mod in model.model.named_modules():
        if name.startswith("projector_") or name in ("cls_token",):
            continue

    # ---- Reference dense output.
    with torch.no_grad():
        dense_ref = model(x, patch_size=10, output="dense")

    print(f"\ndense output: {tuple(dense_ref.shape)}")
    print(f"executed transformer blocks: {sorted(exec_counts)}")

    # ---- Perturbation probe: for each executed transformer block, register a
    # forward hook that zeroes the block's output, then check whether the dense
    # output changes. A block whose output does not affect dense output is not
    # on the dense dependency path (a dependency compiler could skip it).
    print("\n== perturbation dependency probe (block -> dense output sensitive?) ==")
    dependency: dict[str, bool] = {}

    def _zero_hook(_mod, _inp, out):
        if isinstance(out, torch.Tensor):
            return torch.zeros_like(out)
        return out

    for blk_name in sorted(exec_counts):
        parts = blk_name.split(".")
        node = model.model
        for p in parts:
            node = getattr(node, p) if not p.isdigit() else node[int(p)]
        handle = node.register_forward_hook(_zero_hook)
        try:
            with torch.no_grad():
                dense_pert = model(x, patch_size=10, output="dense")
        except Exception as exc:
            dependency[blk_name] = None
            print(f"  {blk_name}: ERROR {type(exc).__name__}: {exc}")
        else:
            delta = (dense_ref - dense_pert).abs().max().item()
            dependency[blk_name] = delta > 1e-6
            print(f"  {blk_name}: sensitive={dependency[blk_name]} (max|delta|={delta:.3e})")
        finally:
            handle.remove()

    # ---- blocks.6 (CrossBlockMulti) is invoked via AnyModule.forward_release
    # directly, bypassing nn.Module.__call__, so forward hooks never see it.
    # Verify it explicitly: count calls and test necessity by zeroing its output.
    print("\n== explicit blocks.6 (CrossBlockMulti) verification ==")
    cross_block = model.model.blocks[-1]
    orig_fr = cross_block.forward_release
    calls = {"n": 0}

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return orig_fr(*args, **kwargs)

    cross_block.forward_release = _counting
    with torch.no_grad():
        _ = model(x, patch_size=10, output="dense")
    print(f"  blocks.6 forward_release calls under dense: {calls['n']}")
    dependency["blocks.6 (CrossBlockMulti)"] = None

    def _zeroing(*args, **kwargs):
        out = orig_fr(*args, **kwargs)
        return torch.zeros_like(out)

    cross_block.forward_release = _zeroing
    with torch.no_grad():
        dense_pert = model(x, patch_size=10, output="dense")
    delta = (dense_ref - dense_pert).abs().max().item()
    dependency["blocks.6 (CrossBlockMulti)"] = delta > 1e-6
    print(f"  blocks.6 sensitive={delta > 1e-6} (max|delta|={delta:.3e})")
    cross_block.forward_release = orig_fr

    print("\n== summary ==")
    removable = [k for k, v in dependency.items() if v is False]
    necessary = [k for k, v in dependency.items() if v is True]
    print(f"necessary blocks ({len(necessary)}): {necessary}")
    print(f"removable blocks (not on dense dependency path): {removable}")
    print(f"=> {len(necessary)}/{len(dependency)} transformer blocks necessary under dense contract")
    print(f"=> {len(removable)} removable ({len(removable)/len(dependency)*100:.1f}%)")

    for h in handles:
        h.remove()
    print("\nPROBE DONE")


if __name__ == "__main__":
    main()
