"""R24 SkySense++ S2 segmentation-head training entry point (cloud).

Trains only the 1x1 segmentation head on top of the frozen, pretrained
SkySense++ S2 ViT-L backbone (CE + Lovasz 1:1, AdamW, cosine-with-warmup).
The backbone checkpoint is audited SkySense++ S2 weights (1.2GB safetensors);
the annotation channel fed to the backbone is either the ground-truth
WorldCover target mapped into the semantic vocabulary
(``gt_worldcover_leakage_documented``) or an all-zero constant map (``empty``
control).  Real data, real weights and CUDA are required; the sealed test
split is never opened.  Best and last checkpoints carry run metadata and the
backbone weights SHA256 so certification can bind to the exact artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch

from geotoken3path.data.skysensepp import (
    annotation_from_target,
    build_skysensepp_loader,
    croma_dynamic_normalize_batch_r24,
)
from geotoken3path.losses import segmentation_objective
from geotoken3path.metrics import confusion_matrix, mean_iou
from geotoken3path.models.skysensepp_seg import build_skysensepp_model
from geotoken3path.utils.test_seal import assert_test_access_allowed

_CONTRACT_NAMES = ("a", "b")
_ANNOTATION_SOURCES = ("gt", "empty")
_WEIGHTS_SHA_BLOCKS = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_WEIGHTS_SHA_BLOCKS), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine_warmup_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _windows(batch: dict[str, torch.Tensor], micro_batch: int):
    """Slice one frozen 16-row loader batch into normalization windows."""
    sample_count = int(batch["optical"].shape[0])
    if sample_count % micro_batch:
        raise RuntimeError(
            f"loader batch has {sample_count} rows, which is not divisible by "
            f"the micro_batch {micro_batch}"
        )
    for start in range(0, sample_count, micro_batch):
        end = start + micro_batch
        window_valid = int(batch["valid_count"].item()) - start
        window_valid = max(0, min(micro_batch, window_valid))
        yield {
            "optical": batch["optical"][start:end].contiguous(),
            "target": batch["target"][start:end].contiguous(),
            "valid_count": torch.tensor(window_valid, dtype=torch.int64),
        }


def _is_fully_ignored(target: torch.Tensor) -> bool:
    """A window whose rows are all ignore-index (validation pad-only window)."""
    return bool((target == 255).all())


def _annotation_for(target: torch.Tensor, annotation_source: str) -> torch.Tensor:
    if annotation_source == "gt":
        return annotation_from_target(target)
    return torch.zeros_like(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--weights", required=True, help="Audited SkySense++ S2 safetensors")
    parser.add_argument("--contract", choices=_CONTRACT_NAMES, default="b")
    parser.add_argument("--annotation", choices=_ANNOTATION_SOURCES, default="gt")
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution-scale",
        choices=("smoke", "acceptance"),
        default="acceptance",
        help="Split-seal gate label forwarded to the frozen sen12ts loader",
    )
    args = parser.parse_args()

    manifest_path = Path(args.data_manifest)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    if not manifest_path.is_absolute() or not weights_path.is_absolute() or not output_dir.is_absolute():
        parser.error("cloud training requires absolute --data-manifest, --weights and --output-dir")
    if not weights_path.is_file():
        parser.error(f"SkySense++ weights checkpoint not found: {weights_path}")
    if args.micro_batch <= 0 or 16 % args.micro_batch:
        parser.error("micro_batch must divide the frozen 16-row loader batch (1,2,4,8,16)")
    if args.grad_accum <= 0 or args.epochs <= 0 or args.resolution <= 0:
        parser.error("grad_accum, epochs and resolution must be positive")
    assert_test_access_allowed(
        {"execution_scale": args.execution_scale, "test_seal_status": "sealed"},
        "validation",
    )
    target_device = torch.device(args.device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        parser.error("R24 formal training requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    resolution = int(args.resolution)

    model = build_skysensepp_model(
        contract=args.contract,
        safetensors_path=str(weights_path),
        num_classes=11,
        seed=0,
    )
    model.to(target_device)
    model.train()
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not backbone_frozen or not trainable or any(not n.startswith("head.") for n in trainable):
        parser.error("R24 model must freeze the backbone and train the head only")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    micro_batch = int(args.micro_batch)
    grad_accum = int(args.grad_accum)
    windows_per_loader_batch = 16 // micro_batch
    loader_kwargs = {
        "batch_size": 16,
        "num_workers": 4,
        "execution_scale": args.execution_scale,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "seed": args.seed,
    }
    augmentation = {
        "name": "paired_geometric_v1",
        "enabled": True,
        "train_only": True,
        "deterministic": True,
        "orientation_space": "D4",
        "operations": [
            "horizontal_flip", "vertical_flip", "rotate_90", "rotate_180",
            "rotate_270", "transpose", "anti_transpose",
        ],
    }
    train_loader, _ = build_skysensepp_loader(
        manifest_path, split="train", augmentation=augmentation, **loader_kwargs,
    )
    validation_loader, _ = build_skysensepp_loader(
        manifest_path, split="validation", augmentation=None, **loader_kwargs,
    )
    steps_per_epoch = (len(train_loader) * windows_per_loader_batch) // grad_accum
    if steps_per_epoch < 1:
        parser.error("training split is too small for one optimizer step per epoch")
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = max(1, int(math.ceil(0.05 * total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_multiplier(
            step, total_steps=total_steps, warmup_steps=warmup_steps,
        ),
    )

    autocast_enabled = True  # protocol AMP (R24 acceptance precision: amp)
    amp_device = "cuda"
    effective_batch_samples = micro_batch * grad_accum

    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"best_skysensepp_{args.contract}_checkpoint.pt"
    last_path = output_dir / f"last_skysensepp_{args.contract}_checkpoint.pt"
    weights_sha = _sha256(weights_path)

    best_miou = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        window_count = 0
        accumulated = 0
        for batch in train_loader:
            for window in _windows(batch, micro_batch):
                normalized = croma_dynamic_normalize_batch_r24(window, micro_batch=micro_batch)
                optical10 = normalized["optical10"].to(target_device, non_blocking=True)
                target = window["target"].to(target_device, non_blocking=True)
                annotation = _annotation_for(target, args.annotation).to(target_device, non_blocking=True)
                with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=autocast_enabled):
                    logits = model(pixel_values=optical10, annotation=annotation)["logits"]
                    loss, _ = segmentation_objective(
                        logits, target, objective_name="ce_lovasz",
                    )
                    loss = loss / grad_accum
                loss.backward()
                running_loss += float(loss.detach())
                window_count += 1
                accumulated += 1
                if accumulated % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        train_loss_value = running_loss / max(1, window_count)

        # --- Validation: deterministic whole-split mIoU under AMP.
        model.eval()
        classes = int(model.num_classes)
        matrix = torch.zeros(classes, classes, dtype=torch.int64)
        with torch.no_grad():
            for batch in validation_loader:
                for window in _windows(batch, micro_batch):
                    target = window["target"]
                    if _is_fully_ignored(target):
                        continue
                    normalized = croma_dynamic_normalize_batch_r24(window, micro_batch=micro_batch)
                    optical10 = normalized["optical10"].to(target_device, non_blocking=True)
                    target_device_t = target.to(target_device, non_blocking=True)
                    annotation = _annotation_for(target_device_t, args.annotation)
                    with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=autocast_enabled):
                        logits = model(pixel_values=optical10, annotation=annotation)["logits"]
                    matrix += confusion_matrix(logits, target_device_t, classes).cpu()
        val_miou = float(mean_iou(matrix)) if matrix is not None else float("nan")
        peak_megabytes = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss_value),
                "val_mIoU_percent": val_miou * 100.0,
                "peak_vram_mb": peak_megabytes,
            }
        )
        print(
            f"epoch {epoch}/{args.epochs} train_loss={train_loss_value:.6f} "
            f"val_mIoU={val_miou * 100.0:.4f}% peak_vram_mb={peak_megabytes:.1f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        checkpoint_metadata = {
            "model": "skysensepp_s2_full",
            "contract": args.contract,
            "annotation": args.annotation,
            "annotation_note": (
                "gt_worldcover_leakage_documented"
                if args.annotation == "gt"
                else "constant_zero_control"
            ),
            "resolution": resolution,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "effective_batch_samples": effective_batch_samples,
            "seed": args.seed,
            "epochs": int(args.epochs),
            "objective": "ce_lovasz_1to1",
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "scheduler": "cosine_with_warmup",
            "warmup_fraction": 0.05,
            "gradient_clip_norm": 1.0,
            "num_classes": int(model.num_classes),
            "head_seed": 0,
            "weights_sha256": weights_sha,
            "weights_path": str(weights_path),
            "val_mIoU_percent": val_miou * 100.0,
            "epoch": epoch,
        }

        def _checkpoint_payload(miou: float, epoch_number: int) -> dict[str, object]:
            payload = {
                "head_state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.head.state_dict().items()
                },
                "epoch": epoch_number,
                "val_mIoU_percent": miou * 100.0,
            }
            payload["metadata"] = dict(checkpoint_metadata)
            payload["metadata"]["val_mIoU_percent"] = miou * 100.0
            payload["metadata"]["epoch"] = epoch_number
            return payload

        torch.save(_checkpoint_payload(val_miou, epoch), last_path)
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(_checkpoint_payload(val_miou, epoch), best_path)

    summary = {
        "status": "complete",
        "best_val_mIoU_percent": best_miou * 100.0,
        "checkpoints": {
            "best": str(best_path),
            "last": str(last_path),
        },
        "metadata": dict(checkpoint_metadata),
        "history": history,
    }
    summary_path = output_dir / "skysensepp_train_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "best_val_mIoU_percent": best_miou * 100.0,
                      "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
