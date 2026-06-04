"""
LENS ML — EfficientNet-B7 DDP Training Script
Supports:
  A) HuggingFace streaming (CIFAKE, GenImage) — --use-hf-streaming
  B) Local Parquet manifests                  — --train-manifest / --val-manifest

Run (HF streaming, single GPU):
  python ml/training/train_efficientnet.py \
    --use-hf-streaming --hf-dataset cifake \
    --out-dir checkpoints/efficientnet \
    --epochs 5 --batch-size 32 --tracking none

Run (4×A100 DDP):
  torchrun --nproc_per_node=4 ml/training/train_efficientnet.py \
    --use-hf-streaming --hf-dataset cifake \
    --out-dir checkpoints/efficientnet --epochs 60 --batch-size 256
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.efficientnet_forensic import (
    EfficientNetForensic, ForensicLoss, FAMILY_CLASSES
)
from lens_datasets.augmentation import build_train_augmentation, build_val_augmentation
from training.utils import (
    AUCTracker, CheckpointManager, EarlyStopping,
    build_cosine_schedule, cleanup_ddp, get_amp_scaler,
    is_main_process, log_metrics, reduce_mean,
    set_seed, setup_ddp, setup_tracking
)

log = logging.getLogger(__name__)
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_CLASSES)}


# ── Local Parquet dataset ─────────────────────────────────────────────────────
class ForensicImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform, image_size: int = 224) -> None:
        self.df        = manifest.reset_index(drop=True)
        self.transform = transform
        self.size      = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import numpy as np
        import io
        row  = self.df.iloc[idx]
        try:
            if "image" in row and isinstance(row["image"], dict):
                img_data = row["image"]
                img_bytes = img_data.get("bytes") or img_data.get("array")
                if isinstance(img_bytes, bytes):
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    img = Image.open(img_data.get("path")).convert("RGB")
            elif "image" in row and isinstance(row["image"], bytes):
                img = Image.open(io.BytesIO(row["image"])).convert("RGB")
            else:
                path = row.get("dst_path") or row.get("out_path") or row["src_path"]
                img = Image.open(path).convert("RGB")

            arr = self.transform(image=np.array(img))["image"]
            arr = torch.from_numpy(arr.transpose(2, 0, 1))
        except Exception:
            arr = torch.zeros(3, self.size, self.size)
        label  = int(row["label"])
        family = FAMILY_TO_IDX.get(str(row.get("family", "REAL")), 0)
        return arr, label, family


# ── Loader factory (HF streaming OR local Parquet) ────────────────────────────
def _build_loaders(args, world_size: int, rank: int):
    """Returns (train_loader, val_loader, train_sampler_or_None)."""

    if args.use_hf_streaming:
        from lens_datasets.remote.hf_streaming_loader import (
            build_genimage_stream, HFStreamDataset
        )
        token     = args.hf_token or os.environ.get("HF_TOKEN")
        train_aug = build_train_augmentation(args.image_size)
        val_aug   = build_val_augmentation(args.image_size)
        per_gpu   = args.batch_size // world_size

        train_stream = build_genimage_stream(
            split="train", hf_token=token,
            image_size=args.image_size, dataset_key=args.hf_dataset
        )
        # CIFAKE uses "test" for val; other datasets use "validation"
        val_split  = "test" if args.hf_dataset == "cifake" else "validation"
        val_stream = build_genimage_stream(
            split=val_split, hf_token=token,
            image_size=args.image_size, dataset_key=args.hf_dataset
        )

        train_ds = HFStreamDataset("train", train_stream, transform=train_aug,
                                    image_size=args.image_size, shuffle_buffer=2000)
        val_ds   = HFStreamDataset("val",   val_stream,   transform=val_aug,
                                    image_size=args.image_size,
                                    shuffle_buffer=200, max_samples=5000)

        train_loader = DataLoader(train_ds, batch_size=per_gpu,
                                   num_workers=args.workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=per_gpu * 2,
                                   num_workers=2, pin_memory=True)
        return train_loader, val_loader, None   # no DistributedSampler for IterableDataset

    else:
        train_df  = pd.read_parquet(args.train_manifest)
        val_df    = pd.read_parquet(args.val_manifest)
        train_aug = build_train_augmentation(args.image_size)
        val_aug   = build_val_augmentation(args.image_size)
        per_gpu   = args.batch_size // world_size

        train_ds      = ForensicImageDataset(train_df, train_aug, args.image_size)
        val_ds        = ForensicImageDataset(val_df,   val_aug,   args.image_size)
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

        train_loader = DataLoader(
            train_ds, batch_size=per_gpu, sampler=train_sampler,
            num_workers=args.workers, pin_memory=True,
            prefetch_factor=2 if args.workers > 0 else None,
            persistent_workers=True if args.workers > 0 else False
        )
        val_loader   = DataLoader(
            val_ds, batch_size=per_gpu * 2, sampler=val_sampler,
            num_workers=args.workers, pin_memory=True
        )
        return train_loader, val_loader, train_sampler


# ── Training epoch ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, steps = 0.0, 0
    for imgs, labels, families in loader:
        imgs     = imgs.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        families = families.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            preds  = model(imgs)
            losses = criterion(preds, labels, families)
            loss   = losses["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps += 1

    scheduler.step()
    return {"train/loss": total_loss / max(steps, 1)}


@torch.no_grad()
def validate(model, loader, criterion, device, auc_tracker):
    model.eval()
    auc_tracker.reset()
    total_loss, steps = 0.0, 0
    for imgs, labels, families in loader:
        imgs     = imgs.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        families = families.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            preds  = model(imgs)
            losses = criterion(preds, labels, families)
        auc_tracker.update(preds["binary"], labels)
        total_loss += losses["loss"].item()
        steps += 1
    metrics = auc_tracker.compute()
    metrics["val/loss"] = total_loss / max(steps, 1)
    return metrics


# ── Main training function ─────────────────────────────────────────────────────
def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if world_size > 1:
        setup_ddp(rank, world_size)
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    log.info("Rank %d | Device: %s | Streaming: %s | Dataset: %s",
             rank, device, args.use_hf_streaming,
             args.hf_dataset if args.use_hf_streaming else "local")

    train_loader, val_loader, train_sampler = _build_loaders(args, world_size, rank)

    model = EfficientNetForensic(pretrained=True, dropout=args.dropout, variant=args.model_variant).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = build_cosine_schedule(optimizer, args.epochs, args.warmup_epochs)
    criterion = ForensicLoss()
    scaler    = get_amp_scaler()
    auc_tracker = AUCTracker(device)

    ckpt_mgr = early_stop = tracker = None
    if is_main_process():
        ckpt_mgr   = CheckpointManager(Path(args.out_dir), top_k=3)
        early_stop = EarlyStopping(patience=args.patience, mode="max")
        if args.tracking != "none":
            tracker = setup_tracking(args.project, args.run_name,
                                      vars(args), args.tracking,
                                      args.tracking_uri, args.api_key)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, scheduler, device)
        val_m   = validate(model, val_loader, criterion, device, auc_tracker)
        val_auc = val_m["auroc"]
        if world_size > 1:
            val_auc = reduce_mean(torch.tensor(val_auc, device=device)).item()

        if is_main_process():
            log.info("Epoch %3d | train_loss=%.4f | val_auc=%.4f",
                     epoch, train_m["train/loss"], val_auc)
            if tracker:
                log_metrics(tracker, {**train_m, **{f"val/{k}": v
                             for k, v in val_m.items()}, "epoch": epoch},
                            epoch, args.tracking)
            state = {"epoch": epoch, "model": getattr(model, "module", model).state_dict(),
                     "val_auc": val_auc, "args": vars(args)}
            ckpt_mgr.save(state, val_auc, epoch, prefix="efficientnet")
            if early_stop.step(val_auc):
                log.info("Early stop at epoch %d", epoch)
                break

    if world_size > 1:
        cleanup_ddp()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS EfficientNet-B7 Training")

    # Dataset source (mutually exclusive: HF streaming OR local Parquet)
    p.add_argument("--use-hf-streaming", action="store_true",
                   help="Stream dataset from HuggingFace Hub (zero local storage)")
    p.add_argument("--hf-dataset",  default="cifake",
                   choices=["cifake","deepfake_faces","genimage","genimage_mj"],
                   help="Which HF dataset to stream (only with --use-hf-streaming)")
    p.add_argument("--hf-token",    default=None,
                   help="HF token (default: reads $HF_TOKEN env var)")
    p.add_argument("--train-manifest", type=Path, default=None)
    p.add_argument("--val-manifest",   type=Path, default=None)

    # Training
    p.add_argument("--out-dir",        required=True, type=Path)
    p.add_argument("--model-variant",  default="tf_efficientnet_b0.ns_jft_in1k",
                   help="timm model variant (e.g. tf_efficientnet_b0.ns_jft_in1k)")
    p.add_argument("--image-size",     type=int,   default=224)
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tracking",       choices=["wandb","mlflow","tensorboard","none"],
                   default="none")
    p.add_argument("--project",        default="LENS")
    p.add_argument("--run-name",       default="efficientnet-b7")
    p.add_argument("--tracking-uri",   default=None)
    p.add_argument("--api-key",        default=None)
    args = p.parse_args()

    # Validate arguments
    if not args.use_hf_streaming and (not args.train_manifest or not args.val_manifest):
        p.error("Provide --use-hf-streaming OR both --train-manifest and --val-manifest")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        train(rank, world_size, args)
    else:
        train(0, 1, args)


if __name__ == "__main__":
    main()
