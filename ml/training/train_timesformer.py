"""
LENS ML — TimeSformer Video DDP Training Script
Phase 1 (epochs 0-9): spatial attention frozen, only temporal + head trained
Phase 2 (epochs 10+): full model fine-tuned jointly

Run:
  torchrun --nproc_per_node=4 train_timesformer.py \
    --train-manifest manifests/split_train.parquet \
    --val-manifest   manifests/split_val.parquet \
    --pretrained     /data/weights/timesformer_k600.pth \
    --out-dir        checkpoints/timesformer
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.timesformer_video import (
    TimeSformerForensic,
    freeze_spatial_attention,
    unfreeze_spatial_attention,
)
from lens_datasets.augmentation import build_train_augmentation, build_val_augmentation
from training.utils import (
    AUCTracker, CheckpointManager, EarlyStopping,
    build_cosine_schedule, cleanup_ddp, get_amp_scaler,
    is_main_process, log_metrics, reduce_mean,
    set_seed, setup_ddp, setup_tracking
)

log = logging.getLogger(__name__)
NUM_FRAMES = 8


# ── Video clip dataset ─────────────────────────────────────────────────────────
class VideoClipDataset(Dataset):
    """
    Loads video clips from manifest. For each item, reads N frames
    evenly spaced across the video and returns (T, C, H, W).
    Manifest must have columns: [video_path or out_path, label].
    """
    def __init__(self, manifest: pd.DataFrame, transform,
                 num_frames: int = 8, image_size: int = 224,
                 is_train: bool = True) -> None:
        self.df        = manifest.reset_index(drop=True)
        self.transform = transform
        self.num_frames = num_frames
        self.size      = image_size
        self.is_train  = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_video_frames(self, path: str) -> np.ndarray:
        """Returns (T, H, W, C) uint8 RGB."""
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size),
                                   interpolation=cv2.INTER_LANCZOS4)
            else:
                frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)   # (T, H, W, C)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        vpath = row.get("video_path") or row.get("out_path") or row["src_path"]
        label = int(row["label"])

        frames = self._load_video_frames(str(vpath))   # (T, H, W, 3) uint8
        # Apply same augmentation to every frame
        aug_frames = []
        first = self.transform(image=frames[0])
        aug_frames.append(first["image"])
        for frame in frames[1:]:
            aug_frames.append(self.transform(image=frame)["image"])

        # Stack: (T, H, W, C) → (T, C, H, W)
        clip = torch.from_numpy(
            np.stack(aug_frames, axis=0).transpose(0, 3, 1, 2)
        )   # (T, C, H, W)
        return clip, label


# ── Training epoch ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, steps = 0.0, 0
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True).float()    # (B, T, C, H, W)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(clips)                               # (B, 1)
            loss   = criterion(logits.squeeze(1), labels)

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
    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast():
            logits = model(clips)
            loss   = criterion(logits.squeeze(1), labels)
        auc_tracker.update(logits, labels.long())
        total_loss += loss.item()
        steps += 1
    metrics = auc_tracker.compute()
    metrics["val/loss"] = total_loss / max(steps, 1)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_ddp(rank, world_size)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{rank}")

    train_df = pd.read_parquet(args.train_manifest)
    val_df   = pd.read_parquet(args.val_manifest)

    # Filter to videos only if manifest is mixed
    for df in (train_df, val_df):
        if "video_path" in df.columns or "out_path" in df.columns:
            pass   # already video manifest

    train_aug = build_train_augmentation(args.image_size)
    val_aug   = build_val_augmentation(args.image_size)

    train_ds = VideoClipDataset(train_df, train_aug, NUM_FRAMES,
                                 args.image_size, is_train=True)
    val_ds   = VideoClipDataset(val_df,   val_aug,   NUM_FRAMES,
                                 args.image_size, is_train=False)

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
    per_gpu = args.batch_size // world_size

    train_loader = DataLoader(train_ds, batch_size=per_gpu, sampler=train_sampler,
                               num_workers=args.workers, pin_memory=True,
                               prefetch_factor=2, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=per_gpu * 2, sampler=val_sampler,
                               num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.pretrained:
        model = TimeSformerForensic.from_pretrained(args.pretrained, dropout=args.dropout)
    else:
        model = TimeSformerForensic(dropout=args.dropout)
    model = model.to(device)

    # Phase 1: freeze spatial attention
    freeze_spatial_attention(model)
    log.info("[Rank %d] Phase 1: spatial attention frozen", rank)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_cosine_schedule(optimizer, args.epochs,
                                       warmup_epochs=args.warmup_epochs)
    criterion = nn.BCEWithLogitsLoss()
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

    FREEZE_EPOCHS = 10

    for epoch in range(args.epochs):
        # Phase 2 transition: unfreeze spatial attention at epoch 10
        if epoch == FREEZE_EPOCHS:
            unfreeze_spatial_attention(model.module)
            # Re-create optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1,  # lower LR for spatial
                weight_decay=args.weight_decay
            )
            scheduler = build_cosine_schedule(
                optimizer, args.epochs - FREEZE_EPOCHS, warmup_epochs=2
            )
            if is_main_process():
                log.info("Phase 2: spatial attention unfrozen at epoch %d", epoch)

        train_sampler.set_epoch(epoch)
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, scheduler, device)
        val_m   = validate(model, val_loader, criterion, device, auc_tracker)
        val_auc = reduce_mean(torch.tensor(val_m["auroc"], device=device)).item()

        if is_main_process():
            log.info("Epoch %3d | loss=%.4f | val_auc=%.4f | phase=%s",
                     epoch, train_m["train/loss"], val_auc,
                     "1-frozen" if epoch < FREEZE_EPOCHS else "2-joint")
            all_m = {**train_m, **{f"val/{k}": v for k, v in val_m.items()},
                     "epoch": epoch}
            if tracker:
                log_metrics(tracker, all_m, epoch, args.tracking)

            state = {"epoch": epoch, "model": model.module.state_dict(),
                     "val_auc": val_auc, "args": vars(args)}
            ckpt_mgr.save(state, val_auc, epoch, "timesformer")

            if early_stop.step(val_auc):
                log.info("Early stop at epoch %d", epoch)
                break

    cleanup_ddp()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(filename)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True, type=Path)
    p.add_argument("--val-manifest",   required=True, type=Path)
    p.add_argument("--out-dir",        required=True, type=Path)
    p.add_argument("--pretrained",     type=Path, default=None)
    p.add_argument("--image-size",     type=int,   default=224)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--warmup-epochs",  type=int,   default=3)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--patience",       type=int,   default=8)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tracking",       default="wandb")
    p.add_argument("--project",        default="LENS")
    p.add_argument("--run-name",       default="timesformer-forensic")
    p.add_argument("--tracking-uri",   default=None)
    p.add_argument("--api-key",        default=None)
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)


if __name__ == "__main__":
    main()
