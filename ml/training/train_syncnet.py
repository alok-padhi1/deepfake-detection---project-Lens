"""
LENS ML — SyncNet Cross-Modal Training Script
500K authentic + 200K manipulated video clips.
InfoNCE sync loss on authentic + BCE fake detection loss.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.syncnet_crossmodal import SyncNetCrossModal, SyncNetLoss
from training.utils import (
    AUCTracker, CheckpointManager, EarlyStopping,
    build_cosine_schedule, cleanup_ddp, get_amp_scaler,
    is_main_process, log_metrics, reduce_mean,
    set_seed, setup_ddp, setup_tracking
)

log = logging.getLogger(__name__)

N_VIDEO_FRAMES  = 30      # 1 second at 30fps
N_AUDIO_FRAMES  = 50      # 1 second / 20ms = 50 frames
AUDIO_SR        = 16_000
AUDIO_FRAME_LEN = 320     # 20ms @ 16kHz
IMAGE_SIZE      = 224


# ── AV Dataset ────────────────────────────────────────────────────────────────
class AVDataset(Dataset):
    """
    Expects manifest with: video_path, audio_path (or derives from video),
    label (0=real, 1=fake).
    Loads 1-second window from a random temporal position.
    """
    def __init__(self, manifest: pd.DataFrame, image_size: int = 224,
                 is_train: bool = True) -> None:
        self.df       = manifest.reset_index(drop=True)
        self.size     = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_video_window(self, path: str) -> np.ndarray:
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = 0
        if self.is_train and total > N_VIDEO_FRAMES:
            start_frame = np.random.randint(0, total - N_VIDEO_FRAMES)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(N_VIDEO_FRAMES):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size),
                                   interpolation=cv2.INTER_LANCZOS4)
            else:
                frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()

        # (T, H, W, C) → (T, C, H, W) float32 normalized
        arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
        return arr.transpose(0, 3, 1, 2)   # (T, C, H, W)

    def _load_audio_window(self, path: str, start_frame: int,
                            fps: float) -> np.ndarray:
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != AUDIO_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=AUDIO_SR)
        except Exception:
            wav = np.zeros(AUDIO_SR, dtype=np.float32)

        start_sample = int(start_frame / fps * AUDIO_SR)
        end_sample   = start_sample + AUDIO_SR   # 1 second
        if end_sample > len(wav):
            wav = np.pad(wav, (0, end_sample - len(wav)))
        wav = wav[start_sample:end_sample]

        # Reshape into 50 × 320 frames
        n_complete = len(wav) // AUDIO_FRAME_LEN
        if n_complete < N_AUDIO_FRAMES:
            wav = np.pad(wav, (0, N_AUDIO_FRAMES * AUDIO_FRAME_LEN - len(wav)))
            n_complete = N_AUDIO_FRAMES
        frames = wav[:N_AUDIO_FRAMES * AUDIO_FRAME_LEN].reshape(
            N_AUDIO_FRAMES, AUDIO_FRAME_LEN
        )
        return frames.astype(np.float32)   # (50, 320)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        vpath = str(row.get("video_path", row.get("out_path", "")))
        label = float(row["label"])

        # Video
        try:
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            start_frame = 0
            if self.is_train and total > N_VIDEO_FRAMES:
                start_frame = np.random.randint(0, total - N_VIDEO_FRAMES)
            vid = self._load_video_window(vpath)
        except Exception:
            vid = np.zeros((N_VIDEO_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE),
                           dtype=np.float32)
            fps = 30.0
            start_frame = 0

        # Audio (from separate .wav or extracted from video)
        audio_path = str(row.get("audio_path", vpath))
        aud = self._load_audio_window(audio_path, start_frame, fps)

        return (
            torch.from_numpy(vid),   # (T, C, H, W)
            torch.from_numpy(aud),   # (50, 320)
            torch.tensor(label),
        )


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, steps = 0.0, 0
    for vid, aud, labels in loader:
        vid    = vid.to(device, non_blocking=True)
        aud    = aud.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            preds  = model(vid, aud)
            losses = criterion(preds, labels)
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
    for vid, aud, labels in loader:
        vid    = vid.to(device, non_blocking=True)
        aud    = aud.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            preds  = model(vid, aud)
            losses = criterion(preds, labels)
        auc_tracker.update(preds["logit"], labels.long())
        total_loss += losses["loss"].item()
        steps += 1
    m = auc_tracker.compute()
    m["val/loss"] = total_loss / max(steps, 1)
    return m


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_ddp(rank, world_size)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{rank}")

    df = pd.read_parquet(args.manifest)
    train_df = df[df["split"].isin(["train"])].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    train_ds = AVDataset(train_df, IMAGE_SIZE, is_train=True)
    val_ds   = AVDataset(val_df,   IMAGE_SIZE, is_train=False)
    per_gpu  = args.batch_size // world_size

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=per_gpu, sampler=train_sampler,
                               num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=per_gpu * 2, sampler=val_sampler,
                               num_workers=args.workers, pin_memory=True)

    model     = SyncNetCrossModal().to(device)
    model     = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = SyncNetLoss(lambda_fake=1.0, lambda_sync=args.lambda_sync)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = build_cosine_schedule(optimizer, args.epochs, args.warmup_epochs)
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
        train_sampler.set_epoch(epoch)
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, scheduler, device)
        val_m   = validate(model, val_loader, criterion, device, auc_tracker)
        val_auc = reduce_mean(torch.tensor(val_m["auroc"], device=device)).item()

        if is_main_process():
            log.info("Epoch %3d | loss=%.4f | val_auc=%.4f",
                     epoch, train_m["train/loss"], val_auc)
            if tracker:
                log_metrics(tracker, {**train_m, **{f"val/{k}": v
                             for k, v in val_m.items()}, "epoch": epoch},
                            epoch, args.tracking)
            ckpt_mgr.save({"epoch": epoch, "model": model.module.state_dict(),
                            "val_auc": val_auc}, val_auc, epoch, "syncnet")
            if early_stop.step(val_auc):
                break

    cleanup_ddp()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",      required=True, type=Path)
    p.add_argument("--out-dir",       required=True, type=Path)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int,   default=3)
    p.add_argument("--lambda-sync",   type=float, default=0.5)
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--tracking",      default="wandb")
    p.add_argument("--project",       default="LENS")
    p.add_argument("--run-name",      default="syncnet-crossmodal")
    p.add_argument("--tracking-uri",  default=None)
    p.add_argument("--api-key",       default=None)
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)


if __name__ == "__main__":
    main()
