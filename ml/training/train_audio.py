"""
LENS ML — Audio Dual Model Training Script (ResNet34Mel + RawNet3 simultaneous)
Supports:
  A) HuggingFace streaming (ASVspoof 2019/2021) — --use-hf-streaming
  B) Local Parquet manifests                   — --manifest

Optimized for single-GPU execution or DDP.
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.audio_dual import AudioDualModel, AudioDualLoss
from lens_datasets.augmentation import augment_melspec
from training.utils import (
    AUCTracker, CheckpointManager, EarlyStopping,
    build_cosine_schedule, cleanup_ddp, get_amp_scaler,
    is_main_process, log_metrics, reduce_mean,
    set_seed, setup_ddp, setup_tracking
)

log = logging.getLogger(__name__)

TARGET_SR    = 16_000
MAX_DUR_S    = 4.0
MAX_SAMPLES  = int(TARGET_SR * MAX_DUR_S)   # 64000
MEL_T_DIM    = 400                          # expected time dimension of saved Mel
FRAME_SAMPLES= 320                          # 20ms @ 16kHz (for SyncNet compat)


# ── Dataset ────────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    """
    Loads pre-processed .npy Mel-spectrograms and waveforms.
    Falls back gracefully if one branch is missing.
    """
    def __init__(self, manifest: pd.DataFrame, is_train: bool = True,
                 mel_augment: bool = True) -> None:
        self.df         = manifest.reset_index(drop=True)
        self.is_train   = is_train
        self.mel_augment= mel_augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, path: str) -> np.ndarray:
        mel = np.load(path).astype(np.float32)   # (128, T)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if mel.shape[1] < MEL_T_DIM:
            mel = np.pad(mel, ((0, 0), (0, MEL_T_DIM - mel.shape[1])))
        else:
            mel = mel[:, :MEL_T_DIM]
        if self.is_train and self.mel_augment:
            mel = augment_melspec(mel)
        return mel[np.newaxis]   # (1, 128, T)

    def _load_wav(self, path: str) -> np.ndarray:
        wav = np.load(path).astype(np.float32)   # (T_samples,)
        if len(wav) < MAX_SAMPLES:
            wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
        else:
            wav = wav[:MAX_SAMPLES]
        return wav[np.newaxis]   # (1, T)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = float(row["label"])

        mel = None
        if "mel_path" in row and pd.notna(row["mel_path"]):
            try:
                mel = torch.from_numpy(self._load_mel(row["mel_path"]))
            except Exception:
                mel = torch.zeros(1, 128, MEL_T_DIM)

        wav = None
        if "wav_path" in row and pd.notna(row["wav_path"]):
            try:
                wav = torch.from_numpy(self._load_wav(row["wav_path"]))
            except Exception:
                wav = torch.zeros(1, MAX_SAMPLES)

        return mel, wav, torch.tensor(label)


# ── Collate fn (handle None modalities) ──────────────────────────────────────
def audio_collate_fn(batch):
    mels, wavs, labels = zip(*batch)
    labels_t = torch.stack(labels)

    mel_t = None
    if any(m is not None for m in mels):
        mel_t = torch.stack([
            m if m is not None else torch.zeros(1, 128, MEL_T_DIM)
            for m in mels
        ])

    wav_t = None
    if any(w is not None for w in wavs):
        wav_t = torch.stack([
            w if w is not None else torch.zeros(1, MAX_SAMPLES)
            for w in wavs
        ])

    return mel_t, wav_t, labels_t


# ── Train / val loops ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, steps = 0.0, 0
    for mel, wav, labels in loader:
        labels = labels.to(device, non_blocking=True)
        mel    = mel.to(device, non_blocking=True) if mel is not None else None
        wav    = wav.to(device, non_blocking=True) if wav is not None else None

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            preds  = model(mel=mel, waveform=wav)
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
    for mel, wav, labels in loader:
        labels = labels.to(device, non_blocking=True)
        mel    = mel.to(device, non_blocking=True) if mel is not None else None
        wav    = wav.to(device, non_blocking=True) if wav is not None else None

        with torch.amp.autocast('cuda'):
            preds  = model(mel=mel, waveform=wav)
            losses = criterion(preds, labels)

        logit_list = [preds[k] for k in ("mel_logit", "raw_logit")
                      if k in preds]
        if logit_list:
            logit = torch.stack(logit_list, dim=0).mean(0)
            auc_tracker.update(logit, labels.long())
        total_loss += losses["loss"].item()
        steps += 1

    metrics = auc_tracker.compute()
    metrics["val/loss"] = total_loss / max(steps, 1)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if world_size > 1:
        setup_ddp(rank, world_size)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if args.use_hf_streaming:
        from lens_datasets.remote.hf_audio_loader import build_asvspoof_stream
        token = args.hf_token or os.environ.get("HF_TOKEN")
        per_gpu = args.batch_size // world_size
        
        train_ds = build_asvspoof_stream(
            dataset_key=args.hf_dataset, split="train",
            hf_token=token, shuffle_buffer=500
        )
        val_split = "validation"
        val_ds = build_asvspoof_stream(
            dataset_key=args.hf_dataset, split=val_split,
            hf_token=token, shuffle_buffer=100, max_samples=1000
        )
        train_loader = DataLoader(
            train_ds, batch_size=per_gpu,
            collate_fn=audio_collate_fn,
            num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=per_gpu * 2,
            collate_fn=audio_collate_fn,
            num_workers=0, pin_memory=True
        )
        train_sampler = None
    else:
        df = pd.read_parquet(args.manifest)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)

        train_ds = AudioDataset(train_df, is_train=True)
        val_ds   = AudioDataset(val_df,   is_train=False)

        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
        per_gpu = args.batch_size // world_size

        train_loader = DataLoader(train_ds, batch_size=per_gpu,
                                   sampler=train_sampler,
                                   collate_fn=audio_collate_fn,
                                   num_workers=args.workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=per_gpu * 2,
                                   sampler=val_sampler,
                                   collate_fn=audio_collate_fn,
                                   num_workers=args.workers, pin_memory=True)

    model     = AudioDualModel(dropout=args.dropout).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = AudioDualLoss(lambda_mel=1.0, lambda_raw=1.0, lambda_reg=args.lambda_reg)
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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                   scaler, scheduler, device)
        val_m   = validate(model, val_loader, criterion, device, auc_tracker)
        val_auc = val_m["auroc"]
        if world_size > 1:
            val_auc = reduce_mean(torch.tensor(val_auc, device=device)).item()

        if is_main_process():
            log.info("Epoch %3d | loss=%.4f | val_auc=%.4f",
                     epoch, train_m["train/loss"], val_auc)
            all_m = {**train_m, **{f"val/{k}": v for k, v in val_m.items()},
                     "epoch": epoch}
            if tracker:
                log_metrics(tracker, all_m, epoch, args.tracking)
            
            target_model = model.module if world_size > 1 else model
            ckpt_mgr.save({"epoch": epoch, "model": target_model.state_dict(),
                            "val_auc": val_auc}, val_auc, epoch, "audio_dual")
            if early_stop.step(val_auc):
                break

    if world_size > 1:
        cleanup_ddp()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--use-hf-streaming", action="store_true")
    p.add_argument("--hf-dataset",  default="asvspoof19",
                   choices=["asvspoof19", "asvspoof21_la", "asvspoof21_df"])
    p.add_argument("--hf-token",    default=None)
    p.add_argument("--manifest",    type=Path, default=None)
    p.add_argument("--out-dir",     required=True, type=Path)
    p.add_argument("--epochs",      type=int,   default=40)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float, default=1e-5)
    p.add_argument("--warmup-epochs",type=int,  default=3)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--lambda-reg",  type=float, default=0.2)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--workers",     type=int,   default=0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--tracking",    default="none")
    p.add_argument("--project",     default="LENS")
    p.add_argument("--run-name",    default="audio-dual")
    p.add_argument("--tracking-uri",default=None)
    p.add_argument("--api-key",     default=None)
    args = p.parse_args()

    if not args.use_hf_streaming and not args.manifest:
        p.error("Provide --use-hf-streaming OR --manifest")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    train(rank, world_size, args)


if __name__ == "__main__":
    main()
