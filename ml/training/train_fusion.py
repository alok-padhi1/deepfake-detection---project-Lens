"""
LENS ML — Bayesian Fusion MLP Training Script
Trained on validation set score vectors collected from all upstream models.
Runs isotonic calibration after training.

Usage:
  python train_fusion.py \
    --score-vectors manifests/val_score_vectors.parquet \
    --out-dir       checkpoints/fusion \
    --calibrate
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAUROC

from models.bayesian_fusion import (
    BayesianFusionMLP, FusionLoss, INPUT_DIM,
    IsotonicCalibrator, SCORE_SCHEMA
)
from training.utils import (
    AUCTracker, CheckpointManager, EarlyStopping,
    build_cosine_schedule, set_seed, setup_tracking, log_metrics
)

log = logging.getLogger(__name__)


# ── Score vector dataset ───────────────────────────────────────────────────────
class ScoreVectorDataset(Dataset):
    """
    Loads a Parquet file with columns for each score dimension + label + media_type.
    Expected columns: all keys from SCORE_SCHEMA, plus 'label' and 'media_type'.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.score_cols = list(SCORE_SCHEMA.keys())

        # Fill missing score columns with 0.5 (uninformative)
        for col in self.score_cols:
            if col not in self.df.columns:
                self.df[col] = 0.5

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        scores = torch.tensor(
            [float(row[c]) for c in self.score_cols],
            dtype=torch.float32
        )
        label      = float(row["label"])
        media_type = str(row.get("media_type", "image"))
        return scores, torch.tensor(label), media_type


def _collate(batch):
    scores, labels, media_types = zip(*batch)
    return torch.stack(scores), torch.stack(labels), list(media_types)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, steps = 0.0, 0
    for scores, labels, media_types in loader:
        scores = scores.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        # For mixed media_type batches, compute loss per type and average
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        for mt in set(media_types):
            mask = torch.tensor([m == mt for m in media_types], device=device)
            if mask.sum() == 0:
                continue
            logits = model(scores[mask], media_type=mt)
            l = criterion(logits, labels[mask])
            loss = loss + l / len(set(media_types))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return {"train/loss": total_loss / max(steps, 1)}


@torch.no_grad()
def validate(model, loader, device, auc_tracker):
    model.eval()
    auc_tracker.reset()
    all_logits, all_labels = [], []
    for scores, labels, media_types in loader:
        scores = scores.to(device)
        labels = labels.to(device)
        for mt in set(media_types):
            mask = torch.tensor([m == mt for m in media_types], device=device)
            if mask.sum() == 0:
                continue
            logits = model(scores[mask], media_type=mt)
            auc_tracker.update(logits, labels[mask].long())
            all_logits.append(logits.cpu())
            all_labels.append(labels[mask].cpu())
    metrics = auc_tracker.compute()
    return (metrics,
            torch.cat(all_logits).numpy(),
            torch.cat(all_labels).numpy())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS Fusion MLP Training")
    p.add_argument("--score-vectors", required=True, type=Path,
                   help="Parquet with score columns + label + media_type")
    p.add_argument("--out-dir",       required=True, type=Path)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--calibrate",     action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--tracking",      default="wandb")
    p.add_argument("--project",       default="LENS")
    p.add_argument("--run-name",      default="fusion-mlp")
    p.add_argument("--tracking-uri",  default=None)
    p.add_argument("--api-key",       default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.score_vectors)
    n  = len(df)
    n_val  = int(0.2 * n)
    n_train= n - n_val

    full_ds    = ScoreVectorDataset(df)
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, collate_fn=_collate,
                               num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                               shuffle=False, collate_fn=_collate,
                               num_workers=0, pin_memory=True)

    model     = BayesianFusionMLP(dropout=args.dropout).to(device)
    criterion = FusionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = build_cosine_schedule(optimizer, args.epochs, warmup_epochs=3)
    auc_tracker = AUCTracker(device)
    ckpt_mgr    = CheckpointManager(out_dir, top_k=3)
    early_stop  = EarlyStopping(patience=args.patience, mode="max")

    tracker = None
    if args.tracking != "none":
        tracker = setup_tracking(args.project, args.run_name,
                                  vars(args), args.tracking,
                                  args.tracking_uri, args.api_key)

    best_val_logits = best_val_labels = None

    for epoch in range(args.epochs):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, val_logits, val_labels = validate(model, val_loader,
                                                        device, auc_tracker)
        val_auc = val_metrics["auroc"]
        scheduler.step()

        log.info("Epoch %3d | train_loss=%.4f | val_auc=%.4f",
                 epoch, train_m["train/loss"], val_auc)

        if tracker:
            log_metrics(tracker, {**train_m, **{f"val/{k}": v
                         for k, v in val_metrics.items()}, "epoch": epoch},
                        epoch, args.tracking)

        ckpt_mgr.save({"epoch": epoch, "model": model.state_dict(),
                        "val_auc": val_auc}, val_auc, epoch, "fusion")

        if val_auc >= (ckpt_mgr.best_score or 0.0):
            best_val_logits = val_logits
            best_val_labels = val_labels

        if early_stop.step(val_auc):
            log.info("Early stop at epoch %d", epoch)
            break

    # ── Isotonic calibration ──────────────────────────────────────────────────
    if args.calibrate and best_val_logits is not None:
        log.info("Fitting isotonic calibration...")
        calibrator = IsotonicCalibrator()
        for mt in ("image", "video", "audio"):
            # Use all val logits for calibration (simplified: not split by type here)
            calibrator.fit(best_val_logits.flatten(), best_val_labels,
                           media_type=mt)
        cal_path = out_dir / "isotonic_calibrators.pkl"
        calibrator.save(cal_path)
        log.info("Calibrators saved to %s", cal_path)

    log.info("Fusion training complete. Best val AUC: %.4f", ckpt_mgr.best_score)


if __name__ == "__main__":
    main()
