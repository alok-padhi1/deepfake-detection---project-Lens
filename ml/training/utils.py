"""
LENS ML — Training utilities shared across all training scripts.
Includes: checkpointing, EarlyStopping, AUC tracking,
          WandB/MLflow logging setup, DDP helpers.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

log = logging.getLogger(__name__)


# ── Checkpoint manager ────────────────────────────────────────────────────────
class CheckpointManager:
    """
    Saves top-K checkpoints by validation AUC.
    Always saves 'last.pth'.
    """
    def __init__(self, save_dir: Path, top_k: int = 3,
                 mode: str = "max") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k  = top_k
        self.mode   = mode
        self._best: List[tuple] = []   # (score, path)

    def save(
        self,
        state: Dict[str, Any],
        score: float,
        epoch: int,
        prefix: str = "ckpt",
    ) -> Path:
        fname = self.save_dir / f"{prefix}_epoch{epoch:03d}_score{score:.4f}.pth"
        torch.save(state, fname)

        # Always save last
        last = self.save_dir / f"{prefix}_last.pth"
        shutil.copy2(fname, last)

        # Track top-K
        self._best.append((score, fname))
        self._best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        if len(self._best) > self.top_k:
            _, to_delete = self._best.pop()
            if to_delete.exists() and "last" not in str(to_delete):
                to_delete.unlink()

        # Copy best instead of symlink to avoid OS privilege issues (e.g. Windows Developer Mode)
        best_link = self.save_dir / f"{prefix}_best.pth"
        best_path = self._best[0][1]
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        shutil.copy2(best_path, best_link)

        log.info("Checkpoint saved: %s (score=%.4f)", fname.name, score)
        return fname

    @property
    def best_score(self) -> Optional[float]:
        return self._best[0][0] if self._best else None


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "max") -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self._best     = float("-inf") if mode == "max" else float("inf")
        self._counter  = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "max" and score > self._best + self.min_delta) or
            (self.mode == "min" and score < self._best - self.min_delta)
        )
        if improved:
            self._best   = score
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience

    @property
    def counter(self) -> int:
        return self._counter


# ── AUC tracker ───────────────────────────────────────────────────────────────
class AUCTracker:
    def __init__(self, device: torch.device) -> None:
        self.auroc = BinaryAUROC().to(device)
        self.auprc = BinaryAveragePrecision().to(device)
        self._device = device

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits.squeeze(1).detach())
        t     = targets.long().to(self._device)
        self.auroc.update(probs.to(self._device), t)
        self.auprc.update(probs.to(self._device), t)

    def compute(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc.compute().item(),
            "auprc": self.auprc.compute().item(),
        }

    def reset(self) -> None:
        self.auroc.reset()
        self.auprc.reset()


# ── Learning rate scheduler ───────────────────────────────────────────────────
def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 5,
    min_lr_factor: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup + cosine annealing.
    """
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return min_lr_factor + (1.0 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Tracking setup ────────────────────────────────────────────────────────────
def setup_tracking(
    project: str,
    run_name: str,
    config: Dict,
    backend: str = "wandb",          # "wandb" | "mlflow" | "tensorboard"
    tracking_uri: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    if backend == "wandb":
        import wandb
        if api_key:
            wandb.login(key=api_key)
        return wandb.init(project=project, name=run_name, config=config)

    elif backend == "mlflow":
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(project)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)
        return mlflow

    elif backend == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_dir = f"runs/{project}/{run_name}"
        return SummaryWriter(log_dir)

    raise ValueError(f"Unknown tracking backend: {backend}")


def log_metrics(tracker: Any, metrics: Dict[str, float],
                step: int, backend: str = "wandb") -> None:
    if backend == "wandb":
        tracker.log(metrics, step=step)
    elif backend == "mlflow":
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    elif backend == "tensorboard":
        for k, v in metrics.items():
            tracker.add_scalar(k, v, step)


# ── DDP helpers ───────────────────────────────────────────────────────────────
def setup_ddp(rank: int, world_size: int,
              backend: str = "nccl") -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all DDP ranks."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


# ── Mixed precision context ───────────────────────────────────────────────────
def get_amp_scaler() -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler()


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
