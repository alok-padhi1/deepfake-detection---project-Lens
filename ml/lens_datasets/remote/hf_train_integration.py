"""
LENS ML — HuggingFace Streaming Training Integration
Drop-in replacement for local DataLoader when using HF streaming.
Plug into train_efficientnet.py with --use-hf-streaming flag.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def build_hf_train_loader(
    hf_token: str,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    shuffle_buffer: int = 2000,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Build a DataLoader streaming GenImage from HuggingFace Hub.
    Zero local storage. Ready to pass to train_efficientnet.py.
    """
    from lens_datasets.remote.hf_streaming_loader import (
        build_genimage_stream, HFStreamDataset
    )
    from lens_datasets.augmentation import build_train_augmentation

    log.info("Building HF streaming loader (train)...")
    ds_stream = build_genimage_stream(
        split="train",
        hf_token=hf_token,
        image_size=image_size,
    )
    aug = build_train_augmentation(image_size)
    wrapped = HFStreamDataset(
        "train", ds_stream,
        transform=aug,
        image_size=image_size,
        shuffle_buffer=shuffle_buffer,
        max_samples=max_samples,
    )
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_hf_val_loader(
    hf_token: str,
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 4,
    max_samples: Optional[int] = 5000,
) -> DataLoader:
    from lens_datasets.remote.hf_streaming_loader import (
        build_genimage_stream, HFStreamDataset
    )
    from lens_datasets.augmentation import build_val_augmentation

    log.info("Building HF streaming loader (val)...")
    ds_stream = build_genimage_stream(
        split="validation",
        hf_token=hf_token,
        image_size=image_size,
    )
    aug = build_val_augmentation(image_size)
    wrapped = HFStreamDataset(
        "val", ds_stream,
        transform=aug,
        image_size=image_size,
        shuffle_buffer=100,
        max_samples=max_samples,
    )
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
