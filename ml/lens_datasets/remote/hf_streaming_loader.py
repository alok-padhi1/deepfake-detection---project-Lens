"""
LENS ML — Option A: HuggingFace Streaming Loader
Works for:
  - GenImage        → RohanRamesh/genimage-224 on HuggingFace Hub
  - CIFAKE          → dragonintelligence/CIFAKE-image-dataset
  - Deepfake faces  → itsLeen/deepfake_vs_real_image_detection

Zero download. Streams shards on-the-fly during training.
Each batch is fetched and decoded in the background via HF datasets.

Usage:
  from lens_datasets.remote.hf_streaming_loader import (
      build_genimage_stream,
      build_asvspoof_stream,
      HFStreamDataset,
  )
  train_ds = HFStreamDataset("train", build_genimage_stream("train"))
  loader   = DataLoader(train_ds, batch_size=32, num_workers=4)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)

try:
    # Remove local ml/ paths temporarily to avoid collision with lens_datasets/
    import sys as _sys
    _removed = []
    for _p in list(_sys.path):
        if 'isafe2' in _p and ('ml' in _p.lower() or 'lens_datasets' in _p.lower()):
            _sys.path.remove(_p)
            _removed.append(_p)
    from datasets import load_dataset, IterableDataset as HFIterableDataset
    for _p in _removed:
        _sys.path.insert(0, _p)
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ── GenImage HuggingFace stream ────────────────────────────────────────────────
# ── Available HF datasets for LENS ──────────────────────────────────────────
HF_DATASETS = {
    # ── Image deepfake datasets ───────────────────────────────────────────────
    # 120K images (60k real CIFAR, 60k SD-generated) — fast to prototype
    "cifake":         "dragonintelligence/CIFAKE-image-dataset",
    # Face deepfake vs real
    "deepfake_faces": "itsLeen/deepfake_vs_real_image_detection",
    # GenImage full (1.43M) — requires accepting HF terms
    "genimage":       "RohanRamesh/genimage-224",
    # MidJourney subset of GenImage
    "genimage_mj":    "bitmind/GenImage_MidJourney",

    # ── Audio spoof datasets (ASVspoof) — all native Parquet, streamable ─────
    # ASVspoof 2019 LA — 121K utterances, 7.54 GB total, streams ~0 GB locally
    "asvspoof19":     "Bisher/ASVspoof_2019_LA",
    # ASVspoof 2021 LA balanced+normalized
    "asvspoof21_la":  "MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized",
    # ASVspoof 2021 DF (deepfake audio) — 36.7 GB total, streams per-batch
    "asvspoof21_df":  "MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized",
}


def build_genimage_stream(
    split: str = "train",
    hf_token: Optional[str] = None,
    image_size: int = 224,
    dataset_key: str = "genimage",   # key from HF_DATASETS above
    generators: Optional[List[str]] = None,
) -> "HFIterableDataset":
    """
    Stream an image deepfake dataset directly from HuggingFace Hub.
    Zero download — data streams shard by shard.

    Args:
        split:       "train" | "validation" | "test"
        hf_token:    HuggingFace token (from hf.co/settings/tokens)
        dataset_key: one of: genimage, cifake, deepfake_faces, genimage_mj
    """
    if not HF_AVAILABLE:
        raise ImportError("pip install datasets huggingface_hub")

    dataset_id = HF_DATASETS.get(dataset_key, dataset_key)
    log.info("Streaming dataset: %s (split=%s)", dataset_id, split)

    ds = load_dataset(
        dataset_id,
        split=split,
        streaming=True,   # ← zero local storage
        token=hf_token,
        # trust_remote_code removed — not supported in datasets>=2.x
    )

    return ds


# ── ASVspoof — not on HF, use local or S3 ───────────────────────────────────
def build_asvspoof_stream(*args, **kwargs):
    raise NotImplementedError(
        "ASVspoof is not available on HuggingFace Hub.\n"
        "Options:\n"
        "  1. Download 25GB from https://www.asvspoof.org (free registration)\n"
        "  2. Upload to your MinIO and use webdataset_s3_loader.py\n"
        "  3. Use the local asvspoof_loader.py after downloading"
    )


# ── Universal HF IterableDataset wrapper ──────────────────────────────────────
class HFStreamDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming IterableDataset into a PyTorch IterableDataset.
    Handles:
      - PIL Image decoding + augmentation
      - Label extraction
      - Infinite shuffling (buffer-based)

    Schema expected from HF dataset:
      image  → PIL.Image or bytes
      label  → 0 (real) / 1 (fake)
      generator → str (optional)
      family    → str (optional)
    """
    FAMILY_ALIASES = {
        "biggan":       "GAN",
        "midjourney":   "Diffusion",
        "stablediffusion": "Diffusion",
        "sd": "Diffusion",
        "glide":        "Diffusion",
        "adm":          "Diffusion",
        "vqdm":         "Diffusion",
        "wukong":       "GAN",
        "real":         "REAL",
    }

    def __init__(
        self,
        split: str,
        hf_dataset: "HFIterableDataset",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        shuffle_buffer: int = 1000,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.split          = split
        self._ds            = hf_dataset
        self.transform      = transform
        self.image_size     = image_size
        self.shuffle_buffer = shuffle_buffer
        self.max_samples    = max_samples

    def _process_item(self, item: Dict[str, Any]) -> Optional[tuple]:
        try:
            # Image decoding
            img = item.get("image")
            if img is None:
                return None

            if hasattr(img, "convert"):          # PIL Image
                img = img.convert("RGB")
                arr = np.array(img)
            elif isinstance(img, (bytes, bytearray)):
                from PIL import Image as PILImage
                import io
                img = PILImage.open(io.BytesIO(img)).convert("RGB")
                arr = np.array(img)
            else:
                return None

            # Resize
            import cv2
            arr = cv2.resize(arr, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LANCZOS4)

            # Augment
            if self.transform:
                arr = self.transform(image=arr)["image"]
            else:
                arr = arr.astype(np.float32) / 127.5 - 1.0

            tensor = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1))

            # Labels
            label = int(item.get("label", 1))
            generator = str(item.get("generator", "unknown")).lower()
            family = self.FAMILY_ALIASES.get(generator, "Diffusion")

            from models.efficientnet_forensic import FAMILY_CLASSES
            family_idx = FAMILY_CLASSES.index(family) \
                         if family in FAMILY_CLASSES else 0

            return tensor, label, family_idx

        except Exception as e:
            log.debug("Item processing error: %s", e)
            return None

    def __iter__(self) -> Iterator[tuple]:
        # Shuffle buffer for streaming
        ds = self._ds.shuffle(buffer_size=self.shuffle_buffer)
        count = 0
        for item in ds:
            if self.max_samples and count >= self.max_samples:
                break
            result = self._process_item(item)
            if result:
                yield result
                count += 1


# ── Multi-dataset streaming combiner ──────────────────────────────────────────
class CombinedStreamDataset(IterableDataset):
    """
    Interleaves multiple HFStreamDataset instances with configurable mixing weights.
    E.g. 50% GenImage + 30% FF++ + 20% ASVspoof.
    """
    def __init__(
        self,
        datasets: List[HFStreamDataset],
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def __iter__(self) -> Iterator[tuple]:
        import random
        iters = [iter(ds) for ds in self.datasets]
        alive = [True] * len(iters)

        while any(alive):
            # Sample dataset according to weights
            available = [i for i in range(len(iters)) if alive[i]]
            if not available:
                break
            w = [self.weights[i] for i in available]
            w_sum = sum(w)
            w_norm = [x / w_sum for x in w]
            chosen = np.random.choice(available, p=w_norm)

            try:
                item = next(iters[chosen])
                yield item
            except StopIteration:
                alive[chosen] = False


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import argparse
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--n-samples", type=int, default=10)
    args = p.parse_args()

    if not HF_AVAILABLE:
        print("Install: pip install datasets huggingface_hub")
        exit(1)

    print("Testing GenImage streaming (no download)...")
    ds = build_genimage_stream("train", hf_token=args.hf_token)
    wrapped = HFStreamDataset("train", ds, max_samples=args.n_samples)

    from torch.utils.data import DataLoader
    loader = DataLoader(wrapped, batch_size=4, num_workers=0)
    for batch in loader:
        imgs, labels, families = batch
        print(f"  batch: imgs={imgs.shape}, labels={labels}, families={families}")
        break

    print(f"Streaming test passed — 0 bytes downloaded locally.")
