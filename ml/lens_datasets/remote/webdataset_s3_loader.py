"""
LENS ML — Option B: WebDataset + S3/GCS Streaming Loader
Works for: DFDC, FaceForensics++, any large video/image dataset stored in cloud.

WebDataset = tar shards stored in S3/GCS.
Training reads shards sequentially over HTTP(S) — no full download needed.
Only 1-2 shards buffered in memory at a time (~500MB max).

Workflow:
  1. Upload dataset to S3 once (one-time)
  2. Convert to WebDataset .tar shards (script included below)
  3. Train directly from s3:// URL — zero local storage

Install:
  pip install webdataset boto3 s3fs

Usage:
  from lens_datasets.remote.webdataset_s3_loader import build_s3_loader
  loader = build_s3_loader(
      s3_urls="s3://my-bucket/lens/dfdc/train/shard-{000000..000499}.tar",
      batch_size=64,
  )
"""

from __future__ import annotations

import io
import logging
import os
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

try:
    import webdataset as wds
    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False
    log.warning("pip install webdataset  ← required for S3 streaming")


# ── S3 authentication helper ───────────────────────────────────────────────────
def _configure_s3_env(
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
    aws_region:     str = "us-east-1",
    endpoint_url:   Optional[str] = None,   # for MinIO or custom S3
) -> None:
    """Set AWS credentials in environment for webdataset to pick up."""
    if aws_access_key:
        os.environ["AWS_ACCESS_KEY_ID"]     = aws_access_key
    if aws_secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
    os.environ["AWS_DEFAULT_REGION"] = aws_region
    if endpoint_url:
        os.environ["AWS_ENDPOINT_URL"] = endpoint_url   # MinIO support


# ── Image decoding ─────────────────────────────────────────────────────────────
def _decode_image(data: bytes, image_size: int = 224) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    return img


# ── Build WebDataset pipeline from S3 ─────────────────────────────────────────
def build_s3_loader(
    s3_urls: str,
    batch_size: int = 64,
    image_size: int = 224,
    transform: Optional[Callable] = None,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    resampled: bool = True,        # enable infinite resampling for DDP
    shardshuffle: int = 500,
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
    aws_region:     str = "us-east-1",
    endpoint_url:   Optional[str] = None,
) -> DataLoader:
    """
    Build a DataLoader that streams directly from S3 WebDataset shards.

    Args:
        s3_urls:     Brace-expanded S3 URL e.g.
                     "s3://bucket/dfdc/train/shard-{000000..000499}.tar"
                     OR list of URLs
        batch_size:  samples per batch
        image_size:  resize target
        transform:   albumentations Compose (optional)
        resampled:   True for infinite DDP-safe stream
        endpoint_url: for MinIO/custom S3: "http://minio:9000"
    """
    if not WDS_AVAILABLE:
        raise ImportError("pip install webdataset")

    _configure_s3_env(aws_access_key, aws_secret_key, aws_region, endpoint_url)

    def preprocess(sample: dict) -> Optional[Tuple]:
        # WebDataset keys: __key__, jpg/png, cls.txt / label.txt
        img_bytes = sample.get("jpg") or sample.get("png") or sample.get("jpeg")
        if img_bytes is None:
            return None

        img = _decode_image(img_bytes, image_size)
        if img is None:
            return None

        if transform:
            img = transform(image=img)["image"].astype(np.float32)
        else:
            img = img.astype(np.float32) / 127.5 - 1.0

        tensor = torch.from_numpy(img.transpose(2, 0, 1))   # (C, H, W)

        # Label
        label_bytes = sample.get("cls") or sample.get("label") or b"1"
        if isinstance(label_bytes, bytes):
            label_str = label_bytes.decode("utf-8").strip()
        else:
            label_str = str(label_bytes).strip()
        label = int(label_str) if label_str.isdigit() else 1

        # Family
        meta_bytes = sample.get("json") or b"{}"
        if isinstance(meta_bytes, bytes):
            import json
            meta = json.loads(meta_bytes)
        else:
            meta = {}
        family = meta.get("family", "GAN")
        from models.efficientnet_forensic import FAMILY_CLASSES
        family_idx = FAMILY_CLASSES.index(family) if family in FAMILY_CLASSES else 1

        return tensor, label, family_idx

    pipeline = (
        wds.WebDataset(
            s3_urls,
            resampled=resampled,
            shardshuffle=shardshuffle,
            nodesplitter=wds.split_by_node,       # DDP-safe shard splitting
            handler=wds.warn_and_continue,
        )
        .shuffle(shuffle_buffer)
        .map(preprocess, handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=True)
    )

    return DataLoader(
        pipeline,
        batch_size=None,       # already batched by WebDataset
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


# ── Convert local dataset to WebDataset shards ────────────────────────────────
def create_wds_shards(
    manifest_parquet: str,
    output_dir: str,
    shard_size_mb: int = 512,
    image_size: int = 224,
    upload_to_s3: bool = False,
    s3_bucket: str = "",
    s3_prefix: str = "lens/shards/",
) -> List[str]:
    """
    Convert a Parquet manifest (produced by our loaders) into WebDataset .tar shards.
    Optionally uploads directly to S3.

    Returns list of shard paths.
    """
    import json
    import tarfile
    import tempfile
    from pathlib import Path

    import pandas as pd

    df = pd.read_parquet(manifest_parquet)
    log.info("Converting %d samples to WebDataset shards", len(df))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx  = 0
    shard_size = 0
    shard_paths = []
    current_tar = None

    def _open_shard():
        nonlocal shard_idx, shard_size, current_tar
        fname = out_dir / f"shard-{shard_idx:06d}.tar"
        current_tar = tarfile.open(fname, "w")
        shard_size  = 0
        shard_paths.append(str(fname))
        return current_tar

    def _close_shard():
        if current_tar:
            current_tar.close()
        if upload_to_s3 and s3_bucket and shard_paths:
            _upload_shard(shard_paths[-1], s3_bucket, s3_prefix)

    def _upload_shard(path: str, bucket: str, prefix: str) -> None:
        import boto3
        s3 = boto3.client("s3")
        key = prefix + Path(path).name
        s3.upload_file(path, bucket, key)
        log.info("Uploaded shard to s3://%s/%s", bucket, key)
        if upload_to_s3:
            Path(path).unlink()   # free local disk after upload

    def _add_file(tar: tarfile.TarFile, key: str,
                  data: bytes, ext: str) -> None:
        info = tarfile.TarInfo(name=f"{key}.{ext}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    import json as _json

    _open_shard()
    for i, row in df.iterrows():
        path = str(row.get("dst_path") or row.get("out_path") or row.get("src_path", ""))
        if not path or not Path(path).exists():
            continue

        with open(path, "rb") as f:
            img_bytes = f.read()

        # Re-encode to JPEG at target size to normalize
        arr = _decode_image(img_bytes, image_size)
        if arr is None:
            continue
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_bytes = encoded.tobytes()

        key      = f"{i:09d}"
        label    = int(row.get("label", 1))
        family   = str(row.get("family", "GAN"))
        meta     = _json.dumps({"family": family,
                                 "generator": str(row.get("generator", "")),
                                 "split": str(row.get("split", "train"))})

        _add_file(current_tar, key, img_bytes,      "jpg")
        _add_file(current_tar, key, str(label).encode(), "cls")
        _add_file(current_tar, key, meta.encode(),  "json")

        shard_size += len(img_bytes)
        if shard_size >= shard_size_mb * 1024 * 1024:
            _close_shard()
            shard_idx += 1
            _open_shard()

    _close_shard()
    log.info("Created %d shards in %s", len(shard_paths), out_dir)
    return shard_paths


# ── GCS variant ───────────────────────────────────────────────────────────────
def build_gcs_loader(
    gcs_urls: str,        # "gs://bucket/lens/dfdc/shard-{000000..000499}.tar"
    batch_size: int = 64,
    image_size: int = 224,
    transform=None,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Same as build_s3_loader but for Google Cloud Storage.
    Set GOOGLE_APPLICATION_CREDENTIALS env var to service account JSON.
    """
    # webdataset natively handles gs:// via gcsfs
    return build_s3_loader(
        gcs_urls, batch_size, image_size,
        transform, num_workers, **kwargs
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument("--s3-url",  help="e.g. s3://bucket/shards/shard-{000000..000004}.tar")
    p.add_argument("--convert-manifest", help="Parquet to convert to shards first")
    p.add_argument("--shard-out", default="/tmp/shards")
    p.add_argument("--s3-bucket", default="")
    p.add_argument("--endpoint", default=None, help="MinIO endpoint")
    args = p.parse_args()

    if args.convert_manifest:
        shards = create_wds_shards(
            args.convert_manifest, args.shard_out,
            upload_to_s3=bool(args.s3_bucket),
            s3_bucket=args.s3_bucket,
        )
        print(f"Created {len(shards)} shards")

    if args.s3_url:
        loader = build_s3_loader(
            args.s3_url, batch_size=8,
            endpoint_url=args.endpoint
        )
        for imgs, labels, families in loader:
            print(f"S3 stream batch: {imgs.shape}, labels={labels}")
            break
        print("S3 streaming test passed.")
