"""
LENS ML — Unified Augmentation Pipeline
Applies stochastic augmentations matching real-world distribution shifts.

Augmentations:
  1. JPEG recompression         Q ∈ [55, 95]
  2. Gaussian blur              σ ∈ [0.5, 2.0]
  3. Random crop + resize       crop ratio ∈ [0.75, 1.0]
  4. Color jitter               brightness, contrast, saturation, hue
  5. H.264 re-encode (video)    CRF ∈ [23, 33] (target mean=28)

Two modes:
  • Image augmentation:  returns augmented PIL/tensor via albumentations
  • Video augmentation:  applies H.264 re-encode on a clip via FFmpeg subprocess
"""

from __future__ import annotations

import io
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageFilter


# ── JPEG round-trip via PIL ────────────────────────────────────────────────────
class JPEGRoundtrip(A.ImageOnlyTransform):
    """Encode image to JPEG in-memory then decode back."""
    def __init__(self, quality_low: int = 55, quality_high: int = 95, p: float = 0.5):
        super().__init__(p=p)
        self.quality_low  = quality_low
        self.quality_high = quality_high

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        quality = random.randint(self.quality_low, self.quality_high)
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))

    def get_transform_init_args_names(self) -> tuple:
        return ("quality_low", "quality_high")


# ── Training augmentation pipeline ────────────────────────────────────────────
def build_train_augmentation(
    image_size: int = 224,
    jpeg_p: float = 0.5,
    blur_p: float = 0.3,
    color_p: float = 0.4,
) -> A.Compose:
    """
    Returns an albumentations Compose pipeline for training.
    Input: H×W×3 uint8 numpy array.
    Output: image_size × image_size × 3 float32 in [-1, 1].
    """
    return A.Compose([
        # ── Spatial ───────────────────────────────────────────────────────────
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.75, 1.0), ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LANCZOS4,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3
        ),

        # ── Color ─────────────────────────────────────────────────────────────
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05, p=color_p
        ),
        A.ToGray(p=0.05),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # ── Compression artifacts ─────────────────────────────────────────────
        JPEGRoundtrip(quality_low=55, quality_high=95, p=jpeg_p),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.0), p=blur_p),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.15),
        A.ImageCompression(quality_range=(60, 100), p=0.2),

        # ── Grid distort (subtle) ─────────────────────────────────────────────
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),

        # ── Normalize to [-1, 1] ──────────────────────────────────────────────
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            max_pixel_value=255.0
        ),
    ])


# ── Validation / test pipeline (no augmentation) ──────────────────────────────
def build_val_augmentation(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            max_pixel_value=255.0
        ),
    ])


# ── Video H.264 re-encode ──────────────────────────────────────────────────────
def h264_reencode(
    frames: List[np.ndarray],
    fps: int = 30,
    crf: Optional[int] = None,
    crf_range: Tuple[int, int] = (23, 33),
) -> List[np.ndarray]:
    """
    Re-encode a list of BGR frames through H.264 (CRF=28 default) and decode back.
    Requires ffmpeg on PATH.

    Args:
        frames:    List of (H, W, 3) uint8 BGR numpy arrays
        fps:       Frame rate
        crf:       Fixed CRF; if None, samples uniformly from crf_range
        crf_range: (min, max) CRF when crf is None

    Returns:
        List of (H, W, 3) uint8 BGR numpy arrays after re-encoding
    """
    if not frames:
        return frames
    actual_crf = crf if crf is not None else random.randint(*crf_range)
    h, w = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as tmp:
        in_pattern  = os.path.join(tmp, "frame_%05d.png")
        out_file    = os.path.join(tmp, "clip.mp4")
        out_pattern = os.path.join(tmp, "out_%05d.png")

        for i, frame in enumerate(frames):
            cv2.imwrite(in_pattern % (i + 1), frame)

        encode_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", in_pattern,
            "-c:v", "libx264",
            "-crf", str(actual_crf),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            out_file,
        ]
        subprocess.run(encode_cmd, check=True, capture_output=True)

        decode_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", out_file,
            out_pattern,
        ]
        subprocess.run(decode_cmd, check=True, capture_output=True)

        decoded: List[np.ndarray] = []
        for i in range(len(frames)):
            frame_path = out_pattern % (i + 1)
            if os.path.exists(frame_path):
                decoded.append(cv2.imread(frame_path))
            else:
                decoded.append(frames[i])   # fallback to original

    return decoded


# ── Clip augmentation (applies H.264 + image augs per frame) ──────────────────
def augment_clip(
    frames: List[np.ndarray],          # BGR uint8
    image_size: int = 224,
    apply_h264: bool = True,
    h264_p: float = 0.4,
    jpeg_p: float = 0.3,
    blur_p: float = 0.2,
) -> np.ndarray:
    """
    Augment a video clip (list of BGR frames).
    Returns: float32 tensor (T, C, H, W) normalized to [-1, 1]
    """
    import torch

    if apply_h264 and random.random() < h264_p:
        frames = h264_reencode(frames)

    aug = build_train_augmentation(image_size, jpeg_p=jpeg_p, blur_p=blur_p)
    # Apply the SAME spatial transform to every frame for temporal consistency
    replay = aug(image=cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    augmented = [replay["image"]]
    for frame in frames[1:]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = A.ReplayCompose.replay(replay, image=rgb)
        augmented.append(out["image"])

    arr = np.stack(augmented, axis=0)               # (T, H, W, C) float32
    arr = np.transpose(arr, (0, 3, 1, 2))           # (T, C, H, W)
    return arr


# ── Mel-spectrogram augmentation ──────────────────────────────────────────────
def augment_melspec(mel: np.ndarray,
                    freq_mask_max: int = 20,
                    time_mask_max: int = 20,
                    num_freq_masks: int = 2,
                    num_time_masks: int = 2) -> np.ndarray:
    """
    Apply SpecAugment (frequency + time masking) to a (128, T) Mel-spectrogram.
    Returns augmented (128, T) float32 array.
    """
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_max)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0:f0 + f, :] = mel.min()

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_max)
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, t0:t0 + t] = mel.min()

    return mel


if __name__ == "__main__":
    # Quick smoke test
    import time
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    aug = build_train_augmentation()
    t0 = time.perf_counter()
    for _ in range(100):
        aug(image=dummy)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"100 augmentations: {elapsed:.1f}ms ({elapsed/100:.2f}ms/img)")

    mel = np.random.randn(128, 400).astype(np.float32)
    aug_mel = augment_melspec(mel)
    print(f"Mel aug shape: {aug_mel.shape} — min={aug_mel.min():.3f}")
