"""
LENS ML — HF Audio Streaming Loader (ASVspoof)
Streams ASVspoof 2019 LA / 2021 LA / 2021 DF directly from HuggingFace Hub.
Native Parquet format — no download, no registration required.

Datasets:
  Bisher/ASVspoof_2019_LA          → 121K utterances, 7.54 GB
  MoaazTalab/ASVspoof_2021_LA_...  → 8.37 GB
  MoaazTalab/ASVspoof_2021_DF_...  → 36.7 GB

Usage:
  from lens_datasets.remote.hf_audio_loader import build_asvspoof_loader
  loader = build_asvspoof_loader("asvspoof19", split="train", batch_size=64)
  for mel, wav, labels in loader:
      ...  # train audio model
"""

from __future__ import annotations

import io
import logging
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

log = logging.getLogger(__name__)

try:
    import sys as _sys
    _removed = []
    for _p in list(_sys.path):
        if 'isafe2' in _p and ('ml' in _p.lower() or 'lens_datasets' in _p.lower()):
            _sys.path.remove(_p)
            _removed.append(_p)
    from datasets import load_dataset
    for _p in _removed:
        _sys.path.insert(0, _p)
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

TARGET_SR     = 16_000
MAX_DUR_S     = 4.0
MAX_SAMPLES   = int(TARGET_SR * MAX_DUR_S)   # 64000
MEL_BINS      = 128
FFT_SIZE      = 512
WIN_MS        = 25
HOP_MS        = 10
MEL_T_DIM     = 400

AUDIO_DATASET_IDS = {
    "asvspoof19":    "Bisher/ASVspoof_2019_LA",
    "asvspoof21_la": "MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized",
    "asvspoof21_df": "MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized",
}


# ── Audio processing helpers ───────────────────────────────────────────────────
def _bytes_to_waveform(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes → 16kHz mono float32 waveform."""
    try:
        import soundfile as sf
        import librosa
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    except Exception:
        wav = np.zeros(MAX_SAMPLES, dtype=np.float32)

    if len(wav) < MAX_SAMPLES:
        wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
    return wav[:MAX_SAMPLES].astype(np.float32)


def _waveform_to_mel(wav: np.ndarray) -> np.ndarray:
    """Compute 128-bin log-Mel spectrogram → (1, 128, T) float32."""
    import librosa
    win  = int(TARGET_SR * WIN_MS / 1000)
    hop  = int(TARGET_SR * HOP_MS / 1000)
    S    = librosa.feature.melspectrogram(
        y=wav, sr=TARGET_SR, n_mels=MEL_BINS, n_fft=FFT_SIZE,
        win_length=win, hop_length=hop, fmin=20, fmax=7600, power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    log_S = (log_S - log_S.mean()) / (log_S.std() + 1e-6)
    if log_S.shape[1] < MEL_T_DIM:
        log_S = np.pad(log_S, ((0, 0), (0, MEL_T_DIM - log_S.shape[1])))
    return log_S[np.newaxis, :, :MEL_T_DIM]   # (1, 128, 400)


# ── Streaming audio dataset ───────────────────────────────────────────────────
class HFAudioStreamDataset(IterableDataset):
    """
    Streams ASVspoof utterances from HuggingFace Hub.
    Yields (mel_tensor, wav_tensor, label) tuples.

    Expected HF dataset schema (Bisher/ASVspoof_2019_LA):
      audio → dict with 'bytes' and 'sampling_rate'
      label → 0 (bonafide/real) or 1 (spoof/fake)
      OR: label_str → 'bonafide' / 'spoof'
    """
    def __init__(
        self,
        hf_dataset,
        shuffle_buffer: int = 500,
        max_samples: Optional[int] = None,
        return_mel: bool = True,
        return_wav: bool = True,
    ) -> None:
        super().__init__()
        self._ds           = hf_dataset
        self.shuffle_buffer= shuffle_buffer
        self.max_samples   = max_samples
        self.return_mel    = return_mel
        self.return_wav    = return_wav

    def _extract_audio(self, item: dict) -> Optional[np.ndarray]:
        audio = item.get("audio")
        if audio is None:
            return None
        if isinstance(audio, dict):
            raw = audio.get("bytes") or audio.get("array")
            if raw is None:
                return None
            if isinstance(raw, bytes):
                return _bytes_to_waveform(raw)
            # already decoded array
            wav = np.array(raw, dtype=np.float32)
            sr  = audio.get("sampling_rate", TARGET_SR)
            if sr != TARGET_SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            if len(wav) < MAX_SAMPLES:
                wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
            return wav[:MAX_SAMPLES].astype(np.float32)
        if isinstance(audio, bytes):
            return _bytes_to_waveform(audio)
        return None

    def _extract_label(self, item: dict) -> int:
        label = item.get("label")
        if label is not None:
            return int(label)
        label_str = str(item.get("label_str", item.get("class", "spoof"))).lower()
        return 0 if "bonafide" in label_str or "real" in label_str else 1

    def __iter__(self) -> Iterator:
        ds = self._ds.shuffle(buffer_size=self.shuffle_buffer)
        count = 0
        for item in ds:
            if self.max_samples and count >= self.max_samples:
                break
            try:
                wav = self._extract_audio(item)
                if wav is None:
                    continue
                label = self._extract_label(item)

                mel_t = torch.from_numpy(_waveform_to_mel(wav)) if self.return_mel \
                        else torch.zeros(1, MEL_BINS, MEL_T_DIM)
                wav_t = torch.from_numpy(wav[np.newaxis]) if self.return_wav \
                        else torch.zeros(1, MAX_SAMPLES)

                yield mel_t, wav_t, torch.tensor(float(label))
                count += 1
            except Exception as e:
                log.debug("Audio item error: %s", e)
                continue


# ── Main builder ──────────────────────────────────────────────────────────────
def build_asvspoof_stream(
    dataset_key: str = "asvspoof19",
    split: str = "train",
    hf_token: Optional[str] = None,
    shuffle_buffer: int = 500,
    max_samples: Optional[int] = None,
) -> HFAudioStreamDataset:
    """
    Build a streaming ASVspoof dataset from HuggingFace Hub.

    Args:
        dataset_key: "asvspoof19" | "asvspoof21_la" | "asvspoof21_df"
        split:       "train" | "validation" | "test"
        hf_token:    HuggingFace token
        max_samples: cap for quick testing (None = all)

    Returns:
        HFAudioStreamDataset (PyTorch IterableDataset)
    """
    if not HF_AVAILABLE:
        raise ImportError("pip install datasets")

    dataset_id = AUDIO_DATASET_IDS.get(dataset_key, dataset_key)
    log.info("Streaming audio: %s split=%s", dataset_id, split)

    ds = load_dataset(
        dataset_id,
        split=split,
        streaming=True,
        token=hf_token,
    )
    return HFAudioStreamDataset(ds, shuffle_buffer, max_samples)


def build_asvspoof_loader(
    dataset_key: str = "asvspoof19",
    split: str = "train",
    hf_token: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """DataLoader wrapper — ready to pass directly to train_audio.py."""
    ds = build_asvspoof_stream(dataset_key, split, hf_token, max_samples=max_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("HF_TOKEN")

    print("Testing ASVspoof 2019 LA streaming...")
    loader = build_asvspoof_loader(
        "asvspoof19", split="train",
        hf_token=token, batch_size=4, max_samples=8
    )
    for mel, wav, labels in loader:
        print(f"  mel: {mel.shape}  wav: {wav.shape}  labels: {labels}")
        break
    print("ASVspoof streaming OK — 0 bytes downloaded locally.")
