"""
LENS ML — ASVspoof 2019 LA + 2021 DF Audio Loader
Preprocesses all audio to 16kHz mono PCM (float32 numpy arrays).
Generates Mel-spectrogram tensors (128 bins) and raw waveform tensors.

ASVspoof 2019 LA layout:
  $ASVSPOOF_ROOT/
    LA/
      ASVspoof2019_LA_train/
        flac/  (*.flac)
      ASVspoof2019_LA_dev/
        flac/
      ASVspoof2019_LA_eval/
        flac/
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

ASVspoof 2021 DF layout:
  $ASVSPOOF21_ROOT/
    flac/  (*.flac)
    ASVspoof2021.LA.cm.eval.trl.txt  (or DF equivalent)
"""

from __future__ import annotations

import logging
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

log = logging.getLogger(__name__)

TARGET_SR    = 16_000      # 16kHz
MEL_BINS     = 128
FFT_SIZE     = 512
WIN_MS       = 25          # window = 25ms
HOP_MS       = 10          # hop = 10ms
MAX_DURATION = 4.0         # seconds — pad/truncate
SPOOF_LABEL  = 1
BONAFIDE_LABEL = 0


@dataclass
class ASVspoof_Config:
    asvspoof19_root: Optional[Path]    # path to LA/ dir
    asvspoof21_root: Optional[Path]    # path to 2021 DF dir
    out_root: Path
    num_workers: int = 8
    max_duration_s: float = MAX_DURATION


# ── Protocol parser ───────────────────────────────────────────────────────────
def _parse_protocol_2019(proto_path: Path) -> pd.DataFrame:
    """
    Format: SPEAKER_ID UTERANCE_ID - SYSTEM_ID LABEL
    LABEL: spoof / bonafide
    """
    rows = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, _, system_id, label_str = parts
            rows.append({
                "utt_id":    utt_id,
                "speaker":   speaker_id,
                "system_id": system_id,
                "label":     SPOOF_LABEL if label_str == "spoof" else BONAFIDE_LABEL,
                "label_str": label_str,
            })
    return pd.DataFrame(rows)


def _parse_protocol_2021(proto_path: Path) -> pd.DataFrame:
    rows = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[1]
            label_str = parts[-1] if len(parts) >= 5 else "spoof"
            rows.append({
                "utt_id":    utt_id,
                "speaker":   parts[0] if len(parts) >= 1 else "UNK",
                "system_id": parts[3] if len(parts) >= 4 else "UNK",
                "label":     SPOOF_LABEL if label_str == "spoof" else BONAFIDE_LABEL,
                "label_str": label_str,
            })
    return pd.DataFrame(rows)


# ── Audio processing ──────────────────────────────────────────────────────────
def load_audio(path: Path, target_sr: int = TARGET_SR,
               max_dur: float = MAX_DURATION) -> np.ndarray:
    """Load flac/wav, resample to target_sr, pad/truncate to max_dur seconds."""
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)   # stereo → mono
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    max_samples = int(target_sr * max_dur)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    else:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))

    return waveform.astype(np.float32)


def compute_melspec(waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Returns log-power Mel-spectrogram (128 × T) as float32.
    win_length = 25ms, hop_length = 10ms, n_fft = 512
    """
    win_length = int(sr * WIN_MS / 1000)    # 400
    hop_length = int(sr * HOP_MS / 1000)    # 160

    S = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=MEL_BINS, n_fft=FFT_SIZE,
        win_length=win_length, hop_length=hop_length,
        fmin=20, fmax=7600, power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return log_S   # shape: (128, T)


# ── Per-file worker ───────────────────────────────────────────────────────────
def _process_file(args: tuple) -> Optional[dict]:
    (audio_path, utt_id, label, label_str, system_id,
     speaker, split, out_root, max_dur) = args
    audio_path = Path(audio_path)
    out_root   = Path(out_root)

    wav_dir = out_root / split / "wav" / label_str
    mel_dir = out_root / split / "mel" / label_str
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    wav_path = wav_dir / f"{utt_id}.npy"
    mel_path = mel_dir / f"{utt_id}.npy"

    try:
        waveform = load_audio(audio_path, TARGET_SR, max_dur)
        np.save(str(wav_path), waveform)

        mel = compute_melspec(waveform)
        np.save(str(mel_path), mel)
    except Exception as e:
        log.debug("Failed %s: %s", audio_path, e)
        return None

    return {
        "utt_id":    utt_id,
        "label":     label,
        "label_str": label_str,
        "system_id": system_id,
        "speaker":   speaker,
        "split":     split,
        "wav_path":  str(wav_path),
        "mel_path":  str(mel_path),
        "duration_s": max_dur,
        "sr":        TARGET_SR,
    }


# ── 2019 LA builder ───────────────────────────────────────────────────────────
def _build_2019(cfg: ASVspoof_Config) -> List[dict]:
    root = Path(cfg.asvspoof19_root) / "LA"
    proto_root = root / "ASVspoof2019_LA_cm_protocols"

    SPLIT_MAP = {
        "train": (root / "ASVspoof2019_LA_train" / "flac",
                  proto_root / "ASVspoof2019.LA.cm.train.trn.txt"),
        "val":   (root / "ASVspoof2019_LA_dev"   / "flac",
                  proto_root / "ASVspoof2019.LA.cm.dev.trl.txt"),
        "test":  (root / "ASVspoof2019_LA_eval"  / "flac",
                  proto_root / "ASVspoof2019.LA.cm.eval.trl.txt"),
    }

    tasks: List[tuple] = []
    for split, (flac_dir, proto_file) in SPLIT_MAP.items():
        if not proto_file.exists():
            log.warning("Missing protocol: %s", proto_file)
            continue
        meta = _parse_protocol_2019(proto_file)
        for _, row in meta.iterrows():
            flac = flac_dir / f"{row['utt_id']}.flac"
            if not flac.exists():
                continue
            tasks.append((
                str(flac), row["utt_id"], row["label"], row["label_str"],
                row["system_id"], row["speaker"], split,
                str(cfg.out_root / "asvspoof19"), cfg.max_duration_s
            ))
    return tasks


# ── 2021 DF builder ───────────────────────────────────────────────────────────
def _build_2021(cfg: ASVspoof_Config) -> List[tuple]:
    root = Path(cfg.asvspoof21_root)
    flac_dir = root / "flac"

    proto_candidates = list(root.glob("*.trl.txt")) + list(root.glob("*.trn.txt"))
    if not proto_candidates:
        log.warning("No protocol file found in %s", root)
        return []
    proto_file = proto_candidates[0]
    meta = _parse_protocol_2021(proto_file)

    tasks: List[tuple] = []
    for _, row in meta.iterrows():
        flac = flac_dir / f"{row['utt_id']}.flac"
        if not flac.exists():
            continue
        tasks.append((
            str(flac), row["utt_id"], row["label"], row["label_str"],
            row["system_id"], row["speaker"], "test",
            str(cfg.out_root / "asvspoof21"), cfg.max_duration_s
        ))
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────
def build_asvspoof(cfg: ASVspoof_Config) -> pd.DataFrame:
    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    all_tasks: List[tuple] = []

    if cfg.asvspoof19_root:
        all_tasks.extend(_build_2019(cfg))
        log.info("ASVspoof 2019 tasks: %d", len(all_tasks))

    prev = len(all_tasks)
    if cfg.asvspoof21_root:
        all_tasks.extend(_build_2021(cfg))
        log.info("ASVspoof 2021 tasks: %d", len(all_tasks) - prev)

    all_records: List[dict] = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        futs = {ex.submit(_process_file, t): t for t in all_tasks}
        with tqdm(total=len(all_tasks), desc="ASVspoof audio") as bar:
            for fut in as_completed(futs):
                rec = fut.result()
                if rec:
                    all_records.append(rec)
                bar.update(1)

    df = pd.DataFrame(all_records)
    manifest = Path(cfg.out_root) / "asvspoof_manifest.parquet"
    df.to_parquet(manifest, index=False)
    log.info("ASVspoof done: %d utterances → %s", len(df), manifest)
    return df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LENS ASVspoof Loader")
    p.add_argument("--asvspoof19-root", type=Path, default=None)
    p.add_argument("--asvspoof21-root", type=Path, default=None)
    p.add_argument("--out-root",        required=True, type=Path)
    p.add_argument("--workers",         type=int, default=8)
    p.add_argument("--max-dur",         type=float, default=4.0)
    args = p.parse_args()

    cfg = ASVspoof_Config(
        asvspoof19_root=args.asvspoof19_root,
        asvspoof21_root=args.asvspoof21_root,
        out_root=args.out_root,
        num_workers=args.workers,
        max_duration_s=args.max_dur,
    )
    df = build_asvspoof(cfg)
    if len(df) > 0:
        print(df.groupby(["split", "label_str"]).size().to_string())
