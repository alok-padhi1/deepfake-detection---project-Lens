"""
LENS ML — TorchServe Audio Handler (ResNet34Mel + RawNet3)
Input:  audio bytes (WAV or FLAC)
Output: JSON with fake_probability per model + ensemble
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from ts.torch_handler.base_handler import BaseHandler

log = logging.getLogger(__name__)

TARGET_SR    = 16_000
MAX_DUR_S    = 4.0
MAX_SAMPLES  = int(TARGET_SR * MAX_DUR_S)
MEL_BINS     = 128
WIN_MS       = 25
HOP_MS       = 10
FFT_SIZE     = 512
MEL_T_DIM    = 400


class AudioHandler(BaseHandler):
    def initialize(self, context) -> None:
        super().initialize(context)
        self._branch = context.model_yaml_config.get("branch", "both")
        # "mel" | "raw" | "both"

    def _load_audio(self, body: bytes) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(body)
            tmp = f.name
        try:
            wav, sr = sf.read(tmp, dtype="float32", always_2d=False)
        finally:
            os.unlink(tmp)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        if len(wav) < MAX_SAMPLES:
            wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
        return wav[:MAX_SAMPLES]

    def _compute_mel(self, wav: np.ndarray) -> np.ndarray:
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
        return log_S[:, :MEL_T_DIM][np.newaxis]   # (1, 128, 400)

    def preprocess(self, data: List[Dict]):
        mels, wavs = [], []
        for item in data:
            body = item.get("body") or item.get("data", b"")
            if isinstance(body, str):
                body = base64.b64decode(body)
            wav  = self._load_audio(body)
            mel  = self._compute_mel(wav)
            mels.append(torch.from_numpy(mel))
            wavs.append(torch.from_numpy(wav[np.newaxis]))   # (1, T)
        return (torch.stack(mels).to(self.device),
                torch.stack(wavs).to(self.device))

    def inference(self, data):
        mel_t, wav_t = data
        with torch.no_grad():
            out = self.model(mel=mel_t, waveform=wav_t)
        return out

    def postprocess(self, data: Dict) -> List[Dict]:
        results = []
        B = (data.get("mel_logit") or data.get("raw_logit")).shape[0]
        for i in range(B):
            item: Dict = {}
            if "mel_logit" in data:
                item["mel_fake_prob"]  = round(
                    float(torch.sigmoid(data["mel_logit"][i]).item()), 4)
            if "raw_logit" in data:
                item["raw_fake_prob"]  = round(
                    float(torch.sigmoid(data["raw_logit"][i]).item()), 4)
            # Ensemble average
            probs = [v for k, v in item.items() if k.endswith("_fake_prob")]
            item["fake_probability"] = round(float(np.mean(probs)), 4)
            item["is_fake"]          = bool(item["fake_probability"] > 0.5)
            item["model"]            = "audio-dual"
            results.append(item)
        return results
