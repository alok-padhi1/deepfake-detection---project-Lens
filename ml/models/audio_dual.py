"""
LENS ML — Model 3: Audio Dual Model
Branch A: ResNet-34 on 128-bin Mel-spectrograms (25ms/10ms/512 FFT/16kHz)
Branch B: RawNet3 on raw PCM waveforms with learned sinc filterbank
Shared embedding regularization loss (alignment between branches).
Both models produce a binary (real/fake) logit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════════
#  Branch A: ResNet-34 Mel-spectrogram Model
# ════════════════════════════════════════════════════════════════════════════════

class MelResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet34Mel(nn.Module):
    """
    ResNet-34 adapted for 128-bin log-Mel spectrogram input.
    Input: (B, 1, 128, T) — single-channel Mel
    Output: embedding (B, 128), logit (B, 1)
    """
    EMBED_DIM = 128

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.embed   = nn.Linear(512, self.EMBED_DIM)
        self.head    = nn.Linear(self.EMBED_DIM, 1)

    def _make_layer(self, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(MelResBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel: (B, 1, 128, T)"""
        x = self.stem(mel)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)         # (B, 512)
        x = self.dropout(x)
        emb   = F.normalize(self.embed(x), dim=-1)   # (B, 128) L2-normed
        logit = self.head(emb)                        # (B, 1)
        return {"embedding": emb, "logit": logit}


# ════════════════════════════════════════════════════════════════════════════════
#  Branch B: RawNet3 with Sinc Filterbank
# ════════════════════════════════════════════════════════════════════════════════

class SincConv(nn.Module):
    """
    Learnable Sinc-function filterbank (band-pass filters).
    Operates directly on raw waveform.
    Reference: SincNet (Ravanelli & Bengio 2018), extended in RawNet3.
    """
    def __init__(self, num_filters: int = 128, kernel_size: int = 251,
                 sample_rate: int = 16_000, min_low_hz: float = 50.0,
                 min_band_hz: float = 50.0) -> None:
        super().__init__()
        self.num_filters  = num_filters
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz

        # Initialize filter center frequencies linearly spaced (mel scale)
        low_hz  = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_lo  = self._hz_to_mel(low_hz)
        mel_hi  = self._hz_to_mel(high_hz)
        mel_pts = torch.linspace(mel_lo, mel_hi, num_filters + 1)
        hz_pts  = self._mel_to_hz(mel_pts)

        self.low_hz_  = nn.Parameter(hz_pts[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_pts[1:] - hz_pts[:-1]).unsqueeze(1))

        # Hamming window
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1,
                          dtype=torch.float32)
        self.register_buffer("n", n)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * torch.arange(kernel_size,
                                           dtype=torch.float32) / (kernel_size - 1))
        self.register_buffer("window", window)

    @staticmethod
    def _hz_to_mel(hz) -> torch.Tensor | float:
        if isinstance(hz, torch.Tensor):
            return 2595.0 * torch.log10(1.0 + hz / 700.0)
        import math
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel) -> torch.Tensor | float:
        if isinstance(mel, torch.Tensor):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T)  →  (B, num_filters, T')"""
        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2.0)
        band = (high - low)[:, 0]

        f_times_t_low  = torch.matmul(low,  self.n.unsqueeze(0))
        f_times_t_high = torch.matmul(high, self.n.unsqueeze(0))

        band_pass_low  = 2 * low  * self._sinc(f_times_t_low  * 2 / self.sample_rate)
        band_pass_high = 2 * high * self._sinc(f_times_t_high * 2 / self.sample_rate)
        band_pass      = (band_pass_high - band_pass_low) / (2 * band.unsqueeze(1))
        filters        = band_pass * self.window.unsqueeze(0)   # (F, K)
        filters        = filters.unsqueeze(1)                   # (F, 1, K)

        return F.conv1d(x, filters, stride=1,
                        padding=self.kernel_size // 2, groups=1)

    @staticmethod
    def _sinc(x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.where(x == 0, torch.tensor(1e-20, device=x.device), x)
        return torch.sin(torch.pi * x_safe) / (torch.pi * x_safe)


class RawNetBlock(nn.Module):
    """Residual block with FMS (Feature Map Scaling) for RawNet3."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        # FMS
        self.fms   = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3, inplace=True)
        x = self.bn2(self.conv2(x))
        scale = self.fms(x).unsqueeze(-1)
        x = x * scale + res
        return F.leaky_relu(x, 0.3, inplace=True)


class RawNet3(nn.Module):
    """
    Simplified RawNet3 operating on raw PCM waveform.
    Input: (B, 1, T)  where T = 16000 * max_dur (e.g. 64000 for 4s)
    Output: embedding (B, 128), logit (B, 1)
    """
    EMBED_DIM = 128

    def __init__(self, sinc_filters: int = 128, sinc_kernel: int = 251,
                 sample_rate: int = 16_000, dropout: float = 0.3) -> None:
        super().__init__()
        self.sinc = SincConv(sinc_filters, sinc_kernel, sample_rate)
        self.bn_sinc = nn.BatchNorm1d(sinc_filters)

        # Strided conv blocks to downsample
        self.strided = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(sinc_filters if i == 0 else 128, 128, 3,
                          stride=3, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.3, inplace=True),
            ) for i in range(4)
        ])

        self.res_blocks = nn.Sequential(*[RawNetBlock(128) for _ in range(6)])

        self.gru    = nn.GRU(128, 128, num_layers=2, batch_first=True,
                              bidirectional=False, dropout=dropout)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.embed  = nn.Linear(128, self.EMBED_DIM)
        self.head   = nn.Linear(self.EMBED_DIM, 1)
        self.dropout= nn.Dropout(dropout)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """waveform: (B, 1, T)"""
        x = torch.abs(self.sinc(waveform))                      # (B, F, T)
        x = F.leaky_relu(self.bn_sinc(x), 0.3, inplace=True)
        for conv in self.strided:
            x = conv(x)                                          # downsample
        x = self.res_blocks(x)                                  # (B, 128, T')
        x = x.permute(0, 2, 1)                                  # (B, T', 128)
        x, _ = self.gru(x)                                      # (B, T', 128)
        x = x.permute(0, 2, 1)                                  # (B, 128, T')
        x = self.pool(x).flatten(1)                             # (B, 128)
        x = self.dropout(x)
        emb   = F.normalize(self.embed(x), dim=-1)             # (B, 128)
        logit = self.head(emb)                                  # (B, 1)
        return {"embedding": emb, "logit": logit}


# ════════════════════════════════════════════════════════════════════════════════
#  Dual Audio Model with Shared Embedding Regularization
# ════════════════════════════════════════════════════════════════════════════════

class AudioDualModel(nn.Module):
    """
    Container: ResNet34Mel + RawNet3 + shared regularization loss.
    """
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.mel_model = ResNet34Mel(dropout)
        self.raw_model = RawNet3(dropout=dropout)

    def forward(
        self,
        mel: Optional[torch.Tensor] = None,       # (B, 1, 128, T)
        waveform: Optional[torch.Tensor] = None,  # (B, 1, T_samples)
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if mel is not None:
            m = self.mel_model(mel)
            out["mel_logit"]     = m["logit"]
            out["mel_embedding"] = m["embedding"]
        if waveform is not None:
            r = self.raw_model(waveform)
            out["raw_logit"]     = r["logit"]
            out["raw_embedding"] = r["embedding"]
        return out


class AudioDualLoss(nn.Module):
    """
    L = λ_mel * BCE(mel_logit) + λ_raw * BCE(raw_logit)
      + λ_reg * (1 - cosine_similarity(mel_emb, raw_emb))
    """
    def __init__(self, lambda_mel: float = 1.0, lambda_raw: float = 1.0,
                 lambda_reg: float = 0.2) -> None:
        super().__init__()
        self.lm  = lambda_mel
        self.lr  = lambda_raw
        self.lrg = lambda_reg
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: torch.Tensor,          # (B,) float 0/1
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=targets.device)

        if "mel_logit" in preds:
            l_mel = self.bce(preds["mel_logit"].squeeze(1), targets)
            losses["loss_mel"] = l_mel
            total = total + self.lm * l_mel

        if "raw_logit" in preds:
            l_raw = self.bce(preds["raw_logit"].squeeze(1), targets)
            losses["loss_raw"] = l_raw
            total = total + self.lr * l_raw

        if "mel_embedding" in preds and "raw_embedding" in preds:
            cos_sim = F.cosine_similarity(
                preds["mel_embedding"], preds["raw_embedding"], dim=-1
            ).mean()
            l_reg = 1.0 - cos_sim
            losses["loss_reg"] = l_reg
            total = total + self.lrg * l_reg

        losses["loss"] = total
        return losses


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T_mel, T_wav = 4, 400, 64000

    model = AudioDualModel().to(device)
    mel   = torch.randn(B, 1, 128, T_mel, device=device)
    wav   = torch.randn(B, 1, T_wav, device=device)

    t0 = time.perf_counter()
    out = model(mel=mel, waveform=wav)
    print(f"Forward: {(time.perf_counter()-t0)*1000:.1f}ms")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    targets = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)
    criterion = AudioDualLoss()
    losses = criterion(out, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
