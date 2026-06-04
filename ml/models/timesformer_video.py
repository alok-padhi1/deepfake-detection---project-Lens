"""
LENS ML — Model 2: TimeSformer Video Temporal Forensic Model
Pretrained on Kinetics-600.
Spatial attention frozen for first 10 epochs, then unfrozen for joint fine-tuning.
8-frame clip input at 224×224.
Sliding window inference with stride=4, max-pool aggregation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import TimesformerModel, TimesformerConfig


# ── Spatial attention freeze/unfreeze utilities ───────────────────────────────
def _get_spatial_attention_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all parameters belonging to spatial (non-temporal) attention."""
    params = []
    for name, param in model.named_parameters():
        # TimeSformer spatial blocks = "timesformer.encoder.layer.*.attention.attention"
        # but NOT "temporal_attention"
        if "attention" in name and "temporal" not in name:
            params.append(param)
    return params


def freeze_spatial_attention(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "attention" in name and "temporal" not in name:
            param.requires_grad_(False)


def unfreeze_spatial_attention(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "attention" in name and "temporal" not in name:
            param.requires_grad_(True)


# ── Binary classification head ────────────────────────────────────────────────
class VideoForensicHead(nn.Module):
    def __init__(self, hidden_size: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),   # binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full TimeSformer Forensic Model ──────────────────────────────────────────
class TimeSformerForensic(nn.Module):
    """
    TimeSformer backbone (divided space-time attention) + binary forensic head.

    Usage:
        model = TimeSformerForensic.from_pretrained("facebook/timesformer-base-finetuned-k600")
    """
    NUM_FRAMES   = 8
    IMAGE_SIZE   = 224
    PATCH_SIZE   = 16

    def __init__(self, config: Optional[TimesformerConfig] = None,
                 dropout: float = 0.3) -> None:
        super().__init__()
        if config is None:
            config = TimesformerConfig(
                image_size=self.IMAGE_SIZE,
                patch_size=self.PATCH_SIZE,
                num_frames=self.NUM_FRAMES,
                num_channels=3,
                num_labels=1,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                attention_type="divided_space_time",
            )
        self.timesformer = TimesformerModel(config)
        self.head = VideoForensicHead(config.hidden_size, dropout)

    @classmethod
    def from_pretrained(cls, ckpt_path: str, **kwargs) -> "TimeSformerForensic":
        """Load Kinetics-600 pretrained weights, replacing classification head."""
        obj = cls(**kwargs)
        state = torch.load(ckpt_path, map_location="cpu")
        # Strip final classifier weights from pretrained checkpoint
        model_state = state.get("model", state)
        incompatible = obj.timesformer.load_state_dict(
            {k.replace("timesformer.", ""): v for k, v in model_state.items()
             if "head" not in k and "classifier" not in k},
            strict=False
        )
        print(f"TimeSformer pretrained load: missing={incompatible.missing_keys[:5]}")
        return obj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, T, C, H, W) — T=8 frames
        Returns:
            logit: (B, 1)
        """
        # HuggingFace TimeSformer expects (B, T, C, H, W)
        outputs = self.timesformer(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]   # (B, hidden_size)
        return self.head(cls_token)


# ── Sliding window inference ──────────────────────────────────────────────────
@torch.no_grad()
def sliding_window_inference(
    model: TimeSformerForensic,
    frames: torch.Tensor,            # (T_total, C, H, W) — full video
    clip_len: int = 8,
    stride:   int = 4,
    device:   str = "cuda",
) -> float:
    """
    Run sliding window inference over a video.
    Args:
        frames:   (T_total, C, H, W) float tensor, normalized
        clip_len: frames per clip (=8)
        stride:   window stride (=4)
    Returns:
        aggregated_score: float — max-pooled sigmoid probability across windows
    """
    model.eval()
    T = frames.shape[0]
    starts = list(range(0, max(1, T - clip_len + 1), stride))
    if not starts:
        starts = [0]

    clip_scores: List[float] = []
    for s in starts:
        end = min(s + clip_len, T)
        clip = frames[s:end]
        # Pad last clip if shorter than clip_len
        if clip.shape[0] < clip_len:
            pad = frames[-1:].expand(clip_len - clip.shape[0], -1, -1, -1)
            clip = torch.cat([clip, pad], dim=0)

        clip = clip.unsqueeze(0).to(device)   # (1, T, C, H, W)
        logit = model(clip)                    # (1, 1)
        score = torch.sigmoid(logit).item()
        clip_scores.append(score)

    # Max-pool aggregation: conservative — take worst-case (highest fake probability)
    return float(max(clip_scores))


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimeSformerForensic().to(device)
    model.eval()

    # Freeze spatial attn
    freeze_spatial_attention(model)
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"Frozen spatial attn: {frozen}/{total} param groups")

    x = torch.randn(2, 8, 3, 224, 224, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    print(f"Forward: {(time.perf_counter()-t0)*1000:.1f}ms | shape={out.shape}")

    # Sliding window
    full_video = torch.randn(90, 3, 224, 224, device=device)
    score = sliding_window_inference(model, full_video, device=device)
    print(f"Sliding window score (90 frames): {score:.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
