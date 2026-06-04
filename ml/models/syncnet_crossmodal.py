"""
LENS ML — Model 4: SyncNet Cross-Modal Authenticity Model
Fine-tuned on 500K authentic + 200K manipulated video clips.
Dual-stream: video facial patch tokens + audio tokens.
Cross-attention over 1-second temporal windows.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Video stream: facial patch tokenizer ──────────────────────────────────────
class FacialPatchTokenizer(nn.Module):
    """
    Extracts 16×16 patch embeddings from facial region crops.
    Input:  (B, T, C, H, W)  — T frames, 224×224 crops
    Output: (B, T*N_patches, D_vid)
    """
    PATCH_SIZE = 16
    IMG_SIZE   = 224
    N_PATCHES  = (224 // 16) ** 2  # 196

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, embed_dim,
            kernel_size=self.PATCH_SIZE,
            stride=self.PATCH_SIZE,
            bias=False,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.N_PATCHES, embed_dim) * 0.02
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W)"""
        B, T = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x)                                    # (BT, D, 14, 14)
        x = rearrange(x, "bt d h w -> bt (h w) d")         # (BT, 196, D)
        x = x + self.pos_embed
        x = self.norm(x)
        x = rearrange(x, "(b t) n d -> b (t n) d", b=B, t=T)   # (B, T*196, D)
        return x


# ── Audio stream: 20ms frame tokenizer ───────────────────────────────────────
class AudioFrameTokenizer(nn.Module):
    """
    Projects short-time audio frames (20ms at 16kHz = 320 samples) into tokens.
    Input:  (B, N_frames, 320) raw audio frames
    Output: (B, N_frames, D_aud)
    """
    FRAME_SAMPLES = 320    # 20ms @ 16kHz

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.FRAME_SAMPLES, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Embedding(4096, embed_dim)   # positional encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_frames, 320)"""
        tokens = self.proj(x)                            # (B, N, D)
        N = x.shape[1]
        pos = self.pos_embed(
            torch.arange(N, device=x.device).unsqueeze(0)
        )
        return tokens + pos


# ── Cross-attention module (video → audio, audio → video) ────────────────────
class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention between video and audio token sequences.
    Applied within 1-second temporal windows.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.v2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.a2v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.ffn_v  = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_a  = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

    def forward(
        self,
        vid_tokens: torch.Tensor,   # (B, Nv, D)
        aud_tokens: torch.Tensor,   # (B, Na, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Video attends to audio
        v_out, _ = self.v2a(vid_tokens, aud_tokens, aud_tokens)
        vid_tokens = self.norm_v(vid_tokens + v_out)
        vid_tokens = self.norm_v2(vid_tokens + self.ffn_v(vid_tokens))

        # Audio attends to video
        a_out, _ = self.a2v(aud_tokens, vid_tokens, vid_tokens)
        aud_tokens = self.norm_a(aud_tokens + a_out)
        aud_tokens = self.norm_a2(aud_tokens + self.ffn_a(aud_tokens))

        return vid_tokens, aud_tokens


# ── SyncNet Cross-Modal Model ─────────────────────────────────────────────────
class SyncNetCrossModal(nn.Module):
    """
    SyncNet-style audio-visual synchrony model for deepfake detection.

    Training setup:
        - 500K authentic clips: lip-sync contrastive loss (matched A/V = 1)
        - 200K manipulated clips: binary authenticity cross-entropy loss
    """
    WINDOW_FRAMES    = 30    # 1 second at 30fps
    AUDIO_FRAMES_1S  = 50    # 1s / 20ms = 50 audio frames

    def __init__(
        self,
        embed_dim: int = 256,
        n_cross_attn_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vid_tokenizer = FacialPatchTokenizer(embed_dim)
        self.aud_tokenizer = AudioFrameTokenizer(embed_dim)

        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, n_heads, dropout)
            for _ in range(n_cross_attn_layers)
        ])

        # Pool video and audio into single vectors per window
        self.vid_pool = nn.AdaptiveAvgPool1d(1)
        self.aud_pool = nn.AdaptiveAvgPool1d(1)

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Sync score head (for contrastive loss on authentic clips)
        self.sync_proj = nn.Linear(embed_dim, embed_dim)

    def _process_window(
        self,
        vid: torch.Tensor,   # (B, T_window, 3, H, W)
        aud: torch.Tensor,   # (B, N_audio, 320)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_tok = self.vid_tokenizer(vid)    # (B, T*196, D)
        aud_tok = self.aud_tokenizer(aud)    # (B, N, D)

        for layer in self.cross_attn_layers:
            vid_tok, aud_tok = layer(vid_tok, aud_tok)

        # Pool to single vector
        v_vec = self.vid_pool(vid_tok.permute(0, 2, 1)).squeeze(-1)  # (B, D)
        a_vec = self.aud_pool(aud_tok.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        # Binary logit
        fused = torch.cat([v_vec, a_vec], dim=-1)   # (B, 2D)
        logit = self.classifier(fused)               # (B, 1)

        return v_vec, a_vec, logit

    def forward(
        self,
        vid: torch.Tensor,       # (B, T, 3, H, W)  — full clip (≥30 frames)
        aud: torch.Tensor,       # (B, N_audio, 320) — 20ms audio frames
        window_stride: int = 15, # stride in frames for sliding window
    ) -> Dict[str, torch.Tensor]:
        T = vid.shape[1]
        window_logits = []
        v_vecs, a_vecs = [], []

        # Sliding window over 1-second windows
        for start in range(0, max(1, T - self.WINDOW_FRAMES + 1), window_stride):
            end = min(start + self.WINDOW_FRAMES, T)
            v_win = vid[:, start:end]

            # Corresponding audio frames (1s = 50 × 20ms frames)
            aud_start = int(start * self.AUDIO_FRAMES_1S / self.WINDOW_FRAMES)
            aud_end   = aud_start + self.AUDIO_FRAMES_1S
            a_win = aud[:, aud_start:aud_end]

            # Pad if needed
            if v_win.shape[1] < self.WINDOW_FRAMES:
                pad_v = vid[:, -1:].expand(-1, self.WINDOW_FRAMES - v_win.shape[1],
                                            -1, -1, -1)
                v_win = torch.cat([v_win, pad_v], dim=1)
            if a_win.shape[1] < self.AUDIO_FRAMES_1S:
                pad_a = aud[:, -1:].expand(-1, self.AUDIO_FRAMES_1S - a_win.shape[1],
                                            -1)
                a_win = torch.cat([a_win, pad_a], dim=1)

            v_vec, a_vec, logit = self._process_window(v_win, a_win)
            window_logits.append(logit)
            v_vecs.append(v_vec)
            a_vecs.append(a_vec)

        # Max-pool over windows (most suspicious window wins)
        all_logits = torch.stack(window_logits, dim=1)    # (B, W, 1)
        final_logit, _ = all_logits.max(dim=1)            # (B, 1)

        # Average sync vectors for contrastive loss
        v_mean = torch.stack(v_vecs, dim=1).mean(dim=1)   # (B, D)
        a_mean = torch.stack(a_vecs, dim=1).mean(dim=1)   # (B, D)

        return {
            "logit":    final_logit,
            "v_embed":  F.normalize(self.sync_proj(v_mean), dim=-1),
            "a_embed":  F.normalize(self.sync_proj(a_mean), dim=-1),
        }


# ── SyncNet Loss (contrastive sync + binary fake detection) ──────────────────
class SyncNetLoss(nn.Module):
    """
    L = λ_fake * BCE(logit, fake_label)
      + λ_sync * InfoNCE(v_embed, a_embed) on authentic pairs only
    """
    def __init__(self, lambda_fake: float = 1.0,
                 lambda_sync: float = 0.5, temp: float = 0.07) -> None:
        super().__init__()
        self.lambda_fake = lambda_fake
        self.lambda_sync = lambda_sync
        self.temp        = temp
        self.bce         = nn.BCEWithLogitsLoss()

    def _infonce(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss for matched v/a pairs. (B, D) × (B, D) → scalar."""
        sim = torch.matmul(v, a.T) / self.temp   # (B, B)
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        fake_targets: torch.Tensor,     # (B,) float 0=real, 1=fake
    ) -> Dict[str, torch.Tensor]:
        l_fake = self.bce(preds["logit"].squeeze(1), fake_targets)

        # Sync loss only on authentic clips
        real_mask = (fake_targets == 0)
        if real_mask.sum() > 1:
            l_sync = self._infonce(
                preds["v_embed"][real_mask],
                preds["a_embed"][real_mask],
            )
        else:
            l_sync = torch.tensor(0.0, device=fake_targets.device)

        total = self.lambda_fake * l_fake + self.lambda_sync * l_sync
        return {"loss": total, "loss_fake": l_fake, "loss_sync": l_sync}


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2
    model = SyncNetCrossModal().to(device)
    vid   = torch.randn(B, 30, 3, 224, 224, device=device)
    aud   = torch.randn(B, 50, 320, device=device)

    t0 = time.perf_counter()
    out = model(vid, aud)
    print(f"Forward: {(time.perf_counter()-t0)*1000:.1f}ms")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    targets = torch.tensor([0.0, 1.0], device=device)
    criterion = SyncNetLoss()
    losses = criterion(out, targets)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
