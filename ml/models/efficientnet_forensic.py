"""
LENS ML — EfficientNet-B4 Forensic Model (RTX 3050 / 4GB VRAM optimized)
B4 fits in 4GB with batch=16. Forensic accuracy nearly identical to B7 on binary tasks.
"""
from __future__ import annotations
from typing import Dict
import torch, torch.nn as nn, torch.nn.functional as F
import timm

FAMILY_CLASSES = ["REAL", "GAN", "Diffusion", "Neural"]
N_FAMILIES     = len(FAMILY_CLASSES)

class PatchHead(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c, 128, 2, 2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )
    def forward(self, x): return self.up(x).flatten(1)

class EfficientNetForensic(nn.Module):
    """EfficientNet-B4 + ForensicPrismHead. Fits in 4GB VRAM with batch=16."""
    def __init__(self, pretrained=True, dropout=0.3, variant="tf_efficientnet_b4.ns_jft_in1k"):
        super().__init__()
        self.backbone = timm.create_model(
            variant, pretrained=pretrained,
            features_only=True, out_indices=(2, 3, 4)
        )
        ch = self.backbone.feature_info.channels()
        self.lat = nn.Sequential(
            nn.Conv2d(ch[0]+ch[1], ch[2], 1, bias=False),
            nn.BatchNorm2d(ch[2]), nn.GELU()
        )
        C = ch[2]
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.binary = nn.Sequential(nn.Linear(C,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256,1))
        self.family = nn.Sequential(nn.Linear(C,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256,N_FAMILIES))
        self.patch  = PatchHead(C)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f0, f1, f2 = self.backbone(x)
        h, w = f0.shape[2:]
        fused  = self.lat(torch.cat([f0, F.interpolate(f1,(h,w),mode='bilinear',align_corners=False)],1))
        global_feat = self.pool(f2).flatten(1)
        return {"binary": self.binary(global_feat),
                "family": self.family(global_feat),
                "patch":  self.patch(fused)}

class ForensicLoss(nn.Module):
    def __init__(self, lb=1.0, lf=0.5, lp=0.3, label_smoothing=0.05):
        super().__init__()
        self.lb, self.lf, self.lp = lb, lf, lp
        self.bce       = nn.BCEWithLogitsLoss()
        self.ce        = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.patch_bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, binary_targets, family_targets):
        l_bin   = self.bce(preds["binary"].squeeze(1), binary_targets.float())
        l_fam   = self.ce(preds["family"], family_targets)
        pt      = binary_targets.float().unsqueeze(1).expand_as(preds["patch"])
        l_patch = self.patch_bce(preds["patch"], pt)
        total   = self.lb*l_bin + self.lf*l_fam + self.lp*l_patch
        return {"loss": total, "loss_binary": l_bin, "loss_family": l_fam}

if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = EfficientNetForensic(pretrained=False).to(device)
    x = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        t0  = time.perf_counter()
        out = model(x)
        t1  = time.perf_counter()
    params = sum(p.numel() for p in model.parameters())/1e6
    print(f"B4 | params={params:.1f}M | forward={1000*(t1-t0):.1f}ms | binary={out['binary'].shape}")
