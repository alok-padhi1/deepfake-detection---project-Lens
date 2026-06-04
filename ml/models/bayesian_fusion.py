"""
LENS ML — Model 5: Bayesian Fusion MLP
Input: score vector from all active modules + media quality metrics
Trained on validation set score vectors.
Isotonic calibration post-training for true probability output.
Module weights differ per media type (image / video / audio).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

# ── Input feature schema ───────────────────────────────────────────────────────
# Score indices in the fusion input vector:
SCORE_SCHEMA = {
    # Model scores (sigmoid probabilities, 0=real, 1=fake)
    "efficientnet_binary":   0,
    "efficientnet_family_0": 1,   # P(REAL)
    "efficientnet_family_1": 2,   # P(GAN)
    "efficientnet_family_2": 3,   # P(Diffusion)
    "efficientnet_family_3": 4,   # P(Neural)
    "efficientnet_patch_max": 5,  # max of 196 patch logits → sigmoid
    "efficientnet_patch_mean":6,  # mean patch score
    "timesformer_binary":    7,
    "mel_binary":            8,
    "raw_binary":            9,
    "syncnet_binary":        10,
    # Media quality metrics
    "jpeg_quality_est":      11,  # estimated JPEG quality [0,1]
    "blur_score":            12,  # Laplacian variance, normalized [0,1]
    "noise_score":           13,  # estimated noise level [0,1]
    "resolution_norm":       14,  # min(H,W)/1024 clamped to [0,1]
    "fps_norm":              15,  # fps/60 clamped to [0,1]
    "has_audio":             16,  # 0 or 1
    "duration_norm":         17,  # duration_s / 300 clamped to [0,1]
}
INPUT_DIM = len(SCORE_SCHEMA)    # 18

# Per media-type module availability masks
MEDIA_TYPE_MASKS = {
    "image": [1,1,1,1,1,1,1, 0, 0,0, 0,  1,1,1,1,0,0,0],
    "video": [1,1,1,1,1,1,1, 1, 0,0, 1,  1,1,1,1,1,0,1],
    "audio": [0,0,0,0,0,0,0, 0, 1,1, 0,  0,0,0,0,0,1,1],
}
assert all(len(v) == INPUT_DIM for v in MEDIA_TYPE_MASKS.values()), \
    f"Mask length must be {INPUT_DIM}"


# ── MLP with uncertainty estimation ───────────────────────────────────────────
class BayesianFusionMLP(nn.Module):
    """
    Multi-layer perceptron fusion model.
    Uses MC Dropout for epistemic uncertainty estimation.
    Accepts a media_type mask to zero out unavailable module scores.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: List[int] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

        # Per-media-type learned importance weights (log-scale, softmax-normalized)
        self.media_type_weights = nn.ParameterDict({
            mt: nn.Parameter(torch.ones(input_dim))
            for mt in MEDIA_TYPE_MASKS
        })

    def _apply_mask(
        self, x: torch.Tensor, media_type: str
    ) -> torch.Tensor:
        mask = torch.tensor(
            MEDIA_TYPE_MASKS[media_type], dtype=torch.float32, device=x.device
        )
        weights = torch.sigmoid(self.media_type_weights[media_type])
        return x * mask * weights

    def forward(
        self,
        scores: torch.Tensor,           # (B, INPUT_DIM)
        media_type: str = "image",      # "image" / "video" / "audio"
    ) -> torch.Tensor:
        """Returns raw logit (B, 1)."""
        x = self._apply_mask(scores, media_type)
        return self.net(x)

    @torch.no_grad()
    def predict_proba_mc(
        self,
        scores: torch.Tensor,
        media_type: str = "image",
        n_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation.
        Returns:
            mean_prob: (B,) — mean probability
            std_prob:  (B,) — epistemic uncertainty (std across MC samples)
        """
        self.train()   # enable dropout
        samples = []
        for _ in range(n_samples):
            logit = self.forward(scores, media_type)
            samples.append(torch.sigmoid(logit).squeeze(1))
        self.eval()

        samples_t = torch.stack(samples, dim=0)   # (N_samples, B)
        return samples_t.mean(0), samples_t.std(0)


# ── Isotonic calibration wrapper ──────────────────────────────────────────────
class IsotonicCalibrator:
    """
    Fits isotonic regression on validation logits to map raw outputs to
    true posterior probabilities.
    """
    def __init__(self) -> None:
        self._calibrators: Dict[str, IsotonicRegression] = {}

    def fit(
        self,
        logits: np.ndarray,     # (N,) raw logits from model
        labels: np.ndarray,     # (N,) int 0/1
        media_type: str = "image",
    ) -> None:
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
        cal.fit(probs.reshape(-1, 1), labels)
        self._calibrators[media_type] = cal
        log.info("Isotonic calibration fitted for media_type=%s on %d samples",
                 media_type, len(labels))

    def calibrate(self, logits: np.ndarray, media_type: str = "image") -> np.ndarray:
        """Map raw logits → calibrated probabilities."""
        if media_type not in self._calibrators:
            raise ValueError(f"No calibrator fitted for media_type={media_type}")
        probs = 1.0 / (1.0 + np.exp(-logits))
        return self._calibrators[media_type].predict(probs.reshape(-1, 1)).flatten()

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._calibrators, f)
        log.info("Calibrators saved to %s", path)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._calibrators = pickle.load(f)
        log.info("Calibrators loaded from %s", path)


# ── Fusion loss ────────────────────────────────────────────────────────────────
class FusionLoss(nn.Module):
    """Simple BCE loss for the fusion MLP."""
    def __init__(self, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ls  = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        if self.ls > 0:
            t = t * (1 - self.ls) + 0.5 * self.ls
        return self.bce(logits.squeeze(1), t)


# ── Score vector builder (inference-time) ─────────────────────────────────────
def build_score_vector(
    efficientnet_preds: Optional[Dict] = None,
    timesformer_logit: Optional[float] = None,
    mel_logit: Optional[float] = None,
    raw_logit: Optional[float] = None,
    syncnet_logit: Optional[float] = None,
    quality_metrics: Optional[Dict] = None,
) -> np.ndarray:
    """
    Construct the 18-dimensional input vector from individual model outputs.
    Missing values are filled with 0.5 (uninformative prior).
    """
    vec = np.full(INPUT_DIM, 0.5, dtype=np.float32)

    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    if efficientnet_preds:
        vec[0] = _sigmoid(efficientnet_preds.get("binary", 0.0))
        family_probs = (torch.softmax(
            torch.tensor(efficientnet_preds.get("family", [0.0]*4)), dim=0
        ).numpy())
        vec[1:5] = family_probs
        patch = np.array(efficientnet_preds.get("patch", [0.0] * 196))
        patch_probs = 1.0 / (1.0 + np.exp(-patch))
        vec[5] = float(patch_probs.max())
        vec[6] = float(patch_probs.mean())

    if timesformer_logit is not None:
        vec[7] = _sigmoid(timesformer_logit)
    if mel_logit is not None:
        vec[8] = _sigmoid(mel_logit)
    if raw_logit is not None:
        vec[9] = _sigmoid(raw_logit)
    if syncnet_logit is not None:
        vec[10] = _sigmoid(syncnet_logit)

    qm = quality_metrics or {}
    vec[11] = float(np.clip(qm.get("jpeg_quality", 0.85), 0, 1))
    vec[12] = float(np.clip(qm.get("blur_score",   0.5),  0, 1))
    vec[13] = float(np.clip(qm.get("noise_score",  0.1),  0, 1))
    vec[14] = float(np.clip(qm.get("resolution_norm", 0.5), 0, 1))
    vec[15] = float(np.clip(qm.get("fps_norm",     0.5),  0, 1))
    vec[16] = float(bool(qm.get("has_audio", False)))
    vec[17] = float(np.clip(qm.get("duration_norm", 0.1), 0, 1))

    return vec


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BayesianFusionMLP().to(device)
    model.eval()

    B = 8
    scores = torch.rand(B, INPUT_DIM, device=device)
    for mt in ("image", "video", "audio"):
        logits = model(scores, media_type=mt)
        mean_p, std_p = model.predict_proba_mc(scores, media_type=mt)
        print(f"[{mt}] logit shape={logits.shape}, "
              f"mean_prob={mean_p.mean().item():.3f}±{std_p.mean().item():.3f}")

    # Calibration smoke test
    cal = IsotonicCalibrator()
    logits_np = np.random.randn(100)
    labels_np = (logits_np > 0).astype(int)
    cal.fit(logits_np, labels_np, "image")
    out = cal.calibrate(logits_np, "image")
    print(f"Calibrated probs range: [{out.min():.3f}, {out.max():.3f}]")

    vec = build_score_vector(
        efficientnet_preds={"binary": 1.2, "family": [0.1, 0.7, 0.1, 0.1],
                             "patch": [0.5] * 196},
        timesformer_logit=0.8,
        quality_metrics={"jpeg_quality": 0.7, "has_audio": True}
    )
    print(f"Score vector: {vec}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
