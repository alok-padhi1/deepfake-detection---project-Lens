"""
LENS ML — TorchServe Image Handler (EfficientNet-B7)
Handles: image/jpeg, image/png, image/webp
Returns: JSON with binary_prob, family_probs, patch_heatmap
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

log = logging.getLogger(__name__)

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:
    _HAS_ALB = False


class ImageHandler(BaseHandler):
    IMAGE_SIZE = 224

    def initialize(self, context) -> None:
        super().initialize(context)
        self.transform = self._build_transform()

    def _build_transform(self):
        if _HAS_ALB:
            import cv2
            return A.Compose([
                A.Resize(self.IMAGE_SIZE, self.IMAGE_SIZE,
                         interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                             max_pixel_value=255.0),
            ])
        return None

    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        images = []
        for item in data:
            body = item.get("body") or item.get("data", b"")
            if isinstance(body, str):
                body = base64.b64decode(body)
            img = Image.open(io.BytesIO(body)).convert("RGB")
            arr = np.array(img)

            if self.transform:
                arr = self.transform(image=arr)["image"].astype(np.float32)
            else:
                arr = arr.astype(np.float32) / 127.5 - 1.0
                arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize(
                    (self.IMAGE_SIZE, self.IMAGE_SIZE)))

            images.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        return torch.stack(images).to(self.device)

    def inference(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data: Dict[str, torch.Tensor]) -> List[Dict]:
        binary_probs = torch.sigmoid(data["binary"]).squeeze(1).cpu().tolist()
        family_probs = F.softmax(data["family"], dim=-1).cpu().tolist()
        patch_probs  = torch.sigmoid(data["patch"]).cpu().tolist()

        family_names = ["REAL", "GAN", "Diffusion", "Neural"]
        results = []
        for i, bp in enumerate(binary_probs):
            patch_grid = np.array(patch_probs[i]).reshape(14, 14).tolist()
            results.append({
                "fake_probability":   round(float(bp), 4),
                "is_fake":            bool(bp > 0.5),
                "generator_family":   dict(zip(family_names, [round(p, 4)
                                              for p in family_probs[i]])),
                "patch_heatmap":      patch_grid,   # 14×14 confidence grid
                "model":              "efficientnet-b7-forensic",
            })
        return results
