"""
LENS ML — TorchServe Video Handler (TimeSformer + SyncNet)
Input:  multipart/form-data with video file OR base64-encoded video bytes
Output: JSON with fake_probability, per_window_scores, model
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from models.timesformer_video import sliding_window_inference

log = logging.getLogger(__name__)
NUM_FRAMES = 8
IMAGE_SIZE = 224


class VideoHandler(BaseHandler):
    def initialize(self, context) -> None:
        super().initialize(context)
        self._model_name = context.manifest.get("model", {}).get(
            "modelName", "timesformer"
        )

    def _decode_video(self, body: bytes) -> List[np.ndarray]:
        """Write bytes to temp file, extract frames."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(body)
            tmp_path = f.name
        try:
            cap    = cv2.VideoCapture(tmp_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE),
                                   interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)
            cap.release()
        finally:
            os.unlink(tmp_path)
        return frames

    def preprocess(self, data: List[Dict]) -> List[torch.Tensor]:
        all_clips = []
        for item in data:
            body = item.get("body") or item.get("data", b"")
            if isinstance(body, str):
                body = base64.b64decode(body)
            frames = self._decode_video(body)
            if not frames:
                frames = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)]
            # (T, H, W, C) → (T, C, H, W) float32 normalized
            arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0
            clip = torch.from_numpy(arr.transpose(0, 3, 1, 2))
            all_clips.append(clip)
        return all_clips

    def inference(self, data: List[torch.Tensor]) -> List[float]:
        scores = []
        for clip in data:
            score = sliding_window_inference(
                self.model, clip,
                clip_len=NUM_FRAMES, stride=4, device=str(self.device)
            )
            scores.append(score)
        return scores

    def postprocess(self, data: List[float]) -> List[Dict]:
        return [
            {
                "fake_probability": round(float(s), 4),
                "is_fake":          bool(s > 0.5),
                "model":            self._model_name,
            }
            for s in data
        ]
