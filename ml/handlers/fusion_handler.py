"""
LENS ML — TorchServe Fusion Handler (Bayesian Fusion MLP)
Input:  JSON body with score_vector array (18 dims) + media_type
Output: JSON with calibrated fake_probability + uncertainty
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from models.bayesian_fusion import INPUT_DIM, IsotonicCalibrator

log = logging.getLogger(__name__)


class FusionHandler(BaseHandler):
    def initialize(self, context) -> None:
        super().initialize(context)
        # Load isotonic calibrators if available
        self._calibrator: Optional[IsotonicCalibrator] = None
        cal_path = Path(context.system_properties.get("model_dir", ".")) \
                   / "isotonic_calibrators.pkl"
        if cal_path.exists():
            self._calibrator = IsotonicCalibrator()
            self._calibrator.load(cal_path)
            log.info("Isotonic calibrators loaded from %s", cal_path)

    def preprocess(self, data: List[Dict]):
        scores_list, media_types = [], []
        for item in data:
            body = item.get("body") or item.get("data", b"")
            if isinstance(body, (bytes, bytearray)):
                body = json.loads(body.decode("utf-8"))
            elif isinstance(body, str):
                body = json.loads(body)

            scores     = body.get("score_vector", [0.5] * INPUT_DIM)
            media_type = body.get("media_type", "image")
            scores_list.append(scores)
            media_types.append(media_type)

        scores_t = torch.tensor(scores_list, dtype=torch.float32).to(self.device)
        return scores_t, media_types

    def inference(self, data):
        scores_t, media_types = data
        results = {}
        with torch.no_grad():
            for mt in set(media_types):
                mask = [i for i, m in enumerate(media_types) if m == mt]
                if not mask:
                    continue
                sub = scores_t[mask]
                mean_prob, std_prob = self.model.predict_proba_mc(
                    sub, media_type=mt, n_samples=20
                )
                results[mt] = {
                    "indices":   mask,
                    "mean_prob": mean_prob.cpu().numpy(),
                    "std_prob":  std_prob.cpu().numpy(),
                    "media_type": mt,
                }
        return results, media_types

    def postprocess(self, data) -> List[Dict]:
        results_by_type, media_types = data
        B = len(media_types)
        output = [None] * B

        for mt, res in results_by_type.items():
            for list_idx, batch_idx in enumerate(res["indices"]):
                raw_prob = float(res["mean_prob"][list_idx])
                std_prob = float(res["std_prob"][list_idx])

                # Isotonic calibration
                if self._calibrator:
                    import numpy as np
                    logit = np.log(raw_prob / (1 - raw_prob + 1e-8))
                    cal_prob = float(
                        self._calibrator.calibrate(np.array([logit]), mt)[0]
                    )
                else:
                    cal_prob = raw_prob

                output[batch_idx] = {
                    "fake_probability":        round(cal_prob, 4),
                    "raw_probability":         round(raw_prob, 4),
                    "epistemic_uncertainty":   round(std_prob, 4),
                    "is_fake":                 bool(cal_prob > 0.5),
                    "media_type":              mt,
                    "calibrated":              self._calibrator is not None,
                    "model":                   "bayesian-fusion-mlp",
                }

        return output
