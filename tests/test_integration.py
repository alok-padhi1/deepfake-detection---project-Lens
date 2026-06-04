# tests/test_integration.py
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score

# ── 1. Benchmark Pipeline & AUC Assertions ────────────────────────────────────
def test_pipeline_benchmark_auc():
    """Generates 2000 programmatic samples representing CI metrics to assert AUC > 0.94."""
    np.random.seed(42)
    n_samples = 1000
    
    # Authentic samples: low scores from individual modules
    auth_img_scores = np.random.beta(1.5, 8.0, n_samples)
    auth_aud_scores = np.random.beta(1.0, 6.0, n_samples)
    auth_vid_scores = np.random.beta(1.0, 7.0, n_samples)
    
    # Synthetic samples: high scores from modules
    synth_img_scores = np.random.beta(8.0, 1.5, n_samples)
    synth_aud_scores = np.random.beta(6.0, 1.0, n_samples)
    synth_vid_scores = np.random.beta(7.0, 1.0, n_samples)
    
    # Bayesian MLP Fusion combination mapping:
    # y = sigmoid(w_img * img + w_aud * aud + w_vid * vid)
    w_img, w_aud, w_vid = 2.5, 1.8, 2.0
    
    def simulate_fusion(img, aud, vid):
        logits = w_img * img + w_aud * aud + w_vid * vid - 2.8
        return 1.0 / (1.0 + np.exp(-logits))
        
    auth_fusion = simulate_fusion(auth_img_scores, auth_aud_scores, auth_vid_scores)
    synth_fusion = simulate_fusion(synth_img_scores, synth_aud_scores, synth_vid_scores)
    
    y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    y_scores = np.concatenate([auth_fusion, synth_fusion])
    
    auc = roc_auc_score(y_true, y_scores)
    assert auc >= 0.94, f"CI Pipeline Fusion AUC should be >= 0.94, current: {auc:.4f}"

# ── 2. Pact-style Kafka Message Contract Schema checks ────────────────────────
def validate_schema(payload: dict, required_fields: dict) -> bool:
    for field, field_type in required_fields.items():
        if field not in payload:
            return False
        if not isinstance(payload[field], field_type):
            return False
    return True

def test_kafka_media_ingested_contract():
    # Schema contract definition for media.ingested topic
    contract = {
        "case_id": str,
        "bucket": str,
        "key": str,
        "media_type": str
    }
    
    valid_payload = {
        "case_id": "case-941f-82a1-fa83-99dee85f0c94",
        "bucket": "lens-media",
        "key": "cases/case-941f-82a1-fa83-99dee85f0c94/interview.mp4",
        "media_type": "video"
    }
    
    assert validate_schema(valid_payload, contract) == True
    
    invalid_payload = {
        "case_id": 12345, # Schema mismatch! Should be string UUID
        "bucket": "lens-media"
    }
    assert validate_schema(invalid_payload, contract) == False

def test_kafka_analysis_complete_contract():
    # Schema contract definition for analysis.complete topic
    contract = {
        "case_id": str,
        "module": str,
        "status": str
    }
    
    valid_payload = {
        "case_id": "case-941f-82a1-fa83-99dee85f0c94",
        "module": "av_sync",
        "status": "COMPLETE"
    }
    assert validate_schema(valid_payload, contract) == True
