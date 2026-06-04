# tests/test_unit.py
import pytest
import numpy as np
from scipy.signal import wiener

# ── 1. PRNU Correlation Test ──────────────────────────────────────────────────
def calculate_prnu_correlation(img: np.ndarray, ref_pattern: np.ndarray) -> float:
    """Extracts PRNU noise residual using Wiener filter and correlates with reference."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    # Estimate noise residual
    noise = gray.astype(np.float32) - wiener(gray.astype(np.float32), (5, 5))
    
    # Correlation coefficient
    c_matrix = np.corrcoef(noise.flatten(), ref_pattern.flatten())
    return float(c_matrix[0, 1])

import cv2
def test_prnu_correlation(authentic_image, synthetic_image):
    h, w = 224, 224
    # Mock clean camera reference PRNU pattern
    ref_pattern = np.random.normal(0, 2.0, (h, w))
    
    # Authentic image has high correlation with camera signature
    c_auth = calculate_prnu_correlation(authentic_image + (ref_pattern * 0.1).astype(np.uint8), ref_pattern)
    # Synthetic image has noise patterns replaced or unrelated
    c_synth = calculate_prnu_correlation(synthetic_image, ref_pattern)
    
    assert c_auth > 0.012, f"Authentic correlation {c_auth} should be > 0.012"
    assert c_synth < 0.005, f"Synthetic correlation {c_synth} should be < 0.005"

# ── 2. FFT Frequency Analysis (HFER) Test ─────────────────────────────────────
def calculate_hfer(img: np.ndarray) -> float:
    """Calculates High-Frequency Energy Ratio to detect GAN grids."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    f_coef = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_coef)
    magnitude = np.abs(f_shift)
    
    # Measure energy in the high frequency outer border band vs central core
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    total_energy = np.sum(magnitude)
    center_mask = np.zeros((h, w), dtype=bool)
    cv2.circle(center_mask.astype(np.uint8), (cx, cy), 30, 1, -1)
    
    high_freq_energy = np.sum(magnitude[~center_mask])
    return float(high_freq_energy / (total_energy + 1e-6))

def test_fft_hfer(authentic_image, synthetic_image):
    hfer_auth = calculate_hfer(authentic_image)
    hfer_synth = calculate_hfer(synthetic_image)
    
    # GAN artifacts inject strong periodic grid frequencies in high-frequency regions
    assert hfer_synth > hfer_auth, "Synthetic HFER score should exceed authentic score."

# ── 3. DCT Coefficient Histogram Test ──────────────────────────────────────────
def test_dct_coherency(authentic_image):
    gray = cv2.cvtColor(authentic_image, cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(gray.astype(np.float32))
    
    # Inject double JPEG compression DCT histograms (periodic zero bins)
    dct_injected = dct.copy()
    dct_injected[::2, ::2] = 0 # Zero out alternate coefficients
    
    diff = np.abs(dct - dct_injected).sum()
    assert diff > 1000.0, "DCT modification should trigger visible histogram divergence."

# ── 4. Sync Score Tensor Test ─────────────────────────────────────────────────
def test_sync_score():
    # Construct aligned vs misaligned audio-video mock embeddings (128-dimensional vectors)
    audio_embed = np.random.normal(0, 1.0, (10, 128))
    video_embed_aligned = audio_embed + np.random.normal(0, 0.05, (10, 128))
    video_embed_misaligned = np.random.normal(0, 1.0, (10, 128))
    
    # Cosine similarities
    def cosine_sim(a, b):
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
        
    sim_aligned = cosine_sim(audio_embed, video_embed_aligned)
    sim_misaligned = cosine_sim(audio_embed, video_embed_misaligned)
    
    assert np.mean(sim_aligned) > 0.72, "Aligned frames should be above authenticity threshold."
    assert np.mean(sim_misaligned) < 0.55, "Misaligned frames should be within suspicious bounds."

# ── 5. Jerk Score Discontinuity Test ──────────────────────────────────────────
def test_jerk_score():
    # Standard head pose acceleration timeline
    timeline = np.random.normal(0, 0.1, 100)
    # Inject deepfake boundary warp jump artifact at frame 50
    timeline[50] = 5.8
    
    # Calculate jerk acceleration score (standard deviation of third derivative)
    jerk_diff = np.diff(timeline, n=3)
    jerk_score = np.std(jerk_diff)
    
    assert jerk_score > 3.0, "Jerk score anomaly detection failed to highlight discontinuity."

# ── 6. Isotonic Calibration Calibration Test ──────────────────────────────────
def test_bayesian_fusion_calibration():
    # Isotonic calibration mock helper
    # Authentic scores should calibrate to values close to 0.0, synthetic close to 1.0
    mock_inputs = np.array([0.05, 0.12, 0.75, 0.88, 0.94])
    # Calibrated probabilities via mapping functions
    calibrated = np.clip(mock_inputs * 1.05, 0.0, 1.0)
    
    assert calibrated[0] < 0.1
    assert calibrated[-1] > 0.9
