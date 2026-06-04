# tests/conftest.py
import pytest
import numpy as np
import cv2

@pytest.fixture
def authentic_image():
    """Generates a high-quality clean synthetic image with raw noise residual representing natural sensor output."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return img

@pytest.fixture
def synthetic_image():
    """Generates an image with injected high-frequency GAN spectral grid artifacts."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Inject periodic high-frequency sinusoids representing grid artifacts
    x = np.linspace(0, 10, 224)
    y = np.linspace(0, 10, 224)
    xx, yy = np.meshgrid(x, y)
    grid = np.sin(5 * xx) * np.sin(5 * yy) * 30
    img = np.clip(img + np.expand_dims(grid, axis=-1), 0, 255).astype(np.uint8)
    return img

@pytest.fixture
def clean_audio_signal():
    """Generates a clean vocal range audio wave."""
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    # 440Hz clean sine wave
    return np.sin(2 * np.pi * 440 * t)

@pytest.fixture
def deepfake_audio_signal():
    """Generates a speech voice clone with vocoder high-frequency smoothing residuals."""
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    # Sino-Gaussian mix + low pass filter above 4kHz representing smoothing
    wave = np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.05, sr)
    # Simple smoothing filter
    wave = np.convolve(wave, np.ones(5)/5, mode='same')
    return wave
