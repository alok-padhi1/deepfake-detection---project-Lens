import re

with open("lens_test_server.py", "r") as f:
    content = f.read()

new_engines = """def prnu_analysis(img_bgr):
    \"\"\"PRNU sensor noise residual extraction via Wiener denoising (Edge Masked).\"\"\"
    import cv2
    import numpy as np
    from scipy.signal import wiener
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denoised = wiener(gray, (5, 5))
    noise_residual = gray - denoised
    noise_energy = float(np.std(noise_residual))
    
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1) == 0
    
    block_size = 64
    h, w = gray.shape
    tamper_scores = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_mask = mask[y:y+block_size, x:x+block_size]
            if np.sum(block_mask) > (block_size * block_size * 0.3):
                block = noise_residual[y:y+block_size, x:x+block_size]
                tamper_scores.append(float(np.std(block[block_mask])))
                
    if not tamper_scores: tamper_scores = [np.std(noise_residual)]
    consistency = float(np.std(tamper_scores) / (np.mean(tamper_scores) + 1e-6))
    score = min(1.0, max(0.0, 1.0 - consistency * 0.8))
    return {
        "name": "PRNU Sensor Forensics",
        "score": round(score, 4),
        "noise_energy": round(noise_energy, 4),
        "block_consistency": round(1.0 - consistency, 4),
        "blocks_analyzed": len(tamper_scores),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def ela_analysis(img_bgr):
    \"\"\"Error Level Analysis via JPEG recompression divergence (Adaptive).\"\"\"
    import cv2
    import numpy as np
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, enc = cv2.imencode('.jpg', img_bgr, encode_param)
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, recompressed).astype(np.float32)
    ela_map = np.mean(diff, axis=2)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lum_var = np.var(gray) + 1e-6
    mean_err = float(np.mean(ela_map))
    max_err = float(np.max(ela_map))
    std_err = float(np.std(ela_map))
    
    normalized_err = std_err / (np.sqrt(lum_var) * 0.05 + 1e-6)
    score = min(1.0, max(0.0, 1.0 - (normalized_err / (mean_err + 1e-6)) * 0.15))
    return {
        "name": "Error Level Analysis (ELA)",
        "score": round(score, 4),
        "mean_error": round(mean_err, 4),
        "max_error": round(max_err, 4),
        "std_error": round(std_err, 4),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def fft_analysis(img_bgr):
    \"\"\"2D FFT frequency domain analysis (Content-Normalized).\"\"\"
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 6
    Y, X = np.ogrid[:h, :w]
    center_mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
    low_energy = float(np.sum(mag[center_mask]))
    high_energy = float(np.sum(mag[~center_mask]))
    hfer = high_energy / (low_energy + high_energy + 1e-6)
    
    entropy = float(np.std(gray) + 1e-6)
    normalized_hfer = hfer / (np.log(entropy + 2) * 0.5 + 1e-6)
    
    radial_profile = []
    for ri in range(1, min(cy, cx)):
        ring = ((X - cx)**2 + (Y - cy)**2 >= (ri-1)**2) & ((X - cx)**2 + (Y - cy)**2 < ri**2)
        radial_profile.append(float(np.mean(mag[ring])))
    rp = np.array(radial_profile)
    spectral_peaks = int(np.sum(np.diff(np.sign(np.diff(rp))) < 0)) if len(rp) > 2 else 0
    score = min(1.0, max(0.0, 1.0 - (normalized_hfer * 0.5) - (spectral_peaks * 0.005)))
    return {
        "name": "FFT Frequency Analysis",
        "score": round(score, 4),
        "high_freq_ratio": round(hfer, 4),
        "spectral_peaks": spectral_peaks,
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }

def dct_analysis(img_bgr):
    \"\"\"DCT coefficient histogram for double-JPEG compression detection (Adaptive).\"\"\"
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    gray = gray[:bh, :bw]
    dct_coeffs = []
    for y in range(0, bh, 8):
        for x in range(0, bw, 8):
            dct_coeffs.append(cv2.dct(gray[y:y+8, x:x+8]).flatten())
    all_coeffs = np.concatenate(dct_coeffs)
    hist, _ = np.histogram(all_coeffs, bins=256, range=(-128, 128))
    hist_norm = hist / (hist.sum() + 1e-6)
    zero_ratio = float(hist_norm[128])
    periodicity = float(np.std(np.diff(hist_norm[120:136])))
    
    # AI generators lack periodicity, real images might have some due to compression
    # We soften the penalty for real images slightly
    score = min(1.0, max(0.0, 0.9 - periodicity * 30))
    return {
        "name": "DCT Compression Forensics",
        "score": round(score, 4),
        "zero_coefficient_ratio": round(zero_ratio, 4),
        "histogram_periodicity": round(periodicity, 6),
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }

def texture_analysis(img_bgr):
    \"\"\"LBP + Laplacian variance for texture consistency (Edge-Masked).\"\"\"
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))
    
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1) == 0

    h, w = gray.shape
    block_vars = []
    bs = 64
    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            block_mask = mask[y:y+bs, x:x+bs]
            if np.sum(block_mask) > (bs * bs * 0.3):
                b = cv2.Laplacian(gray[y:y+bs, x:x+bs], cv2.CV_64F)
                b_flat = b[block_mask]
                if len(b_flat) > 10:
                    block_vars.append(float(np.var(b_flat)))
    bv = np.array(block_vars) if block_vars else np.array([lap_var])
    consistency = float(np.std(bv) / (np.mean(bv) + 1e-6))
    score = min(1.0, max(0.0, 1.0 - consistency * 0.5))
    return {
        "name": "Texture Consistency",
        "score": round(score, 4),
        "laplacian_variance": round(lap_var, 2),
        "block_consistency": round(1.0 - consistency, 4),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def wavelet_analysis(img_bgr):
    \"\"\"DWT-based noise pattern analysis.\"\"\"
    import cv2
    import numpy as np
    import pywt
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    coeffs = pywt.dwt2(gray, 'db4')
    cA, (cH, cV, cD) = coeffs
    detail_energy = float(np.mean(np.abs(cH)) + np.mean(np.abs(cV)) + np.mean(np.abs(cD)))
    approx_energy = float(np.mean(np.abs(cA)))
    ratio = detail_energy / (approx_energy + 1e-6)
    h_std = float(np.std(cH))
    v_std = float(np.std(cV))
    d_std = float(np.std(cD))
    anisotropy = float(max(h_std, v_std, d_std) / (min(h_std, v_std, d_std) + 1e-6))
    
    # Allow slightly more natural anisotropy
    score = min(1.0, max(0.0, 1.0 - abs(anisotropy - 1.0) * 0.2))
    return {
        "name": "Wavelet Noise Analysis",
        "score": round(score, 4),
        "detail_to_approx_ratio": round(ratio, 4),
        "noise_anisotropy": round(anisotropy, 4),
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }"""

import re
pattern = re.compile(r'def prnu_analysis\(img_bgr\):.*?def compute_fusion\(results\):', re.DOTALL)
new_content = pattern.sub(new_engines + "\n\ndef compute_fusion(results):", content)

with open("lens_test_server.py", "w") as f:
    f.write(new_content)
print("Updated successfully")
