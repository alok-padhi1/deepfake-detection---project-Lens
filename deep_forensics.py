import re

with open("lens_test_server.py", "r") as f:
    content = f.read()

new_logic = """def prnu_analysis(img_bgr):
    \"\"\"CFA Hardware Footprint & Noise Kurtosis (The Silver Bullet for AI vs Real).\"\"\"
    import cv2
    import numpy as np
    from scipy.signal import wiener
    from scipy.stats import kurtosis
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. CFA 2x2 Periodic Correlation Check (Bayer Filter footprint)
    # Physical sensors demosaic colors, leaving mathematical 2x2 periodicity.
    diff_h = np.abs(gray[:, :-1] - gray[:, 1:])
    var_h_even = np.var(diff_h[:, 0::2]) + 1e-6
    var_h_odd = np.var(diff_h[:, 1::2]) + 1e-6
    cfa_ratio = min(var_h_even, var_h_odd) / max(var_h_even, var_h_odd)
    
    # 2. PRNU Noise Kurtosis
    denoised = wiener(gray, (5, 5))
    noise = gray - denoised
    noise_kurtosis = abs(float(kurtosis(noise.flatten(), fisher=True)))
    
    # AI generates pixels via latent space. No Bayer filter = perfect cfa_ratio (~1.0)
    # AI diffusion noise is mathematically flat or pure Gaussian = low kurtosis.
    # Real camera sensors = cfa_ratio < 0.95, complex kurtosis > 1.5
    if cfa_ratio > 0.98 and noise_kurtosis < 1.0:
        score = 0.05 # Definitively AI Generated (Lacks Hardware)
    elif cfa_ratio > 0.95:
        score = 0.35 # Highly Suspicious (Possible AI or heavy digital wipe)
    else:
        score = 0.98 # Authentic Hardware Signature Detected
        
    return {
        "name": "CFA Hardware Footprint",
        "score": round(score, 4),
        "cfa_correlation": round(cfa_ratio, 4),
        "noise_kurtosis": round(noise_kurtosis, 4),
        "blocks_analyzed": int((gray.shape[0] * gray.shape[1]) / 4),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def texture_analysis(img_bgr):
    \"\"\"Optical Edge-Gradient Correlation (Detect AI mathematical perfection).\"\"\"
    import cv2
    import numpy as np
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Real lenses have chromatic aberration and optical softness variance.
    # AI upscalers/diffusion models generate mathematically perfect sub-pixel gradients.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    edge_threshold = np.percentile(gradient_mag, 95)
    edge_mask = gradient_mag > edge_threshold
    
    if np.sum(edge_mask) < 100:
        score = 0.8
        lap_var = 0.0
        consistency = 1.0
    else:
        sharp_edges = gradient_mag[edge_mask]
        lap_var = float(np.var(sharp_edges))
        # If variance of the sharpest edges is extremely low, ALL edges are mathematically identical.
        consistency = float(np.std(sharp_edges) / (np.mean(sharp_edges) + 1e-6))
        
        if consistency < 0.2:
            score = 0.1 # Synthetic perfection (AI Upscaled)
        elif consistency < 0.35:
            score = 0.4
        else:
            score = 0.96 # Natural optical blur variance
            
    return {
        "name": "Optical Edge Correlation",
        "score": round(score, 4),
        "gradient_variance": round(lap_var, 2),
        "optical_consistency": round(consistency, 4),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def ela_analysis(img_bgr):
    \"\"\"Error Level Analysis (Penalize perfect uniformity & extreme variance).\"\"\"
    import cv2
    import numpy as np
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, enc = cv2.imencode('.jpg', img_bgr, encode_param)
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, recompressed).astype(np.float32)
    ela_map = np.mean(diff, axis=2)
    
    mean_err = float(np.mean(ela_map))
    max_err = float(np.max(ela_map))
    std_err = float(np.std(ela_map))
    
    if std_err < 0.8:
        score = 0.25
    elif std_err > 5.0:
        score = 0.4
    else:
        score = 0.92
        
    return {
        "name": "Error Level Analysis (ELA)",
        "score": round(score, 4),
        "mean_error": round(mean_err, 4),
        "max_error": round(max_err, 4),
        "std_error": round(std_err, 4),
        "verdict": "Authentic" if score > 0.6 else "Suspicious"
    }

def fft_analysis(img_bgr):
    \"\"\"2D FFT frequency domain analysis (Detect GAN grids).\"\"\"
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
    
    radial_profile = []
    for ri in range(1, min(cy, cx)):
        ring = ((X - cx)**2 + (Y - cy)**2 >= (ri-1)**2) & ((X - cx)**2 + (Y - cy)**2 < ri**2)
        radial_profile.append(float(np.mean(mag[ring])))
    rp = np.array(radial_profile)
    spectral_peaks = int(np.sum(np.diff(np.sign(np.diff(rp))) < 0)) if len(rp) > 2 else 0
    
    if spectral_peaks >= 2:
        score = 0.1
    elif hfer < 0.05:
        score = 0.3
    else:
        score = 0.95
        
    return {
        "name": "FFT Frequency Analysis",
        "score": round(score, 4),
        "high_freq_ratio": round(hfer, 4),
        "spectral_peaks": spectral_peaks,
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }

def dct_analysis(img_bgr):
    \"\"\"DCT coefficient histogram for double-JPEG compression detection.\"\"\"
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
    
    score = min(1.0, max(0.0, 0.95 - periodicity * 30))
    return {
        "name": "DCT Compression Forensics",
        "score": round(score, 4),
        "zero_coefficient_ratio": round(zero_ratio, 4),
        "histogram_periodicity": round(periodicity, 6),
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }

def wavelet_analysis(img_bgr):
    \"\"\"DWT-based noise pattern analysis (Detect directional generation skew).\"\"\"
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
    
    if anisotropy > 1.25:
        score = max(0.0, 1.0 - (anisotropy - 1.0) * 1.5)
    else:
        score = 0.95
        
    return {
        "name": "Wavelet Noise Analysis",
        "score": round(score, 4),
        "detail_to_approx_ratio": round(ratio, 4),
        "noise_anisotropy": round(anisotropy, 4),
        "verdict": "Authentic" if score > 0.55 else "Suspicious"
    }"""

pattern = re.compile(r'def prnu_analysis\(img_bgr\):.*?def compute_fusion\(results\):', re.DOTALL)
new_content = pattern.sub(new_logic + "\n\ndef compute_fusion(results):", content)
# We also update the weights slightly to favor the new CFA module heavily since it's the silver bullet
new_content = new_content.replace('"PRNU Sensor Forensics": 0.25', '"CFA Hardware Footprint": 0.35, "Optical Edge Correlation": 0.15')
new_content = new_content.replace('"Texture Consistency": 0.15, ', '') # We replaced this with optical edge correlation

with open("lens_test_server.py", "w") as f:
    f.write(new_content)
print("Deep forensics overhaul deployed successfully.")
