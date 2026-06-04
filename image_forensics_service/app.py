# app.py
import io
import json
import logging
import time
import threading
import math
import requests
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.signal
import pywt
from PIL import Image
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import redis
from minio import Minio
from confluent_kafka import Consumer, Producer, KafkaError
from qdrant_client import QdrantClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import settings

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
log = logging.getLogger("image_forensics")

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
METRIC_REQUESTS_TOTAL = Counter("image_forensics_requests_total", "Total image analysis requests received", ["status"])
METRIC_LATENCY = Histogram("image_forensics_latency_seconds", "Latency of image analysis pipeline in seconds")

# ── FastAPI App for Health & Metrics ──────────────────────────────────────────
app = FastAPI(title=settings.APP_NAME)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness and Readiness probe endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ── Helper: Wavelet-based Wiener Filter Denoising ──────────────────────────────
def wavelet_wiener_denoise(img_gray: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    """
    Perform Wavelet-based Wiener filter denoising.
    Shrinks 2D DWT detail coefficients to estimate and extract the noise residual.
    """
    # 2D DWT using Daubechies 4 wavelet
    coeffs = pywt.wavedec2(img_gray, 'db4', level=4)
    shrunk_coeffs = [coeffs[0]]  # Keep LL approximation coefficients untouched
    
    for level_coeffs in coeffs[1:]:
        shrunk_level = []
        for detail in level_coeffs:
            # Estimate detail coefficients variance and apply Wiener shrinkage
            c_sq = detail ** 2
            shrunk = detail * (c_sq / (c_sq + sigma**2 + 1e-12))
            shrunk_level.append(shrunk)
        shrunk_coeffs.append(tuple(shrunk_level))
        
    # Reconstruct denoised image
    img_denoised = pywt.waverec2(shrunk_coeffs, 'db4')
    # Crop borders to match original image size
    h, w = img_gray.shape
    return img_denoised[:h, :w]

# ── Helper: Pearson Cross-Correlation ──────────────────────────────────────────
def pearson_correlation(R: np.ndarray, K: np.ndarray) -> float:
    """Computes the Pearson correlation coefficient between two 2D patterns."""
    R_flat = R.flatten()
    K_flat = K.flatten()
    if len(R_flat) != len(K_flat):
        # Resize K to match R
        K_img = Image.fromarray(K)
        K_resized = np.array(K_img.resize((R.shape[1], R.shape[0])))
        K_flat = K_resized.flatten()
        
    R_mean, K_mean = np.mean(R_flat), np.mean(K_flat)
    R_diff, K_diff = R_flat - R_mean, K_flat - K_mean
    num = np.sum(R_diff * K_diff)
    den = np.sqrt(np.sum(R_diff**2) * np.sum(K_diff**2))
    if den == 0:
        return 0.0
    return float(num / den)

# ── Helper: Block-based PRNU Tamper Localization ─────────────────────────────
def prnu_tamper_map(R: np.ndarray, K: np.ndarray, block_size: int = 64) -> List[List[float]]:
    """Calculates sub-pixel correlation map over 64x64 blocks for localization."""
    h, w = R.shape
    if K.shape != R.shape:
        K = np.array(Image.fromarray(K).resize((w, h)))
        
    tamper_grid = []
    for y in range(0, h, block_size):
        row = []
        for x in range(0, w, block_size):
            r_block = R[y:y+block_size, x:x+block_size]
            k_block = K[y:y+block_size, x:x+block_size]
            if r_block.size < 16:
                row.append(0.0)
                continue
            row.append(pearson_correlation(r_block, k_block))
        tamper_grid.append(row)
    return tamper_grid

# ── Helper: 2D FFT Magnitude & Radial/Azimuthal Profiles ──────────────────────
def analyze_frequency_fft(img_gray: np.ndarray) -> Tuple[List[float], float, List[float]]:
    """Computes 2D FFT, log-radial averaging, and HFER."""
    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    mag = np.abs(f_shift)
    
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x**2 + y**2)
    
    # Radial average
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), mag.ravel())
    nr = np.bincount(r_int.ravel())
    radial_avg = np.log(tbin / (nr + 1e-12) + 1.0)
    
    # HFER (High-Frequency Energy Ratio)
    r_max = np.max(r)
    high_freq_mask = r > (0.8 * r_max)
    hfer = float(np.sum(mag[high_freq_mask]) / (np.sum(mag) + 1e-12))
    
    # Azimuthal profile for upsampling detection
    theta = np.arctan2(y, x)
    theta_deg = ((theta + np.pi) * 180 / np.pi).astype(int) % 360
    azimuthal_sum = np.bincount(theta_deg.ravel(), mag.ravel(), minlength=360)
    azimuthal_count = np.bincount(theta_deg.ravel(), minlength=360)
    azimuthal_avg = (azimuthal_sum / (azimuthal_count + 1e-12)).tolist()
    
    return radial_avg.tolist(), hfer, azimuthal_avg

# ── Helper: 2D DCT & Histogram Divergence ──────────────────────────────────────
def analyze_dct(img_gray: np.ndarray) -> Tuple[List[float], float]:
    """Computes 2D DCT coefficient histogram and Symmetric KL Divergence."""
    from scipy.fft import dctn
    dct_coeffs = dctn(img_gray, norm='ortho')
    ac_coeffs = dct_coeffs.ravel()[1:] # skip DC term
    
    hist, bin_edges = np.histogram(ac_coeffs, bins=100, range=(-100, 100), density=True)
    hist = hist + 1e-12
    hist = hist / np.sum(hist)
    
    # Learned Authentic Reference (Standard Laplacian scale=1.0)
    from scipy.stats import laplace
    ref_dist = laplace.pdf((bin_edges[:-1] + bin_edges[1:]) / 2, loc=0, scale=1.0)
    ref_dist = ref_dist + 1e-12
    ref_dist = ref_dist / np.sum(ref_dist)
    
    # Symmetric KL
    kl_pq = np.sum(hist * np.log(hist / ref_dist))
    kl_qp = np.sum(ref_dist * np.log(ref_dist / hist))
    sym_kl = float(0.5 * (kl_pq + kl_qp))
    
    return hist.tolist(), sym_kl

# ── Helper: Texture Analysis (LBP, GLCM, Gradients) ──────────────────────────
def analyze_texture(img_gray: np.ndarray) -> Tuple[List[float], Dict[str, float], float]:
    """Computes LBP, GLCM, and Gradient Magnitude JS Divergence."""
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    
    # LBP at 4 scales
    lbp_hist = []
    for radius in [1, 2, 4, 8]:
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        h, _ = np.histogram(lbp.ravel(), bins=n_points+2, density=True)
        lbp_hist.extend(h.tolist())
        
    # GLCM
    glcm = graycomatrix((img_gray * 255).astype(np.uint8), distances=[1, 2, 4],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
                         symmetric=True, normed=True)
    glcm_feats = {}
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        glcm_feats[prop] = float(np.mean(graycoprops(glcm, prop)))
        
    # Gradient Magnitudes JS Divergence
    from scipy.ndimage import sobel
    sx = sobel(img_gray, axis=0)
    sy = sobel(img_gray, axis=1)
    grad_mag = np.sqrt(sx**2 + sy**2)
    h_grad, _ = np.histogram(grad_mag.ravel(), bins=50, range=(0, 2), density=True)
    h_grad = h_grad + 1e-12
    h_grad = h_grad / np.sum(h_grad)
    
    # Reference grad manifold
    ref_grad = np.exp(-np.linspace(0, 2, 50))
    ref_grad = ref_grad + 1e-12
    ref_grad = ref_grad / np.sum(ref_grad)
    
    # Jensen-Shannon Divergence
    m = 0.5 * (h_grad + ref_grad)
    js_div = float(0.5 * (np.sum(h_grad * np.log(h_grad / m)) + np.sum(ref_grad * np.log(ref_grad / m))))
    
    return lbp_hist, glcm_feats, js_div

# ── Helper: Deep CNN Batch Inference (TorchServe gRPC/HTTP client) ──────────────
def analyze_deep_cnn(img: Image.Image) -> Tuple[float, str, List[List[float]]]:
    """Slices image into 16x16 grid and batches inference to TorchServe."""
    w, h = img.size
    tile_w, tile_h = w // 16, h // 16
    
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        response = requests.post(settings.TORCHSERVE_URL, data=img_bytes, timeout=5.0)
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list) and len(res) > 0:
                data = res[0]
                cnn_score = float(data.get("fake_probability", 0.5))
                fam_dict = data.get("generator_family", {})
                family = max(fam_dict, key=fam_dict.get) if fam_dict else "UNKNOWN"
                heatmap = data.get("patch_heatmap", [[0.5]*16 for _ in range(16)])
                return cnn_score, family, heatmap
            elif isinstance(res, dict):
                cnn_score = float(res.get("fake_probability", 0.5))
                fam_dict = res.get("generator_family", {})
                family = max(fam_dict, key=fam_dict.get) if fam_dict else "UNKNOWN"
                heatmap = res.get("patch_heatmap", [[0.5]*16 for _ in range(16)])
                return cnn_score, family, heatmap
    except Exception as e:
        log.warning(f"TorchServe connection failed: {e}. Falling back to default baseline values.")
        
    return 0.5, "UNKNOWN", [[0.5]*16 for _ in range(16)]

# ── Central Execution Pipeline ──────────────────────────────────────────────────
def process_image_analysis(image_bytes: bytes, camera_model: str) -> Dict[str, Any]:
    """Runs all forensic pipelines sequentially and structures the result."""
    t0 = time.time()
    
    # 1. Decode to Pillow RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Decoded image has zero dimensions.")
        
    img_gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    
    # 2. PRNU Pipeline
    # Extract noise residual R = I - W(I)
    R = img_gray - wavelet_wiener_denoise(img_gray, sigma=3.5)
    
    # Retrieve Camera Model Reference PRNU pattern from Qdrant
    prnu_score = 0.0
    prnu_map = [[0.0]]
    try:
        qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        # Search for pattern by camera_model
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter={"must": [{"key": "camera_model", "match": {"value": camera_model}}]},
            limit=1
        )
        if results and results[0]:
            # Load stored camera model PRNU reference
            ref_pattern_list = results[0][0].payload.get("prnu_pattern")
            if ref_pattern_list:
                K = np.array(ref_pattern_list, dtype=np.float32)
                prnu_score = pearson_correlation(R, K)
                prnu_map = prnu_tamper_map(R, K, block_size=64)
    except Exception as e:
        log.warning(f"Failed loading reference PRNU from Qdrant: {e}. Defaulting to dummy reference pattern.")
        K = np.zeros_like(R)
        prnu_score = pearson_correlation(R, K)
        prnu_map = prnu_tamper_map(R, K, block_size=64)

    # 3. Frequency Domain
    radial_avg, hfer, azimuthal_avg = analyze_frequency_fft(img_gray)
    dct_hist, sym_kl = analyze_dct(img_gray)
    
    # 4. Texture Forensics
    lbp_hist, glcm_feats, js_div = analyze_texture(img_gray)
    
    # 5. Deep CNN Inference
    cnn_score, generator_family, patch_heatmap = analyze_deep_cnn(img)
    
    # Aggregate Score calculations
    freq_score = float(hfer * 0.7 + (min(sym_kl, 5.0)/5.0) * 0.3)
    texture_score = float(js_div)
    
    latency = time.time() - t0
    METRIC_LATENCY.observe(latency)
    
    return {
        "prnu_score": prnu_score,
        "prnu_map": prnu_map,
        "freq_score": freq_score,
        "freq_radial_avg": radial_avg,
        "freq_azimuthal_avg": azimuthal_avg,
        "freq_dct_hist": dct_hist,
        "texture_score": texture_score,
        "texture_lbp_hist": lbp_hist,
        "texture_glcm": glcm_feats,
        "cnn_score": cnn_score,
        "patch_confidence_grid": patch_heatmap,
        "generator_family": generator_family,
        "latency_sec": latency
    }

# ── Kafka Consumer & Publisher Loop ───────────────────────────────────────────
def kafka_worker_thread(partition_id: int):
    """Kafka consumer thread worker representing a single concurrent partition worker."""
    thread_name = f"Worker-Partition-{partition_id}"
    threading.current_thread().name = thread_name
    log.info("Worker thread initialized and starting.")
    
    # Configure Kafka Consumer
    conf = {
        'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
        'group.id': settings.KAFKA_CONSUMER_GROUP,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    }
    if settings.KAFKA_SASL_USERNAME:
        conf.update({
            'security.protocol': settings.KAFKA_SECURITY_PROTOCOL,
            'sasl.mechanism': settings.KAFKA_SASL_MECHANISM,
            'sasl.username': settings.KAFKA_SASL_USERNAME,
            'sasl.password': settings.KAFKA_SASL_PASSWORD
        })
        
    consumer = Consumer(conf)
    consumer.subscribe([settings.KAFKA_INPUT_TOPIC])
    
    # Configure Kafka Producer
    prod_conf = {'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS}
    if settings.KAFKA_SASL_USERNAME:
        prod_conf.update({
            'security.protocol': settings.KAFKA_SECURITY_PROTOCOL,
            'sasl.mechanism': settings.KAFKA_SASL_MECHANISM,
            'sasl.username': settings.KAFKA_SASL_USERNAME,
            'sasl.password': settings.KAFKA_SASL_PASSWORD
        })
    producer = Producer(prod_conf)
    
    # Initialize Storage and Cache clients inside thread
    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD)
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE
    )
    
    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    log.error(f"Kafka error: {msg.error()}")
                    continue
            
            # Parse Event
            payload = json.loads(msg.value().decode('utf-8'))
            case_id = payload.get("case_id")
            s3_key = payload.get("s3_key")
            camera_model = payload.get("camera_model", "GENERIC")
            
            if not case_id or not s3_key:
                log.error("Received payload missing case_id or s3_key. Skipping.")
                consumer.commit(msg, asynchronous=False)
                continue
                
            log.info(f"Processing event case_id={case_id} s3_key={s3_key}")
            
            # Download Media from MinIO S3
            try:
                response = minio_client.get_object(settings.MINIO_BUCKET_NAME, s3_key)
                image_bytes = response.read()
                response.close()
                response.release_conn()
                
                # Execute Pipeline
                result = process_image_analysis(image_bytes, camera_model)
                result.update({
                    "case_id": case_id,
                    "module": "image_forensics",
                    "status": "SUCCESS",
                    "timestamp": time.time(),
                    "error_reason": None
                })
                METRIC_REQUESTS_TOTAL.labels(status="SUCCESS").inc()
                
            except Exception as ex:
                log.exception(f"Pipeline error processing case_id={case_id}")
                # Create standard failure result payload to ensure pipeline never crashes
                result = {
                    "case_id": case_id,
                    "module": "image_forensics",
                    "status": "ERROR",
                    "error_reason": str(ex),
                    "prnu_score": 0.0,
                    "prnu_map": [[0.0]],
                    "freq_score": 0.0,
                    "freq_radial_avg": [],
                    "freq_azimuthal_avg": [],
                    "freq_dct_hist": [],
                    "texture_score": 0.0,
                    "texture_lbp_hist": [],
                    "texture_glcm": {},
                    "cnn_score": 0.5,
                    "patch_confidence_grid": [[0.5]*16 for _ in range(16)],
                    "generator_family": "UNKNOWN",
                    "timestamp": time.time()
                }
                METRIC_REQUESTS_TOTAL.labels(status="ERROR").inc()
                
            # Publish result to Redis with TTL
            try:
                redis_client.setex(f"analysis:image:{case_id}", settings.REDIS_TTL, json.dumps(result))
            except Exception as rx:
                log.error(f"Failed writing results to Redis: {rx}")
                
            # Publish result to Kafka completion topic
            try:
                producer.produce(
                    settings.KAFKA_OUTPUT_TOPIC,
                    key=case_id.encode('utf-8'),
                    value=json.dumps(result).encode('utf-8')
                )
                producer.flush()
            except Exception as kx:
                log.error(f"Failed publishing to completion Kafka: {kx}")
                
            # Commit offset
            consumer.commit(msg, asynchronous=False)
            
        except Exception as e:
            log.exception(f"Unexpected worker thread exception: {e}")
            time.sleep(2)

# ── Start concurrent consumer threads and FastAPI Server ───────────────────────
if __name__ == "__main__":
    # Start 8 concurrent Kafka workers corresponding to partitions
    for i in range(8):
        t = threading.Thread(target=kafka_worker_thread, args=(i,), daemon=True)
        t.start()
        
    log.info("All 8 partition workers successfully initialized in background.")
    
    # Start FastAPI metrics/health server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)
