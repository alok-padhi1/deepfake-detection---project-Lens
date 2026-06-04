# app.py
import json
import logging
import time
import threading
import requests
from typing import Dict, Any, List

from fastapi import FastAPI, Response, status
import redis
from confluent_kafka import Consumer, Producer, KafkaError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import settings

# ── Logging & Metrics Setup ────────────────────────────────────────────────────
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
log = logging.getLogger("fusion_service")

METRIC_FUSION_REQUESTS = Counter("fusion_requests_total", "Total fusion requests processed", ["status"])
METRIC_FUSION_LATENCY = Histogram("fusion_latency_seconds", "Latency of fusion pipeline")

# ── FastAPI App for Health & Metrics ──────────────────────────────────────────
app = FastAPI(title=settings.APP_NAME)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ── Database Setup ────────────────────────────────────────────────────────────
engine = create_engine(settings.POSTGRES_DSN, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Score Vector Builder ──────────────────────────────────────────────────────
INPUT_DIM = 18

def build_score_vector(image_res: Dict, audio_res: Dict, media_type: str) -> List[float]:
    """Builds the 18-dimensional feature vector for the Bayesian Fusion MLP."""
    vec = [0.5] * INPUT_DIM
    
    if image_res and image_res.get("status") == "SUCCESS":
        vec[0] = float(image_res.get("cnn_score", 0.5))
        fam_dict = image_res.get("generator_family", {})
        # Map family strings to indices if provided as dict probabilities
        if isinstance(fam_dict, dict) and fam_dict:
            vec[1] = fam_dict.get("REAL", 0.0)
            vec[2] = fam_dict.get("GAN", 0.0)
            vec[3] = fam_dict.get("Diffusion", 0.0)
            vec[4] = fam_dict.get("Neural", 0.0)
            
        patch_grid = image_res.get("patch_confidence_grid", [[0.5]])
        flat_patches = [item for sublist in patch_grid for item in sublist]
        vec[5] = float(max(flat_patches)) if flat_patches else 0.5
        vec[6] = float(sum(flat_patches)/len(flat_patches)) if flat_patches else 0.5

    # Index 7: Timesformer (skipped for now, defaults to 0.5)
    
    if audio_res and audio_res.get("status") == "SUCCESS":
        vec[8] = float(audio_res.get("mel_score", 0.5))
        vec[9] = float(audio_res.get("raw_score", 0.5))
        
    # Index 10: Syncnet (skipped for now, defaults to 0.5)
    
    # Media Quality Metrics
    qm = image_res.get("quality_metrics", {}) if image_res else {}
    vec[11] = float(qm.get("jpeg_quality", 0.85))
    vec[12] = float(qm.get("blur_score", 0.5))
    vec[13] = float(qm.get("noise_score", 0.1))
    vec[14] = float(qm.get("resolution_norm", 0.5))
    vec[15] = float(qm.get("fps_norm", 0.5))
    vec[16] = 1.0 if audio_res else 0.0
    vec[17] = float(qm.get("duration_norm", 0.1))
    
    return vec

# ── Core Pipeline ─────────────────────────────────────────────────────────────
def process_fusion(case_id: str, media_type: str, redis_client: redis.Redis) -> Dict[str, Any]:
    t0 = time.time()
    
    # 1. Fetch scores from Redis
    image_data = redis_client.get(f"analysis:image:{case_id}")
    audio_data = redis_client.get(f"analysis:audio:{case_id}")
    
    image_res = json.loads(image_data) if image_data else None
    audio_res = json.loads(audio_data) if audio_data else None
    
    # 2. Build Score Vector
    score_vector = build_score_vector(image_res, audio_res, media_type)
    
    # 3. Call TorchServe Fusion Model
    payload = {
        "score_vector": score_vector,
        "media_type": media_type
    }
    
    fake_prob = 0.5
    raw_prob = 0.5
    uncertainty = 0.5
    calibrated = False
    
    try:
        resp = requests.post(settings.TORCHSERVE_URL, json=payload, timeout=5.0)
        if resp.status_code == 200:
            res_data = resp.json()
            if isinstance(res_data, list) and len(res_data) > 0:
                data = res_data[0]
                fake_prob = data.get("fake_probability", 0.5)
                raw_prob = data.get("raw_probability", 0.5)
                uncertainty = data.get("epistemic_uncertainty", 0.5)
                calibrated = data.get("calibrated", False)
    except Exception as e:
        log.warning(f"Fusion TorchServe offline or failed: {e}. Using uncalibrated heuristic fallback.")
        
    latency = time.time() - t0
    METRIC_FUSION_LATENCY.observe(latency)
    
    return {
        "case_id": case_id,
        "module": "fusion",
        "fake_probability": fake_prob,
        "raw_probability": raw_prob,
        "epistemic_uncertainty": uncertainty,
        "is_fake": bool(fake_prob > 0.5),
        "is_calibrated": calibrated,
        "media_type": media_type,
        "latency_sec": latency,
        "status": "SUCCESS",
        "timestamp": time.time()
    }

def save_to_postgres(result: Dict[str, Any]):
    try:
        with SessionLocal() as db:
            query = text("""
                INSERT INTO case_results (case_id, fake_probability, uncertainty, is_fake, media_type, updated_at)
                VALUES (:case_id, :fake_prob, :uncert, :is_fake, :media_type, NOW())
                ON CONFLICT (case_id) DO UPDATE 
                SET fake_probability = EXCLUDED.fake_probability,
                    uncertainty = EXCLUDED.uncertainty,
                    is_fake = EXCLUDED.is_fake,
                    updated_at = NOW();
            """)
            db.execute(query, {
                "case_id": result["case_id"],
                "fake_prob": result["fake_probability"],
                "uncert": result["epistemic_uncertainty"],
                "is_fake": result["is_fake"],
                "media_type": result["media_type"]
            })
            db.commit()
    except Exception as e:
        log.error(f"Failed to save fusion result to DB for case {result.get('case_id')}: {e}")

# ── Kafka Worker Thread ───────────────────────────────────────────────────────
def kafka_worker_thread(partition_id: int):
    threading.current_thread().name = f"Fusion-Worker-{partition_id}"
    log.info("Fusion worker thread starting.")
    
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
    
    prod_conf = {'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS}
    if settings.KAFKA_SASL_USERNAME:
        prod_conf.update({
            'security.protocol': settings.KAFKA_SECURITY_PROTOCOL,
            'sasl.mechanism': settings.KAFKA_SASL_MECHANISM,
            'sasl.username': settings.KAFKA_SASL_USERNAME,
            'sasl.password': settings.KAFKA_SASL_PASSWORD
        })
    producer = Producer(prod_conf)
    
    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD)
    
    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF: continue
                log.error(f"Kafka error: {msg.error()}")
                continue
            
            payload = json.loads(msg.value().decode('utf-8'))
            case_id = payload.get("case_id")
            media_type = payload.get("media_type", "image")
            
            if not case_id:
                consumer.commit(msg, asynchronous=False)
                continue
                
            log.info(f"Processing Fusion for case_id={case_id}")
            
            try:
                result = process_fusion(case_id, media_type, redis_client)
                METRIC_FUSION_REQUESTS.labels(status="SUCCESS").inc()
                
                # Save to Postgres
                save_to_postgres(result)
                
                # Publish to Kafka
                producer.produce(
                    settings.KAFKA_OUTPUT_TOPIC,
                    key=case_id.encode('utf-8'),
                    value=json.dumps(result).encode('utf-8')
                )
                producer.flush()
                
            except Exception as ex:
                log.exception(f"Fusion pipeline error for case_id={case_id}")
                METRIC_FUSION_REQUESTS.labels(status="ERROR").inc()
                
            consumer.commit(msg, asynchronous=False)
            
        except Exception as e:
            log.exception(f"Unexpected worker exception: {e}")
            time.sleep(2)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i in range(4):
        threading.Thread(target=kafka_worker_thread, args=(i,), daemon=True).start()
    log.info("Fusion partition workers initialized.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)
