import os
import json
import logging
import psycopg2
import requests
from kafka import KafkaProducer
from psycopg2.extras import DictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("drift-detector")

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://lens-monitoring-prometheus.monitoring.svc.cluster.local:9090")
PG_HOST = os.environ.get("PG_HOST", "postgresql.lens.svc.cluster.local")
PG_USER = os.environ.get("PG_USER", "lens_admin")
PG_PASS = os.environ.get("PG_PASS", "postgres_password")
PG_DB = os.environ.get("PG_DB", "lens_db")
KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "strimzi-kafka-bootstrap.lens.svc.cluster.local:9092")
DRIFT_TOPIC = "drift.detected"
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.08"))

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def get_db_connection():
    return psycopg2.connect(host=PG_HOST, dbname=PG_DB, user=PG_USER, password=PG_PASS)

def fetch_prometheus_rolling_mean():
    """Queries Prometheus for the 7-day rolling mean of module score distributions."""
    query = "avg_over_time(module_score_distribution{namespace='lens'}[7d])"
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    current_means = {}
    for result in data['data']['result']:
        metric = result['metric']
        module = metric.get('module')
        media_type = metric.get('media_type')
        if module and media_type:
            key = f"{module}_{media_type}"
            current_means[key] = float(result['value'][1])
            
    return current_means

def fetch_baselines_from_db():
    """Fetches the baseline scores established after the last retraining."""
    conn = get_db_connection()
    baselines = {}
    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Create table if it doesn't exist for fresh setups
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_baselines (
                module_name VARCHAR(100),
                media_type VARCHAR(50),
                baseline_mean FLOAT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (module_name, media_type)
            )
        """)
        cur.execute("SELECT module_name, media_type, baseline_mean FROM model_baselines")
        rows = cur.fetchall()
        for row in rows:
            key = f"{row['module_name']}_{row['media_type']}"
            baselines[key] = float(row['baseline_mean'])
    conn.close()
    return baselines

def analyze_drift():
    logger.info("Starting model drift analysis...")
    try:
        current_means = fetch_prometheus_rolling_mean()
        baselines = fetch_baselines_from_db()
    except Exception as e:
        logger.error(f"Failed to fetch metrics or baselines: {e}")
        return

    for key, current_mean in current_means.items():
        if key not in baselines:
            logger.warning(f"No baseline found for {key}. Initialising baseline with current 7-day mean: {current_mean}")
            parts = key.split('_')
            media_type = parts[-1]
            module_name = '_'.join(parts[:-1])
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO model_baselines (module_name, media_type, baseline_mean) VALUES (%s, %s, %s) ON CONFLICT (module_name, media_type) DO NOTHING",
                    (module_name, media_type, current_mean)
                )
            conn.commit()
            conn.close()
            continue

        baseline_mean = baselines[key]
        drift_delta = abs(current_mean - baseline_mean)
        
        logger.info(f"Module {key}: Current 7d-mean={current_mean:.4f}, Baseline={baseline_mean:.4f}, Delta={drift_delta:.4f}")

        if drift_delta > DRIFT_THRESHOLD:
            logger.warning(f"🚨 DRIFT DETECTED for {key}! Delta {drift_delta:.4f} > Threshold {DRIFT_THRESHOLD}")
            parts = key.split('_')
            media_type = parts[-1]
            module_name = '_'.join(parts[:-1])
            
            payload = {
                "module_name": module_name,
                "media_type": media_type,
                "drift_delta": drift_delta,
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
                "action": "trigger_retraining"
            }
            producer.send(DRIFT_TOPIC, payload)
            logger.info(f"Published drift alert to {DRIFT_TOPIC}")

    producer.flush()
    logger.info("Drift analysis complete.")

if __name__ == "__main__":
    analyze_drift()
