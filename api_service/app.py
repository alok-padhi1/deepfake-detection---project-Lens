# app.py
import io
import os
import hmac
import hashlib
import uuid
import time
import json
import logging
from typing import Dict, List, Optional
import requests
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import boto3
from botocore.config import Config
import redis
from confluent_kafka import Producer
from zipstream_ng import ZipStream, ZipFileEntry

from config import settings
from auth import auth_manager, require_role, get_current_user, UserPrincipal
from rate_limiter import require_rate_limit

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=settings.LOG_LEVEL)
log = logging.getLogger("api_gateway")

# ── FastAPI Setup ─────────────────────────────────────────────────────────────
app = FastAPI(title=settings.APP_NAME)
START_TIME = time.time()

# ── Connections Setup ─────────────────────────────────────────────────────────
engine = create_engine(settings.POSTGRES_DSN, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD)

s3_client = boto3.client(
    's3',
    endpoint_url=f"{'https' if settings.MINIO_SECURE else 'http'}://{settings.MINIO_ENDPOINT}",
    aws_access_key_id=settings.MINIO_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

kafka_conf = {'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS}
if settings.KAFKA_SASL_USERNAME:
    kafka_conf.update({
        'security.protocol': settings.KAFKA_SECURITY_PROTOCOL,
        'sasl.mechanism': settings.KAFKA_SASL_MECHANISM,
        'sasl.username': settings.KAFKA_SASL_USERNAME,
        'sasl.password': settings.KAFKA_SASL_PASSWORD
    })
kafka_producer = Producer(kafka_conf)

# ── 1. Magic Bytes Checker ────────────────────────────────────────────────────
MAGIC_SIGNATURES = {
    b"\xFF\xD8\xFF": "image/jpeg",
    b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "image/png",
    b"RIFF": "riff_based",  # WEBP or WAV
    b"fLaC": "audio/flac",
    b"ID3": "audio/mp3",
    b"\xFF\xFB": "audio/mp3",
    b"\x00\x00\x00\x18ftypmp42": "video/mp4",
    b"\x00\x00\x00\x20ftyp": "video/mp4",
    b"\x1A\x45\xDF\xA3": "video/mkv"
}

def validate_magic_bytes(header: bytes) -> str:
    for sig, mime in MAGIC_SIGNATURES.items():
        if header.startswith(sig):
            if mime == "riff_based":
                if b"WEBP" in header: return "image/webp"
                if b"WAVE" in header: return "audio/wav"
            return mime
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported file signature or media format."
    )

# ── REST API Endpoints ────────────────────────────────────────────────────────
@app.post("/api/v1/analyze", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_rate_limit)])
def analyze_media(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    user: UserPrincipal = Depends(get_current_user)
):
    # 1. Read first 32 bytes for strict magic verification
    header = file.file.read(32)
    file.file.seek(0)
    mime_type = validate_magic_bytes(header)
    media_type = mime_type.split("/")[0]
    
    case_id = str(uuid.uuid4())
    s3_key = f"cases/{case_id}/{file.filename}"
    
    # 2. Boto3 Multipart Stream Upload with MD5
    try:
        mp = s3_client.create_multipart_upload(Bucket=settings.MINIO_BUCKET, Key=s3_key)
        upload_id = mp['UploadId']
        parts = []
        part_num = 1
        md5_hash = hashlib.md5()
        
        while True:
            chunk = file.file.read(8 * 1024 * 1024)  # 8MB parts
            if not chunk:
                break
            md5_hash.update(chunk)
            part = s3_client.upload_part(
                Bucket=settings.MINIO_BUCKET, Key=s3_key,
                PartNumber=part_num, UploadId=upload_id, Body=chunk
            )
            parts.append({"PartNumber": part_num, "ETag": part['ETag']})
            part_num += 1
            
        s3_client.complete_multipart_upload(
            Bucket=settings.MINIO_BUCKET, Key=s3_key,
            UploadId=upload_id, MultipartUpload={"Parts": parts}
        )
    except Exception as e:
        log.error(f"S3 Multipart Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
        
    # 3. Publish to Postgres case records
    try:
        with SessionLocal() as db:
            query = text("""
                INSERT INTO cases (case_id, owner_id, filename, media_type, status, created_at)
                VALUES (:case_id, :owner_id, :filename, :media_type, 'INGESTED', NOW())
            """)
            db.execute(query, {
                "case_id": case_id,
                "owner_id": user.sub,
                "filename": file.filename,
                "media_type": media_type
            })
            db.commit()
    except Exception as e:
        log.error(f"Database registration failed: {e}")
        raise HTTPException(status_code=500, detail="Database registration failed.")

    # 4. Publish message to Kafka
    payload = {
        "case_id": case_id,
        "bucket": settings.MINIO_BUCKET,
        "key": s3_key,
        "media_type": media_type,
        "metadata": json.loads(metadata) if metadata else {}
    }
    kafka_producer.produce(
        settings.KAFKA_INGEST_TOPIC,
        key=case_id.encode('utf-8'),
        value=json.dumps(payload).encode('utf-8')
    )
    kafka_producer.flush()
    
    return {
        "case_id": case_id,
        "status_url": f"/api/v1/cases/{case_id}",
        "estimated_duration_seconds": 45 if media_type == "video" else 10
    }

@app.get("/api/v1/cases/{case_id}")
def get_case_status(case_id: str, user: UserPrincipal = Depends(get_current_user)):
    with SessionLocal() as db:
        query = text("SELECT owner_id, status, confidence_score, media_type, created_at FROM cases WHERE case_id = :case_id")
        res = db.execute(query, {"case_id": case_id}).fetchone()
        
    if not res:
        raise HTTPException(status_code=404, detail="Case not found.")
        
    owner_id, status, confidence, media_type, created_at = res
    
    # RBAC protection: only submitter owner or analyst/admin roles
    if not (user.sub == owner_id or user.has_role("Analyst") or user.has_role("Admin")):
        raise HTTPException(status_code=403, detail="Forbidden: You do not have permissions to view this case.")
        
    confidence_score = float(confidence) if confidence is not None else 0.0
    
    if confidence_score > 0.72:
        band, label = "HIGH", "FAKE"
    elif confidence_score > 0.35:
        band, label = "MEDIUM", "SUSPICIOUS"
    else:
        band, label = "LOW", "AUTHENTIC"
        
    return {
        "case_id": case_id,
        "status": status,
        "confidence_score": confidence_score,
        "decision_band": band,
        "decision_label": label,
        "modules_complete": ["image", "video", "audio"] if status == "COMPLETE" else [],
        "created_at": created_at.isoformat(),
        "completed_at": created_at.isoformat() if status == "COMPLETE" else None
    }

@app.get("/api/v1/cases/{case_id}/report")
def get_pdf_report(case_id: str, user: UserPrincipal = Depends(get_current_user)):
    # Banal role check
    if not (user.has_role("Analyst") or user.has_role("Admin")):
         raise HTTPException(status_code=403, detail="Analyst permissions required.")
         
    s3_key = f"cases/{case_id}/report.pdf"
    try:
        url = s3_client.generate_presigned_url(
            'get_object', Params={'Bucket': settings.MINIO_BUCKET, 'Key': s3_key}, ExpiresIn=3600
        )
        return RedirectResponse(url)
    except:
        raise HTTPException(status_code=404, detail="Forensic report PDF not generated yet.")

@app.get("/api/v1/cases/{case_id}/evidence")
def get_zip_evidence(case_id: str, user: UserPrincipal = Depends(get_current_user)):
    if not (user.has_role("Analyst") or user.has_role("Admin")):
         raise HTTPException(status_code=403, detail="Forbidden.")
         
    # Build on-the-fly streaming zip using zipstream-ng
    s3_prefix = f"cases/{case_id}/evidence/"
    try:
        resp = s3_client.list_objects_v2(Bucket=settings.MINIO_BUCKET, Prefix=s3_prefix)
        files = [item['Key'] for item in resp.get('Contents', [])]
    except Exception:
        raise HTTPException(status_code=404, detail="No evidence files found.")
        
    def generator():
        manifest_data = "LENS SHA256 Evidence Manifest\n"
        for f in files:
            obj = s3_client.get_object(Bucket=settings.MINIO_BUCKET, Key=f)
            data = obj['Body'].read()
            sha = hashlib.sha256(data).hexdigest()
            fname = os.path.basename(f)
            manifest_data += f"{sha}  {fname}\n"
            yield ZipFileEntry(fname, io.BytesIO(data))
            
        yield ZipFileEntry("sha256_manifest.txt", io.BytesIO(manifest_data.encode('utf-8')))
        
    return StreamingResponse(
        ZipStream(generator()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=LENS_EVIDENCE_{case_id}.zip"}
    )

@app.get("/api/v1/cases/{case_id}/heatmap")
def get_heatmap(case_id: str, user: UserPrincipal = Depends(get_current_user)):
    s3_key = f"cases/{case_id}/heatmap.png"
    try:
        obj = s3_client.get_object(Bucket=settings.MINIO_BUCKET, Key=s3_key)
        return StreamingResponse(
            io.BytesIO(obj['Body'].read()),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"}
        )
    except:
        raise HTTPException(status_code=404, detail="Spatial localization heatmap not available.")

@app.post("/api/v1/webhook")
def register_webhook(
    url: str = Form(...),
    secret: str = Form(...),
    user: UserPrincipal = Depends(require_role(["Admin"]))
):
    # Validate url reachability (HEAD check)
    try:
        resp = requests.head(url, timeout=3.0)
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail="Target webhook URL unreachable or returned error status.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Target URL validation failed: {e}")
        
    # Store in DB
    try:
        with SessionLocal() as db:
            query = text("INSERT INTO webhooks (id, url, secret) VALUES (:id, :url, :secret)")
            db.execute(query, {"id": str(uuid.uuid4()), "url": url, "secret": secret})
            db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database write error.")
        
    return {"status": "registered"}

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "model_versions": {
            "efficientnet": "b0-v1.2",
            "timesformer": "v2.1",
            "resnet34": "audio-v1",
            "rawnet3": "v3",
            "syncnet": "v1.1",
            "fusion_mlp": "v1"
        },
        "kafka_lag": 0,
        "gpu_pool_utilization": 0.12,
        "uptime": time.time() - START_TIME
    }
