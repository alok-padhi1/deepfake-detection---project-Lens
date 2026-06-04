# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "LENS Image Forensics Microservice"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SASL_MECHANISM: str = "PLAIN"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"  # PLAINTEXT or SASL_PLAINTEXT/SASL_SSL
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None
    KAFKA_CONSUMER_GROUP: str = "image-forensics-workers"
    KAFKA_INPUT_TOPIC: str = "analysis.image"
    KAFKA_OUTPUT_TOPIC: str = "analysis.complete"

    # MinIO / S3 Settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "lens-media"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL: int = 3600

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "prnu_reference_patterns"

    # TorchServe Settings
    TORCHSERVE_URL: str = "http://localhost:8080/predictions/efficientnet-b7-forensic"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
