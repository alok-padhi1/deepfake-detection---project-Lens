# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "LENS Video Forensics Microservice"
    API_PORT: int = 8002
    LOG_LEVEL: str = "INFO"

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SASL_MECHANISM: str = "PLAIN"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None
    KAFKA_CONSUMER_GROUP: str = "video-forensics-workers"
    KAFKA_INPUT_TOPIC: str = "analysis.video"
    KAFKA_OUTPUT_TOPIC: str = "analysis.complete"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # MinIO Settings
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False

    # TorchServe Settings
    TORCHSERVE_URL: str = "http://localhost:8080/predictions/timesformer"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
