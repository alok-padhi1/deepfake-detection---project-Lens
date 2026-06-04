# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "LENS Fusion Microservice"
    API_PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SASL_MECHANISM: str = "PLAIN"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None
    KAFKA_CONSUMER_GROUP: str = "fusion-workers"
    KAFKA_INPUT_TOPIC: str = "analysis.fusion.trigger"
    KAFKA_OUTPUT_TOPIC: str = "case.final_result"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # Postgres Settings
    POSTGRES_DSN: str = "postgresql://postgres:postgres@localhost:5432/lensdb"

    # TorchServe Settings
    TORCHSERVE_URL: str = "http://localhost:8080/predictions/bayesian-fusion-mlp"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
