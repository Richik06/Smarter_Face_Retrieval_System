"""
Centralised configuration loaded from environment variables.
All tuneable parameters live here - no magic numbers scattered around the code.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EVENT_DATA_DIR: Path = Path("event_data")

    DETECTOR_BACKEND: str = "retinaface"

    EMBEDDING_MODEL: str = "Facenet512"
    EMBEDDING_DIM: int = 512

    DBSCAN_EPS: float = 0.35
    DBSCAN_MIN_SAMPLES: int = 1
    DBSCAN_METRIC: str = "cosine"

    SIMILARITY_THRESHOLD: float = 0.55
    USE_FAISS: bool = True

    CLOUDINARY_CLOUD_NAME: str = "demo-cloud"
    CLOUDINARY_API_KEY: str = "1234567890"
    CLOUDINARY_API_SECRET: str = "abcdefg"
    CLOUDINARY_UPLOAD_FOLDER: str = "face-retrieval"

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
