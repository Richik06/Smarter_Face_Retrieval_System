"""
Centralised configuration loaded from environment variables.
All tuneable parameters live here – no magic numbers scattered around the code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Storage 
    EVENT_DATA_DIR: Path = Path("event_data")

    # ── Face detection
    # Backend used by DeepFace for detection
    DETECTOR_BACKEND: str = "retinaface"   # options: retinaface, mtcnn, opencv

    # ── Face embedding 
    # Primary model; switch to "ArcFace" for the optional alternative
    EMBEDDING_MODEL: str = "Facenet512"
    EMBEDDING_DIM: int = 512               # Facenet512 → 512-d; ArcFace → 512-d

    # ── DBSCAN clustering
    # Cosine distance threshold between faces of the *same* person.
    # Lower  = stricter (more clusters / fewer false-merges).
    # Higher = looser   (fewer clusters / possible false-merges).
    DBSCAN_EPS: float = 0.35
    DBSCAN_MIN_SAMPLES: int = 1   # 1 → every point can be a core point (no noise)
    DBSCAN_METRIC: str = "cosine"

    # ── Similarity search 
    # Minimum cosine similarity required to accept a cluster match (0–1).
    SIMILARITY_THRESHOLD: float = 0.55

    # ── FAISS 
    USE_FAISS: bool = True                 # set False to fall back to NumPy

    # ── Misc 
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure the root storage directory exists at import time
settings.EVENT_DATA_DIR.mkdir(parents=True, exist_ok=True)