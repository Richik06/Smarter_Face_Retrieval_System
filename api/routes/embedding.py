"""
Embedding router
────────────────
POST /get-embedding  – return the embedding vector for a single face image
"""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.embedding_service import EmbeddingService
from utils.image_utils import decode_upload
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


# ── Response model 

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    embedding_dim: int
    model: str


# ── Endpoint 
@router.post(
    "/get-embedding",
    response_model=EmbeddingResponse,
    summary="Generate a face embedding from a single image",
    description=(
        "Accepts a single image containing one face. "
        "Returns the L2-normalised embedding vector produced by the configured "
        "embedding model (default: Facenet512). "
        "If multiple faces are detected, the largest (most prominent) is used."
    ),
)
async def get_embedding(
    image: UploadFile = File(..., description="Image file containing a face."),
):
    logger.info("POST /get-embedding  filename=%s", image.filename)

    try:
        img = decode_upload(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    try:
        embedding = EmbeddingService.get_single_embedding(img)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Embedding generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    from config import settings
    return EmbeddingResponse(
        embedding=embedding.tolist(),
        embedding_dim=len(embedding),
        model=settings.EMBEDDING_MODEL,
    )