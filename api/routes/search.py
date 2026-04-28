"""
Search router
─────────────
POST /search-face  – find a person's cluster in an event and return their images
"""

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from utils.image_utils import decode_upload
from utils.logger import setup_logger
from utils.storage import event_exists, clusters_exist

logger = setup_logger(__name__)
router = APIRouter()


# ── Response model 
class SearchResponse(BaseModel):
    matched_cluster_id: Optional[int]
    similarity: float
    matched_images: List[str]
    message: str


# ── Endpoint
@router.post(
    "/search-face",
    response_model=SearchResponse,
    summary="Find all images of a person within an event",
    description=(
        "Upload a user's face photo and an event_id. "
        "The service generates the face embedding, compares it against every "
        "cluster centroid stored for that event (using cosine similarity), and "
        "returns the image paths from the best-matching cluster. "
        "A match is only returned when similarity ≥ SIMILARITY_THRESHOLD."
    ),
)
async def search_face(
    event_id: str = Form(..., description="Event to search within."),
    image: UploadFile = File(..., description="User's face image."),
):
    logger.info(
        "POST /search-face  event_id=%s  filename=%s", event_id, image.filename
    )

    # ── Validate event exists
    if not event_exists(event_id):
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found.",
        )
    if not clusters_exist(event_id):
        raise HTTPException(
            status_code=409,
            detail=f"Event '{event_id}' has not been clustered yet. "
                   "Call /process-event first.",
        )

    # ── Decode image 
    try:
        img = decode_upload(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # ── Generate embedding
    try:
        query_embedding = EmbeddingService.get_single_embedding(img)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Embedding failed during search: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Search 
    try:
        result = SearchService.search(
            query_embedding=query_embedding,
            event_id=event_id,
        )
    except Exception as exc:
        logger.exception("Search failed for event '%s': %s", event_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return SearchResponse(**result)