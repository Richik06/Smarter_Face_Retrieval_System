"""
Events router
─────────────
POST /process-event   – accept images, run full pipeline, return cluster manifest
POST /recluster-event – re-run DBSCAN on existing embeddings (after tuning or new images)
GET  /event/{event_id} – retrieve current cluster manifest for an event
"""

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from services.event_service import EventService
from utils.logger import setup_logger
from utils.storage import event_exists, load_clusters

logger = setup_logger(__name__)
router = APIRouter()


# ── Response models
class ClusterSummary(BaseModel):
    cluster_id: int
    num_images: int
    image_paths: List[str]


class ProcessEventResponse(BaseModel):
    event_id: str
    num_people_detected: int
    clusters: List[ClusterSummary]
    images_with_no_face: Optional[List[str]] = None
    warning: Optional[str] = None
    message: Optional[str] = None


# ── Endpoints

@router.post(
    "/process-event",
    response_model=ProcessEventResponse,
    summary="Upload event images, detect faces, and cluster them",
    description=(
        "Accepts one or more images for an event. "
        "Each detected face is embedded and grouped into clusters using DBSCAN. "
        "One cluster = one person. "
        "Supports incremental uploads: calling this endpoint again with the same "
        "event_id will append new images to the existing pool and re-cluster."
    ),
)
async def process_event(
    event_id: str = Form(..., description="Unique identifier for the event."),
    images: List[UploadFile] = File(..., description="One or more image files."),
    recluster_eps: Optional[float] = Form(
        None,
        description=(
            "Override DBSCAN eps for this run. "
            "Leave blank to use the server default."
        ),
    ),
):
    logger.info(
        "POST /process-event  event_id=%s  num_images=%d", event_id, len(images)
    )
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    try:
        result = EventService.process_event(
            event_id=event_id,
            images=images,
            recluster_eps=recluster_eps,
        )
    except Exception as exc:
        logger.exception("Error processing event '%s': %s", event_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@router.post(
    "/recluster-event",
    response_model=ProcessEventResponse,
    summary="Re-cluster an existing event with new parameters",
    description=(
        "Triggers DBSCAN again on the already-stored embeddings for an event. "
        "Use this to tune eps/min_samples or after appending new images."
    ),
)
async def recluster_event(
    event_id: str = Form(...),
    eps: Optional[float] = Form(None, description="New DBSCAN eps override."),
    min_samples: Optional[int] = Form(None, description="New DBSCAN min_samples override."),
):
    logger.info("POST /recluster-event  event_id=%s  eps=%s", event_id, eps)
    if not event_exists(event_id):
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found. Process it first via /process-event.",
        )

    try:
        result = EventService.recluster_event(
            event_id=event_id, eps=eps, min_samples=min_samples
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Re-cluster error for event '%s': %s", event_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@router.get(
    "/event/{event_id}",
    summary="Retrieve current cluster manifest for an event",
)
async def get_event(event_id: str):
    logger.info("GET /event/%s", event_id)
    manifest = load_clusters(event_id)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cluster data found for event '{event_id}'.",
        )
    return manifest