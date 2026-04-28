"""
Google Drive router
────────────────────
POST /process-event-from-drive  – download images from a Google Drive folder
                                   link, then run the full face pipeline
"""

from typing import List, Optional

from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel

from services.event_service import EventService
from utils.logger import setup_logger
from services.drive_service import GoogleDriveService

logger = setup_logger(__name__)
router = APIRouter()


# ── Response models 

class ClusterSummary(BaseModel):
    cluster_id: int
    num_images: int
    image_paths: List[str]


class DriveEventResponse(BaseModel):
    event_id: str
    num_people_detected: int
    clusters: List[ClusterSummary]
    source: str = "google_drive"
    images_downloaded: Optional[int] = None
    new_faces_processed: Optional[int] = None
    images_with_no_face: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    warning: Optional[str] = None


# ── Endpoint 

@router.post(
    "/process-event-from-drive",
    response_model=DriveEventResponse,
    summary="Import event images from a Google Drive folder link",
    description=(
        "Accepts a public Google Drive folder URL and an event_id. "
        "Downloads all images from the folder automatically, then runs the "
        "full face detection → embedding → clustering pipeline. "
        "No need to upload images one by one. "
        "\n\n**Requirements for the Drive folder:**\n"
        "- Must be shared publicly ('Anyone with the link can view')\n"
        "- Supported formats: JPG, JPEG, PNG, WEBP, BMP\n"
        "\n\n**Setup options:**\n"
        "- Install `gdown` for automatic public folder downloads (recommended)\n"
        "- Or set `GOOGLE_DRIVE_API_KEY` in .env for API-based downloads"
    ),
)
async def process_event_from_drive(
    event_id: str = Form(
        ...,
        description="Unique identifier for the event e.g. 'wedding2024'",
    ),
    drive_link: str = Form(
        ...,
        description=(
            "Public Google Drive folder URL. "
            "Example: https://drive.google.com/drive/folders/ABC123XYZ"
        ),
    ),
    recluster_eps: Optional[float] = Form(
        None,
        description="Override DBSCAN eps for this run. Leave blank for default.",
    ),
):
    logger.info(
        "POST /process-event-from-drive  event_id=%s  link=%s",
        event_id, drive_link,
    )

    # ── Validate Drive link 
    if not GoogleDriveService.is_drive_link(drive_link):
        raise HTTPException(
            status_code=400,
            detail=(
                "The provided URL does not appear to be a Google Drive link. "
                "Expected format: https://drive.google.com/drive/folders/<ID>"
            ),
        )

    # ── Run pipeline 
    try:
        result = EventService.process_event_from_drive(
            event_id=event_id,
            drive_link=drive_link,
            recluster_eps=recluster_eps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception(
            "Error processing Drive event '%s': %s", event_id, exc
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return result