"""
Application-layer routes for the web client.

These endpoints keep the existing ML logic intact while adapting Cloudinary URL
uploads into the current event and face-search pipeline.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from typing import List, Literal

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

from api.routes.events import ProcessEventResponse
from api.routes.search import SearchResponse
from services.cloudinary_service import CloudinaryService
from services.embedding_service import EmbeddingService
from services.event_service import EventService
from services.remote_image_service import RemoteAsset, RemoteImageService
from services.search_service import SearchService
from utils.image_utils import decode_bytes
from utils.logger import setup_logger
from utils.storage import (
    clusters_exist,
    event_exists,
    images_dir,
    load_asset_urls,
    load_clusters,
    save_asset_urls,
    save_clusters,
)

logger = setup_logger(__name__)
router = APIRouter(prefix="/app")


class CloudinarySignatureRequest(BaseModel):
    event_id: str = Field(..., min_length=1)
    asset_type: Literal["events", "queries"] = "events"


class CloudinarySignatureResponse(BaseModel):
    api_key: str
    cloud_name: str
    folder: str
    signature: str
    timestamp: str
    upload_url: str
    use_filename: str
    unique_filename: str


class ProcessEventUrlsRequest(BaseModel):
    event_id: str = Field(..., min_length=1)
    image_urls: List[HttpUrl] = Field(..., min_length=1)
    recluster_eps: float | None = None


class SearchFaceUrlRequest(BaseModel):
    event_id: str = Field(..., min_length=1)
    image_url: HttpUrl


class DownloadAllRequest(BaseModel):
    event_id: str = Field(..., min_length=1)
    image_urls: List[str] = Field(..., min_length=1)


@router.post(
    "/cloudinary/signature",
    response_model=CloudinarySignatureResponse,
    summary="Create a signed payload for direct browser-to-Cloudinary uploads",
)
async def create_cloudinary_signature(payload: CloudinarySignatureRequest):
    folder = CloudinaryService.build_folder(payload.event_id, payload.asset_type)
    return CloudinaryService.build_signature_payload(folder)


@router.post(
    "/process-event-urls",
    response_model=ProcessEventResponse,
    summary="Process Cloudinary image URLs with the existing event pipeline",
)
async def process_event_urls(payload: ProcessEventUrlsRequest):
    logger.info(
        "POST /app/process-event-urls event_id=%s num_images=%d",
        payload.event_id,
        len(payload.image_urls),
    )

    try:
        remote_batch = await RemoteImageService.fetch_event_uploads(
            payload.event_id,
            payload.image_urls,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch image URL: {exc}")

    try:
        result = await run_in_threadpool(
            EventService.process_event,
            payload.event_id,
            remote_batch.uploads,
            payload.recluster_eps,
        )
    except Exception as exc:
        logger.exception("Error processing remote event '%s': %s", payload.event_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    asset_urls = load_asset_urls(payload.event_id) or {}
    asset_urls.update(remote_batch.url_map)
    save_asset_urls(payload.event_id, asset_urls)

    manifest = load_clusters(payload.event_id)
    if manifest is not None:
        manifest = RemoteImageService.apply_url_map(manifest, asset_urls)
        save_clusters(payload.event_id, manifest)
        result["clusters"] = [
            {
                "cluster_id": cluster["cluster_id"],
                "num_images": cluster["num_images"],
                "image_paths": cluster["image_paths"],
            }
            for cluster in manifest["clusters"].values()
        ]

    return result


@router.post(
    "/search-face-url",
    response_model=SearchResponse,
    summary="Search an event using a Cloudinary-hosted query image",
)
async def search_face_url(payload: SearchFaceUrlRequest):
    logger.info("POST /app/search-face-url event_id=%s", payload.event_id)

    if not event_exists(payload.event_id):
        raise HTTPException(status_code=404, detail=f"Event '{payload.event_id}' not found.")
    if not clusters_exist(payload.event_id):
        raise HTTPException(
            status_code=409,
            detail=f"Event '{payload.event_id}' has not been clustered yet.",
        )

    try:
        remote_asset = await RemoteImageService.fetch_single_asset(str(payload.image_url))
        query_img = decode_bytes(remote_asset.content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch image URL: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    try:
        query_embedding = await run_in_threadpool(
            EmbeddingService.get_single_embedding,
            query_img,
        )
        result = await run_in_threadpool(
            SearchService.search,
            query_embedding,
            payload.event_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Remote search failed for event '%s': %s", payload.event_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    asset_urls = load_asset_urls(payload.event_id) or {}
    result["matched_images"] = RemoteImageService.rewrite_image_list(
        result["matched_images"],
        asset_urls,
    )
    result["matched_images"] = _rewrite_local_image_list(
        payload.event_id,
        result["matched_images"],
    )
    return SearchResponse(**result)


@router.get(
    "/event-assets/{event_id}/{image_path:path}",
    summary="Serve a processed event image for the web client",
)
async def get_event_asset(event_id: str, image_path: str):
    root = images_dir(event_id).resolve()
    target = (root / unquote(image_path)).resolve()

    if not _is_relative_to(target, root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(target)


@router.post(
    "/download-all",
    summary="Download all matched images as a single ZIP archive",
)
async def download_all(payload: DownloadAllRequest):
    logger.info(
        "POST /app/download-all event_id=%s num_images=%d",
        payload.event_id,
        len(payload.image_urls),
    )

    try:
        assets = await _load_download_assets(payload.event_id, payload.image_urls)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch image URL: {exc}")

    archive = await run_in_threadpool(_build_zip_archive, payload.event_id, assets)
    headers = {
        "Content-Disposition": f'attachment; filename="{payload.event_id}-matches.zip"'
    }
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


def _build_zip_archive(event_id: str, assets: List[RemoteAsset]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, asset in enumerate(assets, start=1):
            suffix_parts = asset.filename.rsplit(".", 1)
            suffix = suffix_parts[1] if len(suffix_parts) == 2 else "jpg"
            archive.writestr(f"{event_id}-match-{index:02d}.{suffix}", asset.content)
    buffer.seek(0)
    return buffer


async def _load_download_assets(event_id: str, image_urls: List[str]) -> List[RemoteAsset]:
    local_assets: List[RemoteAsset] = []
    remote_urls: List[str] = []

    for image_url in image_urls:
        local_path = _local_asset_path(event_id, image_url)
        if local_path:
            local_assets.append(
                RemoteAsset(
                    filename=local_path.name,
                    content=local_path.read_bytes(),
                    content_type="image/jpeg",
                    source_url=image_url,
                )
            )
        else:
            remote_urls.append(image_url)

    remote_assets = await RemoteImageService.fetch_assets(remote_urls) if remote_urls else []
    return [*local_assets, *remote_assets]


def _rewrite_local_image_list(event_id: str, image_paths: List[str]) -> List[str]:
    return [_local_image_url(event_id, image_path) or image_path for image_path in image_paths]


def _local_image_url(event_id: str, image_path: str) -> str | None:
    try:
        root = images_dir(event_id).resolve()
        target = Path(image_path).resolve()
    except OSError:
        return None

    if not target.is_file() or not _is_relative_to(target, root):
        return None

    rel_path = target.relative_to(root).as_posix()
    return f"/app/event-assets/{event_id}/{quote(rel_path)}"


def _local_asset_path(event_id: str, image_url: str) -> Path | None:
    parsed = urlparse(image_url)
    path = parsed.path if parsed.scheme else image_url
    prefix = f"/app/event-assets/{event_id}/"

    if not path.startswith(prefix):
        return None

    root = images_dir(event_id).resolve()
    target = (root / unquote(path.removeprefix(prefix))).resolve()
    if not target.is_file() or not _is_relative_to(target, root):
        return None
    return target


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
