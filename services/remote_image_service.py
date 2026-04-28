"""
Helpers for fetching remote image URLs and adapting them to the existing
UploadFile-style event pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import unquote, urlparse

import httpx

from utils.storage import images_dir


@dataclass
class InMemoryUpload:
    filename: str
    file: io.BytesIO


@dataclass
class RemoteUploadBatch:
    uploads: List[InMemoryUpload]
    url_map: Dict[str, str]


@dataclass
class RemoteAsset:
    filename: str
    content: bytes
    content_type: str
    source_url: str


class RemoteImageService:
    @classmethod
    async def fetch_event_uploads(
        cls,
        event_id: str,
        image_urls: Iterable[str],
    ) -> RemoteUploadBatch:
        assets = await cls.fetch_assets(image_urls)
        uploads: List[InMemoryUpload] = []
        url_map: Dict[str, str] = {}

        for asset in assets:
            uploads.append(
                InMemoryUpload(filename=asset.filename, file=io.BytesIO(asset.content))
            )
            local_path = str(images_dir(event_id) / asset.filename)
            url_map[local_path] = asset.source_url

        return RemoteUploadBatch(uploads=uploads, url_map=url_map)

    @classmethod
    async def fetch_single_asset(cls, image_url: str) -> RemoteAsset:
        assets = await cls.fetch_assets([image_url])
        return assets[0]

    @classmethod
    async def fetch_assets(cls, image_urls: Iterable[str]) -> List[RemoteAsset]:
        urls = [str(url) for url in image_urls]
        if not urls:
            raise ValueError("At least one image URL is required.")

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
        ) as client:
            tasks = [cls._fetch_one(client, url, index) for index, url in enumerate(urls)]
            return await asyncio.gather(*tasks)

    @staticmethod
    async def _fetch_one(
        client: httpx.AsyncClient,
        url: str,
        index: int,
    ) -> RemoteAsset:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"URL does not point to an image: {url}")

        filename = RemoteImageService._filename_from_url(url, content_type, index)
        return RemoteAsset(
            filename=filename,
            content=response.content,
            content_type=content_type or "image/jpeg",
            source_url=url,
        )

    @staticmethod
    def _filename_from_url(url: str, content_type: str, index: int) -> str:
        parsed = urlparse(url)
        source_name = Path(unquote(parsed.path)).name
        stem = Path(source_name).stem if source_name else f"asset-{index + 1}"
        ext = Path(source_name).suffix.lower()
        if not ext:
            ext = RemoteImageService._extension_from_content_type(content_type)

        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in stem)
        safe_stem = "-".join(part for part in safe_stem.split("-") if part) or f"asset-{index + 1}"
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        return f"{safe_stem}-{url_hash}{ext}"

    @staticmethod
    def _extension_from_content_type(content_type: str) -> str:
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return mapping.get(content_type, ".jpg")

    @staticmethod
    def apply_url_map(
        manifest: Dict[str, object],
        url_map: Dict[str, str],
    ) -> Dict[str, object]:
        updated = deepcopy(manifest)
        clusters = updated.get("clusters", {})
        for cluster in clusters.values():
            image_paths = cluster.get("image_paths", [])
            cluster["image_paths"] = [url_map.get(path, path) for path in image_paths]
        return updated

    @staticmethod
    def rewrite_image_list(image_paths: List[str], url_map: Dict[str, str]) -> List[str]:
        return [url_map.get(path, path) for path in image_paths]
