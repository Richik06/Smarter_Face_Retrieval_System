"""
Cloudinary helpers for direct browser uploads.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict

from config import settings


def _sanitize_segment(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "untitled"


class CloudinaryService:
    @staticmethod
    def build_folder(event_id: str, asset_type: str) -> str:
        safe_event = _sanitize_segment(event_id)
        safe_type = _sanitize_segment(asset_type)
        return f"{settings.CLOUDINARY_UPLOAD_FOLDER}/{safe_type}/{safe_event}"

    @staticmethod
    def build_signature_payload(folder: str) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        params = {
            "folder": folder,
            "timestamp": timestamp,
            "unique_filename": "true",
            "use_filename": "true",
        }
        signature = CloudinaryService._sign(params)
        return {
            "api_key": settings.CLOUDINARY_API_KEY,
            "cloud_name": settings.CLOUDINARY_CLOUD_NAME,
            "folder": folder,
            "signature": signature,
            "timestamp": timestamp,
            "upload_url": (
                f"https://api.cloudinary.com/v1_1/"
                f"{settings.CLOUDINARY_CLOUD_NAME}/image/upload"
            ),
            "use_filename": "true",
            "unique_filename": "true",
        }

    @staticmethod
    def _sign(params: Dict[str, str]) -> str:
        filtered = {key: value for key, value in params.items() if value not in (None, "")}
        joined = "&".join(f"{key}={filtered[key]}" for key in sorted(filtered))
        digest = hashlib.sha1(
            f"{joined}{settings.CLOUDINARY_API_SECRET}".encode("utf-8")
        )
        return digest.hexdigest()
