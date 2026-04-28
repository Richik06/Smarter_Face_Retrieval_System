"""
GoogleDriveService
──────────────────
Downloads images from a publicly shared Google Drive folder link.

Supports two URL formats:
  1. Folder link : https://drive.google.com/drive/folders/<FOLDER_ID>
  2. File link   : https://drive.google.com/file/d/<FILE_ID>/view

How it works (no OAuth needed for public folders)
──────────────────────────────────────────────────
1. Parse the folder ID from the URL
2. Use Google Drive API (v3) with an API key  OR
   fall back to the gdown library which handles public folders
3. Download all image files into a local temp directory
4. Return list of local file paths to be processed normally

Setup
─────
Option A — gdown (simplest, no API key needed):
    pip install gdown

Option B — Google Drive API key (more reliable for large folders):
    1. Go to https://console.cloud.google.com
    2. Create a project → Enable "Google Drive API"
    3. Create an API key (Credentials → API key)
    4. Set GOOGLE_DRIVE_API_KEY=<your_key> in .env

The service tries gdown first, then falls back to the Drive API if a key is set.
"""

from __future__ import annotations

import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Google Drive API base URL
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


class GoogleDriveService:
    """Downloads images from a public Google Drive folder or file link."""

    # ── Public API

    @classmethod
    def download_images_from_link(
        cls,
        drive_link: str,
        dest_dir: Path,
    ) -> Tuple[List[Path], List[str]]:
        """
        Download all images from a Google Drive link to *dest_dir*.

        Parameters
        ----------
        drive_link : str
            Public Google Drive folder or file URL.
        dest_dir   : Path
            Local directory where images will be saved.

        Returns
        -------
        (downloaded_paths, errors)
          downloaded_paths : list of local Path objects for each image
          errors           : list of error messages for failed downloads
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        link_type, resource_id = cls._parse_drive_link(drive_link)
        if resource_id is None:
            raise ValueError(
                f"Could not parse Google Drive link: '{drive_link}'. "
                "Expected format: https://drive.google.com/drive/folders/<ID> "
                "or https://drive.google.com/file/d/<ID>/view"
            )

        logger.info(
            "Google Drive link parsed → type=%s  id=%s", link_type, resource_id
        )

        if link_type == "file":
            return cls._download_single_file(resource_id, dest_dir)
        else:
            return cls._download_folder(resource_id, dest_dir)

    # ── Link parsing 

    @staticmethod
    def _parse_drive_link(url: str) -> Tuple[str, Optional[str]]:
        """
        Extract (type, id) from a Google Drive URL.
        type is "folder" or "file".
        Returns (type, None) if the link cannot be parsed.
        """
        # Folder: .../drive/folders/<ID>
        folder_match = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", url)
        if folder_match:
            return "folder", folder_match.group(1)

        # File: .../file/d/<ID>/
        file_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
        if file_match:
            return "file", file_match.group(1)

        # Fallback: id= query param
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if "id" in params:
            return "file", params["id"][0]

        return "unknown", None

    # ── Folder download (gdown)

    @classmethod
    def _download_folder(
        cls, folder_id: str, dest_dir: Path
    ) -> Tuple[List[Path], List[str]]:
        """
        Download all images from a public Google Drive folder.
        Tries gdown first, then falls back to Drive API if key is available.
        """
        # Try gdown
        try:
            return cls._download_folder_gdown(folder_id, dest_dir)
        except ImportError:
            logger.warning("gdown not installed. Trying Drive API fallback.")
        except Exception as exc:
            logger.warning("gdown folder download failed: %s. Trying API.", exc)

        # Try Drive API
        if settings.GOOGLE_DRIVE_API_KEY:
            return cls._download_folder_api(folder_id, dest_dir)

        raise RuntimeError(
            "Could not download Google Drive folder. "
            "Install gdown (`pip install gdown`) or set GOOGLE_DRIVE_API_KEY in .env."
        )

    @staticmethod
    def _download_folder_gdown(
        folder_id: str, dest_dir: Path
    ) -> Tuple[List[Path], List[str]]:
        """Use gdown to download an entire public folder."""
        import gdown  # type: ignore

        url = f"https://drive.google.com/drive/folders/{folder_id}"
        logger.info("Downloading folder via gdown: %s → %s", url, dest_dir)

        # gdown.download_folder downloads into a subfolder inside dest_dir
        downloaded = gdown.download_folder(
            url=url,
            output=str(dest_dir),
            quiet=False,
            use_cookies=False,
        )

        if not downloaded:
            return [], ["gdown returned no files — folder may be private or empty."]

        # Filter to image files only
        image_paths: List[Path] = []
        errors: List[str] = []

        for item in downloaded:
            p = Path(item)
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(p)
            else:
                logger.debug("Skipping non-image file: %s", p.name)

        logger.info(
            "gdown downloaded %d image(s) from folder %s.", len(image_paths), folder_id
        )
        return image_paths, errors

    @classmethod
    def _download_folder_api(
        cls, folder_id: str, dest_dir: Path
    ) -> Tuple[List[Path], List[str]]:
        """
        Use Google Drive API v3 to list and download all image files in a folder.
        Requires GOOGLE_DRIVE_API_KEY to be set.
        """
        api_key = settings.GOOGLE_DRIVE_API_KEY
        image_paths: List[Path] = []
        errors: List[str] = []
        page_token: Optional[str] = None

        logger.info("Listing Drive folder %s via API.", folder_id)

        while True:
            params = {
                "key": api_key,
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": "nextPageToken, files(id, name, mimeType)",
                "pageSize": 100,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = requests.get(f"{DRIVE_API_BASE}/files", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for file_info in data.get("files", []):
                mime = file_info.get("mimeType", "")
                name = file_info.get("name", "unknown")
                file_id = file_info["id"]

                if not mime.startswith("image/"):
                    logger.debug("Skipping non-image: %s (%s)", name, mime)
                    continue

                try:
                    path = cls._download_file_api(file_id, name, dest_dir, api_key)
                    image_paths.append(path)
                except Exception as exc:
                    msg = f"Failed to download '{name}': {exc}"
                    logger.warning(msg)
                    errors.append(msg)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(
            "Drive API downloaded %d image(s) from folder %s.", len(image_paths), folder_id
        )
        return image_paths, errors

    # ── Single file download 

    @classmethod
    def _download_single_file(
        cls, file_id: str, dest_dir: Path
    ) -> Tuple[List[Path], List[str]]:
        """Download a single file from Google Drive."""
        # Try gdown first
        try:
            import gdown  # type: ignore
            url = f"https://drive.google.com/uc?id={file_id}"
            out_path = dest_dir / f"{file_id}.jpg"
            result = gdown.download(url=url, output=str(out_path), quiet=False)
            if result:
                p = Path(result)
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    return [p], []
                return [], [f"Downloaded file '{p.name}' is not a supported image type."]
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("gdown single file failed: %s", exc)

        # Try direct download
        if settings.GOOGLE_DRIVE_API_KEY:
            try:
                path = cls._download_file_api(
                    file_id, f"{file_id}.jpg", dest_dir, settings.GOOGLE_DRIVE_API_KEY
                )
                return [path], []
            except Exception as exc:
                return [], [str(exc)]

        # Last resort: direct export URL (works for some public files)
        try:
            path = cls._download_file_direct(file_id, dest_dir)
            return [path], []
        except Exception as exc:
            return [], [f"Could not download file {file_id}: {exc}"]

    @staticmethod
    def _download_file_api(
        file_id: str, filename: str, dest_dir: Path, api_key: str
    ) -> Path:
        """Download a file using the Drive API."""
        url = f"{DRIVE_API_BASE}/files/{file_id}"
        params = {"key": api_key, "alt": "media"}
        resp = requests.get(url, params=params, stream=True, timeout=60)
        resp.raise_for_status()

        # Use content-type to fix extension if needed
        dest = dest_dir / filename
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug("Downloaded: %s", dest)
        return dest

    @staticmethod
    def _download_file_direct(file_id: str, dest_dir: Path) -> Path:
        """Direct download for public files (no API key)."""
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        resp = session.get(url, stream=True, timeout=60)

        # Handle large-file confirmation page
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                url = f"{url}&confirm={value}"
                resp = session.get(url, stream=True, timeout=60)
                break

        resp.raise_for_status()
        dest = dest_dir / f"{file_id}.jpg"
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest

    # ── Utilities

    @staticmethod
    def is_drive_link(url: str) -> bool:
        """Return True when *url* looks like a Google Drive link."""
        return "drive.google.com" in url or "docs.google.com" in url