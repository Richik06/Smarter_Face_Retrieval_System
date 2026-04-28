"""
Image I/O and pre-processing helpers.
"""

import base64
import io
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Decoding 

def decode_upload(upload: UploadFile) -> np.ndarray:
    """
    Read a FastAPI UploadFile and return a BGR NumPy array (OpenCV format).
    Raises ValueError if the bytes cannot be decoded as an image.
    """
    raw_bytes = upload.file.read()
    return decode_bytes(raw_bytes)


def decode_bytes(raw: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR NumPy array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes – unsupported format.")
    return img


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode a base-64 encoded image string → BGR NumPy array."""
    # Strip optional data-URI header, e.g. "data:image/jpeg;base64,..."
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return decode_bytes(raw)


# ── Saving
def save_image(img: np.ndarray, dest_dir: Path, filename: Optional[str] = None) -> Path:
    """
    Write *img* (BGR) to *dest_dir*.
    Returns the absolute path of the saved file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{uuid.uuid4().hex}.jpg"
    dest = dest_dir / filename
    cv2.imwrite(str(dest), img)
    logger.debug("Saved image → %s", dest)
    return dest


# ── Validation 

def is_valid_image(img: np.ndarray) -> bool:
    """Return True when *img* is a non-empty 3-channel array."""
    return img is not None and img.ndim == 3 and img.shape[2] == 3 and img.size > 0


# ── Misc 
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) → RGB (DeepFace / PIL convention)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)