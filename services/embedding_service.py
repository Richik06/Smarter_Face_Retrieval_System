"""
EmbeddingService
────────────────
Wraps DeepFace to provide:
  • Face detection   (RetinaFace by default)
  • Face embedding   (Facenet512 by default; ArcFace as alternative)

Key design decisions
────────────────────
1. We call DeepFace.represent() which internally detects + embeds in one pass.
2. enforce_detection=False means images with no face return an empty list rather
   than raising an exception – critical for batch robustness.
3. All embeddings are L2-normalised before storage so that cosine similarity
   reduces to a simple dot product.
4. A class-level singleton pattern avoids reloading heavy model weights on
   every request.
"""

from __future__ import annotations

import threading
from typing import List, Tuple

import cv2
import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# DeepFace is imported lazily to avoid slow startup when the module is loaded
# (the first actual call will trigger weight download if needed).
_deepface = None
_df_lock = threading.Lock()


def _get_deepface():
    global _deepface
    if _deepface is None:
        with _df_lock:
            if _deepface is None:
                import deepface.DeepFace as df  # noqa: N812
                _deepface = df
    return _deepface


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised version of vector *v*. Safe against zero vectors."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v


class EmbeddingService:
    """Stateless service – all methods are class-level."""

    # ── Public API

    @classmethod
    def warm_up(cls) -> None:
        """
        Force model weights to load once at startup.
        Generates a dummy embedding on a blank image.
        """
        blank = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            cls.get_embeddings_from_image(blank)
        except Exception:
            pass  # blank image may yield no face – that is fine
        logger.info(
            "Embedding model '%s' with detector '%s' loaded.",
            settings.EMBEDDING_MODEL,
            settings.DETECTOR_BACKEND,
        )

    @classmethod
    def get_embeddings_from_image(
        cls,
        img: np.ndarray,
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        Detect all faces in *img* and return their embeddings.

        Parameters
        ----------
        img : np.ndarray
            BGR image (OpenCV convention).

        Returns
        -------
        List of (embedding_vector, facial_area_dict) tuples.
        embedding_vector : np.ndarray  shape (dim,)  float32  L2-normalised
        facial_area_dict : dict  keys: x, y, w, h
        """
        df = _get_deepface()
        results = []

        try:
            # DeepFace.represent accepts BGR arrays directly.
            representations = df.represent(
                img_path=img,
                model_name=settings.EMBEDDING_MODEL,
                detector_backend=settings.DETECTOR_BACKEND,
                enforce_detection=False,   # don't crash on no-face images
                align=True,
                normalization="base",
            )
        except Exception as exc:
            logger.warning("DeepFace.represent failed: %s", exc)
            return results

        for rep in representations:
            raw_embedding = np.array(rep["embedding"], dtype=np.float32)
            facial_area = rep.get("facial_area", {})

            # Skip 'detections' that are essentially the full frame
            # (happens when no face is found and enforce_detection=False)
            if cls._is_dummy_detection(facial_area, img.shape):
                logger.debug("Skipping dummy full-frame detection.")
                continue

            embedding = _l2_normalize(raw_embedding)
            results.append((embedding, facial_area))

        logger.debug("Detected %d face(s) in image.", len(results))
        return results

    @classmethod
    def get_single_embedding(cls, img: np.ndarray) -> np.ndarray:
        """
        Return the embedding of the *largest* (most prominent) face in *img*.
        Raises ValueError if no face is detected.
        """
        detections = cls.get_embeddings_from_image(img)
        if not detections:
            raise ValueError("No face detected in the provided image.")

        if len(detections) == 1:
            return detections[0][0]

        # Pick largest bounding box (most prominent face)
        best = max(
            detections,
            key=lambda t: t[1].get("w", 0) * t[1].get("h", 0),
        )
        return best[0]

    # ── Internal helpers

    @staticmethod
    def _is_dummy_detection(facial_area: dict, img_shape: tuple) -> bool:
        """
        DeepFace returns the entire frame as the 'face' when no face is found
        and enforce_detection=False. Detect this by checking whether the
        bounding box covers ≥ 95 % of the image area.
        """
        if not facial_area:
            return True
        h_img, w_img = img_shape[:2]
        w_face = facial_area.get("w", 0)
        h_face = facial_area.get("h", 0)
        if w_img == 0 or h_img == 0:
            return False
        coverage = (w_face * h_face) / (w_img * h_img)
        return coverage > 0.95