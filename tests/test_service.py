"""
Test suite for the Face Retrieval Microservice.

Run with:
    pytest tests/ -v

Requires pytest and httpx:
    pip install pytest httpx pytest-asyncio
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ── Make project root importable 
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


# Fixtures


@pytest.fixture(scope="session")
def tmp_event_dir(tmp_path_factory):
    """Temporary directory used as EVENT_DATA_DIR for all tests."""
    return tmp_path_factory.mktemp("event_data")


@pytest.fixture(scope="session", autouse=True)
def patch_settings(tmp_event_dir):
    """Override settings to use a temp directory and fast/cheap models."""
    with patch("config.settings") as mock_settings:
        mock_settings.EVENT_DATA_DIR = tmp_event_dir
        mock_settings.DETECTOR_BACKEND = "opencv"
        mock_settings.EMBEDDING_MODEL = "Facenet512"
        mock_settings.EMBEDDING_DIM = 512
        mock_settings.DBSCAN_EPS = 0.35
        mock_settings.DBSCAN_MIN_SAMPLES = 1
        mock_settings.DBSCAN_METRIC = "cosine"
        mock_settings.SIMILARITY_THRESHOLD = 0.50
        mock_settings.USE_FAISS = False
        mock_settings.CLOUDINARY_CLOUD_NAME = "demo-cloud"
        mock_settings.CLOUDINARY_API_KEY = "1234567890"
        mock_settings.CLOUDINARY_API_SECRET = "abcdefg"
        mock_settings.CLOUDINARY_UPLOAD_FOLDER = "face-retrieval"
        mock_settings.LOG_LEVEL = "DEBUG"
        yield mock_settings


@pytest.fixture(scope="session")
def client(tmp_event_dir, patch_settings):
    """FastAPI test client with mocked heavy dependencies."""
    # Mock DeepFace to avoid loading real model weights during tests
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_embedding /= np.linalg.norm(fake_embedding)

    mock_repr = [
        {
            "embedding": fake_embedding.tolist(),
            "facial_area": {"x": 10, "y": 10, "w": 100, "h": 100},
        }
    ]

    with patch("services.embedding_service._get_deepface") as mock_df_getter:
        mock_df = MagicMock()
        mock_df.represent.return_value = mock_repr
        mock_df_getter.return_value = mock_df

        from main import app
        with TestClient(app) as c:
            yield c


def _make_fake_image_bytes() -> bytes:
    """Create a minimal valid JPEG-ish byte sequence using OpenCV."""
    import cv2
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Draw a rough face-like circle so RetinaFace has something
    cv2.circle(img, (100, 100), 60, (200, 180, 160), -1)
    cv2.circle(img, (80, 85), 10, (50, 50, 50), -1)   # left eye
    cv2.circle(img, (120, 85), 10, (50, 50, 50), -1)  # right eye
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()



# Health


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"



# Embedding service unit tests


class TestEmbeddingService:
    def test_l2_normalise(self):
        from services.embedding_service import _l2_normalize
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = _l2_normalize(v)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_zero_vector_safe(self):
        from services.embedding_service import _l2_normalize
        v = np.zeros(512, dtype=np.float32)
        normed = _l2_normalize(v)
        assert np.all(normed == 0)

    def test_is_dummy_detection_full_frame(self):
        from services.embedding_service import EmbeddingService
        # An area that covers 100 % of the image → dummy
        assert EmbeddingService._is_dummy_detection(
            {"x": 0, "y": 0, "w": 200, "h": 200}, (200, 200, 3)
        )

    def test_is_dummy_detection_real_face(self):
        from services.embedding_service import EmbeddingService
        assert not EmbeddingService._is_dummy_detection(
            {"x": 10, "y": 10, "w": 80, "h": 80}, (200, 200, 3)
        )



# Clustering service unit tests


class TestClusteringService:
    def _make_embeddings(self, n=10, dim=512):
        e = np.random.rand(n, dim).astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / norms

    def test_cluster_returns_correct_length(self):
        from services.clustering_service import ClusteringService
        embs = self._make_embeddings(5)
        labels = ClusteringService.cluster_embeddings(embs)
        assert len(labels) == 5

    def test_empty_embeddings(self):
        from services.clustering_service import ClusteringService
        labels = ClusteringService.cluster_embeddings(np.array([]))
        assert len(labels) == 0

    def test_build_manifest_structure(self):
        from services.clustering_service import ClusteringService
        embs = self._make_embeddings(4)
        labels = np.array([0, 0, 1, 1])
        meta = [
            {"image_path": f"img_{i}.jpg", "face_index": 0, "cluster_id": None}
            for i in range(4)
        ]
        manifest = ClusteringService.build_cluster_manifest("ev1", embs, labels, meta)
        assert manifest["num_clusters"] == 2
        assert "0" in manifest["clusters"]
        assert "1" in manifest["clusters"]

    def test_noise_points_become_singletons(self):
        from services.clustering_service import ClusteringService
        embs = self._make_embeddings(3)
        labels = np.array([-1, -1, -1])
        meta = [
            {"image_path": f"img_{i}.jpg", "face_index": 0, "cluster_id": None}
            for i in range(3)
        ]
        manifest = ClusteringService.build_cluster_manifest("ev2", embs, labels, meta)
        # Each noise point → own cluster → 3 clusters total
        assert manifest["num_clusters"] == 3

    def test_centroid_is_unit_vector(self):
        from services.clustering_service import ClusteringService
        embs = self._make_embeddings(6)
        labels = np.array([0, 0, 0, 1, 1, 1])
        meta = [
            {"image_path": f"img_{i}.jpg", "face_index": 0, "cluster_id": None}
            for i in range(6)
        ]
        manifest = ClusteringService.build_cluster_manifest("ev3", embs, labels, meta)
        for cluster in manifest["clusters"].values():
            centroid = np.array(cluster["centroid"])
            assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5



# Search service unit tests

class TestSearchService:
    def _make_unit(self, dim=512):
        v = np.random.rand(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_numpy_search_finds_closest(self):
        from services.search_service import SearchService
        query = self._make_unit()
        centroids = np.vstack([self._make_unit() for _ in range(5)])
        # Replace one centroid with the query itself → should be the winner
        centroids[2] = query
        idx, score = SearchService._numpy_search(query, centroids)
        assert idx == 2
        assert abs(score - 1.0) < 1e-5

    def test_no_match_below_threshold(self, tmp_event_dir, patch_settings):
        """Similarity below threshold → no match returned."""
        from services.search_service import SearchService
        from utils.storage import save_clusters, ensure_event_dirs

        event_id = "test_search_below_thresh"
        ensure_event_dirs(event_id)

        # Centroid pointing in a completely different direction
        centroid = self._make_unit()
        query = -centroid    # opposite direction → similarity ≈ -1

        manifest = {
            "event_id": event_id,
            "num_clusters": 1,
            "clusters": {
                "0": {
                    "cluster_id": 0,
                    "centroid": centroid.tolist(),
                    "num_faces": 1,
                    "num_images": 1,
                    "image_paths": ["img_a.jpg"],
                }
            },
        }
        save_clusters(event_id, manifest)

        result = SearchService.search(query, event_id)
        assert result["matched_cluster_id"] is None



# Storage utility tests

class TestStorage:
    def test_save_load_embeddings(self, tmp_event_dir):
        from utils.storage import save_embeddings, load_embeddings
        embs = np.random.rand(10, 512).astype(np.float32)
        save_embeddings("store_test", embs)
        loaded = load_embeddings("store_test")
        assert loaded is not None
        np.testing.assert_array_almost_equal(embs, loaded)

    def test_save_load_meta(self, tmp_event_dir):
        from utils.storage import save_meta, load_meta
        meta = [{"image_path": "a.jpg", "face_index": 0, "cluster_id": 1}]
        save_meta("meta_test", meta)
        loaded = load_meta("meta_test")
        assert loaded == meta

    def test_missing_embeddings_returns_none(self):
        from utils.storage import load_embeddings
        assert load_embeddings("nonexistent_event_xyz") is None

    def test_missing_clusters_returns_none(self):
        from utils.storage import load_clusters
        assert load_clusters("nonexistent_event_xyz") is None

    def test_save_load_asset_urls(self, tmp_event_dir):
        from utils.storage import load_asset_urls, save_asset_urls

        asset_urls = {"event_data/demo/images/a.jpg": "https://example.com/a.jpg"}
        save_asset_urls("asset_url_test", asset_urls)
        loaded = load_asset_urls("asset_url_test")
        assert loaded == asset_urls



# API integration tests


class TestGetEmbeddingEndpoint:
    def test_returns_embedding(self, client):
        img_bytes = _make_fake_image_bytes()
        r = client.post(
            "/get-embedding",
            files={"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "embedding" in body
        assert len(body["embedding"]) == 512
        assert body["embedding_dim"] == 512

    def test_invalid_file_returns_400(self, client):
        r = client.post(
            "/get-embedding",
            files={"image": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert r.status_code in (400, 422, 500)


class TestProcessEventEndpoint:
    EVENT_ID = "integration_test_event"

    def test_process_event_success(self, client):
        img_bytes = _make_fake_image_bytes()
        r = client.post(
            "/process-event",
            data={"event_id": self.EVENT_ID},
            files=[
                ("images", ("img1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
                ("images", ("img2.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ],
        )
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == self.EVENT_ID
        assert "num_people_detected" in body
        assert "clusters" in body

    def test_no_images_returns_400(self, client):
        r = client.post(
            "/process-event",
            data={"event_id": "empty_event"},
        )
        assert r.status_code in (400, 422)

    def test_get_event_manifest(self, client):
        r = client.get(f"/event/{self.EVENT_ID}")
        assert r.status_code == 200
        body = r.json()
        assert "clusters" in body

    def test_get_nonexistent_event_returns_404(self, client):
        r = client.get("/event/totally_unknown_event_99999")
        assert r.status_code == 404


class TestSearchFaceEndpoint:
    EVENT_ID = "search_integration_event"

    def _setup_event(self, client):
        img_bytes = _make_fake_image_bytes()
        client.post(
            "/process-event",
            data={"event_id": self.EVENT_ID},
            files=[("images", ("img.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
        )

    def test_search_returns_result(self, client):
        self._setup_event(client)
        img_bytes = _make_fake_image_bytes()
        r = client.post(
            "/search-face",
            data={"event_id": self.EVENT_ID},
            files={"image": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "matched_cluster_id" in body
        assert "matched_images" in body
        assert "similarity" in body

    def test_search_unknown_event_returns_404(self, client):
        img_bytes = _make_fake_image_bytes()
        r = client.post(
            "/search-face",
            data={"event_id": "ghost_event_00000"},
            files={"image": ("query.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert r.status_code == 404


class TestAppIntegrationEndpoints:
    def test_cloudinary_signature(self, client):
        response = client.post(
            "/app/cloudinary/signature",
            json={"event_id": "launch-night", "asset_type": "events"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["cloud_name"] == "demo-cloud"
        assert body["api_key"] == "1234567890"
        assert body["folder"].endswith("/events/launch-night")
        assert body["signature"]

    def test_process_event_urls(self, client):
        from services.remote_image_service import RemoteUploadBatch

        fake_batch = RemoteUploadBatch(uploads=[], url_map={})
        fake_result = {
            "event_id": "remote_event",
            "num_people_detected": 1,
            "clusters": [],
        }

        with patch(
            "api.routes.integration.RemoteImageService.fetch_event_uploads",
            new=AsyncMock(return_value=fake_batch),
        ), patch(
            "api.routes.integration.EventService.process_event",
            return_value=fake_result,
        ):
            response = client.post(
                "/app/process-event-urls",
                json={
                    "event_id": "remote_event",
                    "image_urls": ["https://example.com/photo-one.jpg"],
                },
            )

        assert response.status_code == 200
        assert response.json()["event_id"] == "remote_event"

    def test_search_face_url(self, client):
        from services.remote_image_service import RemoteAsset
        from utils.storage import ensure_event_dirs, save_clusters

        event_id = "remote_search_event"
        ensure_event_dirs(event_id)
        save_clusters(
            event_id,
            {
                "event_id": event_id,
                "num_clusters": 1,
                "clusters": {
                    "0": {
                        "cluster_id": 0,
                        "centroid": (np.ones(512) / np.sqrt(512)).tolist(),
                        "num_faces": 1,
                        "num_images": 1,
                        "image_paths": ["https://example.com/match.jpg"],
                    }
                },
            },
        )

        fake_asset = RemoteAsset(
            filename="query.jpg",
            content=_make_fake_image_bytes(),
            content_type="image/jpeg",
            source_url="https://example.com/query.jpg",
        )

        with patch(
            "api.routes.integration.RemoteImageService.fetch_single_asset",
            new=AsyncMock(return_value=fake_asset),
        ), patch(
            "api.routes.integration.EmbeddingService.get_single_embedding",
            return_value=np.ones(512, dtype=np.float32) / np.sqrt(512),
        ), patch(
            "api.routes.integration.SearchService.search",
            return_value={
                "matched_cluster_id": 0,
                "similarity": 0.92,
                "matched_images": ["https://example.com/match.jpg"],
                "message": "Match found.",
            },
        ):
            response = client.post(
                "/app/search-face-url",
                json={
                    "event_id": event_id,
                    "image_url": "https://example.com/query.jpg",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["matched_cluster_id"] == 0
        assert body["matched_images"] == ["https://example.com/match.jpg"]



# Duplicate detection


class TestDuplicateDetection:
    def test_identical_embedding_flagged(self):
        from services.event_service import EventService
        v = np.random.rand(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert EventService._is_duplicate(v, [v])

    def test_different_embedding_not_flagged(self):
        from services.event_service import EventService
        v1 = np.array([1.0] + [0.0] * 511, dtype=np.float32)
        v2 = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
        assert not EventService._is_duplicate(v2, [v1])

    def test_empty_pool_never_duplicate(self):
        from services.event_service import EventService
        v = np.random.rand(512).astype(np.float32)
        assert not EventService._is_duplicate(v, [])
