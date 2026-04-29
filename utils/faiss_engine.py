"""
faiss_engine.py
───────────────
Centralised FAISS index management with automatic GPU / CPU selection.

Device priority (when FAISS_DEVICE = "auto"):
  1. faiss-gpu   → CUDA GPU (fastest — 5–50x over CPU for large indexes)
  2. faiss-cpu   → CPU (always available, very fast for < 500 clusters)
  3. NumPy        → pure Python fallback (no faiss installed)

Index type selection (automatic):
  • n <  FAISS_IVF_THRESHOLD → IndexFlatIP       (exact, zero error)
  • n >= FAISS_IVF_THRESHOLD → IndexIVFFlat      (approximate, ~99% recall)

GPU notes
─────────
• faiss-gpu keeps the index in GPU VRAM — searches are batched on the GPU.
• For very large events (10,000+ faces) the GPU speedup is dramatic.
• The index is always SAVED to disk in CPU format (faiss.index_gpu_to_cpu)
  and LOADED back to GPU on the next search. This is the standard pattern.
• If GPU memory is exhausted, we automatically fall back to CPU index.

Install
───────
    pip install faiss-gpu        # CUDA 11/12 wheels on PyPI
    # OR build from source for newer CUDA:
    # conda install -c pytorch faiss-gpu cudatoolkit=11.8
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Thread-safe GPU resource singleton ───────────────────────────────────────
_gpu_res = None
_gpu_lock = threading.Lock()


def _get_gpu_resource():
    """
    Return a (cached) faiss.StandardGpuResources object.
    Creates one per process — creating per-search would be very slow.
    """
    global _gpu_res
    if _gpu_res is None:
        with _gpu_lock:
            if _gpu_res is None:
                try:
                    import faiss
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(256 * 1024 * 1024)  # 256 MB temp pool
                    _gpu_res = res
                    logger.info(
                        "FAISS GPU resource initialised (GPU %d).",
                        settings.FAISS_GPU_ID,
                    )
                except Exception as exc:
                    logger.warning("Could not init FAISS GPU resource: %s", exc)
    return _gpu_res


# ── Device detection ──────────────────────────────────────────────────────────

def _detect_device() -> str:
    """
    Resolve the effective device based on FAISS_DEVICE setting and
    what is actually installed/available.

    Returns "gpu" or "cpu".
    """
    requested = settings.FAISS_DEVICE.lower().strip()

    if requested == "cpu":
        logger.debug("FAISS device: CPU (forced by config).")
        return "cpu"

    # Try GPU
    try:
        import faiss
        if not hasattr(faiss, "StandardGpuResources"):
            raise ImportError("faiss-gpu not installed (no StandardGpuResources).")

        # Check CUDA is actually available by trying to init resources
        res = _get_gpu_resource()
        if res is None:
            raise RuntimeError("GPU resource init returned None.")

        # Verify the requested GPU index exists
        n_gpus = faiss.get_num_gpus()
        if n_gpus == 0:
            raise RuntimeError("faiss.get_num_gpus() == 0 — no CUDA GPUs found.")
        if settings.FAISS_GPU_ID >= n_gpus:
            logger.warning(
                "FAISS_GPU_ID=%d but only %d GPU(s) found. Using GPU 0.",
                settings.FAISS_GPU_ID, n_gpus,
            )

        if requested == "gpu":
            logger.info("FAISS device: GPU %d (forced by config).", settings.FAISS_GPU_ID)
        else:
            logger.info(
                "FAISS device: GPU %d (auto-detected, %d GPU(s) available).",
                settings.FAISS_GPU_ID, n_gpus,
            )
        return "gpu"

    except ImportError as exc:
        if requested == "gpu":
            raise RuntimeError(
                f"FAISS_DEVICE=gpu but faiss-gpu is not installed: {exc}\n"
                "Install it with:  pip install faiss-gpu"
            )
        logger.info("faiss-gpu not available (%s). Using CPU.", exc)
        return "cpu"

    except Exception as exc:
        if requested == "gpu":
            raise RuntimeError(f"FAISS GPU init failed: {exc}")
        logger.warning(
            "GPU unavailable (%s). Falling back to CPU FAISS.", exc
        )
        return "cpu"


# Cache the resolved device so we only log once per process
_resolved_device: Optional[str] = None
_device_lock = threading.Lock()


def get_device() -> str:
    """Return the resolved device ('gpu' or 'cpu'), cached after first call."""
    global _resolved_device
    if _resolved_device is None:
        with _device_lock:
            if _resolved_device is None:
                _resolved_device = _detect_device()
    return _resolved_device


def faiss_available() -> bool:
    """Return True when faiss (cpu or gpu) can be imported."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


# ── Index construction ────────────────────────────────────────────────────────

def build_index(vectors: np.ndarray):
    """
    Build the most appropriate FAISS index for *vectors*.

    Selection logic
    ───────────────
    n < IVF_THRESHOLD  → FlatIP   (exact, always correct)
    n ≥ IVF_THRESHOLD  → IVFFlat  (approximate, ~99% recall, much faster)

    The index is built on CPU first, then moved to GPU if device == "gpu".
    This is the correct FAISS pattern — GPU indexes are not serialisable
    directly, so they always wrap a CPU index.
    """
    if not faiss_available():
        return None

    import faiss

    n = len(vectors)
    vecs = vectors.astype(np.float32)
    device = get_device()

    try:
        cpu_index = _build_cpu_index(vecs, n)

        if device == "gpu":
            gpu_index = _move_to_gpu(cpu_index)
            if gpu_index is not None:
                logger.info(
                    "FAISS index on GPU %d (%d vectors, type=%s).",
                    settings.FAISS_GPU_ID, n, type(cpu_index).__name__,
                )
                return gpu_index
            else:
                logger.warning("GPU move failed — keeping CPU index.")

        logger.info(
            "FAISS index on CPU (%d vectors, type=%s).",
            n, type(cpu_index).__name__,
        )
        return cpu_index

    except Exception as exc:
        logger.error("FAISS index build failed: %s", exc)
        return None


def _build_cpu_index(vecs: np.ndarray, n: int):
    """Build a CPU FAISS index (FlatIP or IVFFlat)."""
    import faiss

    dim = vecs.shape[1]

    if n >= settings.FAISS_IVF_THRESHOLD:
        nlist = min(settings.FAISS_NLIST, max(1, n // 4))
        logger.debug(
            "Building IVFFlat (n=%d ≥ threshold=%d, nlist=%d).",
            n, settings.FAISS_IVF_THRESHOLD, nlist,
        )
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
        index.add(vecs)
        index.nprobe = min(settings.FAISS_NPROBE, nlist)
    else:
        logger.debug("Building FlatIP (n=%d).", n)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

    return index


def _move_to_gpu(cpu_index):
    """
    Transfer a CPU index to the configured GPU.
    Returns the GPU index, or None if the transfer fails.
    """
    try:
        import faiss
        res = _get_gpu_resource()
        if res is None:
            return None
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False          # float32 for accuracy
        gpu_index = faiss.index_cpu_to_gpu(
            res, settings.FAISS_GPU_ID, cpu_index, co
        )
        return gpu_index
    except Exception as exc:
        logger.warning("index_cpu_to_gpu failed: %s", exc)
        return None


# ── Index persistence ─────────────────────────────────────────────────────────

def save_index(index, path) -> bool:
    """
    Save a FAISS index to disk.
    GPU indexes are first converted back to CPU before writing
    (FAISS cannot serialise GPU indexes directly).

    Returns True on success.
    """
    if index is None:
        return False
    try:
        import faiss
        cpu_index = index
        # Convert GPU → CPU if needed
        if hasattr(faiss, "GpuIndex") and isinstance(index, faiss.GpuIndex):
            cpu_index = faiss.index_gpu_to_cpu(index)
        elif hasattr(faiss, "GpuIndexFlat") and isinstance(index, faiss.GpuIndexFlat):
            cpu_index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(cpu_index, str(path))
        logger.debug("Saved FAISS index → %s", path)
        return True
    except Exception as exc:
        logger.warning("Could not save FAISS index: %s", exc)
        return False


def load_index(path):
    """
    Load a FAISS index from disk.
    If device == "gpu", automatically moves it to GPU after loading.
    Returns None if the file does not exist or load fails.
    """
    try:
        import faiss
        if not path.exists():
            return None
        cpu_index = faiss.read_index(str(path))
        logger.debug(
            "Loaded CPU FAISS index from %s (%d vectors).", path, cpu_index.ntotal
        )

        device = get_device()
        if device == "gpu":
            gpu_index = _move_to_gpu(cpu_index)
            if gpu_index is not None:
                logger.debug("Moved loaded index to GPU %d.", settings.FAISS_GPU_ID)
                return gpu_index
            logger.warning("GPU transfer failed after load — using CPU index.")

        return cpu_index

    except Exception as exc:
        logger.warning("Could not load FAISS index from %s: %s", path, exc)
        return None


# ── Search ────────────────────────────────────────────────────────────────────

def search_index(index, query: np.ndarray, k: int) -> Tuple[list, list]:
    """
    Search *index* for the top-k nearest neighbours of *query*.

    Returns (indices, scores) — both plain Python lists.
    Restores IVF nprobe after loading if needed.
    """
    try:
        # Restore nprobe for IVF indexes (lost after GPU transfer)
        if hasattr(index, "nprobe"):
            try:
                index.nprobe = min(settings.FAISS_NPROBE, index.nlist)
            except Exception:
                pass

        q = query.reshape(1, -1).astype(np.float32)
        actual_k = min(k, index.ntotal)
        scores, indices = index.search(q, actual_k)

        valid = [
            (int(i), float(s))
            for i, s in zip(indices[0], scores[0])
            if i >= 0
        ]
        if not valid:
            return [], []
        idxs, scrs = zip(*valid)
        return list(idxs), list(scrs)

    except Exception as exc:
        logger.error("FAISS search failed: %s", exc)
        return [], []


# ── Diagnostics ───────────────────────────────────────────────────────────────

def get_faiss_info() -> dict:
    """
    Return a diagnostic dict about the current FAISS setup.
    Useful for the /health or /info endpoint.
    """
    info: dict = {
        "faiss_installed": faiss_available(),
        "requested_device": settings.FAISS_DEVICE,
        "resolved_device": _resolved_device or "not yet resolved",
        "gpu_id": settings.FAISS_GPU_ID,
        "use_faiss": settings.USE_FAISS,
    }

    if faiss_available():
        try:
            import faiss
            info["faiss_version"] = faiss.__version__ if hasattr(faiss, "__version__") else "unknown"
            info["num_gpus"] = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
            info["gpu_support"] = hasattr(faiss, "StandardGpuResources")
        except Exception:
            pass

    return info