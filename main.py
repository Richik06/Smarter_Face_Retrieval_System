"""
Face-Based Image Retrieval Microservice
Entry point for the FastAPI application.
"""

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import embedding, events, search
from utils.logger import setup_logger

#Logging
logger = setup_logger(__name__)


#  Lifespan 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info("Starting Face Retrieval Microservice …")
    # Pre-warm the DeepFace model so the first real request isn't slow.
    try:
        from services.embedding_service import EmbeddingService
        EmbeddingService.warm_up()
        logger.info("Embedding model warmed up successfully.")
    except Exception as exc:
        logger.warning("Model warm-up skipped: %s", exc)
    yield
    logger.info("Shutting down Face Retrieval Microservice.")


#Application 
app = FastAPI(
    title="Face Retrieval Microservice",
    description=(
        "AI microservice for face-based image retrieval using clustering. "
        "Detects faces, generates embeddings, clusters them per event, "
        "and returns all images belonging to a queried person."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


#Routers
app.include_router(events.router, tags=["Events"])
app.include_router(embedding.router, tags=["Embedding"])
app.include_router(search.router, tags=["Search"])


# ── Health check
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "face-retrieval-microservice"}

@app.get("/")
async def root():
    return {
        "service": "Face Retrieval Microservice",
        "status": "running",
        "docs": "http://127.0.0.1:8000/docs"
    }


# ── CLI entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,          # keep at 1; model state is in-process
    )