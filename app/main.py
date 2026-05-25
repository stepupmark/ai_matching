"""
FastAPI application entry point for the AI Job Matching system.

Wires all services together at startup and stores them on app.state
for clean dependency injection.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.services.embedding import EmbeddingService
from app.services.matcher import MatcherService
from app.services.metadata_store import MetadataStore
from app.services.openai_parser import OpenAIParser
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _init_services(app: FastAPI) -> None:
    """Initialise all services and attach them to app.state."""
    logger.info("Initialising Candidate Scorecard Matching services ...")

    embedding_service = EmbeddingService()
    _ = embedding_service.model  # eagerly load model
    logger.info("Embedding model '%s' loaded.", embedding_service.model_name)

    vector_store = VectorStore()
    metadata_store = MetadataStore()
    parser = OpenAIParser()
    reranker = RerankerService()

    matcher_service = MatcherService(
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_service=embedding_service,
        parser=parser,
        reranker=reranker,
    )

    # Attach to app.state for clean DI
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.metadata_store = metadata_store
    app.state.parser = parser
    app.state.reranker = reranker
    app.state.matcher_service = matcher_service

    logger.info(
        "Services initialised — %d candidates, %d JDs in store.",
        vector_store.candidate_count,
        vector_store.jd_count,
    )


# ── Lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup / shutdown."""
    _init_services(app)
    yield
    # Shutdown — persist state
    vs: VectorStore | None = getattr(app.state, "vector_store", None)
    if vs is not None:
        vs.save()
    ms: MetadataStore | None = getattr(app.state, "metadata_store", None)
    if ms is not None:
        ms.close()
    logger.info("Shutdown complete.")


# ── FastAPI application ────────────────────────────────────────────

app = FastAPI(
    title="AI Job Matching API",
    description="Production-level AI-powered job matching using vector retrieval.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
