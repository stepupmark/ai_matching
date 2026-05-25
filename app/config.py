"""
Application configuration with environment variable support.
Uses pydantic-settings for robust config management.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: str = "*"

    # --- Data Paths ---
    data_dir: str = str(Path(__file__).resolve().parent.parent / "data")
    faiss_index_dir: str = ""  # defaults to {data_dir}/indices
    metadata_db_path: str = ""  # defaults to {data_dir}/metadata.db
    skills_db_path: str = ""  # defaults to {data_dir}/skills.json

    # --- Embedding Model ---
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_model_version: str = "1.0"
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 outputs 384-d vectors
    embedding_device: str = "cpu"  # "cuda" or "cpu"
    embedding_batch_size: int = 32

    # --- Vector Store (FAISS) ---
    faiss_index_type: str = "flat"  # "flat" or "ivf" or "hnsw"
    faiss_nprobe: int = 10  # IVF probes
    faiss_hnsw_m: int = 32  # HNSW connections per node
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 64

    # --- File Upload ---
    max_file_size_mb: int = 50  # max upload file size in MB

    # --- Bulk Upload ---
    max_upload_files: int = 50  # max scorecards per bulk request

    # --- Matching ---
    default_top_k: int = 50
    max_top_k: int = 200
    rerank_default: bool = False
    rerank_limit: int = 30
    cache_ttl_seconds: int = 3600  # 1 hour

    # --- OpenAI Reranker ---
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_rerank_temperature: float = 0.1
    openai_rerank_max_tokens: int = 512
    openai_timeout_seconds: int = 30

    # --- Tenant ---
    default_tenant_id: str = "default"

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def resolved_faiss_index_dir(self) -> str:
        return self.faiss_index_dir or os.path.join(self.data_dir, "indices")

    @property
    def resolved_metadata_db_path(self) -> str:
        return self.metadata_db_path or os.path.join(self.data_dir, "metadata.db")

    @property
    def resolved_skills_db_path(self) -> str:
        return self.skills_db_path or os.path.join(self.data_dir, "skills.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
