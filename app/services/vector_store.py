"""
FAISS-based vector store for candidate-scorecard and job-description embeddings.

Supports:
- IndexFlatIP (exact cosine similarity via normalized vectors)
- IndexIDMap for direct look‑up by external ID
- In-memory vector cache for reconstruction
- Separate indices per type (candidate / jd)
- Persistence to / from disk
- Metadata-filtered searches via an in-memory whitelist pass

Performance note: `save()` is called explicitly after inserts to persist to disk.
Call `save()` after batch operations for efficiency.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages FAISS indices for candidate-scorecard and JD vectors."""

    def __init__(self, index_dir: Optional[str] = None) -> None:
        self._index_dir = Path(index_dir or settings.resolved_faiss_index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # internal_id → external_id  (int → str)
        self._candidate_id_map: dict[int, str] = {}
        self._jd_id_map: dict[int, str] = {}

        # Reverse: external_id → internal_id
        self._candidate_reverse_map: dict[str, int] = {}
        self._jd_reverse_map: dict[str, int] = {}

        # In-memory vector cache
        self._candidate_vectors: dict[int, np.ndarray] = {}
        self._jd_vectors: dict[int, np.ndarray] = {}

        self._candidate_index: Optional[faiss.Index] = None
        self._jd_index: Optional[faiss.Index] = None

        self._next_candidate_id: int = 1
        self._next_jd_id: int = 1

        self._dimension = settings.embedding_dimension

        self._load_or_create_indices()

    # ── Index initialisation ───────────────────────────────────────

    def _load_or_create_indices(self) -> None:
        """Load existing indices from disk or create new ones."""
        candidate_path = self._index_dir / "candidates.index"
        jd_path = self._index_dir / "jds.index"
        maps_path = self._index_dir / "id_maps.pkl"
        vectors_path = self._index_dir / "vectors.pkl"

        if candidate_path.exists() and jd_path.exists():
            try:
                self._candidate_index = faiss.read_index(str(candidate_path))
                self._jd_index = faiss.read_index(str(jd_path))
                if maps_path.exists():
                    with open(maps_path, "rb") as f:
                        data = pickle.load(f)
                    self._candidate_id_map = data.get("candidate_id_map", {})
                    self._jd_id_map = data.get("jd_id_map", {})
                    self._candidate_reverse_map = data.get("candidate_reverse_map", {})
                    self._jd_reverse_map = data.get("jd_reverse_map", {})
                    self._next_candidate_id = data.get("next_candidate_id", max(self._candidate_id_map.keys(), default=0) + 1)
                    self._next_jd_id = data.get("next_jd_id", max(self._jd_id_map.keys(), default=0) + 1)
                if vectors_path.exists():
                    with open(vectors_path, "rb") as f:
                        vdata = pickle.load(f)
                    self._candidate_vectors = vdata.get("candidate_vectors", {})
                    self._jd_vectors = vdata.get("jd_vectors", {})
                logger.info(
                    "Loaded FAISS indices: %d candidates, %d JDs.",
                    self._candidate_index.ntotal,
                    self._jd_index.ntotal,
                )
                return
            except Exception as exc:
                logger.warning("Failed to load indices, creating new ones: %s", exc)

        self._candidate_index = self._create_index()
        self._jd_index = self._create_index()
        logger.info("Created new FAISS indices.")

    def _create_index(self) -> faiss.Index:
        """Create an IndexIDMap(IndexFlatIP)."""
        base = faiss.IndexFlatIP(self._dimension)
        return faiss.IndexIDMap(base)

    # ── Persistence ────────────────────────────────────────────────

    def save(self) -> None:
        """Persist both indices, ID maps, and vector cache to disk."""
        candidate_path = self._index_dir / "candidates.index"
        jd_path = self._index_dir / "jds.index"
        maps_path = self._index_dir / "id_maps.pkl"
        vectors_path = self._index_dir / "vectors.pkl"

        faiss.write_index(self._candidate_index, str(candidate_path))
        faiss.write_index(self._jd_index, str(jd_path))

        with open(maps_path, "wb") as f:
            pickle.dump(
                {
                    "candidate_id_map": self._candidate_id_map,
                    "jd_id_map": self._jd_id_map,
                    "candidate_reverse_map": self._candidate_reverse_map,
                    "jd_reverse_map": self._jd_reverse_map,
                    "next_candidate_id": self._next_candidate_id,
                    "next_jd_id": self._next_jd_id,
                },
                f,
            )

        with open(vectors_path, "wb") as f:
            pickle.dump(
                {
                    "candidate_vectors": self._candidate_vectors,
                    "jd_vectors": self._jd_vectors,
                },
                f,
            )

    # ── Insert ─────────────────────────────────────────────────────

    def add_candidate(self, external_id: str, vector: np.ndarray) -> None:
        """Add or upsert a candidate-scorecard vector."""
        internal_id = self._candidate_reverse_map.get(external_id)
        if internal_id is not None:
            self._candidate_index.remove_ids(np.array([internal_id], dtype=np.int64))
            self._candidate_vectors.pop(internal_id, None)
        else:
            internal_id = self._next_candidate_id
            self._next_candidate_id += 1

        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        ids = np.array([internal_id], dtype=np.int64)
        self._candidate_index.add_with_ids(vec, ids)

        self._candidate_vectors[internal_id] = vec.ravel().copy()

        self._candidate_id_map[internal_id] = external_id
        self._candidate_reverse_map[external_id] = internal_id
        self.save()

    def add_jd(self, external_id: str, vector: np.ndarray) -> None:
        """Add or upsert a JD vector."""
        internal_id = self._jd_reverse_map.get(external_id)
        if internal_id is not None:
            self._jd_index.remove_ids(np.array([internal_id], dtype=np.int64))
            self._jd_vectors.pop(internal_id, None)
        else:
            internal_id = self._next_jd_id
            self._next_jd_id += 1

        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        ids = np.array([internal_id], dtype=np.int64)
        self._jd_index.add_with_ids(vec, ids)

        self._jd_vectors[internal_id] = vec.ravel().copy()

        self._jd_id_map[internal_id] = external_id
        self._jd_reverse_map[external_id] = internal_id
        self.save()

    # ── Query ──────────────────────────────────────────────────────

    def search_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        allowed_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Search the candidate index for nearest neighbours.

        Returns list of (external_id, cosine_similarity) sorted descending.
        If *allowed_ids* is provided, only those external IDs are returned.
        """
        if self._candidate_index.ntotal == 0:
            return []

        k = min(top_k, self._candidate_index.ntotal)
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self._candidate_index.search(query, k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            external_id = self._candidate_id_map.get(int(idx))
            if external_id is None:
                continue
            if allowed_ids is not None and external_id not in allowed_ids:
                continue
            results.append((external_id, float(dist)))

        return results

    def get_candidate_vector(self, external_id: str) -> Optional[np.ndarray]:
        """Retrieve a candidate vector by external ID from in-memory cache."""
        internal_id = self._candidate_reverse_map.get(external_id)
        if internal_id is None:
            return None
        vec = self._candidate_vectors.get(internal_id)
        if vec is not None:
            return vec.copy()
        return None

    def get_jd_vector(self, external_id: str) -> Optional[np.ndarray]:
        """Retrieve a JD vector by external ID from in-memory cache."""
        internal_id = self._jd_reverse_map.get(external_id)
        if internal_id is None:
            return None
        vec = self._jd_vectors.get(internal_id)
        if vec is not None:
            return vec.copy()
        return None

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def candidate_count(self) -> int:
        return self._candidate_index.ntotal if self._candidate_index else 0

    @property
    def jd_count(self) -> int:
        return self._jd_index.ntotal if self._jd_index else 0
