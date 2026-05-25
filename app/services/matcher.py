"""
Core matching orchestrator for the candidate scorecard system.

Coordinates the retrieval pipeline:
  1. Load JD embedding from FAISS
  2. Query vector index for topK nearest candidates
  3. Merge with scorecard metadata (skills, experience, interview_score)
  4. Compute hybrid scores: vector_similarity(40%) + skill_match(30%) + experience_fit(15%) + role_alignment(10%) + interview_score(5%)
  5. Optionally rerank with LLM
  6. Return sorted results

Optimised for structured scorecard data at scale (1000+ candidates).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.models.schemas import CandidateScorecardCreate, MatchResult, MatchResponse
from app.services.embedding import EmbeddingService
from app.services.metadata_store import MetadataStore
from app.services.openai_parser import OpenAIParser
from app.services.reranker import RerankerResult, RerankerService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MatcherService:
    """Orchestrates the candidate-JD matching pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        embedding_service: EmbeddingService,
        parser: OpenAIParser,
        reranker: RerankerService,
    ) -> None:
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_service = embedding_service
        self._parser = parser
        self._reranker = reranker

    # ── Public properties ──────────────────────────────────────────

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def metadata_store(self) -> MetadataStore:
        return self._metadata_store

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    # ── Helper: build searchable profile text from scorecard data ──

    @staticmethod
    def _build_profile_text(
        candidate_name: str,
        role_applied: str,
        skills: list[str],
        experience_years: float,
    ) -> str:
        """
        Build a semantic search profile from structured scorecard data.

        This text is what gets embedded for vector search. It's designed
        to read naturally so the embedding model captures semantic meaning
        that aligns with JD text.
        """
        skills_str = ", ".join(skills)
        return (
            f"Candidate: {candidate_name}. "
            f"Role Applied: {role_applied}. "
            f"Skills: {skills_str}. "
            f"Experience: {experience_years} years."
        )

    # ── Candidate indexing ─────────────────────────────────────────

    def index_candidate(
        self,
        scorecard: CandidateScorecardCreate,
        candidate_id: str,
    ) -> str:
        """
        Index a candidate scorecard: build profile text → embed → store.

        Args:
            scorecard: Structured scorecard data from the API.
            candidate_id: Unique ID for this candidate.

        Returns:
            The candidate_id.
        """
        profile_text = self._build_profile_text(
            candidate_name=scorecard.candidate_name,
            role_applied=scorecard.role_applied,
            skills=scorecard.skills,
            experience_years=scorecard.experience_years,
        )
        vector = self._embedding_service.embed(profile_text)
        self._vector_store.add_candidate(candidate_id, vector)
        self._metadata_store.upsert_candidate(
            candidate_id=candidate_id,
            tenant_id=scorecard.tenant_id,
            candidate_name=scorecard.candidate_name,
            role_applied=scorecard.role_applied,
            skills=scorecard.skills,
            interview_score=scorecard.interview_score,
            experience_years=scorecard.experience_years,
            generated_profile=profile_text,
            embedding_model_version=self._embedding_service.model_version,
        )
        logger.info(
            "Indexed candidate %s (%s) — %d skills, %.1f yr exp, interview score %.1f",
            candidate_id, scorecard.candidate_name,
            len(scorecard.skills), scorecard.experience_years,
            scorecard.interview_score,
        )
        return candidate_id

    # ── JD indexing ────────────────────────────────────────────────

    def index_jd(self, text: str, tenant_id: str, jd_id: str) -> str:
        """Parse, embed, and store a job description. Returns the jd_id."""
        parsed = self._parser.parse(text)
        vector = self._embedding_service.embed(text)
        self._vector_store.add_jd(jd_id, vector)
        self._metadata_store.upsert_jd(
            jd_id=jd_id,
            tenant_id=tenant_id,
            parsed_text=text,
            skills=parsed.skills,
            experience_years=parsed.experience_years,
            role=parsed.role,
            location=parsed.location,
            embedding_model_version=self._embedding_service.model_version,
        )
        logger.info("Indexed JD %s (%d skills)", jd_id, len(parsed.skills))
        return jd_id

    # ── Matching ───────────────────────────────────────────────────

    def match(
        self,
        jd_id: str,
        top_k: int = 50,
        rerank: bool = False,
        tenant_id: Optional[str] = None,
        experience_min: Optional[float] = None,
        experience_max: Optional[float] = None,
    ) -> MatchResponse:
        """
        Execute the OpenAI-powered matching pipeline.

        Pipeline:
          1. Load JD metadata from the store
          2. Vector search for topK candidates (pre-filter)
          3. Apply experience filters
          4. Evaluate all candidates via OpenAI — scoring by:
               role match (highest), experience, skills, interview_score
          5. Separate into direct matches and related-role matches
             (e.g. Frontend Developer for a Fullstack Developer JD)
          6. Sort: direct matches first (by score), then related (by score)
          7. Build and return response
        """
        # --- Step 1: Load JD metadata ---
        jd_meta = self._metadata_store.get_jd(jd_id)
        if jd_meta is None:
            logger.warning("JD %s not found in metadata store", jd_id)
            return MatchResponse(jd_id=jd_id, total_candidates=0, matches=[])

        jd_vector = self._vector_store.get_jd_vector(jd_id)
        if jd_vector is None:
            jd_vector = self._embedding_service.embed(jd_meta["parsed_text"])

        # --- Step 2: Determine candidate pool ---
        allowed_ids: Optional[set[str]] = None
        if tenant_id:
            allowed_ids = set(self._metadata_store.get_active_candidate_ids(tenant_id))

        # --- Step 3: Vector search (pre-filter) ---
        search_results = self._vector_store.search_candidates(
            jd_vector, top_k=top_k, allowed_ids=allowed_ids,
        )

        if not search_results:
            return MatchResponse(
                jd_id=jd_id,
                total_candidates=self._metadata_store.candidate_count,
                matches=[],
            )

        # --- Step 4: Merge metadata & apply experience filters ---
        jd_skills = set(jd_meta.get("skills", []))
        jd_role = jd_meta.get("role") or ""

        candidates: list[dict] = []
        for candidate_id, similarity in search_results:
            meta = self._metadata_store.get_candidate(candidate_id)
            if meta is None:
                continue

            candidate_exp = meta.get("experience_years", 0) or 0
            if experience_min is not None and candidate_exp < experience_min:
                continue
            if experience_max is not None and candidate_exp > experience_max:
                continue

            candidate_skills = set(meta.get("skills", []))
            matched_skills = sorted(jd_skills & candidate_skills)
            missing_skills = sorted(jd_skills - candidate_skills)

            candidates.append({
                "candidate_id": candidate_id,
                "candidate_name": meta.get("candidate_name", ""),
                "score": 0.0,  # will be set by OpenAI evaluation
                "summary": "",
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "interview_score": meta.get("interview_score", 0),
                "ai_reason": None,
                "similarity": similarity,
                "metadata": meta,
            })

        if not candidates:
            return MatchResponse(
                jd_id=jd_id,
                total_candidates=self._metadata_store.candidate_count,
                matches=[],
            )

        # --- Step 5: Evaluate via OpenAI (primary matching) ---
        rerank_limit = min(len(candidates), settings.rerank_limit)
        top_candidates = candidates[:rerank_limit]

        jd_text = jd_meta.get("parsed_text", "")
        evaluated = self._reranker.evaluate_matches(
            jd_text=jd_text,
            jd_skills=list(jd_skills),
            jd_role=jd_role,
            candidates=top_candidates,
        )

        evaluated_map: dict[str, RerankerResult] = {}
        for r in evaluated:
            evaluated_map[r.candidate_id] = r

        for c in candidates:
            cid = c["candidate_id"]
            if cid in evaluated_map:
                rr = evaluated_map[cid]
                c["score"] = rr.score
                c["summary"] = rr.summary
                c["matched_skills"] = rr.matched_skills
                c["missing_skills"] = rr.missing_skills
                c["ai_reason"] = rr.ai_reason
                c["match_type"] = rr.match_type
                c["suggested_roles"] = rr.suggested_roles
            else:
                c["match_type"] = "direct"
                c["suggested_roles"] = []

        # --- Step 6: Sort — direct matches first, then related, by score ---
        direct = [c for c in candidates if c.get("match_type", "direct") == "direct"]
        related = [c for c in candidates if c.get("match_type", "direct") != "direct"]

        direct.sort(key=lambda c: c["score"], reverse=True)
        related.sort(key=lambda c: c["score"], reverse=True)

        ordered = direct + related

        # --- Step 7: Build response ---
        matches = [
            MatchResult(
                candidate_id=c["candidate_id"],
                candidate_name=c["candidate_name"],
                score=round(c["score"], 1),
                summary=c["summary"],
                matched_skills=c["matched_skills"],
                missing_skills=c["missing_skills"],
                interview_score=c["interview_score"],
                ai_reason=c["ai_reason"],
                match_type=c.get("match_type", "direct"),
                suggested_roles=c.get("suggested_roles", []),
            )
            for c in ordered
        ]

        return MatchResponse(
            jd_id=jd_id,
            total_candidates=self._metadata_store.candidate_count,
            matches=matches,
        )
