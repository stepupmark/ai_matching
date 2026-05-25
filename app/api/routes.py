"""
FastAPI route handlers for the Candidate Scorecard Matching system.

Endpoints:
  - POST /api/candidate/upload        — Upload a single candidate scorecard
  - POST /api/candidate/bulk-upload   — Bulk upload multiple scorecards
  - POST /api/jd/parse                — Parse & index a JD from text
  - GET  /api/aijob-matching/{jd_id}  — Match candidates against a JD
  - GET  /health                      — Health check
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.config import settings
from app.models.schemas import (
    CandidateScorecardCreate,
    CandidateScorecardResponse,
    BulkCandidateResponse,
    BulkCandidateResult,
    JDParsed,
    JDCreate,
    MatchRequestParams,
    MatchResponse,
    HealthResponse,
    generate_id,
)
from app.services.document_reader import extract_text, UnsupportedFormatError, DocumentReadError
from app.services.matcher import MatcherService

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency injection ───────────────────────────────────────────

def get_matcher(request: Request) -> MatcherService:
    matcher: Optional[MatcherService] = getattr(request.app.state, "matcher_service", None)
    if matcher is None:
        raise HTTPException(
            status_code=503,
            detail="AI Job Matching service is not ready. The server may still be starting up.",
        )
    return matcher


# ── Health ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health(matcher: MatcherService = Depends(get_matcher)) -> HealthResponse:
    """Health check — reports counts of indexed candidates and JDs."""
    return HealthResponse(
        status="ok",
        embedding_model=matcher.embedding_service.model_name,
        candidate_count=matcher.vector_store.candidate_count,
        jd_count=matcher.vector_store.jd_count,
        version="1.0.0",
    )


# ═══════════════════════════════════════════════════════════════════
#  CANDIDATE SCORECARD ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/api/candidate/upload",
    response_model=CandidateScorecardResponse,
    status_code=201,
    tags=["Candidate"],
    summary="Upload a candidate scorecard",
)
async def upload_candidate(
    payload: CandidateScorecardCreate,
    matcher: MatcherService = Depends(get_matcher),
) -> CandidateScorecardResponse:
    """
    Upload and index a candidate's interview scorecard.

    Accepts structured JSON with:
      - candidate_name  — Full name
      - role_applied    — Role the candidate applied for
      - skills          — List of skills assessed in the interview
      - interview_score — Score out of 100
      - experience_years — Years of experience
    """
    candidate_id = payload.candidate_id or generate_id()
    matcher.index_candidate(scorecard=payload, candidate_id=candidate_id)
    meta = matcher.metadata_store.get_candidate(candidate_id)
    return CandidateScorecardResponse(
        candidate_id=candidate_id,
        candidate_name=meta.get("candidate_name", "") if meta else payload.candidate_name,
        role_applied=meta.get("role_applied", "") if meta else payload.role_applied,
        skills=meta.get("skills", []) if meta else [],
        interview_score=meta.get("interview_score", 0.0) if meta else payload.interview_score,
        experience_years=meta.get("experience_years", 0.0) if meta else payload.experience_years,
        embedding_model_version=matcher.embedding_service.model_version,
    )


@router.post(
    "/api/candidate/bulk-upload",
    response_model=BulkCandidateResponse,
    status_code=201,
    tags=["Candidate"],
    summary="Bulk upload multiple candidate scorecards",
)
async def bulk_upload_candidates(
    payload: list[CandidateScorecardCreate],
    matcher: MatcherService = Depends(get_matcher),
) -> BulkCandidateResponse:
    """
    Upload and index multiple candidate scorecards in a single request.

    Accepts up to **50** scorecards at once. Each scorecard is embedded
    and indexed individually. Returns per-item status.
    """
    if not payload:
        raise HTTPException(status_code=400, detail="No scorecards provided")

    if len(payload) > settings.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many scorecards. Maximum is {settings.max_upload_files} per request.",
        )

    results: list[BulkCandidateResult] = []

    for scorecard in payload:
        try:
            candidate_id = scorecard.candidate_id or generate_id()
            matcher.index_candidate(scorecard=scorecard, candidate_id=candidate_id)
            results.append(BulkCandidateResult(
                candidate_id=candidate_id,
                candidate_name=scorecard.candidate_name,
                status="success",
                skills=scorecard.skills,
                interview_score=scorecard.interview_score,
                experience_years=scorecard.experience_years,
            ))
        except Exception as exc:
            logger.error("Bulk upload failed for '%s': %s", scorecard.candidate_name, exc)
            results.append(BulkCandidateResult(
                candidate_id="",
                candidate_name=scorecard.candidate_name,
                status="failed",
                error=str(exc),
            ))

    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info(
        "Bulk upload: %d/%d succeeded, %d failed",
        succeeded, len(results), failed,
    )

    return BulkCandidateResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


# ═══════════════════════════════════════════════════════════════════
#  JOB DESCRIPTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/api/jd/parse",
    response_model=JDParsed,
    status_code=201,
    tags=["Job Description"],
    summary="Parse a job description from text",
)
async def parse_jd_text(
    payload: JDCreate,
    matcher: MatcherService = Depends(get_matcher),
) -> JDParsed:
    """
    Parse and index a job description from raw text.

    Stores embedding + metadata for subsequent matching requests.
    """
    jd_id = payload.jd_id or generate_id()
    matcher.index_jd(text=payload.text, tenant_id=payload.tenant_id, jd_id=jd_id)
    meta = matcher.metadata_store.get_jd(jd_id)
    return JDParsed(
        jd_id=jd_id,
        skills=meta.get("skills", []) if meta else [],
        experience_years=meta.get("experience_years") if meta else None,
        role=meta.get("role") if meta else None,
        location=meta.get("location") if meta else None,
        embedding_model_version=matcher.embedding_service.model_version,
    )


@router.post(
    "/api/jd/upload-file",
    response_model=JDParsed,
    status_code=201,
    tags=["Job Description"],
    summary="Upload a JD as PDF/DOCX/TXT file",
)
async def upload_jd_file(
    file: UploadFile = File(..., description="JD file (PDF, DOCX, or TXT)"),
    tenant_id: str = "default",
    matcher: MatcherService = Depends(get_matcher),
) -> JDParsed:
    """
    Upload and index a job description from a file.

    Supports PDF, DOCX, and TXT formats. The file is automatically
    extracted to plain text, parsed with OpenAI, embedded, and indexed
    in the vector store for subsequent matching.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()

    try:
        text = extract_text(content, file.filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DocumentReadError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    jd_id = generate_id()
    matcher.index_jd(text=text, tenant_id=tenant_id, jd_id=jd_id)

    meta = matcher.metadata_store.get_jd(jd_id)
    return JDParsed(
        jd_id=jd_id,
        filename=file.filename,
        skills=meta.get("skills", []) if meta else [],
        experience_years=meta.get("experience_years") if meta else None,
        role=meta.get("role") if meta else None,
        location=meta.get("location") if meta else None,
        embedding_model_version=matcher.embedding_service.model_version,
    )


# ═══════════════════════════════════════════════════════════════════
#  MATCHING ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@router.get(
    "/api/aijob-matching/{jd_id}",
    response_model=MatchResponse,
    tags=["Matching"],
    summary="Match candidates against a job description",
)
async def match(
    jd_id: str,
    params: MatchRequestParams = Depends(),
    matcher: MatcherService = Depends(get_matcher),
) -> MatchResponse:
    """
    Get matching candidates for a job description.

    Uses hybrid scoring:
      - Vector similarity (40%) + Skill match (30%)
      - Experience fit (15%) + Role alignment (10%) + Interview score (5%)

    Works with 1000+ candidates in milliseconds.
    """
    if not jd_id:
        raise HTTPException(status_code=400, detail="jd_id is required")

    return matcher.match(
        jd_id=jd_id,
        top_k=params.top_k,
        rerank=params.rerank,
        tenant_id=params.tenant_id,
        experience_min=params.experience_min,
        experience_max=params.experience_max,
    )
