"""
Pydantic models for API request/response serialization.

Supports the candidate scorecard flow:
  - Upload structured scorecards (role_applied, skills, interview_score, experience)
  - Upload job descriptions (text-based)
  - Match candidates against JDs with weighted scoring
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
#  Internal ID generation helpers
# ──────────────────────────────────────────────

def generate_id() -> str:
    """Generate a unique identifier (UUID v4 hex)."""
    return uuid.uuid4().hex


def current_timestamp() -> datetime:
    return datetime.utcnow()


# ──────────────────────────────────────────────
#  Candidate Scorecard
# ──────────────────────────────────────────────

class CandidateScorecardCreate(BaseModel):
    """Request payload to upload a candidate's interview scorecard."""
    candidate_name: str = Field(..., min_length=1, description="Candidate name")
    role_applied: str = Field(..., min_length=1, description="Role the candidate applied for")
    skills: list[str] = Field(..., min_length=1, description="Skills assessed during interview")
    interview_score: float = Field(..., ge=0, le=100, description="Interview score out of 100")
    experience_years: float = Field(..., ge=0, description="Years of professional experience")
    tenant_id: str = Field("default", description="Tenant / employer identifier")
    candidate_id: Optional[str] = Field(None, description="Optional external candidate ID")


class CandidateScorecardResponse(BaseModel):
    """Result returned after a candidate scorecard is indexed."""
    candidate_id: str
    candidate_name: str
    role_applied: str
    skills: list[str] = []
    interview_score: float = 0.0
    experience_years: float = 0.0
    embedding_model_version: str = ""


class BulkCandidateResult(BaseModel):
    """Result for a single candidate within a bulk upload."""
    candidate_id: str
    candidate_name: str
    status: str = "success"  # "success" or "failed"
    error: Optional[str] = None
    skills: list[str] = []
    interview_score: float = 0.0
    experience_years: float = 0.0


class BulkCandidateResponse(BaseModel):
    """Response from a bulk candidate scorecard upload."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[BulkCandidateResult] = []


# ──────────────────────────────────────────────
#  Job Description
# ──────────────────────────────────────────────

class JDCreate(BaseModel):
    """Request payload to parse a job description (text mode)."""
    text: str = Field(..., min_length=1, description="Full text of the job description")
    tenant_id: str = Field("default", description="Tenant / employer identifier")
    jd_id: Optional[str] = Field(None, description="Optional external JD ID")


class JDParsed(BaseModel):
    """Result returned after JD is parsed and indexed."""
    jd_id: str
    filename: Optional[str] = None
    skills: list[str] = []
    experience_years: Optional[float] = None
    role: Optional[str] = None
    location: Optional[str] = None
    embedding_model_version: str = ""


# ──────────────────────────────────────────────
#  Matching
# ──────────────────────────────────────────────

class MatchRequestParams(BaseModel):
    """Query parameters for the match endpoint."""
    top_k: int = Field(50, ge=1, le=200, description="Number of candidates to retrieve")
    rerank: bool = Field(False, description="Enable LLM-based reranking")
    tenant_id: Optional[str] = Field(None, description="Filter by tenant")
    experience_min: Optional[float] = Field(None, ge=0, description="Minimum experience years")
    experience_max: Optional[float] = Field(None, ge=0, description="Maximum experience years")


class MatchResult(BaseModel):
    """A single candidate match result."""
    candidate_id: str
    candidate_name: str = ""
    score: float = Field(..., ge=0, le=100, description="Match score 0–100")
    summary: str = ""
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    interview_score: float = 0.0
    ai_reason: Optional[str] = None
    match_type: str = Field("direct", description="Type of match: 'direct' or 'related'")
    suggested_roles: list[str] = Field([], description="Suggested related roles if match_type is 'related'")


class MatchResponse(BaseModel):
    """Response from the AI job matching endpoint."""
    jd_id: str
    total_candidates: int = 0
    matches: list[MatchResult] = []


# ──────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    embedding_model: str = ""
    candidate_count: int = 0
    jd_count: int = 0
    version: str = "1.0.0"
