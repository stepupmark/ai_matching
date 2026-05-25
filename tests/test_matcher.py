"""Integration tests for the MatcherService (candidate scorecard flow).

Uses the real embedding model (all-MiniLM-L6-v2) and FAISS.
"""

from __future__ import annotations

import pytest

from app.models.schemas import CandidateScorecardCreate, generate_id
from app.services.embedding import EmbeddingService
from app.services.matcher import MatcherService
from app.services.metadata_store import MetadataStore
from app.services.openai_parser import OpenAIParser
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStore
from tests.conftest import SAMPLE_CANDIDATE, SAMPLE_CANDIDATE_2, SAMPLE_CANDIDATE_3, SAMPLE_JD


@pytest.fixture
def matcher(temp_data_dir) -> MatcherService:  # noqa: ARG001
    vs = VectorStore()
    ms = MetadataStore()
    es = EmbeddingService()
    parser = OpenAIParser(api_key="")  # No API key → rule-based fallback for JD parsing
    reranker = RerankerService(api_key="")  # No key → fallback
    return MatcherService(
        vector_store=vs,
        metadata_store=ms,
        embedding_service=es,
        parser=parser,
        reranker=reranker,
    )


def _make_scorecard(**overrides) -> CandidateScorecardCreate:
    data = {**SAMPLE_CANDIDATE, "tenant_id": "t1", **overrides}
    return CandidateScorecardCreate(**data)


def test_index_and_match(matcher: MatcherService) -> None:
    """Index candidates and a JD, then verify correct ranking."""
    cid1 = matcher.index_candidate(
        _make_scorecard(candidate_name="John", role_applied="Senior Software Engineer"),
        candidate_id=generate_id(),
    )
    cid2 = matcher.index_candidate(
        _make_scorecard(**SAMPLE_CANDIDATE_2, tenant_id="t1"),
        candidate_id=generate_id(),
    )
    cid3 = matcher.index_candidate(
        _make_scorecard(**SAMPLE_CANDIDATE_3, tenant_id="t1"),
        candidate_id=generate_id(),
    )
    jid = matcher.index_jd(SAMPLE_JD, tenant_id="t1", jd_id=generate_id())

    result = matcher.match(jd_id=jid, top_k=10)

    assert result.jd_id == jid
    assert result.total_candidates >= 3
    assert len(result.matches) >= 1

    match_map = {m.candidate_id: m for m in result.matches}
    assert cid1 in match_map
    assert cid2 in match_map
    assert cid3 in match_map

    # John (SWE, 6yr, 85 score) should rank highest for a Senior SWE JD
    r1_score = match_map[cid1].score
    r2_score = match_map[cid2].score
    r3_score = match_map[cid3].score

    assert r1_score >= r2_score, (
        f"Expected SWE candidate ({r1_score}) to score >= data scientist ({r2_score})"
    )
    assert r1_score >= r3_score, (
        f"Expected SWE candidate ({r1_score}) to score >= junior dev ({r3_score})"
    )

    # Verify response shape
    m = match_map[cid1]
    assert 0 <= m.score <= 100
    assert isinstance(m.matched_skills, list)
    assert isinstance(m.missing_skills, list)
    assert m.candidate_name == "John"
    assert m.interview_score == 85.0
    assert len(m.summary) > 0
    assert m.match_type in ("direct", "related")
    assert isinstance(m.suggested_roles, list)


def test_match_no_jd(matcher: MatcherService) -> None:
    """Matching a non-existent JD returns empty."""
    result = matcher.match(jd_id="nonexistent")
    assert result.matches == []
    assert result.total_candidates == 0


def test_match_tenant_scoping(matcher: MatcherService) -> None:
    """Tenant filtering works correctly."""
    cid_a = matcher.index_candidate(
        _make_scorecard(tenant_id="tenant_a"),
        candidate_id=generate_id(),
    )
    matcher.index_candidate(
        _make_scorecard(**SAMPLE_CANDIDATE_2, tenant_id="tenant_b"),
        candidate_id=generate_id(),
    )
    jid = matcher.index_jd(SAMPLE_JD, tenant_id="tenant_a", jd_id=generate_id())

    # Without filter → returns all
    result_all = matcher.match(jd_id=jid, top_k=10)
    assert result_all.total_candidates >= 2

    # With tenant filter → only tenant_a's candidates
    result_filtered = matcher.match(jd_id=jid, top_k=10, tenant_id="tenant_a")
    assert result_filtered.total_candidates >= 1
    match_ids = [m.candidate_id for m in result_filtered.matches]
    assert cid_a in match_ids


def test_experience_filter(matcher: MatcherService) -> None:
    """Experience range filtering works."""
    cid1 = matcher.index_candidate(
        _make_scorecard(tenant_id="t1", experience_years=6.0),
        candidate_id=generate_id(),
    )
    matcher.index_candidate(
        _make_scorecard(**SAMPLE_CANDIDATE_3, tenant_id="t1", experience_years=1.5),
        candidate_id=generate_id(),
    )
    jid = matcher.index_jd(SAMPLE_JD, tenant_id="t1", jd_id=generate_id())

    # Filter for 5+ years → only the 6yr candidate
    result = matcher.match(jd_id=jid, top_k=10, experience_min=5.0)
    match_ids = [m.candidate_id for m in result.matches]
    assert cid1 in match_ids
    assert len(result.matches) >= 1
