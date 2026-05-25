"""API-level integration tests using FastAPI TestClient.

Tests the candidate scorecard upload → JD parse → match pipeline.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.conftest import SAMPLE_CANDIDATE, SAMPLE_CANDIDATE_2, SAMPLE_JD


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "all-MiniLM" in data["embedding_model"]
    assert "candidate_count" in data
    assert "jd_count" in data


def test_upload_candidate(client: TestClient) -> None:
    resp = client.post("/api/candidate/upload", json=SAMPLE_CANDIDATE)
    assert resp.status_code == 201
    data = resp.json()
    assert data["candidate_name"] == "John Doe"
    assert data["role_applied"] == "Senior Software Engineer"
    assert len(data["skills"]) > 0
    assert "python" in [s.lower() for s in data["skills"]]
    assert data["interview_score"] == 85.0
    assert data["experience_years"] == 6.0
    assert "candidate_id" in data


def test_upload_candidate_validates_fields(client: TestClient) -> None:
    """Missing required fields should return 422."""
    resp = client.post("/api/candidate/upload", json={"candidate_name": "No Skills"})
    assert resp.status_code == 422


def test_bulk_upload_candidates(client: TestClient) -> None:
    payload = [SAMPLE_CANDIDATE, SAMPLE_CANDIDATE_2]
    resp = client.post("/api/candidate/bulk-upload", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert r["status"] == "success"
        assert "candidate_id" in r


def test_bulk_upload_empty(client: TestClient) -> None:
    resp = client.post("/api/candidate/bulk-upload", json=[])
    assert resp.status_code == 400


def test_parse_jd(client: TestClient) -> None:
    resp = client.post("/api/jd/parse", json={"text": SAMPLE_JD})
    assert resp.status_code == 201
    data = resp.json()
    assert "jd_id" in data
    assert len(data["skills"]) > 0


def test_match_endpoint(client: TestClient) -> None:
    """Full pipeline: upload candidate → parse JD → match."""
    client.post("/api/candidate/upload", json=SAMPLE_CANDIDATE)
    jd_resp = client.post("/api/jd/parse", json={"text": SAMPLE_JD})
    jd_id = jd_resp.json()["jd_id"]

    resp = client.get(f"/api/aijob-matching/{jd_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jd_id"] == jd_id
    assert data["total_candidates"] >= 1
    assert len(data["matches"]) >= 1

    m = data["matches"][0]
    assert "candidate_id" in m
    assert "candidate_name" in m
    assert 0 <= m["score"] <= 100
    assert "summary" in m
    assert "matched_skills" in m
    assert "missing_skills" in m
    assert "interview_score" in m
    assert "match_type" in m
    assert "suggested_roles" in m
    assert m["match_type"] in ("direct", "related")
    assert isinstance(m["suggested_roles"], list)
    assert m["candidate_name"] == "John Doe"


def test_match_multiple_candidates(client: TestClient) -> None:
    """Multiple candidates: SWE applicant should rank higher for SWE JD."""
    client.post("/api/candidate/upload", json=SAMPLE_CANDIDATE)  # SWE
    client.post("/api/candidate/upload", json=SAMPLE_CANDIDATE_2)  # Data Scientist
    jd_resp = client.post("/api/jd/parse", json={"text": SAMPLE_JD})
    jd_id = jd_resp.json()["jd_id"]

    resp = client.get(f"/api/aijob-matching/{jd_id}", params={"top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["matches"]) >= 2

    # First result should be the SWE candidate
    top = data["matches"][0]
    assert top["candidate_name"] == "John Doe"


def test_upload_jd_file_txt(client: TestClient) -> None:
    """Upload a JD as a TXT file."""
    resp = client.post(
        "/api/jd/upload-file",
        files={"file": ("backend_engineer.txt", SAMPLE_JD.encode("utf-8"), "text/plain")},
        data={"tenant_id": "default"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "jd_id" in data
    assert data["filename"] == "backend_engineer.txt"
    assert len(data["skills"]) > 0
    # Verify it can be used for matching
    resp2 = client.get(f"/api/aijob-matching/{data['jd_id']}")
    assert resp2.status_code == 200


def test_upload_jd_file_unsupported_format(client: TestClient) -> None:
    """Uploading an unsupported file format should return 400."""
    resp = client.post(
        "/api/jd/upload-file",
        files={"file": ("job.html", b"<html>not supported</html>", "text/html")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_upload_jd_file_empty_filename(client: TestClient) -> None:
    """Upload without a filename should return 400."""
    resp = client.post(
        "/api/jd/upload-file",
        files={"file": ("", b"some content", "text/plain")},
    )
    assert resp.status_code == 400


def test_match_not_found(client: TestClient) -> None:
    resp = client.get("/api/aijob-matching/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["matches"] == []
    assert data["total_candidates"] == 0


def test_match_with_params(client: TestClient) -> None:
    client.post("/api/candidate/upload", json=SAMPLE_CANDIDATE)
    jd_resp = client.post("/api/jd/parse", json={"text": SAMPLE_JD})
    jd_id = jd_resp.json()["jd_id"]

    resp = client.get(
        f"/api/aijob-matching/{jd_id}",
        params={"top_k": 5, "rerank": False},
    )
    assert resp.status_code == 200
    assert len(resp.json()["matches"]) >= 1
