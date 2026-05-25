"""Tests for file upload and bulk resume upload endpoints."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from tests.conftest import SAMPLE_RESUME, SAMPLE_JD

# ── TXT file helpers ──────────────────────────────────────────────

def _make_txt_upload(content: str, filename: str = "resume.txt"):
    return io.BytesIO(content.encode("utf-8"))


# ── File upload tests ──────────────────────────────────────────────

class TestResumeFileUpload:
    def test_upload_txt_resume(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_RESUME, "resume.txt")
        resp = client.post(
            "/api/resume/upload-file",
            files={"file": ("resume.txt", file, "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "resume_id" in data
        assert data["filename"] == "resume.txt"
        assert len(data["skills"]) > 0
        assert "python" in [s.lower() for s in data["skills"]]

    def test_upload_resume_with_tenant(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_RESUME, "resume.txt")
        resp = client.post(
            "/api/resume/upload-file",
            files={"file": ("resume.txt", file, "text/plain")},
            data={"tenant_id": "acme_corp"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "resume_id" in data

    def test_upload_resume_wrong_extension(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_RESUME, "resume.png")
        resp = client.post(
            "/api/resume/upload-file",
            files={"file": ("resume.png", file, "image/png")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_upload_empty_filename(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_RESUME, "")
        resp = client.post(
            "/api/resume/upload-file",
            files={"file": ("", file, "text/plain")},
        )
        assert resp.status_code == 400


class TestJDFileUpload:
    def test_upload_txt_jd(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_JD, "job_description.txt")
        resp = client.post(
            "/api/jd/upload-file",
            files={"file": ("jd.txt", file, "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "jd_id" in data
        assert data["filename"] == "jd.txt"
        assert len(data["skills"]) > 0

    def test_upload_jd_wrong_extension(self, client: TestClient) -> None:
        file = _make_txt_upload(SAMPLE_JD, "jd.csv")
        resp = client.post(
            "/api/jd/upload-file",
            files={"file": ("jd.csv", file, "text/csv")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()


class TestBulkUpload:
    def test_bulk_upload_multiple_resumes(self, client: TestClient) -> None:
        file1 = _make_txt_upload(SAMPLE_RESUME, "john.txt")
        file2 = _make_txt_upload(SAMPLE_RESUME.replace("John", "Jane"), "jane.txt")
        resp = client.post(
            "/api/resume/bulk-upload",
            files=[
                ("files", ("john.txt", file1, "text/plain")),
                ("files", ("jane.txt", file2, "text/plain")),
            ],
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["status"] == "success"
            assert "resume_id" in r

    def test_bulk_upload_empty(self, client: TestClient) -> None:
        resp = client.post("/api/resume/bulk-upload", files=[])
        assert resp.status_code == 400

    def test_bulk_upload_partial_failure(self, client: TestClient) -> None:
        """One good file, one unsupported format."""
        file1 = _make_txt_upload(SAMPLE_RESUME, "good.txt")
        file2 = io.BytesIO(b"not a document")
        resp = client.post(
            "/api/resume/bulk-upload",
            files=[
                ("files", ("good.txt", file1, "text/plain")),
                ("files", ("bad.csv", file2, "text/csv")),
            ],
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 1
        assert data["failed"] == 1
        # Find which failed
        fail_results = [r for r in data["results"] if r["status"] == "failed"]
        assert len(fail_results) == 1
        assert "unsupported" in fail_results[0]["error"].lower()

    def test_bulk_upload_too_many_files(self, client: TestClient) -> None:
        """Exceed the max files limit."""
        from app.config import settings
        files = [
            ("files", (f"r{i}.txt", _make_txt_upload(f"Resume {i}"), "text/plain"))
            for i in range(settings.max_upload_files + 1)
        ]
        resp = client.post("/api/resume/bulk-upload", files=files)
        assert resp.status_code == 400
        assert "too many" in resp.json()["detail"].lower()


class TestMatchAfterFileUpload:
    """End-to-end: upload resume via file, upload JD via file, then match."""

    def test_match_after_file_upload(self, client: TestClient) -> None:
        # Upload resume as file
        resume_file = _make_txt_upload(SAMPLE_RESUME, "resume.txt")
        resume_resp = client.post(
            "/api/resume/upload-file",
            files={"file": ("resume.txt", resume_file, "text/plain")},
        )
        assert resume_resp.status_code == 201

        # Upload JD as file
        jd_file = _make_txt_upload(SAMPLE_JD, "jd.txt")
        jd_resp = client.post(
            "/api/jd/upload-file",
            files={"file": ("jd.txt", jd_file, "text/plain")},
        )
        assert jd_resp.status_code == 201
        jd_id = jd_resp.json()["jd_id"]

        # Match
        resp = client.get(f"/api/aijob-matching/{jd_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_candidates"] >= 1
        assert len(data["matches"]) >= 1
        m = data["matches"][0]
        assert 0 <= m["score"] <= 100
        assert len(m["matched_skills"]) > 0
