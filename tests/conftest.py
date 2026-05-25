"""
Shared pytest fixtures for Candidate Scorecard Matching tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.config import settings


# Clean up stale files if they exist
_stale = [
    "app/services/document_reader.py",
    "tests/test_document_reader.py",
    "tests/test_file_upload.py",
]
for _f in _stale:
    try:
        os.remove(_f)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def temp_data_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Redirect data to a temporary directory for each test."""
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(settings, "data_dir", str(tmp))
    monkeypatch.setattr(settings, "faiss_index_dir", str(tmp / "indices"))
    monkeypatch.setattr(settings, "metadata_db_path", str(tmp / "metadata.db"))
    yield tmp


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient backed by the real app."""
    from app.main import app
    with TestClient(app) as c:
        yield c


# ── Sample data ────────────────────────────────────────────────────

# Raw text for parser tests (simulates a resume body)
SAMPLE_RESUME = """
John Doe
Senior Software Engineer

Experience: 6 years of software development

Skills:
- Python, TypeScript, FastAPI, PostgreSQL
- Docker, Kubernetes, AWS
- CI/CD pipelines, Microservices
- Apache Kafka, RabbitMQ

Education:
- B.S. Computer Science, MIT

Location: San Francisco, CA
"""

SAMPLE_CANDIDATE = {
    "candidate_name": "John Doe",
    "role_applied": "Senior Software Engineer",
    "skills": ["Python", "TypeScript", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS"],
    "interview_score": 85.0,
    "experience_years": 6.0,
}

SAMPLE_CANDIDATE_2 = {
    "candidate_name": "Jane Smith",
    "role_applied": "Data Scientist",
    "skills": ["Python", "TensorFlow", "scikit-learn", "PyTorch", "SQL", "Tableau"],
    "interview_score": 72.0,
    "experience_years": 4.0,
}

SAMPLE_CANDIDATE_3 = {
    "candidate_name": "Bob Wilson",
    "role_applied": "Junior Developer",
    "skills": ["Python", "JavaScript", "React"],
    "interview_score": 60.0,
    "experience_years": 1.5,
}

SAMPLE_JD = """
Job Description: Senior Software Engineer

We are looking for a Senior Software Engineer to join our platform team.

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python and TypeScript
- Experience with FastAPI or similar web frameworks
- Hands-on experience with PostgreSQL
- Knowledge of Docker and Kubernetes
- Experience building microservices
- Familiarity with CI/CD pipelines

Nice to have:
- Experience with Apache Kafka
- Knowledge of AWS cloud services

Location: San Francisco, CA (Remote OK)
"""
