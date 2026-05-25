"""Tests for the ResumeJDParser (rule-based) and OpenAIParser."""

from app.services.parser import ResumeJDParser
from app.services.openai_parser import OpenAIParser
from tests.conftest import SAMPLE_RESUME, SAMPLE_JD


# ── Rule-based parser tests ──────────────────────────────────────

def test_parse_resume_extracts_skills() -> None:
    parser = ResumeJDParser()
    result = parser.parse(SAMPLE_RESUME)
    assert "python" in result.skills
    assert "typescript" in result.skills
    assert "fastapi" in result.skills
    assert "postgresql" in result.skills or "postgres" in result.skills
    assert "docker" in result.skills
    assert "kubernetes" in result.skills or "k8s" in result.skills


def test_parse_resume_extracts_experience() -> None:
    parser = ResumeJDParser()
    result = parser.parse(SAMPLE_RESUME)
    assert result.experience_years is not None
    assert result.experience_years > 0


def test_parse_resume_extracts_role() -> None:
    parser = ResumeJDParser()
    result = parser.parse(SAMPLE_RESUME)
    assert result.role is not None
    assert "software" in result.role.lower()


def test_parse_jd_extracts_skills() -> None:
    parser = ResumeJDParser()
    result = parser.parse(SAMPLE_JD)
    assert "python" in result.skills
    assert "typescript" in result.skills
    assert "fastapi" in result.skills
    assert "docker" in result.skills


def test_parse_empty_text() -> None:
    parser = ResumeJDParser()
    result = parser.parse("")
    assert result.skills == []
    assert result.experience_years is None
    assert result.role is None
    assert result.location is None


# ── OpenAI parser tests (fallback path) ──────────────────────────
# These tests exercise the rule-based fallback since no API key is set.

def test_openai_parser_fallback_no_key(monkeypatch) -> None:
    """Without an API key, OpenAIParser falls back to rule-based parsing."""
    monkeypatch.setattr("app.config.settings.openai_api_key", None)
    parser = OpenAIParser(api_key="")  # Explicitly no key
    result = parser.parse(SAMPLE_RESUME)
    assert len(result.skills) > 0
    assert "python" in result.skills


def test_openai_parser_fallback_on_empty_text(monkeypatch) -> None:
    """Fallback handles empty text gracefully."""
    monkeypatch.setattr("app.config.settings.openai_api_key", None)
    parser = OpenAIParser(api_key="")
    result = parser.parse("")
    assert result.skills == []
    assert result.experience_years is None
    assert result.role is None
    assert result.location is None


def test_openai_parser_parse_jd_via_fallback(monkeypatch) -> None:
    """JD parsing via OpenAIParser fallback yields expected skills."""
    monkeypatch.setattr("app.config.settings.openai_api_key", None)
    parser = OpenAIParser(api_key="")
    result = parser.parse(SAMPLE_JD)
    assert "python" in result.skills
    assert "docker" in result.skills
    assert "typescript" in result.skills


def test_openai_parser_model_name() -> None:
    """The model_name property returns the configured model."""
    parser = OpenAIParser(api_key="")
    assert parser.model_name == "gpt-4o-mini"
