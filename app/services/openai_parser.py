"""
OpenAI-powered parser for resumes and job descriptions.

Replaces the rule-based ResumeJDParser with LLM-based extraction for
more accurate and context-aware parsing of skills, experience, role,
and location from unstructured text.

Falls back to the rule-based parser (ResumeJDParser) when:
  - The OpenAI API key is not configured
  - The API call fails or times out
  - The response cannot be parsed
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.config import settings
from app.services.parser import ResumeJDParser, ParsedDocument

logger = logging.getLogger(__name__)


class OpenAIParser:
    """
    Parses resume / JD text using OpenAI GPT-4o-mini for structured extraction.

    Uses JSON mode to get clean structured output. Falls back to the
    rule-based ResumeJDParser if OpenAI is unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        fallback_parser: Optional[ResumeJDParser] = None,
    ) -> None:
        self._api_key = api_key or settings.openai_api_key
        self._model = settings.openai_model
        self._temperature = settings.openai_rerank_temperature
        self._max_tokens = 1024  # parsing needs more tokens than reranking
        self._timeout = settings.openai_timeout_seconds
        self._fallback = fallback_parser or ResumeJDParser()
        self._client = None

    # ── Lazy OpenAI client ─────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    # ── Main parse method ──────────────────────────────────────────

    def parse(self, text: str) -> ParsedDocument:
        """
        Parse text using OpenAI, falling back to rule-based on failure.

        Args:
            text: Raw text content of a resume or job description.

        Returns:
            ParsedDocument with extracted skills, experience_years, role, location.
        """
        if not self._api_key:
            logger.debug("No OpenAI API key — using rule-based parser fallback")
            return self._fallback.parse(text)

        try:
            return self._call_llm(text)
        except Exception as exc:
            logger.warning(
                "OpenAI parsing failed (%s) — falling back to rule-based parser",
                exc,
            )
            return self._fallback.parse(text)

    # ── LLM call ───────────────────────────────────────────────────

    def _call_llm(self, text: str) -> ParsedDocument:
        """Call OpenAI to extract structured attributes from the text."""
        # Truncate very long texts to avoid excessive token usage
        truncated = text[:8000] if len(text) > 8000 else text

        system_prompt = """You are an expert resume and job description parser.

Extract the following fields from the provided text. Return ONLY valid JSON.

Fields to extract:
- skills: array of strings — technical and professional skills mentioned (e.g., ["Python", "AWS", "React", "Project Management"]). Be thorough but relevant.
- experience_years: number or null — years of professional experience mentioned. If a range is given (e.g., "5-7 years"), return the higher number. If no experience is mentioned, return null.
- role: string or null — the most likely job title / role of the person (for resumes) or the position being hired for (for job descriptions).
- location: string or null — the geographic location mentioned (city, state, or "Remote").

Output format:
{
  "skills": ["Skill1", "Skill2", ...],
  "experience_years": 5.0,
  "role": "Software Engineer",
  "location": "San Francisco, CA"
}"""

        user_prompt = f"Extract structured information from this text:\n\n{truncated}"

        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("OpenAI returned invalid JSON: %s — falling back", exc)
            return self._fallback.parse(text)

        if not isinstance(parsed, dict):
            logger.warning("OpenAI returned non-dict JSON — falling back")
            return self._fallback.parse(text)

        skills_raw = parsed.get("skills", [])
        if isinstance(skills_raw, list):
            skills = [str(s) for s in skills_raw if isinstance(s, (str, int, float))]
        else:
            skills = []

        experience_years = parsed.get("experience_years")
        if experience_years is not None:
            try:
                experience_years = float(experience_years)
            except (ValueError, TypeError):
                experience_years = None

        role = parsed.get("role")
        if role is not None:
            role = str(role)
        location = parsed.get("location")
        if location is not None:
            location = str(location)

        logger.debug(
            "OpenAI parsed: %d skills, %.1f yr exp, role=%s, loc=%s",
            len(skills),
            experience_years or 0,
            role,
            location,
        )

        return ParsedDocument(
            skills=skills,
            experience_years=experience_years,
            role=role,
            location=location,
        )
