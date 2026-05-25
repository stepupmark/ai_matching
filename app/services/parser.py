"""
Resume and Job Description text parser.

Extracts structured attributes (skills, years of experience, role, location)
from raw unstructured text using rule-based heuristics.

Skills extraction uses word-boundary regex matching to avoid false positives
(e.g., "go" should not match "going" or "goal").
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.services.skills_base import SKILLS_SET, LOCATIONS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

# Role/job title keywords (lowercase)
_ROLE_KEYWORDS = [
    "software engineer", "senior software engineer", "staff software engineer",
    "principal engineer", "lead engineer", "full stack developer",
    "frontend developer", "backend developer", "front-end developer",
    "back-end developer", "web developer", "devops engineer", "sre",
    "site reliability engineer", "data scientist", "data engineer",
    "machine learning engineer", "ml engineer", "ai engineer",
    "research scientist", "product manager", "engineering manager",
    "tech lead", "architect", "solution architect", "cloud architect",
    "security engineer", "qa engineer", "test engineer",
    "systems engineer", "network engineer", "database administrator",
    "data analyst", "business analyst", "scrum master",
    "project manager", "program manager", "technical writer",
    "ux designer", "ui designer", "product designer",
    "intern", "junior developer", "entry level",
]

# Experience patterns
_EXP_PATTERNS = [
    re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)", re.IGNORECASE),
    re.compile(r"(?:experience|exp)\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(\d+)\+?\s*\+\s*years?", re.IGNORECASE),
]

# Pre-compile word-boundary regex patterns for all skills
_SKILL_PATTERNS: list[tuple[str, re.Pattern]] = sorted(
    [(s, re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)) for s in SKILLS_SET],
    key=lambda x: len(x[0]),
    reverse=True,
)


# ── Public API ─────────────────────────────────────────────────────

class ParsedDocument:
    """Structured output from parsing a resume or job description."""

    def __init__(
        self,
        skills: list[str],
        experience_years: Optional[float],
        role: Optional[str],
        location: Optional[str],
    ) -> None:
        self.skills = skills
        self.experience_years = experience_years
        self.role = role
        self.location = location

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"skills={self.skills!r}, "
            f"exp={self.experience_years!r}, "
            f"role={self.role!r}, "
            f"loc={self.location!r})"
        )


class ResumeJDParser:
    """Parser for extracting structured attributes from resume / JD text."""

    def parse(self, text: str) -> ParsedDocument:
        """Parse full text and return structured attributes."""
        skills = self._extract_skills(text)
        experience_years = self._extract_experience(text)
        role = self._extract_role(text)
        location = self._extract_location(text)

        logger.debug(
            "Parsed: %d skills, %.1f yr exp, role=%s, loc=%s",
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

    # ── Skills extraction ──────────────────────────────────────────

    @staticmethod
    def _extract_skills(text: str) -> list[str]:
        """
        Extract known skills using word-boundary regex matching.

        Using \\b boundaries prevents false positives where a skill name
        appears as part of another word (e.g., "go" won't match "going").
        Multi-word skills are matched first (longest-first ordering).
        """
        found: list[str] = []
        seen_positions: list[tuple[int, int]] = []

        for skill_name, pattern in _SKILL_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                # Skip if this match overlaps with a previously matched (longer) skill
                overlapping = any(
                    not (end <= s or start >= e)
                    for s, e in seen_positions
                )
                if not overlapping:
                    found.append(skill_name)
                    seen_positions.append((start, end))
                    break  # one match per skill is enough

        return found

    # ── Experience extraction ──────────────────────────────────────

    @staticmethod
    def _extract_experience(text: str) -> Optional[float]:
        """Extract years of experience from text."""
        for pattern in _EXP_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except (IndexError, ValueError):
                    continue
        return None

    # ── Role extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_role(text: str) -> Optional[str]:
        """Extract the most likely job title/role from text."""
        text_lower = text.lower()
        lines = text_lower.split("\n")
        for line in lines[:30]:  # scan first 30 lines
            line = line.strip()
            for role in _ROLE_KEYWORDS:
                idx = line.find(role)
                if idx != -1:
                    # Return the original-case version from the text
                    orig_idx = text.lower().find(role)
                    if orig_idx != -1:
                        return text[orig_idx: orig_idx + len(role)]
                    return role.title()
        return None

    # ── Location extraction ────────────────────────────────────────

    @staticmethod
    def _extract_location(text: str) -> Optional[str]:
        """Extract location from text using word-boundary matching."""
        text_lower = text.lower()
        for loc in sorted(LOCATIONS, key=len, reverse=True):
            pattern = re.compile(r"\b" + re.escape(loc) + r"\b", re.IGNORECASE)
            if pattern.search(text_lower):
                return loc.title()
        return None
