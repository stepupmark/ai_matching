"""
OpenAI-powered matching service — the core evaluation engine.

Replaces the old hybrid-scoring + optional-rerank pipeline with a single
LLM-based evaluation that:
  1. Scores each candidate on role match (highest priority), experience,
     skills match, and interview score.
  2. Identifies "direct" (role matches JD) vs. "related" (different but
     related role) matches — e.g. a Fullstack Developer JD would also
     surface Frontend or Backend candidates as "related" matches.
  3. Returns per-candidate structured output including suggested role titles
     for related matches.

The LLM prompt instructs the model to weigh:
  - Role match:          highest priority
  - Experience fit:      second highest
  - Skills matched:      third
  - Interview score:     fourth (as a quality signal)

When OpenAI is unavailable, falls back to a simple skill-overlap score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.config import settings

logger = logging.getLogger(__name__)


# ── Data class ─────────────────────────────────────────────────────

@dataclass
class RerankerResult:
    """Output from a single candidate after OpenAI evaluation."""
    candidate_id: str
    score: float  # 0–100
    summary: str
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    ai_reason: Optional[str] = None
    match_type: str = "direct"  # "direct" or "related"
    suggested_roles: list[str] = field(default_factory=list)


# ── Protocol ───────────────────────────────────────────────────────

class Reranker(Protocol):
    """Interface all rerankers must implement."""

    def evaluate_matches(
        self,
        jd_text: str,
        jd_skills: list[str],
        jd_role: str,
        candidates: list[dict],
    ) -> list[RerankerResult]:
        ...


# ── Implementation ────────────────────────────────────────────────

class RerankerService:
    """
    Evaluates candidates against a job description using OpenAI.

    The LLM is instructed to prioritise:
      1. Role match (is the candidate's role aligned with the JD?)
      2. Experience fit (years of experience)
      3. Skills matched (overlap with required skills)
      4. Interview score (as a quality signal)

    Candidates whose role is different-but-related (e.g. Frontend Developer
    for a Fullstack Developer JD) are flagged with match_type="related"
    along with suggested role titles.

    Falls back to a simple skill-overlap score if the API key is missing
    or the call fails.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.openai_api_key
        self._model = settings.openai_model
        self._temperature = settings.openai_rerank_temperature
        self._max_tokens = settings.openai_rerank_max_tokens
        self._timeout = settings.openai_timeout_seconds
        self._client = None

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

    # ── Primary method ─────────────────────────────────────────────

    def evaluate_matches(
        self,
        jd_text: str,
        jd_skills: list[str],
        jd_role: str,
        candidates: list[dict],
    ) -> list[RerankerResult]:
        """
        Evaluate candidates against a job description using OpenAI.

        Args:
            jd_text: Full job description text.
            jd_skills: List of required skills from the JD.
            jd_role: The role/title extracted from the JD.
            candidates: List of candidate dicts with metadata.

        Returns:
            List of RerankerResult (sorted by score descending).
        """
        if not candidates:
            return []

        if not self._api_key:
            logger.warning("No OpenAI API key — using skill-overlap fallback scoring")
            return self._fallback(candidates)

        try:
            return self._call_llm(jd_text, jd_skills, jd_role, candidates)
        except Exception as exc:
            logger.error("OpenAI evaluation failed: %s — falling back", exc)
            return self._fallback(candidates)

    # ── LLM call ───────────────────────────────────────────────────

    def _call_llm(
        self,
        jd_text: str,
        jd_skills: list[str],
        jd_role: str,
        candidates: list[dict],
    ) -> list[RerankerResult]:
        """Call OpenAI to evaluate all candidates against the JD."""

        candidates_block = ""
        for i, c in enumerate(candidates):
            meta = c.get("metadata", {})
            candidates_block += f"""
Candidate {i + 1}:
- ID: {c['candidate_id']}
- Candidate Name: {meta.get('candidate_name', 'N/A')}
- Role Applied For: {meta.get('role_applied', 'N/A')}
- Skills: {', '.join(meta.get('skills', []))}
- Experience: {meta.get('experience_years', 'N/A')} years
- Interview Score: {meta.get('interview_score', 0)}/100
- Initial similarity score: {c.get('similarity', 0):.4f}
"""

        system_prompt = """You are an expert AI recruiter evaluating job-candidate matches.

For each candidate, produce an evaluation object with these fields:
- candidate_id: string (match the ID provided)
- score: integer 0-100 (overall match quality)
- summary: short 1-sentence summary of the match
- matched_skills: list of skills the candidate has that match the JD
- missing_skills: list of required skills the candidate lacks
- ai_reason: 1-2 sentence explanation of the score
- match_type: "direct" if the candidate's role directly aligns with the JD role, or "related" if the candidate's role is different but meaningfully related (e.g. Frontend Developer for a Fullstack Developer JD, or Software Engineer for a Senior Software Engineer JD)
- suggested_roles: if match_type is "related", provide 1-3 suggested role titles that would be a good fit for this candidate based on their profile; if match_type is "direct", leave this empty

SCORING PRIORITY — weight factors in this order:
1. Role match (highest priority) — does the candidate's applied role align with the JD role? Direct alignment = higher score. Related roles get a moderate score.
2. Experience fit (second priority) — how well does the candidate's years of experience match the JD requirement?
3. Skills matched (third priority) — what proportion of required skills does the candidate have?
4. Interview score (quality signal) — treat this as a bonus signal of candidate quality.

Output valid JSON as an object with a single key "candidates" whose value is an array of these evaluation objects, in the same order as the input candidates."""

        user_prompt = f"""Job Description:
{jd_text[:2500]}

Required Role: {jd_role or 'Not specified'}
Required Skills: {', '.join(jd_skills)}

Candidates to evaluate:
{candidates_block}

Return a JSON object with key "candidates" containing an array of evaluation objects, one per candidate, in the same order as the input. Each object must have: candidate_id, score (0-100), summary, matched_skills, missing_skills, ai_reason, match_type ("direct" or "related"), suggested_roles (list of strings)."""

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
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON — falling back")
            return self._fallback(candidates)

        # Safely extract the candidate list
        items: list = []
        if isinstance(parsed, dict):
            items = parsed.get("candidates", parsed.get("matches", []))
            if isinstance(items, dict):
                items = [items]
        elif isinstance(parsed, list):
            items = parsed
        else:
            logger.warning("Unexpected LLM response type: %s — falling back", type(parsed).__name__)
            return self._fallback(candidates)

        results: list[RerankerResult] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            ref = candidates[i] if i < len(candidates) else {}
            ref_meta = ref.get("metadata", {})

            # Determine match_type — default based on role comparison
            match_type = item.get("match_type", "direct")
            if match_type not in ("direct", "related"):
                match_type = "direct"

            suggested = item.get("suggested_roles", [])
            if not isinstance(suggested, list):
                suggested = []

            results.append(RerankerResult(
                candidate_id=item.get("candidate_id", ref.get("candidate_id", "")),
                score=float(item.get("score", ref.get("score", 0))),
                summary=item.get("summary", ref.get("summary", "")),
                matched_skills=item.get("matched_skills", ref.get("matched_skills", [])),
                missing_skills=item.get("missing_skills", ref.get("missing_skills", [])),
                ai_reason=item.get("ai_reason", None),
                match_type=match_type,
                suggested_roles=suggested,
            ))

        return results

    # ── Legacy rerank alias ────────────────────────────────────────

    def rerank(
        self,
        jd_text: str,
        jd_skills: list[str],
        candidates: list[dict],
    ) -> list[RerankerResult]:
        """
        Legacy rerank method — delegates to evaluate_matches.
        Kept for backward compatibility.
        """
        return self.evaluate_matches(
            jd_text=jd_text,
            jd_skills=jd_skills,
            jd_role="",
            candidates=candidates,
        )

    # ── Fallback ───────────────────────────────────────────────────

    @staticmethod
    def _fallback(candidates: list[dict]) -> list[RerankerResult]:
        """
        Fallback scoring based on similarity + skill overlap ratio.
        Used when OpenAI is unavailable.

        Score = similarity_norm * 50 + skill_ratio * 50
        - similarity_norm: cosine similarity normalized to [0, 1]
        - skill_ratio: matched / (matched + missing)
        """
        results: list[RerankerResult] = []
        for c in candidates:
            sim = c.get("similarity", 0)
            sim_norm = (sim + 1.0) / 2.0  # [-1, 1] → [0, 1]

            matched = len(c.get("matched_skills", []))
            missing = len(c.get("missing_skills", []))
            total_skills = matched + missing
            skill_ratio = matched / max(total_skills, 1)

            score = (sim_norm * 50.0) + (skill_ratio * 50.0)

            # Build a basic summary
            summary_parts = []
            if matched:
                summary_parts.append(f"{matched} skills matched")
            if missing:
                summary_parts.append(f"{missing} missing")
            summary = ", ".join(summary_parts) if summary_parts else "No skills matched"

            results.append(RerankerResult(
                candidate_id=c["candidate_id"],
                score=round(min(score, 100.0), 1),
                summary=summary,
                matched_skills=c.get("matched_skills", []),
                missing_skills=c.get("missing_skills", []),
                ai_reason=None,
                match_type="direct",
                suggested_roles=[],
            ))
        return results


def create_reranker(api_key: Optional[str] = None) -> RerankerService:
    """Factory: returns the appropriate reranker based on configuration."""
    return RerankerService(api_key=api_key)
