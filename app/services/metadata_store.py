"""
SQLite-based metadata store for resumes and job descriptions.

Stores parsed attributes (skills, experience, role, location) alongside
embedding model version and tenant scoping information.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class MetadataStore:
    """Thread-safe SQLite metadata store."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = Path(db_path or settings.resolved_metadata_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── Connection management ──────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id        TEXT PRIMARY KEY,
                    tenant_id           TEXT NOT NULL DEFAULT 'default',
                    candidate_name      TEXT NOT NULL,
                    role_applied        TEXT NOT NULL,
                    skills              TEXT NOT NULL DEFAULT '[]',
                    interview_score     REAL NOT NULL DEFAULT 0,
                    experience_years    REAL NOT NULL DEFAULT 0,
                    generated_profile   TEXT NOT NULL DEFAULT '',
                    embedding_model_version TEXT NOT NULL DEFAULT '',
                    active              INTEGER NOT NULL DEFAULT 1,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_descriptions (
                    jd_id           TEXT PRIMARY KEY,
                    tenant_id       TEXT NOT NULL DEFAULT 'default',
                    parsed_text     TEXT NOT NULL,
                    skills          TEXT NOT NULL DEFAULT '[]',
                    experience_years REAL,
                    role            TEXT,
                    location        TEXT,
                    embedding_model_version TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_tenant ON candidates(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_candidates_active ON candidates(active);
                CREATE INDEX IF NOT EXISTS idx_jds_tenant ON job_descriptions(tenant_id);
            """)
            self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Candidate operations ───────────────────────────────────────

    def upsert_candidate(
        self,
        candidate_id: str,
        tenant_id: str,
        candidate_name: str,
        role_applied: str,
        skills: list[str],
        interview_score: float,
        experience_years: float,
        generated_profile: str,
        embedding_model_version: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO candidates
                    (candidate_id, tenant_id, candidate_name, role_applied, skills,
                     interview_score, experience_years, generated_profile,
                     embedding_model_version, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    candidate_name          = excluded.candidate_name,
                    role_applied            = excluded.role_applied,
                    skills                  = excluded.skills,
                    interview_score         = excluded.interview_score,
                    experience_years        = excluded.experience_years,
                    generated_profile       = excluded.generated_profile,
                    embedding_model_version = excluded.embedding_model_version,
                    active                  = 1,
                    updated_at              = excluded.updated_at
                """,
                (
                    candidate_id,
                    tenant_id,
                    candidate_name,
                    role_applied,
                    json.dumps(skills),
                    interview_score,
                    experience_years,
                    generated_profile,
                    embedding_model_version,
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def get_candidate(self, candidate_id: str) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ? AND active = 1", (candidate_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_active_candidate_ids(self, tenant_id: Optional[str] = None) -> list[str]:
        query = "SELECT candidate_id FROM candidates WHERE active = 1"
        params: list = []
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [r["candidate_id"] for r in rows]

    def deactivate_candidate(self, candidate_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE candidates SET active = 0, updated_at = ? WHERE candidate_id = ?",
                (datetime.utcnow().isoformat(), candidate_id),
            )
            self.conn.commit()

    # ── JD operations ──────────────────────────────────────────────

    def upsert_jd(
        self,
        jd_id: str,
        tenant_id: str,
        parsed_text: str,
        skills: list[str],
        experience_years: Optional[float],
        role: Optional[str],
        location: Optional[str],
        embedding_model_version: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO job_descriptions
                    (jd_id, tenant_id, parsed_text, skills, experience_years,
                     role, location, embedding_model_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(jd_id) DO UPDATE SET
                    parsed_text             = excluded.parsed_text,
                    skills                  = excluded.skills,
                    experience_years        = excluded.experience_years,
                    role                    = excluded.role,
                    location                = excluded.location,
                    embedding_model_version = excluded.embedding_model_version,
                    updated_at              = excluded.updated_at
                """,
                (
                    jd_id,
                    tenant_id,
                    parsed_text,
                    json.dumps(skills),
                    experience_years,
                    role,
                    location,
                    embedding_model_version,
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def get_jd(self, jd_id: str) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM job_descriptions WHERE jd_id = ?", (jd_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse JSON fields
        for field in ("skills",):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    @property
    def candidate_count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM candidates WHERE active = 1"
            ).fetchone()
        return row["cnt"] if row else 0

    @property
    def jd_count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM job_descriptions"
            ).fetchone()
        return row["cnt"] if row else 0
