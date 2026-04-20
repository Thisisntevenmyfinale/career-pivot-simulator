"""
SQLite Persistence Layer
========================
Replaces the JSON flat-file persistence with a proper relational database.

Why SQLite over JSON:
  - Atomic writes — no partial file corruption on crash
  - Structured pipeline and outcome tables — proper relational schema
  - Concurrent read safety (WAL mode)
  - Query capability: "show all applications with status=rejected"
  - Metadata: timestamps, row counts, last-modified per key
  - Standard Python stdlib — no external dependencies

Architecture:
  Database:  pivot_os.db (SQLite, WAL mode)
  Tables:
    kv_store      — all serialised session state (key TEXT, value TEXT/JSON)
    pipeline_jobs — structured job pipeline with full schema
    outcome_log   — structured outcome history with full schema
    audit_log     — every save event with timestamp + byte size

  The kv_store handles complex nested objects (pivot_dna, cv_profile, etc.)
  as JSON blobs, while pipeline_jobs and outcome_log get proper columns
  so they're queryable and inspectable without deserialisation.

Backwards compatibility:
  - If pivot_profile.json exists, it is migrated to SQLite on first load
  - The JSON file is kept as a backup after migration
  - save_profile() and load_profile() maintain the exact same interface
    as persistence.py — zero changes needed in app.py callers

Drop-in replacement: import from this module instead of persistence.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pivot_os.db")
JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pivot_profile.json")

_VERSION = 5  # DB schema version

# Keys stored in kv_store (complex objects)
KV_KEYS = [
    "cv_text", "cv_profile", "onet_match", "skill_gap_results",
    "pivot_dna", "voice_profile", "cohort_intelligence", "cohort_pivot_key",
    "quality_log", "calibration_data",
    "momentum_streak_days", "momentum_last_date", "momentum_journal",
    "mock_interview_report", "interview_questions", "interview_answers",
    "interview_evals", "interview_prep_done", "roi_results", "skill_proofs",
    "hm_dossier", "hm_dossier_name", "zwilling_messages", "zwilling_initialized",
    "advisor_result", "war_room_result", "war_room_company",
    "daily_brief_date", "daily_brief_content",
    "ops_previous", "writing_memory", "rejection_interpretations",
    "rag_corpus_hash",
    # Note: rag_embeddings excluded — too large, rebuild each session
]


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version  INTEGER PRIMARY KEY,
    applied  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv_store (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    company     TEXT,
    status      TEXT,
    date_added  TEXT,
    date_updated TEXT,
    source      TEXT,
    cover_letter TEXT,
    notes       TEXT,
    raw_json    TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcome_log (
    id              TEXT,
    job_title       TEXT,
    company         TEXT,
    actual_stage    TEXT,
    reached_response INTEGER,
    reached_interview INTEGER,
    is_offer        INTEGER,
    predicted_roi   REAL,
    notes           TEXT,
    date            TEXT,
    raw_json        TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (id, date)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event       TEXT NOT NULL,
    rows_saved  INTEGER,
    size_bytes  INTEGER,
    ts          TEXT NOT NULL
);
"""


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    with _db() as conn:
        conn.executescript(_DDL)
        # Check version
        row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
        if row is None or row[0] < _VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied) VALUES (?, ?)",
                (_VERSION, datetime.now().isoformat()),
            )


def db_exists() -> bool:
    return os.path.exists(DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(state: Any) -> bool:
    """
    Save session state to SQLite.
    Structured tables for pipeline/outcomes, kv_store for everything else.
    """
    try:
        _init_db()
        now = datetime.now().isoformat()

        with _db() as conn:
            # ── kv_store ─────────────────────────────────────────────────
            for key in KV_KEYS:
                val = state.get(key)
                if val is None or val == "" or val == [] or val == {}:
                    continue
                try:
                    serialised = json.dumps(val, default=str)
                    conn.execute(
                        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                        (key, serialised, now),
                    )
                except Exception:
                    pass

            # ── pipeline_jobs ─────────────────────────────────────────────
            pipeline = state.get("pipeline_jobs") or []
            if pipeline:
                conn.execute("DELETE FROM pipeline_jobs")
                for job in pipeline:
                    if not isinstance(job, dict):
                        continue
                    conn.execute(
                        """INSERT OR REPLACE INTO pipeline_jobs
                           (id, title, company, status, date_added, date_updated,
                            source, cover_letter, notes, raw_json, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(job.get("id", "")),
                            str(job.get("title", "")),
                            str(job.get("company", "")),
                            str(job.get("status", "")),
                            str(job.get("date_added", "")),
                            str(job.get("date_updated", "")),
                            str(job.get("source", "")),
                            str(job.get("cover_letter", ""))[:2000],
                            str(job.get("notes", "")),
                            json.dumps(job, default=str),
                            now,
                        ),
                    )

            # ── outcome_log ───────────────────────────────────────────────
            outcomes = state.get("outcome_log") or []
            if outcomes:
                conn.execute("DELETE FROM outcome_log")
                for o in outcomes:
                    if not isinstance(o, dict):
                        continue
                    conn.execute(
                        """INSERT OR REPLACE INTO outcome_log
                           (id, job_title, company, actual_stage, reached_response,
                            reached_interview, is_offer, predicted_roi, notes,
                            date, raw_json, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(o.get("id", "")),
                            str(o.get("job_title", "")),
                            str(o.get("company", "")),
                            str(o.get("actual_stage", "")),
                            int(bool(o.get("reached_response"))),
                            int(bool(o.get("reached_interview"))),
                            int(bool(o.get("is_offer"))),
                            float(o.get("predicted_roi") or 0),
                            str(o.get("notes", "")),
                            str(o.get("date", "")),
                            json.dumps(o, default=str),
                            now,
                        ),
                    )

            # ── audit_log ─────────────────────────────────────────────────
            db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
            conn.execute(
                "INSERT INTO audit_log (event, rows_saved, size_bytes, ts) VALUES (?, ?, ?, ?)",
                ("save_profile", len(pipeline) + len(outcomes), db_size, now),
            )

        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_profile() -> Optional[Dict[str, Any]]:
    """
    Load saved profile from SQLite.
    Returns dict of key→value, or None if no DB exists.
    Falls back to JSON migration if only JSON file exists.
    """
    # Auto-migrate from JSON if DB doesn't exist yet
    if not db_exists() and os.path.exists(JSON_PATH):
        _migrate_from_json()

    if not db_exists():
        return None

    try:
        _init_db()
        data: Dict[str, Any] = {}

        with _db() as conn:
            # Load kv_store
            rows = conn.execute("SELECT key, value FROM kv_store").fetchall()
            for row in rows:
                try:
                    data[row["key"]] = json.loads(row["value"])
                except Exception:
                    data[row["key"]] = row["value"]

            # Load pipeline_jobs from structured table
            jobs_rows = conn.execute(
                "SELECT raw_json FROM pipeline_jobs ORDER BY date_added"
            ).fetchall()
            if jobs_rows:
                data["pipeline_jobs"] = [json.loads(r["raw_json"]) for r in jobs_rows]

            # Load outcome_log from structured table
            outcome_rows = conn.execute(
                "SELECT raw_json FROM outcome_log ORDER BY date"
            ).fetchall()
            if outcome_rows:
                data["outcome_log"] = [json.loads(r["raw_json"]) for r in outcome_rows]

        return data if data else None
    except Exception:
        return None


def profile_exists() -> bool:
    return db_exists() or os.path.exists(JSON_PATH)


def delete_profile() -> bool:
    try:
        if db_exists():
            os.remove(DB_PATH)
        return True
    except Exception:
        return False


def get_profile_meta() -> Dict[str, Any]:
    """Return metadata about saved profile."""
    if not db_exists():
        # Fall back to JSON meta
        if os.path.exists(JSON_PATH):
            size = os.path.getsize(JSON_PATH)
            return {"saved_at": "unknown", "size_kb": round(size / 1024, 1),
                    "storage": "json (legacy)", "has_cv": False, "has_dna": False,
                    "pipeline_count": 0, "outcome_count": 0}
        return {}
    try:
        size = os.path.getsize(DB_PATH)
        with _db() as conn:
            saved_at = (conn.execute(
                "SELECT ts FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone() or {}).get("ts", "unknown")

            has_cv = bool(conn.execute(
                "SELECT 1 FROM kv_store WHERE key='cv_text' LIMIT 1"
            ).fetchone())
            has_dna = bool(conn.execute(
                "SELECT 1 FROM kv_store WHERE key='pivot_dna' LIMIT 1"
            ).fetchone())
            pipeline_count = (conn.execute(
                "SELECT COUNT(*) as c FROM pipeline_jobs"
            ).fetchone() or {}).get("c", 0)
            outcome_count = (conn.execute(
                "SELECT COUNT(*) as c FROM outcome_log"
            ).fetchone() or {}).get("c", 0)

        return {
            "saved_at":      saved_at,
            "size_kb":       round(size / 1024, 1),
            "storage":       "SQLite (WAL)",
            "has_cv":        has_cv,
            "has_dna":       has_dna,
            "pipeline_count": pipeline_count,
            "outcome_count":  outcome_count,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline queries (new capability not possible with JSON)
# ─────────────────────────────────────────────────────────────────────────────

def query_pipeline(
    status: Optional[str] = None,
    company: Optional[str] = None,
    since_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query pipeline jobs with filters. Only possible with SQLite.
    Returns list of job dicts matching all provided filters.
    """
    if not db_exists():
        return []
    try:
        _init_db()
        clauses = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if company:
            clauses.append("company LIKE ?")
            params.append(f"%{company}%")
        if since_days:
            from datetime import date, timedelta
            cutoff = (date.today() - timedelta(days=since_days)).isoformat()
            clauses.append("date_added >= ?")
            params.append(cutoff)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with _db() as conn:
            rows = conn.execute(
                f"SELECT raw_json FROM pipeline_jobs {where} ORDER BY date_added DESC",
                params,
            ).fetchall()
        return [json.loads(r["raw_json"]) for r in rows]
    except Exception:
        return []


def get_rejection_funnel() -> Dict[str, int]:
    """Return stage counts from outcome_log — queryable because it's structured."""
    if not db_exists():
        return {}
    try:
        _init_db()
        with _db() as conn:
            rows = conn.execute(
                "SELECT actual_stage, COUNT(*) as cnt FROM outcome_log GROUP BY actual_stage"
            ).fetchall()
        return {r["actual_stage"]: r["cnt"] for r in rows}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# JSON → SQLite migration
# ─────────────────────────────────────────────────────────────────────────────

def _migrate_from_json() -> bool:
    """
    One-time migration: read pivot_profile.json → write to SQLite.
    Keeps the JSON file as pivot_profile.json.bak for safety.
    """
    if not os.path.exists(JSON_PATH):
        return False
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        old_data.pop("_saved_at", None)
        old_data.pop("_version", None)

        # Build a mock state dict and call save_profile
        save_profile(old_data)

        # Rename original as backup
        bak = JSON_PATH + ".bak"
        if not os.path.exists(bak):
            os.rename(JSON_PATH, bak)
        return True
    except Exception:
        return False
