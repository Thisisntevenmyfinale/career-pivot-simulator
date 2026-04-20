"""
Persistence Layer
=================
File-based profile persistence so PivotOS remembers everything across sessions.

Saves to pivot_profile.json in the project directory.
Loads on app startup, saves on demand (call save_profile()).

Persisted state:
  - cv_text, cv_profile
  - pivot_dna, voice_profile, cohort_intelligence
  - pipeline_jobs (list of dicts)
  - outcome_tracker data
  - calibration_data (personal ROI model)
  - momentum data (streak, journal, last_date)
  - mock_interview_report
  - interview_evals, interview_questions
  - zwilling_messages
  - roi_results
  - skill_proofs
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

PROFILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pivot_profile.json")

# Keys from session_state that we persist
PERSISTENT_KEYS = [
    "cv_text",
    "cv_profile",
    "onet_match",
    "skill_gap_results",
    "pivot_dna",
    "voice_profile",
    "cohort_intelligence",
    "cohort_pivot_key",
    "pipeline_jobs",
    "quality_log",
    "outcome_log",
    "calibration_data",
    "momentum_streak_days",
    "momentum_last_date",
    "momentum_journal",
    "mock_interview_report",
    "interview_questions",
    "interview_answers",
    "interview_evals",
    "interview_prep_done",
    "roi_results",
    "skill_proofs",
    "hm_dossier",
    "hm_dossier_name",
    "zwilling_messages",
    "zwilling_initialized",
    "advisor_result",
    "war_room_result",
    "war_room_company",
    "daily_brief_date",
    "daily_brief_content",
]


def _serialize(obj: Any) -> Any:
    """Make objects JSON-serializable."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return str(obj)


def save_profile(state: Any) -> bool:
    """
    Save current session state to disk.
    state should be st.session_state (dict-like).
    Returns True on success.
    """
    try:
        data: Dict[str, Any] = {
            "_saved_at": datetime.now().isoformat(),
            "_version": 4,
        }
        for key in PERSISTENT_KEYS:
            val = state.get(key)
            if val is not None and val != "" and val != [] and val != {}:
                try:
                    # Test if serializable
                    json.dumps(val, default=_serialize)
                    data[key] = val
                except Exception:
                    # Try converting to dict
                    try:
                        data[key] = json.loads(json.dumps(val, default=_serialize))
                    except Exception:
                        pass  # Skip non-serializable values

        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_serialize)
        return True
    except Exception:
        return False


def load_profile() -> Optional[Dict[str, Any]]:
    """
    Load saved profile from disk.
    Returns dict of key→value, or None if no profile exists.
    """
    if not os.path.exists(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove metadata keys
        data.pop("_saved_at", None)
        data.pop("_version", None)
        return data
    except Exception:
        return None


def profile_exists() -> bool:
    return os.path.exists(PROFILE_PATH)


def delete_profile() -> bool:
    try:
        if os.path.exists(PROFILE_PATH):
            os.remove(PROFILE_PATH)
        return True
    except Exception:
        return False


def get_profile_meta() -> Dict[str, Any]:
    """Return metadata about the saved profile (saved_at, size)."""
    if not os.path.exists(PROFILE_PATH):
        return {}
    try:
        size = os.path.getsize(PROFILE_PATH)
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "saved_at": data.get("_saved_at", "unknown"),
            "size_kb": round(size / 1024, 1),
            "has_cv": bool(data.get("cv_text")),
            "has_dna": bool(data.get("pivot_dna")),
            "pipeline_count": len(data.get("pipeline_jobs", [])),
            "outcome_count": len(data.get("outcome_log", [])),
        }
    except Exception:
        return {}
