"""
Per-JD Offer Predictor
======================
Answers the #1 question every career changer has: "Should I apply to THIS job?"

Not a keyword matcher (that's Jobscan). This is an outcome predictor:
P(offer | this JD, this candidate profile, personal calibration).

Three-layer analysis:
  Layer 1: Skill extraction from JD — required vs. nice-to-have, ATS keywords
  Layer 2: Profile-JD matching — cosine-style weighted overlap against cv_profile + skill_gap
  Layer 3: Calibrated prediction — applies personal response rate + pipeline context

Output is a Go / Borderline / No-Go decision with full reasoning,
not a vague "good fit" score. The point is to save the candidate from
wasting effort on applications with < 20% chance — and focus energy
on the 60%+ opportunities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# JD Extraction (Layer 1)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """You are a talent acquisition expert analyzing a job description.
Extract structured requirements. Be precise — only extract what's explicitly stated.

Output JSON only:
{
  "role_title": "exact title from JD",
  "seniority": "junior|mid|senior|lead|director",
  "required_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "years_experience_min": <int or null>,
  "years_experience_max": <int or null>,
  "must_have_pm_title": <bool>,
  "culture_signals": ["data-driven", "fast-paced"],
  "red_flags": ["requires CS degree", "10+ years PM required"],
  "ats_keywords": ["top 8 keywords a resume must contain to pass ATS"],
  "one_line_summary": "What this role actually does in plain English"
}"""


def _extract_jd(client: Any, jd_text: str) -> Optional[Dict]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": f"Job description:\n\n{jd_text[:4000]}"},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction (Layer 2 + 3)
# ─────────────────────────────────────────────────────────────────────────────

_PREDICT_SYSTEM = """You are a brutal, data-driven hiring prediction model.
You have: the candidate's profile, the extracted JD requirements, and personal calibration data.
Predict P(offer) — realistic, not encouraging. Be specific about what will and won't work.

Output JSON only:
{
  "fit_score": <int 0-100>,
  "offer_probability": <int 0-100>,
  "ats_risk": "low|medium|high",
  "seniority_match": "under|match|over",
  "go_no_go": "go|borderline|no_go",
  "go_no_go_reason": "One sentence. Blunt.",
  "required_skills_found": ["skills candidate clearly has"],
  "required_skills_missing": ["skills JD requires, candidate lacks"],
  "ats_keywords_present": ["keywords in candidate profile"],
  "ats_keywords_absent": ["keywords missing — must add to resume"],
  "top_strengths": ["2-3 strongest selling points for THIS role"],
  "top_risks": ["2-3 most likely rejection reasons for THIS role"],
  "pivot_credibility": "strong|moderate|weak",
  "pivot_credibility_reason": "Why a HM would/wouldn't believe the pivot for this specific role"
}"""


def _predict(client: Any, jd_data: Dict, cv_profile: Dict, pivot_dna: Dict,
             skill_gap: Dict, calibration: Dict) -> Optional[Dict]:
    candidate_summary = {
        "role": cv_profile.get("extracted_role", ""),
        "years_experience": cv_profile.get("years_experience", 0),
        "top_skills": cv_profile.get("top_skills", [])[:15],
        "key_achievements": cv_profile.get("key_achievements", [])[:3],
        "education": cv_profile.get("education", ""),
        "pivot_hook": pivot_dna.get("pivot_hook", ""),
        "strongest_argument": pivot_dna.get("strongest_transferable_argument", ""),
        "personal_response_rate": calibration.get("personal_response_rate", 0.15) if calibration else 0.15,
        "dominant_rejection_stage": calibration.get("dominant_rejection_stage", "") if calibration else "",
        "skill_gaps": [g.get("skill") for g in (skill_gap.get("gaps") or [])[:5]],
        "fit_percentile": skill_gap.get("fit_percentile", 50),
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.15,
            messages=[
                {"role": "system", "content": _PREDICT_SYSTEM},
                {"role": "user", "content": (
                    f"JD requirements:\n{json.dumps(jd_data, indent=2)}\n\n"
                    f"Candidate profile:\n{json.dumps(candidate_summary, indent=2)}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_jd(
    oai_key: str,
    *,
    jd_text: str,
    cv_profile: Dict,
    pivot_dna: Dict,
    skill_gap_results: Dict,
    calibration_data: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Full per-JD analysis. Returns prediction dict or None on failure.

    Keys:
      jd_data          — extracted JD requirements
      fit_score        — 0-100 profile-JD overlap
      offer_probability — 0-100 calibrated P(offer)
      go_no_go         — "go" / "borderline" / "no_go"
      go_no_go_reason  — one blunt sentence
      ats_risk         — "low" / "medium" / "high"
      seniority_match  — "under" / "match" / "over"
      required_skills_found / missing
      ats_keywords_absent  — must add to resume NOW
      top_strengths / top_risks
      pivot_credibility + reason
    """
    if not oai_key or not jd_text.strip():
        return None
    if not cv_profile:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    jd_data = _extract_jd(client, jd_text)
    if not jd_data:
        return None

    prediction = _predict(
        client, jd_data,
        cv_profile   = cv_profile,
        pivot_dna    = pivot_dna or {},
        skill_gap    = skill_gap_results or {},
        calibration  = calibration_data or {},
    )
    if not prediction:
        return None

    return {**jd_data, **prediction}


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def go_no_go_color(verdict: str) -> str:
    return {"go": "#057642", "borderline": "#D97706", "no_go": "#DC2626"}.get(verdict, "#555")


def go_no_go_label(verdict: str) -> str:
    return {"go": "GO — Apply now", "borderline": "BORDERLINE — Fix these issues first",
            "no_go": "NO-GO — Probability too low"}.get(verdict, verdict.upper())


def offer_prob_color(p: int) -> str:
    if p >= 60: return "#057642"
    if p >= 35: return "#D97706"
    return "#DC2626"
