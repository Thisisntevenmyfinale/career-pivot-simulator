"""
Job Posting Analyzer
=====================
Extracts required skills from a real job posting and computes a personalised
match score — comparing what the job actually demands against:
  1. The user's personal CV skill vector (if available)
  2. Their current O*NET role profile (fallback)

This bridges the gap between abstract career analysis and a specific
application decision: "Should I apply to THIS job?"

Architecture
------------
Pass 1 — LLM extracts structured job metadata and required skills
Pass 2 — maps extracted skills onto the O*NET skill space (same approach as cv_parser)
Pass 3 — computes match score, advantage/gap breakdown, application readiness verdict
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _offline_job_analysis(
    job_text: str,
    skill_columns: List[str],
) -> Dict[str, Any]:
    """Keyword-based fallback when OpenAI unavailable."""
    text_lower = job_text.lower()
    found_skills = [s for s in skill_columns if s.lower() in text_lower][:15]
    return {
        "role_title": "Unknown (offline mode)",
        "company": "Unknown",
        "experience_years_required": None,
        "job_skill_vector": pd.Series(0.0, index=skill_columns),
        "required_skills_raw": found_skills,
        "nice_to_have_skills": [],
        "top_matches": [],
        "top_gaps": [],
        "match_score": 0.0,
        "application_readiness": "Unknown",
        "readiness_rationale": "Offline mode — LLM not available.",
        "key_insights": ["Connect to OpenAI to get full analysis."],
        "source": "offline",
    }


def analyze_job_posting(
    job_text: str,
    skill_columns: List[str],
    matrix: pd.DataFrame,
    current_role: str,
    target_role: str,
    cv_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse a job posting and compute match against user profile.

    Returns a dict with:
      role_title, company, job_skill_vector (pd.Series),
      required_skills_raw, top_matches, top_gaps,
      match_score (0-100), application_readiness, key_insights
    """
    job_text = (job_text or "").strip()
    if not job_text:
        return {"error": "No job posting text provided.", "source": "empty"}

    if not prefer_online:
        return _offline_job_analysis(job_text, skill_columns)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_job_analysis(job_text, skill_columns)

    # ── Pass 1: Extract job metadata ──────────────────────────────────────────
    pass1_prompt = f"""You are a job requirements analyst. Extract structured information from this job posting.

Respond ONLY with valid JSON:
{{
  "role_title": "exact job title",
  "company": "company name or 'Not specified'",
  "experience_years_required": 2,
  "education_required": "Bachelor's / Master's / PhD / Not specified",
  "required_skills": [
    {{"skill": "...", "importance": "must-have" | "preferred", "evidence": "brief quote from posting"}}
  ],
  "key_responsibilities": ["...", "..."],
  "domain": "tech / finance / healthcare / marketing / etc."
}}

JOB POSTING:
{job_text[:3500]}
"""

    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": pass1_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        meta = json.loads(r1.choices[0].message.content or "{}")
    except Exception as e:
        return {**_offline_job_analysis(job_text, skill_columns), "error": repr(e)}

    role_title: str = meta.get("role_title", "Unknown")
    company: str = meta.get("company", "Unknown")
    required_skills_raw: List[str] = [s["skill"] for s in meta.get("required_skills", [])]

    # ── Pass 2: Map to O*NET skill space ──────────────────────────────────────
    BATCH_SIZE = 60
    onet_scores: Dict[str, float] = {}

    skill_context = "\n".join([
        f"- {s['skill']} ({s.get('importance','required')}): {s.get('evidence','')}"
        for s in meta.get("required_skills", [])[:30]
    ])

    for batch_start in range(0, len(skill_columns), BATCH_SIZE):
        batch = skill_columns[batch_start: batch_start + BATCH_SIZE]
        batch_str = "\n".join([f"- {s}" for s in batch])

        pass2_prompt = f"""Map job posting requirements to O*NET skill dimensions.

JOB: {role_title} at {company}

SKILLS REQUIRED BY JOB:
{skill_context}

Rate each O*NET skill dimension on 0–7 scale based on how much the job REQUIRES it:
0 = Not required / not mentioned
1–2 = Minor / peripheral
3–4 = Regularly needed
5–6 = Core requirement
7 = Essential / primary skill

O*NET DIMENSIONS:
{batch_str}

Respond ONLY with JSON: {{"ratings": {{"<skill>": <0-7>, ...}}}}
"""
        try:
            r2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": pass2_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=800,
            )
            batch_result = json.loads(r2.choices[0].message.content or "{}")
            for sk, val in batch_result.get("ratings", {}).items():
                if sk not in onet_scores:
                    try:
                        onet_scores[sk] = float(np.clip(float(val), 0.0, 7.0))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            continue

    job_vec = pd.Series(0.0, index=skill_columns, dtype=float)
    for sk in skill_columns:
        if sk in onet_scores:
            job_vec[sk] = onet_scores[sk]

    # ── Pass 3: Compute match score ───────────────────────────────────────────
    # Pick the "user" vector: personal CV > current role O*NET
    user_vec: Optional[pd.Series] = None
    user_label = "current role (O*NET)"

    if cv_profile and "skill_vector" in cv_profile and cv_profile["skill_vector"] is not None:
        user_vec = cv_profile["skill_vector"].reindex(skill_columns, fill_value=0.0)
        user_label = f"your CV ({cv_profile.get('extracted_role', 'uploaded profile')})"
    elif current_role in matrix.index:
        user_vec = matrix.loc[current_role].reindex(skill_columns, fill_value=0.0)

    match_score = 0.0
    top_matches: List[Dict] = []
    top_gaps: List[Dict] = []

    if user_vec is not None and job_vec.sum() > 0:
        # Cosine similarity
        u = user_vec.values.astype(float)
        j = job_vec.values.astype(float)
        norm_u = np.linalg.norm(u)
        norm_j = np.linalg.norm(j)
        if norm_u > 0 and norm_j > 0:
            match_score = float(np.clip(np.dot(u, j) / (norm_u * norm_j) * 100, 0, 100))

        # Top matches: skills the job wants AND you have
        overlap = pd.Series(
            np.minimum(user_vec.values, job_vec.values),
            index=skill_columns
        )
        top_matches_idx = overlap[overlap > 1.0].sort_values(ascending=False).head(6).index
        for sk in top_matches_idx:
            top_matches.append({
                "skill": sk,
                "your_level": round(float(user_vec[sk]), 2),
                "job_requires": round(float(job_vec[sk]), 2),
            })

        # Top gaps: skills the job wants that you lack
        gap = (job_vec - user_vec).clip(lower=0)
        top_gaps_idx = gap[gap > 1.0].sort_values(ascending=False).head(6).index
        for sk in top_gaps_idx:
            top_gaps.append({
                "skill": sk,
                "your_level": round(float(user_vec[sk]), 2),
                "job_requires": round(float(job_vec[sk]), 2),
                "gap": round(float(gap[sk]), 2),
            })

    # Application readiness
    if match_score >= 72:
        readiness = "Strong"
        readiness_rationale = "Your profile closely matches what this job requires. Apply now."
    elif match_score >= 52:
        readiness = "Moderate"
        readiness_rationale = "Solid overlap with manageable gaps. A strong application with targeted framing is viable."
    elif match_score >= 35:
        readiness = "Stretch"
        readiness_rationale = "Meaningful gaps exist. Consider closing 1-2 key gaps before applying, or apply with a clear upskilling narrative."
    else:
        readiness = "Not Ready"
        readiness_rationale = "Significant skill distance. This role may be a future target rather than an immediate application."

    # Key insights
    key_insights = [
        f"Match score vs. {user_label}: {match_score:.0f}/100",
        f"Job requires {len([s for s in job_vec if job_vec[s] >= 4])} core skills at level ≥ 4/7",
    ]
    if top_matches:
        key_insights.append(f"Your strongest advantage: {top_matches[0]['skill']}")
    if top_gaps:
        key_insights.append(f"Biggest gap to address: {top_gaps[0]['skill']} (you: {top_gaps[0]['your_level']:.1f}, job needs: {top_gaps[0]['job_requires']:.1f})")
    key_insights.append(f"Application readiness: {readiness} — {readiness_rationale}")

    return {
        "role_title": role_title,
        "company": company,
        "experience_years_required": meta.get("experience_years_required"),
        "education_required": meta.get("education_required", "Not specified"),
        "domain": meta.get("domain", ""),
        "key_responsibilities": meta.get("key_responsibilities", [])[:4],
        "required_skills_raw": required_skills_raw[:12],
        "nice_to_have_skills": [
            s["skill"] for s in meta.get("required_skills", []) if s.get("importance") == "preferred"
        ][:6],
        "job_skill_vector": job_vec,
        "top_matches": top_matches,
        "top_gaps": top_gaps,
        "match_score": round(match_score, 1),
        "application_readiness": readiness,
        "readiness_rationale": readiness_rationale,
        "key_insights": key_insights,
        "user_label": user_label,
        "source": "online",
    }
