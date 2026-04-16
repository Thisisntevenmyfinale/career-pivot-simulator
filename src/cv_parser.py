"""
CV Parser — Personal Skill Extraction
======================================
Extracts skill levels from a user's CV text and maps them to the O*NET skill
space. This transforms the entire pipeline from generic (O*NET role average)
to personal (YOUR actual skill profile).

Architecture
------------
1. LLM Pass 1 — free extraction: pull every skill, technology, and competency
   mentioned in the CV, with evidence and estimated proficiency.
2. LLM Pass 2 — O*NET mapping: given the extracted skills + the O*NET skill
   dimension list, rate each dimension 0–7 (matching O*NET importance scale).
3. Returns a pd.Series indexed by O*NET skill names — a drop-in replacement
   for any row in the skill matrix.

Why this matters
----------------
O*NET role vectors represent an *average* occupational profile.
A personal CV vector represents *this specific person*.
The gap analysis then answers: "What do YOU specifically need to develop?"
instead of "What does a typical person in Role A need?"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Offline proficiency mapping ──────────────────────────────────────────────
_OFFLINE_LEVEL_KEYWORDS: Dict[str, float] = {
    "expert": 6.5,
    "senior": 6.0,
    "lead": 6.0,
    "principal": 6.5,
    "advanced": 5.5,
    "proficient": 5.0,
    "experienced": 4.5,
    "working knowledge": 3.5,
    "familiar": 3.0,
    "basic": 2.5,
    "beginner": 2.0,
    "learning": 2.0,
}


def _offline_skill_vector(cv_text: str, skill_columns: List[str]) -> pd.Series:
    """
    Fallback when OpenAI is unavailable.
    Scans the CV text for direct mentions of O*NET skill names and assigns
    levels based on co-occurring proficiency keywords.
    """
    text_lower = cv_text.lower()
    result: Dict[str, float] = {}

    for skill in skill_columns:
        if skill.lower() in text_lower:
            # Look for proficiency words near the skill mention
            idx = text_lower.find(skill.lower())
            context = text_lower[max(0, idx - 60) : idx + 60]
            level = 3.5  # default: mentioned = working knowledge
            for kw, lvl in _OFFLINE_LEVEL_KEYWORDS.items():
                if kw in context:
                    level = lvl
                    break
            result[skill] = level

    series = pd.Series(0.0, index=skill_columns, dtype=float)
    for skill, level in result.items():
        if skill in series.index:
            series[skill] = level

    return series


def extract_cv_skills_online(
    cv_text: str,
    skill_columns: List[str],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-pass LLM extraction:
      Pass 1 — free-form: extract all skills/technologies mentioned in the CV
      Pass 2 — mapping: score each O*NET skill dimension 0-7

    Returns a dict with skill_vector, metadata, and confidence.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not available", "source": "offline"}

    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception as e:
        return {"error": f"OpenAI client init failed: {repr(e)}", "source": "offline"}

    # ── Pass 1: free extraction ───────────────────────────────────────────────
    pass1_prompt = f"""You are a CV skill analyst. Extract every skill, technology, tool, methodology,
and domain competency mentioned in this CV. For each, note:
- The skill name
- Proficiency level: one of [beginner, basic, working, proficient, advanced, expert]
- Brief evidence from the CV (max 10 words)

Also extract:
- The person's most recent role title
- Approximate years of total experience
- Highest education level

Respond ONLY with valid JSON in this exact structure:
{{
  "extracted_role": "...",
  "years_experience": 0.0,
  "education_level": "...",
  "skills": [
    {{"skill": "...", "level": "...", "evidence": "..."}}
  ]
}}

CV TEXT:
{cv_text[:4000]}
"""

    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": pass1_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1200,
        )
        pass1 = json.loads(r1.choices[0].message.content or "{}")
    except Exception as e:
        return {"error": f"Pass 1 extraction failed: {repr(e)}", "source": "offline"}

    extracted_skills: List[Dict] = pass1.get("skills", [])
    extracted_role: str = pass1.get("extracted_role", "Unknown")
    years_exp: float = float(pass1.get("years_experience", 0.0) or 0.0)
    education: str = pass1.get("education_level", "Unknown")

    if not extracted_skills:
        vec = pd.Series(0.0, index=skill_columns, dtype=float)
        return {
            "skill_vector": vec,
            "extracted_role": extracted_role,
            "years_experience": years_exp,
            "education_level": education,
            "top_skills": [],
            "skills_extracted_count": 0,
            "skills_mapped_count": 0,
            "confidence": 0.2,
            "source": "online_empty",
        }

    # ── Pass 2: map to O*NET skill space ─────────────────────────────────────
    # Batch the O*NET skills in groups to stay within token limits
    BATCH_SIZE = 60
    onet_scores: Dict[str, float] = {}

    # Build a lookup from extracted skills for context injection
    skill_context = "\n".join(
        [f"- {s['skill']} ({s['level']}): {s['evidence']}" for s in extracted_skills[:40]]
    )

    for batch_start in range(0, len(skill_columns), BATCH_SIZE):
        batch = skill_columns[batch_start : batch_start + BATCH_SIZE]
        batch_str = "\n".join([f"- {s}" for s in batch])

        pass2_prompt = f"""You are mapping a person's CV skills to a standardized O*NET occupational skill framework.

PERSON'S EXTRACTED SKILLS (from their CV):
{skill_context}

PERSON CONTEXT: {extracted_role}, ~{years_exp:.0f} years experience, {education}

For each O*NET skill dimension below, estimate this person's level on a 0–7 scale:
0 = No evidence in CV
1–2 = Tangentially mentioned / basic
3–4 = Working knowledge / used regularly
5–6 = Proficient / advanced
7 = Expert / lead-level / published work

O*NET SKILL DIMENSIONS TO RATE:
{batch_str}

Respond ONLY with valid JSON:
{{"ratings": {{"<skill_name>": <0-7 float>, ...}}}}

Rate ONLY the skills listed above. Use 0 for anything not evidenced.
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
            ratings = batch_result.get("ratings", {})
            for skill_name, val in ratings.items():
                if skill_name in onet_scores:
                    continue
                try:
                    onet_scores[skill_name] = float(np.clip(float(val), 0.0, 7.0))
                except (TypeError, ValueError):
                    pass
        except Exception:
            # Partial failure: skip this batch
            continue

    # Build the final Series
    vec = pd.Series(0.0, index=skill_columns, dtype=float)
    mapped_count = 0
    for skill in skill_columns:
        if skill in onet_scores:
            vec[skill] = onet_scores[skill]
            if onet_scores[skill] > 0:
                mapped_count += 1

    # Top skills for display
    top_skills = vec[vec > 0].sort_values(ascending=False).head(10).index.tolist()

    confidence = min(0.9, 0.3 + 0.05 * len(extracted_skills) + 0.01 * mapped_count)

    return {
        "skill_vector": vec,
        "extracted_role": extracted_role,
        "years_experience": years_exp,
        "education_level": education,
        "top_skills": top_skills,
        "skills_extracted_count": len(extracted_skills),
        "skills_mapped_count": mapped_count,
        "confidence": round(confidence, 2),
        "extracted_skills_raw": extracted_skills[:20],  # for display
        "source": "online",
    }


def parse_cv(
    cv_text: str,
    skill_columns: List[str],
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public entry point. Returns the same structure whether online or offline.
    Always includes 'skill_vector' (pd.Series) and 'source'.
    """
    cv_text = (cv_text or "").strip()
    if not cv_text:
        return {
            "skill_vector": pd.Series(0.0, index=skill_columns, dtype=float),
            "extracted_role": "",
            "years_experience": 0.0,
            "education_level": "",
            "top_skills": [],
            "skills_extracted_count": 0,
            "skills_mapped_count": 0,
            "confidence": 0.0,
            "source": "empty",
            "error": "No CV text provided.",
        }

    if prefer_online:
        result = extract_cv_skills_online(
            cv_text=cv_text,
            skill_columns=skill_columns,
            model=model,
            api_key=api_key,
        )
        if "error" not in result or result.get("source") == "online_empty":
            return result

    # Fallback
    vec = _offline_skill_vector(cv_text, skill_columns)
    top_skills = vec[vec > 0].sort_values(ascending=False).head(10).index.tolist()
    return {
        "skill_vector": vec,
        "extracted_role": "",
        "years_experience": 0.0,
        "education_level": "",
        "top_skills": top_skills,
        "skills_extracted_count": int((vec > 0).sum()),
        "skills_mapped_count": int((vec > 0).sum()),
        "confidence": 0.4 if (vec > 0).any() else 0.1,
        "source": "offline",
    }


def compute_personal_gap_df(
    personal_vector: pd.Series,
    target_role: str,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute gap between personal skill vector and target role's O*NET profile.

    Returns the same schema as model_logic.compute_gap_df:
      skill | current_importance | target_importance | gap
    but 'current_importance' now reflects YOUR actual skills, not the role average.
    """
    if target_role not in matrix.index:
        return pd.DataFrame(columns=["skill", "current_importance", "target_importance", "gap"])

    target_vec = matrix.loc[target_role]

    # Align indices
    common_skills = personal_vector.index.intersection(target_vec.index)
    personal_aligned = personal_vector.reindex(common_skills, fill_value=0.0)
    target_aligned = target_vec.reindex(common_skills, fill_value=0.0)

    gap = (target_aligned - personal_aligned).clip(lower=0.0)

    df = pd.DataFrame(
        {
            "skill": common_skills,
            "current_importance": personal_aligned.values.round(3),
            "target_importance": target_aligned.values.round(3),
            "gap": gap.values.round(3),
        }
    )

    # Only include skills where either role cares about it
    df = df[(df["current_importance"] > 0) | (df["target_importance"] > 0)].copy()
    df = df.sort_values("target_importance", ascending=False).reset_index(drop=True)
    return df
