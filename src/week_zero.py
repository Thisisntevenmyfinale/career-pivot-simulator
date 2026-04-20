"""
Week Zero Success Protocol
===========================
The pivot doesn't end when the offer is accepted. It ends when the
probationary period is successfully completed — typically 3-6 months.

40% of career changers feel out of their depth in the first 90 days.
Most make the same mistakes:
  - Trying to prove themselves immediately (wrong — listen first)
  - Not mapping stakeholder relationships fast enough
  - Missing the unwritten cultural rules of the new role
  - Failing to build early wins that confirm the hiring decision

This module generates a personalised 30-day success plan for the first
month in the new role, based on:
  - The target occupation and company type
  - The candidate's background (where they're coming from)
  - The specific skill gaps that were identified during the search
  - Cohort intelligence on what new PMs typically struggle with

The plan is NOT generic onboarding advice. It's calibrated to:
  - THIS specific career transition
  - THIS person's background and gaps
  - THIS company type's expectations

Output structure:
  day_1_5:    The "listen and map" phase — never impress, only absorb
  day_6_15:   The "quick win identification" phase — find the first deliverable
  day_16_30:  The "first impact" phase — deliver something visible
  day_31_90:  The "establishing track record" phase — summary
  stakeholder_map:    Who to meet in week 1 and what to ask each person
  early_win_strategy: How to identify and execute the first quick win
  failure_modes:      The 3 most common ways new PMs fail in this transition
  success_signals:    How they'll know they're on track at day 30
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_WEEK_ZERO_SYSTEM = """You are an executive onboarding coach specialising in career changers entering product roles.
You have coached 200+ people through their first 90 days as a PM after pivoting from a different function.

Generate a highly specific 30-day success plan. NOT generic onboarding advice.
Calibrated to:
  - The person's specific background (where they're coming from)
  - The target role type and company stage
  - Their known skill gaps from the job search
  - Common failure modes for this specific transition type

Rules:
  - Every task must be specific and completable in < 2 hours
  - Day 1-5: ONLY listening and mapping. Zero "proving yourself."
  - Week 2: Identify ONE quick win. Not execute — identify.
  - Week 3-4: Execute the quick win. Make it visible.
  - Address the psychological challenges directly — imposter syndrome is real

Output JSON only:
{
  "mindset_for_day_one": "The one mental shift that makes or breaks the first month",
  "day_1_5": [
    {"day": "Day 1", "tasks": ["specific task"], "why": "reason"}
  ],
  "day_6_15": [
    {"day": "Day 6-8", "tasks": ["specific task"], "why": "reason"}
  ],
  "day_16_30": [
    {"day": "Day 16-20", "tasks": ["specific task"], "why": "reason"}
  ],
  "day_31_90": "One paragraph: what the next 60 days should focus on",
  "stakeholder_map": [
    {"role": "who to meet", "ask_them": "specific question to ask", "goal": "what you're learning"}
  ],
  "early_win_strategy": "How to find and execute the first deliverable that proves the hiring decision right",
  "failure_modes": ["3 specific ways people from your background fail in this transition"],
  "success_signals": ["How you'll know you're on track at day 30"],
  "imposter_syndrome_note": "Direct, honest note about handling the feeling of being unqualified",
  "leverage_from_old_role": "The one superpower from your previous career that most new PMs don't have"
}"""


def generate_week_zero_plan(
    oai_key: str,
    *,
    target_role: str,
    company: str,
    company_stage: str,
    previous_role: str,
    skill_gaps: Optional[List[str]] = None,
    pivot_dna: Optional[Dict] = None,
    cv_profile: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a 30-day onboarding success plan for a career changer starting a new PM role.
    Returns structured plan dict or None on failure.
    """
    if not oai_key or not target_role:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    context = {
        "new_role":          target_role,
        "company":           company,
        "company_stage":     company_stage,
        "previous_role":     previous_role,
        "known_gaps":        (skill_gaps or [])[:5],
        "unfair_advantage":  (pivot_dna  or {}).get("unfair_advantage", ""),
        "three_word_brand":  (pivot_dna  or {}).get("three_word_brand", ""),
        "career_narrative":  (pivot_dna  or {}).get("career_narrative", "")[:200],
        "top_achievements":  (cv_profile or {}).get("key_achievements", [])[:3],
        "years_experience":  (cv_profile or {}).get("years_experience", 0),
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": _WEEK_ZERO_SYSTEM},
                {"role": "user",   "content": (
                    f"New role: {target_role} at {company} ({company_stage})\n"
                    f"Transitioning from: {previous_role}\n"
                    f"Context: {json.dumps(context, indent=2)}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["role"]    = target_role
        result["company"] = company
        return result
    except Exception:
        return None
