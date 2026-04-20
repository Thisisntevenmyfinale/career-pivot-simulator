"""
Pivot Defensibility Briefing
=============================
Generates the hardest questions a hiring manager will ask — specifically
about the WEAKEST points of THIS candidate's pivot — with tailored answers.

The insight: most candidates prepare for questions they can already answer well.
This module finds the questions they will AVOID and forces them to prepare for those.

Architecture:
  Input: company + role + full candidate profile
  Output: 3-5 hardest questions, each with:
    - Why this specific HM at this specific company would ask it
    - The best possible answer (grounded in actual CV data, not generic advice)
    - The exact trap/mistake to avoid
    - The "killer story" that flips the question into a strength

This is not generic interview prep. It's adversarial scenario planning
calibrated to the specific weakness profile of the pivot.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_DEFENSIBILITY_SYSTEM = """You are a senior hiring manager who has interviewed 200+ product managers.
You are evaluating a candidate making a career pivot. Your job: generate the hardest questions
you would ask THIS specific candidate, given their WEAKEST points.

Do not generate softball questions. Do not generate questions the candidate is obviously strong on.
Generate the questions that will make them sweat — the ones that probe the pivot's credibility.

For each question, provide:
- why a HM at this specific company would ask it (be specific about company context)
- the best possible answer using the candidate's ACTUAL data (specific, not generic)
- the exact trap (what most pivot candidates say that kills the application)
- a reframe: how to turn this weak point into a credibility signal

Output JSON only:
{
  "killer_story": "The one STAR story from their CV that should open every conversation — and why",
  "hardest_questions": [
    {
      "question": "exact interview question",
      "why_asked": "why THIS HM at THIS company asks this",
      "best_answer": "specific answer using their actual CV data — concrete, not generic",
      "trap_to_avoid": "the #1 mistake candidates make answering this",
      "reframe": "how to turn this into a strength signal"
    }
  ],
  "weakest_points": ["2-3 specific vulnerabilities of this exact pivot"],
  "strongest_openings": ["2-3 opening statements that immediately build credibility"],
  "topics_to_avoid_volunteering": ["topics that raise red flags if brought up unprompted"]
}"""


def generate_defensibility_briefing(
    oai_key: str,
    *,
    company: str,
    job_title: str,
    cv_profile: Dict,
    pivot_dna: Dict,
    skill_gap_results: Optional[Dict] = None,
    mock_interview_report: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a pre-application defensibility briefing for a specific company+role.
    Returns dict with hardest_questions, killer_story, etc. or None on failure.
    """
    if not oai_key or not company or not job_title:
        return None
    if not cv_profile:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    # Extract mock interview weak spots if available
    mock_weak = []
    if mock_interview_report:
        dims = mock_interview_report.get("dimension_scores", {})
        for dim, score in dims.items():
            if isinstance(score, (int, float)) and score < 70:
                mock_weak.append(f"{dim}: {score}/100")
        improvements = mock_interview_report.get("top_improvements", [])
        mock_weak.extend(improvements[:2])

    candidate = {
        "current_role":           cv_profile.get("extracted_role", ""),
        "years_experience":       cv_profile.get("years_experience", 0),
        "top_skills":             cv_profile.get("top_skills", [])[:12],
        "key_achievements":       cv_profile.get("key_achievements", [])[:4],
        "education":              cv_profile.get("education", ""),
        "pivot_hook":             pivot_dna.get("pivot_hook", ""),
        "strongest_argument":     pivot_dna.get("strongest_transferable_argument", ""),
        "pivot_risk":             pivot_dna.get("pivot_risk", ""),
        "mitigation":             pivot_dna.get("mitigation", ""),
        "unfair_advantage":       pivot_dna.get("unfair_advantage", ""),
        "top_skill_gaps":         [g.get("skill") for g in ((skill_gap_results or {}).get("gaps") or [])[:4]],
        "mock_interview_score":   (mock_interview_report or {}).get("overall_score"),
        "known_weak_dimensions":  mock_weak[:4],
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.25,
            messages=[
                {"role": "system", "content": _DEFENSIBILITY_SYSTEM},
                {"role": "user", "content": (
                    f"Company: {company}\n"
                    f"Role: {job_title}\n"
                    f"Candidate profile:\n{json.dumps(candidate, indent=2)}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["company"]   = company
        result["job_title"] = job_title
        return result
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Skill Proof Engine
# ─────────────────────────────────────────────────────────────────────────────

_PROOF_TASK_SYSTEM = """You are a hiring manager who has seen thousands of PM candidates.
You know exactly what separates "I claim I can do X" from "I can prove I can do X."

Generate ONE specific, completable proof-of-skill task for the given skill gap.
This task must:
- Be completable in 2-4 hours
- Produce an artifact the candidate can show in an interview or attach to an application
- Mirror real work a PM does, not an abstract exercise
- Be scoped to the candidate's pivot context (not generic)

Output JSON only:
{
  "task_title": "short, specific title",
  "task_type": "case_study|spec_writing|data_analysis|user_research|competitive_analysis|prototype",
  "task_description": "3-4 sentence exact description of what to produce",
  "deliverable": "exactly what the output looks like (e.g. '1-page PRD with 3 user stories')",
  "evaluation_criteria": ["criterion 1", "criterion 2", "criterion 3"],
  "estimated_time": "X hours",
  "credibility_signal": "What this artifact proves to a hiring manager — be specific",
  "interview_angle": "How to naturally reference this artifact in an interview answer"
}"""

_PROOF_EVAL_SYSTEM = """You are a senior PM hiring manager evaluating a proof-of-skill submission.
Be demanding. A score of 80+ means 'this could go in a PM portfolio.' Below 60 means 'needs significant rework.'
Be specific in your feedback — vague praise or vague criticism is useless.

Output JSON only:
{
  "score": <int 0-100>,
  "verdict": "strong|adequate|needs_work",
  "what_works": "specific 1-2 sentences on the strongest part",
  "what_to_improve": "specific 1-2 sentences on the single most important improvement",
  "credibility_level": "entry|mid|senior",
  "interview_ready": <bool>,
  "suggested_revision": "One specific concrete revision to make this stronger"
}"""


def generate_proof_task(
    oai_key: str,
    *,
    skill: str,
    target_role: str,
    pivot_context: str = "",
) -> Optional[Dict[str, Any]]:
    if not oai_key or not skill:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": _PROOF_TASK_SYSTEM},
                {"role": "user", "content": (
                    f"Skill to prove: {skill}\n"
                    f"Target role: {target_role}\n"
                    f"Candidate context: {pivot_context[:300]}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["skill"] = skill
        return result
    except Exception:
        return None


def evaluate_proof_submission(
    oai_key: str,
    *,
    skill: str,
    task_description: str,
    deliverable: str,
    submission_text: str,
    evaluation_criteria: List[str],
) -> Optional[Dict[str, Any]]:
    if not oai_key or not submission_text.strip():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _PROOF_EVAL_SYSTEM},
                {"role": "user", "content": (
                    f"Skill being proved: {skill}\n"
                    f"Task: {task_description}\n"
                    f"Expected deliverable: {deliverable}\n"
                    f"Evaluation criteria:\n" + "\n".join(f"- {c}" for c in evaluation_criteria) +
                    f"\n\nCandidate submission:\n\n{submission_text[:3000]}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None
