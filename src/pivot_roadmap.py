"""
Pivot Path Execution Roadmap
=============================
Turns the Occupation Space bridge-role analysis + skill gap + cohort data
into a concrete, week-by-week execution plan.

Not generic career advice. Not "update your LinkedIn." Actual tasks
calibrated to THIS candidate's specific gap profile, pipeline state,
and cohort timeline benchmarks.

The plan is built in two layers:
  Layer 1: Structural tasks (from bridge roles, skill gaps, application volume)
           — deterministic, no API needed for the skeleton
  Layer 2: LLM enrichment (specific task descriptions, sequencing logic)
           — adds context-specific detail to each milestone

Output structure:
  phases:
    kickoff (week 1-2)    — foundation tasks, zero ambiguity
    bridge (week 3-6)     — bridge role applications + proof work
    primary (week 7-12)   — direct applications to primary targets
    compound (month 3-6)  — if no offer yet, escalation strategy
  critical_path           — the single most important thing to do first
  milestones              — [{week, milestone, completion_signal}]
  blocking_gaps           — skill gaps that MUST be addressed before primary apps
  timeline_estimate       — "10-14 weeks to first offer" (calibrated)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_ROADMAP_SYSTEM = """You are a career transition strategist with a ruthless focus on outcomes.
You are given a candidate's complete pivot profile and you must generate a concrete
week-by-week execution roadmap.

Rules:
- Every task must be specific and completable. "Update LinkedIn" is not a task.
  "Add 3 bullet points to LinkedIn Experience section referencing the 8 PRDs written" is a task.
- Respect the cohort timeline — don't over-promise
- The bridge role strategy is critical for career switchers — don't skip it
- Week 1 tasks must be doable in < 5 hours total

Output JSON only:
{
  "critical_path": "The single most important thing to do THIS WEEK",
  "timeline_estimate": "X–Y weeks to first offer (based on cohort + profile)",
  "blocking_gaps": ["skills that must be addressed before primary applications"],
  "phases": {
    "kickoff": {
      "label": "Week 1-2: Foundation",
      "tasks": [
        {"task": "specific action", "time_estimate": "2h", "why": "one sentence"}
      ]
    },
    "bridge": {
      "label": "Week 3-6: Bridge role applications",
      "tasks": [...]
    },
    "primary": {
      "label": "Week 7-12: Primary target applications",
      "tasks": [...]
    },
    "compound": {
      "label": "Month 3-6: If no offer yet — escalation",
      "tasks": [...]
    }
  },
  "milestones": [
    {"week": 2, "milestone": "First bridge role application submitted", "completion_signal": "application in pipeline with status=applied"},
    {"week": 4, "milestone": "First phone screen", "completion_signal": "outcome_log entry with reached_response=true"}
  ],
  "weekly_cadence": "X applications/week, Y networking contacts, Z skill proof hours"
}"""


def generate_pivot_roadmap(
    oai_key: str,
    *,
    current_occ: str,
    target_occ: str,
    skill_gap_results: Optional[Dict] = None,
    cohort_intelligence: Optional[Dict] = None,
    pivot_dna: Optional[Dict] = None,
    pipeline_jobs: Optional[List] = None,
    outcome_log: Optional[List] = None,
    calibration_data: Optional[Dict] = None,
    bridge_occupations: Optional[List] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a personalised 30/60/90-day pivot execution roadmap.
    Returns structured roadmap dict, or None on failure.
    """
    if not oai_key:
        return _fallback_roadmap(current_occ, target_occ, skill_gap_results, cohort_intelligence)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return _fallback_roadmap(current_occ, target_occ, skill_gap_results, cohort_intelligence)

    # Build context summary
    pipeline_summary = {
        "total_applications": len(pipeline_jobs or []),
        "statuses": _count_statuses(pipeline_jobs or []),
        "total_outcomes": len(outcome_log or []),
        "personal_response_rate": (calibration_data or {}).get("personal_response_rate"),
    }
    gap_summary = {
        "fit_percentile": (skill_gap_results or {}).get("fit_percentile", 50),
        "top_gaps": [(g.get("skill"), g.get("severity")) for g in
                     ((skill_gap_results or {}).get("gaps") or [])[:5]],
    }
    cohort_summary = {
        "median_timeline_weeks": (cohort_intelligence or {}).get("median_timeline_weeks", 14),
        "median_applications":   (cohort_intelligence or {}).get("median_applications", 32),
        "what_worked":           (cohort_intelligence or {}).get("what_worked", "")[:300],
        "what_failed":           (cohort_intelligence or {}).get("what_failed", "")[:150],
    }
    bridge_summary = [b.get("occupation") for b in (bridge_occupations or [])[:3]]

    user_content = (
        f"Current occupation: {current_occ}\n"
        f"Target occupation: {target_occ}\n"
        f"Pivot hook: {(pivot_dna or {}).get('pivot_hook','')}\n"
        f"Target companies: {(pivot_dna or {}).get('target_companies', [])}\n"
        f"Skill gaps: {json.dumps(gap_summary)}\n"
        f"Pipeline state: {json.dumps(pipeline_summary)}\n"
        f"Cohort benchmarks: {json.dumps(cohort_summary)}\n"
        f"Bridge roles identified: {bridge_summary}\n"
        f"Strongest argument: {(pivot_dna or {}).get('strongest_transferable_argument','')[:300]}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": _ROADMAP_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["generated"] = True
        return result
    except Exception:
        return _fallback_roadmap(current_occ, target_occ, skill_gap_results, cohort_intelligence)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (no API)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_roadmap(current_occ, target_occ, skill_gap, cohort) -> Dict[str, Any]:
    weeks = (cohort or {}).get("median_timeline_weeks", 14)
    apps  = (cohort or {}).get("median_applications", 32)
    gaps  = [(g.get("skill","")) for g in ((skill_gap or {}).get("gaps") or [])[:3]]

    return {
        "generated": False,
        "critical_path": f"Close your top skill gap ({gaps[0] if gaps else 'identified gap'}) with a concrete portfolio proof this week.",
        "timeline_estimate": f"{weeks}–{weeks+4} weeks to first offer (cohort median: {apps} applications)",
        "blocking_gaps": gaps,
        "phases": {
            "kickoff": {
                "label": "Week 1-2: Foundation",
                "tasks": [
                    {"task": "Complete Skill Proof task for your top gap", "time_estimate": "3h", "why": "Gives you something concrete to reference in applications"},
                    {"task": "Research and add 5 target companies to pipeline", "time_estimate": "2h", "why": "Volume baseline — cohort median needs 32 applications"},
                    {"task": "Run ATS scan on your CV against 3 target JDs", "time_estimate": "1h", "why": "Keyword gaps cost you before a human ever reads it"},
                ],
            },
            "bridge": {
                "label": "Week 3-6: Bridge role applications",
                "tasks": [
                    {"task": "Apply to 2-3 bridge roles per week (adjacent to target)", "time_estimate": "4h/week", "why": "Bridge roles have higher acceptance rate and build the PM title story"},
                    {"task": "Refine pivot narrative based on first recruiter feedback", "time_estimate": "2h", "why": "Phone screen data is the fastest signal you'll get"},
                    {"task": "Add 2 skill proofs to portfolio", "time_estimate": "4h", "why": "Converts stated skills into demonstrated skills"},
                ],
            },
            "primary": {
                "label": "Week 7-12: Primary target applications",
                "tasks": [
                    {"task": "Apply to your top 5 target companies with tailored materials", "time_estimate": "6h", "why": "Quality matters more than volume at this stage"},
                    {"task": "Run war room prep for each first-round interview", "time_estimate": "2h/interview", "why": "Cohort data shows interview prep is the #1 converter"},
                    {"task": "Request feedback from every rejection — specifically", "time_estimate": "30min", "why": "3 data points reveal the pattern the Rejection Interpreter will flag"},
                ],
            },
            "compound": {
                "label": "Month 3-6: Escalation if needed",
                "tasks": [
                    {"task": "Reassess OPS — if < 40%, rebuild CV and Pivot DNA from scratch", "time_estimate": "4h", "why": "Systemic issue requires systemic fix"},
                    {"task": "Activate warm intro network for each remaining target company", "time_estimate": "ongoing", "why": "Referrals convert 4x better than direct applications at month 4+"},
                ],
            },
        },
        "milestones": [
            {"week": 2,  "milestone": "First application submitted",            "completion_signal": "pipeline_jobs entry with status=applied"},
            {"week": 4,  "milestone": "First phone screen",                      "completion_signal": "outcome_log with reached_response=true"},
            {"week": 8,  "milestone": "First interview",                          "completion_signal": "outcome_log with reached_interview=true"},
            {"week": weeks, "milestone": "First offer",                          "completion_signal": "outcome_log with is_offer=true"},
        ],
        "weekly_cadence": "3-4 applications/week, 2 networking contacts, 2-3h skill proof work",
    }


def _count_statuses(pipeline: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for job in pipeline:
        s = job.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    return counts
