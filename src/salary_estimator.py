"""
Salary Impact Estimator
========================
Estimates the compensation impact of a career pivot and generates a
month-by-month salary trajectory for the transition period.

Answers the #1 real-world question: "Will I earn less while pivoting,
and how long until the target role pays more?"

Architecture
------------
Single LLM call (gpt-4o-mini) with structured JSON output.
Uses US Bureau of Labor Statistics knowledge + O*NET occupational context.
All figures are estimates / simulations — clearly labelled as such.

Returns
-------
{
  current_median: int          — median annual salary in USD for current role
  current_range: [low, high]   — 25th–75th percentile range
  target_entry_median: int     — median salary entering the target role
  target_entry_range: [l, h]
  target_senior_median: int    — median salary after 3-5 yrs in target role
  target_senior_range: [l, h]
  months_to_breakeven: int     — months until target salary > current salary
  entry_delta_pct: float       — % change at entry level (can be negative)
  ceiling_delta_pct: float     — % change at senior level
  trajectory: List[{month, salary, phase}]
  insights: List[str]          — 3-4 plain-English takeaways
  source: str
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _offline_salary(
    current_role: str,
    target_role: str,
    match_score: float,
) -> Dict[str, Any]:
    """Rule-based fallback estimates when OpenAI is unavailable."""
    base = 75_000
    target_entry = int(base * (0.85 if match_score < 50 else 0.92))
    target_senior = int(base * 1.45)
    months_bre = 18 if match_score < 50 else 12

    trajectory = []
    for m in range(37):
        if m == 0:
            s = base
        elif m <= months_bre:
            # Gradual dip then climb back to current
            frac = m / months_bre
            s = int(target_entry + (base - target_entry) * frac)
        else:
            # Growth beyond current
            extra = (m - months_bre) / 24
            s = int(base + (target_senior - base) * min(extra, 1.0))
        trajectory.append({"month": m, "salary": s,
                            "phase": "Transition" if m <= months_bre else "Growth"})

    entry_delta = round((target_entry - base) / base * 100, 1)
    ceiling_delta = round((target_senior - base) / base * 100, 1)

    return {
        "current_median": base,
        "current_range": [int(base * 0.80), int(base * 1.25)],
        "target_entry_median": target_entry,
        "target_entry_range": [int(target_entry * 0.85), int(target_entry * 1.15)],
        "target_senior_median": target_senior,
        "target_senior_range": [int(target_senior * 0.85), int(target_senior * 1.20)],
        "months_to_breakeven": months_bre,
        "entry_delta_pct": entry_delta,
        "ceiling_delta_pct": ceiling_delta,
        "trajectory": trajectory,
        "insights": [
            f"Entry-level {target_role} roles typically pay {abs(entry_delta):.0f}% "
            f"{'less' if entry_delta < 0 else 'more'} than {current_role}.",
            f"Senior {target_role} roles offer ~{ceiling_delta:.0f}% higher ceiling.",
            f"Estimated break-even: {months_bre} months after starting in the target role.",
            "Upload your CV for personalised salary estimates.",
        ],
        "source": "offline",
    }


def estimate_salary_impact(
    current_role: str,
    target_role: str,
    match_score: float,
    years_experience: float = 0.0,
    location: str = "United States",
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate the salary impact of transitioning from current_role to target_role.

    All figures are LLM-based estimates using US labour-market knowledge.
    They are clearly labelled as simulations in the UI.
    """
    if not prefer_online:
        return _offline_salary(current_role, target_role, match_score)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_salary(current_role, target_role, match_score)

    exp_context = f"{years_experience:.0f} years of experience" if years_experience > 0 else "experience level unknown"

    prompt = f"""You are a compensation analyst with deep knowledge of US labour markets.
Estimate realistic salary figures for a career transition.

TRANSITION:
- From: {current_role} ({exp_context})
- To: {target_role}
- Skill match score: {match_score:.0f}/100
- Location: {location}

Provide realistic salary estimates based on actual US Bureau of Labor Statistics
data and industry knowledge. Be specific and accurate — not generic.

Respond ONLY with valid JSON:
{{
  "current_median": 85000,
  "current_range": [68000, 108000],
  "target_entry_median": 78000,
  "target_entry_range": [62000, 95000],
  "target_senior_median": 130000,
  "target_senior_range": [105000, 165000],
  "months_to_breakeven": 14,
  "entry_delta_pct": -8.2,
  "ceiling_delta_pct": 52.9,
  "trajectory": [
    {{"month": 0, "salary": 85000, "phase": "Current"}},
    {{"month": 6, "salary": 72000, "phase": "Transition"}},
    {{"month": 12, "salary": 80000, "phase": "Transition"}},
    {{"month": 18, "salary": 90000, "phase": "Growth"}},
    {{"month": 24, "salary": 105000, "phase": "Growth"}},
    {{"month": 36, "salary": 120000, "phase": "Growth"}}
  ],
  "insights": [
    "Specific insight 1 about this exact transition",
    "Specific insight 2 (cite the actual numbers)",
    "Specific insight 3 about the break-even timeline",
    "Specific insight 4 about long-term upside"
  ]
}}

Rules:
- trajectory must have exactly 7 data points: months 0, 3, 6, 12, 18, 24, 36
- months_to_breakeven is when target salary first exceeds current_median
- entry_delta_pct is negative if entry target pay is below current pay
- ceiling_delta_pct is % difference between target_senior_median and current_median
- Be realistic — if this is a hard pivot (low match score), entry salaries are typically lower
- Insights must be specific to these two roles, not generic
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=700,
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        # Validate and clean trajectory
        traj = data.get("trajectory", [])
        if not traj or len(traj) < 3:
            raise ValueError("Bad trajectory")

        return {
            "current_median": int(data.get("current_median", 80000)),
            "current_range": [int(x) for x in data.get("current_range", [60000, 100000])],
            "target_entry_median": int(data.get("target_entry_median", 75000)),
            "target_entry_range": [int(x) for x in data.get("target_entry_range", [55000, 95000])],
            "target_senior_median": int(data.get("target_senior_median", 110000)),
            "target_senior_range": [int(x) for x in data.get("target_senior_range", [85000, 140000])],
            "months_to_breakeven": int(data.get("months_to_breakeven", 15)),
            "entry_delta_pct": float(data.get("entry_delta_pct", 0.0)),
            "ceiling_delta_pct": float(data.get("ceiling_delta_pct", 0.0)),
            "trajectory": [
                {"month": int(p["month"]), "salary": int(p["salary"]), "phase": str(p.get("phase", ""))}
                for p in traj
            ],
            "insights": [str(i) for i in data.get("insights", [])[:4]],
            "source": "online",
        }

    except Exception as e:
        result = _offline_salary(current_role, target_role, match_score)
        result["source"] = f"offline (error: {repr(e)[:60]})"
        return result
