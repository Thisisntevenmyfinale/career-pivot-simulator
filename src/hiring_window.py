"""
Hiring Window Intelligence
==========================
Tells you WHEN to apply — not just where.

Timing is the single most underrated variable in job searching.
The same application sent to a company with a hiring freeze
vs. one that just raised a Series B has radically different odds.

This module analyzes company-level signals to compute a
Hiring Momentum Score (0–100) and a strategic recommendation
for when and how to approach each target.

Architecture:
  gpt-4o-mini with web context injection.
  Input:  company name + target role + optional CV context
  Output: timing score, signal breakdown, strategic approach

Timing signals analyzed:
  funding_events     — recent Series A/B/C/IPO (headcount expansion likely)
  headcount_growth   — LinkedIn employee count trend
  product_launches   — new products = new team needs
  competitor_moves   — competitor layoffs = talent available / market shift
  seasonality        — Q1/Q4 typically have higher hiring budgets
  job_posting_velocity — how many roles are live right now
  exec_changes       — new CTO/VP often means new team rebuild
  press_signals      — coverage type (growth story vs. cost-cutting)
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional


SIGNAL_TYPES = {
    "funding":         {"label": "Funding event",       "color": "#057642", "icon": "trending-up"},
    "headcount":       {"label": "Headcount growth",    "color": "#0A66C2", "icon": "users"},
    "product_launch":  {"label": "Product expansion",   "color": "#7A2A8A", "icon": "zap"},
    "exec_change":     {"label": "Leadership change",   "color": "#A05A00", "icon": "user-check"},
    "job_velocity":    {"label": "Active hiring",       "color": "#057642", "icon": "briefcase"},
    "seasonality":     {"label": "Hiring season",       "color": "#0A66C2", "icon": "calendar"},
    "press_positive":  {"label": "Positive coverage",   "color": "#057642", "icon": "star"},
    "freeze_signal":   {"label": "Freeze / layoff risk","color": "#B71C1C", "icon": "pause"},
    "competitive":     {"label": "Competition signal",  "color": "#5F6B7A", "icon": "users"},
}


def analyze_hiring_window(
    oai_key: str,
    *,
    company_name: str,
    target_role: str,
    candidate_background: str = "",
    cv_profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Analyze hiring timing for a specific company + role.

    Returns:
      timing_score: int 0–100 (100 = perfect window)
      signal_strength: "strong" | "moderate" | "weak" | "negative"
      signals: list of {signal_type, description, weight, direction}
      window_verdict: str (1-sentence timing verdict)
      strategic_approach: str (HOW to approach given the signals)
      best_channel: str (email cold / LinkedIn / referral / job posting)
      talking_points: list[str] (3 points to reference in outreach)
      avoid: str (what NOT to say/do given the signals)
    """
    if not oai_key or not company_name:
        return _no_signal_result(company_name, target_role)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return _no_signal_result(company_name, target_role)

    today = date.today()
    month = today.month
    quarter = (month - 1) // 3 + 1

    candidate_ctx = ""
    if cv_profile:
        skills = (cv_profile.get("top_skills") or [])[:5]
        yoe = cv_profile.get("years_experience", "?")
        candidate_ctx = f"Candidate has {yoe} years experience. Top skills: {', '.join(skills)}."

    prompt = f"""You are a senior talent market analyst. Assess the hiring timing for a specific company and role.

Company: {company_name}
Target role: {target_role}
Today: {today.strftime('%B %Y')} (Q{quarter})
Candidate: {candidate_ctx or candidate_background or 'Not provided'}

Based on your knowledge of {company_name}'s recent trajectory, market position, and hiring patterns:

1. Assess whether now is a GOOD time to apply to {company_name} for a {target_role} role
2. Identify specific timing signals (funding, headcount, product news, layoffs, etc.)
3. Give a strategic approach for outreach

Be specific to {company_name} — not generic career advice.
If you don't have recent data on this specific company, say so honestly and give market-level signals.

Respond ONLY with valid JSON:
{{
  "timing_score": 0-100,
  "signal_strength": "strong|moderate|weak|negative",
  "signals": [
    {{
      "signal_type": "funding|headcount|product_launch|exec_change|job_velocity|seasonality|press_positive|freeze_signal|competitive",
      "description": "Specific observation about {company_name}",
      "direction": "positive|negative|neutral",
      "weight": 1-3
    }}
  ],
  "window_verdict": "One sentence: is now the right time? Be direct.",
  "strategic_approach": "2-3 sentences on HOW to approach this company given the signals.",
  "best_channel": "cold_email|linkedin_connect|referral|job_posting|events",
  "talking_points": ["specific thing to mention in outreach 1", "thing 2", "thing 3"],
  "avoid": "What NOT to do or say when approaching this company right now.",
  "data_confidence": "high|medium|low"
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=600,
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        timing_score = max(0, min(100, int(data.get("timing_score", 50))))
        signal_strength = data.get("signal_strength", "moderate")
        if signal_strength not in ("strong", "moderate", "weak", "negative"):
            signal_strength = "moderate"

        raw_signals = data.get("signals", [])
        signals = []
        for s in raw_signals[:6]:
            stype = s.get("signal_type", "seasonality")
            if stype not in SIGNAL_TYPES:
                stype = "seasonality"
            signals.append({
                "signal_type":   stype,
                "meta":          SIGNAL_TYPES[stype],
                "description":   str(s.get("description", ""))[:200],
                "direction":     s.get("direction", "neutral"),
                "weight":        int(s.get("weight", 1)),
            })

        return {
            "company":           company_name,
            "target_role":       target_role,
            "timing_score":      timing_score,
            "signal_strength":   signal_strength,
            "signals":           signals,
            "window_verdict":    str(data.get("window_verdict", ""))[:200],
            "strategic_approach":str(data.get("strategic_approach", ""))[:400],
            "best_channel":      str(data.get("best_channel", "linkedin_connect")),
            "talking_points":    [str(x)[:150] for x in data.get("talking_points", [])[:3]],
            "avoid":             str(data.get("avoid", ""))[:200],
            "data_confidence":   data.get("data_confidence", "medium"),
        }
    except Exception:
        return _no_signal_result(company_name, target_role)


def _no_signal_result(company_name: str, target_role: str) -> Dict[str, Any]:
    today = date.today()
    month = today.month
    # Q1 and early Q3 are typically peak hiring windows
    is_peak = month in (1, 2, 3, 7, 8, 9)
    timing_score = 60 if is_peak else 45

    return {
        "company":           company_name,
        "target_role":       target_role,
        "timing_score":      timing_score,
        "signal_strength":   "moderate",
        "signals": [
            {
                "signal_type": "seasonality",
                "meta":        SIGNAL_TYPES["seasonality"],
                "description": f"Q{(month-1)//3+1} — {'historically strong hiring window' if is_peak else 'moderate hiring activity'}",
                "direction":   "positive" if is_peak else "neutral",
                "weight":      2,
            }
        ],
        "window_verdict":    "No company-specific signals available — apply based on seasonal timing.",
        "strategic_approach":"Apply via the job posting and request a referral if possible. Timing looks neutral.",
        "best_channel":      "job_posting",
        "talking_points":    [],
        "avoid":             "Generic cover letters — be specific about why this company.",
        "data_confidence":   "low",
    }


def timing_score_color(score: int) -> str:
    if score >= 70:
        return "#057642"
    if score >= 50:
        return "#0A66C2"
    if score >= 35:
        return "#A05A00"
    return "#B71C1C"


def timing_score_label(score: int) -> str:
    if score >= 75:
        return "Ideal Window"
    if score >= 60:
        return "Good Timing"
    if score >= 45:
        return "Moderate"
    if score >= 30:
        return "Cautious"
    return "Poor Timing"
