"""
Unified AI Advisor
==================
The brain that reads everything and tells you exactly what to do next.

Instead of juggling 8 separate tools, this synthesizes ALL session state
into one clear, prioritized recommendation — the "what do I do right now?"
that every job seeker actually needs.

Reads:
- O*NET fit score + top skill gaps
- Pipeline health (response rate, rejection patterns)
- ATS scores across applications
- Interview performance if available
- Time in search
- Negotiation status

Outputs:
- One primary action (the highest-leverage thing to do in the next 2 hours)
- Three supporting actions (in priority order)
- A plain-English status summary ("Your momentum is...")
- A momentum score (0-100)
- A phase label ("Exploration" | "Active Application" | "Interview Loop" | "Negotiation")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def synthesize_advisor_recommendation(
    # O*NET / fit data
    fit_score: Optional[float] = None,
    target_role: str = "",
    current_role: str = "",
    top_skill_gaps: Optional[List[str]] = None,
    # Pipeline data
    pipeline_stats: Optional[Dict[str, Any]] = None,
    pipeline_diagnosis: Optional[Dict[str, Any]] = None,
    rejection_analysis: Optional[Dict[str, Any]] = None,
    weeks_searching: int = 0,
    # Application quality
    avg_ats_score: Optional[int] = None,
    last_ats_score: Optional[int] = None,
    has_cover_letter: bool = False,
    # Interview data
    mock_interview_score: Optional[int] = None,
    mock_interview_complete: bool = False,
    # Negotiation status
    has_offer: bool = False,
    offer_analyzed: bool = False,
    # General
    cv_uploaded: bool = False,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Synthesize all session state into one prioritized recommendation.

    Returns:
      {
        "phase": str,                 # current phase of search
        "momentum_score": int,        # 0-100
        "momentum_label": str,        # "Stalled" | "Building" | "Strong" | "On Fire"
        "status_summary": str,        # 2-3 sentence plain English status
        "primary_action": str,        # the ONE thing to do next
        "primary_action_why": str,    # why this action, not another
        "supporting_actions": List[str],  # 2-3 other high-value actions
        "green_signals": List[str],   # what's going well
        "warning_signals": List[str], # what needs attention
        "time_to_offer_estimate": str,
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "phase": "Exploration",
        "momentum_score": 0,
        "momentum_label": "Getting Started",
        "status_summary": "Upload your CV and run an O*NET fit analysis to activate your personalized career advisor.",
        "primary_action": "Upload your CV and enter your current + target role to get started.",
        "primary_action_why": "The advisor needs your profile to generate personalized guidance.",
        "supporting_actions": [
            "Use Quick Apply mode to discover matching job listings",
            "Browse the O*NET skill gap analysis to understand your pivot distance",
        ],
        "green_signals": [],
        "warning_signals": ["No profile data yet — advisor is in minimal mode"],
        "time_to_offer_estimate": "Unknown — needs more data",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    # Build context string
    stats = pipeline_stats or {}
    gaps_str = ", ".join(top_skill_gaps[:5]) if top_skill_gaps else "not analyzed"

    context_parts = [
        f"CANDIDATE: {current_role or 'unknown'} → {target_role or 'unknown'}",
        f"CV UPLOADED: {cv_uploaded}",
        f"O*NET FIT SCORE: {fit_score:.0f}% percentile" if fit_score else "O*NET fit: not analyzed",
        f"TOP SKILL GAPS: {gaps_str}",
        f"WEEKS SEARCHING: {weeks_searching}",
        "",
        "PIPELINE STATUS:",
        f"  Total applications: {stats.get('total', 0)}",
        f"  Response rate: {stats.get('response_rate', 0)}% (healthy = 20-35%)",
        f"  Interview rate: {stats.get('interview_rate', 0)}%",
        f"  Offers: {stats.get('offers', 0)}",
        f"  Avg ATS score: {stats.get('avg_ats_score', avg_ats_score or 0)}/100",
        "",
        f"LAST ATS SCORE: {last_ats_score}/100" if last_ats_score else "ATS: not scanned",
        f"HAS COVER LETTER: {has_cover_letter}",
        f"MOCK INTERVIEW: {'Completed, score ' + str(mock_interview_score) + '/100' if mock_interview_complete else 'Not taken'}",
        f"OFFER STATUS: {'Has offer' + (' (analyzed)' if offer_analyzed else ' (not yet analyzed)') if has_offer else 'No offer yet'}",
    ]

    if pipeline_diagnosis:
        context_parts += [
            "",
            f"PIPELINE DIAGNOSIS: {pipeline_diagnosis.get('top_bottleneck', '')}",
            f"  Health score: {pipeline_diagnosis.get('health_score', 0)}/100",
            f"  Highest leverage action: {pipeline_diagnosis.get('highest_leverage_action', '')}",
        ]

    if rejection_analysis and rejection_analysis.get("primary_pattern"):
        context_parts += [
            "",
            f"REJECTION PATTERN: {rejection_analysis.get('primary_pattern', '')}",
            f"  Root cause: {rejection_analysis.get('root_cause', '')}",
        ]

    context = "\n".join(context_parts)

    prompt = f"""You are a senior career strategist. Based on this candidate's complete data, synthesize the most important guidance.

{context}

Your job: Identify exactly where this person is, what's working, what's broken, and the single highest-leverage next action.

Be direct. Be specific. No generic advice. Reference their actual numbers.

PHASE OPTIONS: "Exploration" (just starting, no applications), "Active Application" (applying but no interviews), "Interview Loop" (getting interviews), "Negotiation" (has offer)

MOMENTUM LABELS: "Stalled" (0-25), "Building" (26-50), "Strong" (51-75), "On Fire" (76-100)

Respond ONLY with valid JSON:
{{
  "phase": "Active Application",
  "momentum_score": 42,
  "momentum_label": "Building",
  "status_summary": "You've applied to 12 roles in 3 weeks but your 8% response rate is below the 20-35% healthy benchmark. The data points to an ATS problem — your average score of 61/100 means most applications are getting filtered before a human sees them.",
  "primary_action": "Re-run ATS scan on your top 3 target roles and use the Fix & Regenerate loop to push each score above 75 before reapplying.",
  "primary_action_why": "With an 8% response rate and avg ATS score of 61, fixing keyword coverage is 5× more impactful than sending more applications at your current quality level.",
  "supporting_actions": [
    "Add a 'Technical Skills' section to your CV with the 5 most-flagged missing keywords",
    "Run mock interview practice — you're close to getting calls and need to be ready",
    "Narrow your target to 3 specific companies and generate Company Intel briefs"
  ],
  "green_signals": [
    "Strong O*NET fit at 73rd percentile — targeting the right roles",
    "12 applications in 3 weeks shows good search velocity"
  ],
  "warning_signals": [
    "Response rate 8% vs. healthy 20-35% — systematic ATS issue",
    "ATS average 61/100 — keyword gaps blocking human review"
  ],
  "time_to_offer_estimate": "8-12 weeks at current pace; 4-6 weeks if ATS scores improve"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.25,
            max_tokens=700,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


def generate_rejection_reframe(
    job_title: str,
    company: str,
    rejection_stage: str = "unknown",
    search_context: str = "",
    total_applications: int = 0,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Rejection Reframe Engine — dual output: emotional support + technical diagnosis.

    Returns:
      {
        "reframe": str,          # honest, non-patronizing emotional reframe
        "statistical_context": str,  # what the data says about this stage
        "technical_diagnosis": str,  # what likely went wrong
        "next_action": str,      # most specific thing to do right now
        "motivation_line": str,  # one line to keep going
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "reframe": "Rejection is data. Every no narrows the gap to the right yes.",
        "statistical_context": "Add API key for data-driven rejection analysis.",
        "technical_diagnosis": "Log rejection stage details for pattern analysis.",
        "next_action": "Log the rejection stage and continue tracking.",
        "motivation_line": "You're building pipeline. Keep going.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    prompt = f"""You are a career coach who is direct, data-driven, and genuinely supportive — not a cheerleader.

REJECTION EVENT:
- Role: {job_title} at {company}
- Stage: {rejection_stage}
- Total applications so far: {total_applications}
- Search context: {search_context or "career pivot"}

Provide two things:
1. An honest, non-patronizing emotional reframe (not "you're amazing" — that's empty). Acknowledge the sting, then reframe with data/perspective.
2. A technical diagnosis of what likely went wrong at this stage, with one specific fix.

Keep the reframe under 60 words. Be real with them.

Respond ONLY with valid JSON:
{{
  "reframe": "Getting rejected at the final round genuinely hurts — you were close. That's also the best signal you can get: your application is working, your story lands, you're just not winning the last conversation yet. That's a coachable problem.",
  "statistical_context": "Final round rejection rate averages 70-80% — most candidates who reach this stage don't get the offer. That's not failure; that's a numbers game you're now inside.",
  "technical_diagnosis": "Final round rejections almost always come down to one of three things: salary mismatch, culture fit signals, or a stronger internal candidate. At this stage, the bottleneck is rarely your skills.",
  "next_action": "Send a graceful follow-up email asking for one piece of feedback — 30% of candidates who ask receive it, and it's the fastest way to improve your final-round conversion.",
  "motivation_line": "You made a final round. That's the top 10-15% of applicants. Do it again."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=450,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}
