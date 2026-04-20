"""
Rejection Pattern Detector
===========================
Detects systematic problems in a job search before the user recognises them.

The human brain rationalises each individual rejection ("they went with someone
more senior", "poor cultural fit", etc.). This module aggregates across ALL
rejections, finds the statistical pattern, and surfaces it as an alert —
before the user repeats the mistake a 5th time.

Detection logic (no API needed — pure Python on structured data):

  Pattern 1: ATS Filter Block
    Signal: ≥2 outcomes with actual_stage = "no_response"
    Cause:  Resume not passing keyword screening
    Action: ATS keyword gap fix + resume reformat

  Pattern 2: Phone Screen Stall
    Signal: ≥2 outcomes with actual_stage = "phone_screen", reached_interview=False
    Cause:  Pivot narrative unconvincing, or compensation mismatch
    Action: Sharpen pivot hook, rehearse 60-sec pitch

  Pattern 3: Interview Drop-off
    Signal: ≥2 outcomes with reached_interview=True, is_offer=False
    Cause:  Deep-skills gap exposed in interview
    Action: Mock interview + skill proof work

  Pattern 4: Application Volume Alarm
    Signal: < cohort median applications with 0 interviews
    Cause:  Insufficient volume for statistical signal
    Action: Increase application rate

  Pattern 5: Quality Drought
    Signal: avg quality_log score < 55 across last 5 applications
    Cause:  Consistently weak applications being sent
    Action: Application quality gate must be used
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_rejection_pattern(
    outcome_log: List[Dict],
    quality_log: Optional[List[Dict]] = None,
    cohort_intelligence: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyse outcome_log for systematic rejection patterns.

    Returns dict if a pattern is detected, None if insufficient data (< 2 outcomes)
    or no clear pattern found.

    Keys:
      pattern_type         — internal slug
      severity             — "mild" / "concerning" / "critical"
      alert_title          — short human-readable title
      alert_message        — specific description referencing actual numbers
      recommended_action   — concrete next step
      supporting_data      — dict of stats that support the alert
      consecutive_count    — how many consecutive same-pattern rejections
      pattern_confidence   — 0.0–1.0
    """
    if not outcome_log or len(outcome_log) < 2:
        return None

    outcomes = list(outcome_log)  # don't mutate caller's list

    # ── Pattern 1: ATS Filter Block ──────────────────────────────────────────
    no_response = [o for o in outcomes if o.get("actual_stage") == "no_response"
                   or not o.get("reached_response")]
    if len(no_response) >= 2:
        nr_rate = len(no_response) / len(outcomes)
        severity = "critical" if nr_rate >= 0.7 else ("concerning" if nr_rate >= 0.5 else "mild")
        return {
            "pattern_type":      "ats_filter",
            "severity":          severity,
            "alert_title":       "ATS Filter is blocking you",
            "alert_message":     (
                f"{len(no_response)} of your {len(outcomes)} tracked applications received zero response. "
                f"This strongly suggests an ATS keyword gap — your resume isn't matching the role's "
                f"screening criteria before a human ever sees it."
            ),
            "recommended_action": (
                "Run the ATS Compatibility Scan on your 3 most recent applications. "
                "Add missing keywords directly to your CV skills section (verbatim from JDs). "
                "Reformat: skills section must appear in the top third of the resume."
            ),
            "supporting_data":   {
                "no_response_count":   len(no_response),
                "total_outcomes":      len(outcomes),
                "no_response_rate_pct": round(nr_rate * 100),
            },
            "consecutive_count":    _count_consecutive(outcomes, lambda o: not o.get("reached_response")),
            "pattern_confidence":   round(min(0.5 + (nr_rate - 0.3) * 0.8, 0.97), 2),
        }

    # ── Pattern 2: Phone Screen Stall ────────────────────────────────────────
    phone_screen_stall = [
        o for o in outcomes
        if o.get("reached_response") and not o.get("reached_interview")
    ]
    if len(phone_screen_stall) >= 2:
        ps_rate = len(phone_screen_stall) / max(
            len([o for o in outcomes if o.get("reached_response")]), 1
        )
        severity = "critical" if len(phone_screen_stall) >= 4 else ("concerning" if len(phone_screen_stall) >= 3 else "mild")
        return {
            "pattern_type":      "phone_screen_block",
            "severity":          severity,
            "alert_title":       "You're stalling at Phone Screen",
            "alert_message":     (
                f"{len(phone_screen_stall)} of your applications reached a recruiter call "
                f"but didn't advance. The pivot narrative isn't landing — or your compensation "
                f"expectations are misaligned."
            ),
            "recommended_action": (
                "Refine your 60-second pivot pitch: lead with the specific product decision you "
                "owned (not 'I worked with the PM team'). Rehearse with the Zwilling. "
                "Check: are you stating salary expectations before they ask?"
            ),
            "supporting_data":   {
                "phone_screen_stalls":  len(phone_screen_stall),
                "stall_rate_pct":       round(ps_rate * 100),
                "reached_response":     len([o for o in outcomes if o.get("reached_response")]),
            },
            "consecutive_count":    _count_consecutive(
                outcomes,
                lambda o: o.get("reached_response") and not o.get("reached_interview")
            ),
            "pattern_confidence":   round(min(0.45 + (len(phone_screen_stall) * 0.15), 0.95), 2),
        }

    # ── Pattern 3: Interview Drop-off ────────────────────────────────────────
    interview_dropoff = [
        o for o in outcomes
        if o.get("reached_interview") and not o.get("is_offer")
    ]
    if len(interview_dropoff) >= 2:
        return {
            "pattern_type":      "interview_dropoff",
            "severity":          "concerning" if len(interview_dropoff) >= 3 else "mild",
            "alert_title":       "Getting to interviews — but not offers",
            "alert_message":     (
                f"{len(interview_dropoff)} interviews didn't convert to offers. "
                f"The gap is in technical depth or pivot credibility under pressure."
            ),
            "recommended_action": (
                "Run a fresh mock interview focused on 'technical PM' questions and the "
                "'build vs. buy' scenario. Reviewers consistently flag deep-skills gaps in "
                "late interview stages. One strong case study on a real product decision "
                "can close this gap."
            ),
            "supporting_data":   {
                "interview_dropoffs":  len(interview_dropoff),
                "reached_interview":   len([o for o in outcomes if o.get("reached_interview")]),
            },
            "consecutive_count":    _count_consecutive(
                outcomes,
                lambda o: o.get("reached_interview") and not o.get("is_offer")
            ),
            "pattern_confidence":   round(min(0.40 + (len(interview_dropoff) * 0.18), 0.92), 2),
        }

    # ── Pattern 4: Quality Drought (from quality_log) ────────────────────────
    if quality_log and len(quality_log) >= 3:
        recent_scores = [q.get("score", 0) for q in quality_log[-5:] if isinstance(q.get("score"), (int, float))]
        if recent_scores and (avg := sum(recent_scores) / len(recent_scores)) < 55:
            return {
                "pattern_type":      "quality_drought",
                "severity":          "critical" if avg < 40 else "concerning",
                "alert_title":       "Application quality is consistently low",
                "alert_message":     (
                    f"Your last {len(recent_scores)} applications averaged a quality score of "
                    f"{avg:.0f}/100 — below the threshold where positive outcomes are reliably generated. "
                    f"Quantity cannot compensate for quality at this level."
                ),
                "recommended_action": (
                    "Stop submitting applications below score 65. Use the Quality Gate to "
                    "improve each application before sending. Focus on 3 high-quality "
                    "applications per week, not 10 mediocre ones."
                ),
                "supporting_data":   {
                    "avg_quality_score": round(avg, 1),
                    "applications_analysed": len(recent_scores),
                    "threshold": 65,
                },
                "consecutive_count":    len([s for s in recent_scores if s < 55]),
                "pattern_confidence":   0.88,
            }

    return None


def _count_consecutive(outcomes: List[Dict], predicate) -> int:
    """Count trailing consecutive outcomes matching predicate (most recent first)."""
    count = 0
    for o in reversed(outcomes):
        if predicate(o):
            count += 1
        else:
            break
    return count


def pattern_severity_color(severity: str) -> str:
    return {"mild": "#D97706", "concerning": "#DC2626", "critical": "#7C2D12"}.get(severity, "#555")


def pattern_severity_icon(severity: str) -> str:
    return {"mild": "⚠️", "concerning": "🚨", "critical": "🔴"}.get(severity, "⚠️")
