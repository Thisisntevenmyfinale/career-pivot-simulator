"""
Application ROI Calculator
===========================
The most important question nobody answers: WHERE should I focus my energy?

Sending 50 applications randomly has a terrible expected value.
Sending 10 targeted applications has 5× the ROI.

This module calculates Expected Interview Rate per Application for each job
and ranks them by effort-adjusted return — so the user focuses on the right opportunities.

Inputs:
- ATS score (pre-computed)
- Hiring velocity (from company intel / market pulse)
- Pivot compatibility (how common is this pivot path at this company type)
- Role fit (O*NET similarity)
- Competition signal (company tier / application volume)

Output:
- Expected Interviews per Application (EI/app)
- Expected Interviews per 4h of effort (EI/4h)
- Priority rank: "High ROI" | "Medium ROI" | "Low ROI" | "Skip"
- Plain-English reasoning
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic ROI Calculation (no LLM — pure Python)
# ─────────────────────────────────────────────────────────────────────────────

COMPETITION_MULTIPLIERS = {
    "enterprise":   0.4,   # FAANG / 10k+ employees — brutal ATS filter
    "mid_market":   0.75,  # 500-5000 employees — moderate competition
    "growth":       0.90,  # Series B-D — actively hiring, less competition
    "startup":      1.0,   # <200 employees — humans read every CV
    "unknown":      0.65,
}

HIRING_SIGNAL_MULTIPLIERS = {
    "strong":      1.30,
    "moderate":    1.00,
    "weak":        0.60,
    "concerning":  0.30,
    "unknown":     0.85,
}

PIVOT_COMPATIBILITY = {
    "high":    1.20,   # company known to hire pivot candidates
    "medium":  1.00,
    "low":     0.60,   # company rarely takes career changers
    "unknown": 0.90,
}


def compute_application_roi(
    job_title: str,
    company: str,
    ats_score: Optional[int] = None,              # 0-100
    fit_percentile: Optional[float] = None,        # 0-100
    hiring_signal: str = "unknown",               # from company intel
    company_stage: str = "unknown",               # startup | growth | mid_market | enterprise
    pivot_compatibility: str = "unknown",          # high | medium | low
    effort_hours: float = 4.0,                    # hours to write + customize
    base_response_rate: float = 0.22,             # industry average: 22%
) -> Dict[str, Any]:
    """
    Calculate expected interview ROI for a single application.

    Returns a dict with EI/app, EI/effort, priority label, and reasoning.
    """
    # ATS factor: below 60 = heavy penalty, above 80 = strong signal
    if ats_score is None:
        ats_factor = 0.80  # unknown → assume average
    elif ats_score >= 85:
        ats_factor = 1.35
    elif ats_score >= 75:
        ats_factor = 1.10
    elif ats_score >= 65:
        ats_factor = 0.90
    elif ats_score >= 55:
        ats_factor = 0.60
    else:
        ats_factor = 0.30  # below 55 = very low chance of passing screen

    # Fit factor: higher O*NET match = more likely to succeed in interview
    if fit_percentile is None:
        fit_factor = 1.00
    elif fit_percentile >= 75:
        fit_factor = 1.25
    elif fit_percentile >= 50:
        fit_factor = 1.05
    elif fit_percentile >= 30:
        fit_factor = 0.90
    else:
        fit_factor = 0.70

    competition_mult = COMPETITION_MULTIPLIERS.get(company_stage, 0.65)
    hiring_mult = HIRING_SIGNAL_MULTIPLIERS.get(hiring_signal, 0.85)
    pivot_mult = PIVOT_COMPATIBILITY.get(pivot_compatibility, 0.90)

    # Expected interview rate for this application
    ei_per_app = base_response_rate * ats_factor * fit_factor * competition_mult * hiring_mult * pivot_mult

    # Clamp to realistic range
    ei_per_app = max(0.01, min(0.85, ei_per_app))

    # ROI per effort
    ei_per_4h = (ei_per_app / effort_hours) * 4.0

    # Priority label
    if ei_per_app >= 0.35:
        priority = "High ROI"
        priority_color = "#057642"
        priority_icon = "🟢"
    elif ei_per_app >= 0.18:
        priority = "Medium ROI"
        priority_color = "#A05A00"
        priority_icon = "🟡"
    elif ei_per_app >= 0.08:
        priority = "Low ROI"
        priority_color = "#7A2A8A"
        priority_icon = "🟠"
    else:
        priority = "Skip"
        priority_color = "#B71C1C"
        priority_icon = "🔴"

    # Build reasoning
    factors = []
    if ats_score is not None:
        if ats_score >= 80:
            factors.append(f"ATS {ats_score} is strong")
        elif ats_score >= 65:
            factors.append(f"ATS {ats_score} is acceptable")
        else:
            factors.append(f"ATS {ats_score} is a bottleneck — fix before applying")
    if hiring_signal in ("strong", "moderate"):
        factors.append(f"company is actively hiring ({hiring_signal} signal)")
    elif hiring_signal in ("weak", "concerning"):
        factors.append(f"hiring signal is {hiring_signal} — risk of slow process or freeze")
    if company_stage == "enterprise":
        factors.append("large company = heavy ATS filter + slow process")
    elif company_stage in ("growth", "startup"):
        factors.append("growth stage = faster process, more pivot-friendly")
    if fit_percentile is not None and fit_percentile >= 70:
        factors.append(f"O*NET fit at {fit_percentile:.0f}th percentile is a strong signal")

    reasoning = ". ".join(factors) + "." if factors else "Limited data — estimate based on industry averages."

    return {
        "job_title": job_title,
        "company": company,
        "ei_per_app": round(ei_per_app, 3),
        "ei_per_app_pct": round(ei_per_app * 100, 1),
        "ei_per_4h": round(ei_per_4h, 3),
        "priority": priority,
        "priority_color": priority_color,
        "priority_icon": priority_icon,
        "reasoning": reasoning,
        "ats_factor": round(ats_factor, 2),
        "fit_factor": round(fit_factor, 2),
        "competition_mult": competition_mult,
        "hiring_mult": hiring_mult,
    }


def rank_applications_by_roi(
    applications: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Takes a list of application ROI dicts and returns them sorted by ei_per_app descending.
    Each dict must have already been computed by compute_application_roi().
    """
    return sorted(applications, key=lambda x: x.get("ei_per_app", 0), reverse=True)


def get_portfolio_roi_summary(
    roi_results: List[Dict[str, Any]],
    pipeline_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Summarise portfolio-level ROI stats for the Command Center.
    """
    if not roi_results:
        return {
            "avg_ei_per_app": 0,
            "expected_interviews_total": 0,
            "high_roi_count": 0,
            "skip_count": 0,
            "top_recommendation": "Run ROI calculator on your applications to see where to focus.",
        }

    avg_ei = sum(r.get("ei_per_app", 0) for r in roi_results) / len(roi_results)
    total_apps = (pipeline_stats or {}).get("total", len(roi_results))
    expected_interviews = avg_ei * total_apps
    high_roi = sum(1 for r in roi_results if r.get("priority") == "High ROI")
    skip = sum(1 for r in roi_results if r.get("priority") == "Skip")

    if avg_ei >= 0.28:
        top_rec = f"Your portfolio is well-targeted. Expected {expected_interviews:.1f} interviews from current applications."
    elif avg_ei >= 0.15:
        top_rec = f"Mid-range ROI. Focus next applications on {high_roi} high-ROI opportunities and improve ATS scores on the others."
    else:
        top_rec = f"ROI is low — most applications are unlikely to convert. Fix ATS scores first, then retarget toward growth-stage companies."

    return {
        "avg_ei_per_app": round(avg_ei, 3),
        "avg_ei_pct": round(avg_ei * 100, 1),
        "expected_interviews_total": round(expected_interviews, 1),
        "high_roi_count": high_roi,
        "skip_count": skip,
        "total_evaluated": len(roi_results),
        "top_recommendation": top_rec,
    }
