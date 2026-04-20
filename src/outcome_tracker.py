"""
Outcome Tracker + Calibration Motor
=====================================
The flywheel at the heart of PivotOS.

Every job application has a predicted ROI (from roi_calculator.py).
Every outcome (response / rejection / interview / offer) is recorded.
The Calibration Motor compares prediction vs. reality and updates
the user's personal response rate multiplier over time.

After 5+ outcomes: "Your real response rate is 31% vs. our initial
estimate of 22%. Your ROI scores are recalibrated accordingly."

Outcome stages (ordered by pipeline progression):
  no_response   → applied, never heard back
  viewed        → profile viewed but no call
  phone_screen  → got on a call, then rejected
  first_round   → in-person/video round 1, then rejected
  final_round   → reached final, then rejected or offer
  offer         → received offer

Rejection Intelligence:
  Each stage points to a different root cause + specific fix.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Stage definitions
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_STAGES = [
    "no_response",
    "viewed_no_call",
    "phone_screen",
    "first_round",
    "final_round",
    "offer",
]

STAGE_LABELS = {
    "no_response":    "No response",
    "viewed_no_call": "Viewed — no call",
    "phone_screen":   "Phone screen → rejected",
    "first_round":    "1st round → rejected",
    "final_round":    "Final round → rejected",
    "offer":          "Offer received",
}

STAGE_COLORS = {
    "no_response":    "#5F6B7A",
    "viewed_no_call": "#0A66C2",
    "phone_screen":   "#7A2A8A",
    "first_round":    "#A05A00",
    "final_round":    "#B24020",
    "offer":          "#057642",
}

# Each stage maps to: root cause + specific actions
STAGE_DIAGNOSIS = {
    "no_response": {
        "root_cause": "ATS filter or low keyword match — your application isn't reaching a human",
        "actions": [
            "Run ATS scan on your CV for each rejected job — look for <65 scores",
            "Add exact keywords from the job description to your CV bullets",
            "Switch focus to growth-stage companies (startup/series B-D) — humans read every application",
            "Check your email subject line if applying directly — generic subjects get filtered",
        ],
        "priority_fix": "ATS score",
    },
    "viewed_no_call": {
        "root_cause": "CV/LinkedIn passed the filter but didn't sell the pivot — recruiter moved on",
        "actions": [
            "Rewrite your LinkedIn headline to lead with your target role, not your current title",
            "Add a 3-line 'About' summary that opens with your pivot narrative hook",
            "Ensure your most relevant experience is in the top third of your CV",
            "Consider adding a 'Career Transition' note directly below your name on your CV",
        ],
        "priority_fix": "LinkedIn + CV top section",
    },
    "phone_screen": {
        "root_cause": "Pivot narrative is unclear or unconvincing — recruiter can't sell you internally",
        "actions": [
            "Practice your 60-second pivot pitch until it sounds completely natural",
            "Lead with what you bring TO the role, not what you're leaving behind",
            "Have a specific answer ready for 'why this company specifically?'",
            "Prepare 2-3 concrete examples of transferable work from your current background",
        ],
        "priority_fix": "Pivot narrative + pitch",
    },
    "first_round": {
        "root_cause": "Interview performance — STAR structure weak, or technical gaps surfacing",
        "actions": [
            "Review your Mock Interview score — focus on the lowest dimension",
            "Prepare 5 STAR stories that cover: leadership, failure, cross-functional, impact, learning",
            "Research the company's current product/strategy — show domain awareness",
            "Fill your top skill gap with a Proof-of-Skill project before the next round",
        ],
        "priority_fix": "Interview preparation",
    },
    "final_round": {
        "root_cause": "Close but not chosen — cultural fit, competing candidate, or salary mismatch",
        "actions": [
            "Ask for specific feedback — 'Would you be willing to share what made the other candidate stronger?'",
            "Review your closing statement — did you ask for the role explicitly?",
            "Check if salary expectation misalignment was a factor — research the range before finals",
            "Stay in touch — final-round candidates are often the first call for the next opening",
        ],
        "priority_fix": "Closing + cultural fit signals",
    },
    "offer": {
        "root_cause": None,
        "actions": [
            "Negotiate — 85% of hiring managers expect a counter and have budget to move",
            "Run salary impact analysis before accepting",
            "Get the full comp picture: equity, bonus, benefits, remote policy",
        ],
        "priority_fix": "Negotiation",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Outcome log operations
# ─────────────────────────────────────────────────────────────────────────────

def create_outcome_entry(
    job_id: str,
    job_title: str,
    company: str,
    predicted_roi: Optional[float],
    actual_stage: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Create a single outcome log entry."""
    return {
        "id": job_id,
        "job_title": job_title,
        "company": company,
        "predicted_roi": predicted_roi,
        "actual_stage": actual_stage,
        "reached_response": actual_stage != "no_response",
        "reached_interview": actual_stage in ("first_round", "final_round", "offer"),
        "reached_final": actual_stage in ("final_round", "offer"),
        "is_offer": actual_stage == "offer",
        "notes": notes,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Motor
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(outcome_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare predicted ROI vs. actual outcomes and compute a personal
    calibration multiplier.

    Requires at least 3 outcomes for meaningful calibration.
    Returns calibration dict with personal_response_rate and adjustment_factor.
    """
    if not outcome_log or len(outcome_log) < 3:
        return {
            "calibrated": False,
            "n_outcomes": len(outcome_log),
            "min_required": 3,
            "personal_response_rate": 0.22,  # industry default
            "adjustment_factor": 1.0,
            "insight": None,
        }

    n = len(outcome_log)
    n_responded = sum(1 for o in outcome_log if o.get("reached_response"))
    n_interviewed = sum(1 for o in outcome_log if o.get("reached_interview"))
    n_offers = sum(1 for o in outcome_log if o.get("is_offer"))

    personal_response_rate = n_responded / n
    personal_interview_rate = n_interviewed / n
    personal_offer_rate = n_offers / n

    # Compare to predicted average
    predicted_rois = [o["predicted_roi"] for o in outcome_log if o.get("predicted_roi") is not None]
    avg_predicted = sum(predicted_rois) / len(predicted_rois) if predicted_rois else 0.22

    # Adjustment factor: actual / predicted (clamped)
    if avg_predicted > 0:
        raw_adj = personal_response_rate / avg_predicted
        adjustment_factor = max(0.3, min(2.5, raw_adj))
    else:
        adjustment_factor = 1.0

    # Identify dominant rejection stage
    rejection_stages = [o["actual_stage"] for o in outcome_log if not o.get("is_offer")]
    stage_counts: Dict[str, int] = {}
    for s in rejection_stages:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    dominant_stage = max(stage_counts, key=stage_counts.get) if stage_counts else None
    dominant_pct = (stage_counts.get(dominant_stage, 0) / n * 100) if dominant_stage else 0

    # Generate insight
    if personal_response_rate > avg_predicted * 1.2:
        insight = (
            f"Your real response rate ({personal_response_rate*100:.0f}%) is "
            f"{((personal_response_rate/avg_predicted)-1)*100:.0f}% above our initial estimate. "
            f"Your ROI scores are now calibrated upward."
        )
    elif personal_response_rate < avg_predicted * 0.8:
        insight = (
            f"Your real response rate ({personal_response_rate*100:.0f}%) is below estimate "
            f"({avg_predicted*100:.0f}%). This suggests a systematic issue — likely "
            f"{STAGE_DIAGNOSIS.get(dominant_stage or 'no_response', {}).get('root_cause', 'ATS or narrative')}."
        )
    else:
        insight = (
            f"Your response rate ({personal_response_rate*100:.0f}%) matches predictions closely. "
            f"Model is well-calibrated for your profile."
        )

    return {
        "calibrated": True,
        "n_outcomes": n,
        "personal_response_rate": round(personal_response_rate, 3),
        "personal_interview_rate": round(personal_interview_rate, 3),
        "personal_offer_rate": round(personal_offer_rate, 3),
        "avg_predicted_roi": round(avg_predicted, 3),
        "adjustment_factor": round(adjustment_factor, 2),
        "dominant_rejection_stage": dominant_stage,
        "dominant_rejection_pct": round(dominant_pct, 0),
        "stage_counts": stage_counts,
        "insight": insight,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Application Narrative Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_rejection_pattern(
    outcome_log: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Identify the dominant rejection pattern and return a specific diagnosis
    with actionable fixes. Requires at least 3 outcomes.
    """
    if len(outcome_log) < 3:
        return None

    non_offers = [o for o in outcome_log if not o.get("is_offer")]
    if not non_offers:
        return None

    stage_counts: Dict[str, int] = {}
    for o in non_offers:
        s = o.get("actual_stage", "no_response")
        stage_counts[s] = stage_counts.get(s, 0) + 1

    dominant = max(stage_counts, key=stage_counts.get)
    dominant_n = stage_counts[dominant]
    dominant_pct = dominant_n / len(non_offers) * 100

    diagnosis = STAGE_DIAGNOSIS.get(dominant, STAGE_DIAGNOSIS["no_response"])

    # Severity
    if dominant_pct >= 60:
        severity = "critical"
        severity_label = "Critical pattern detected"
    elif dominant_pct >= 40:
        severity = "high"
        severity_label = "Clear pattern detected"
    else:
        severity = "moderate"
        severity_label = "Emerging pattern"

    return {
        "dominant_stage": dominant,
        "dominant_label": STAGE_LABELS.get(dominant, dominant),
        "dominant_pct": round(dominant_pct, 0),
        "dominant_n": dominant_n,
        "total_rejections": len(non_offers),
        "severity": severity,
        "severity_label": severity_label,
        "root_cause": diagnosis["root_cause"],
        "actions": diagnosis["actions"],
        "priority_fix": diagnosis["priority_fix"],
        "stage_counts": stage_counts,
    }


def get_funnel_stats(outcome_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Conversion funnel from applications to offers."""
    n = len(outcome_log)
    if n == 0:
        return {}
    responded = sum(1 for o in outcome_log if o.get("reached_response"))
    interviewed = sum(1 for o in outcome_log if o.get("reached_interview"))
    final = sum(1 for o in outcome_log if o.get("reached_final"))
    offers = sum(1 for o in outcome_log if o.get("is_offer"))
    return {
        "applied": n,
        "responded": responded,
        "interviewed": interviewed,
        "final": final,
        "offers": offers,
        "response_rate": round(responded / n * 100, 1),
        "interview_rate": round(interviewed / n * 100, 1),
        "offer_rate": round(offers / n * 100, 1) if n else 0,
    }
