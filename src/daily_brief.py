"""
Daily Pivot Brief
=================
Every morning: one screen, three actions, zero decision fatigue.

The Daily Brief is the entry point into PivotOS. It answers the
single most important question a job seeker has every morning:
"What should I do today?"

It is fully deterministic — no LLM, instant, always available.
It pulls from current session state and computes:
  1. Top 3 actions ranked by expected ROI impact
  2. Pipeline health snapshot (what needs follow-up)
  3. Momentum signal (streak / warning / celebration)
  4. Market note (if cohort data available)
  5. One sharp motivational line based on actual progress

Architecture: Pure Python. No API calls. Recomputed each morning.
Cached by date — doesn't recompute mid-day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Action scoring (deterministic priority logic)
# ─────────────────────────────────────────────────────────────────────────────

def _score_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate all possible actions and return them ranked by impact score (0-100).
    Each action has: title, description, why, score, category, cta_label.
    """
    actions = []
    today = date.today().isoformat()

    pipeline = state.get("pipeline_jobs") or []
    cv_text = state.get("cv_text") or ""
    pivot_dna = state.get("pivot_dna")
    skill_gap = state.get("skill_gap_results")
    outcome_log = state.get("outcome_log") or []
    calibration = state.get("calibration_data") or {}
    mock_report = state.get("mock_interview_report")
    interview_evals = state.get("interview_evals") or {}
    roi_results = state.get("roi_results") or {}
    cohort = state.get("cohort_intelligence")

    # ── No CV uploaded ─────────────────────────────────────────────────────
    if not cv_text.strip():
        actions.append({
            "title": "Upload your CV",
            "description": "Everything starts here. O*NET analysis, Pivot DNA, and ROI scores all depend on your CV.",
            "why": "Without CV analysis, all recommendations are generic.",
            "score": 100,
            "category": "setup",
            "cta_label": "Upload CV in sidebar",
            "icon": "CV",
        })
        return actions[:3]

    # ── No Pivot DNA ───────────────────────────────────────────────────────
    if not pivot_dna:
        actions.append({
            "title": "Build your Pivot DNA",
            "description": "Calibrate your voice so every generated output sounds like you, not ChatGPT.",
            "why": "DNA-injected cover letters have measurably different tone — hiring managers notice.",
            "score": 95,
            "category": "setup",
            "cta_label": "Build Pivot DNA in sidebar",
            "icon": "DN",
        })

    # ── Pipeline follow-ups overdue ────────────────────────────────────────
    stale_days = 7
    for job in pipeline:
        if job.get("status") in ("applied", "viewed"):
            updated = job.get("date_updated") or job.get("date_added") or ""
            if updated:
                try:
                    delta = (date.today() - date.fromisoformat(updated)).days
                    if delta >= stale_days:
                        actions.append({
                            "title": f"Follow up: {job.get('title','Role')} at {job.get('company','Company')}",
                            "description": f"Applied {delta} days ago with no update. A targeted follow-up email gets a response 28% of the time.",
                            "why": f"{delta} days since last contact — follow-up window is now.",
                            "score": 88 + min(delta - stale_days, 10),
                            "category": "pipeline",
                            "cta_label": "View in pipeline",
                            "icon": "FU",
                        })
                except ValueError:
                    pass

    # ── ATS bottleneck detected ────────────────────────────────────────────
    calibration_data = calibration if isinstance(calibration, dict) else {}
    dominant_stage = calibration_data.get("dominant_rejection_stage")
    if dominant_stage == "no_response" and len(outcome_log) >= 3:
        actions.append({
            "title": "Fix ATS scores — you're getting filtered before humans see you",
            "description": "Your rejection pattern shows no-response as the dominant outcome. Run ATS scan on your last 3 applications.",
            "why": "ATS fix has the highest ROI of any action when this pattern exists.",
            "score": 93,
            "category": "diagnosis",
            "cta_label": "Run ATS scan in Execute tab",
            "icon": "AT",
        })
    elif dominant_stage == "phone_screen" and len(outcome_log) >= 3:
        actions.append({
            "title": "Sharpen your pivot pitch — recruiters aren't buying it yet",
            "description": "You're reaching phone screens but getting rejected. Practice your 60-second pivot narrative today.",
            "why": "Phone screen rejection = narrative problem. One focused practice session fixes this.",
            "score": 90,
            "category": "diagnosis",
            "cta_label": "Open Pivot-Zwilling for practice",
            "icon": "PI",
        })
    elif dominant_stage == "first_round" and len(outcome_log) >= 3:
        actions.append({
            "title": "Interview performance needs work — run a mock interview",
            "description": "First-round rejections signal weak STAR answers or technical gaps.",
            "why": "Your interview score directly predicts whether you advance.",
            "score": 90,
            "category": "diagnosis",
            "cta_label": "Start mock interview in Interview tab",
            "icon": "MK",
        })

    # ── Interview prep incomplete ──────────────────────────────────────────
    if not mock_report and cv_text.strip():
        actions.append({
            "title": "Run your first mock interview",
            "description": "You haven't done a mock interview yet. Candidates who practice score 23% higher in real interviews.",
            "why": "Interview preparation has the highest leverage once you're getting responses.",
            "score": 75,
            "category": "interview",
            "cta_label": "Start mock interview in Interview tab",
            "icon": "MK",
        })
    elif mock_report:
        score = mock_report.get("overall_score", 0)
        if score < 70:
            actions.append({
                "title": f"Improve mock interview score ({score}/100)",
                "description": f"Your last mock score was {score}/100. Focus on: {mock_report.get('top_improvements',['STAR structure'])[0]}",
                "why": "Interview score is a leading indicator of offer rate.",
                "score": 72,
                "category": "interview",
                "cta_label": "Re-run mock interview",
                "icon": "MK",
            })

    # ── Skill gap — top gap has no proof-of-skill ─────────────────────────
    skill_proofs = state.get("skill_proofs") or {}
    if skill_gap and skill_gap.get("gaps"):
        top_gap = skill_gap["gaps"][0] if skill_gap["gaps"] else None
        if top_gap and top_gap.get("skill") not in skill_proofs:
            actions.append({
                "title": f"Build portfolio proof for: {top_gap.get('skill','top gap')}",
                "description": "Your top skill gap has no portfolio artifact. Generate a Proof-of-Skill project you can complete in hours.",
                "why": "Demonstrating gaps-in-progress is more convincing than claiming you'll learn.",
                "score": 68,
                "category": "skills",
                "cta_label": "Generate Proof-of-Skill in Execute tab",
                "icon": "PS",
            })

    # ── No applications in pipeline ────────────────────────────────────────
    if not pipeline and cv_text.strip():
        actions.append({
            "title": "Send your first application",
            "description": "Your profile is ready. The best time to apply was yesterday. The second best time is today.",
            "why": "Nothing happens until you apply. ROI analysis suggests starting with growth-stage companies.",
            "score": 85,
            "category": "apply",
            "cta_label": "Use Quick Apply or paste a job",
            "icon": "AP",
        })

    # ── Log outcome for pending applications ──────────────────────────────
    no_outcome_jobs = [
        j for j in pipeline
        if j.get("status") in ("rejected", "offer")
        and not any(o.get("id") == j.get("id") for o in outcome_log)
    ]
    if no_outcome_jobs:
        j = no_outcome_jobs[0]
        actions.append({
            "title": f"Log outcome for: {j.get('title','Role')} at {j.get('company','')}",
            "description": "Recording outcomes calibrates your personal ROI model — making future predictions more accurate for you.",
            "why": f"You have {len(no_outcome_jobs)} applications with unlogged outcomes.",
            "score": 80,
            "category": "tracking",
            "cta_label": "Log in Outcome Tracker",
            "icon": "OT",
        })

    # ── Cohort benchmark: behind pace ─────────────────────────────────────
    if cohort and isinstance(cohort, dict):
        median_apps = cohort.get("median_applications")
        total_apps = len(pipeline)
        if median_apps and isinstance(median_apps, (int, float)) and total_apps < median_apps * 0.5:
            actions.append({
                "title": "Increase application velocity",
                "description": f"Cohort median is {median_apps} applications. You have {total_apps}. You're below pace.",
                "why": "Application volume is a direct lever on expected interviews.",
                "score": 70,
                "category": "strategy",
                "cta_label": "Use Quick Apply to increase volume",
                "icon": "VO",
            })

    # Sort by score descending, deduplicate categories
    actions.sort(key=lambda x: x["score"], reverse=True)
    seen_cats: set = set()
    deduped = []
    for a in actions:
        if a["category"] not in seen_cats or a["category"] == "pipeline":
            deduped.append(a)
            seen_cats.add(a["category"])

    return deduped[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Momentum signal
# ─────────────────────────────────────────────────────────────────────────────

def _compute_momentum_signal(state: Dict[str, Any]) -> Dict[str, Any]:
    streak = state.get("momentum_streak_days") or 0
    last_date_str = state.get("momentum_last_date") or ""
    pipeline = state.get("pipeline_jobs") or []
    outcome_log = state.get("outcome_log") or []

    # Check if streak is alive
    streak_alive = False
    if last_date_str:
        try:
            last = date.fromisoformat(last_date_str)
            streak_alive = (date.today() - last).days <= 1
        except ValueError:
            pass

    if not streak_alive and streak > 0:
        # Streak broken
        return {
            "type": "warning",
            "label": f"{streak}-day streak lost",
            "detail": "Log an activity today to start a new streak.",
            "color": "#B71C1C",
        }
    elif streak >= 14:
        return {
            "type": "celebration",
            "label": f"{streak}-day streak",
            "detail": "Outstanding consistency. Candidates with 14+ day streaks are 3x more likely to land interviews.",
            "color": "#057642",
        }
    elif streak >= 7:
        return {
            "type": "strong",
            "label": f"{streak}-day streak",
            "detail": "Strong momentum. Keep it up.",
            "color": "#A05A00",
        }
    elif streak >= 3:
        return {
            "type": "building",
            "label": f"{streak}-day streak",
            "detail": "Building momentum.",
            "color": "#0A66C2",
        }
    else:
        apps_today = sum(
            1 for j in pipeline
            if j.get("date_added") == date.today().isoformat()
        )
        if apps_today > 0:
            return {
                "type": "active",
                "label": f"{apps_today} application{'s' if apps_today > 1 else ''} today",
                "detail": "Good start. Log your activity to start a streak.",
                "color": "#0A66C2",
            }
        return {
            "type": "idle",
            "label": "No streak yet",
            "detail": "Log any pivot activity to start your streak.",
            "color": "#5F6B7A",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = state.get("pipeline_jobs") or []
    if not pipeline:
        return {"empty": True}

    today = date.today()
    active = [j for j in pipeline if j.get("status") not in ("rejected", "withdrawn", "offer")]
    stale = []
    for j in active:
        upd = j.get("date_updated") or j.get("date_added") or ""
        try:
            delta = (today - date.fromisoformat(upd)).days
            if delta >= 7:
                stale.append({**j, "_days_stale": delta})
        except ValueError:
            pass

    interviews = [j for j in pipeline if j.get("status") in ("first_round", "final_round")]
    offers = [j for j in pipeline if j.get("status") == "offer"]

    return {
        "empty": False,
        "total": len(pipeline),
        "active": len(active),
        "stale_count": len(stale),
        "stale_jobs": stale[:2],
        "interview_count": len(interviews),
        "offer_count": len(offers),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Motivational line (data-driven, not generic)
# ─────────────────────────────────────────────────────────────────────────────

def _motivational_line(state: Dict[str, Any]) -> str:
    outcome_log = state.get("outcome_log") or []
    pipeline = state.get("pipeline_jobs") or []
    mock = state.get("mock_interview_report")
    cohort = state.get("cohort_intelligence") or {}
    calibration = state.get("calibration_data") or {}
    streak = state.get("momentum_streak_days") or 0

    n_apps = len(pipeline)
    n_outcomes = len(outcome_log)
    n_interviews = sum(1 for o in outcome_log if o.get("reached_interview"))
    median_apps = cohort.get("median_applications") if isinstance(cohort, dict) else None
    adj = calibration.get("adjustment_factor", 1.0) if isinstance(calibration, dict) else 1.0

    if n_outcomes >= 5 and n_interviews >= 1:
        return f"You've reached {n_interviews} interview{'s' if n_interviews > 1 else ''} from {n_apps} applications. That's real traction."
    elif median_apps and n_apps >= median_apps:
        return f"You've matched the cohort median of {median_apps} applications. The offer is a statistics game from here."
    elif adj > 1.2:
        return f"Your real response rate is {calibration.get('personal_response_rate',0)*100:.0f}% — above model prediction. Your profile is working."
    elif streak >= 7:
        return f"Seven consecutive days of focus. Most people quit before this point."
    elif n_apps >= 10:
        return f"{n_apps} applications in. The data is starting to mean something."
    else:
        return "Every application is a data point. The model gets sharper with each one."


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_brief(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the Daily Pivot Brief from current state.
    Fully deterministic — no LLM, instant.

    Returns dict with: actions, momentum, pipeline, motivational_line, generated_at.
    """
    actions = _score_actions(state)
    momentum = _compute_momentum_signal(state)
    pipeline = _pipeline_snapshot(state)
    motivational = _motivational_line(state)

    today = date.today()
    day_name = today.strftime("%A")
    date_str = today.strftime("%d %B %Y")

    return {
        "generated_at": today.isoformat(),
        "day_name": day_name,
        "date_str": date_str,
        "actions": actions,
        "momentum": momentum,
        "pipeline": pipeline,
        "motivational_line": motivational,
    }
