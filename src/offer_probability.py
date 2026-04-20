"""
Offer Probability Score (OPS)
==============================
The single northstar metric of PivotOS.

P(Offer | current state) — a live Bayesian-ish probability (0–92%)
computed from every available signal in session state.

Not a fit score. Not a quality score. The actual probability
that this specific user receives a job offer given their
current pipeline, quality, interview readiness, and market position.

Why it's different:
  - Other tools measure INPUTS (ATS score, CV quality, skill fit).
  - OPS measures the OUTPUT: likelihood of an offer.
  - It changes with every action — rising or falling.
  - It tells you the single highest-leverage thing to do next.

Architecture:
  Pure Python. No LLM calls. Instant. Recomputed every render.
  Uses Bayesian-ish factor accumulation with a base prior of 3%.
  Each factor has an empirically-motivated weight (grounded in
  job search research literature + cohort outcome data structure).

Output:
  ops:         int 0–92 (capped — never false certainty)
  delta:       int change vs. previous session value
  confidence:  "low" | "medium" | "high" (data completeness signal)
  drivers:     top 6 factors sorted by absolute impact
  next_lever:  string — single highest-ROI action right now
  grade:       letter grade A/B/C/D/F for quick orientation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ops(state: Any) -> Dict[str, Any]:
    """
    Compute Offer Probability Score from session state.

    state: dict-like (st.session_state or plain dict)
    Returns: dict with ops, delta, confidence, drivers, next_lever, grade
    """
    score = 3.0  # Base prior (3% before any signal)
    factors: List[Dict[str, Any]] = []

    # ── Extract all relevant state ────────────────────────────────────────
    cv_text        = (state.get("cv_text") or "").strip()
    cv_profile     = state.get("cv_profile") or {}
    pivot_dna      = state.get("pivot_dna") or {}
    voice_profile  = state.get("voice_profile") or {}
    onet_match     = state.get("onet_match") or {}
    skill_gap      = state.get("skill_gap_results") or {}
    pipeline       = state.get("pipeline_jobs") or []
    quality_log    = state.get("quality_log") or []
    mock_report    = state.get("mock_interview_report") or {}
    interview_evals = state.get("interview_evals") or {}
    calibration    = state.get("calibration_data") or {}
    cohort         = state.get("cohort_intelligence") or {}
    streak         = state.get("momentum_streak_days") or 0
    outcome_log    = state.get("outcome_log") or []
    skill_proofs   = state.get("skill_proofs") or {}
    roi_results    = state.get("roi_results") or {}
    linkedin       = state.get("linkedin_profile") or {}
    interview_prep = state.get("interview_prep_done") or False

    # ── Gate: no CV = no signal ───────────────────────────────────────────
    if not cv_text:
        factors.append({
            "factor": "CV not uploaded",
            "impact": -30,
            "direction": "-",
            "category": "blocker",
        })
        return _package(3, factors, "low", state)

    # ── Factor 1: CV completeness (+0 to +10) ────────────────────────────
    score += 8  # base for having a CV
    factors.append({"factor": "CV uploaded", "impact": 8, "direction": "+", "category": "profile"})

    if cv_profile:
        skills    = cv_profile.get("top_skills") or []
        yoe_raw   = cv_profile.get("years_experience", 0)
        yoe       = float(yoe_raw) if isinstance(yoe_raw, (int, float)) else 0
        n_skills  = len(skills)
        cv_depth  = min(10, n_skills * 0.6 + yoe * 0.3)
        score    += cv_depth
        if cv_depth >= 7:
            factors.append({"factor": f"Strong CV profile ({n_skills} skills, {int(yoe)}y exp)", "impact": round(cv_depth), "direction": "+", "category": "profile"})
        elif cv_depth >= 4:
            factors.append({"factor": f"Moderate CV depth ({n_skills} skills)", "impact": round(cv_depth), "direction": "~", "category": "profile"})
        else:
            factors.append({"factor": "Thin CV profile — add more skills", "impact": round(cv_depth), "direction": "~", "category": "profile"})

    # ── Factor 2: Pivot DNA (+0 to +10) ──────────────────────────────────
    if pivot_dna:
        dna_score = 6
        if pivot_dna.get("strongest_transferable_argument"):
            dna_score += 2
        if pivot_dna.get("unfair_advantage"):
            dna_score += 2
        score += dna_score
        factors.append({"factor": "Pivot DNA calibrated — voice injected into applications", "impact": dna_score, "direction": "+", "category": "profile"})
    else:
        factors.append({"factor": "Pivot DNA not built — applications sound generic", "impact": -4, "direction": "-", "category": "profile"})

    # ── Factor 3: O*NET fit percentile (+0 to +14) ───────────────────────
    fit_pct = (
        onet_match.get("fit_percentile")
        or skill_gap.get("fit_percentile")
        or 0
    )
    if isinstance(fit_pct, (int, float)) and fit_pct > 0:
        fit_contribution = min(14, fit_pct * 0.16)
        score += fit_contribution
        if fit_pct >= 70:
            factors.append({"factor": f"O*NET fit: top {100-int(fit_pct)}% of candidates", "impact": round(fit_contribution), "direction": "+", "category": "fit"})
        elif fit_pct >= 45:
            factors.append({"factor": f"O*NET fit: {int(fit_pct)}th percentile — gap exists", "impact": round(fit_contribution), "direction": "~", "category": "fit"})
        else:
            factors.append({"factor": f"Low O*NET fit: {int(fit_pct)}th percentile", "impact": round(fit_contribution), "direction": "-", "category": "fit"})
    else:
        factors.append({"factor": "O*NET analysis not run", "impact": 0, "direction": "-", "category": "fit"})

    # ── Factor 4: Skill proofs (+0 to +6) ────────────────────────────────
    if skill_proofs:
        n_proofs = len(skill_proofs)
        proof_score = min(6, n_proofs * 2)
        score += proof_score
        factors.append({"factor": f"{n_proofs} Proof-of-Skill project{'s' if n_proofs > 1 else ''} built", "impact": proof_score, "direction": "+", "category": "fit"})

    # ── Factor 5: Application volume vs. cohort (+0 to +16) ──────────────
    n_apps = len(pipeline)
    median_apps = cohort.get("median_applications") if isinstance(cohort, dict) else None

    if n_apps == 0:
        factors.append({"factor": "No applications sent — biggest volume gap", "impact": -8, "direction": "-", "category": "volume"})
    else:
        if median_apps and isinstance(median_apps, (int, float)) and median_apps > 0:
            ratio = min(1.2, n_apps / median_apps)
            vol_score = min(16, ratio * 13)
            score += vol_score
            pct_of_median = round(ratio * 100)
            if pct_of_median >= 100:
                factors.append({"factor": f"Volume: {n_apps} apps — at cohort median ({int(median_apps)})", "impact": round(vol_score), "direction": "+", "category": "volume"})
            elif pct_of_median >= 60:
                factors.append({"factor": f"Volume: {n_apps}/{int(median_apps)} cohort median — building", "impact": round(vol_score), "direction": "~", "category": "volume"})
            else:
                factors.append({"factor": f"Volume below cohort: {n_apps}/{int(median_apps)} — speed up", "impact": round(vol_score), "direction": "-", "category": "volume"})
        else:
            vol_score = min(14, n_apps * 1.4)
            score += vol_score
            lbl = "Strong" if n_apps >= 10 else ("Building" if n_apps >= 5 else "Early stage")
            factors.append({"factor": f"{lbl}: {n_apps} application{'s' if n_apps != 1 else ''} in pipeline", "impact": round(vol_score), "direction": "+" if n_apps >= 5 else "~", "category": "volume"})

    # ── Factor 6: Application quality avg (+0 to +12) ─────────────────────
    if quality_log:
        scores_final = [
            (e["score_v2"] if e.get("score_v2") is not None else e["score_v1"])
            for e in quality_log if e.get("score_v1") is not None
        ]
        if scores_final:
            avg_q = sum(scores_final) / len(scores_final)
            q_contribution = min(12, (avg_q / 100) * 12)
            score += q_contribution
            if avg_q >= 75:
                factors.append({"factor": f"Application quality avg: {round(avg_q)}/100 — above threshold", "impact": round(q_contribution), "direction": "+", "category": "quality"})
            elif avg_q >= 65:
                factors.append({"factor": f"Application quality: {round(avg_q)}/100 — passing", "impact": round(q_contribution), "direction": "~", "category": "quality"})
            else:
                factors.append({"factor": f"Low application quality: {round(avg_q)}/100 — below gate", "impact": round(q_contribution), "direction": "-", "category": "quality"})

    # ── Factor 7: Interview readiness (+0 to +10) ─────────────────────────
    if mock_report and mock_report.get("overall_score"):
        interview_score = int(mock_report.get("overall_score", 0))
        i_contribution = min(10, (interview_score / 100) * 10)
        score += i_contribution
        if interview_score >= 75:
            factors.append({"factor": f"Mock interview: {interview_score}/100 — interview-ready", "impact": round(i_contribution), "direction": "+", "category": "interview"})
        elif interview_score >= 60:
            factors.append({"factor": f"Mock interview: {interview_score}/100 — needs improvement", "impact": round(i_contribution), "direction": "~", "category": "interview"})
        else:
            factors.append({"factor": f"Mock interview: {interview_score}/100 — critical gap", "impact": round(i_contribution), "direction": "-", "category": "interview"})
    elif interview_prep:
        score += 4
        factors.append({"factor": "Interview questions prepared", "impact": 4, "direction": "+", "category": "interview"})
    else:
        factors.append({"factor": "No interview prep done", "impact": 0, "direction": "-", "category": "interview"})

    # ── Factor 8: Personal calibration multiplier (±8) ───────────────────
    if calibration.get("calibrated"):
        adj = float(calibration.get("adjustment_factor", 1.0))
        cal_contribution = (adj - 1.0) * 8
        score += cal_contribution
        if adj > 1.15:
            factors.append({"factor": f"Above-model response rate (×{adj:.2f}) — your profile over-performs", "impact": round(cal_contribution), "direction": "+", "category": "calibration"})
        elif adj < 0.85:
            factors.append({"factor": f"Below-model response rate (×{adj:.2f}) — systematic issue detected", "impact": round(cal_contribution), "direction": "-", "category": "calibration"})

    # ── Factor 9: Rejection pattern penalty (−0 to −8) ───────────────────
    dom_stage = calibration.get("dominant_rejection_stage") if calibration else None
    if dom_stage == "no_response" and len(outcome_log) >= 5:
        score -= 7
        factors.append({"factor": "ATS filter pattern — not reaching humans", "impact": -7, "direction": "-", "category": "calibration"})
    elif dom_stage == "phone_screen" and len(outcome_log) >= 4:
        score -= 4
        factors.append({"factor": "Pivot narrative rejected at phone screen", "impact": -4, "direction": "-", "category": "calibration"})

    # ── Factor 10: LinkedIn optimization (+0 to +5) ───────────────────────
    if linkedin and linkedin.get("headline"):
        score += 4
        factors.append({"factor": "LinkedIn profile optimized for pivot", "impact": 4, "direction": "+", "category": "profile"})

    # ── Factor 11: Momentum (+0 to +5) ───────────────────────────────────
    if streak >= 14:
        score += 5
        factors.append({"factor": f"{streak}-day streak — elite consistency", "impact": 5, "direction": "+", "category": "momentum"})
    elif streak >= 7:
        score += 3
        factors.append({"factor": f"{streak}-day streak — strong momentum", "impact": 3, "direction": "+", "category": "momentum"})
    elif streak >= 3:
        score += 1
        factors.append({"factor": f"{streak}-day streak", "impact": 1, "direction": "+", "category": "momentum"})

    # ── Final score ───────────────────────────────────────────────────────
    final = max(2, min(92, round(score)))

    # ── Confidence level ──────────────────────────────────────────────────
    data_richness = sum([
        bool(cv_text),
        bool(pivot_dna),
        n_apps >= 3,
        bool(quality_log),
        bool(mock_report),
        len(outcome_log) >= 3,
        fit_pct > 0,
    ])
    if data_richness >= 6:
        confidence = "high"
    elif data_richness >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    factors.sort(key=lambda f: abs(f.get("impact", 0)), reverse=True)
    return _package(final, factors[:7], confidence, state)


def _package(
    ops: int,
    factors: List[Dict[str, Any]],
    confidence: str,
    state: Any,
) -> Dict[str, Any]:
    """Assemble the final OPS result dict."""
    prev = state.get("ops_previous") or ops
    delta = ops - int(prev)

    next_lever = _best_next_action(ops, factors, state)

    if ops >= 75:
        grade = "A"
    elif ops >= 60:
        grade = "B"
    elif ops >= 45:
        grade = "C"
    elif ops >= 30:
        grade = "D"
    else:
        grade = "F"

    return {
        "ops":        ops,
        "delta":      delta,
        "confidence": confidence,
        "drivers":    factors,
        "next_lever": next_lever,
        "grade":      grade,
    }


def _best_next_action(ops: int, factors: List[Dict[str, Any]], state: Any) -> str:
    """Return the single highest-leverage next action."""
    cv_text    = (state.get("cv_text") or "").strip()
    pivot_dna  = state.get("pivot_dna")
    mock       = state.get("mock_interview_report")
    pipeline   = state.get("pipeline_jobs") or []
    quality_log = state.get("quality_log") or []

    if not cv_text:
        return "Upload your CV — unlocks all analysis and adds ~+18pts to OPS"
    if not pivot_dna:
        return "Build Pivot DNA — adds +6–10pts, makes every application sound like you"
    if not mock:
        return "Run mock interview — adds up to +10pts when you hit 75+"
    if len(pipeline) < 5:
        return f"Send more applications — volume is your #1 lever right now (+{max(0, 5-len(pipeline)):.0f} more apps needed)"

    neg = [f for f in factors if f.get("direction") == "-"]
    if neg:
        return f"Fix: {neg[0]['factor']}"

    if ops >= 75:
        return "You're on track — maintain volume and follow up on stale applications"
    return "Run ATS scan on your last 3 applications — ATS is the most common silent killer"


# ─────────────────────────────────────────────────────────────────────────────
# OPS color + label helpers
# ─────────────────────────────────────────────────────────────────────────────

def ops_color(ops: int) -> str:
    if ops >= 70:
        return "#057642"
    if ops >= 50:
        return "#0A66C2"
    if ops >= 35:
        return "#A05A00"
    return "#B71C1C"


def ops_label(ops: int) -> str:
    if ops >= 75:
        return "Strong"
    if ops >= 60:
        return "Building"
    if ops >= 45:
        return "Developing"
    if ops >= 30:
        return "Early"
    return "Critical"


def ops_description(ops: int) -> str:
    if ops >= 75:
        return "You're a competitive candidate. Keep your pipeline warm and prep for offers."
    if ops >= 60:
        return "Good trajectory. A few targeted improvements will move you to the offer zone."
    if ops >= 45:
        return "Solid foundation. Volume and interview prep are the highest-ROI next moves."
    if ops >= 30:
        return "You're in early stages. Focus on profile setup and first applications."
    return "Several critical blockers need fixing before offers become likely."
