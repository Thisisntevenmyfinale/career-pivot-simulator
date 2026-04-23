"""
P(offer) Trend Engine
=====================
Computes P(offer) history from outcome log and Brier predictions,
showing measurable improvement as the calibration loop accumulates data.

This module is the quantitative backbone of the "long-term value" story:
every outcome logged makes the system's predictions more accurate.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import datetime
import math


# ─────────────────────────────────────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_p_offer_trend(
    outcome_log: List[Dict],
    base_prob: float = 0.05,
    ops_val: float = 0.0,
    cal_factor: float = 1.0,
    brier_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute P(offer) trend over time from outcome log.

    Returns a dict with:
      - has_data: bool
      - dates: list[str]
      - p_offer_values: list[float]   (one per entry + start baseline)
      - brier_scores: list[float]     (running Brier score per entry)
      - current_p: float
      - start_p: float
      - trend_direction: "up" | "down" | "flat"
      - applications_logged: int
      - response_rate_actual: float   (%)
      - correction_factor: float
      - accuracy_gain_pct: float      (how much better vs baseline)
      - milestone_message: str
    """
    if not outcome_log:
        _fallback_p = round(min(35, max(0.5, base_prob
            * (1.0 + max(0, (ops_val - 50)) / 100)
            * cal_factor * brier_factor * 100)), 1)
        return {
            "has_data": False,
            "current_p": _fallback_p,
        }

    entries = sorted(outcome_log, key=lambda x: x.get("timestamp", ""))

    dates: List[str] = ["Start"]
    p_offer_values: List[float] = [round(base_prob * 100, 1)]
    brier_scores: List[float] = []

    responses = 0
    predictions_sq_err: List[float] = []
    running_cal = 1.0

    for idx, entry in enumerate(entries):
        apps_so_far = idx + 1
        got_response = entry.get("outcome") in ("response", "interview", "offer", "hired")
        if got_response:
            responses += 1

        # Update running calibration factor (needs ≥3 outcomes)
        if apps_so_far >= 3:
            actual_rate = responses / apps_so_far
            predicted_rate = base_prob
            running_cal = actual_rate / max(predicted_rate, 0.001)
            running_cal = max(0.1, min(5.0, running_cal))

        # Track Brier score for the prediction vs outcome
        predicted_p = float(entry.get("predicted_roi", base_prob * 100) or base_prob * 100) / 100
        actual_p = 1.0 if got_response else 0.0
        sq_err = (predicted_p - actual_p) ** 2
        predictions_sq_err.append(sq_err)
        brier = sum(predictions_sq_err) / len(predictions_sq_err)
        brier_scores.append(round(brier, 4))

        # Compute current P(offer) with all factors
        ops = float(entry.get("ops_score", ops_val) or ops_val)
        ops_factor = 1.0 + max(0, (ops - 50)) / 100
        current_p = round(min(35, max(0.5, base_prob * ops_factor * running_cal * brier_factor * 100)), 1)

        # Date label
        ts = entry.get("timestamp", "")
        if ts:
            try:
                d = datetime.datetime.fromisoformat(str(ts))
                dates.append(d.strftime("%b %d"))
            except Exception:
                dates.append(f"App {apps_so_far}")
        else:
            dates.append(f"App {apps_so_far}")

        p_offer_values.append(current_p)

    n = len(entries)
    start_p = p_offer_values[0]
    end_p = p_offer_values[-1]

    trend = (
        "up" if end_p > start_p + 0.3
        else "down" if end_p < start_p - 0.3
        else "flat"
    )

    # Accuracy gain: lower Brier score = better (start is 0.25 for 50/50 random)
    if brier_scores:
        brier_now = brier_scores[-1]
        random_brier = 0.25
        accuracy_gain = round(max(0, (random_brier - brier_now) / random_brier * 100), 1)
    else:
        accuracy_gain = 0.0

    # Milestone message
    response_rate = round(responses / max(n, 1) * 100, 1)
    if n < 3:
        milestone = f"Log {3 - n} more outcome{'s' if 3 - n > 1 else ''} to unlock personal calibration"
    elif n < 10:
        milestone = f"Calibration active · {n} outcomes logged · predictions personalised to your response rate"
    else:
        milestone = f"Full calibration active · {n} outcomes · {accuracy_gain}% more accurate than baseline"

    return {
        "has_data": True,
        "dates": dates,
        "p_offer_values": p_offer_values,
        "brier_scores": brier_scores,
        "current_p": end_p,
        "start_p": start_p,
        "trend_direction": trend,
        "applications_logged": n,
        "responses": responses,
        "response_rate_actual": response_rate,
        "correction_factor": round(running_cal, 2),
        "accuracy_gain_pct": accuracy_gain,
        "brier_latest": round(brier_scores[-1], 4) if brier_scores else None,
        "milestone_message": milestone,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Goal Progress
# ─────────────────────────────────────────────────────────────────────────────

def compute_goal_progress(
    current_p: float,
    target_p: float = 15.0,
    elite_p: float = 25.0,
) -> Dict[str, Any]:
    """
    How far is the user toward their P(offer) goal?

    Tiers:
      - Baseline:  5%  (industry cold-application average)
      - Target:   15%  (strong pivot candidate, warm-intro mix)
      - Elite:    25%  (referral + top OPS + calibrated)
    """
    baseline = 5.0

    if current_p >= elite_p:
        tier = "elite"
        tier_label = "Elite"
        tier_color = "#057642"
    elif current_p >= target_p:
        tier = "target"
        tier_label = "Target Reached"
        tier_color = "#0A66C2"
    elif current_p >= baseline:
        tier = "baseline"
        tier_label = "Building"
        tier_color = "#A05A00"
    else:
        tier = "below"
        tier_label = "Getting Started"
        tier_color = "#B71C1C"

    # Progress toward next tier
    if current_p < target_p:
        next_tier_p = target_p
        next_label = "Target"
        pct_to_next = round((current_p - baseline) / max(target_p - baseline, 0.01) * 100, 1)
    elif current_p < elite_p:
        next_tier_p = elite_p
        next_label = "Elite"
        pct_to_next = round((current_p - target_p) / max(elite_p - target_p, 0.01) * 100, 1)
    else:
        next_tier_p = elite_p
        next_label = "Elite"
        pct_to_next = 100.0

    pct_to_next = max(0.0, min(100.0, pct_to_next))

    return {
        "tier": tier,
        "tier_label": tier_label,
        "tier_color": tier_color,
        "pct_to_next": pct_to_next,
        "next_tier_label": next_label,
        "gap_to_next": round(max(0, next_tier_p - current_p), 1),
        "baseline_p": baseline,
        "target_p": target_p,
        "elite_p": elite_p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loop Stage Summary (for the closed-loop architecture visualizer)
# ─────────────────────────────────────────────────────────────────────────────

def loop_stage_summary(
    cv_profile: Dict,
    jd_result: Dict,
    app_package: Dict,
    app_eval: Dict,
    debate_result: Dict,
    outcome_log: List[Dict],
    calibration_data: Dict,
    brier_stats: Dict,
) -> List[Dict[str, Any]]:
    """
    Summarise the current state of each stage in the closed loop.
    Each stage has: name, status, metric, description.
    """
    stages = []

    # Stage 1: PREDICT
    ops = int(cv_profile.get("ops_score", 0) or 0) if cv_profile else 0
    p_base = 5.0
    p_ops_adj = round(min(35, p_base * (1.0 + max(0, (ops - 50)) / 100)), 1)
    stages.append({
        "num": "1",
        "name": "PREDICT",
        "sub": "Zero-shot P(offer) estimate",
        "status": "done" if cv_profile else "pending",
        "metric": f"{p_ops_adj}% P(offer)" if cv_profile else "Upload CV",
        "metric_color": "#0A66C2" if cv_profile else "rgba(0,0,0,0.30)",
        "detail": (
            f"OPS {ops}/100 · base 5% × {1.0 + max(0,(ops-50)/100):.2f}× = {p_ops_adj}%"
            if cv_profile else
            "Upload CV to get your OPS skill-match score"
        ),
        "action": "Upload CV → run OPS scorer" if not cv_profile else None,
    })

    # Stage 2: GENERATE
    pkg = app_package or {}
    eval_score = int((app_eval or {}).get("overall_score", 0) or 0)
    stages.append({
        "num": "2",
        "name": "GENERATE",
        "sub": "Application via gpt-4o",
        "status": "done" if pkg else "pending",
        "metric": f"{eval_score}/100 quality" if pkg else "No application yet",
        "metric_color": "#057642" if eval_score >= 70 else ("#A05A00" if eval_score >= 55 else "#B71C1C") if pkg else "rgba(0,0,0,0.30)",
        "detail": (
            f"Cover letter + CV rewrite + InMail generated · quality score: {eval_score}/100"
            if pkg else
            "Paste a job description to generate a tailored application"
        ),
        "action": "Paste job description → Generate Application" if not pkg else None,
    })

    # Stage 3: EVALUATE (adversarial debate)
    debate = debate_result or {}
    hire_prob = int(debate.get("hire_probability_pct", 0) or 0)
    stages.append({
        "num": "3",
        "name": "EVALUATE",
        "sub": "3-agent adversarial debate",
        "status": "done" if debate else "pending",
        "metric": f"{hire_prob}% hire probability" if debate else "Not evaluated",
        "metric_color": "#057642" if hire_prob >= 70 else ("#A05A00" if hire_prob >= 50 else "#B71C1C") if debate else "rgba(0,0,0,0.30)",
        "detail": (
            f"Advocate vs. Skeptic vs. Judge · verdict: {debate.get('verdict','—')}"
            if debate else
            "3-agent debate benchmarks your application before you send it"
        ),
        "action": "Run Adversarial Verdict" if pkg and not debate else None,
    })

    # Stage 4: MEASURE (outcome log)
    n_outcomes = len(outcome_log or [])
    n_responses = sum(1 for e in (outcome_log or []) if e.get("outcome") in ("response","interview","offer","hired"))
    rr = round(n_responses / max(n_outcomes, 1) * 100)
    stages.append({
        "num": "4",
        "name": "MEASURE",
        "sub": "Outcome logging",
        "status": "done" if n_outcomes >= 3 else ("partial" if n_outcomes > 0 else "pending"),
        "metric": f"{n_outcomes} logged · {rr}% response rate" if n_outcomes else "No outcomes yet",
        "metric_color": "#057642" if rr >= 20 else ("#A05A00" if rr >= 10 else "#B71C1C") if n_outcomes else "rgba(0,0,0,0.30)",
        "detail": (
            f"{n_outcomes} applications resolved · {n_responses} responses · {3 - n_outcomes} more needed for calibration"
            if 0 < n_outcomes < 3 else
            f"Calibration active · personal correction factor: ×{(calibration_data or {}).get('adjustment_factor',1.0):.2f}"
            if n_outcomes >= 3 else
            "Log outcomes (response/rejection) after submitting applications"
        ),
        "action": "Log outcomes in Outcome Tracker" if n_outcomes < 3 else None,
    })

    # Stage 5: CALIBRATE
    cal = calibration_data or {}
    is_cal = cal.get("calibrated", False)
    adj = float(cal.get("adjustment_factor", 1.0) or 1.0)
    brier = brier_stats or {}
    brier_score = brier.get("brier_score")
    stages.append({
        "num": "5",
        "name": "CALIBRATE",
        "sub": "Brier score + personal correction",
        "status": "done" if is_cal else "pending",
        "metric": (
            f"×{adj:.2f} correction · Brier {brier_score:.3f}"
            if is_cal and brier_score else
            f"×{adj:.2f} correction" if is_cal else
            "Needs 3+ outcomes"
        ),
        "metric_color": "#057642" if is_cal else "rgba(0,0,0,0.30)",
        "detail": (
            f"Your real response rate: {round(float(cal.get('actual_rate',0))*100,1)}% vs. model estimate {round(float(cal.get('predicted_rate',0.05))*100,1)}% · correction ×{adj:.2f} applied to all predictions"
            if is_cal else
            "After 3+ outcomes: predictions auto-correct to your personal response rate"
        ),
        "action": None,
    })

    return stages
