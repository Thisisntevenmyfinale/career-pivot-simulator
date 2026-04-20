"""
Brier Score Calibration Engine
================================
The #1 Mistake working with AI backends: not evaluating AI in zero-shot tasks.

This module closes that loop explicitly.

Every time the JD Analyzer predicts P(offer), that prediction is persisted here.
Every time the user records an actual outcome, the prediction is resolved and
the Brier Score is recomputed.

Brier Score = mean( (p_predicted - y_actual)^2 )
  0.00 = perfect calibration
  0.25 = uninformative (always predicting 0.5)
  > 0.25 = worse than guessing

The Reliability Diagram (reliability buckets) asks:
  "Of all the JDs where we predicted 60–70% offer probability,
   what fraction actually led to offers?"
If that fraction is 30%, the model is systematically overconfident.

The Correction Factor adjusts every future prediction:
  p_corrected = p_raw × correction_factor
  correction_factor = empirical_offer_rate / predicted_offer_rate

This is the "evaluate the AI" loop — LLM predictions are never accepted raw.
They are validated against ground-truth outcomes and corrected over time.

Architecture note (for documentation):
  Why not just trust the model? Because gpt-4o-mini has no access to your
  specific company, role level, or regional market. Its P(offer) is a prior
  over all PM applications globally. The Brier calibration learns YOUR prior —
  your specific profile against your specific target market.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SQLite persistence
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS brier_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id       TEXT UNIQUE,
    company             TEXT,
    job_title           TEXT,
    predicted_prob      REAL,       -- 0-100 scale from JD Analyzer
    fit_score           REAL,
    go_no_go            TEXT,
    actual_outcome      INTEGER,    -- NULL until resolved; 1=offer, 0=no offer
    predicted_at        REAL,
    resolved_at         REAL
)
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_TABLE_DDL)
    conn.commit()
    return conn


def log_prediction(
    db_path: str,
    *,
    prediction_id: str,
    company: str,
    job_title: str,
    predicted_prob: float,
    fit_score: Optional[float] = None,
    go_no_go: Optional[str] = None,
) -> None:
    """Persist a JD Analyzer prediction. Called every time analyze_jd() returns a result."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            """INSERT OR IGNORE INTO brier_log
               (prediction_id, company, job_title, predicted_prob, fit_score, go_no_go, predicted_at)
               VALUES (?,?,?,?,?,?,?)""",
            (prediction_id, company, job_title, predicted_prob, fit_score, go_no_go, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def resolve_prediction(db_path: str, prediction_id: str, *, got_offer: bool) -> None:
    """Record the actual outcome once the user logs it."""
    try:
        conn = _get_conn(db_path)
        conn.execute(
            "UPDATE brier_log SET actual_outcome=?, resolved_at=? WHERE prediction_id=?",
            (int(got_offer), time.time(), prediction_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_all_predictions(db_path: str) -> List[Dict[str, Any]]:
    """Return all logged predictions (resolved and unresolved)."""
    try:
        conn = _get_conn(db_path)
        rows = conn.execute(
            "SELECT prediction_id, company, job_title, predicted_prob, fit_score, "
            "go_no_go, actual_outcome, predicted_at, resolved_at FROM brier_log "
            "ORDER BY predicted_at DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "prediction_id": r[0], "company": r[1], "job_title": r[2],
                "predicted_prob": r[3], "fit_score": r[4], "go_no_go": r[5],
                "actual_outcome": r[6], "predicted_at": r[7], "resolved_at": r[8],
                "resolved": r[6] is not None,
            }
            for r in rows
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Calibration computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_brier_stats(db_path: str) -> Dict[str, Any]:
    """
    Compute Brier score and reliability diagram from resolved predictions.
    Requires ≥3 resolved predictions for meaningful statistics.

    Returns:
      brier_score, brier_quality, correction_factor, reliability (list of buckets),
      mean_predicted_pct, mean_actual_pct, bias_direction, n_resolved, n_pending
    """
    all_preds = get_all_predictions(db_path)
    resolved = [p for p in all_preds if p["resolved"]]
    pending  = [p for p in all_preds if not p["resolved"]]

    base = {
        "n_total":    len(all_preds),
        "n_resolved": len(resolved),
        "n_pending":  len(pending),
        "pending":    pending,
    }

    if len(resolved) < 3:
        return {**base, "insufficient_data": True, "min_required": 3}

    # Normalise to 0-1
    preds   = [r["predicted_prob"] / 100.0 for r in resolved]
    actuals = [float(r["actual_outcome"])   for r in resolved]

    # Brier Score
    brier = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(resolved)

    # Mean calibration error
    mean_pred   = sum(preds)   / len(preds)
    mean_actual = sum(actuals) / len(actuals)
    bias        = mean_pred - mean_actual  # positive = model is overconfident

    # Correction factor (clamp to reasonable range)
    correction = (mean_actual / mean_pred) if mean_pred > 0.001 else 1.0
    correction = round(max(0.3, min(3.0, correction)), 3)

    # Reliability diagram — 5 equal-width buckets (0-20, 20-40, … 80-100)
    buckets: Dict[int, Dict] = {i: {"sum_pred": 0.0, "sum_actual": 0.0, "n": 0} for i in range(5)}
    for p, a in zip(preds, actuals):
        b = min(4, int(p * 5))
        buckets[b]["sum_pred"]   += p
        buckets[b]["sum_actual"] += a
        buckets[b]["n"]          += 1

    reliability = []
    for b, v in buckets.items():
        if v["n"] > 0:
            reliability.append({
                "bucket":         f"{b*20}–{(b+1)*20}%",
                "mean_predicted": round(v["sum_pred"]   / v["n"] * 100, 1),
                "mean_actual":    round(v["sum_actual"] / v["n"] * 100, 1),
                "n":              v["n"],
                "gap":            round((v["sum_pred"] / v["n"] - v["sum_actual"] / v["n"]) * 100, 1),
            })

    # Quality labels
    if brier < 0.08:
        quality, quality_color = "Excellent", "#057642"
    elif brier < 0.15:
        quality, quality_color = "Good",      "#D97706"
    elif brier < 0.25:
        quality, quality_color = "Fair",      "#D97706"
    else:
        quality, quality_color = "Poor — worse than guessing", "#DC2626"

    if bias > 0.08:
        bias_dir = "overconfident"
        bias_note = "The model predicts higher offer probabilities than your actual results. Correction factor will shrink future predictions."
    elif bias < -0.08:
        bias_dir = "underconfident"
        bias_note = "The model predicts lower offer probabilities than your actual results. Correction factor will boost future predictions."
    else:
        bias_dir  = "well-calibrated"
        bias_note = "Model predictions are closely aligned with your empirical outcomes."

    return {
        **base,
        "insufficient_data":   False,
        "brier_score":         round(brier, 4),
        "brier_quality":       quality,
        "brier_quality_color": quality_color,
        "mean_predicted_pct":  round(mean_pred   * 100, 1),
        "mean_actual_pct":     round(mean_actual * 100, 1),
        "bias_pct":            round(bias        * 100, 1),
        "bias_direction":      bias_dir,
        "bias_note":           bias_note,
        "correction_factor":   correction,
        "reliability":         reliability,
    }


def apply_correction(raw_prob: float, stats: Optional[Dict]) -> float:
    """Apply calibration correction to a raw AI prediction. Always safe to call."""
    if not stats or stats.get("insufficient_data"):
        return raw_prob
    factor = stats.get("correction_factor", 1.0)
    return round(max(0.0, min(100.0, raw_prob * factor)), 1)
