from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with a safe zero-vector guard."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _score_from_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Return a 0-100 non-negative cosine score."""
    sim = max(0.0, _cosine_similarity(a, b))
    return float(np.clip(sim * 100.0, 0.0, 100.0))


def simulate_skill_investment(
    matrix: pd.DataFrame,
    *,
    current_role: str,
    target_role: str,
    selected_skills: List[str],
    uplift_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Simulate a counterfactual skill investment.

    The selected current-role skill weights are moved part of the way toward
    the target-role values. This creates a transparent 'what-if' simulator.
    """
    if current_role not in matrix.index:
        raise ValueError(f"Current occupation not found: {current_role}")
    if target_role not in matrix.index:
        raise ValueError(f"Target occupation not found: {target_role}")

    current = matrix.loc[current_role].astype(float).copy()
    target = matrix.loc[target_role].astype(float).copy()

    before_score = _score_from_vectors(current.values, target.values)

    valid_skills = [s for s in selected_skills if s in matrix.columns]
    if not valid_skills:
        return {
            "before_score": before_score,
            "after_score": before_score,
            "uplift": 0.0,
            "applied_skills": [],
            "details_df": pd.DataFrame(
                columns=["skill", "current_before", "target_value", "current_after", "delta_applied"]
            ),
        }

    uplift_ratio = float(np.clip(uplift_ratio, 0.0, 1.0))
    updated = current.copy()

    rows: List[Dict[str, Any]] = []
    for skill in valid_skills:
        cur_val = float(current[skill])
        tgt_val = float(target[skill])
        gap = max(0.0, tgt_val - cur_val)
        new_val = cur_val + (uplift_ratio * gap)
        updated[skill] = new_val

        rows.append(
            {
                "skill": skill,
                "current_before": cur_val,
                "target_value": tgt_val,
                "current_after": float(new_val),
                "delta_applied": float(new_val - cur_val),
            }
        )

    after_score = _score_from_vectors(updated.values, target.values)
    details_df = pd.DataFrame(rows).sort_values("delta_applied", ascending=False).reset_index(drop=True)

    return {
        "before_score": float(before_score),
        "after_score": float(after_score),
        "uplift": float(after_score - before_score),
        "applied_skills": valid_skills,
        "details_df": details_df,
    }


def suggest_best_investment_skills(
    gap_df: pd.DataFrame,
    *,
    top_k: int = 8,
) -> pd.DataFrame:
    """Return top candidate skills for manual investment simulation."""
    df = gap_df.copy()
    required = {"skill", "gap", "current_importance", "target_importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gap_df missing required columns: {sorted(missing)}")

    df["skill"] = df["skill"].astype(str)
    for c in ["gap", "current_importance", "target_importance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df[df["gap"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["skill", "gap", "target_importance", "investment_priority"])

    df["investment_priority"] = df["gap"] * df["target_importance"]
    df = df.sort_values(
        ["investment_priority", "gap", "target_importance"],
        ascending=False,
    ).head(int(top_k))

    return df[["skill", "gap", "target_importance", "investment_priority"]].reset_index(drop=True)