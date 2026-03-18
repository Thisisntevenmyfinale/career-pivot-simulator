"""
Consensus aggregation and Judge memo generation.

This module:
1. Aggregates scores from 5 reviewers across 5 strategies
2. Detects consensus and major disagreements
3. Generates final Judge memo recommending best strategy
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import pandas as pd
import numpy as np

from src.review_schemas import (
    ReviewerEvaluation,
    ConsensusResult,
    JudgeMemo,
)


def _get_api_key_optional() -> str:
    """Return OpenAI API key."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output."""
    if not text or not text.strip():
        raise ValueError("Empty LLM output.")

    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from LLM output.")


# ============================================================
# Consensus Computation (Pure Python)
# ============================================================

def compute_consensus(
    evaluations: List[ReviewerEvaluation],
) -> ConsensusResult:
    """
    Aggregate scores from all reviewers and compute consensus metrics.
    
    This is 100% Python-based, no LLM involved.
    """

    if not evaluations:
        raise ValueError("No evaluations provided")

    # Build a DataFrame: rows=strategies, columns=reviewers, values=scores
    strategy_codes = set()
    for eval_obj in evaluations:
        for score in eval_obj.strategy_scores:
            strategy_codes.add(score.strategy_code)

    strategy_codes = sorted(list(strategy_codes))

    # Score matrix: strategy_code -> {reviewer_persona -> overall_score}
    score_matrix: Dict[str, Dict[str, float]] = {code: {} for code in strategy_codes}

    for eval_obj in evaluations:
        for score in eval_obj.strategy_scores:
            score_matrix[score.strategy_code][eval_obj.reviewer_persona] = score.overall_score

    # Compute mean scores per strategy
    mean_scores: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}

    for code in strategy_codes:
        scores_list = list(score_matrix[code].values())
        if scores_list:
            mean_scores[code] = float(np.mean(scores_list))
            std_scores[code] = float(np.std(scores_list))
        else:
            mean_scores[code] = 0.0
            std_scores[code] = 0.0

    # Rank strategies
    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)

    winner_code = ranked[0][0] if ranked else "HYBRID"
    winner_score = ranked[0][1] if ranked else 0.0
    runner_up_code = ranked[1][0] if len(ranked) > 1 else winner_code
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Compute consensus strength: inverse of variance across reviewers
    # High consensus = low variance = high consensus_strength
    all_stds = [std_scores.get(code, 0.0) for code in strategy_codes]
    mean_std = float(np.mean(all_stds)) if all_stds else 0.0
    consensus_strength = max(0.0, 100.0 - (mean_std * 5.0))  # heuristic scaling

    # Detect major disagreements
    major_disagreements: List[Dict[str, Any]] = []
    for code in strategy_codes:
        std = std_scores.get(code, 0.0)
        if std > 15.0:  # high variance = disagreement
            reviewer_scores = score_matrix[code]
            min_score = min(reviewer_scores.values()) if reviewer_scores else 0.0
            max_score = max(reviewer_scores.values()) if reviewer_scores else 0.0
            
            min_reviewer = [k for k, v in reviewer_scores.items() if v == min_score][0] if reviewer_scores else "Unknown"
            max_reviewer = [k for k, v in reviewer_scores.items() if v == max_score][0] if reviewer_scores else "Unknown"
            
            major_disagreements.append(
                {
                    "strategy": code,
                    "range": f"{min_score:.0f}-{max_score:.0f}",
                    "strongest_advocate": max_reviewer,
                    "strongest_critic": min_reviewer,
                    "spread": float(max_score - min_score),
                }
            )

    major_disagreements.sort(key=lambda x: x["spread"], reverse=True)

    strategy_rankings = [(code, score) for code, score in ranked]

    return ConsensusResult(
        winner_strategy=winner_code,
        winner_score=winner_score,
        runner_up_strategy=runner_up_code,
        runner_up_score=runner_up_score,
        consensus_strength=consensus_strength,
        major_disagreements=major_disagreements,
        strategy_rankings=strategy_rankings,
    )


# ============================================================
# Judge Memo Generation
# ============================================================

def _build_judge_prompt(
    *,
    current_role: str,
    target_role: str,
    consensus_result: ConsensusResult,
    evaluations: List[ReviewerEvaluation],
    gap_summary: str,
) -> str:
    """Build prompt for final Judge memo."""

    rankings_text = "\n".join(
        [f"  {i+1}. {code}: {score:.1f}/100" for i, (code, score) in enumerate(consensus_result.strategy_rankings)]
    )

    disagreements_text = ""
    if consensus_result.major_disagreements:
        disagreements_text = "\n\nMajor disagreements:\n"
        for d in consensus_result.major_disagreements[:3]:
            disagreements_text += f"  - {d['strategy']}: {d['strongest_advocate']} loves it ({d['range']}), {d['strongest_critic']} is skeptical\n"

    return f"""
You are the final Judge in a career pivot evaluation.

Five expert reviewers (Hiring Manager, Recruiter, Portfolio Evaluator, Risk Analyst, Career Coach) 
have evaluated five different strategies for pivoting from {current_role} to {target_role}.

Rankings (by average score):
{rankings_text}

Consensus strength: {consensus_result.consensus_strength:.0f}/100 (higher = more agreement among reviewers)

{disagreements_text}

Gap summary: {gap_summary}

Based on the consensus results above, produce a final recommendation memo.

Return ONLY valid JSON with this exact shape:
{{
  "verdict": "Highly Feasible | Feasible with Conditions | Challenging",
  "recommended_strategy": "{consensus_result.winner_strategy}",
  "executive_summary": "3-4 sentences summarizing the best path forward",
  "key_success_factors": ["factor1", "factor2", "factor3"],
  "critical_risks": ["risk1", "risk2"],
  "first_30_day_actions": ["action1", "action2", "action3"],
  "interview_narrative": "How to pitch this pivot to a hiring manager",
  "success_timeline": "e.g., '6-9 months'",
  "confidence_level": "High | Medium | Low"
}}

Be specific, grounded in the reviewer feedback, and provide actionable guidance.
""".strip()


def generate_judge_memo(
    *,
    current_role: str,
    target_role: str,
    consensus_result: ConsensusResult,
    evaluations: List[ReviewerEvaluation],
    gap_summary: str,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate final Judge memo using LLM + consensus data.
    """

    if not prefer_online:
        return _offline_judge_memo(
            current_role=current_role,
            target_role=target_role,
            consensus_result=consensus_result,
        )

    api_key = _get_api_key_optional()
    if not api_key:
        return _offline_judge_memo(
            current_role=current_role,
            target_role=target_role,
            consensus_result=consensus_result,
        )

    try:
        from openai import OpenAI
    except Exception:
        return _offline_judge_memo(
            current_role=current_role,
            target_role=target_role,
            consensus_result=consensus_result,
        )

    try:
        client = OpenAI(api_key=api_key)

        prompt = _build_judge_prompt(
            current_role=current_role,
            target_role=target_role,
            consensus_result=consensus_result,
            evaluations=evaluations,
            gap_summary=gap_summary,
        )

        resp = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw_text = resp.content[0].text if resp.content else ""
        memo_obj = _extract_json_object(raw_text)

        return {
            "memo": JudgeMemo(
                verdict=memo_obj.get("verdict", "Feasible with Conditions"),
                recommended_strategy=memo_obj.get("recommended_strategy", consensus_result.winner_strategy),
                executive_summary=memo_obj.get("executive_summary", ""),
                key_success_factors=memo_obj.get("key_success_factors", []),
                critical_risks=memo_obj.get("critical_risks", []),
                first_30_day_actions=memo_obj.get("first_30_day_actions", []),
                interview_narrative=memo_obj.get("interview_narrative", ""),
                success_timeline=memo_obj.get("success_timeline", "6-9 months"),
                confidence_level=memo_obj.get("confidence_level", "Medium"),
            ),
            "source": "OpenAI Judge",
            "trace": {"raw": raw_text},
        }

    except Exception as e:
        return {
            "memo": None,
            "source": "Offline (error)",
            "trace": {"error": repr(e)},
        }


def _offline_judge_memo(
    *,
    current_role: str,
    target_role: str,
    consensus_result: ConsensusResult,
) -> Dict[str, Any]:
    """Offline fallback judge memo."""

    return {
        "memo": JudgeMemo(
            verdict="Feasible with Conditions",
            recommended_strategy=consensus_result.winner_strategy,
            executive_summary=f"The {consensus_result.winner_strategy} strategy shows the strongest consensus among reviewers ({consensus_result.consensus_strength:.0f}/100 agreement). It balances risk, timing, and credibility for your transition from {current_role} to {target_role}.",
            key_success_factors=["Build relevant portfolio artifacts", "Maintain consistent skill development", "Network in target industry"],
            critical_risks=["Market saturation for target role", "Skill gaps may be wider than expected"],
            first_30_day_actions=["Identify 2-3 core missing skills", "Start one portfolio project", "Connect with 5 people in target role"],
            interview_narrative="Position this transition as a deliberate, well-researched career move that leverages your transferable strengths.",
            success_timeline="6-9 months",
            confidence_level="Medium",
        ),
        "source": "Offline (deterministic)",
        "trace": {},
    }


# ============================================================
# Counterfactual Re-Ranking
# ============================================================

def rerank_after_skill_investment(
    *,
    evaluations: List[ReviewerEvaluation],
    invested_skills: List[str],
    uplift_ratio: float = 0.5,
) -> ConsensusResult:
    """
    Simulate re-evaluation if user invests in specific skills.
    
    Simple heuristic: boost scores for strategies that emphasize those skills.
    """

    # Clone evaluations and apply modest score boosts to strategies
    # that benefit from the invested skills.

    boosted_evaluations = []
    for eval_obj in evaluations:
        boosted_scores = []
        for score in eval_obj.strategy_scores:
            # Heuristic: strategies like SKILL_FIRST and HYBRID benefit most
            # from early skill investment
            boost = 0.0
            if score.strategy_code in {"SKILL_FIRST", "HYBRID"}:
                boost = 8.0 * uplift_ratio
            elif score.strategy_code == "PORTFOLIO":
                boost = 5.0 * uplift_ratio
            
            new_overall = min(100.0, score.overall_score + boost)
            boosted_score = score.copy(update={"overall_score": new_overall})
            boosted_scores.append(boosted_score)

        boosted_eval = eval_obj.copy(update={"strategy_scores": boosted_scores})
        boosted_evaluations.append(boosted_eval)

    return compute_consensus(boosted_evaluations)
