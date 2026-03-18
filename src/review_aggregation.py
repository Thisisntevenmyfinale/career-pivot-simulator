from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from src.review_schemas import (
    ReviewerEvaluation,
    ConsensusResult,
    JudgeMemo,
)


def _get_api_key_optional() -> str:
    try:
        import streamlit as st
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        return ""


def _extract_json_object(text: str) -> Dict[str, Any]:
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


def compute_consensus(
    evaluations: List[ReviewerEvaluation],
) -> ConsensusResult:
    if not evaluations:
        raise ValueError("No evaluations provided")

    strategy_codes = set()
    for eval_obj in evaluations:
        for score in eval_obj.strategy_scores:
            strategy_codes.add(score.strategy_code)

    strategy_codes = sorted(list(strategy_codes))
    score_matrix: Dict[str, Dict[str, float]] = {code: {} for code in strategy_codes}

    for eval_obj in evaluations:
        for score in eval_obj.strategy_scores:
            score_matrix[score.strategy_code][eval_obj.reviewer_persona] = float(score.overall_score)

    mean_scores: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}

    for code in strategy_codes:
        scores_list = list(score_matrix[code].values())
        mean_scores[code] = float(np.mean(scores_list)) if scores_list else 0.0
        std_scores[code] = float(np.std(scores_list)) if scores_list else 0.0

    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    winner_code = ranked[0][0] if ranked else "HYBRID"
    winner_score = ranked[0][1] if ranked else 0.0
    runner_up_code = ranked[1][0] if len(ranked) > 1 else winner_code
    runner_up_score = ranked[1][1] if len(ranked) > 1 else winner_score

    all_stds = [std_scores.get(code, 0.0) for code in strategy_codes]
    mean_std = float(np.mean(all_stds)) if all_stds else 0.0
    consensus_strength = max(0.0, min(100.0, 100.0 - (mean_std * 5.0)))

    major_disagreements: List[Dict[str, Any]] = []
    for code in strategy_codes:
        std = std_scores.get(code, 0.0)
        if std > 15.0:
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

    return ConsensusResult(
        winner_strategy=winner_code,
        winner_score=winner_score,
        runner_up_strategy=runner_up_code,
        runner_up_score=runner_up_score,
        consensus_strength=consensus_strength,
        major_disagreements=major_disagreements,
        strategy_rankings=[(code, score) for code, score in ranked],
    )


def _build_judge_prompt(
    *,
    current_role: str,
    target_role: str,
    consensus_result: ConsensusResult,
    evaluations: List[ReviewerEvaluation],
    gap_summary: str,
) -> str:
    rankings_text = "\n".join(
        [f"{i+1}. {code}: {score:.1f}/100" for i, (code, score) in enumerate(consensus_result.strategy_rankings)]
    )

    disagreements_text = ""
    if consensus_result.major_disagreements:
        disagreements_text = "\nMajor disagreements:\n"
        for d in consensus_result.major_disagreements[:3]:
            disagreements_text += (
                f"- {d['strategy']}: {d['strongest_advocate']} vs {d['strongest_critic']} "
                f"(spread {d['spread']:.0f})\n"
            )

    return f"""
You are the final judge in a career pivot decision system.

Current role: {current_role}
Target role: {target_role}

Consensus rankings:
{rankings_text}

Consensus strength: {consensus_result.consensus_strength:.0f}/100

{disagreements_text}

Gap summary:
{gap_summary}

Return ONLY valid JSON:
{{
  "verdict": "Highly Feasible | Feasible with Conditions | Challenging",
  "recommended_strategy": "{consensus_result.winner_strategy}",
  "executive_summary": "3-4 sentence summary",
  "key_success_factors": ["string", "string", "string"],
  "critical_risks": ["string", "string"],
  "first_30_day_actions": ["string", "string", "string"],
  "interview_narrative": "How to pitch this pivot",
  "success_timeline": "e.g. 6-9 months",
  "confidence_level": "High | Medium | Low"
}}
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

        resp = client.responses.create(
            model=model,
            input=prompt,
        )

        raw_text = (resp.output_text or "").strip()
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
    return {
        "memo": JudgeMemo(
            verdict="Feasible with Conditions",
            recommended_strategy=consensus_result.winner_strategy,
            executive_summary=(
                f"The {consensus_result.winner_strategy} strategy has the strongest overall reviewer support "
                f"for moving from {current_role} to {target_role}."
            ),
            key_success_factors=[
                "Build visible proof of capability",
                "Close the highest-signal gaps first",
                "Tell a coherent transition story",
            ],
            critical_risks=[
                "Weak credibility signal in the market",
                "Timeline drift during transition",
            ],
            first_30_day_actions=[
                "Prioritize the highest-signal missing skills",
                "Start one visible evidence-building project",
                "Refine the pivot narrative",
            ],
            interview_narrative="Frame the pivot as a deliberate move backed by evidence, targeted skill-building, and proof of execution.",
            success_timeline="6-9 months",
            confidence_level="Medium",
        ),
        "source": "Offline (deterministic)",
        "trace": {},
    }


def rerank_after_skill_investment(
    *,
    evaluations: List[ReviewerEvaluation],
    invested_skills: List[str],
    uplift_ratio: float = 0.5,
) -> ConsensusResult:
    boosted_evaluations = []

    for eval_obj in evaluations:
        boosted_scores = []
        for score in eval_obj.strategy_scores:
            boost = 0.0
            if score.strategy_code in {"SKILL_FIRST", "HYBRID"}:
                boost = 8.0 * uplift_ratio
            elif score.strategy_code == "PORTFOLIO":
                boost = 5.0 * uplift_ratio

            boosted_scores.append(
                score.copy(update={"overall_score": min(100.0, float(score.overall_score) + boost)})
            )

        boosted_evaluations.append(
            eval_obj.copy(update={"strategy_scores": boosted_scores})
        )

    return compute_consensus(boosted_evaluations)