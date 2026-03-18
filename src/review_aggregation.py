from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from src.review_schemas import ConsensusResult, JudgeMemo, ReviewerEvaluation


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_consensus(
    evaluations: List[ReviewerEvaluation],
) -> ConsensusResult:
    if not evaluations:
        raise ValueError("No evaluations provided")

    strategy_codes = sorted(
        {
            score.strategy_code
            for ev in evaluations
            for score in ev.strategy_scores
        }
    )
    if not strategy_codes:
        raise ValueError("No strategy scores found")

    reviewer_weights = {
        ev.reviewer_persona: max(0.5, min(2.0, _safe_float(getattr(ev, "reviewer_weight", 1.0), 1.0)))
        for ev in evaluations
    }

    score_matrix: Dict[str, Dict[str, float]] = {code: {} for code in strategy_codes}

    for ev in evaluations:
        for score in ev.strategy_scores:
            score_matrix[score.strategy_code][ev.reviewer_persona] = float(score.overall_score)

    adjusted_scores: Dict[str, float] = {}
    raw_means: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
    robustness: Dict[str, float] = {}
    major_disagreements: List[Dict[str, Any]] = []
    reviewer_alignment_summary: List[Dict[str, Any]] = []
    strategy_diagnostics: List[Dict[str, Any]] = []

    for code in strategy_codes:
        reviewer_scores = score_matrix[code]
        if not reviewer_scores:
            adjusted_scores[code] = 0.0
            raw_means[code] = 0.0
            std_scores[code] = 0.0
            penalties[code] = 0.0
            robustness[code] = 0.0
            continue

        reviewers = list(reviewer_scores.keys())
        vals = np.asarray([reviewer_scores[r] for r in reviewers], dtype=float)
        weights = np.asarray([reviewer_weights.get(r, 1.0) for r in reviewers], dtype=float)

        raw_mean = float(np.mean(vals))
        weighted_mean = float(np.average(vals, weights=weights))
        std = float(np.std(vals))
        spread = float(np.max(vals) - np.min(vals))

        disagreement_penalty = min(16.0, std * 0.9 + spread * 0.12)
        adjusted = max(0.0, weighted_mean - disagreement_penalty)
        robust = max(0.0, min(100.0, weighted_mean - std * 1.8))

        adjusted_scores[code] = adjusted
        raw_means[code] = raw_mean
        std_scores[code] = std
        penalties[code] = disagreement_penalty
        robustness[code] = robust

        if std >= 5.0 or spread >= 12.0:
            strongest_advocate = max(reviewer_scores.items(), key=lambda kv: kv[1])[0]
            strongest_critic = min(reviewer_scores.items(), key=lambda kv: kv[1])[0]
            major_disagreements.append(
                {
                    "strategy": code,
                    "range": f"{min(vals):.0f}-{max(vals):.0f}",
                    "strongest_advocate": strongest_advocate,
                    "strongest_critic": strongest_critic,
                    "spread": spread,
                    "std_dev": std,
                }
            )

        strategy_diagnostics.append(
            {
                "strategy": code,
                "confidence_adjusted_score": round(adjusted, 2),
                "raw_mean_score": round(raw_mean, 2),
                "disagreement_penalty": round(disagreement_penalty, 2),
                "std_dev": round(std, 2),
                "robustness_score": round(robust, 2),
            }
        )

    ranked = sorted(adjusted_scores.items(), key=lambda kv: kv[1], reverse=True)
    winner_code, winner_score = ranked[0]
    runner_up_code, runner_up_score = ranked[1] if len(ranked) > 1 else ranked[0]

    all_stds = np.asarray(list(std_scores.values()), dtype=float)
    mean_std = float(np.mean(all_stds)) if all_stds.size else 0.0

    consensus_strength = max(0.0, min(100.0, 100.0 - mean_std * 8.0))
    controversy_score = max(0.0, min(100.0, np.mean(list(penalties.values())) * 4.0 if penalties else 0.0))
    robustness_score = max(0.0, min(100.0, robustness.get(winner_code, winner_score)))
    fragile_winner = (winner_score - runner_up_score) < 4.0 or std_scores.get(winner_code, 0.0) > 8.0

    for ev in evaluations:
        per_scores = {s.strategy_code: float(s.overall_score) for s in ev.strategy_scores}
        best = max(per_scores.items(), key=lambda kv: kv[1])[0]
        worst = min(per_scores.items(), key=lambda kv: kv[1])[0]
        reviewer_alignment_summary.append(
            {
                "reviewer_persona": ev.reviewer_persona,
                "preferred_strategy": best,
                "least_preferred_strategy": worst,
                "reviewer_weight": reviewer_weights.get(ev.reviewer_persona, 1.0),
            }
        )

    major_disagreements.sort(key=lambda x: x["spread"], reverse=True)
    strategy_diagnostics.sort(key=lambda x: x["confidence_adjusted_score"], reverse=True)

    return ConsensusResult(
        winner_strategy=winner_code,
        winner_score=winner_score,
        runner_up_strategy=runner_up_code,
        runner_up_score=runner_up_score,
        consensus_strength=consensus_strength,
        robustness_score=robustness_score,
        controversy_score=controversy_score,
        fragile_winner=fragile_winner,
        major_disagreements=major_disagreements,
        reviewer_alignment_summary=reviewer_alignment_summary,
        strategy_diagnostics=strategy_diagnostics,
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

    disagreement_text = ""
    if consensus_result.major_disagreements:
        disagreement_text = "Major disagreements:\n"
        for d in consensus_result.major_disagreements[:4]:
            disagreement_text += (
                f"- {d['strategy']}: {d['strongest_advocate']} vs {d['strongest_critic']} "
                f"(spread {d['spread']:.1f}, std {d['std_dev']:.1f})\n"
            )

    return f"""
You are the final judge in a career pivot decision engine.

Synthesize:
- competing strategy rankings
- reviewer disagreement
- robustness and controversy
- labor-market realism

Current role: {current_role}
Target role: {target_role}

Consensus rankings:
{rankings_text}

Consensus strength: {consensus_result.consensus_strength:.1f}/100
Robustness score: {consensus_result.robustness_score:.1f}/100
Controversy score: {consensus_result.controversy_score:.1f}/100
Fragile winner: {consensus_result.fragile_winner}

{disagreement_text}

Gap summary:
{gap_summary}

Return ONLY valid JSON:
{{
  "verdict": "Highly Feasible | Feasible with Conditions | Challenging",
  "recommended_strategy": "{consensus_result.winner_strategy}",
  "executive_summary": "3-5 sentence synthesis",
  "key_success_factors": ["string", "string", "string"],
  "critical_risks": ["string", "string"],
  "first_30_day_actions": ["string", "string", "string"],
  "interview_narrative": "How to pitch this pivot convincingly",
  "success_timeline": "e.g. 6-9 months",
  "confidence_level": "High | Medium | Low"
}}
""".strip()


def _offline_judge_memo(
    *,
    current_role: str,
    target_role: str,
    consensus_result: ConsensusResult,
) -> Dict[str, Any]:
    verdict = "Feasible with Conditions"
    if consensus_result.winner_score >= 82 and consensus_result.robustness_score >= 75 and not consensus_result.fragile_winner:
        verdict = "Highly Feasible"
    elif consensus_result.winner_score < 65 or consensus_result.controversy_score > 40:
        verdict = "Challenging"

    confidence = "Medium"
    if consensus_result.robustness_score >= 75 and not consensus_result.fragile_winner:
        confidence = "High"
    elif consensus_result.fragile_winner or consensus_result.controversy_score >= 35:
        confidence = "Low"

    return {
        "memo": JudgeMemo(
            verdict=verdict,
            recommended_strategy=consensus_result.winner_strategy,
            executive_summary=(
                f"The {consensus_result.winner_strategy} strategy currently leads for moving from "
                f"{current_role} to {target_role}. This recommendation is based on confidence-adjusted reviewer support, "
                f"not just raw averages, so disagreement and fragility reduce the final rank rather than being ignored."
            ),
            key_success_factors=[
                "Prioritize the highest-leverage missing skills",
                "Create visible evidence that makes the pivot legible to outsiders",
                "Use a strategy-specific narrative instead of generic career-change language",
            ],
            critical_risks=[
                "Weak external credibility if execution quality is low",
                "The best-looking plan may fail if its trade-offs are ignored in practice",
            ],
            first_30_day_actions=[
                "Choose one high-signal proof artifact or milestone",
                "Refine the transition narrative around the strongest transferable anchors",
                "Stress-test the first month plan against real market feedback",
            ],
            interview_narrative=(
                "Present the pivot as deliberate, evidence-backed, and role-specific: explain what transfers, "
                "show what has been built to close the gaps, and make the strategy sound realistic rather than aspirational."
            ),
            success_timeline="6-9 months",
            confidence_level=confidence,
        ),
        "source": "Offline (deterministic)",
        "trace": {},
    }


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
        resp = client.responses.create(model=model, input=prompt)
        raw_text = (resp.output_text or "").strip()
        memo_obj = _extract_json_object(raw_text)

        return {
            "memo": JudgeMemo(
                verdict=str(memo_obj.get("verdict", "Feasible with Conditions")),
                recommended_strategy=str(memo_obj.get("recommended_strategy", consensus_result.winner_strategy)),
                executive_summary=str(memo_obj.get("executive_summary", "")),
                key_success_factors=list(memo_obj.get("key_success_factors", []))[:5],
                critical_risks=list(memo_obj.get("critical_risks", []))[:5],
                first_30_day_actions=list(memo_obj.get("first_30_day_actions", []))[:5],
                interview_narrative=str(memo_obj.get("interview_narrative", "")),
                success_timeline=str(memo_obj.get("success_timeline", "6-9 months")),
                confidence_level=str(memo_obj.get("confidence_level", "Medium")),
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


def rerank_after_skill_investment(
    *,
    evaluations: List[ReviewerEvaluation],
    invested_skills: List[str],
    uplift_ratio: float = 0.5,
) -> ConsensusResult:
    if not evaluations:
        raise ValueError("No evaluations provided")

    invested_skills = [str(x).strip() for x in (invested_skills or []) if str(x).strip()]
    uplift_ratio = max(0.0, min(1.0, float(uplift_ratio)))
    skill_count = len(invested_skills)

    base_boost = {
        "DIRECT": 2.0,
        "STEPPING": 4.5,
        "SKILL_FIRST": 9.0,
        "PORTFOLIO": 6.5,
        "HYBRID": 7.5,
    }

    boosted_evaluations: List[ReviewerEvaluation] = []

    for ev in evaluations:
        boosted_scores = []
        for score in ev.strategy_scores:
            code = score.strategy_code
            leverage = min(1.0, 0.35 + 0.18 * skill_count)
            boost = base_boost.get(code, 4.0) * leverage * uplift_ratio

            if ev.reviewer_persona == "PortfolioEval" and code == "PORTFOLIO":
                boost += 1.0 * uplift_ratio
            if ev.reviewer_persona == "RiskAnalyst" and code in {"SKILL_FIRST", "STEPPING", "HYBRID"}:
                boost += 0.8 * uplift_ratio
            if ev.reviewer_persona == "HiringManager" and code in {"HYBRID", "PORTFOLIO", "SKILL_FIRST"}:
                boost += 0.6 * uplift_ratio

            boosted_scores.append(
                score.model_copy(
                    update={
                        "overall_score": min(100.0, float(score.overall_score) + boost),
                        "justification": (
                            f"{score.justification} Counterfactual rerank applied after investment in "
                            f"{len(invested_skills)} selected skills."
                        )[:500],
                    }
                )
            )

        boosted_evaluations.append(
            ev.model_copy(update={"strategy_scores": boosted_scores})
        )

    return compute_consensus(boosted_evaluations)