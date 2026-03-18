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


def _reviewer_weights(evaluations: List[ReviewerEvaluation]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ev in evaluations:
        out[ev.reviewer_persona] = max(0.5, min(2.0, _safe_float(getattr(ev, "reviewer_weight", 1.0), 1.0)))
    return out


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

    reviewer_weights = _reviewer_weights(evaluations)

    score_matrix: Dict[str, Dict[str, float]] = {code: {} for code in strategy_codes}
    dimension_matrix: Dict[str, List[float]] = {code: [] for code in strategy_codes}

    for ev in evaluations:
        for score in ev.strategy_scores:
            score_matrix[score.strategy_code][ev.reviewer_persona] = float(score.overall_score)
            dimension_matrix[score.strategy_code].append(
                float(score.alignment_with_role + score.market_feasibility + score.time_efficiency + score.risk_assessment + score.narrative_strength) / 5.0
            )

    weighted_means: Dict[str, float] = {}
    raw_means: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}
    penalties: Dict[str, float] = {}
    robustness: Dict[str, float] = {}

    for code in strategy_codes:
        reviewer_scores = score_matrix.get(code, {})
        if not reviewer_scores:
            weighted_means[code] = 0.0
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
        spread = float(np.max(vals) - np.min(vals)) if len(vals) else 0.0

        disagreement_penalty = min(18.0, std * 0.9 + spread * 0.15)
        confidence_adjusted = max(0.0, weighted_mean - disagreement_penalty)

        weighted_means[code] = confidence_adjusted
        raw_means[code] = raw_mean
        std_scores[code] = std
        penalties[code] = disagreement_penalty
        robustness[code] = max(0.0, min(100.0, weighted_mean - std * 2.2))

    ranked = sorted(weighted_means.items(), key=lambda x: x[1], reverse=True)
    winner_code, winner_score = ranked[0] if ranked else ("HYBRID", 0.0)
    runner_up_code, runner_up_score = ranked[1] if len(ranked) > 1 else (winner_code, winner_score)

    # global agreement metrics
    all_std = np.asarray(list(std_scores.values()), dtype=float)
    mean_std = float(np.mean(all_std)) if all_std.size else 0.0

    consensus_strength = float(max(0.0, min(100.0, 100.0 - mean_std * 7.0)))
    controversy_score = float(max(0.0, min(100.0, np.mean(list(penalties.values())) * 4.0 if penalties else 0.0)))
    robustness_score = float(max(0.0, min(100.0, robustness.get(winner_code, winner_score))))

    score_margin = winner_score - runner_up_score
    fragile_winner = bool(score_margin < 4.0 or std_scores.get(winner_code, 0.0) > 10.0)

    major_disagreements: List[Dict[str, Any]] = []
    strategy_diagnostics: List[Dict[str, Any]] = []
    reviewer_alignment_summary: List[Dict[str, Any]] = []

    for code in strategy_codes:
        reviewer_scores = score_matrix[code]
        vals = list(reviewer_scores.values())
        if reviewer_scores:
            min_score = min(vals)
            max_score = max(vals)
            min_reviewer = [k for k, v in reviewer_scores.items() if v == min_score][0]
            max_reviewer = [k for k, v in reviewer_scores.items() if v == max_score][0]
        else:
            min_score = max_score = 0.0
            min_reviewer = max_reviewer = "Unknown"

        if std_scores.get(code, 0.0) > 8.0 or (max_score - min_score) >= 15.0:
            major_disagreements.append(
                {
                    "strategy": code,
                    "range": f"{min_score:.0f}-{max_score:.0f}",
                    "strongest_advocate": max_reviewer,
                    "strongest_critic": min_reviewer,
                    "spread": float(max_score - min_score),
                    "std_dev": float(std_scores.get(code, 0.0)),
                }
            )

        strategy_diagnostics.append(
            {
                "strategy": code,
                "confidence_adjusted_score": round(weighted_means.get(code, 0.0), 2),
                "raw_mean_score": round(raw_means.get(code, 0.0), 2),
                "disagreement_penalty": round(penalties.get(code, 0.0), 2),
                "std_dev": round(std_scores.get(code, 0.0), 2),
                "robustness_score": round(robustness.get(code, 0.0), 2),
            }
        )

    major_disagreements.sort(key=lambda x: x["spread"], reverse=True)
    strategy_diagnostics.sort(key=lambda x: x["confidence_adjusted_score"], reverse=True)

    for ev in evaluations:
        person_scores = {s.strategy_code: float(s.overall_score) for s in ev.strategy_scores}
        if not person_scores:
            continue

        best_code = max(person_scores.items(), key=lambda kv: kv[1])[0]
        worst_code = min(person_scores.items(), key=lambda kv: kv[1])[0]
        reviewer_alignment_summary.append(
            {
                "reviewer_persona": ev.reviewer_persona,
                "preferred_strategy": best_code,
                "least_preferred_strategy": worst_code,
                "reviewer_weight": reviewer_weights.get(ev.reviewer_persona, 1.0),
            }
        )

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
        [f"{i + 1}. {code}: {score:.1f}/100" for i, (code, score) in enumerate(consensus_result.strategy_rankings)]
    )

    disagreements_text = ""
    if consensus_result.major_disagreements:
        disagreements_text += "Major disagreements:\n"
        for d in consensus_result.major_disagreements[:4]:
            disagreements_text += (
                f"- {d['strategy']}: {d['strongest_advocate']} vs {d['strongest_critic']} "
                f"(spread {d['spread']:.1f}, std {d['std_dev']:.1f})\n"
            )

    reviewer_pref_text = ""
    if consensus_result.reviewer_alignment_summary:
        reviewer_pref_text += "Reviewer alignment summary:\n"
        for row in consensus_result.reviewer_alignment_summary[:5]:
            reviewer_pref_text += (
                f"- {row['reviewer_persona']}: prefers {row['preferred_strategy']}, "
                f"least prefers {row['least_preferred_strategy']}\n"
            )

    return f"""
You are the final judge in a career-pivot decision engine.

You must synthesize:
- ranked multi-strategy outputs
- reviewer disagreements
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

{disagreements_text}

{reviewer_pref_text}

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
  "interview_narrative": "How the candidate should pitch this pivot",
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
    if consensus_result.winner_score >= 82 and consensus_result.consensus_strength >= 75:
        verdict = "Highly Feasible"
    elif consensus_result.winner_score < 65:
        verdict = "Challenging"

    confidence = "Medium"
    if consensus_result.robustness_score >= 80 and not consensus_result.fragile_winner:
        confidence = "High"
    elif consensus_result.fragile_winner or consensus_result.controversy_score >= 45:
        confidence = "Low"

    return {
        "memo": JudgeMemo(
            verdict=verdict,
            recommended_strategy=consensus_result.winner_strategy,
            executive_summary=(
                f"The {consensus_result.winner_strategy} strategy currently leads for moving from "
                f"{current_role} to {target_role}. Its recommendation is based on confidence-adjusted reviewer "
                f"support rather than raw mean score alone, which helps penalize fragile or controversial winners."
            ),
            key_success_factors=[
                "Prioritize the highest-signal missing skills",
                "Create external proof of capability early",
                "Align the interview narrative with the chosen strategy logic",
            ],
            critical_risks=[
                "Weak market signal if execution quality is low",
                "Recommendation could change if critical gaps remain unaddressed",
            ],
            first_30_day_actions=[
                "Choose the highest-leverage proof artifact",
                "Tighten the pivot narrative around transferable anchors",
                "Target the top missing skills that most affect credibility",
            ],
            interview_narrative=(
                "Frame the pivot as a deliberate, evidence-backed transition: explain the transferable strengths, "
                "show what has been built to close the role-relevant gaps, and position the move as realistic rather than aspirational."
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
    invested_skills = [str(s).strip() for s in (invested_skills or []) if str(s).strip()]
    uplift_ratio = max(0.0, min(1.0, float(uplift_ratio)))

    if not evaluations:
        raise ValueError("No evaluations provided")

    skill_count = len(invested_skills)
    leverage = min(1.0, 0.35 + 0.18 * skill_count)

    # strategy-specific sensitivity to skill investment
    base_boost = {
        "DIRECT": 2.0,
        "STEPPING": 4.0,
        "SKILL_FIRST": 9.0,
        "PORTFOLIO": 6.5,
        "HYBRID": 7.5,
    }

    boosted_evaluations: List[ReviewerEvaluation] = []

    for ev in evaluations:
        boosted_scores = []

        for score in ev.strategy_scores:
            code = score.strategy_code
            boost = base_boost.get(code, 4.0) * uplift_ratio * leverage

            # persona-specific interpretation of skill investment
            persona_bonus = 0.0
            if ev.reviewer_persona == "HiringManager" and code in {"SKILL_FIRST", "HYBRID", "PORTFOLIO"}:
                persona_bonus = 0.8 * uplift_ratio
            elif ev.reviewer_persona == "Recruiter" and code in {"STEPPING", "HYBRID", "PORTFOLIO"}:
                persona_bonus = 0.6 * uplift_ratio
            elif ev.reviewer_persona == "PortfolioEval" and code == "PORTFOLIO":
                persona_bonus = 1.2 * uplift_ratio
            elif ev.reviewer_persona == "RiskAnalyst" and code in {"SKILL_FIRST", "HYBRID", "STEPPING"}:
                persona_bonus = 0.9 * uplift_ratio
            elif ev.reviewer_persona == "CareerCoach" and code in {"SKILL_FIRST", "HYBRID"}:
                persona_bonus = 0.7 * uplift_ratio

            total_boost = boost + persona_bonus
            new_overall = min(100.0, float(score.overall_score) + total_boost)

            boosted_scores.append(
                score.model_copy(
                    update={
                        "overall_score": new_overall,
                        "justification": (
                            f"{score.justification} Counterfactual rerank applied after investment in "
                            f"{len(invested_skills)} skills."
                        )[:500],
                    }
                )
            )

        boosted_evaluations.append(
            ev.model_copy(update={"strategy_scores": boosted_scores})
        )

    return compute_consensus(boosted_evaluations)