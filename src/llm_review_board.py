from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd

from src.review_schemas import (
    Strategy,
    StrategyArchetype,
    StrategyPhase,
    ReviewerScore,
    ReviewerEvaluation,
)


def _get_api_key_optional() -> str:
    try:
        import streamlit as st
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        return ""


def _sanitize_text(s: str, max_len: int = 400) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_len]


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


STRATEGY_ARCHETYPES = {
    "DIRECT": StrategyArchetype(
        name="Direct Pivot",
        code="DIRECT",
        description="Immediate jump to target role. Highest risk but fastest.",
        estimated_days=45,
        risk_level="high",
    ),
    "STEPPING": StrategyArchetype(
        name="Stepping-Stone Pivot",
        code="STEPPING",
        description="Use intermediate roles as bridges. Lower risk, longer timeline.",
        estimated_days=150,
        risk_level="medium",
    ),
    "SKILL_FIRST": StrategyArchetype(
        name="Skill-First Pivot",
        code="SKILL_FIRST",
        description="Master critical missing skills first, then pivot.",
        estimated_days=120,
        risk_level="low",
    ),
    "PORTFOLIO": StrategyArchetype(
        name="Portfolio-First Pivot",
        code="PORTFOLIO",
        description="Build portfolio artifacts to prove capability before applying.",
        estimated_days=100,
        risk_level="medium",
    ),
    "HYBRID": StrategyArchetype(
        name="Hybrid / Balanced",
        code="HYBRID",
        description="Combine stepping-stone, skill-building, and portfolio work.",
        estimated_days=110,
        risk_level="medium",
    ),
}


def _build_strategy_gen_prompt(
    *,
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
    gap_summary: str,
) -> str:
    return f"""
You are a senior career strategist.

Create one structured career pivot strategy.

Current role: {current_role}
Target role: {target_role}

Archetype:
- name: {archetype.name}
- code: {archetype.code}
- description: {archetype.description}
- estimated_days: {archetype.estimated_days}
- risk_level: {archetype.risk_level}

Missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Transferable anchors:
{json.dumps(transfer_skills, ensure_ascii=False)}

Gap summary:
{gap_summary}

Return ONLY valid JSON:
{{
  "archetype": {{
    "name": "{archetype.name}",
    "code": "{archetype.code}",
    "description": "{archetype.description}",
    "estimated_days": {archetype.estimated_days},
    "risk_level": "{archetype.risk_level}"
  }},
  "summary": "2-3 sentence strategy summary",
  "phases": [
    {{
      "phase": "0-30 days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string"]
    }},
    {{
      "phase": "30-90 days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string"]
    }},
    {{
      "phase": "90+ days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string"]
    }}
  ],
  "key_missing_skills": ["string", "string"],
  "transferable_anchors": ["string", "string"],
  "success_criteria": ["string", "string"],
  "potential_risks": ["string", "string"],
  "resources_needed": ["string", "string"]
}}
""".strip()


def _build_reviewer_prompt(
    *,
    reviewer_persona: str,
    strategy_json: Dict[str, Any],
    current_role: str,
    target_role: str,
) -> str:
    persona_context = {
        "HiringManager": "You are a hiring manager focused on role readiness and credibility.",
        "Recruiter": "You are a recruiter focused on market feasibility and sellability.",
        "PortfolioEval": "You are a portfolio evaluator focused on proof of skill.",
        "RiskAnalyst": "You are a risk analyst focused on downside, timing, and realism.",
        "CareerCoach": "You are a career coach focused on sustainable growth and narrative coherence.",
    }

    return f"""
{persona_context.get(reviewer_persona, "")}

Evaluate this pivot strategy for a transition from {current_role} to {target_role}.

Strategy:
{json.dumps(strategy_json, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "reviewer_persona": "{reviewer_persona}",
  "strategy_code": "{strategy_json.get('archetype', {}).get('code', 'UNKNOWN')}",
  "alignment_with_role": 0,
  "market_feasibility": 0,
  "time_efficiency": 0,
  "risk_assessment": 0,
  "narrative_strength": 0,
  "justification": "2-3 sentence explanation",
  "concerns": ["string", "string"],
  "overall_score": 0
}}
Scores:
- the five dimensions are 0-10
- overall_score is 0-100
""".strip()


def generate_all_strategies(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    missing = gap_df[gap_df["gap"] > 0].copy()
    missing_skills = missing.sort_values("gap", ascending=False)["skill"].head(6).astype(str).tolist()

    transfer = gap_df.copy()
    transfer["overlap"] = transfer[["current_importance", "target_importance"]].min(axis=1)
    transfer_skills = transfer.sort_values("overlap", ascending=False)["skill"].head(5).astype(str).tolist()

    gap_summary = f"{len(missing)} skill gaps identified. Strongest transferable skills: {', '.join(transfer_skills[:3])}"

    if not prefer_online:
        return _offline_strategies(
            current_role=current_role,
            target_role=target_role,
            missing_skills=missing_skills,
            transfer_skills=transfer_skills,
        )

    api_key = _get_api_key_optional()
    if not api_key:
        return _offline_strategies(
            current_role=current_role,
            target_role=target_role,
            missing_skills=missing_skills,
            transfer_skills=transfer_skills,
        )

    try:
        from openai import OpenAI
    except Exception:
        return _offline_strategies(
            current_role=current_role,
            target_role=target_role,
            missing_skills=missing_skills,
            transfer_skills=transfer_skills,
        )

    trace = {
        "mode": "online",
        "model": model,
        "strategies_generated": 0,
        "errors": [],
        "raw_outputs": {},
    }

    client = OpenAI(api_key=api_key)
    strategies: List[Strategy] = []

    for code, archetype in STRATEGY_ARCHETYPES.items():
        try:
            prompt = _build_strategy_gen_prompt(
                current_role=current_role,
                target_role=target_role,
                archetype=archetype,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
                gap_summary=gap_summary,
            )

            resp = client.responses.create(
                model=model,
                input=prompt,
            )

            raw_text = (resp.output_text or "").strip()
            trace["raw_outputs"][code] = raw_text

            obj = _extract_json_object(raw_text)

            strategy = Strategy(
                archetype=StrategyArchetype(**obj.get("archetype", {})),
                current_role=current_role,
                target_role=target_role,
                summary=_sanitize_text(obj.get("summary", ""), 500),
                phases=[StrategyPhase(**p) for p in obj.get("phases", [])],
                key_missing_skills=obj.get("key_missing_skills", missing_skills)[:6],
                transferable_anchors=obj.get("transferable_anchors", transfer_skills)[:5],
                success_criteria=obj.get("success_criteria", [])[:4],
                potential_risks=obj.get("potential_risks", [])[:3],
                resources_needed=obj.get("resources_needed", [])[:5],
            )
            strategies.append(strategy)
            trace["strategies_generated"] += 1

        except Exception as e:
            trace["errors"].append(f"{code}: {repr(e)}")

    if not strategies:
        return _offline_strategies(
            current_role=current_role,
            target_role=target_role,
            missing_skills=missing_skills,
            transfer_skills=transfer_skills,
        )

    return {
        "strategies": strategies,
        "trace": trace,
        "source": "OpenAI",
    }


def evaluate_strategies_by_reviewers(
    *,
    strategies: List[Strategy],
    current_role: str,
    target_role: str,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    reviewer_personas = [
        "HiringManager",
        "Recruiter",
        "PortfolioEval",
        "RiskAnalyst",
        "CareerCoach",
    ]

    if not prefer_online:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    api_key = _get_api_key_optional()
    if not api_key:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    try:
        from openai import OpenAI
    except Exception:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    trace = {
        "mode": "online",
        "model": model,
        "evaluations_completed": 0,
        "errors": [],
        "raw_outputs": {},
    }

    client = OpenAI(api_key=api_key)
    evaluations: List[ReviewerEvaluation] = []

    for persona in reviewer_personas:
        try:
            scores: List[ReviewerScore] = []

            for strategy in strategies:
                prompt = _build_reviewer_prompt(
                    reviewer_persona=persona,
                    strategy_json=asdict(strategy),
                    current_role=current_role,
                    target_role=target_role,
                )

                resp = client.responses.create(
                    model=model,
                    input=prompt,
                )

                raw_text = (resp.output_text or "").strip()
                trace["raw_outputs"][f"{persona}_{strategy.archetype.code}"] = raw_text

                obj = _extract_json_object(raw_text)
                score = ReviewerScore(
                    reviewer_persona=persona,
                    strategy_code=strategy.archetype.code,
                    overall_score=float(obj.get("overall_score", 50.0)),
                    alignment_with_role=float(obj.get("alignment_with_role", 5.0)),
                    market_feasibility=float(obj.get("market_feasibility", 5.0)),
                    time_efficiency=float(obj.get("time_efficiency", 5.0)),
                    risk_assessment=float(obj.get("risk_assessment", 5.0)),
                    narrative_strength=float(obj.get("narrative_strength", 5.0)),
                    justification=_sanitize_text(obj.get("justification", ""), 400),
                    concerns=obj.get("concerns", [])[:3],
                )
                scores.append(score)

            evaluations.append(
                ReviewerEvaluation(
                    reviewer_persona=persona,
                    strategy_scores=scores,
                    overall_recommendation="Strategy selection pending consensus aggregation.",
                    strongest_strategy=max(scores, key=lambda s: s.overall_score).strategy_code if scores else "HYBRID",
                    weakest_strategy=min(scores, key=lambda s: s.overall_score).strategy_code if scores else "DIRECT",
                )
            )
            trace["evaluations_completed"] += 1

        except Exception as e:
            trace["errors"].append(f"{persona}: {repr(e)}")

    if not evaluations:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    return {
        "evaluations": evaluations,
        "trace": trace,
        "source": "OpenAI",
    }


def _offline_strategies(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    strategies = []
    for _, archetype in STRATEGY_ARCHETYPES.items():
        strategies.append(
            Strategy(
                archetype=archetype,
                current_role=current_role,
                target_role=target_role,
                summary=f"A {archetype.name.lower()} strategy for transitioning from {current_role} to {target_role}.",
                phases=[
                    StrategyPhase(
                        phase="0-30 days",
                        objective="Foundation phase",
                        deliverables=["Gap review", "Initial action plan"],
                        key_actions=["Assess fit", "Prioritize gaps"],
                    ),
                    StrategyPhase(
                        phase="30-90 days",
                        objective="Capability building",
                        deliverables=["One concrete artifact", "Interview prep"],
                        key_actions=["Build evidence", "Practice narrative"],
                    ),
                    StrategyPhase(
                        phase="90+ days",
                        objective="Execution",
                        deliverables=["Applications", "Interviews"],
                        key_actions=["Apply", "Iterate based on feedback"],
                    ),
                ],
                key_missing_skills=missing_skills[:6],
                transferable_anchors=transfer_skills[:5],
                success_criteria=["Role readiness", "Convincing story"],
                potential_risks=["Weak signal", "Timeline drift"],
                resources_needed=["Time", "Practice", "Feedback"],
            )
        )

    return {
        "strategies": strategies,
        "trace": {"mode": "offline", "errors": ["No API key or API unavailable"]},
        "source": "Offline (deterministic)",
    }


def _offline_evaluations(
    *,
    strategies: List[Strategy],
    reviewer_personas: List[str],
) -> Dict[str, Any]:
    evaluations = []

    for persona in reviewer_personas:
        scores = []
        for i, strategy in enumerate(strategies):
            base_score = min(95.0, 55.0 + i * 6.0)
            scores.append(
                ReviewerScore(
                    reviewer_persona=persona,
                    strategy_code=strategy.archetype.code,
                    overall_score=base_score,
                    alignment_with_role=min(10.0, 6.5 + i * 0.4),
                    market_feasibility=min(10.0, 6.0 + i * 0.3),
                    time_efficiency=max(1.0, 10.0 - (strategy.archetype.estimated_days / 365.0) * 5.0),
                    risk_assessment=10.0 - {"high": 3.0, "medium": 1.5, "low": 0.5}.get(strategy.archetype.risk_level, 1.5),
                    narrative_strength=min(10.0, 6.8 + i * 0.35),
                    justification=f"Offline evaluation for {persona} on {strategy.archetype.code}.",
                    concerns=[],
                )
            )

        evaluations.append(
            ReviewerEvaluation(
                reviewer_persona=persona,
                strategy_scores=scores,
                overall_recommendation="Consensus pending aggregation.",
                strongest_strategy=max(scores, key=lambda s: s.overall_score).strategy_code,
                weakest_strategy=min(scores, key=lambda s: s.overall_score).strategy_code,
            )
        )

    return {
        "evaluations": evaluations,
        "trace": {"mode": "offline", "errors": ["No API key or API unavailable"]},
        "source": "Offline (deterministic)",
    }