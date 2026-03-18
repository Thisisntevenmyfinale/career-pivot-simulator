"""
Adversarial Review Board: Generate competing strategies and reviewer evaluations.

This module:
1. Generates 5 different strategy archetypes for a pivot
2. Calls LLM to evaluate each via 5 reviewer personas
3. Returns structured scores + raw reasoning
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from dataclasses import asdict

import pandas as pd

from src.review_schemas import (
    Strategy,
    StrategyArchetype,
    StrategyPhase,
    ReviewerScore,
    ReviewerEvaluation,
)


# ============================================================
# Helpers
# ============================================================

def _get_api_key_optional() -> str:
    """Return OpenAI API key from env or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _sanitize_text(s: str, max_len: int = 400) -> str:
    """Normalize whitespace and truncate."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_len]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM output, even if wrapped in prose."""
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
# Strategy Archetypes (Deterministic)
# ============================================================

STRATEGY_ARCHETYPES = {
    "DIRECT": StrategyArchetype(
        name="Direct Pivot",
        code="DIRECT",
        description="Immediate jump to target role. Highest risk but fastest. Assumes strong skill overlap or willingness to learn on the job.",
        estimated_days=45,
        risk_level="high",
    ),
    "STEPPING": StrategyArchetype(
        name="Stepping-Stone Pivot",
        code="STEPPING",
        description="Use intermediate roles as bridges. Lower risk, longer timeline. Builds credibility incrementally.",
        estimated_days=150,
        risk_level="medium",
    ),
    "SKILL_FIRST": StrategyArchetype(
        name="Skill-First Pivot",
        code="SKILL_FIRST",
        description="Master critical missing skills first, then pivot. Thorough but time-intensive.",
        estimated_days=120,
        risk_level="low",
    ),
    "PORTFOLIO": StrategyArchetype(
        name="Portfolio-First Pivot",
        code="PORTFOLIO",
        description="Build 2-3 portfolio artifacts that demonstrate target-role capability, then leverage those for credibility.",
        estimated_days=100,
        risk_level="medium",
    ),
    "HYBRID": StrategyArchetype(
        name="Hybrid / Balanced",
        code="HYBRID",
        description="Combine stepping-stone + parallel skill building + portfolio work. Most sustainable but requires discipline.",
        estimated_days=110,
        risk_level="medium",
    ),
}


# ============================================================
# Prompts for Strategy Generation
# ============================================================

def _build_strategy_gen_prompt(
    *,
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
    gap_summary: str,
) -> str:
    """Build prompt for LLM to generate a specific strategy archetype."""

    return f"""
You are a senior career strategist designing a {archetype.name} for a career pivot.

Current role: {current_role}
Target role: {target_role}

Archetype: {archetype.name}
Description: {archetype.description}
Expected timeline: {archetype.estimated_days} days
Risk level: {archetype.risk_level}

Top missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Transferable anchors:
{json.dumps(transfer_skills, ensure_ascii=False)}

Gap summary:
{gap_summary}

Return ONLY valid JSON with this exact shape:
{{
  "archetype": {{
    "name": "{archetype.name}",
    "code": "{archetype.code}",
    "description": "{archetype.description}",
    "estimated_days": {archetype.estimated_days},
    "risk_level": "{archetype.risk_level}"
  }},
  "summary": "2-3 sentence pitch of this strategy",
  "phases": [
    {{
      "phase": "0-30 days",
      "objective": "...",
      "deliverables": ["...", "..."],
      "key_actions": ["...", "..."]
    }},
    {{
      "phase": "30-60 days",
      ...
    }},
    {{
      "phase": "60-90+ days",
      ...
    }}
  ],
  "key_missing_skills": ["skill1", "skill2", ...],
  "transferable_anchors": ["anchor1", "anchor2", ...],
  "success_criteria": ["criterion1", "criterion2"],
  "potential_risks": ["risk1", "risk2"],
  "resources_needed": ["resource1", "resource2", ...]
}}

Be specific, grounded in the data, and realistic.
""".strip()


def _build_reviewer_prompt(
    *,
    reviewer_persona: str,
    strategy_json: Dict[str, Any],
    current_role: str,
    target_role: str,
) -> str:
    """Build prompt for a specific reviewer persona to evaluate a strategy."""

    persona_context = {
        "HiringManager": "You are a hiring manager at a company hiring for the target role. You care about: fit for the role, ability to hit the ground running, credibility in the space.",
        "Recruiter": "You are a tech recruiter with 10 years experience. You care about: market demand, timing, how easy it is to sell this candidate, realistic salary progression.",
        "PortfolioEval": "You are a portfolio reviewer evaluating candidates for a role. You care about: artifacts that prove capability, project quality, relevance to target domain.",
        "RiskAnalyst": "You are a risk analyst evaluating career change feasibility. You care about: what could go wrong, financial/opportunity costs, fallback plans, timeline realism.",
        "CareerCoach": "You are a career coach helping someone plan a transition. You care about: personal growth, narrative coherence, sustainable pace, long-term fulfillment.",
    }

    persona_desc = persona_context.get(reviewer_persona, "")

    return f"""
{persona_desc}

You are evaluating the following career pivot strategy:

Current role: {current_role}
Target role: {target_role}

Strategy details:
{json.dumps(strategy_json, ensure_ascii=False, indent=2)}

Score this strategy across five dimensions (each 0-10):
1. alignment_with_role: How well does this strategy prepare for the target role?
2. market_feasibility: Is this realistic in the job market?
3. time_efficiency: Is the timeline reasonable? (Lower days = higher score; 365+ days = lower score)
4. risk_assessment: How risky is this? (Lower risk = higher score)
5. narrative_strength: Can the candidate tell a convincing story to hiring managers?

Return ONLY valid JSON with this exact shape:
{{
  "reviewer_persona": "{reviewer_persona}",
  "strategy_code": "{strategy_json.get('archetype', {}).get('code', 'UNKNOWN')}",
  "alignment_with_role": <0-10>,
  "market_feasibility": <0-10>,
  "time_efficiency": <0-10>,
  "risk_assessment": <0-10>,
  "narrative_strength": <0-10>,
  "justification": "2-3 sentences explaining your overall assessment",
  "concerns": ["concern1", "concern2"],
  "overall_score": <0-100>
}}

Be critical but fair. Explain tradeoffs.
""".strip()


# ============================================================
# Main Orchestration
# ============================================================

def generate_all_strategies(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate all 5 strategy archetypes for a pivot.
    
    Returns:
        {
            "strategies": [Strategy, ...],
            "trace": {...},
            "source": "online" | "offline"
        }
    """

    # Extract high-level gap summary
    missing = gap_df[gap_df["gap"] > 0].copy()
    missing_skills = missing.sort_values("gap", ascending=False)["skill"].head(6).tolist()
    
    transfer = gap_df.copy()
    transfer["overlap"] = transfer[["current_importance", "target_importance"]].min(axis=1)
    transfer_skills = transfer.sort_values("overlap", ascending=False)["skill"].head(5).tolist()

    gap_summary = f"{len(missing)} significant skill gaps identified. Strongest overlaps in: {', '.join(transfer_skills[:3])}"

    if not prefer_online:
        # Offline mode: return deterministic fallback
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

    strategies: List[Strategy] = []
    client = OpenAI(api_key=api_key)

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

            resp = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            raw_text = resp.content[0].text if resp.content else ""
            trace["raw_outputs"][code] = raw_text
            
            strategy_obj = _extract_json_object(raw_text)
            
            # Validate + construct Strategy object
            strategy = Strategy(
                archetype=StrategyArchetype(**strategy_obj.get("archetype", {})),
                current_role=current_role,
                target_role=target_role,
                summary=_sanitize_text(strategy_obj.get("summary", ""), 500),
                phases=[
                    StrategyPhase(**p) for p in strategy_obj.get("phases", [])
                ],
                key_missing_skills=strategy_obj.get("key_missing_skills", missing_skills)[:6],
                transferable_anchors=strategy_obj.get("transferable_anchors", transfer_skills)[:5],
                success_criteria=strategy_obj.get("success_criteria", []),
                potential_risks=strategy_obj.get("potential_risks", []),
                resources_needed=strategy_obj.get("resources_needed", []),
            )
            
            strategies.append(strategy)
            trace["strategies_generated"] += 1

        except Exception as e:
            trace["errors"].append(f"{code}: {repr(e)}")

    return {
        "strategies": strategies,
        "trace": trace,
        "source": "OpenAI" if strategies else "Offline (failed)",
    }


def evaluate_strategies_by_reviewers(
    *,
    strategies: List[Strategy],
    current_role: str,
    target_role: str,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Have 5 reviewer personas evaluate all strategies.
    
    Returns:
        {
            "evaluations": [ReviewerEvaluation, ...],
            "trace": {...},
            "source": "online" | "offline"
        }
    """

    reviewer_personas = [
        "HiringManager",
        "Recruiter",
        "PortfolioEval",
        "RiskAnalyst",
        "CareerCoach",
    ]

    if not prefer_online:
        return _offline_evaluations(
            strategies=strategies,
            reviewer_personas=reviewer_personas,
        )

    api_key = _get_api_key_optional()
    if not api_key:
        return _offline_evaluations(
            strategies=strategies,
            reviewer_personas=reviewer_personas,
        )

    try:
        from openai import OpenAI
    except Exception:
        return _offline_evaluations(
            strategies=strategies,
            reviewer_personas=reviewer_personas,
        )

    trace = {
        "mode": "online",
        "model": model,
        "evaluations_completed": 0,
        "errors": [],
        "raw_outputs": {},
    }

    evaluations: List[ReviewerEvaluation] = []
    client = OpenAI(api_key=api_key)

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

                resp = client.messages.create(
                    model=model,
                    max_tokens=800,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                raw_text = resp.content[0].text if resp.content else ""
                key = f"{persona}_{strategy.archetype.code}"
                trace["raw_outputs"][key] = raw_text

                score_obj = _extract_json_object(raw_text)
                score = ReviewerScore(
                    reviewer_persona=persona,
                    strategy_code=strategy.archetype.code,
                    overall_score=score_obj.get("overall_score", 50.0),
                    alignment_with_role=score_obj.get("alignment_with_role", 5.0),
                    market_feasibility=score_obj.get("market_feasibility", 5.0),
                    time_efficiency=score_obj.get("time_efficiency", 5.0),
                    risk_assessment=score_obj.get("risk_assessment", 5.0),
                    narrative_strength=score_obj.get("narrative_strength", 5.0),
                    justification=_sanitize_text(score_obj.get("justification", ""), 400),
                    concerns=score_obj.get("concerns", [])[:3],
                )
                scores.append(score)

            eval_obj = ReviewerEvaluation(
                reviewer_persona=persona,
                strategy_scores=scores,
                overall_recommendation="Strategy selection pending full consensus analysis.",
                strongest_strategy=max(scores, key=lambda s: s.overall_score).strategy_code if scores else "HYBRID",
                weakest_strategy=min(scores, key=lambda s: s.overall_score).strategy_code if scores else "DIRECT",
            )
            evaluations.append(eval_obj)
            trace["evaluations_completed"] += 1

        except Exception as e:
            trace["errors"].append(f"{persona}: {repr(e)}")

    return {
        "evaluations": evaluations,
        "trace": trace,
        "source": "OpenAI" if evaluations else "Offline (failed)",
    }


# ============================================================
# Offline Fallbacks
# ============================================================

def _offline_strategies(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    """Return deterministic offline strategies."""

    strategies = []
    for code, archetype in STRATEGY_ARCHETYPES.items():
        strategy = Strategy(
            archetype=archetype,
            current_role=current_role,
            target_role=target_role,
            summary=f"A {archetype.name.lower()} strategy for transitioning from {current_role} to {target_role}.",
            phases=[
                StrategyPhase(
                    phase="0-30 days",
                    objective="Foundation phase",
                    deliverables=["1 foundational project"],
                    key_actions=["Assess current skill level", "Identify gaps", "Begin skill building"],
                ),
                StrategyPhase(
                    phase="30-90 days",
                    objective="Intermediate phase",
                    deliverables=["Portfolio artifact"],
                    key_actions=["Build 1 portfolio project", "Practice interviews"],
                ),
                StrategyPhase(
                    phase="90+ days",
                    objective="Target achievement",
                    deliverables=["Completed pivot"],
                    key_actions=["Apply to roles", "Interview", "Negotiate offer"],
                ),
            ],
            key_missing_skills=missing_skills[:6],
            transferable_anchors=transfer_skills[:5],
            success_criteria=["Secure target role", "30-day ramp time", "Salary match"],
            potential_risks=["Market saturation", "Credential gaps"],
            resources_needed=["Courses", "Mentorship", "Portfolio projects"],
        )
        strategies.append(strategy)

    return {
        "strategies": strategies,
        "trace": {"mode": "offline", "note": "API unavailable"},
        "source": "Offline (deterministic)",
    }


def _offline_evaluations(
    *,
    strategies: List[Strategy],
    reviewer_personas: List[str],
) -> Dict[str, Any]:
    """Return deterministic offline evaluations."""

    evaluations = []
    for persona in reviewer_personas:
        scores = []
        for i, strategy in enumerate(strategies):
            # Simple deterministic scoring
            base_score = 50.0 + (i * 8)
            score = ReviewerScore(
                reviewer_persona=persona,
                strategy_code=strategy.archetype.code,
                overall_score=min(base_score, 95.0),
                alignment_with_role=min(8.0 + (i * 0.3), 10.0),
                market_feasibility=7.0 + (i * 0.2),
                time_efficiency=10.0 - (strategy.archetype.estimated_days / 365.0) * 5.0,
                risk_assessment=10.0 - {"high": 3.0, "medium": 1.5, "low": 0.0}.get(strategy.archetype.risk_level, 1.5),
                narrative_strength=7.0 + (i * 0.25),
                justification=f"Offline evaluation for {persona}: {strategy.archetype.name}",
                concerns=[],
            )
            scores.append(score)

        eval_obj = ReviewerEvaluation(
            reviewer_persona=persona,
            strategy_scores=scores,
            overall_recommendation="Hybrid strategy shows best balance",
            strongest_strategy="HYBRID",
            weakest_strategy="DIRECT",
        )
        evaluations.append(eval_obj)

    return {
        "evaluations": evaluations,
        "trace": {"mode": "offline"},
        "source": "Offline (deterministic)",
    }