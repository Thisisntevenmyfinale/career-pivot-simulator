from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.review_schemas import (
    ReviewerEvaluation,
    ReviewerScore,
    Strategy,
    StrategyArchetype,
    StrategyPhase,
)


def _get_api_key_optional() -> str:
    try:
        import streamlit as st
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        return ""


def _sanitize_text(s: Any, max_len: int = 400) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:max_len]


def _sanitize_list(items: Any, *, max_n: int, max_len: int = 120) -> List[str]:
    if not isinstance(items, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for item in items:
        val = _sanitize_text(item, max_len=max_len)
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
        if len(cleaned) >= max_n:
            break
    return cleaned


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


STRATEGY_ARCHETYPES: Dict[str, StrategyArchetype] = {
    "DIRECT": StrategyArchetype(
        name="Direct Pivot",
        code="DIRECT",
        description="Immediate jump to the target role. Optimizes for speed, but requires high credibility.",
        estimated_days=45,
        risk_level="high",
    ),
    "STEPPING": StrategyArchetype(
        name="Stepping-Stone Pivot",
        code="STEPPING",
        description="Uses one or more adjacent roles as bridges. Optimizes for market realism and lower downside risk.",
        estimated_days=150,
        risk_level="medium",
    ),
    "SKILL_FIRST": StrategyArchetype(
        name="Skill-First Pivot",
        code="SKILL_FIRST",
        description="Prioritizes closing the highest-signal capability gaps before pushing hard on the pivot.",
        estimated_days=120,
        risk_level="low",
    ),
    "PORTFOLIO": StrategyArchetype(
        name="Portfolio-First Pivot",
        code="PORTFOLIO",
        description="Builds visible proof of capability first. Optimizes for evidence and credibility in applications.",
        estimated_days=100,
        risk_level="medium",
    ),
    "HYBRID": StrategyArchetype(
        name="Hybrid / Balanced",
        code="HYBRID",
        description="Combines targeted upskilling, evidence-building, and realistic market entry. Optimizes for robustness.",
        estimated_days=110,
        risk_level="medium",
    ),
}

REVIEWER_PERSONAS: List[str] = [
    "HiringManager",
    "Recruiter",
    "PortfolioEval",
    "RiskAnalyst",
    "CareerCoach",
]

REVIEWER_WEIGHTS: Dict[str, float] = {
    "HiringManager": 1.25,
    "Recruiter": 1.10,
    "PortfolioEval": 1.00,
    "RiskAnalyst": 1.15,
    "CareerCoach": 0.95,
}


def _strategy_style_hints(code: str) -> Dict[str, Any]:
    code = str(code).upper()
    mapping = {
        "DIRECT": {
            "optimization_goal": "maximize speed to target role",
            "must_emphasize": [
                "credible transferable strengths",
                "fast interview readiness",
                "minimum viable proof of fit",
            ],
            "must_avoid": [
                "long preparation timelines",
                "too many intermediate detours",
                "overbuilding before applying",
            ],
            "signal_profile": {
                "speed_bias": 9.0,
                "risk_bias": 2.5,
                "evidence_burden": 7.5,
                "market_signal_strength": 7.5,
                "skill_gap_focus": 4.5,
            },
        },
        "STEPPING": {
            "optimization_goal": "minimize transition risk by using adjacent roles",
            "must_emphasize": [
                "credible bridge roles",
                "progressive market entry",
                "reduced downside risk",
            ],
            "must_avoid": [
                "implausible direct jump",
                "missing bridge narrative",
                "weak sequencing",
            ],
            "signal_profile": {
                "speed_bias": 4.0,
                "risk_bias": 8.5,
                "evidence_burden": 5.5,
                "market_signal_strength": 8.0,
                "skill_gap_focus": 6.0,
            },
        },
        "SKILL_FIRST": {
            "optimization_goal": "close critical missing skills before applying aggressively",
            "must_emphasize": [
                "highest-signal skill gaps",
                "sequenced learning",
                "measurable capability lift",
            ],
            "must_avoid": [
                "vague learning",
                "course collecting without outputs",
                "premature applications",
            ],
            "signal_profile": {
                "speed_bias": 3.0,
                "risk_bias": 8.0,
                "evidence_burden": 6.5,
                "market_signal_strength": 6.0,
                "skill_gap_focus": 9.0,
            },
        },
        "PORTFOLIO": {
            "optimization_goal": "maximize external proof of capability through tangible evidence",
            "must_emphasize": [
                "visible outputs",
                "portfolio artifacts",
                "proof before claims",
            ],
            "must_avoid": [
                "abstract learning only",
                "generic projects with weak signaling",
                "unclear evidence strategy",
            ],
            "signal_profile": {
                "speed_bias": 5.0,
                "risk_bias": 6.0,
                "evidence_burden": 9.0,
                "market_signal_strength": 8.5,
                "skill_gap_focus": 7.0,
            },
        },
        "HYBRID": {
            "optimization_goal": "balance speed, risk, learning, and proof of execution",
            "must_emphasize": [
                "balanced sequencing",
                "risk-aware progress",
                "both skill and signal building",
            ],
            "must_avoid": [
                "strategy drift",
                "trying everything at once",
                "weak prioritization",
            ],
            "signal_profile": {
                "speed_bias": 6.5,
                "risk_bias": 7.0,
                "evidence_burden": 7.0,
                "market_signal_strength": 7.5,
                "skill_gap_focus": 7.5,
            },
        },
    }
    return mapping.get(code, mapping["HYBRID"])


def _summarize_gap_df(gap_df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    df = gap_df.copy()

    for col in ["gap", "current_importance", "target_importance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    missing = df[df["gap"] > 0].copy()
    missing_skills = (
        missing.sort_values(["gap", "target_importance"], ascending=False)["skill"]
        .astype(str)
        .head(6)
        .tolist()
    )

    transfer = df.copy()
    transfer["overlap"] = transfer[["current_importance", "target_importance"]].min(axis=1)
    transfer_skills = (
        transfer.sort_values(["overlap", "target_importance"], ascending=False)["skill"]
        .astype(str)
        .head(5)
        .tolist()
    )

    avg_gap = float(missing["gap"].mean()) if not missing.empty else 0.0
    high_signal_missing = (
        missing[missing["target_importance"] >= 3.0]
        .sort_values(["gap", "target_importance"], ascending=False)["skill"]
        .astype(str)
        .head(4)
        .tolist()
    )

    gap_summary = (
        f"{len(missing)} positive skill gaps identified. "
        f"Average gap: {avg_gap:.2f}. "
        f"Highest-signal missing skills: {', '.join(high_signal_missing) if high_signal_missing else 'none'}. "
        f"Top transferable anchors: {', '.join(transfer_skills[:3]) if transfer_skills else 'none'}."
    )

    return missing_skills, transfer_skills, gap_summary


def _build_strategy_gen_prompt(
    *,
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
    gap_summary: str,
) -> str:
    hints = _strategy_style_hints(archetype.code)

    return f"""
You are designing one strategy inside a career-pivot decision engine.

Your task is NOT to write generic career advice.
Your task is to produce ONE strategy that is clearly differentiated from the other archetypes.

Current role: {current_role}
Target role: {target_role}

Strategy archetype:
- name: {archetype.name}
- code: {archetype.code}
- description: {archetype.description}
- estimated_days: {archetype.estimated_days}
- risk_level: {archetype.risk_level}

Optimization goal:
{hints["optimization_goal"]}

Must emphasize:
{json.dumps(hints["must_emphasize"], ensure_ascii=False)}

Must avoid:
{json.dumps(hints["must_avoid"], ensure_ascii=False)}

Top missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Top transferable anchors:
{json.dumps(transfer_skills, ensure_ascii=False)}

Gap summary:
{gap_summary}

Write a strategy that feels genuinely different from the other archetypes.
Make trade-offs explicit. Keep it practical and market-aware.

Return ONLY valid JSON:
{{
  "archetype": {{
    "name": "{archetype.name}",
    "code": "{archetype.code}",
    "description": "{archetype.description}",
    "estimated_days": {archetype.estimated_days},
    "risk_level": "{archetype.risk_level}"
  }},
  "summary": "2-4 sentence summary",
  "phases": [
    {{
      "phase": "0-30 days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string", "string"]
    }},
    {{
      "phase": "30-90 days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string", "string"]
    }},
    {{
      "phase": "90+ days",
      "objective": "string",
      "deliverables": ["string", "string"],
      "key_actions": ["string", "string", "string"]
    }}
  ],
  "key_missing_skills": ["string", "string"],
  "transferable_anchors": ["string", "string"],
  "success_criteria": ["string", "string", "string"],
  "potential_risks": ["string", "string"],
  "resources_needed": ["string", "string", "string"],
  "best_for_profile": "Who this strategy suits best",
  "evidence_strategy": "How this strategy creates labor-market proof",
  "key_tradeoff": "Main trade-off in one sentence",
  "confidence_rationale": "Why this strategy is or is not realistic",
  "speed_bias": {hints["signal_profile"]["speed_bias"]},
  "risk_bias": {hints["signal_profile"]["risk_bias"]},
  "evidence_burden": {hints["signal_profile"]["evidence_burden"]},
  "market_signal_strength": {hints["signal_profile"]["market_signal_strength"]},
  "skill_gap_focus": {hints["signal_profile"]["skill_gap_focus"]}
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
        "HiringManager": "You are a hiring manager focused on role readiness, credibility, and whether this candidate would survive a real interview loop.",
        "Recruiter": "You are a recruiter focused on market feasibility, positioning, and whether this pivot can be sold externally in a noisy job market.",
        "PortfolioEval": "You are a portfolio evaluator focused on proof of skill, visibility of work, and whether evidence backs the claims.",
        "RiskAnalyst": "You are a risk analyst focused on downside, weak assumptions, timing risk, and fragility of the plan.",
        "CareerCoach": "You are a career coach focused on sustainable growth, motivation, sequencing, and narrative coherence.",
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
  "justification": "2-4 sentence explanation",
  "concerns": ["string", "string"],
  "best_strength": "Best part of this strategy",
  "biggest_risk": "Biggest practical risk",
  "killer_objection": "The hardest objection against this plan",
  "success_condition": "What must be true for this strategy to work",
  "best_candidate_fit": "Who this strategy fits best",
  "overall_score": 0
}}

Scoring rules:
- the five dimensions are 0-10
- overall_score is 0-100
- be discriminative, not polite
- do not give all strategies similar scores
- let your persona preferences matter
""".strip()


def _coerce_float(x: Any, default: float, *, lo: float, hi: float) -> float:
    try:
        val = float(x)
    except Exception:
        val = default
    return max(lo, min(hi, val))


def _normalize_strategy_object(
    *,
    obj: Dict[str, Any],
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Strategy:
    phases_raw = obj.get("phases", [])
    phases: List[StrategyPhase] = []

    for idx, p in enumerate(phases_raw[:4]):
        phase_name = _sanitize_text(
            p.get("phase", ["0-30 days", "30-90 days", "90+ days"][min(idx, 2)]),
            max_len=40,
        )
        phases.append(
            StrategyPhase(
                phase=phase_name or ["0-30 days", "30-90 days", "90+ days"][min(idx, 2)],
                objective=_sanitize_text(p.get("objective", "Build pivot readiness"), max_len=160),
                deliverables=_sanitize_list(p.get("deliverables", []), max_n=4, max_len=120),
                key_actions=_sanitize_list(p.get("key_actions", []), max_n=5, max_len=120),
            )
        )

    if len(phases) < 3:
        fallback_phases = [
            StrategyPhase(
                phase="0-30 days",
                objective="Clarify the pivot positioning and define the first evidence milestones.",
                deliverables=["Positioning brief", "Priority skill list"],
                key_actions=["Map top gaps", "Define target narrative", "Choose first proof artifact"],
            ),
            StrategyPhase(
                phase="30-90 days",
                objective="Build enough capability and evidence to make the pivot credible.",
                deliverables=["Visible artifact", "Interview story bank"],
                key_actions=["Create proof of work", "Practice stories", "Refine role targeting"],
            ),
            StrategyPhase(
                phase="90+ days",
                objective="Push into the market with a strategy-specific execution plan.",
                deliverables=["Applications", "Feedback loop"],
                key_actions=["Apply selectively", "Use feedback to iterate", "Double down on strongest signal"],
            ),
        ]
        phases = fallback_phases

    return Strategy(
        archetype=archetype,
        current_role=current_role,
        target_role=target_role,
        summary=_sanitize_text(obj.get("summary", ""), max_len=700)
        or f"{archetype.name} strategy for moving from {current_role} to {target_role}.",
        phases=phases[:4],
        key_missing_skills=_sanitize_list(obj.get("key_missing_skills", missing_skills), max_n=6),
        transferable_anchors=_sanitize_list(obj.get("transferable_anchors", transfer_skills), max_n=5),
        success_criteria=_sanitize_list(obj.get("success_criteria", []), max_n=5) or [
            "Improved market credibility",
            "Stronger interview readiness",
            "Clearer evidence of fit",
        ],
        potential_risks=_sanitize_list(obj.get("potential_risks", []), max_n=4) or [
            "Weak labor-market signal",
            "Timeline drift",
        ],
        resources_needed=_sanitize_list(obj.get("resources_needed", []), max_n=6) or [
            "Focused time",
            "Feedback from market",
            "Tangible outputs",
        ],
        best_for_profile=_sanitize_text(obj.get("best_for_profile", ""), max_len=240),
        evidence_strategy=_sanitize_text(obj.get("evidence_strategy", ""), max_len=320),
        key_tradeoff=_sanitize_text(obj.get("key_tradeoff", ""), max_len=220),
        confidence_rationale=_sanitize_text(obj.get("confidence_rationale", ""), max_len=320),
        speed_bias=_coerce_float(obj.get("speed_bias", 5.0), 5.0, lo=0.0, hi=10.0),
        risk_bias=_coerce_float(obj.get("risk_bias", 5.0), 5.0, lo=0.0, hi=10.0),
        evidence_burden=_coerce_float(obj.get("evidence_burden", 5.0), 5.0, lo=0.0, hi=10.0),
        market_signal_strength=_coerce_float(obj.get("market_signal_strength", 5.0), 5.0, lo=0.0, hi=10.0),
        skill_gap_focus=_coerce_float(obj.get("skill_gap_focus", 5.0), 5.0, lo=0.0, hi=10.0),
    )


def _normalize_reviewer_score(
    *,
    obj: Dict[str, Any],
    persona: str,
    strategy_code: str,
) -> ReviewerScore:
    dimensions = {
        "alignment_with_role": _coerce_float(obj.get("alignment_with_role", 5.0), 5.0, lo=0.0, hi=10.0),
        "market_feasibility": _coerce_float(obj.get("market_feasibility", 5.0), 5.0, lo=0.0, hi=10.0),
        "time_efficiency": _coerce_float(obj.get("time_efficiency", 5.0), 5.0, lo=0.0, hi=10.0),
        "risk_assessment": _coerce_float(obj.get("risk_assessment", 5.0), 5.0, lo=0.0, hi=10.0),
        "narrative_strength": _coerce_float(obj.get("narrative_strength", 5.0), 5.0, lo=0.0, hi=10.0),
    }

    overall = _coerce_float(obj.get("overall_score", 0.0), 0.0, lo=0.0, hi=100.0)
    if overall <= 0.0:
        overall = sum(dimensions.values()) / len(dimensions) * 10.0

    return ReviewerScore(
        reviewer_persona=persona,
        strategy_code=strategy_code,
        overall_score=overall,
        alignment_with_role=dimensions["alignment_with_role"],
        market_feasibility=dimensions["market_feasibility"],
        time_efficiency=dimensions["time_efficiency"],
        risk_assessment=dimensions["risk_assessment"],
        narrative_strength=dimensions["narrative_strength"],
        justification=_sanitize_text(obj.get("justification", ""), max_len=500)
        or f"{persona} evaluation for {strategy_code}.",
        concerns=_sanitize_list(obj.get("concerns", []), max_n=4, max_len=160),
        best_strength=_sanitize_text(obj.get("best_strength", ""), max_len=220),
        biggest_risk=_sanitize_text(obj.get("biggest_risk", ""), max_len=220),
        killer_objection=_sanitize_text(obj.get("killer_objection", ""), max_len=220),
        success_condition=_sanitize_text(obj.get("success_condition", ""), max_len=220),
        best_candidate_fit=_sanitize_text(obj.get("best_candidate_fit", ""), max_len=220),
    )


def _strategy_similarity_signature(strategy: Strategy) -> set[str]:
    tokens = set()

    for skill in strategy.key_missing_skills[:6]:
        tokens.add(skill.lower())

    for phase in strategy.phases[:3]:
        for action in phase.key_actions[:4]:
            tokens.add(action.lower())

    for crit in strategy.success_criteria[:3]:
        tokens.add(crit.lower())

    return tokens


def _compute_diversity_warnings(strategies: List[Strategy]) -> List[str]:
    warnings: List[str] = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1 = _strategy_similarity_signature(strategies[i])
            s2 = _strategy_similarity_signature(strategies[j])
            if not s1 or not s2:
                continue
            overlap = len(s1 & s2)
            union = max(1, len(s1 | s2))
            ratio = overlap / union
            if ratio >= 0.45:
                warnings.append(
                    f"{strategies[i].archetype.code} and {strategies[j].archetype.code} may be too similar (overlap={ratio:.2f})."
                )
    return warnings


def _offline_strategies(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    strategies: List[Strategy] = []

    offline_profiles = {
        "DIRECT": {
            "summary": "Fastest path to the target role by leveraging existing overlap and targeting immediate market entry.",
            "best_for_profile": "Candidates with strong transferable credibility and low tolerance for long detours.",
            "evidence_strategy": "Use a tight interview narrative plus one fast proof artifact.",
            "key_tradeoff": "Speed is high, but downside risk is also high.",
            "confidence_rationale": "Works best when the current role already signals partial readiness.",
        },
        "STEPPING": {
            "summary": "Use adjacent bridge roles to make the pivot easier to sell and safer in the market.",
            "best_for_profile": "Candidates who need credibility compounding and want lower downside risk.",
            "evidence_strategy": "Create proof gradually through adjacent-role accomplishments.",
            "key_tradeoff": "More realistic, but slower than a direct jump.",
            "confidence_rationale": "Stronger when the target role is materially different from the current role.",
        },
        "SKILL_FIRST": {
            "summary": "Prioritize critical missing capabilities before pushing hard on the market.",
            "best_for_profile": "Candidates with large high-signal skill gaps.",
            "evidence_strategy": "Show capability lift through targeted outputs and practical exercises.",
            "key_tradeoff": "Capability improves first, but market entry is delayed.",
            "confidence_rationale": "Strong when missing skills are the main blocker.",
        },
        "PORTFOLIO": {
            "summary": "Build visible artifacts that prove target-role readiness before relying on resume claims.",
            "best_for_profile": "Candidates entering proof-heavy markets where visible work matters.",
            "evidence_strategy": "Create two or three public-facing proof artifacts tied to target-role work.",
            "key_tradeoff": "Requires effort upfront, but makes the narrative much more credible.",
            "confidence_rationale": "Strong when employers want concrete evidence rather than abstract claims.",
        },
        "HYBRID": {
            "summary": "Balance skill-building, evidence, and market realism to create a robust transition strategy.",
            "best_for_profile": "Candidates who want a practical path without overcommitting to one lever.",
            "evidence_strategy": "Combine targeted skill building with visible market-facing outputs.",
            "key_tradeoff": "Not the fastest single path, but usually the most resilient.",
            "confidence_rationale": "Strong when the pivot requires both proof and credibility, but not a complete reset.",
        },
    }

    for code, archetype in STRATEGY_ARCHETYPES.items():
        profile = offline_profiles[code]
        signal = _strategy_style_hints(code)["signal_profile"]

        strategies.append(
            Strategy(
                archetype=archetype,
                current_role=current_role,
                target_role=target_role,
                summary=profile["summary"],
                phases=[
                    StrategyPhase(
                        phase="0-30 days",
                        objective="Define the pivot positioning and high-priority workstream.",
                        deliverables=["Pivot positioning brief", "Prioritized gap list"],
                        key_actions=["Clarify target angle", "Select highest-signal opportunities", "Set evidence targets"],
                    ),
                    StrategyPhase(
                        phase="30-90 days",
                        objective="Build enough credibility to change the quality of the market conversation.",
                        deliverables=["One visible artifact", "Updated interview story bank"],
                        key_actions=["Create concrete evidence", "Tighten story", "Test assumptions against real postings"],
                    ),
                    StrategyPhase(
                        phase="90+ days",
                        objective="Execute the pivot with a strategy-specific go-to-market plan.",
                        deliverables=["Applications", "Feedback log"],
                        key_actions=["Apply selectively", "Gather feedback", "Iterate based on weak signal areas"],
                    ),
                ],
                key_missing_skills=missing_skills[:6],
                transferable_anchors=transfer_skills[:5],
                success_criteria=[
                    "Better market credibility",
                    "Stronger proof of fit",
                    "Clearer interview narrative",
                ],
                potential_risks=["Weak external signal", "Timeline drift"],
                resources_needed=["Focused time", "Practical output", "Feedback loop"],
                best_for_profile=profile["best_for_profile"],
                evidence_strategy=profile["evidence_strategy"],
                key_tradeoff=profile["key_tradeoff"],
                confidence_rationale=profile["confidence_rationale"],
                speed_bias=signal["speed_bias"],
                risk_bias=signal["risk_bias"],
                evidence_burden=signal["evidence_burden"],
                market_signal_strength=signal["market_signal_strength"],
                skill_gap_focus=signal["skill_gap_focus"],
            )
        )

    return {
        "strategies": strategies,
        "trace": {
            "mode": "offline",
            "errors": ["No API key or API unavailable"],
            "diversity_warnings": _compute_diversity_warnings(strategies),
        },
        "source": "Offline (deterministic)",
    }


def generate_all_strategies(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    missing_skills, transfer_skills, gap_summary = _summarize_gap_df(gap_df)

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

    trace: Dict[str, Any] = {
        "mode": "online",
        "model": model,
        "strategies_generated": 0,
        "errors": [],
        "raw_outputs": {},
        "gap_summary": gap_summary,
        "missing_skills": missing_skills,
        "transfer_skills": transfer_skills,
        "diversity_warnings": [],
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

            resp = client.responses.create(model=model, input=prompt)
            raw_text = (resp.output_text or "").strip()
            trace["raw_outputs"][code] = raw_text

            obj = _extract_json_object(raw_text)
            strategy = _normalize_strategy_object(
                obj=obj,
                current_role=current_role,
                target_role=target_role,
                archetype=archetype,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
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

    trace["diversity_warnings"] = _compute_diversity_warnings(strategies)

    return {
        "strategies": strategies,
        "trace": trace,
        "source": "OpenAI",
    }


def _offline_evaluations(
    *,
    strategies: List[Strategy],
    reviewer_personas: List[str],
) -> Dict[str, Any]:
    evaluations: List[ReviewerEvaluation] = []

    persona_bias = {
        "HiringManager": {"DIRECT": 72, "STEPPING": 79, "SKILL_FIRST": 75, "PORTFOLIO": 81, "HYBRID": 84},
        "Recruiter": {"DIRECT": 68, "STEPPING": 82, "SKILL_FIRST": 73, "PORTFOLIO": 77, "HYBRID": 83},
        "PortfolioEval": {"DIRECT": 65, "STEPPING": 73, "SKILL_FIRST": 76, "PORTFOLIO": 88, "HYBRID": 82},
        "RiskAnalyst": {"DIRECT": 58, "STEPPING": 84, "SKILL_FIRST": 80, "PORTFOLIO": 75, "HYBRID": 86},
        "CareerCoach": {"DIRECT": 67, "STEPPING": 79, "SKILL_FIRST": 81, "PORTFOLIO": 80, "HYBRID": 87},
    }

    for persona in reviewer_personas:
        scores: List[ReviewerScore] = []
        for strategy in strategies:
            code = strategy.archetype.code
            overall = float(persona_bias.get(persona, {}).get(code, 75.0))

            scores.append(
                ReviewerScore(
                    reviewer_persona=persona,
                    strategy_code=code,
                    overall_score=overall,
                    alignment_with_role=min(10.0, max(0.0, overall / 10.0 - 0.2)),
                    market_feasibility=min(10.0, max(0.0, overall / 10.0 - 0.1)),
                    time_efficiency=min(10.0, max(0.0, strategy.speed_bias)),
                    risk_assessment=min(10.0, max(0.0, strategy.risk_bias)),
                    narrative_strength=min(10.0, max(0.0, strategy.market_signal_strength)),
                    justification=f"{persona} offline evaluation of {code} based on its trade-offs and market signal.",
                    concerns=["Execution quality matters", "Weak evidence would reduce this score"],
                    best_strength=f"Strongest trait is {strategy.key_tradeoff or 'its differentiated logic'}.",
                    biggest_risk=strategy.potential_risks[0] if strategy.potential_risks else "Execution risk",
                    killer_objection="The plan fails if the candidate cannot convert theory into visible proof quickly.",
                    success_condition="The candidate executes the stated sequencing with discipline.",
                    best_candidate_fit=strategy.best_for_profile or "Candidates with aligned constraints and motivation.",
                )
            )

        evaluations.append(
            ReviewerEvaluation(
                reviewer_persona=persona,
                strategy_scores=scores,
                overall_recommendation="Use the highest-scoring strategy unless disagreement and fragility are too high.",
                strongest_strategy=max(scores, key=lambda s: s.overall_score).strategy_code if scores else "HYBRID",
                weakest_strategy=min(scores, key=lambda s: s.overall_score).strategy_code if scores else "DIRECT",
                reviewer_weight=REVIEWER_WEIGHTS.get(persona, 1.0),
            )
        )

    return {
        "evaluations": evaluations,
        "trace": {"mode": "offline", "errors": ["No API key or API unavailable"]},
        "source": "Offline (deterministic)",
    }


def evaluate_strategies_by_reviewers(
    *,
    strategies: List[Strategy],
    current_role: str,
    target_role: str,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
) -> Dict[str, Any]:
    reviewer_personas = REVIEWER_PERSONAS[:]

    if not prefer_online:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    api_key = _get_api_key_optional()
    if not api_key:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    try:
        from openai import OpenAI
    except Exception:
        return _offline_evaluations(strategies=strategies, reviewer_personas=reviewer_personas)

    trace: Dict[str, Any] = {
        "mode": "online",
        "model": model,
        "evaluations_completed": 0,
        "errors": [],
        "raw_outputs": {},
        "reviewer_weights": REVIEWER_WEIGHTS,
    }

    client = OpenAI(api_key=api_key)
    evaluations: List[ReviewerEvaluation] = []

    for persona in reviewer_personas:
        try:
            scores: List[ReviewerScore] = []

            for strategy in strategies:
                prompt = _build_reviewer_prompt(
                    reviewer_persona=persona,
                    strategy_json=strategy.model_dump(),
                    current_role=current_role,
                    target_role=target_role,
                )

                resp = client.responses.create(model=model, input=prompt)
                raw_text = (resp.output_text or "").strip()
                trace["raw_outputs"][f"{persona}_{strategy.archetype.code}"] = raw_text

                obj = _extract_json_object(raw_text)
                score = _normalize_reviewer_score(
                    obj=obj,
                    persona=persona,
                    strategy_code=strategy.archetype.code,
                )
                scores.append(score)

            strongest = max(scores, key=lambda s: s.overall_score).strategy_code if scores else "HYBRID"
            weakest = min(scores, key=lambda s: s.overall_score).strategy_code if scores else "DIRECT"

            evaluations.append(
                ReviewerEvaluation(
                    reviewer_persona=persona,
                    strategy_scores=scores,
                    overall_recommendation=(
                        f"{persona} favors {strongest} over {weakest} given its implicit priorities."
                    ),
                    strongest_strategy=strongest,
                    weakest_strategy=weakest,
                    reviewer_weight=REVIEWER_WEIGHTS.get(persona, 1.0),
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