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


def _sanitize_text(value: Any, max_len: int = 400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_len]


def _sanitize_list(items: Any, *, max_n: int, max_len: int = 140) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen = set()
    for item in items:
        s = _sanitize_text(item, max_len=max_len)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _safe_float(x: Any, default: float, lo: float, hi: float) -> float:
    try:
        val = float(x)
    except Exception:
        val = float(default)
    return max(lo, min(hi, val))


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
        description="Immediate jump to the target role. Optimizes for speed and early market exposure.",
        estimated_days=45,
        risk_level="high",
    ),
    "STEPPING": StrategyArchetype(
        name="Stepping-Stone Pivot",
        code="STEPPING",
        description="Uses adjacent bridge roles to reduce market-entry risk and improve credibility.",
        estimated_days=150,
        risk_level="medium",
    ),
    "SKILL_FIRST": StrategyArchetype(
        name="Skill-First Pivot",
        code="SKILL_FIRST",
        description="Prioritizes closing the highest-signal missing capabilities before a strong market push.",
        estimated_days=120,
        risk_level="low",
    ),
    "PORTFOLIO": StrategyArchetype(
        name="Portfolio-First Pivot",
        code="PORTFOLIO",
        description="Creates visible proof of ability first to make the pivot more credible externally.",
        estimated_days=100,
        risk_level="medium",
    ),
    "HYBRID": StrategyArchetype(
        name="Hybrid / Balanced",
        code="HYBRID",
        description="Balances speed, capability-building, portfolio proof, and labor-market realism.",
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
    "RiskAnalyst": 1.20,
    "CareerCoach": 0.95,
}


def _strategy_profile(code: str) -> Dict[str, Any]:
    code = str(code).upper()
    mapping = {
        "DIRECT": {
            "goal": "maximize speed to target role",
            "best_for": "Candidates with unusually strong overlap, high risk tolerance, and immediate willingness to test themselves in the market.",
            "evidence": "Use a tight narrative plus fast, high-signal experience such as auditions, public performance, workshops, or real target-role trials.",
            "tradeoff": "Fastest path, but also the least forgiving if the market rejects the first signal.",
            "confidence": "Only realistic when transferable strengths are unusually strong or the target field accepts rough-entry experimentation.",
            "biases": {"speed": 9.0, "risk": 2.5, "evidence": 7.0, "market": 6.5, "gap": 4.5},
            "must_do": [
                "act fast",
                "accept feedback from the market quickly",
                "avoid over-preparing before first market exposure",
            ],
            "must_avoid": [
                "long detours",
                "course collection without external testing",
                "pretending risk is low",
            ],
        },
        "STEPPING": {
            "goal": "minimize transition risk via adjacent bridge roles",
            "best_for": "Candidates who need credibility compounding and cannot afford a very risky direct jump.",
            "evidence": "Build proof through adjacent experiences that move the profile closer to the target role over time.",
            "tradeoff": "Safer and more sellable, but slower and less emotionally exciting than a direct leap.",
            "confidence": "Most realistic when the gap is meaningful and there are plausible intermediate positions or adjacent signals.",
            "biases": {"speed": 4.0, "risk": 8.5, "evidence": 5.5, "market": 8.5, "gap": 6.0},
            "must_do": [
                "identify bridge roles",
                "sequence the transition logically",
                "show compounding credibility",
            ],
            "must_avoid": [
                "fake bridge roles that do not actually improve fit",
                "aimless drift",
                "unclear sequencing",
            ],
        },
        "SKILL_FIRST": {
            "goal": "close critical missing skills before serious market push",
            "best_for": "Candidates whose biggest blocker is true capability deficit rather than positioning alone.",
            "evidence": "Demonstrate skill acquisition through measurable outputs, practice loops, and role-relevant exercises.",
            "tradeoff": "Improves actual readiness, but delays aggressive market entry.",
            "confidence": "Strong when the target role clearly requires capabilities that are not yet present at a credible level.",
            "biases": {"speed": 3.0, "risk": 8.0, "evidence": 6.5, "market": 6.0, "gap": 9.0},
            "must_do": [
                "prioritize highest-signal gaps",
                "turn learning into measurable outputs",
                "avoid vague self-development",
            ],
            "must_avoid": [
                "learning without proof",
                "trying to close every gap at once",
                "premature applications",
            ],
        },
        "PORTFOLIO": {
            "goal": "maximize visible proof of capability",
            "best_for": "Candidates entering markets where tangible artifacts, demos, or visible public work strongly affect credibility.",
            "evidence": "Create public or semi-public artifacts that make competence legible to external evaluators.",
            "tradeoff": "High proof value, but requires disciplined creation and curation of visible work.",
            "confidence": "Strong when evidence can materially change how outsiders evaluate the pivot.",
            "biases": {"speed": 5.0, "risk": 6.0, "evidence": 9.0, "market": 8.5, "gap": 7.0},
            "must_do": [
                "build concrete artifacts",
                "choose visible proof that hiring-side audiences care about",
                "connect outputs to target-role expectations",
            ],
            "must_avoid": [
                "generic projects",
                "private work no one sees",
                "evidence that does not map to the target role",
            ],
        },
        "HYBRID": {
            "goal": "balance speed, evidence, learning, and realism",
            "best_for": "Candidates who need a robust path instead of a single aggressive bet.",
            "evidence": "Combine targeted upskilling with visible outputs and a realistic market entry sequence.",
            "tradeoff": "Most balanced option, but requires stronger prioritization discipline to avoid becoming too broad.",
            "confidence": "Usually strongest when both readiness and proof matter and no single lever solves everything.",
            "biases": {"speed": 6.5, "risk": 7.0, "evidence": 7.5, "market": 7.5, "gap": 7.5},
            "must_do": [
                "sequence actions carefully",
                "balance proof and learning",
                "avoid overcommitting to one lever too early",
            ],
            "must_avoid": [
                "doing everything at once",
                "weak prioritization",
                "strategy drift",
            ],
        },
    }
    return mapping[code]


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
        f"{len(missing)} positive skill gaps. "
        f"Average gap {avg_gap:.2f}. "
        f"Highest-signal missing skills: {', '.join(high_signal_missing) if high_signal_missing else 'none'}. "
        f"Top transferable anchors: {', '.join(transfer_skills[:3]) if transfer_skills else 'none'}."
    )
    return missing_skills, transfer_skills, gap_summary


def _fallback_strategy_payload(
    *,
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    profile = _strategy_profile(archetype.code)
    b = profile["biases"]

    objective_map = {
        "DIRECT": [
            "Package the pivot quickly and get to real market feedback fast.",
            "Build minimum viable credibility and test it externally.",
            "Push aggressively into auditions, interviews, or real opportunities.",
        ],
        "STEPPING": [
            "Define the bridge-role path that makes the pivot believable.",
            "Accumulate adjacent evidence that narrows the credibility gap.",
            "Convert bridge-role credibility into target-role entry.",
        ],
        "SKILL_FIRST": [
            "Identify the highest-signal missing capabilities and set a focused practice plan.",
            "Convert core skill gaps into measurable improvement.",
            "Enter the market once the minimum capability threshold is more credible.",
        ],
        "PORTFOLIO": [
            "Choose target-role proof artifacts that outsiders will actually value.",
            "Build visible evidence that demonstrates role-relevant ability.",
            "Use the portfolio to change how the market reads the pivot.",
        ],
        "HYBRID": [
            "Define a balanced plan for proof, skill-building, and market entry.",
            "Build both readiness and visible signal in parallel.",
            "Launch with a more robust, evidence-backed transition narrative.",
        ],
    }

    return {
        "summary": (
            f"{archetype.name} is a {profile['goal']} strategy for moving from {current_role} to {target_role}. "
            f"It is designed for candidates where this trade-off profile is more appropriate than the alternatives."
        ),
        "phases": [
            {
                "phase": "0-30 days",
                "objective": objective_map[archetype.code][0],
                "deliverables": ["Pivot positioning brief", "Priority action list"],
                "key_actions": profile["must_do"][:3],
            },
            {
                "phase": "30-90 days",
                "objective": objective_map[archetype.code][1],
                "deliverables": ["Visible evidence milestone", "Narrative refinement"],
                "key_actions": [
                    "Test progress against real market expectations",
                    "Collect feedback from credible outsiders",
                    "Refine the weak parts of the transition story",
                ],
            },
            {
                "phase": "90+ days",
                "objective": objective_map[archetype.code][2],
                "deliverables": ["Market push", "Iteration loop"],
                "key_actions": [
                    "Apply or audition selectively",
                    "Track rejection and feedback patterns",
                    "Double down on what increases credibility",
                ],
            },
        ],
        "key_missing_skills": missing_skills[:6],
        "transferable_anchors": transfer_skills[:5],
        "success_criteria": [
            "Stronger external credibility",
            "Better proof of fit",
            "Clearer market narrative",
        ],
        "potential_risks": [
            profile["tradeoff"],
            "Execution quality may be weaker than planned",
        ],
        "resources_needed": [
            "Focused time",
            "Feedback from real evaluators",
            "Visible evidence of progress",
        ],
        "best_for_profile": profile["best_for"],
        "evidence_strategy": profile["evidence"],
        "key_tradeoff": profile["tradeoff"],
        "confidence_rationale": profile["confidence"],
        "speed_bias": b["speed"],
        "risk_bias": b["risk"],
        "evidence_burden": b["evidence"],
        "market_signal_strength": b["market"],
        "skill_gap_focus": b["gap"],
    }


def _build_strategy_prompt(
    *,
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
    gap_summary: str,
) -> str:
    profile = _strategy_profile(archetype.code)
    fallback = _fallback_strategy_payload(
        current_role=current_role,
        target_role=target_role,
        archetype=archetype,
        missing_skills=missing_skills,
        transfer_skills=transfer_skills,
    )

    return f"""
You are generating ONE strategy inside a multi-strategy career pivot decision engine.

This is critical:
- The strategy must feel genuinely DIFFERENT from the other archetypes.
- Do not write generic career advice.
- Make the trade-off profile explicit.
- Fill every field.
- Return valid JSON only.

Current role: {current_role}
Target role: {target_role}

Archetype:
- code: {archetype.code}
- name: {archetype.name}
- description: {archetype.description}
- estimated_days: {archetype.estimated_days}
- risk_level: {archetype.risk_level}

Optimization goal:
{profile["goal"]}

Must emphasize:
{json.dumps(profile["must_do"], ensure_ascii=False)}

Must avoid:
{json.dumps(profile["must_avoid"], ensure_ascii=False)}

Top missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Top transferable anchors:
{json.dumps(transfer_skills, ensure_ascii=False)}

Gap summary:
{gap_summary}

Use this as the style reference for what a good answer contains, but generate your own better version:
{json.dumps(fallback, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this shape:
{{
  "summary": "2-4 sentence strategy summary",
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
  "best_for_profile": "string",
  "evidence_strategy": "string",
  "key_tradeoff": "string",
  "confidence_rationale": "string",
  "speed_bias": 0,
  "risk_bias": 0,
  "evidence_burden": 0,
  "market_signal_strength": 0,
  "skill_gap_focus": 0
}}
""".strip()


def _normalize_strategy(
    *,
    obj: Dict[str, Any],
    current_role: str,
    target_role: str,
    archetype: StrategyArchetype,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Strategy:
    fallback = _fallback_strategy_payload(
        current_role=current_role,
        target_role=target_role,
        archetype=archetype,
        missing_skills=missing_skills,
        transfer_skills=transfer_skills,
    )
    profile = _strategy_profile(archetype.code)

    raw_phases = obj.get("phases", fallback["phases"])
    phases: List[StrategyPhase] = []
    for idx, p in enumerate(raw_phases[:4]):
        default_phase = fallback["phases"][min(idx, 2)]
        phases.append(
            StrategyPhase(
                phase=_sanitize_text(p.get("phase", default_phase["phase"]), 40),
                objective=_sanitize_text(p.get("objective", default_phase["objective"]), 180),
                deliverables=_sanitize_list(p.get("deliverables", default_phase["deliverables"]), max_n=4),
                key_actions=_sanitize_list(p.get("key_actions", default_phase["key_actions"]), max_n=5),
            )
        )

    if len(phases) < 3:
        phases = [StrategyPhase(**x) for x in fallback["phases"]]

    b = profile["biases"]

    return Strategy(
        archetype=archetype,
        current_role=current_role,
        target_role=target_role,
        summary=_sanitize_text(obj.get("summary", fallback["summary"]), 700),
        phases=phases,
        key_missing_skills=_sanitize_list(obj.get("key_missing_skills", fallback["key_missing_skills"]), max_n=6),
        transferable_anchors=_sanitize_list(obj.get("transferable_anchors", fallback["transferable_anchors"]), max_n=5),
        success_criteria=_sanitize_list(obj.get("success_criteria", fallback["success_criteria"]), max_n=5),
        potential_risks=_sanitize_list(obj.get("potential_risks", fallback["potential_risks"]), max_n=4),
        resources_needed=_sanitize_list(obj.get("resources_needed", fallback["resources_needed"]), max_n=6),
        best_for_profile=_sanitize_text(obj.get("best_for_profile", fallback["best_for_profile"]), 240),
        evidence_strategy=_sanitize_text(obj.get("evidence_strategy", fallback["evidence_strategy"]), 320),
        key_tradeoff=_sanitize_text(obj.get("key_tradeoff", fallback["key_tradeoff"]), 220),
        confidence_rationale=_sanitize_text(obj.get("confidence_rationale", fallback["confidence_rationale"]), 320),
        speed_bias=_safe_float(obj.get("speed_bias", b["speed"]), b["speed"], 0.0, 10.0),
        risk_bias=_safe_float(obj.get("risk_bias", b["risk"]), b["risk"], 0.0, 10.0),
        evidence_burden=_safe_float(obj.get("evidence_burden", b["evidence"]), b["evidence"], 0.0, 10.0),
        market_signal_strength=_safe_float(obj.get("market_signal_strength", b["market"]), b["market"], 0.0, 10.0),
        skill_gap_focus=_safe_float(obj.get("skill_gap_focus", b["gap"]), b["gap"], 0.0, 10.0),
    )


def _signature(strategy: Strategy) -> set[str]:
    tokens = set()
    tokens.add(strategy.archetype.code.lower())
    tokens.update([x.lower() for x in strategy.key_missing_skills[:6]])
    tokens.update([x.lower() for x in strategy.success_criteria[:3]])
    for phase in strategy.phases[:3]:
        tokens.update([x.lower() for x in phase.key_actions[:3]])
    return tokens


def _compute_diversity_warnings(strategies: List[Strategy]) -> List[str]:
    warnings: List[str] = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a = _signature(strategies[i])
            b = _signature(strategies[j])
            if not a or not b:
                continue
            overlap = len(a & b) / max(1, len(a | b))
            if overlap >= 0.45:
                warnings.append(
                    f"{strategies[i].archetype.code} and {strategies[j].archetype.code} may still be too similar ({overlap:.2f} overlap)."
                )
    return warnings


def _build_reviewer_prompt(
    *,
    reviewer_persona: str,
    strategy: Strategy,
    current_role: str,
    target_role: str,
) -> str:
    persona_context = {
        "HiringManager": "You are a hiring manager. Care most about role readiness, credibility, and whether this person could survive real evaluation.",
        "Recruiter": "You are a recruiter. Care most about market positioning, sellability, and whether the pivot can be explained convincingly.",
        "PortfolioEval": "You are a portfolio evaluator. Care most about proof, visible evidence, and whether claims are backed by artifacts or public signal.",
        "RiskAnalyst": "You are a risk analyst. Care most about downside, fragile assumptions, hidden blockers, and time-to-realism.",
        "CareerCoach": "You are a career coach. Care most about sustainability, sequencing, motivation, and whether the plan is psychologically executable.",
    }

    return f"""
{persona_context.get(reviewer_persona, "")}

Evaluate this career pivot strategy for moving from {current_role} to {target_role}.

Strategy JSON:
{json.dumps(strategy.model_dump(), ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "alignment_with_role": 0,
  "market_feasibility": 0,
  "time_efficiency": 0,
  "risk_assessment": 0,
  "narrative_strength": 0,
  "overall_score": 0,
  "justification": "2-4 sentence explanation",
  "concerns": ["string", "string"],
  "best_strength": "string",
  "biggest_risk": "string",
  "killer_objection": "string",
  "success_condition": "string",
  "best_candidate_fit": "string"
}}

Rules:
- score 0-10 for dimensions
- overall_score 0-100
- be discriminative
- do not make all strategies similar
- let your persona preferences visibly matter
""".strip()


def _persona_anchor_score(persona: str, strategy: Strategy) -> Dict[str, float]:
    code = strategy.archetype.code

    # hard persona preferences
    persona_bias = {
        "HiringManager": {"DIRECT": -1.2, "STEPPING": 0.6, "SKILL_FIRST": 0.4, "PORTFOLIO": 0.9, "HYBRID": 1.1},
        "Recruiter": {"DIRECT": -0.8, "STEPPING": 1.0, "SKILL_FIRST": 0.2, "PORTFOLIO": 0.5, "HYBRID": 1.0},
        "PortfolioEval": {"DIRECT": -1.0, "STEPPING": 0.1, "SKILL_FIRST": 0.5, "PORTFOLIO": 1.6, "HYBRID": 0.9},
        "RiskAnalyst": {"DIRECT": -1.8, "STEPPING": 1.2, "SKILL_FIRST": 1.1, "PORTFOLIO": 0.3, "HYBRID": 1.0},
        "CareerCoach": {"DIRECT": -0.9, "STEPPING": 0.8, "SKILL_FIRST": 1.0, "PORTFOLIO": 0.6, "HYBRID": 1.1},
    }
    bias = persona_bias.get(persona, {}).get(code, 0.0)

    alignment = 5.5 + (strategy.skill_gap_focus * 0.10) + (strategy.market_signal_strength * 0.12) + bias
    market = 5.0 + (strategy.market_signal_strength * 0.18) + (strategy.risk_bias * 0.10) + bias
    time_eff = 4.5 + (strategy.speed_bias * 0.22) - (strategy.evidence_burden * 0.06)
    risk = 4.5 + (strategy.risk_bias * 0.22) - (strategy.speed_bias * 0.10)
    narrative = 5.0 + (strategy.market_signal_strength * 0.14) + (strategy.evidence_burden * 0.08) + bias * 0.4

    dims = {
        "alignment_with_role": max(0.0, min(10.0, alignment)),
        "market_feasibility": max(0.0, min(10.0, market)),
        "time_efficiency": max(0.0, min(10.0, time_eff)),
        "risk_assessment": max(0.0, min(10.0, risk)),
        "narrative_strength": max(0.0, min(10.0, narrative)),
    }
    dims["overall_score"] = sum(dims.values()) / 5.0 * 10.0
    return dims


def _normalize_reviewer_score(
    *,
    obj: Dict[str, Any],
    persona: str,
    strategy: Strategy,
) -> ReviewerScore:
    anchor = _persona_anchor_score(persona, strategy)

    # blend weak LLM output with deterministic anchor so results stay useful
    def blended(field: str) -> float:
        llm_val = _safe_float(obj.get(field, anchor[field]), anchor[field], 0.0, 10.0)
        return max(0.0, min(10.0, 0.45 * llm_val + 0.55 * anchor[field]))

    alignment = blended("alignment_with_role")
    market = blended("market_feasibility")
    time_eff = blended("time_efficiency")
    risk = blended("risk_assessment")
    narrative = blended("narrative_strength")

    llm_overall = _safe_float(obj.get("overall_score", 0.0), 0.0, 0.0, 100.0)
    computed = (alignment + market + time_eff + risk + narrative) / 5.0 * 10.0
    if llm_overall <= 0.0:
        overall = computed
    else:
        overall = max(0.0, min(100.0, 0.35 * llm_overall + 0.65 * computed))

    return ReviewerScore(
        reviewer_persona=persona,
        strategy_code=strategy.archetype.code,
        overall_score=overall,
        alignment_with_role=alignment,
        market_feasibility=market,
        time_efficiency=time_eff,
        risk_assessment=risk,
        narrative_strength=narrative,
        justification=_sanitize_text(
            obj.get(
                "justification",
                f"{persona} sees {strategy.archetype.code} as a differentiated strategy with clear trade-offs."
            ),
            500,
        ),
        concerns=_sanitize_list(obj.get("concerns", []), max_n=4) or [
            "Execution quality will strongly affect outcomes.",
            "Market reaction may differ from internal confidence.",
        ],
        best_strength=_sanitize_text(
            obj.get("best_strength", strategy.key_tradeoff or strategy.best_for_profile),
            240,
        ),
        biggest_risk=_sanitize_text(
            obj.get("biggest_risk", (strategy.potential_risks[0] if strategy.potential_risks else "Execution risk")),
            240,
        ),
        killer_objection=_sanitize_text(
            obj.get("killer_objection", "This plan fails if it does not produce external credibility quickly enough."),
            240,
        ),
        success_condition=_sanitize_text(
            obj.get("success_condition", "The candidate must execute the sequencing with discipline and visible proof."),
            240,
        ),
        best_candidate_fit=_sanitize_text(
            obj.get("best_candidate_fit", strategy.best_for_profile),
            240,
        ),
    )


def _offline_strategies(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    strategies: List[Strategy] = []
    for code, archetype in STRATEGY_ARCHETYPES.items():
        payload = _fallback_strategy_payload(
            current_role=current_role,
            target_role=target_role,
            archetype=archetype,
            missing_skills=missing_skills,
            transfer_skills=transfer_skills,
        )
        strategies.append(
            _normalize_strategy(
                obj=payload,
                current_role=current_role,
                target_role=target_role,
                archetype=archetype,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
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
        "diversity_warnings": [],
    }

    client = OpenAI(api_key=api_key)
    strategies: List[Strategy] = []

    for code, archetype in STRATEGY_ARCHETYPES.items():
        try:
            prompt = _build_strategy_prompt(
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
            strategy = _normalize_strategy(
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
            fallback_obj = _fallback_strategy_payload(
                current_role=current_role,
                target_role=target_role,
                archetype=archetype,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
            )
            strategies.append(
                _normalize_strategy(
                    obj=fallback_obj,
                    current_role=current_role,
                    target_role=target_role,
                    archetype=archetype,
                    missing_skills=missing_skills,
                    transfer_skills=transfer_skills,
                )
            )

    trace["diversity_warnings"] = _compute_diversity_warnings(strategies)

    return {
        "strategies": strategies,
        "trace": trace,
        "source": "OpenAI" if trace["strategies_generated"] > 0 else "Offline (deterministic)",
    }


def _offline_evaluations(
    *,
    strategies: List[Strategy],
    reviewer_personas: List[str],
) -> Dict[str, Any]:
    evaluations: List[ReviewerEvaluation] = []

    for persona in reviewer_personas:
        scores: List[ReviewerScore] = []
        for strategy in strategies:
            scores.append(
                _normalize_reviewer_score(
                    obj={},
                    persona=persona,
                    strategy=strategy,
                )
            )

        strongest = max(scores, key=lambda s: s.overall_score).strategy_code
        weakest = min(scores, key=lambda s: s.overall_score).strategy_code

        evaluations.append(
            ReviewerEvaluation(
                reviewer_persona=persona,
                strategy_scores=scores,
                overall_recommendation=f"{persona} currently prefers {strongest} over {weakest} given its priorities.",
                strongest_strategy=strongest,
                weakest_strategy=weakest,
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
        scores: List[ReviewerScore] = []

        for strategy in strategies:
            try:
                prompt = _build_reviewer_prompt(
                    reviewer_persona=persona,
                    strategy=strategy,
                    current_role=current_role,
                    target_role=target_role,
                )
                resp = client.responses.create(model=model, input=prompt)
                raw_text = (resp.output_text or "").strip()
                trace["raw_outputs"][f"{persona}_{strategy.archetype.code}"] = raw_text
                obj = _extract_json_object(raw_text)
            except Exception as e:
                trace["errors"].append(f"{persona}_{strategy.archetype.code}: {repr(e)}")
                obj = {}

            scores.append(
                _normalize_reviewer_score(
                    obj=obj,
                    persona=persona,
                    strategy=strategy,
                )
            )

        strongest = max(scores, key=lambda s: s.overall_score).strategy_code
        weakest = min(scores, key=lambda s: s.overall_score).strategy_code

        evaluations.append(
            ReviewerEvaluation(
                reviewer_persona=persona,
                strategy_scores=scores,
                overall_recommendation=f"{persona} prefers {strongest} and is most skeptical of {weakest}.",
                strongest_strategy=strongest,
                weakest_strategy=weakest,
                reviewer_weight=REVIEWER_WEIGHTS.get(persona, 1.0),
            )
        )
        trace["evaluations_completed"] += 1

    return {
        "evaluations": evaluations,
        "trace": trace,
        "source": "OpenAI" if trace["evaluations_completed"] > 0 else "Offline (deterministic)",
    }