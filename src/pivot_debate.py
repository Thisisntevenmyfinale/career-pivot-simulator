"""
Adversarial Pivot Debate
=========================
A genuine multi-agent adversarial architecture:

  Agent 1 — Advocate   : builds the strongest possible case FOR the pivot
  Agent 2 — Skeptic    : finds every reason AGAINST the pivot
  Agent 3 — Judge      : reads both arguments, weighs evidence, delivers
                         a probability-style verdict with explicit reasoning

Why this is architecturally distinct from the review board
----------------------------------------------------------
The review board runs 5 independent evaluators in parallel — each scores
strategies without seeing the others' reasoning.

The debate is adversarial and sequential:
  1. Advocate argues first, forced to be maximally constructive
  2. Skeptic argues second, forced to be maximally critical
  3. Judge sees BOTH arguments and must explicitly address the strongest
     point on each side before reaching a verdict

This produces qualitatively richer output because the judge cannot ignore
the best objection — it must be addressed and either refuted or accepted.

The probability output (e.g. "68% viable") is not a statistical claim —
it is a calibrated confidence expression from the judge, grounded in the
specific evidence from both sides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DebateRound:
    """One side's argument in the debate."""
    side: str          # "advocate" | "skeptic"
    main_argument: str
    strongest_evidence: List[str] = field(default_factory=list)
    key_risks_acknowledged: List[str] = field(default_factory=list)
    closing_statement: str = ""


@dataclass
class DebateVerdict:
    """Judge's synthesis of both sides."""
    pivot_viability_pct: int         # 0-100 probability-style confidence
    verdict_label: str               # "Strong Case", "Viable with Conditions", "Weak Case", "Not Recommended"
    decisive_factor: str             # the single most important consideration
    strongest_pro_argument: str      # what the advocate got right
    strongest_con_argument: str      # what the skeptic got right
    judge_reasoning: str             # full synthesis
    conditions_for_success: List[str] = field(default_factory=list)
    conditions_for_failure: List[str] = field(default_factory=list)
    recommended_next_action: str = ""
    source: str = "online"


def _build_debate_context(
    current_role: str,
    target_role: str,
    match_score: float,
    gap_summary: str,
    market_signal: Optional[Dict] = None,
    agent_summary: Optional[str] = None,
    consensus_winner: Optional[str] = None,
    cv_profile: Optional[Dict] = None,
) -> str:
    """Build shared evidence context for all three agents."""
    ctx_parts = [
        f"PIVOT: {current_role} → {target_role}",
        f"Skill match score: {match_score:.0f}/100",
        f"Gap analysis: {gap_summary}",
    ]
    if consensus_winner:
        ctx_parts.append(f"Review board recommended strategy: {consensus_winner}")
    if agent_summary:
        ctx_parts.append(f"Agent analysis: {agent_summary}")
    if market_signal and not market_signal.get("error"):
        demand = market_signal.get("job_demand", "Unknown")
        outlook = market_signal.get("growth_outlook", "Unknown")
        hot = ", ".join(market_signal.get("top_employer_skills", [])[:4])
        ctx_parts.append(f"Market signal: Demand={demand}, Outlook={outlook}, Top employer skills: {hot}")
    if cv_profile and cv_profile.get("extracted_role"):
        p = cv_profile
        top = ", ".join(p.get("top_skills", [])[:5])
        ctx_parts.append(
            f"Candidate profile: {p.get('extracted_role')}, "
            f"{p.get('years_experience',0):.0f} yrs, {p.get('education_level','')}, "
            f"Top skills: {top}"
        )
    return "\n".join(ctx_parts)


def run_pivot_debate(
    current_role: str,
    target_role: str,
    match_score: float,
    gap_summary: str,
    market_signal: Optional[Dict] = None,
    agent_summary: Optional[str] = None,
    consensus_winner: Optional[str] = None,
    cv_profile: Optional[Dict] = None,
    model_debate: str = "gpt-4o-mini",
    model_judge: str = "gpt-4o",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the three-agent adversarial debate.

    Returns a dict with:
      advocate: DebateRound
      skeptic:  DebateRound
      verdict:  DebateVerdict
    """
    if not prefer_online:
        return _offline_debate(current_role, target_role, match_score)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_debate(current_role, target_role, match_score)

    context = _build_debate_context(
        current_role, target_role, match_score, gap_summary,
        market_signal, agent_summary, consensus_winner, cv_profile
    )

    # ── Agent 1: Advocate ─────────────────────────────────────────────────────
    advocate_prompt = f"""You are the Advocate in a structured career pivot debate.
Your job: make the STRONGEST POSSIBLE CASE for why this pivot is a good idea.
Be specific, evidence-based, and compelling. Do NOT be balanced — you are arguing FOR.

EVIDENCE BASE:
{context}

Respond ONLY with valid JSON:
{{
  "main_argument": "Your core thesis in 2-3 sentences — the strongest reason this pivot makes sense",
  "strongest_evidence": [
    "3-4 specific evidence points that support the pivot (cite actual data from context)"
  ],
  "key_risks_acknowledged": [
    "1-2 risks you admit exist but argue are manageable or overstated"
  ],
  "closing_statement": "1 sentence — the single most compelling reason to proceed"
}}"""

    # ── Agent 2: Skeptic ──────────────────────────────────────────────────────
    skeptic_prompt = f"""You are the Skeptic in a structured career pivot debate.
Your job: make the STRONGEST POSSIBLE CASE against this pivot.
Be rigorous, evidence-based, and critical. Do NOT be balanced — you are arguing AGAINST.

EVIDENCE BASE:
{context}

Respond ONLY with valid JSON:
{{
  "main_argument": "Your core thesis in 2-3 sentences — the strongest reason this pivot is risky or unlikely to succeed",
  "strongest_evidence": [
    "3-4 specific evidence points that challenge the pivot (cite actual data from context)"
  ],
  "key_risks_acknowledged": [
    "1-2 strengths you admit exist but argue are insufficient"
  ],
  "closing_statement": "1 sentence — the single most important reason to reconsider"
}}"""

    # Run advocate and skeptic in parallel
    try:
        import concurrent.futures

        def call_agent(prompt: str, agent_model: str) -> Dict:
            resp = client.chat.completions.create(
                model=agent_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=700,
            )
            return json.loads(resp.choices[0].message.content or "{}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_adv = ex.submit(call_agent, advocate_prompt, model_debate)
            f_skp = ex.submit(call_agent, skeptic_prompt, model_debate)
            adv_raw = f_adv.result()
            skp_raw = f_skp.result()

    except Exception as e:
        return {**_offline_debate(current_role, target_role, match_score), "error": repr(e)}

    advocate = DebateRound(
        side="advocate",
        main_argument=adv_raw.get("main_argument", ""),
        strongest_evidence=adv_raw.get("strongest_evidence", []),
        key_risks_acknowledged=adv_raw.get("key_risks_acknowledged", []),
        closing_statement=adv_raw.get("closing_statement", ""),
    )
    skeptic = DebateRound(
        side="skeptic",
        main_argument=skp_raw.get("main_argument", ""),
        strongest_evidence=skp_raw.get("strongest_evidence", []),
        key_risks_acknowledged=skp_raw.get("key_risks_acknowledged", []),
        closing_statement=skp_raw.get("closing_statement", ""),
    )

    # ── Agent 3: Judge ────────────────────────────────────────────────────────
    judge_prompt = f"""You are the Judge in a structured career pivot debate.
You have read arguments from both an Advocate and a Skeptic. Your job:
synthesise the evidence, address the strongest argument on EACH side, and deliver
a calibrated verdict.

EVIDENCE BASE:
{context}

ADVOCATE'S ARGUMENT:
{json.dumps(adv_raw, indent=2)}

SKEPTIC'S ARGUMENT:
{json.dumps(skp_raw, indent=2)}

Deliver your verdict. Be specific — cite the actual arguments you are accepting or rejecting.

Respond ONLY with valid JSON:
{{
  "pivot_viability_pct": 65,
  "verdict_label": "Viable with Conditions",
  "decisive_factor": "The single most important factor that tipped your decision",
  "strongest_pro_argument": "The advocate's argument you found most convincing, and why",
  "strongest_con_argument": "The skeptic's argument you found most compelling, and why",
  "judge_reasoning": "3-4 sentences: how you weighed both sides and reached the verdict",
  "conditions_for_success": ["3 specific conditions under which the pivot succeeds"],
  "conditions_for_failure": ["2-3 specific conditions under which it fails"],
  "recommended_next_action": "The single most important concrete action the candidate should take right now"
}}

verdict_label options: "Strong Case" (>75%) | "Viable with Conditions" (50-75%) | "Weak Case" (30-50%) | "Not Recommended" (<30%)"""

    try:
        judge_resp = client.chat.completions.create(
            model=model_judge,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=900,
        )
        judge_raw = json.loads(judge_resp.choices[0].message.content or "{}")
    except Exception as e:
        judge_raw = {
            "pivot_viability_pct": 50,
            "verdict_label": "Viable with Conditions",
            "decisive_factor": "Analysis incomplete due to API error.",
            "strongest_pro_argument": advocate.main_argument,
            "strongest_con_argument": skeptic.main_argument,
            "judge_reasoning": f"Judge API call failed: {repr(e)}",
            "conditions_for_success": [],
            "conditions_for_failure": [],
            "recommended_next_action": "",
        }

    verdict = DebateVerdict(
        pivot_viability_pct=int(judge_raw.get("pivot_viability_pct", 50)),
        verdict_label=str(judge_raw.get("verdict_label", "Viable with Conditions")),
        decisive_factor=str(judge_raw.get("decisive_factor", "")),
        strongest_pro_argument=str(judge_raw.get("strongest_pro_argument", "")),
        strongest_con_argument=str(judge_raw.get("strongest_con_argument", "")),
        judge_reasoning=str(judge_raw.get("judge_reasoning", "")),
        conditions_for_success=list(judge_raw.get("conditions_for_success", [])),
        conditions_for_failure=list(judge_raw.get("conditions_for_failure", [])),
        recommended_next_action=str(judge_raw.get("recommended_next_action", "")),
        source=f"online (advocate+skeptic: {model_debate}, judge: {model_judge})",
    )

    return {
        "advocate": advocate,
        "skeptic": skeptic,
        "verdict": verdict,
        "source": "online",
        "model_debate": model_debate,
        "model_judge": model_judge,
    }


def _offline_debate(
    current_role: str,
    target_role: str,
    match_score: float,
) -> Dict[str, Any]:
    viability = 65 if match_score >= 60 else (45 if match_score >= 40 else 30)
    return {
        "advocate": DebateRound(
            side="advocate",
            main_argument=f"The skill overlap ({match_score:.0f}/100) provides a real foundation for this pivot.",
            strongest_evidence=["Existing transferable skills reduce learning time", "Career pivots with >50% skill overlap have higher success rates"],
            key_risks_acknowledged=["Gaps exist but are addressable through targeted upskilling"],
            closing_statement="The foundation is there — execution determines the outcome.",
        ),
        "skeptic": DebateRound(
            side="skeptic",
            main_argument=f"A match score of {match_score:.0f}/100 means significant gaps that take time and effort to close.",
            strongest_evidence=["Skill gaps require real investment before market credibility is achievable", "Hiring managers prefer candidates with direct experience"],
            key_risks_acknowledged=["Some transferable skills exist but may not be sufficient alone"],
            closing_statement="The risk is real — preparation is non-negotiable.",
        ),
        "verdict": DebateVerdict(
            pivot_viability_pct=viability,
            verdict_label="Viable with Conditions" if viability >= 50 else "Weak Case",
            decisive_factor="Skill match score and gap size",
            strongest_pro_argument="Real transferable foundation exists",
            strongest_con_argument="Gaps require deliberate investment",
            judge_reasoning=f"Offline mode: verdict based on match score {match_score:.0f}/100.",
            conditions_for_success=["Close top 3 skill gaps", "Build portfolio evidence", "Network in target field"],
            conditions_for_failure=["No upskilling investment", "Applying without preparation"],
            recommended_next_action="Start with the skill investment simulator to prioritise development.",
            source="offline",
        ),
        "source": "offline",
    }
