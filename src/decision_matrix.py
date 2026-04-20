"""
Decision Matrix — The Hardest Pivot Decisions
===============================================
Three specific scenarios where career changers need structured help:

  A. Two offers simultaneously — which to take?
  B. Bridge role vs. direct PM application — what's the expected value?
  C. Counter-offer from current employer — stay or go?

These are the highest-stakes decisions in any pivot. Most tools ignore them.
Generic advice ("follow your gut", "think about long-term growth") is useless.

This module provides:
  - Structured multi-factor analysis for each scenario
  - Expected value calculation where possible (using cohort + OPS data)
  - Specific recommendation with reasoning — not a "it depends" hedge
  - The one question to ask yourself that cuts through the noise

Design principle: the system gives a recommendation. Not "here are the factors."
Career changers in decision paralysis need a clear directional answer, not
more information to process. The reasoning is transparent, the call is clear.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A: Two Offers
# ─────────────────────────────────────────────────────────────────────────────

_TWO_OFFERS_SYSTEM = """You are a career strategist for career changers.
Two offers are on the table. Give a clear recommendation — not "it depends."

Analyze on: compensation, growth trajectory, learning speed, company trajectory,
pivot credibility (does this role cement the pivot or leave it ambiguous?),
cultural risk for a career changer, negotiation leverage from the second offer.

Output JSON only:
{
  "recommendation": "offer_a|offer_b",
  "confidence": "high|medium|low",
  "decision_reason": "2-3 sentence clear explanation of the call",
  "the_one_question": "The single question that cuts through the noise for this specific choice",
  "offer_a_score": <int 0-100>,
  "offer_b_score": <int 0-100>,
  "dimensions": [
    {"name": "dimension name", "offer_a": <int 0-10>, "offer_b": <int 0-10>, "weight": "high|medium|low", "note": "why this matters here"}
  ],
  "negotiation_angle": "Can you use offer B to improve offer A, or vice versa? Exactly how.",
  "hidden_risk_a": "The non-obvious risk of taking offer A",
  "hidden_risk_b": "The non-obvious risk of taking offer B",
  "timeline_advice": "How long to take to decide, and what to do during that time"
}"""


def compare_offers(
    oai_key: str,
    *,
    offer_a: Dict,
    offer_b: Dict,
    pivot_dna: Optional[Dict] = None,
    cv_profile: Optional[Dict] = None,
    ops_score: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Compare two job offers. offer_a/b: {company, title, salary, stage, notes}"""
    if not oai_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _TWO_OFFERS_SYSTEM},
                {"role": "user", "content": (
                    f"Offer A: {json.dumps(offer_a)}\n"
                    f"Offer B: {json.dumps(offer_b)}\n"
                    f"Candidate pivot context: {json.dumps({'hook': (pivot_dna or {}).get('pivot_hook',''), 'target_type': (pivot_dna or {}).get('target_company_type',''), 'years_exp': (cv_profile or {}).get('years_experience','?'), 'ops': ops_score})}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B: Bridge Role vs. Direct Application
# ─────────────────────────────────────────────────────────────────────────────

_BRIDGE_SYSTEM = """You are a career strategist specialising in career pivots.
The candidate is deciding: take a bridge role (adjacent, not exact target) or apply directly to target roles now.

This is a genuine strategic choice — bridge roles have real costs (time, opportunity cost) AND real benefits
(PM title, reduced rejection rate). Give a clear quantified recommendation.

Output JSON only:
{
  "recommendation": "bridge|direct|hybrid",
  "confidence": "high|medium|low",
  "decision_reason": "2-3 sentences. Be specific about this candidate's situation.",
  "expected_value_bridge": "estimated weeks to first PM offer if bridge route taken",
  "expected_value_direct": "estimated weeks to first PM offer if direct route taken",
  "bridge_cost": "what the candidate gives up by taking the bridge route",
  "bridge_benefit": "what the candidate gains that justifies the cost",
  "direct_risk": "the main risk of going direct now — why it might fail",
  "hybrid_approach": "if recommendation is hybrid: exactly what to do in parallel",
  "trigger_point": "If direct applications, at what point should they reconsider and pivot to bridge route?",
  "the_one_question": "The single question that clarifies this decision"
}"""


def bridge_vs_direct(
    oai_key: str,
    *,
    bridge_role: str,
    target_role: str,
    current_ops: int,
    current_applications: int,
    cohort_median_apps: int,
    skill_gaps: List[str],
    pivot_dna: Optional[Dict] = None,
    bridge_occupations: Optional[List[Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze bridge role vs. direct application strategy."""
    if not oai_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _BRIDGE_SYSTEM},
                {"role": "user", "content": (
                    f"Bridge role available: {bridge_role}\n"
                    f"Direct target: {target_role}\n"
                    f"Current OPS: {current_ops}/100\n"
                    f"Applications sent: {current_applications} (cohort median: {cohort_median_apps})\n"
                    f"Open skill gaps: {skill_gaps[:5]}\n"
                    f"Identified bridge occupations from skill space: {[b.get('occupation','') for b in (bridge_occupations or [])[:3]]}\n"
                    f"Pivot hook: {(pivot_dna or {}).get('pivot_hook','')}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C: Counter-Offer from Current Employer
# ─────────────────────────────────────────────────────────────────────────────

_COUNTER_SYSTEM = """You are a career strategist who has seen hundreds of counter-offer situations.
You know the data: 80% of people who accept counter-offers leave within 12 months anyway.
But you also know: sometimes a counter-offer is genuinely the right call.

Give a specific, data-grounded recommendation. Don't hedge.

Output JSON only:
{
  "recommendation": "accept|decline|negotiate_counter",
  "confidence": "high|medium|low",
  "decision_reason": "2-3 sentences. Specific to this situation.",
  "the_real_question": "The thing the candidate is actually deciding (it's not the money)",
  "counter_offer_traps": ["the 3 psychological traps in counter-offer situations"],
  "if_accept": "What typically happens in the 6-12 months after accepting — specifically",
  "if_decline": "What the candidate is committing to by walking away",
  "negotiation_angle": "If they want to use the counter to improve the external offer instead",
  "pivot_cost_of_staying": "What staying costs the pivot journey specifically",
  "timeline_question": "How long has the pivot search been running — and how does that change the calculus?"
}"""


def analyze_counter_offer(
    oai_key: str,
    *,
    external_offer_amount: int,
    counter_offer_amount: int,
    current_salary: int,
    months_in_search: int,
    current_ops: int,
    pivot_dna: Optional[Dict] = None,
    currency: str = "EUR",
) -> Optional[Dict[str, Any]]:
    """Analyze whether to accept a counter-offer from current employer."""
    if not oai_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _COUNTER_SYSTEM},
                {"role": "user", "content": (
                    f"Current salary: {current_salary} {currency}\n"
                    f"External offer: {external_offer_amount} {currency}\n"
                    f"Counter-offer from current employer: {counter_offer_amount} {currency}\n"
                    f"Months in pivot search: {months_in_search}\n"
                    f"Current OPS score: {current_ops}/100\n"
                    f"Pivot goal: {(pivot_dna or {}).get('pivot_hook','')}\n"
                    f"Pivot risk: {(pivot_dna or {}).get('pivot_risk','')}"
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def recommendation_color(rec: str) -> str:
    mapping = {
        "offer_a": "#0A66C2", "offer_b": "#7C3AED",
        "bridge": "#D97706", "direct": "#057642", "hybrid": "#0A66C2",
        "accept": "#057642", "decline": "#DC2626", "negotiate_counter": "#D97706",
    }
    return mapping.get(rec, "#555")
