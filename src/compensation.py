"""
Compensation Intelligence Engine
==================================
The highest-ROI feature in the product: salary negotiation for career changers.

Career changers have a structural disadvantage in salary negotiations:
  - They lack the "I have competing offers" leverage of passive candidates
  - They often accept initial offers out of gratitude for the pivot opportunity
  - They don't know if the offer is calibrated to their *current* role or *target* role
  - They fear pushing back will cost them the offer

This module provides:
  1. Market range analysis for the target role (GPT-calibrated from known data)
  2. Offer evaluation: under/fair/above market
  3. Negotiation strategy specific to career changers (different from senior ICs)
  4. Three negotiation scripts: direct ask / anchor high / alternatives to cash
  5. Counter-offer analysis: the "my employer is keeping me" scenario
  6. Walk-away threshold and BATNA (Best Alternative To Negotiated Agreement)

Why career-changer negotiation is different:
  - You negotiate from "gratitude" not "leverage" — must flip this
  - Your current salary is often used as an anchor (wrong frame — resist it)
  - Your "comparable offers" are from a different role family (must translate)
  - Signing bonus is often easier to get than base for career switchers
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_COMP_SYSTEM = """You are a compensation specialist who has helped 500+ career changers negotiate offers.
You know the specific dynamics of pivot negotiation: candidates feel grateful, fear losing the offer,
and don't know their market value in the new role.

Your job: give a brutal, honest compensation analysis with specific negotiation scripts.

Rules:
- Give real numbers, not ranges so wide they're useless
- Be specific about career-changer dynamics (not generic advice)
- The negotiation scripts must be word-for-word usable, not outlines
- Address the fear of losing the offer directly

Output JSON only:
{
  "market_low": <int, annual EUR/USD>,
  "market_mid": <int>,
  "market_high": <int>,
  "offer_assessment": "below_market|fair|above_market",
  "offer_assessment_reason": "one specific sentence",
  "negotiation_headroom": <int, likely additional amount achievable>,
  "pivot_premium_risk": "low|medium|high",
  "pivot_premium_explanation": "why the company might resist vs. accept a push",

  "scripts": {
    "direct_ask": "Word-for-word script for asking for more base salary",
    "anchor_high": "Script for anchoring to a higher number first",
    "alternatives": "Script for when base is firm — pivot to signing bonus/equity/vacation"
  },

  "fear_reframe": "One paragraph reframing the 'they'll rescind the offer' fear with data",
  "walk_away_threshold": <int, minimum you should accept>,
  "walk_away_reasoning": "why below this is not worth taking",

  "batna_assessment": "What happens if you walk away — is it actually as bad as it feels?",
  "timing_advice": "When in the conversation to ask, what not to say first",
  "three_asks_ranked": ["First choice (most likely to get)", "Second", "Third"],

  "counter_offer_context": "If this is a counter from current employer — one paragraph on whether to take it"
}"""


def analyze_offer(
    oai_key: str,
    *,
    offered_amount: int,
    role_title: str,
    company: str,
    location: str,
    company_stage: str,
    years_experience: int,
    current_salary: Optional[int] = None,
    pivot_dna: Optional[Dict] = None,
    is_counter_offer: bool = False,
    currency: str = "EUR",
) -> Optional[Dict[str, Any]]:
    """
    Full compensation analysis for a specific offer.

    Returns negotiation strategy dict or None on failure.
    """
    if not oai_key or not offered_amount or not role_title:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    context = {
        "offered_amount":    offered_amount,
        "currency":          currency,
        "role_title":        role_title,
        "company":           company,
        "location":          location,
        "company_stage":     company_stage,
        "years_experience":  years_experience,
        "current_salary":    current_salary,
        "is_pivot":          True,
        "is_counter_offer":  is_counter_offer,
        "pivot_hook":        (pivot_dna or {}).get("pivot_hook", ""),
        "target_company_type": (pivot_dna or {}).get("target_company_type", ""),
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": _COMP_SYSTEM},
                {"role": "user", "content": (
                    f"Offer to analyze:\n{json.dumps(context, indent=2)}\n\n"
                    f"Candidate context: career changer into {role_title}. "
                    f"They have {years_experience} years of experience in a different field. "
                    f"They need specific, usable scripts — not generic advice."
                )},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["offered_amount"] = offered_amount
        result["currency"]       = currency
        result["role_title"]     = role_title
        result["company"]        = company
        return result
    except Exception:
        return None


def offer_assessment_color(assessment: str) -> str:
    return {
        "below_market": "#DC2626",
        "fair":         "#D97706",
        "above_market": "#057642",
    }.get(assessment, "#555")


def offer_assessment_label(assessment: str) -> str:
    return {
        "below_market": "Below market — push back",
        "fair":         "Fair market — room to improve",
        "above_market": "Above market — accept or minor push",
    }.get(assessment, assessment)
