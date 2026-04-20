"""
Negotiation Coach
=================
The phase every other career tool ignores: after "get the interview" comes
"capture the maximum value from the offer."

Studies show candidates who negotiate earn $5,000-$20,000 more annually —
yet 60% never ask. This module provides:

1. Market salary analysis — where your offer sits vs. the real market
2. Offer intelligence — equity, benefits, total comp breakdown
3. Personalized negotiation script — specific lines, not generic tips
4. Live roleplay — practice against a realistic HR director (gpt-4o)
5. Counter-offer letter — professional, specific, ready to send

Architecture note: gpt-4o for roleplay (needs sustained persona + context);
gpt-4o-mini for analysis (structured JSON extraction, lower cost).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market Salary Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_salary_offer(
    job_title: str,
    company: str = "",
    location: str = "",
    offered_salary: Optional[float] = None,
    offered_equity: str = "",
    offered_benefits: str = "",
    years_experience: float = 0.0,
    current_role: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      market_salary_low/mid/high, offer_quality, negotiation_room,
      key_leverage_points, risk_factors, one_line_verdict
    """
    _fallback: Dict[str, Any] = {
        "market_salary_low": 0,
        "market_salary_mid": 0,
        "market_salary_high": 0,
        "market_percentile": "unknown",
        "offer_quality": "unknown",
        "equity_assessment": "Add API key for full offer analysis.",
        "benefits_assessment": "",
        "negotiation_room": 0,
        "negotiation_room_pct": 0,
        "key_leverage_points": ["Research market rates on Glassdoor, Levels.fyi and LinkedIn Salary before negotiating."],
        "risk_factors": [],
        "total_comp_estimate": 0,
        "one_line_verdict": "Provide offer details and API key for analysis.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    prompt = f"""You are a compensation expert with deep market knowledge. Analyse this job offer with precision.

ROLE: {job_title}
COMPANY: {company or "not specified (assume well-funded company)"}
LOCATION: {location or "United States — assume major tech hub (SF/NYC/Seattle)"}
EXPERIENCE: {years_experience:.0f} years — coming from: {current_role or "not specified"}

OFFER:
- Base salary: {"$" + str(int(offered_salary)) if offered_salary else "NOT DISCLOSED — estimate based on role"}
- Equity: {offered_equity or "not provided"}
- Benefits: {offered_benefits or "not provided"}

Using your knowledge of current (2024-2025) compensation market data:
1. Provide the realistic P25/P50/P75 salary range for this exact role/location/seniority.
2. Assess the offer quality relative to market.
3. Estimate realistically how much more is negotiable (specific number, not a range).
4. Give 3-4 strong leverage points the candidate can use.
5. Flag any financial or contractual risks in this offer structure.

Be concrete. Use real numbers. Do not hedge with "it varies" — give your best estimate.

Respond ONLY with valid JSON:
{{
  "market_salary_low": 95000,
  "market_salary_mid": 120000,
  "market_salary_high": 148000,
  "market_percentile": "below",
  "offer_quality": "fair",
  "equity_assessment": "RSUs worth ~$40k/year vesting — typical for Series B",
  "benefits_assessment": "401k match is below market; unlimited PTO often reduces actual time-off",
  "negotiation_room": 18000,
  "negotiation_room_pct": 15,
  "total_comp_estimate": 145000,
  "key_leverage_points": [
    "Market P50 for this role in NYC is $120k — offer is 12% below that",
    "Your 6 years of product experience reduces their onboarding cost significantly",
    "Mentioning an alternate process in late stages is legitimate and effective"
  ],
  "risk_factors": [
    "2-year equity cliff — leaving before 24 months means zero unvested equity",
    "No guaranteed bonus — variable comp is at manager discretion"
  ],
  "one_line_verdict": "12% below market median. Counter at $132k base. High probability of success."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        for k in ["market_salary_low", "market_salary_mid", "market_salary_high", "negotiation_room"]:
            if k not in result:
                result[k] = 0
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Negotiation Script Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_negotiation_script(
    job_title: str,
    company: str = "",
    offered_salary: Optional[float] = None,
    target_salary: Optional[float] = None,
    market_salary_mid: Optional[int] = None,
    key_leverage_points: Optional[List[str]] = None,
    current_role: str = "",
    years_experience: float = 0.0,
    top_skills: Optional[List[str]] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Returns a complete, personalized negotiation playbook:
    opening, salary ask line, 3 objection responses, email + phone versions.
    """
    _fallback: Dict[str, Any] = {
        "opening_statement": "I'm very excited about this role. I do want to discuss the compensation package.",
        "salary_ask_line": "Based on my market research, I'd like to propose a base salary of [X].",
        "justification_points": ["Add API key for personalized negotiation script."],
        "objection_responses": [],
        "closing_line": "I'm confident we can find a number that works for both of us.",
        "email_version": "",
        "phone_script": "",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    _target = target_salary or (offered_salary * 1.12 if offered_salary else None)
    _mkt = f"${market_salary_mid:,}" if market_salary_mid else "above offer"
    _leverage = "\n".join([f"- {p}" for p in (key_leverage_points or [])]) or "- Market data supports higher compensation"
    _skills = ", ".join(top_skills[:5]) if top_skills else "relevant technical and domain skills"

    prompt = f"""Write a specific, confident, non-generic salary negotiation playbook. This must feel like real advice from a top recruiter, not a template.

SITUATION:
- Role: {job_title} at {company}
- Offered: {"$" + str(int(offered_salary)) if offered_salary else "not yet stated"}
- Target: {"$" + str(int(_target)) if _target else "10-15% above offer"}
- Market P50: {_mkt}
- Candidate: {years_experience:.0f} years exp, from {current_role or "previous role"}, strengths: {_skills}

LEVERAGE AVAILABLE:
{_leverage}

Write a playbook where every line is specific, uses real numbers, and sounds like a confident professional (not an AI template).

Respond ONLY with valid JSON:
{{
  "opening_statement": "I want to start by saying I'm genuinely excited — the scope of the PM role here is exactly what I've been building toward. That said, I do want to talk through the comp before we go further.",
  "salary_ask_line": "Based on current market data — Glassdoor and LinkedIn Salary put the P50 for senior PMs in San Francisco at $148k — I'd like to propose a base of $152k.",
  "justification_points": [
    "Market median is $148k for this title/location — I'm not asking above market",
    "My track record of shipping 0-to-1 products means minimal ramp time — I can contribute from week one",
    "I have a second-round process running — not using it as leverage, but it confirms my market value"
  ],
  "objection_responses": [
    {{
      "objection": "That's outside our salary band for this level",
      "response": "I understand. Can you share the top of the band? If base has a ceiling, I'd be open to discussing a signing bonus to close the gap, or a 6-month review with a target number baked in."
    }},
    {{
      "objection": "We treat all candidates at this level equally",
      "response": "That's fair — I'm not asking for special treatment. I'm asking for a number that reflects what the market pays for this skill set in this city. The data puts that at $148k. Happy to share the sources."
    }},
    {{
      "objection": "We can revisit this at your performance review",
      "response": "I'm open to that — can we put a target in writing now? I'd feel more comfortable committing if there's a documented path: $X at 6 months if I hit [specific milestone]. That protects both of us."
    }}
  ],
  "closing_line": "I want this to be easy for you to say yes to. Let me know what room you have — I'm flexible on structure even if the total number matters.",
  "email_version": "Subject: Re: Offer — {job_title}\\n\\nHi [Name],\\n\\nThank you for the formal offer...",
  "phone_script": "Call structure: (1) Open with genuine excitement, 30 seconds. (2) Pivot: 'I do want to discuss one thing...' (3) Market data anchor. (4) The ask. (5) Silence — do not fill it. (6) Handle objection. (7) Close with commitment."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1400,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Negotiation Roleplay (multi-turn)
# ─────────────────────────────────────────────────────────────────────────────

def run_negotiation_roleplay_turn(
    history: List[Dict[str, str]],
    job_title: str,
    company: str = "",
    offered_salary: Optional[float] = None,
    exchange_num: int = 1,
    max_exchanges: int = 5,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    AI plays HR director. User practices the real negotiation.
    Returns {response, coaching_tip, is_complete, outcome}
    """
    _fallback = {
        "response": "I appreciate you bringing this up. Let me see what flexibility we have.",
        "coaching_tip": "",
        "is_complete": False,
        "outcome": None,
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    is_final = exchange_num >= max_exchanges
    _offered = f"${int(offered_salary):,}" if offered_salary else "the stated offer"

    system = f"""You are the HR Director at {company or "a leading technology company"} negotiating compensation for a {job_title} hire.

CONTEXT:
- Initial offer: {_offered}
- You genuinely want to hire this candidate
- Your budget has ~8-10% flex above initial offer
- You can also offer: signing bonus, equity top-up, earlier performance review, extra PTO
- You respond well to market data; poorly to emotional arguments
- You are professional, measured, slightly time-pressured

RULES:
- Keep responses under 3 sentences. Be realistic.
- If candidate makes a strong, data-backed ask: acknowledge it, show some movement
- If candidate is vague or emotional: hold position, ask for specifics
- This is exchange {exchange_num} of {max_exchanges}
{"- This is the final exchange. Bring it to a resolution: accept their counter, split the difference, or hold firm with a clear final number. Include a coaching tip in square brackets at the end like [Coaching: ...]" if is_final else ""}"""

    messages = [{"role": "system", "content": system}] + history

    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.65,
            max_tokens=250,
        )
        content = r.choices[0].message.content or ""

        coaching_tip = ""
        outcome = None
        if "[Coaching:" in content:
            parts = content.split("[Coaching:")
            content = parts[0].strip()
            coaching_tip = parts[1].replace("]", "").strip() if len(parts) > 1 else ""

        return {
            "response": content,
            "coaching_tip": coaching_tip,
            "is_complete": is_final,
            "outcome": outcome,
            "source": "online",
        }
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Counter-Offer Letter Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_counter_offer_letter(
    job_title: str,
    company: str = "",
    hiring_manager: str = "",
    offered_salary: Optional[float] = None,
    counter_salary: Optional[float] = None,
    justification_points: Optional[List[str]] = None,
    candidate_name: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> str:
    """Returns a ready-to-send counter-offer letter as plain text."""
    _target = f"${int(counter_salary):,}" if counter_salary else "[your target]"
    _offered_str = f"${int(offered_salary):,}" if offered_salary else "the initial offer"
    _just = "\n".join([f"- {p}" for p in (justification_points or [])]) or "- Strong market demand for this skill set"

    if not prefer_online:
        return f"""Dear {hiring_manager or "Hiring Team"},

Thank you for extending an offer for the {job_title} position at {company}. I am genuinely excited about the opportunity and the team.

After careful consideration and reviewing current market compensation data for this role, I would like to respectfully propose a base salary of {_target}, compared to the initial offer of {_offered_str}.

{_just}

I remain very interested in joining {company or "your team"} and am confident we can reach an agreement. I'm also open to discussing structure — if base flexibility is limited, I'm happy to explore alternatives such as a signing bonus or equity adjustment.

Looking forward to your response.

Best regards,
{candidate_name or "[Your Name]"}"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return "[OpenAI unavailable — add API key for letter generation]"

    prompt = f"""Write a professional counter-offer letter. It must feel human and confident, not like a template.

FACTS:
- Candidate: {candidate_name or "the candidate"}
- Role: {job_title} at {company}
- To: {hiring_manager or "the hiring team"}
- Offer received: {_offered_str}
- Counter-ask: {_target}
- Justification:
{_just}

RULES:
- Under 180 words
- Specific about the number
- Grateful but not sycophantic
- No clichés ("passionate", "dream job", "unique opportunity")
- Leave room to negotiate further
- Professional subject line included

Return ONLY the letter text (including subject line), no JSON."""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return r.choices[0].message.content or "[Error generating letter]"
    except Exception:
        return "[Error generating letter — try again]"
