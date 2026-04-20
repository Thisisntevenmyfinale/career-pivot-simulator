"""
Market Pulse — Job Market Timing Intelligence
=============================================
Answers the question every job seeker has but never gets a straight answer to:
"Is now a good time to be applying for THIS role in THIS sector?"

Uses LLM knowledge (+ optional SerpAPI for real-time data) to generate:
- Hiring velocity signal for the target role/sector
- Sector-level health (growing / stable / contracting)
- Best timing windows (e.g., Jan-Feb and Sep-Oct are peak hiring months)
- Top companies currently hiring in the space
- Salary momentum (trending up / flat / down)
- 3 specific companies to prioritize right now

This is NOT a generic "tech hiring is slow" report.
It's specific to the candidate's target role and sector.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def get_market_pulse(
    target_role: str,
    sector: str = "",
    location: str = "",
    current_month: int = 4,   # April
    current_year: int = 2025,
    serp_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate market timing intelligence for a specific target role.

    Returns:
      {
        "hiring_velocity": str,       # "accelerating" | "steady" | "decelerating" | "frozen"
        "velocity_score": int,        # 0-100: 0=no hiring, 100=massive hiring wave
        "sector_health": str,         # "growing" | "stable" | "contracting"
        "timing_verdict": str,        # "Apply now" | "Good timing" | "Neutral" | "Wait if you can"
        "timing_score": int,          # 0-100: how good is RIGHT NOW for applying
        "peak_months": List[str],     # historically best months to apply
        "current_month_context": str, # what does applying in THIS month mean?
        "top_companies_hiring": List[str],  # companies actively growing this function
        "salary_momentum": str,       # "rising" | "flat" | "declining"
        "salary_context": str,        # why + what to expect
        "demand_drivers": List[str],  # what's creating demand for this role
        "risk_factors": List[str],    # what could suppress hiring
        "sourcing_tip": str,          # where to find these roles (not just "LinkedIn")
        "one_line_verdict": str,      # is now a good time, and what should they do?
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "hiring_velocity": "unknown",
        "velocity_score": 50,
        "sector_health": "unknown",
        "timing_verdict": "Neutral",
        "timing_score": 50,
        "peak_months": ["January", "February", "September", "October"],
        "current_month_context": "Add API key for market timing intelligence.",
        "top_companies_hiring": [],
        "salary_momentum": "unknown",
        "salary_context": "",
        "demand_drivers": [],
        "risk_factors": [],
        "sourcing_tip": "LinkedIn, Indeed, and company careers pages are standard.",
        "one_line_verdict": "Add OpenAI API key for market pulse analysis.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    # Optional: fetch real-time hiring news via SerpAPI
    news_context = ""
    if serp_api_key:
        try:
            import requests
            queries = [
                f"{target_role} jobs hiring {current_year}",
                f"{sector or target_role} industry hiring trends layoffs {current_year}",
            ]
            snippets = []
            for q in queries[:1]:  # limit to 1 query to avoid rate limit
                params = {
                    "engine": "google",
                    "q": q,
                    "api_key": serp_api_key,
                    "num": 5,
                }
                resp = requests.get("https://serpapi.com/search", params=params, timeout=8)
                if resp.ok:
                    data = resp.json()
                    for r in data.get("organic_results", [])[:4]:
                        if r.get("snippet"):
                            snippets.append(r["snippet"])
            if snippets:
                news_context = "RECENT SEARCH DATA:\n" + "\n".join([f"- {s}" for s in snippets])
        except Exception:
            pass

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_str = month_names[current_month - 1] if 1 <= current_month <= 12 else "April"

    prompt = f"""You are a labor market analyst specializing in tech and knowledge worker hiring trends.

TARGET ROLE: {target_role}
SECTOR: {sector or "not specified (infer from role)"}
LOCATION: {location or "global / remote-friendly"}
CURRENT DATE: {month_str} {current_year}
{news_context}

Draw on everything you know about hiring cycles, sector health, talent supply/demand, and recent industry events.

Be specific. Name real companies. Give real seasonality data. Quantify where you can.
Flag knowledge limits honestly — don't fabricate recent events you don't know.

Respond ONLY with valid JSON:
{{
  "hiring_velocity": "steady",
  "velocity_score": 62,
  "sector_health": "growing",
  "timing_verdict": "Good timing",
  "timing_score": 70,
  "peak_months": ["January", "February", "September", "October"],
  "current_month_context": "April is a solid month — Q2 hiring budgets are open and companies that didn't fill Q1 roles are under pressure to hire now.",
  "top_companies_hiring": [
    "Stripe (scaling Risk & Compliance function post-IPO prep)",
    "Anthropic (growing product team aggressively)",
    "Figma (rebuilding after Adobe deal collapse — active hiring)",
    "Linear (Series C growth stage — product expansion)"
  ],
  "salary_momentum": "rising",
  "salary_context": "AI-adjacent product roles are seeing 15-25% salary inflation vs. 2022 benchmarks. Senior IC comp is moving faster than management comp.",
  "demand_drivers": [
    "AI product development expanding total addressable demand for product managers",
    "Enterprise SaaS consolidation driving demand for product leaders who can own full-stack roadmaps",
    "Post-2023 layoffs created a talent vacuum at mid-senior levels now being refilled"
  ],
  "risk_factors": [
    "VC funding slowdown has extended hiring timelines at Series A/B companies",
    "Large tech (FAANG) remains in efficiency mode — fewer new IC roles than 2021-22"
  ],
  "sourcing_tip": "The best product roles are sourced through warm intros and YC job board (workatastartup.com) — not LinkedIn. Join Lenny's Slack and Product-Led Growth Slack for community sourcing.",
  "one_line_verdict": "Solid time to apply. Mid-market and growth-stage companies are the primary buyer right now — avoid FAANG and focus energy on Series B-D companies with recent funding."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online" + (" + SerpAPI" if news_context else " (LLM knowledge)")
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


def generate_warm_intro_sequence(
    target_company: str,
    target_role: str,
    your_name: str = "",
    your_background: str = "",
    mutual_connection: str = "",
    connection_type: str = "LinkedIn 2nd-degree",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Network Activation Engine — generates a 3-touch outreach sequence.

    Cold outreach has a <3% response rate.
    A warm intro through a mutual connection has ~35-50% response rate.
    This generates the messages to make that happen.

    Returns:
      {
        "touch_1_subject": str,       # email/LinkedIn subject line
        "touch_1_message": str,       # first outreach (connection request note or email)
        "touch_2_message": str,       # follow-up if no response after 5-7 days
        "touch_3_message": str,       # final touch (referral ask or informational interview)
        "strategy_note": str,         # what you're trying to accomplish with this sequence
        "connection_angle": str,      # the specific angle to use with this connection type
        "dos": List[str],
        "donts": List[str],
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "touch_1_subject": f"Quick intro — interested in {target_role} at {target_company}",
        "touch_1_message": "Add API key for personalized outreach sequence.",
        "touch_2_message": "",
        "touch_3_message": "",
        "strategy_note": "Add OpenAI API key for warm intro sequence generation.",
        "connection_angle": "",
        "dos": ["Be specific about why you're reaching out", "Reference something relevant about their work"],
        "donts": ["Don't ask for a job in the first message", "Don't send walls of text"],
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    prompt = f"""You are a networking strategist. Generate a 3-touch outreach sequence to activate a connection toward a warm intro.

TARGET: {target_role} at {target_company}
YOUR NAME: {your_name or "the candidate"}
YOUR BACKGROUND: {your_background or "career pivoter with relevant transferable skills"}
CONNECTION: {mutual_connection or "a 2nd-degree LinkedIn connection"} ({connection_type})

Rules:
- Touch 1: Short connection request or first email. Under 75 words. No job ask. Build genuine connection.
- Touch 2: Follow-up after 5-7 days of silence. Under 60 words. Light, not pushy.
- Touch 3: The actual ask — informational interview OR referral. Under 80 words. Specific, low-friction ask.
- Each message must feel human, not templated. Reference specifics.
- The goal is an INTRO or a 15-minute call — not a job offer.

Respond ONLY with valid JSON:
{{
  "touch_1_subject": "Quick question from a fellow [mutual interest]",
  "touch_1_message": "Hi [Name], I noticed we're both connected to [Mutual] and that you work on the product team at {target_company}. I'm researching the {target_role} space seriously right now — your work on [specific thing they did] caught my attention. Would love to connect briefly if you're open to it. No agenda beyond a genuine conversation.",
  "touch_2_message": "Hey [Name] — circling back on my earlier note. I know inboxes get buried. If a 15-minute call isn't the right fit, I'd genuinely appreciate any reading/resources you'd point someone at who's serious about breaking into this area.",
  "touch_3_message": "Hi [Name], I'm actively interviewing for {target_role} roles and {target_company} is my top target. If you'd be comfortable passing my name to the hiring team — or even just letting me know if there's a better internal contact — that would mean a lot. I've done my homework on the company and think there's a real fit. Happy to share my profile.",
  "strategy_note": "The sequence moves from genuine relationship-building (touch 1) → low-pressure follow-up (touch 2) → a specific, easy-to-fulfill ask (touch 3). Never ask for a job directly — ask for a conversation or intro.",
  "connection_angle": "Lead with the mutual connection as social proof, then pivot to genuine curiosity about their work before making any ask.",
  "dos": [
    "Reference something specific about their career or work — shows you did homework",
    "Keep every message under 100 words — respects their time",
    "Make the ask specific and easy to say yes or no to"
  ],
  "donts": [
    "Don't attach your CV in touch 1 — it signals you're treating them as a transaction",
    "Don't follow up more than 3 times total",
    "Don't ask for a 30-minute call — 15 minutes has 3× the acceptance rate"
  ]
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=700,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}
