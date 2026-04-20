"""
Company Intelligence
====================
You don't apply blind. You apply with intel.

Before sending any application, a smart candidate knows:
- Is this company growing or shrinking?
- What's the culture really like?
- Is the role likely to survive 12 months?
- What do current employees say about the interview process?
- What's the compensation philosophy?
- What signal can I use in my cover letter to stand out?

This module generates a structured Company Intelligence Brief
using LLM knowledge + optional SerpAPI news search.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def generate_company_brief(
    company_name: str,
    role_title: str = "",
    location: str = "",
    serp_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate a structured company intelligence brief.

    Returns:
      {
        "stage": str,              # "startup" | "growth" | "enterprise" | "public"
        "hiring_signal": str,      # "strong" | "moderate" | "weak" | "concerning"
        "stability_score": int,    # 0-100: how safe is this role long-term?
        "culture_snapshot": str,   # 2-3 line culture read
        "glassdoor_read": str,     # what employees say (LLM knowledge)
        "compensation_philosophy": str,
        "interview_process": str,  # what to expect
        "recent_signals": List[str],  # news/events worth knowing
        "red_flags": List[str],
        "green_flags": List[str],
        "cover_letter_hook": str,  # one specific thing to mention in your letter
        "insider_tip": str,        # non-obvious advantage for this company
        "one_line_verdict": str,   # worth applying? why?
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "stage": "unknown",
        "hiring_signal": "unknown",
        "stability_score": 50,
        "culture_snapshot": "Add API key for company intelligence.",
        "glassdoor_read": "",
        "compensation_philosophy": "",
        "interview_process": "",
        "recent_signals": [],
        "red_flags": [],
        "green_flags": [],
        "cover_letter_hook": "",
        "insider_tip": "",
        "one_line_verdict": "Add OpenAI API key for company brief.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    # Optional: fetch recent news via SerpAPI for better recency
    news_context = ""
    if serp_api_key:
        try:
            import requests
            params = {
                "engine": "google",
                "q": f"{company_name} company news hiring layoffs 2024 2025",
                "api_key": serp_api_key,
                "num": 5,
            }
            resp = requests.get("https://serpapi.com/search", params=params, timeout=8)
            if resp.ok:
                data = resp.json()
                snippets = [r.get("snippet", "") for r in data.get("organic_results", [])[:5] if r.get("snippet")]
                if snippets:
                    news_context = "RECENT NEWS:\n" + "\n".join([f"- {s}" for s in snippets])
        except Exception:
            pass

    prompt = f"""You are a company intelligence analyst. Generate a concise, actionable brief for a job candidate applying to this company.

COMPANY: {company_name}
ROLE BEING APPLIED TO: {role_title or "not specified"}
LOCATION: {location or "not specified"}
{news_context}

Draw on everything you know about this company: culture, business model, hiring patterns, employee reviews, compensation philosophy, interview process, recent news, executive leadership, product trajectory.

Be specific. Use real names, real data points, real events where you know them. Flag uncertainty where relevant.
Do NOT make up specific numbers you don't know. Be honest about knowledge limits.

Respond ONLY with valid JSON:
{{
  "stage": "growth",
  "hiring_signal": "strong",
  "stability_score": 78,
  "culture_snapshot": "High performance, low politics. Engineers have real ownership. Manager quality is inconsistent — the product org tends to be stronger than the data org.",
  "glassdoor_read": "4.1 stars. Common themes: fast-paced, smart colleagues, good equity. Negatives: work-life balance variable by team, some middle management gaps.",
  "compensation_philosophy": "Top-of-market for senior ICs; equity is meaningful. Benefits are strong (401k match at 4%, full health). Salary bands are relatively rigid.",
  "interview_process": "4-5 rounds: recruiter screen → hiring manager → take-home case → panel (2 ICs + PM director) → exec. Process takes 3-4 weeks. Case study is known to be demanding.",
  "recent_signals": [
    "Raised $180M Series C in October 2024 — runway extends past 2027",
    "Announced expansion into EU markets Q1 2025 — likely hiring push in product/go-to-market",
    "CEO published vision piece on AI-native product strategy — signals where investment is going"
  ],
  "red_flags": [
    "3 CPOs in 18 months — product strategy instability at the top"
  ],
  "green_flags": [
    "Net revenue retention >120% — core business is healthy",
    "Engineering blog is active and thoughtful — signals a culture that invests in craft"
  ],
  "cover_letter_hook": "Reference their recent EU expansion — frame your experience as relevant to scaling into new markets. Shows you've done your homework.",
  "insider_tip": "Their take-home case is known to weight 'structured thinking' over 'correct answer'. Show your framework explicitly, not just conclusions.",
  "one_line_verdict": "Strong opportunity. Healthy business, growing market, real equity upside. Prepare intensively for the case study — it's the real filter."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online" + (" + SerpAPI news" if news_context else " (LLM knowledge)")
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


def format_brief_as_markdown(brief: Dict[str, Any], company_name: str, role_title: str = "") -> str:
    """Convert a company brief dict to a readable markdown string."""
    _sig_colors = {
        "strong": "🟢", "moderate": "🟡", "weak": "🟠", "concerning": "🔴", "unknown": "⚪"
    }
    _sig = _sig_colors.get(brief.get("hiring_signal", "unknown"), "⚪")
    lines = [
        f"# Company Brief: {company_name}" + (f" — {role_title}" if role_title else ""),
        f"\n**Stage:** {brief.get('stage','?').title()} · "
        f"**Hiring Signal:** {_sig} {brief.get('hiring_signal','?').title()} · "
        f"**Stability Score:** {brief.get('stability_score',0)}/100\n",
        f"**Verdict:** {brief.get('one_line_verdict','')}\n",
        f"## Culture\n{brief.get('culture_snapshot','')}",
        f"\n**Glassdoor read:** {brief.get('glassdoor_read','')}",
        f"\n## Compensation\n{brief.get('compensation_philosophy','')}",
        f"\n## Interview Process\n{brief.get('interview_process','')}",
    ]
    if brief.get("recent_signals"):
        lines.append("\n## Recent Signals")
        for s in brief["recent_signals"]:
            lines.append(f"- {s}")
    if brief.get("green_flags"):
        lines.append("\n## Green Flags")
        for g in brief["green_flags"]:
            lines.append(f"✓ {g}")
    if brief.get("red_flags"):
        lines.append("\n## Red Flags")
        for r in brief["red_flags"]:
            lines.append(f"⚠ {r}")
    if brief.get("cover_letter_hook"):
        lines.append(f"\n## Cover Letter Hook\n> {brief['cover_letter_hook']}")
    if brief.get("insider_tip"):
        lines.append(f"\n## Insider Tip\n> {brief['insider_tip']}")
    lines.append(f"\n---\n*Source: {brief.get('source','')}*")
    return "\n".join(lines)
