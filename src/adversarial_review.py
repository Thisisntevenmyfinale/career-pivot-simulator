"""
Adversarial Application Review — "Der Ablehnungsrichter"
==========================================================
Every other career tool optimises your application and tells you it's good.
This one tries to kill it.

The insight: hiring managers spend 6-8 seconds on initial review.
In those 6 seconds, they are looking for reasons to REJECT, not reasons to hire.
This module adopts that hostile perspective before the candidate submits.

The adversarial reviewer embodies a sceptical hiring manager who:
  - Has seen 200+ pivot candidates and is tired of generic narratives
  - Is specifically looking for signs of PM-inexperience
  - Reads the cover letter trying to find the one reason to pass
  - Checks: does the CV match the JD? Is the pivot story believable?

Output is deliberately unsparing. A score of 90 from a "helpful" reviewer
means nothing. A score of 90 from a hostile reviewer means the application
is genuinely strong.

Three rejection axes:
  1. Credibility axis — does the pivot argument hold up?
  2. Relevance axis — does the application match THIS role?
  3. Execution axis — is the writing tight, specific, non-generic?
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_ADVERSARIAL_SYSTEM = """You are a senior hiring manager who has reviewed 300+ PM candidates.
You are HOSTILE to this application. Your job is to find every reason to reject it.

You do NOT want to be encouraging. You do NOT soften your critique.
Every weakness you identify is a potential rejection reason — name it explicitly.

You are evaluating:
  1. Credibility: Does this pivot story actually hold up? Or is it "I've always been interested in product"?
  2. Relevance: Does this application actually match THIS role at THIS company?
  3. Execution: Is the writing tight, specific, non-generic? Or is it fluffy?

After finding every weakness, you ALSO identify the 2 strongest points — because you're honest, not just mean.

Output JSON only:
{
  "hostile_verdict": "one brutal sentence: would you throw this in the reject pile or not",
  "reject_probability": <int 0-100, probability this gets rejected in first pass>,
  "rejection_reasons": [
    {
      "axis": "credibility|relevance|execution",
      "severity": "fatal|major|minor",
      "reason": "specific, concrete rejection reason — reference actual text where possible",
      "fix": "exact fix to apply before submitting"
    }
  ],
  "fatal_flaws": ["the dealbreaker issues — if any of these aren't fixed, don't submit"],
  "strongest_points": ["the 2 things that actually work — be specific"],
  "ats_pass_probability": <int 0-100>,
  "human_read_probability": <int 0-100, if it passes ATS will a human spend >30 seconds>,
  "one_thing_to_fix_now": "The single most important change — if you only do one thing",
  "submit_as_is": <bool, should they submit NOW or fix first>
}"""


def run_adversarial_review(
    oai_key: str,
    *,
    cover_letter: str,
    cv_text: str,
    job_title: str,
    company: str,
    jd_text: str = "",
    pivot_dna: Optional[Dict] = None,
    cv_profile: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run an adversarial review of a cover letter + CV against a specific role.

    Returns hostile analysis dict or None on failure.
    """
    if not oai_key or not cover_letter.strip():
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    candidate_context = {
        "current_role": (cv_profile or {}).get("extracted_role", ""),
        "pivot_hook":   (pivot_dna  or {}).get("pivot_hook", ""),
        "pivot_risk":   (pivot_dna  or {}).get("pivot_risk", ""),
    }

    prompt = (
        f"Role: {job_title} at {company}\n"
        f"JD excerpt: {jd_text[:800]}\n\n"
        f"Candidate: making a career pivot. {json.dumps(candidate_context)}\n\n"
        f"Cover letter to destroy:\n---\n{cover_letter[:2500]}\n---\n\n"
        f"CV excerpt:\n---\n{cv_text[:1500]}\n---\n\n"
        f"Find every reason to reject this. Be specific."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": _ADVERSARIAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        result["company"]   = company
        result["job_title"] = job_title
        return result
    except Exception:
        return None


def severity_color(sev: str) -> str:
    return {"fatal": "#7C2D12", "major": "#DC2626", "minor": "#D97706"}.get(sev, "#555")


def severity_label(sev: str) -> str:
    return {"fatal": "FATAL", "major": "Major", "minor": "Minor"}.get(sev, sev.upper())


def reject_prob_color(p: int) -> str:
    if p >= 70: return "#DC2626"
    if p >= 40: return "#D97706"
    return "#057642"
