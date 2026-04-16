"""
Cover Letter & Pivot Narrative Generator
==========================================
Generates three personalised output artifacts for a career pivot:

1. cover_letter      — a full application letter tailored to the pivot
2. elevator_pitch    — 2-3 sentences for networking or intro calls
3. linkedin_about    — a rewritten LinkedIn "About" section
4. talking_points    — 5 concrete talking points for interviews

When a personal CV profile is available (from cv_parser.py), the output
is grounded in the user's actual background. Without a CV, it falls back
to the O*NET role-level analysis.

This is the downstream action artifact — what you *do* with the analysis.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ── Offline templates ─────────────────────────────────────────────────────────

def _offline_narrative(
    current_role: str,
    target_role: str,
    recommended_strategy: str,
    top_transfer: List[str],
    top_missing: List[str],
) -> Dict[str, Any]:
    transfer_str = ", ".join(top_transfer[:3]) if top_transfer else "core analytical skills"
    missing_str = ", ".join(top_missing[:2]) if top_missing else "domain-specific knowledge"

    cover_letter = f"""Dear Hiring Manager,

I am writing to express my interest in a role as {target_role}. Coming from a background in {current_role}, I bring a strong foundation in {transfer_str} — skills that directly support the demands of {target_role}.

My transition strategy focuses on {recommended_strategy.lower().replace("_", " ")}: combining my existing strengths with targeted development in {missing_str}. This approach ensures I can contribute meaningfully from day one while continuing to grow into the full scope of the role.

I am confident that my analytical grounding and commitment to deliberate upskilling make me a strong candidate for this pivot. I would welcome the opportunity to discuss how my background aligns with your team's needs.

Best regards"""

    elevator_pitch = (
        f"I'm a {current_role} transitioning into {target_role}, "
        f"bringing deep expertise in {transfer_str}. "
        f"I'm closing the gap through focused work on {missing_str}."
    )

    linkedin_about = (
        f"Career pivot in progress: {current_role} → {target_role}.\n\n"
        f"My background gives me a strong foundation in {transfer_str}. "
        f"I'm now combining that with hands-on development in {missing_str} "
        f"to build the complete skill set for {target_role}.\n\n"
        f"Open to conversations about opportunities in this space."
    )

    return {
        "cover_letter": cover_letter,
        "elevator_pitch": elevator_pitch,
        "linkedin_about": linkedin_about,
        "talking_points": [
            f"My experience in {current_role} gave me a strong base in {transfer_str}",
            f"I've identified {missing_str} as my key development areas and am actively closing those gaps",
            f"My chosen strategy ({recommended_strategy.replace('_', ' ').title()}) balances speed with credibility",
            "I'm not making a blind leap — I've done rigorous analysis of the skill overlap and gaps",
            "I can contribute immediately in areas where the skill sets overlap, while building toward full capability",
        ],
        "source": "offline",
    }


# ── Online generation ─────────────────────────────────────────────────────────

def generate_pivot_narrative(
    current_role: str,
    target_role: str,
    recommended_strategy: str,
    top_transfer: List[str],
    top_missing: List[str],
    match_score: float = 0.0,
    verdict: str = "Feasible with Conditions",
    cv_profile: Optional[Dict[str, Any]] = None,
    agent_executive_summary: Optional[str] = None,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate personalised pivot narrative artifacts.

    Parameters
    ----------
    current_role / target_role : the pivot direction
    recommended_strategy       : e.g. "HYBRID", "SKILL_FIRST"
    top_transfer               : top transferable skills from gap analysis
    top_missing                : top missing skills from gap analysis
    match_score                : cosine similarity score 0-100
    verdict                    : agent or consensus verdict
    cv_profile                 : dict from cv_parser.parse_cv() — if available, personalises output
    agent_executive_summary    : agent's executive summary — grounding context for the LLM
    """
    if not prefer_online:
        return _offline_narrative(current_role, target_role, recommended_strategy, top_transfer, top_missing)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_narrative(current_role, target_role, recommended_strategy, top_transfer, top_missing)

    # Build personal context block
    personal_context = ""
    if cv_profile and cv_profile.get("extracted_role"):
        p = cv_profile
        top_skills_str = ", ".join(p.get("top_skills", [])[:6])
        personal_context = f"""
PERSONAL CONTEXT (from CV):
- Current role: {p.get('extracted_role', current_role)}
- Years of experience: {p.get('years_experience', 0):.0f}
- Education: {p.get('education_level', 'Not specified')}
- Top skills (O*NET mapped): {top_skills_str}
- Skills extracted count: {p.get('skills_extracted_count', 0)}

Use this personal context to make the output specific and authentic — not generic.
Address the person's actual background, not a hypothetical role average.
"""

    analysis_context = ""
    if agent_executive_summary:
        analysis_context = f"\nAI ANALYSIS SUMMARY:\n{agent_executive_summary}\n"

    prompt = f"""You are a career pivot specialist writing high-quality application materials.

PIVOT DETAILS:
- From: {current_role}
- To: {target_role}
- Recommended strategy: {recommended_strategy.replace("_", " ").title()}
- Skill match score: {match_score:.0f}/100
- Verdict: {verdict}
- Top transferable skills: {", ".join(top_transfer[:5])}
- Key skills to develop: {", ".join(top_missing[:4])}
{personal_context}{analysis_context}
Generate four pieces of content. Respond ONLY with valid JSON:

{{
  "cover_letter": "A professional, specific cover letter (4-5 paragraphs). Reference the actual skills and the pivot strategy. Do NOT use generic filler. Make it feel human and grounded.",
  "elevator_pitch": "2-3 sentences for a networking conversation or intro call. Concise, confident, specific.",
  "linkedin_about": "A rewritten LinkedIn About section (150-200 words). First person. Shows the pivot narrative honestly and compellingly.",
  "talking_points": [
    "5 concrete talking points for interviews — specific claims the person can make about their background and transition plan"
  ]
}}

Rules:
- Be specific — mention actual skills, not just 'relevant experience'
- Acknowledge the pivot honestly — don't pretend it doesn't exist
- Show the deliberate nature of the transition (analysis, strategy, upskilling plan)
- If personal CV data is available, write about THAT PERSON specifically
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1800,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        # Validate structure
        for key in ["cover_letter", "elevator_pitch", "linkedin_about", "talking_points"]:
            if key not in data:
                data[key] = ""
        if not isinstance(data["talking_points"], list):
            data["talking_points"] = []
        data["source"] = "online"
        data["personalized"] = bool(cv_profile and cv_profile.get("extracted_role"))
        return data
    except Exception as e:
        result = _offline_narrative(current_role, target_role, recommended_strategy, top_transfer, top_missing)
        result["error"] = repr(e)
        return result
