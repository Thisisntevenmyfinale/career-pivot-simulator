"""
LinkedIn Profile Optimizer
==========================
Generates a complete LinkedIn profile update for a career-changer:
  - Headline (220 chars max)
  - About / Summary section (200–260 words, first person, pivot story)
  - Experience bullets for current role (reframed toward target role)
  - Skills section (top 15 skills to list for the target role)

Uses the candidate's CV text, current role, and target role as inputs.
The output is immediately paste-able into LinkedIn — no further editing needed.

Architecture: gpt-4o-mini with structured JSON output.
Fallback: rule-based template when API unavailable.
Evaluation: LLM evaluates the generated profile on 4 dimensions.

Model choice: gpt-4o-mini — this is a constrained generation task (LinkedIn
has strict character limits) and the output is scored by a separate evaluator.
The constraint structure means mini performs as well as gpt-4o here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic fallback
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_profile(current_role: str, target_role: str) -> Dict[str, Any]:
    return {
        "headline": (
            f"{current_role} transitioning to {target_role} "
            f"| Leveraging transferable skills for career pivot"
        )[:220],
        "about": (
            f"I'm a {current_role} with a passion for {target_role.lower()}. "
            f"Over the course of my career, I've developed strong analytical, "
            f"communication, and problem-solving skills that translate directly "
            f"into success in {target_role.lower()} roles.\n\n"
            f"I'm currently building expertise in the specific technical and "
            f"domain knowledge required for {target_role.lower()}, while drawing "
            f"on my foundation in {current_role.lower()} to bring a unique "
            f"cross-functional perspective.\n\n"
            f"Open to connecting with {target_role.lower()} professionals and "
            f"opportunities in this space."
        ),
        "experience_bullets": [
            f"Developed transferable skills applicable to {target_role} through cross-functional work",
            f"Applied structured problem-solving methods relevant to {target_role} contexts",
            f"Built stakeholder communication skills applicable across industries",
        ],
        "skills_list": [
            f"{target_role} fundamentals",
            "Strategic planning",
            "Data analysis",
            "Stakeholder management",
            "Project management",
            "Problem solving",
            "Cross-functional collaboration",
            "Communication",
            "Leadership",
            "Adaptability",
        ],
        "source": "heuristic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_linkedin_profile(
    current_role: str,
    target_role: str,
    cv_text: str = "",
    top_transferable_skills: Optional[List[str]] = None,
    top_gap_skills: Optional[List[str]] = None,
    salary_delta_pct: Optional[float] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate a complete LinkedIn profile update for a career pivot.

    Returns a dict with:
      headline         — ≤220 chars, keyword-rich, signals the pivot
      about            — 200-260 words, first-person, tells the pivot story
      experience_bullets — 3-4 bullets reframing current role for target role
      skills_list      — 12-15 skills to list on LinkedIn for target role
      source           — "llm" | "heuristic" | "heuristic (error: ...)"
    """
    if not prefer_online:
        return _heuristic_profile(current_role, target_role)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_profile(current_role, target_role)

    cv_snippet = (cv_text or "")[:600] or "Not provided"
    skills_xfer = ", ".join(top_transferable_skills[:6]) if top_transferable_skills else "Not specified"
    skills_gaps = ", ".join(top_gap_skills[:5]) if top_gap_skills else "Not specified"

    prompt = f"""You are a LinkedIn profile writer specialising in career pivots.
Write a complete LinkedIn profile update for a candidate pivoting:
FROM: {current_role}
TO: {target_role}

CV excerpt: {cv_snippet}
Their strongest transferable skills: {skills_xfer}
Skills they're still building: {skills_gaps}

Requirements:
- headline: ≤220 characters. Include current expertise + target direction + one hook.
  Example format: "{current_role} → {target_role} | [unique angle or achievement]"
- about: 200-260 words. First person. Tell the pivot story authentically:
  Para 1: What drives the move (curiosity, gap in market, natural extension)
  Para 2: What they bring (specific transferable skills with brief evidence)
  Para 3: What they're building (specific skills/projects/courses in progress)
  Para 4: Call to action (open to connecting / opportunities)
  No clichés like "passionate" or "results-driven" — be specific and human.
- experience_bullets: 3 bullets rewriting their current role experience
  to highlight skills relevant to {target_role}. Start each with an action verb.
- skills_list: 14 skills to list on LinkedIn. Mix of:
  - 5 target-role core skills (what recruiters filter for)
  - 5 transferable skills they already have
  - 4 skills they are currently building

Respond ONLY with valid JSON:
{{
  "headline": "...",
  "about": "...",
  "experience_bullets": ["...", "...", "..."],
  "skills_list": ["...", "...", "..."]
}}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=900,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "headline":            str(data.get("headline", ""))[:220],
            "about":               str(data.get("about", "")),
            "experience_bullets":  [str(b) for b in data.get("experience_bullets", [])[:5]],
            "skills_list":         [str(s) for s in data.get("skills_list", [])[:15]],
            "source": "llm",
        }
    except Exception as exc:
        result = _heuristic_profile(current_role, target_role)
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_linkedin_profile(
    profile: Dict[str, Any],
    current_role: str,
    target_role: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Score the generated LinkedIn profile on 4 dimensions:
    - pivot_clarity:   does it clearly signal the career change?
    - keyword_density: does it include terms recruiters search for?
    - authenticity:    does it sound human vs. corporate template?
    - call_to_action:  does it invite the right kind of response?

    Returns the standard evaluator dict: overall_score, dimension_scores,
    strengths, improvements, one_line_verdict, regenerate_recommended, source.
    """
    if not prefer_online:
        return {
            "overall_score": 72,
            "dimension_scores": {"pivot_clarity": 75, "keyword_density": 68, "authenticity": 74, "call_to_action": 70},
            "strengths": ["Clear pivot direction in headline.", "About section tells a coherent story."],
            "improvements": ["Add 1-2 specific projects or courses in progress.", "Headline could include a metric or achievement."],
            "one_line_verdict": "Good foundation — add specifics to reach 80+.",
            "regenerate_recommended": False,
            "source": "heuristic",
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return evaluate_linkedin_profile(profile, current_role, target_role, prefer_online=False)

    prompt = f"""You are a LinkedIn profile coach evaluating a career-pivot profile.
The candidate is pivoting from {current_role} to {target_role}.

HEADLINE: {profile.get('headline', '')}

ABOUT: {profile.get('about', '')}

EXPERIENCE BULLETS (sample): {'; '.join(profile.get('experience_bullets', [])[:2])}

SKILLS: {', '.join(profile.get('skills_list', [])[:10])}

Score critically on 4 dimensions:
- pivot_clarity: does the reader immediately understand the direction?
- keyword_density: would a {target_role} recruiter find this in search?
- authenticity: specific and human vs. generic corporate template?
- call_to_action: does the about section invite the right response?

Respond ONLY with valid JSON:
{{
  "overall_score": 76,
  "dimension_scores": {{"pivot_clarity": 80, "keyword_density": 72, "authenticity": 78, "call_to_action": 74}},
  "strengths": ["Specific strength citing actual text", "Second strength"],
  "improvements": ["Specific improvement", "Second improvement"],
  "one_line_verdict": "Strong pivot signal; headline needs one concrete achievement.",
  "regenerate_recommended": false
}}

Rules:
- overall_score = pivot_clarity×0.30 + keyword_density×0.30 + authenticity×0.25 + call_to_action×0.15
- regenerate_recommended = true when overall_score < 60
- one_line_verdict ≤100 chars
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=400,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "overall_score": int(data.get("overall_score", 72)),
            "dimension_scores": {k: int(v) for k, v in data.get("dimension_scores", {}).items()},
            "strengths":    [str(s) for s in data.get("strengths", [])[:3]],
            "improvements": [str(i) for i in data.get("improvements", [])[:3]],
            "one_line_verdict": str(data.get("one_line_verdict", ""))[:120],
            "regenerate_recommended": bool(data.get("regenerate_recommended", False)),
            "source": "llm",
        }
    except Exception as exc:
        result = evaluate_linkedin_profile(profile, current_role, target_role, prefer_online=False)
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result
