"""
LinkedIn Pivot Content Engine
==============================
LinkedIn is the only channel that simultaneously:
  (a) attracts passive recruiter inbound
  (b) builds pivot credibility publicly
  (c) mobilises the existing network
  (d) creates social proof of PM-adjacent thinking
  ...without a single active application.

Career changers who post strategically get 3x more recruiter inbound.
But most post wrong: either too generic ("excited to share I'm exploring PM!")
or too silent (nothing for months, then suddenly asking for referrals).

This module generates an 8-week content strategy using the candidate's
actual career stories — not invented examples. Every post references
a real achievement, a real project, a real insight from the CV.

Content principles for pivot credibility building:
  1. Show PM-thinking, not PM-aspirations. ("Here's how I analysed this" not "I want to do this")
  2. Specificity beats inspiration. Numbers, names, outcomes — not vague lessons.
  3. The pivot hook appears in week 1, then every 3rd post. Not constantly.
  4. Engagement questions at end of each post — build comment signal for algorithm.
  5. "Building in public" > "announcing in private." Show the work, don't announce the journey.

Output: 8 fully written posts, ready to copy-paste, with:
  - Optimal posting day/time
  - Hashtag strategy per post
  - Expected engagement type (learn vs. debate vs. inspire)
  - Connection to the larger content arc
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_CONTENT_SYSTEM = """You are a LinkedIn content strategist specialising in career pivot stories.
You write posts that build PM credibility through specific stories, not vague inspiration.

Rules:
  - Every post uses REAL data from the candidate's CV — specific numbers, company names, project outcomes
  - No generic "lessons learned" posts. Every post shows PM-adjacent thinking in action.
  - The pivot narrative appears subtly — never "I'm trying to break into PM."
    Instead: demonstrate PM thinking and let readers draw their own conclusion.
  - Each post ends with a genuine engagement question
  - Length: 150-250 words. Not a wall of text. Not too short to be substantive.
  - First line must be a hook that stops the scroll

Output JSON only:
{
  "content_arc": "2-sentence description of the 8-week narrative arc",
  "posts": [
    {
      "week": 1,
      "theme": "short theme label",
      "hook_type": "data|story|question|counterintuitive",
      "post_text": "complete ready-to-post LinkedIn post — first line is the hook, body is the story, ends with question",
      "hashtags": ["#hashtag1", "#hashtag2"],
      "post_day": "Tuesday|Wednesday|Thursday",
      "post_time": "8:00 AM|12:00 PM|6:00 PM",
      "expected_engagement": "learn|debate|inspire|share",
      "why_this_week": "why this post fits here in the arc"
    }
  ],
  "profile_optimization_tip": "One specific LinkedIn profile change to make this week before posting",
  "engagement_strategy": "How to respond to comments to maximise algorithm reach"
}"""


def generate_content_plan(
    oai_key: str,
    *,
    cv_profile: Dict,
    pivot_dna: Dict,
    target_role: str,
    skill_gap_results: Optional[Dict] = None,
    mock_interview_report: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate an 8-week LinkedIn content strategy for pivot credibility building.
    Returns content plan dict or None on failure.
    """
    if not oai_key or not cv_profile or not pivot_dna:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return None

    candidate = {
        "current_role":      cv_profile.get("extracted_role", ""),
        "target_role":       target_role,
        "years_experience":  cv_profile.get("years_experience", 0),
        "key_achievements":  cv_profile.get("key_achievements", [])[:5],
        "top_skills":        cv_profile.get("top_skills", [])[:8],
        "companies":         cv_profile.get("companies", []),
        "pivot_hook":        pivot_dna.get("pivot_hook", ""),
        "strongest_argument": pivot_dna.get("strongest_transferable_argument", ""),
        "unfair_advantage":  pivot_dna.get("unfair_advantage", ""),
        "writing_tone":      pivot_dna.get("writing_tone", "direct, data-grounded"),
        "three_word_brand":  pivot_dna.get("three_word_brand", ""),
        "career_narrative":  pivot_dna.get("career_narrative", "")[:300],
        "target_companies":  pivot_dna.get("target_companies", [])[:3],
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": _CONTENT_SYSTEM},
                {"role": "user", "content": (
                    f"Generate 8 LinkedIn posts for this career changer:\n"
                    f"{json.dumps(candidate, indent=2)}\n\n"
                    f"Every post must use their REAL data — specific numbers, company names, outcomes. "
                    f"No invented examples. No vague inspiration."
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def moat_builder_plan(
    oai_key: str,
    *,
    cv_profile: Dict,
    pivot_dna: Dict,
    target_role: str,
    skill_gap_results: Optional[Dict] = None,
    cohort_intelligence: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a Personal Moat Builder plan: proactive unfair advantages.
    What can this candidate do in 60 days that most PM candidates CAN'T?
    """
    if not oai_key or not cv_profile:
        return None

    _MOAT_SYSTEM = """You are a career strategist obsessed with unfair advantages.
    Career changers can't compete on PM title. They CAN compete on things
    PM lifers don't have: domain expertise, external relationships, unique data access,
    fresh perspectives, and the ability to do things no internal PM would do.

    Generate 5 concrete "moat-building" projects for this specific candidate.
    Each must be something MOST PM CANDIDATES CANNOT DO because they lack this person's background.

    Output JSON only:
    {
      "moat_philosophy": "one sentence: what is this candidate's core unfair advantage",
      "projects": [
        {
          "title": "project title",
          "what": "exactly what to build/write/publish/do in 60 days",
          "why_unique": "why someone without this background couldn't do this",
          "output": "what the deliverable is",
          "interview_angle": "how to naturally surface this in an interview",
          "time_investment": "hours total",
          "visibility_strategy": "where/how to publish or share this"
        }
      ],
      "60_day_plan": "ordered sequence of which projects to do in what order and why"
    }"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        candidate = {
            "current_role":      cv_profile.get("extracted_role", ""),
            "target_role":       target_role,
            "key_achievements":  cv_profile.get("key_achievements", [])[:4],
            "top_skills":        cv_profile.get("top_skills", [])[:8],
            "unfair_advantage":  pivot_dna.get("unfair_advantage", ""),
            "strongest_argument": pivot_dna.get("strongest_transferable_argument", ""),
            "companies":         cv_profile.get("companies", []),
            "top_gaps":          [(g.get("skill")) for g in ((skill_gap_results or {}).get("gaps") or [])[:4]],
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": _MOAT_SYSTEM},
                {"role": "user", "content": (
                    f"Build unfair advantages for:\n{json.dumps(candidate, indent=2)}\n\n"
                    f"Focus on what their SPECIFIC background enables that a typical PM candidate cannot match."
                )},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None
