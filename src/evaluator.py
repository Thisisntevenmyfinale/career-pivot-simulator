"""
Output Quality Evaluator
========================
Second-pass LLM evaluation of AI-generated career materials.

The #1 mistake in AI pipelines: using LLM outputs raw without evaluating their
quality for the specific task. This module adds an explicit evaluation step after
every generation, scoring content across relevant dimensions and providing
actionable feedback to the user.

Evaluated artifacts
-------------------
- Application packages (cover letter, InMail, CV rewrites)
- Learning plans (gap coverage, resource specificity, actionability)

Architecture
------------
Fast gpt-4o-mini second-pass call with structured JSON output.
Falls back to rule-based heuristics (word-count + keyword signals) when the
API is unavailable, so quality scores are always shown — never hidden behind
an API gate.

Returns
-------
{
  overall_score: int          — 0-100 composite quality score
  dimension_scores: dict      — per-dimension breakdown
  strengths: List[str]        — 2-3 specific strengths (citing actual content)
  improvements: List[str]     — 2-3 concrete improvement suggestions
  regenerate_recommended: bool — True if overall_score < 65
  one_line_verdict: str       — ≤100-char summary shown as the quality badge
  source: str                 — "llm" | "heuristic" | "heuristic (error: …)"
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic fallback
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_eval(
    content: str,
    content_type: str = "generic",
    context: str = "",
) -> Dict[str, Any]:
    """Rule-based quality estimate when the LLM API is unavailable."""
    words = content.split()
    wc = len(words)
    sentences = content.count(".") + content.count("!") + content.count("?")
    has_specifics = any(
        w in content.lower()
        for w in ["specifically", "experience with", "expertise", "proven", "delivered",
                  "increased", "reduced", "led", "managed", "built"]
    )
    has_personal = any(
        w in content.lower()
        for w in ["i have", "my background", "i've", "my experience", "i led"]
    )
    substantial = wc > 150

    base = 55
    if substantial:         base += 10
    if has_specifics:       base += 8
    if has_personal:        base += 6
    if wc > 350:            base += 5
    if sentences >= 5:      base += 4
    score = min(base, 82)

    verdict_map = {
        "cover_letter":   "Solid structure — add role-specific keywords to boost score.",
        "learning_plan":  "Good roadmap — more specific resource names would help.",
        "generic":        "Acceptable output — add concrete details to strengthen.",
    }

    return {
        "overall_score": score,
        "dimension_scores": {
            "relevance": score,
            "specificity": max(score - 8, 40),
            "narrative_strength": max(score - 4, 40),
            "completeness": min(score + 5, 90),
        },
        "strengths": [
            "Well-structured content with clear paragraphs.",
            "Appropriate length and professional tone.",
        ],
        "improvements": [
            "Reference specific keywords from the job description.",
            "Add concrete achievements with measurable outcomes.",
        ],
        "regenerate_recommended": score < 65,
        "one_line_verdict": verdict_map.get(content_type, verdict_map["generic"]),
        "source": "heuristic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Application Package Evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_application_package(
    cover_letter: str,
    linkedin_inmail: str,
    cv_rewrites: List[Dict[str, str]],
    job_title: str,
    company: str,
    job_description: str = "",
    cv_text: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated application package.

    Scores four dimensions that predict application success:
    - job_relevance: does the content address the specific job requirements?
    - narrative_specificity: concrete examples vs. generic statements
    - inmail_impact: would a hiring manager respond to this message?
    - cv_rewrite_quality: do the rewrites highlight transferable skills?

    Falls back to rule-based heuristics when the API is unavailable.
    """
    if not prefer_online:
        return _heuristic_eval(cover_letter, "cover_letter")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_eval(cover_letter, "cover_letter")

    jd_snippet    = (job_description or "")[:800] or "Not provided"
    cv_snippet    = (cv_text or "")[:400] or "Not provided"
    cl_snippet    = cover_letter[:1400]
    inmail_snippet = linkedin_inmail[:700]
    rewrites_preview = "; ".join(
        f"{r.get('skill_highlighted', '')}: {r.get('rewritten', '')[:80]}"
        for r in (cv_rewrites or [])[:3]
    )

    prompt = f"""You are a senior hiring manager and career coach evaluating AI-generated application materials.
Be critical and specific — generic praise inflates scores and wastes the candidate's time.

ROLE: {job_title} at {company}
JOB DESCRIPTION (excerpt): {jd_snippet}
CANDIDATE BACKGROUND (CV excerpt): {cv_snippet}

COVER LETTER (excerpt):
{cl_snippet}

LINKEDIN INMAIL (excerpt):
{inmail_snippet}

CV REWRITES (sample): {rewrites_preview}

Evaluate honestly. Cite specific phrases when praising or criticising.

Respond ONLY with valid JSON:
{{
  "overall_score": 78,
  "dimension_scores": {{
    "job_relevance": 85,
    "narrative_specificity": 72,
    "inmail_impact": 76,
    "cv_rewrite_quality": 80
  }},
  "strengths": [
    "Strength 1 — quote or paraphrase what you saw",
    "Strength 2"
  ],
  "improvements": [
    "Improvement 1 — be specific about what's missing",
    "Improvement 2"
  ],
  "regenerate_recommended": false,
  "one_line_verdict": "Strong cover letter; InMail opening too generic — needs a hook."
}}

Scoring rules:
- overall_score = job_relevance×0.35 + narrative_specificity×0.25 + inmail_impact×0.20 + cv_rewrite_quality×0.20
- regenerate_recommended = true when overall_score < 62
- one_line_verdict must be ≤100 characters and say something actionable
- If job description was not provided, weight narrative_specificity higher
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=450,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "overall_score": int(data.get("overall_score", 70)),
            "dimension_scores": {
                k: int(v) for k, v in data.get("dimension_scores", {}).items()
            },
            "strengths":    [str(s) for s in data.get("strengths", [])[:3]],
            "improvements": [str(i) for i in data.get("improvements", [])[:3]],
            "regenerate_recommended": bool(data.get("regenerate_recommended", False)),
            "one_line_verdict": str(data.get("one_line_verdict", ""))[:120],
            "source": "llm",
        }
    except Exception as exc:
        result = _heuristic_eval(cover_letter, "cover_letter")
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Learning Plan Evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_learning_plan(
    plan_markdown: str,
    skill_gaps: List[str],
    target_role: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a generated learning plan for gap coverage and actionability.

    Scores:
    - gap_coverage: what fraction of the skill gaps are addressed?
    - resource_specificity: named courses/books/projects vs. "take an online course"
    - timeline_realism: are time estimates achievable?
    - actionability: can the user start today from this plan?
    """
    if not prefer_online:
        return _heuristic_eval(plan_markdown, "learning_plan")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_eval(plan_markdown, "learning_plan")

    gaps_str = ", ".join(skill_gaps[:10]) if skill_gaps else "not specified"
    plan_snippet = plan_markdown[:1800]

    prompt = f"""You are a career development expert evaluating an AI-generated learning plan.
The candidate is pivoting to: {target_role}
Skill gaps to address: {gaps_str}

LEARNING PLAN:
{plan_snippet}

Score this plan critically. Name specific gaps that are well-covered or missed.

Respond ONLY with valid JSON:
{{
  "overall_score": 74,
  "dimension_scores": {{
    "gap_coverage": 80,
    "resource_specificity": 65,
    "timeline_realism": 78,
    "actionability": 75
  }},
  "strengths": [
    "Specific strength 1 — what gap does it cover well?",
    "Specific strength 2"
  ],
  "improvements": [
    "Specific improvement 1 — which gap is underserved or missing?",
    "Specific improvement 2"
  ],
  "regenerate_recommended": false,
  "one_line_verdict": "Covers 5/7 gaps with specific resources; no timeline for skill 'X'."
}}

Rules:
- overall_score = gap_coverage×0.35 + resource_specificity×0.25 + actionability×0.25 + timeline_realism×0.15
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
            "dimension_scores": {
                k: int(v) for k, v in data.get("dimension_scores", {}).items()
            },
            "strengths":    [str(s) for s in data.get("strengths", [])[:3]],
            "improvements": [str(i) for i in data.get("improvements", [])[:3]],
            "regenerate_recommended": bool(data.get("regenerate_recommended", False)),
            "one_line_verdict": str(data.get("one_line_verdict", ""))[:120],
            "source": "llm",
        }
    except Exception as exc:
        result = _heuristic_eval(plan_markdown, "learning_plan")
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result
