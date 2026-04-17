"""
Interview Coach
===============
Generates role-specific interview questions and evaluates user answers
with coaching feedback. Completes the full career pivot journey:

    Assess → Plan → Validate → Execute → Interview-Ready

Two-layer architecture (mirrors the evaluator pattern):
  Layer 1 — Generation (gpt-4o-mini): produce tailored Q&A questions
  Layer 2 — Evaluation (gpt-4o-mini): score the user's draft answer,
             then generate an improved coached version

Both functions have rule-based fallbacks so the Interview tab always works
even without an API key.

Returns
-------
generate_interview_questions → List[Dict]:
    [{question, type, why_asked, difficulty}]

evaluate_interview_answer → Dict:
    {
      overall_score: int          0-100
      dimension_scores: dict      relevance · specificity · star_structure · keywords
      strengths: List[str]
      improvements: List[str]
      coached_answer: str         improved version of the user's answer
      one_line_verdict: str       ≤100 chars
      regenerate_recommended: bool
      source: str                 "llm" | "heuristic" | "heuristic (error: …)"
    }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic fallbacks
# ──────────────────────────────────────────────────────────────────────────────

_QUESTION_TEMPLATES = [
    {
        "question": "Tell me about yourself and why you're making this career transition.",
        "type": "Behavioural",
        "why_asked": "Screens for self-awareness and a coherent pivot narrative.",
        "difficulty": "Medium",
    },
    {
        "question": "What specific skills from your current background transfer directly to this role?",
        "type": "Competency",
        "why_asked": "Tests whether the candidate can articulate transferable value.",
        "difficulty": "Medium",
    },
    {
        "question": "Describe a time you had to learn a completely new skill under time pressure. What was your approach?",
        "type": "Behavioural (STAR)",
        "why_asked": "Signals learning agility — critical for career changers.",
        "difficulty": "Medium",
    },
    {
        "question": "What is the biggest skill gap you have for this role, and what are you doing about it?",
        "type": "Self-awareness",
        "why_asked": "Tests honesty and proactive gap-filling.",
        "difficulty": "Hard",
    },
    {
        "question": "Where do you see yourself in 3 years in this new field?",
        "type": "Motivation",
        "why_asked": "Validates genuine long-term intent vs. opportunistic pivot.",
        "difficulty": "Easy",
    },
    {
        "question": "How would you handle a situation where a colleague with more domain experience disagrees with your approach?",
        "type": "Behavioural (STAR)",
        "why_asked": "Tests humility and collaboration — key for career changers entering a new domain.",
        "difficulty": "Medium",
    },
]


def _heuristic_questions(target_role: str) -> List[Dict[str, str]]:
    """Return template questions when the API is unavailable."""
    return [dict(q) for q in _QUESTION_TEMPLATES]


def _heuristic_eval_answer(answer: str) -> Dict[str, Any]:
    """Rule-based answer evaluation fallback."""
    words = answer.split()
    wc = len(words)
    has_star = any(
        w in answer.lower()
        for w in ["situation", "task", "action", "result", "when i", "i decided", "the outcome"]
    )
    has_numbers = any(c.isdigit() for c in answer)
    has_specifics = wc > 60

    base = 52
    if has_star:      base += 12
    if has_numbers:   base += 8
    if has_specifics: base += 8
    if wc > 120:      base += 6
    score = min(base, 80)

    return {
        "overall_score": score,
        "dimension_scores": {
            "relevance":      score,
            "specificity":    max(score - 10, 40),
            "star_structure": (score + 10) if has_star else max(score - 15, 35),
            "keywords":       max(score - 8, 40),
        },
        "strengths": [
            "Answer addresses the question directly.",
            "Professional tone and clear structure.",
        ],
        "improvements": [
            "Add a concrete example using the STAR format (Situation → Task → Action → Result).",
            "Quantify the impact where possible (%, £, team size).",
        ],
        "coached_answer": (
            f"[Coached version unavailable offline — API key required]\n\n"
            f"Your draft ({wc} words) is a good start. To strengthen it:\n"
            f"1. Open with the specific situation (1-2 sentences)\n"
            f"2. State your role/task clearly\n"
            f"3. Describe 2-3 concrete actions YOU took\n"
            f"4. Close with a measurable result"
        ),
        "one_line_verdict": "Solid start — add STAR structure and one quantified outcome to score 80+.",
        "regenerate_recommended": score < 55,
        "source": "heuristic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Question generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_interview_questions(
    target_role: str,
    job_description: str = "",
    cv_text: str = "",
    n: int = 6,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> List[Dict[str, str]]:
    """
    Generate n role-specific interview questions for the target role.

    When a job description is provided, questions are tailored to the exact
    posting (including technical requirements and company context). When a CV
    is provided, follow-up questions probe the candidate's specific background.

    Each question dict has:
        question    — the full question text
        type        — Behavioural | Technical | Competency | Motivation | Self-awareness
        why_asked   — 1-sentence explanation of what the interviewer is testing
        difficulty  — Easy | Medium | Hard
    """
    if not prefer_online:
        return _heuristic_questions(target_role)[:n]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_questions(target_role)[:n]

    jd_snippet = (job_description or "")[:1000] or "Not provided"
    cv_snippet = (cv_text or "")[:500] or "Not provided"

    prompt = f"""You are a senior hiring manager preparing to interview a career-changer for:
ROLE: {target_role}
JOB DESCRIPTION (excerpt): {jd_snippet}
CANDIDATE CV (excerpt): {cv_snippet}

Generate exactly {n} interview questions for this candidate. Mix:
- 2 behavioural questions (STAR format expected)
- 2 competency/technical questions specific to {target_role}
- 1 motivation/pivot question (why this role? why now?)
- 1 self-awareness question (gap acknowledgement)

For each question explain what you're actually testing.

Respond ONLY with valid JSON — a list of objects:
[
  {{
    "question": "Full question text?",
    "type": "Behavioural",
    "why_asked": "Tests X in one sentence.",
    "difficulty": "Medium"
  }}
]

difficulty must be one of: Easy | Medium | Hard
type must be one of: Behavioural | Technical | Competency | Motivation | Self-awareness
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=900,
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        # API may return {"questions": [...]} or a bare list wrapped in a key
        items = None
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, list):
                    items = v
                    break
        if not items:
            return _heuristic_questions(target_role)[:n]

        questions = []
        for item in items[:n]:
            questions.append({
                "question":   str(item.get("question", "")),
                "type":       str(item.get("type", "Behavioural")),
                "why_asked":  str(item.get("why_asked", "")),
                "difficulty": str(item.get("difficulty", "Medium")),
            })
        return questions if questions else _heuristic_questions(target_role)[:n]

    except Exception:
        return _heuristic_questions(target_role)[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Answer evaluation + coaching
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_interview_answer(
    question: str,
    answer: str,
    target_role: str,
    job_title: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Score the user's draft answer to an interview question and return
    an improved coached version.

    Scores four dimensions that predict interview success:
    - relevance:      does the answer address what was asked?
    - specificity:    concrete details vs. vague generalisations
    - star_structure: Situation/Task/Action/Result completeness
    - keywords:       role-relevant vocabulary and terminology

    Overall = relevance×0.30 + specificity×0.30 + star_structure×0.25 + keywords×0.15

    The coached_answer field contains a rewritten version of the candidate's
    answer that incorporates all improvement suggestions — ready to use as a
    preparation template.
    """
    if not answer.strip():
        return _heuristic_eval_answer("")

    if not prefer_online:
        return _heuristic_eval_answer(answer)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_eval_answer(answer)

    prompt = f"""You are a career coach evaluating a candidate's interview answer for a {target_role} role.
Be honest and specific — generic feedback wastes the candidate's preparation time.

QUESTION: {question}
CANDIDATE'S ANSWER: {answer}

Evaluate and then write an improved coached version of their answer.
The coached answer should preserve their personal experiences/examples but sharpen structure, add specificity, and mirror {target_role} terminology.

Respond ONLY with valid JSON:
{{
  "overall_score": 72,
  "dimension_scores": {{
    "relevance": 80,
    "specificity": 65,
    "star_structure": 70,
    "keywords": 68
  }},
  "strengths": [
    "Specific strength — quote or paraphrase what they said",
    "Second strength"
  ],
  "improvements": [
    "Specific improvement — what is missing or weak",
    "Second improvement"
  ],
  "coached_answer": "Start with the specific situation: [rewritten full answer here, 150-250 words, using their own examples]",
  "one_line_verdict": "Good structure but vague on outcomes — add one metric to hit 80+.",
  "regenerate_recommended": false
}}

Scoring rules:
- overall_score = relevance×0.30 + specificity×0.30 + star_structure×0.25 + keywords×0.15
- regenerate_recommended = true when overall_score < 55
- one_line_verdict ≤100 chars, actionable
- coached_answer: PRESERVE their real experiences, only improve structure/language/specificity
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=700,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "overall_score": int(data.get("overall_score", 65)),
            "dimension_scores": {
                k: int(v) for k, v in data.get("dimension_scores", {}).items()
            },
            "strengths":    [str(s) for s in data.get("strengths", [])[:3]],
            "improvements": [str(i) for i in data.get("improvements", [])[:3]],
            "coached_answer": str(data.get("coached_answer", ""))[:2000],
            "one_line_verdict": str(data.get("one_line_verdict", ""))[:120],
            "regenerate_recommended": bool(data.get("regenerate_recommended", False)),
            "source": "llm",
        }
    except Exception as exc:
        result = _heuristic_eval_answer(answer)
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result
