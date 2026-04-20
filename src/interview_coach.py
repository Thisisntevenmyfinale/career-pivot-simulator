"""
Interview Coach
===============
Generates role-specific interview questions, evaluates user answers with
coaching feedback, and runs a live multi-turn Mock Interview Simulator.

Pipeline
--------
    Assess → Plan → Validate → Execute → Interview-Ready → Mock Interview → Offer

Three modes:
  1. generate_interview_questions   — produce N tailored questions
  2. evaluate_interview_answer      — score + coach a single answer
  3. run_mock_interview_turn        — one turn of a live conversational interview
  4. generate_mock_interview_report — post-interview performance analysis (gpt-4o)

Mock Interview Architecture
---------------------------
  gpt-4o plays a realistic senior interviewer at the target company.
  It receives the full conversation history each turn and responds with:
    - Brief acknowledgement of the previous answer (1 sentence)
    - One targeted follow-up OR a new topic question
  After max_exchanges, it gives a closing statement and sets is_complete=True.
  A separate report call then analyses the full conversation across 5 dimensions.

All functions have rule-based fallbacks so the Interview tab always works
even without an API key.
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


# ──────────────────────────────────────────────────────────────────────────────
# Mock Interview Simulator — multi-turn conversational interview
# ──────────────────────────────────────────────────────────────────────────────

def run_mock_interview_turn(
    history: List[Dict[str, str]],
    job_title: str,
    company: str = "",
    job_description: str = "",
    current_role: str = "",
    target_role: str = "",
    cv_summary: str = "",
    exchange_num: int = 1,
    max_exchanges: int = 6,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Conduct one turn of a live mock interview.

    The interviewer (gpt-4o) receives full conversation history and responds
    with a brief acknowledgement + one focused follow-up or new question.
    After max_exchanges it delivers a closing statement and signals completion.

    Returns:
        {
          "response": str,        # interviewer's next message
          "is_complete": bool,    # True when interview is finished
          "exchange_num": int,    # current exchange count
          "source": str,
        }
    """
    _heuristic_closes = [
        "That's a strong example — thank you. We're wrapping up. You've shown genuine "
        "self-awareness about your pivot. Expect our decision within a week.",
        "Appreciate the detailed answer. That concludes our interview today. "
        "You made a compelling case for your transferable skills.",
    ]
    if not prefer_online:
        return {
            "response": _heuristic_closes[0],
            "is_complete": True,
            "exchange_num": exchange_num,
            "source": "heuristic",
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return {
            "response": _heuristic_closes[0],
            "is_complete": True,
            "exchange_num": exchange_num,
            "source": "heuristic (openai unavailable)",
        }

    remaining = max_exchanges - exchange_num
    company_str = f" at {company}" if company else ""
    jd_snippet = (job_description or "")[:600] or "Not provided"
    cv_str = (cv_summary or "")[:300] or "Not provided"

    system_prompt = f"""You are a senior {job_title} interviewer{company_str}.
You are conducting a real job interview with a candidate transitioning from {current_role or "a previous role"} to {target_role or job_title}.
Candidate CV summary: {cv_str}
Job requirements (excerpt): {jd_snippet}

Your interview style:
- Professional, direct, realistic. Not a cheerleader.
- Ask ONE question per turn. Make it specific to THIS role and THIS candidate's background.
- After each answer: 1 sentence of genuine acknowledgement (not sycophantic), then your next question.
- Vary question types: behavioural (STAR), technical depth, motivation, gap acknowledgement.
- Build on what the candidate said — follow-up naturally when an answer is vague.

{"This is the FINAL exchange. After their answer, give a brief professional closing (2-3 sentences): thank them, mention next steps, end the interview naturally. Do NOT ask another question." if remaining <= 1 else f"You have {remaining} exchanges remaining after this one."}

Respond only with your spoken words as the interviewer. No meta-commentary."""

    messages_for_api = [{"role": "system", "content": system_prompt}] + history

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=0.7,
            max_tokens=300,
        )
        response_text = (resp.choices[0].message.content or "").strip()
        is_done = remaining <= 1
        return {
            "response": response_text,
            "is_complete": is_done,
            "exchange_num": exchange_num,
            "source": "llm",
        }
    except Exception as exc:
        return {
            "response": _heuristic_closes[0],
            "is_complete": True,
            "exchange_num": exchange_num,
            "source": f"heuristic (error: {repr(exc)[:60]})",
        }


def generate_mock_interview_report(
    history: List[Dict[str, str]],
    job_title: str,
    company: str = "",
    target_role: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Analyse the full mock interview conversation and return a structured
    performance report across 5 dimensions.

    Returns:
        {
          "overall_score": int,           0-100
          "hire_recommendation": str,     "Strong Yes" | "Yes" | "Conditional" | "No"
          "hire_probability_pct": int,    0-100
          "dimension_scores": dict,       communication · technical_depth · pivot_narrative
                                          culture_fit · star_structure
          "strongest_moment": str,        quote + analysis of best answer
          "weakest_moment": str,          quote + what to improve
          "top_improvements": List[str],  3 specific actions
          "sample_rewrite": str,          improved version of weakest answer
          "one_line_verdict": str,        ≤120 chars
          "source": str,
        }
    """
    _heuristic_report: Dict[str, Any] = {
        "overall_score": 65,
        "hire_recommendation": "Conditional",
        "hire_probability_pct": 55,
        "dimension_scores": {
            "communication": 65, "technical_depth": 60,
            "pivot_narrative": 70, "culture_fit": 65, "star_structure": 60,
        },
        "strongest_moment": "Candidate showed self-awareness about their career pivot.",
        "weakest_moment": "Technical depth answers lacked specific examples.",
        "top_improvements": [
            "Add quantified outcomes to every STAR answer",
            "Deepen technical vocabulary for the target role",
            "Clarify your 90-day plan for the pivot",
        ],
        "sample_rewrite": "Consider framing your answer with: 'In my previous role as X, I encountered [situation]...'",
        "one_line_verdict": "Solid pivot narrative — strengthen technical depth to convert to Yes.",
        "source": "heuristic",
    }

    if not prefer_online or len(history) < 2:
        return _heuristic_report

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _heuristic_report

    # Build readable transcript
    transcript_lines = []
    for msg in history:
        role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        transcript_lines.append(f"{role}: {msg['content']}")
    transcript = "\n\n".join(transcript_lines)

    company_str = f" at {company}" if company else ""
    prompt = f"""You analysed a mock interview for a {job_title} role{company_str}.
The candidate is transitioning to {target_role or job_title}.

FULL TRANSCRIPT:
{transcript[:4000]}

Score the candidate's performance across 5 dimensions (0-100 each):
1. communication      — clarity, confidence, structure
2. technical_depth    — domain knowledge and role-specific language
3. pivot_narrative    — how compellingly they explained the career change
4. culture_fit        — enthusiasm, professionalism, alignment signals
5. star_structure     — use of Situation/Task/Action/Result in answers

overall_score = communication×0.25 + technical_depth×0.25 + pivot_narrative×0.20 + culture_fit×0.15 + star_structure×0.15

Respond ONLY with valid JSON:
{{
  "overall_score": 72,
  "hire_recommendation": "Yes",
  "hire_probability_pct": 68,
  "dimension_scores": {{
    "communication": 78, "technical_depth": 65, "pivot_narrative": 80, "culture_fit": 72, "star_structure": 60
  }},
  "strongest_moment": "Quote the specific answer that worked best and explain why in 1-2 sentences.",
  "weakest_moment": "Quote the weakest answer and state specifically what was missing.",
  "top_improvements": [
    "Specific improvement 1",
    "Specific improvement 2",
    "Specific improvement 3"
  ],
  "sample_rewrite": "Here is how the weakest answer could have been structured: [rewrite in 100-150 words preserving their real experience]",
  "one_line_verdict": "Strong pivot story, technical depth needs work — add one metric per STAR answer to convert to Strong Yes."
}}

hire_recommendation must be: "Strong Yes" | "Yes" | "Conditional" | "No"
hire_probability_pct: 0-100, consistent with hire_recommendation (Strong Yes ≥ 80, Yes 65-79, Conditional 40-64, No < 40)
one_line_verdict ≤ 120 characters."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "overall_score": int(data.get("overall_score", 65)),
            "hire_recommendation": str(data.get("hire_recommendation", "Conditional")),
            "hire_probability_pct": int(data.get("hire_probability_pct", 55)),
            "dimension_scores": {k: int(v) for k, v in data.get("dimension_scores", {}).items()},
            "strongest_moment": str(data.get("strongest_moment", ""))[:500],
            "weakest_moment": str(data.get("weakest_moment", ""))[:500],
            "top_improvements": [str(x) for x in data.get("top_improvements", [])[:3]],
            "sample_rewrite": str(data.get("sample_rewrite", ""))[:1000],
            "one_line_verdict": str(data.get("one_line_verdict", ""))[:130],
            "source": "llm",
        }
    except Exception as exc:
        result = dict(_heuristic_report)
        result["source"] = f"heuristic (error: {repr(exc)[:60]})"
        return result
