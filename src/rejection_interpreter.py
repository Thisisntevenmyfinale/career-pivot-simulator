"""
Rejection Interpreter
======================
Transforms every rejection into a specific, actionable diagnosis.

Most job seekers treat rejections as opaque "no"s.
The Rejection Interpreter forces each rejection to mean something:
  - Which of the 6 rejection types is this?
  - What specifically failed?
  - What is the ONE thing to fix before the next application?

Architecture:
  gpt-4o-mini (single pass — classification is a structured task,
  not prose generation. Cheaper and fast enough for real-time use.)
  Input:  raw feedback text (often empty), stage reached, job context
  Output: typed rejection + root cause + week plan + OPS impact

Rejection taxonomy (grounded in hiring funnel research):
  ats_filter      — never reached a human
  recruiter_screen — human saw it, didn't pass to HM
  hiring_freeze   — timing / headcount issue (not your fault)
  skills_gap      — concrete skill mismatch surfaced
  culture_signal  — soft/fit rejection after interview
  competition     — strong field, you were runner-up
  unknown         — feedback too vague to classify
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


REJECTION_TYPES = {
    "ats_filter": {
        "label":    "ATS Filter",
        "color":    "#7A2A8A",
        "icon":     "bot",
        "summary":  "Your application was rejected before a human saw it.",
        "fix":      "ATS keyword alignment + formatting",
    },
    "recruiter_screen": {
        "label":    "Recruiter Screen",
        "color":    "#0A66C2",
        "icon":     "user",
        "summary":  "A recruiter reviewed you but didn't pass you to the hiring manager.",
        "fix":      "Pivot narrative + LinkedIn headline rewrite",
    },
    "hiring_freeze": {
        "label":    "Hiring Freeze",
        "color":    "#5F6B7A",
        "icon":     "pause",
        "summary":  "The role was paused — not a reflection of your candidacy.",
        "fix":      "Stay in touch; apply elsewhere immediately",
    },
    "skills_gap": {
        "label":    "Skills Gap",
        "color":    "#A05A00",
        "icon":     "zap",
        "summary":  "A specific skill mismatch was cited or implied.",
        "fix":      "Proof-of-Skill project targeting the specific gap",
    },
    "culture_signal": {
        "label":    "Culture / Fit",
        "color":    "#B24020",
        "icon":     "heart",
        "summary":  "You reached an interview but were rejected on fit signals.",
        "fix":      "Improve closing + culture research + ask for specific feedback",
    },
    "competition": {
        "label":    "Strong Field",
        "color":    "#057642",
        "icon":     "trophy",
        "summary":  "You were competitive but outpaced by another candidate.",
        "fix":      "Stay in their talent pipeline; ask to be considered for future openings",
    },
    "unknown": {
        "label":    "Unknown",
        "color":    "#5F6B7A",
        "icon":     "help-circle",
        "summary":  "Insufficient signal to classify.",
        "fix":      "Email to request specific feedback",
    },
}


def interpret_rejection(
    oai_key: str,
    *,
    feedback_text: str,
    stage: str,
    job_title: str,
    company: str,
    cv_profile: Optional[Dict] = None,
    pivot_dna: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Classify a rejection and produce an actionable diagnosis.

    Returns:
      rejection_type: str (one of REJECTION_TYPES keys)
      confidence: int 0-100
      root_cause: str (1-2 sentences, specific)
      immediate_action: str (single most important thing to do today)
      week_plan: list[str] (3-4 actions for this week)
      reframe: str (psychologically honest reframe — not toxic positivity)
      ops_impact: str ("This typically costs −3pts OPS — fixable in 1 week")
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
    except Exception:
        return _fallback_interpretation(stage, feedback_text)

    stage_context = {
        "no_response":    "The candidate never heard back after applying.",
        "viewed_no_call": "The recruiter viewed the profile but didn't reach out.",
        "phone_screen":   "The candidate had a phone/recruiter screen and was rejected.",
        "first_round":    "The candidate completed a first-round interview and was rejected.",
        "final_round":    "The candidate reached the final round and was not selected.",
        "offer":          "An offer was received — no rejection here.",
    }.get(stage, stage)

    background = ""
    if cv_profile:
        skills = (cv_profile.get("top_skills") or [])[:5]
        background += f"Candidate skills: {', '.join(skills)}. "
    if pivot_dna:
        background += f"Pivot argument: {pivot_dna.get('strongest_transferable_argument','')[:120]}. "

    prompt = f"""You are an expert career coach analyzing a job rejection to help the candidate improve.

REJECTION CONTEXT:
- Role: {job_title} at {company}
- Stage reached: {stage_context}
- Feedback received: "{feedback_text or 'No feedback provided'}"
- Candidate background: {background or 'Not provided'}

Classify this rejection and provide a specific, honest diagnosis.

Rejection types to choose from:
- ats_filter: application never reached a human (no-response pattern)
- recruiter_screen: human saw it but pivot/headline didn't sell
- hiring_freeze: role paused, not candidate's fault
- skills_gap: specific skill mismatch cited or clearly implied
- culture_signal: interview rejection on soft/fit grounds
- competition: runner-up in a strong field
- unknown: feedback too vague to classify

Respond ONLY with valid JSON:
{{
  "rejection_type": "one of the 7 types above",
  "confidence": 0-100,
  "root_cause": "1-2 sentences, be specific. What exactly failed? Reference the stage and any feedback.",
  "immediate_action": "single most impactful thing to do today (specific, concrete, actionable)",
  "week_plan": ["action 1", "action 2", "action 3"],
  "reframe": "honest psychological reframe — not 'keep going!' platitudes. What does this data point actually mean for their strategy?",
  "ops_impact": "e.g. 'This pattern typically costs -5pts OPS — fixable in ~2 weeks with X'"
}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        rtype = data.get("rejection_type", "unknown")
        if rtype not in REJECTION_TYPES:
            rtype = "unknown"
        return {
            "rejection_type":    rtype,
            "type_meta":         REJECTION_TYPES[rtype],
            "confidence":        int(data.get("confidence", 60)),
            "root_cause":        str(data.get("root_cause", "")),
            "immediate_action":  str(data.get("immediate_action", "")),
            "week_plan":         [str(x) for x in data.get("week_plan", [])[:4]],
            "reframe":           str(data.get("reframe", "")),
            "ops_impact":        str(data.get("ops_impact", "")),
        }
    except Exception:
        return _fallback_interpretation(stage, feedback_text)


def _fallback_interpretation(stage: str, feedback_text: str) -> Dict[str, Any]:
    """Rule-based fallback when LLM is unavailable."""
    stage_to_type = {
        "no_response":    "ats_filter",
        "viewed_no_call": "recruiter_screen",
        "phone_screen":   "recruiter_screen",
        "first_round":    "skills_gap",
        "final_round":    "competition",
    }
    rtype = stage_to_type.get(stage, "unknown")
    meta  = REJECTION_TYPES[rtype]

    from src.outcome_tracker import STAGE_DIAGNOSIS
    diag = STAGE_DIAGNOSIS.get(stage, STAGE_DIAGNOSIS["no_response"])

    return {
        "rejection_type":   rtype,
        "type_meta":        meta,
        "confidence":       45,
        "root_cause":       diag["root_cause"],
        "immediate_action": diag["actions"][0] if diag["actions"] else "Review your application materials",
        "week_plan":        diag["actions"][:3],
        "reframe":          "Every rejection narrows the field. This one tells you where the leak is.",
        "ops_impact":       f"This stage pattern typically costs -4pts OPS — address {diag['priority_fix']}",
    }
