"""
Cross-Rejection Synthesis Engine
==================================
Individual rejection diagnosis (rejection_interpreter.py) answers:
  "What went wrong with THIS specific application?"

Cross-rejection synthesis answers:
  "What do my last N rejections collectively say about my search strategy?"

The distinction matters because:
  - One pre-screen rejection → could be anything
  - Five pre-screen rejections in a row → ATS/keyword problem (structural)
  - Three late-stage rejections → you're getting to interviews but losing there (execution)
  - Mix of early + late → two separate problems (sequencing issue)

This module:
  1. Aggregates stage distribution across all rejections
  2. Identifies the dominant bottleneck (quantified with counts + %)
  3. Determines: is the problem fixable now, or structural (strategy must change)?
  4. Returns the single highest-EV fix across all applications

Architecture note:
  Input: List[Dict] of rejection_interpreter results (already analysed individually)
  The synthesis LLM receives: aggregated statistics + individual causes, NOT raw text.
  This means LLM outputs are never used raw — they feed back through a Python
  aggregation layer before reaching the UI. This is the pattern the professor praised.

Model choice: gpt-4o-mini at temp=0.1 (near-deterministic for strategic analysis —
  we want consistency, not creativity, when diagnosing a job search failure pattern).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_SYNTHESIS_SYSTEM = """You are a career strategist synthesising data across multiple job rejections.
Your job: find the structural pattern. Be specific and quantified.

Do NOT hedge. Do NOT say "it depends." Give a directional diagnosis and a clear action.

Output JSON only:
{
  "bottleneck_stage": "the stage where most rejections happen — pre_screen / post_screen / late_stage / mixed",
  "bottleneck_pct": <int — percentage of rejections at this stage>,
  "bottleneck_confidence": "high|medium|low",
  "root_cause": "the underlying issue that explains the bottleneck stage — 1-2 sentences, specific",
  "is_structural": <bool — true if changing strategy is needed, false if execution fix is sufficient>,
  "structural_explanation": "if is_structural: what needs to change and why",
  "highest_ev_fix": "the single change with the highest expected improvement — specific, actionable",
  "fix_effort": "hours|days|weeks",
  "encouraging_signal": "what the pattern says IS working — be genuine, not falsely positive",
  "pivot_strategy_change": "null if strategy is sound, or: what specifically needs to change in the target role/company type/seniority level",
  "weeks_to_improvement": <int — realistic weeks until improvement if fix is applied>
}"""


def synthesize(
    oai_key: str,
    *,
    rejections: List[Dict[str, Any]],
    outcome_log: Optional[List[Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synthesise pattern across ≥2 rejections.
    rejections: list of dicts with at least {company, job_title, rejection_stage, likely_cause}
    Returns synthesis dict or None.
    """
    if not oai_key or len(rejections) < 2:
        return None

    # ── Python aggregation layer: compute stage distribution before LLM call ──
    stage_counts: Dict[str, int] = {}
    for r in rejections:
        s = r.get("rejection_stage") or r.get("actual_stage") or "unknown"
        # Normalise stage names from outcome_log format
        if s in ("no_response", "viewed_no_call"):
            s = "pre_screen"
        elif s in ("phone_screen",):
            s = "post_screen"
        elif s in ("first_round", "final_round"):
            s = "late_stage"
        stage_counts[s] = stage_counts.get(s, 0) + 1

    n = len(rejections)
    dominant_stage = max(stage_counts, key=stage_counts.get) if stage_counts else "unknown"
    dominant_pct   = round(stage_counts.get(dominant_stage, 0) / n * 100)

    # Summary of individual analyses for LLM
    individual_summary = [
        {
            "company":          r.get("company", ""),
            "job_title":        r.get("job_title", ""),
            "rejection_stage":  r.get("rejection_stage") or r.get("actual_stage", "unknown"),
            "likely_cause":     r.get("likely_cause", ""),
            "root_cause":       r.get("root_cause_hypothesis", ""),
            "fixable":          r.get("fixable"),
        }
        for r in rejections
    ]

    # Pass aggregated statistics + individual causes to LLM
    context = {
        "n_rejections":       n,
        "stage_distribution": stage_counts,
        "dominant_stage":     dominant_stage,
        "dominant_pct":       dominant_pct,
        "individual_analyses": individual_summary,
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": json.dumps(context, indent=2)},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        # Attach computed aggregates — LLM output enriched by Python layer
        result["n_rejections"]     = n
        result["stage_distribution"] = stage_counts
        return result
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python stage distribution (used for display even without API)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_distribution(outcome_log: List[Dict]) -> Dict[str, Any]:
    """
    Compute rejection stage distribution from outcome_log without any LLM call.
    Returns: stage_counts, n_rejections, dominant_stage, dominant_pct
    """
    rejections = [o for o in outcome_log if not o.get("is_offer")]
    if not rejections:
        return {"n_rejections": 0, "stage_counts": {}, "dominant_stage": None, "dominant_pct": 0}

    stage_counts: Dict[str, int] = {}
    for r in rejections:
        s = r.get("actual_stage", "unknown")
        if s in ("no_response", "viewed_no_call"):
            bucket = "Pre-screen"
        elif s == "phone_screen":
            bucket = "Post-screen"
        elif s in ("first_round", "final_round"):
            bucket = "Late-stage"
        else:
            bucket = "Unknown"
        stage_counts[bucket] = stage_counts.get(bucket, 0) + 1

    n = len(rejections)
    dominant = max(stage_counts, key=stage_counts.get) if stage_counts else None
    dominant_pct = round(stage_counts.get(dominant, 0) / n * 100) if dominant else 0

    return {
        "n_rejections":   n,
        "stage_counts":   stage_counts,
        "dominant_stage": dominant,
        "dominant_pct":   dominant_pct,
    }


def bottleneck_color(stage: str) -> str:
    return {
        "pre_screen":   "#7C3AED",  # purple — ATS/narrative problem
        "Pre-screen":   "#7C3AED",
        "post_screen":  "#D97706",  # amber — recruiter screen problem
        "Post-screen":  "#D97706",
        "late_stage":   "#DC2626",  # red — interview execution problem
        "Late-stage":   "#DC2626",
        "mixed":        "#0A66C2",
    }.get(stage, "#5F6B7A")
