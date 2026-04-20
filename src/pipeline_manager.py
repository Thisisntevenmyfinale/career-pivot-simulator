"""
Pipeline Manager
================
Treats a job search like a sales pipeline — because that's what it is.
You are the product. The job market is your TAM. Each application is a lead.

Provides:
1. Pipeline data model (PipelineJob) — tracked across the session / importable via JSON
2. Conversion analytics — where are you losing candidates?
3. Rejection intelligence — LLM pattern analysis over your rejections
4. Search velocity benchmarks — are you on pace for your target timeline?
5. Pipeline diagnosis — one clear "what to fix next" recommendation

This is the flywheel: more data → better diagnosis → smarter next action → better outcomes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

APPLICATION_STATUSES = [
    "applied",
    "viewed",
    "first_round",
    "final_round",
    "offer",
    "rejected",
    "withdrawn",
]

STATUS_LABELS = {
    "applied":     "Applied",
    "viewed":      "Viewed",
    "first_round": "1st Round",
    "final_round": "Final Round",
    "offer":       "Offer",
    "rejected":    "Rejected",
    "withdrawn":   "Withdrawn",
}

STATUS_COLORS = {
    "applied":     "#5F6B7A",
    "viewed":      "#0A66C2",
    "first_round": "#7A2A8A",
    "final_round": "#A05A00",
    "offer":       "#057642",
    "rejected":    "#B71C1C",
    "withdrawn":   "#999",
}


def create_pipeline_job(
    title: str,
    company: str,
    location: str = "",
    job_description: str = "",
    ats_score: Optional[int] = None,
    hire_prob: Optional[int] = None,
    cover_letter: str = "",
    source: str = "manual",
    apply_link: str = "",
) -> Dict[str, Any]:
    """Create a new pipeline job entry."""
    return {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "company": company,
        "location": location,
        "status": "applied",
        "date_added": datetime.now().strftime("%Y-%m-%d"),
        "date_updated": datetime.now().strftime("%Y-%m-%d"),
        "ats_score": ats_score,
        "hire_prob": hire_prob,
        "cover_letter": cover_letter,
        "job_description": job_description,
        "source": source,
        "apply_link": apply_link,
        "rejection_stage": "",
        "rejection_notes": "",
        "interview_notes": "",
        "offer_amount": None,
    }


def update_pipeline_job_status(job: Dict[str, Any], new_status: str, notes: str = "") -> Dict[str, Any]:
    """Update a job's status and add notes."""
    job = dict(job)
    job["status"] = new_status
    job["date_updated"] = datetime.now().strftime("%Y-%m-%d")
    if new_status == "rejected":
        job["rejection_stage"] = notes
    elif notes:
        job["interview_notes"] = notes
    return job


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

def compute_pipeline_stats(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute conversion rates, velocity, and stage breakdown.
    All deterministic Python — no LLM needed.
    """
    if not jobs:
        return {
            "total": 0, "active": 0, "rejected": 0, "offers": 0,
            "stage_counts": {}, "response_rate": 0, "interview_rate": 0,
            "offer_rate": 0, "avg_ats_score": 0, "avg_hire_prob": 0,
        }

    total = len(jobs)
    rejected = [j for j in jobs if j["status"] == "rejected"]
    offers = [j for j in jobs if j["status"] == "offer"]
    responded = [j for j in jobs if j["status"] not in ("applied", "withdrawn")]
    interviewed = [j for j in jobs if j["status"] in ("first_round", "final_round", "offer")]
    active = [j for j in jobs if j["status"] not in ("rejected", "withdrawn", "offer")]

    stage_counts: Dict[str, int] = {}
    for j in jobs:
        stage_counts[j["status"]] = stage_counts.get(j["status"], 0) + 1

    ats_scores = [j["ats_score"] for j in jobs if j.get("ats_score")]
    hire_probs = [j["hire_prob"] for j in jobs if j.get("hire_prob")]

    return {
        "total": total,
        "active": len(active),
        "rejected": len(rejected),
        "offers": len(offers),
        "interviewed": len(interviewed),
        "stage_counts": stage_counts,
        "response_rate": int(len(responded) / total * 100) if total else 0,
        "interview_rate": int(len(interviewed) / total * 100) if total else 0,
        "offer_rate": int(len(offers) / total * 100) if total else 0,
        "avg_ats_score": int(sum(ats_scores) / len(ats_scores)) if ats_scores else 0,
        "avg_hire_prob": int(sum(hire_probs) / len(hire_probs)) if hire_probs else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rejection Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def analyze_rejection_patterns(
    rejected_jobs: List[Dict[str, Any]],
    pivot_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    LLM pattern analysis over rejection data.
    Returns: root cause, actionable fixes, predicted next-step.
    """
    _fallback: Dict[str, Any] = {
        "primary_pattern": "Insufficient data — log at least 2 rejections for pattern analysis.",
        "root_cause": "",
        "stage_diagnosis": "",
        "actionable_fixes": ["Continue logging applications and rejections for pattern analysis."],
        "predicted_bottleneck": "",
        "confidence": "low",
        "source": "offline",
    }

    if not rejected_jobs or len(rejected_jobs) < 2:
        return _fallback

    if not prefer_online:
        # Basic offline heuristic
        stages = [j.get("rejection_stage", "unknown") for j in rejected_jobs]
        most_common_stage = max(set(stages), key=stages.count)
        return {
            "primary_pattern": f"Most rejections at stage: {most_common_stage}",
            "root_cause": f"Pattern detected at {most_common_stage} stage — add API key for deeper analysis.",
            "stage_diagnosis": most_common_stage,
            "actionable_fixes": ["Add OpenAI API key for detailed rejection pattern analysis."],
            "predicted_bottleneck": most_common_stage,
            "confidence": "low",
            "source": "offline_heuristic",
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    rejections_text = "\n".join([
        f"- {j['title']} @ {j['company']} | Stage: {j.get('rejection_stage', 'not specified')} | Notes: {j.get('rejection_notes', 'none')}"
        for j in rejected_jobs
    ])

    pivot_ctx = ""
    if pivot_profile:
        pivot_ctx = f"\nCandidate pivot: {pivot_profile.get('current_role','?')} → {pivot_profile.get('target_role','?')}, {pivot_profile.get('years_exp',0):.0f} yrs exp."

    prompt = f"""You are a job search strategist analyzing rejection patterns to diagnose what's going wrong.

REJECTIONS ({len(rejected_jobs)} logged):
{rejections_text}
{pivot_ctx}

Diagnose the pattern. Be specific and direct — not generic. This person needs to know exactly what to fix.

Focus on:
- At which stage are rejections happening? (ATS screening / first round / final round / ghost)
- What is the most likely root cause?
- What concrete actions would fix the bottleneck?

Respond ONLY with valid JSON:
{{
  "primary_pattern": "5 of 7 rejections occurred at the ATS/no-response stage",
  "root_cause": "ATS scores averaging below 60 suggest keyword gaps for technical roles. Career pivot from Marketing → Product means JDs use 'roadmap', 'SQL', 'A/B testing' — terms not yet in your CV.",
  "stage_diagnosis": "pre-human-review",
  "actionable_fixes": [
    "Run ATS scan on every application before submitting — target 75+",
    "Add a 'Technical Skills' section to CV with: SQL, Jira, Figma, A/B testing, roadmap planning",
    "Your cover letter should lead with a product outcome metric — not a career narrative"
  ],
  "predicted_bottleneck": "ATS keyword coverage is blocking all human review",
  "confidence": "high"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=600,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def generate_pipeline_diagnosis(
    jobs: List[Dict[str, Any]],
    current_role: str = "",
    target_role: str = "",
    weeks_searching: int = 0,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    One clear, actionable diagnosis of the entire pipeline.
    Returns the single highest-leverage action to take next.
    """
    stats = compute_pipeline_stats(jobs)

    _fallback: Dict[str, Any] = {
        "overall_health": "unknown",
        "health_score": 0,
        "top_bottleneck": "Log more applications to generate your pipeline diagnosis.",
        "highest_leverage_action": "Continue applying and tracking outcomes.",
        "benchmark_context": "",
        "week_estimate": "4-8 weeks",
        "source": "offline",
    }

    if not jobs or not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    prompt = f"""You are a job search strategist. Diagnose this candidate's pipeline and give the single most important action.

CANDIDATE: {current_role or "unknown"} → {target_role or "unknown"}
WEEKS SEARCHING: {weeks_searching or "unknown"}

PIPELINE STATS:
- Total applications: {stats['total']}
- Response rate: {stats['response_rate']}%
- Interview rate: {stats['interview_rate']}%
- Offer rate: {stats['offer_rate']}%
- Avg ATS score: {stats['avg_ats_score']}/100
- Avg hire probability: {stats['avg_hire_prob']}%
- Stage breakdown: {json.dumps(stats['stage_counts'])}

BENCHMARK (typical career pivot):
- Response rate: 20-35% is healthy
- Interview rate: 10-20% is healthy
- 30-50 applications typical for a pivot role
- Average time: 3-6 months for significant pivot

Diagnose exactly what's wrong (or right) and give the single highest-leverage action.

Respond ONLY with valid JSON:
{{
  "overall_health": "concerning",
  "health_score": 35,
  "top_bottleneck": "ATS screening — 87% of applications get no response, well below the 20-35% healthy benchmark",
  "highest_leverage_action": "Run ATS scan on your last 5 applications and add the missing critical keywords before reapplying or applying to similar roles. A 10-point ATS improvement historically doubles response rates.",
  "benchmark_context": "At {weeks_searching} weeks and {stats['total']} applications, you should have had 3-5 interviews by now. The 4% response rate suggests a systematic application quality issue, not bad luck.",
  "week_estimate": "6-10 weeks to first offer at current pace"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=500,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Serialization (for save/load across sessions)
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_to_json(jobs: List[Dict[str, Any]]) -> str:
    """Serialize pipeline to JSON string for download."""
    export = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "jobs": jobs,
    }
    return json.dumps(export, indent=2, default=str)


def pipeline_from_json(json_str: str) -> List[Dict[str, Any]]:
    """Deserialize pipeline from JSON string (from uploaded file)."""
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "jobs" in data:
            return data["jobs"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []
