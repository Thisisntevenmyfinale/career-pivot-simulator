from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import pandas as pd

from src.llm_pivot_strategy import (
    retrieve_target_evidence,
    top_missing_skills,
    top_transferable_skills,
)


# ============================================================
# Small helpers
# ============================================================
def _get_api_key_optional() -> str:
    """Return an OpenAI API key from env or Streamlit secrets."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key

    try:
        import streamlit as st

        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _sanitize_text(s: str, max_len: int = 300) -> str:
    """Normalize whitespace and keep text compact."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_len]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    if not text or not text.strip():
        raise ValueError("Empty LLM output.")

    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        snippet = raw[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from LLM output.")


# ============================================================
# Offline fallback
# ============================================================
def _offline_projects(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
) -> Dict[str, Any]:
    """Deterministic fallback project ideas when no LLM is available."""
    all_skills = list(dict.fromkeys(missing_skills + transfer_skills))

    projects = []
    for i in range(min(3, max(1, len(all_skills)))):
        chosen = all_skills[i : i + 3] if len(all_skills) >= i + 3 else all_skills[:3]
        title = f"{target_role} Portfolio Project {i+1}"
        projects.append(
            {
                "title": title,
                "summary": f"Build a realistic artifact that demonstrates {', '.join(chosen) if chosen else 'target-role skills'}.",
                "skills": chosen,
                "difficulty": "intermediate" if i > 0 else "foundation",
                "estimated_hours": 20 + (i * 15),
                "portfolio_signal": 0.55 + (0.1 * i),
            }
        )

    return {"projects": projects}


# ============================================================
# Validation / scoring
# ============================================================
def _validate_projects_json(obj: Dict[str, Any], allowed_skills: List[str]) -> tuple[bool, List[str], List[Dict[str, Any]]]:
    """Validate and sanitize project JSON."""
    errors: List[str] = []

    if "projects" not in obj:
        return False, ["Missing top-level key: projects"], []

    cleaned: List[Dict[str, Any]] = []
    for item in obj.get("projects", []):
        title = _sanitize_text(item.get("title", ""), 120)
        summary = _sanitize_text(item.get("summary", ""), 260)
        difficulty = str(item.get("difficulty", "")).strip().lower()
        if difficulty not in {"foundation", "intermediate", "advanced"}:
            difficulty = "intermediate"

        try:
            estimated_hours = int(float(item.get("estimated_hours", 20)))
        except Exception:
            estimated_hours = 20
        estimated_hours = max(4, min(estimated_hours, 200))

        try:
            portfolio_signal = float(item.get("portfolio_signal", 0.5))
        except Exception:
            portfolio_signal = 0.5
        portfolio_signal = max(0.0, min(portfolio_signal, 1.0))

        skills = [str(x) for x in item.get("skills", []) if str(x) in allowed_skills]

        if not title:
            errors.append("Project missing title.")
            continue
        if not skills:
            errors.append(f"Project '{title}' has no valid skills.")
            continue

        cleaned.append(
            {
                "title": title,
                "summary": summary,
                "skills": skills,
                "difficulty": difficulty,
                "estimated_hours": estimated_hours,
                "portfolio_signal": portfolio_signal,
            }
        )

    if not cleaned:
        errors.append("No valid projects survived validation.")

    return len(errors) == 0, errors, cleaned


def _projects_to_dataframe(projects: List[Dict[str, Any]], missing_skills: List[str]) -> pd.DataFrame:
    """Convert cleaned projects into a DataFrame with coverage metrics."""
    rows: List[Dict[str, Any]] = []

    missing_set = set(missing_skills)
    for p in projects:
        skills = list(dict.fromkeys([str(x) for x in p.get("skills", [])]))
        covered_missing = len([s for s in skills if s in missing_set])
        coverage_ratio = covered_missing / max(1, len(missing_set))

        rows.append(
            {
                "title": str(p.get("title", "")),
                "difficulty": str(p.get("difficulty", "")),
                "estimated_hours": int(p.get("estimated_hours", 0)),
                "portfolio_signal": float(p.get("portfolio_signal", 0.0)),
                "skills": ", ".join(skills),
                "covered_missing_skills": int(covered_missing),
                "missing_skill_coverage": float(coverage_ratio),
                "summary": str(p.get("summary", "")),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["missing_skill_coverage", "portfolio_signal", "estimated_hours"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return df


# ============================================================
# Prompt builders
# ============================================================
def _portfolio_prompt(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
    evidence_lines: str,
) -> str:
    """Build the portfolio-project generation prompt."""
    return f"""
You are designing portfolio projects for a career pivot.

Current role: {current_role}
Target role: {target_role}

Missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Transferable skills:
{json.dumps(transfer_skills, ensure_ascii=False)}

Retrieved target-role evidence:
{evidence_lines}

Return ONLY valid JSON with this exact shape:
{{
  "projects": [
    {{
      "title": "...",
      "summary": "...",
      "skills": ["..."],
      "difficulty": "foundation|intermediate|advanced",
      "estimated_hours": 30,
      "portfolio_signal": 0.75
    }}
  ]
}}

Rules:
- Return 3 projects.
- Use only the listed skills.
- Each project must cover at least 2 skills.
- Make projects realistic for someone pivoting into the target role.
- Portfolio_signal must be between 0.0 and 1.0.
""".strip()


def _repair_prompt(original_text: str, errors: List[str]) -> str:
    """Repair invalid project JSON."""
    return f"""
The previous JSON was invalid.

Validation errors:
{json.dumps(errors, ensure_ascii=False)}

Previous output:
{original_text}

Return ONLY corrected JSON.
""".strip()


def _portfolio_markdown(projects_df: pd.DataFrame, current_role: str, target_role: str) -> str:
    """Build a compact markdown summary for the UI."""
    if projects_df.empty:
        return (
            "## Portfolio Project Ideas\n\n"
            f"**Pivot:** {current_role} → {target_role}\n\n"
            "No valid project ideas were generated."
        )

    lines = [
        "## Portfolio Project Ideas",
        "",
        f"**Pivot:** {current_role} → {target_role}",
        "",
    ]

    for i, row in projects_df.head(3).iterrows():
        lines.extend(
            [
                f"### {i+1}. {row['title']}",
                f"- **Difficulty:** {row['difficulty'].title()}",
                f"- **Estimated hours:** {int(row['estimated_hours'])}",
                f"- **Skills:** {row['skills']}",
                f"- **Coverage of missing skills:** {float(row['missing_skill_coverage']):.0%}",
                f"- **Summary:** {row['summary']}",
                "",
            ]
        )

    return "\n".join(lines)


# ============================================================
# Public orchestration API
# ============================================================
def generate_portfolio_projects_bundle(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    data_dir: str = "data/onet_raw",
) -> Dict[str, Any]:
    """Generate structured project ideas plus a Python-ready DataFrame."""
    missing_skills = top_missing_skills(gap_df, max_n=6)
    transfer_skills = top_transferable_skills(gap_df, max_n=5)
    allowed_skills = list(dict.fromkeys(missing_skills + transfer_skills))

    evidence = retrieve_target_evidence(
        target_role,
        candidate_terms=allowed_skills + [target_role],
        data_dir=data_dir,
        max_tasks=6,
        max_tech=4,
        max_activities=4,
    )
    evidence_lines = "\n".join([f"{x.evidence_id} | {x.kind} | {x.text}" for x in evidence.items])

    trace: Dict[str, Any] = {
        "mode": "offline",
        "planner_raw": "",
        "repair_raw": "",
        "validation_errors": [],
        "missing_skills": missing_skills,
        "transfer_skills": transfer_skills,
        "allowed_skills": allowed_skills,
        "retrieved_evidence": [{"evidence_id": x.evidence_id, "kind": x.kind, "text": x.text} for x in evidence.items],
        "cleaned_projects": [],
    }

    offline_obj = _offline_projects(
        current_role=current_role,
        target_role=target_role,
        missing_skills=missing_skills,
        transfer_skills=transfer_skills,
    )
    _, _, offline_cleaned = _validate_projects_json(offline_obj, allowed_skills=allowed_skills)
    offline_df = _projects_to_dataframe(offline_cleaned, missing_skills)
    offline_md = _portfolio_markdown(offline_df, current_role, target_role)

    if not prefer_online:
        trace["cleaned_projects"] = offline_cleaned
        return {
            "markdown": offline_md,
            "source": "Offline projects",
            "projects_df": offline_df,
            "trace": trace,
        }

    api_key = _get_api_key_optional()
    if not api_key:
        trace["cleaned_projects"] = offline_cleaned
        return {
            "markdown": offline_md,
            "source": "Offline projects",
            "projects_df": offline_df,
            "trace": trace,
        }

    try:
        from openai import OpenAI
    except Exception:
        trace["cleaned_projects"] = offline_cleaned
        return {
            "markdown": offline_md,
            "source": "Offline projects",
            "projects_df": offline_df,
            "trace": trace,
        }

    try:
        client = OpenAI(api_key=api_key)

        planner_resp = client.responses.create(
            model=model,
            instructions="Return strictly valid JSON and nothing else.",
            input=_portfolio_prompt(
                current_role=current_role,
                target_role=target_role,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
                evidence_lines=evidence_lines,
            ),
        )
        planner_text = (planner_resp.output_text or "").strip()
        trace["planner_raw"] = planner_text

        planner_obj = _extract_json_object(planner_text)
        valid, errors, cleaned_projects = _validate_projects_json(planner_obj, allowed_skills=allowed_skills)
        trace["validation_errors"] = list(errors)

        if not valid:
            repair_resp = client.responses.create(
                model=model,
                instructions="Return strictly valid JSON and nothing else.",
                input=_repair_prompt(planner_text, errors),
            )
            repair_text = (repair_resp.output_text or "").strip()
            trace["repair_raw"] = repair_text

            repaired_obj = _extract_json_object(repair_text)
            valid, errors, cleaned_projects = _validate_projects_json(repaired_obj, allowed_skills=allowed_skills)
            trace["validation_errors"] = list(errors)

            if not valid:
                trace["cleaned_projects"] = offline_cleaned
                return {
                    "markdown": offline_md,
                    "source": "Offline projects",
                    "projects_df": offline_df,
                    "trace": trace,
                }

        trace["mode"] = "online"
        trace["cleaned_projects"] = cleaned_projects

        projects_df = _projects_to_dataframe(cleaned_projects, missing_skills)
        markdown = "🤖 " + _portfolio_markdown(projects_df, current_role, target_role)

        return {
            "markdown": markdown,
            "source": "OpenAI structured portfolio",
            "projects_df": projects_df,
            "trace": trace,
        }

    except Exception as e:
        trace["validation_errors"] = [f"Exception: {repr(e)}"]
        trace["cleaned_projects"] = offline_cleaned
        return {
            "markdown": offline_md,
            "source": "Offline projects",
            "projects_df": offline_df,
            "trace": trace,
        }