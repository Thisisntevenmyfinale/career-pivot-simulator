from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Data classes
# ============================================================
@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved piece of target-role evidence."""

    evidence_id: str
    kind: str
    text: str
    source_title: str


@dataclass(frozen=True)
class RetrievedEvidence:
    """Container for retrieved O*NET evidence tied to a target occupation."""

    target_role: str
    soc_codes: List[str]
    items: List[EvidenceItem]
    job_zone: Optional[str]


# ============================================================
# Utility helpers
# ============================================================
def _get_api_key_optional() -> str:
    """Return an OpenAI API key from env or Streamlit secrets if available."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key

    try:
        import streamlit as st

        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _sanitize_text(s: str, max_len: int = 500) -> str:
    """Normalize whitespace and keep text bounded for prompts and UI."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_len]


def _sanitize_role_name(s: str) -> str:
    """Short role-name sanitizer for prompts and titles."""
    return _sanitize_text(s, max_len=120)


def _normalize_title(s: str) -> str:
    """Normalize job titles for fuzzy matching."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s/&-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV file if present, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except Exception:
        return pd.DataFrame()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model text, even if wrapped in prose."""
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


def _score_overlap(text: str, query_terms: List[str]) -> int:
    """Simple lexical retrieval score used to prioritize evidence snippets."""
    t = _normalize_title(text)
    score = 0
    for q in query_terms:
        qn = _normalize_title(q)
        if not qn:
            continue
        if qn in t:
            score += 3
        else:
            q_tokens = [tok for tok in qn.split() if len(tok) >= 3]
            score += sum(1 for tok in q_tokens if tok in t)
    return score


# ============================================================
# O*NET retrieval
# ============================================================
@lru_cache(maxsize=1)
def _load_onet_tables(data_dir: str = "data/onet_raw") -> Dict[str, pd.DataFrame]:
    """Load a lightweight subset of O*NET tables used at runtime."""
    root = Path(data_dir)

    return {
        "occupation_data": _safe_read_tsv(root / "Occupation Data.txt"),
        "task_statements": _safe_read_tsv(root / "Task Statements.txt"),
        "technology_skills": _safe_read_tsv(root / "Technology Skills.txt"),
        "work_activities": _safe_read_tsv(root / "Work Activities.txt"),
        "job_zones": _safe_read_tsv(root / "Job Zones.txt"),
    }


def _resolve_soc_codes_for_title(role_title: str, data_dir: str = "data/onet_raw") -> List[str]:
    """Resolve an occupation title to one or more O*NET SOC codes."""
    tables = _load_onet_tables(data_dir)
    occ = tables["occupation_data"]
    if occ.empty or "Title" not in occ.columns or "O*NET-SOC Code" not in occ.columns:
        return []

    title_norm = _normalize_title(role_title)
    occ = occ.copy()
    occ["__title_norm__"] = occ["Title"].astype(str).map(_normalize_title)

    exact = (
        occ.loc[occ["__title_norm__"] == title_norm, "O*NET-SOC Code"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if exact:
        return exact

    partial = occ.loc[
        occ["__title_norm__"].str.contains(re.escape(title_norm), na=False),
        "O*NET-SOC Code",
    ]
    return partial.dropna().astype(str).unique().tolist()[:5]


def _get_job_zone_text(soc_codes: List[str], data_dir: str = "data/onet_raw") -> Optional[str]:
    """Build a short human-readable job-zone summary."""
    tables = _load_onet_tables(data_dir)
    jz = tables["job_zones"]
    if jz.empty or "O*NET-SOC Code" not in jz.columns:
        return None

    sub = jz[jz["O*NET-SOC Code"].astype(str).isin(soc_codes)].copy()
    if sub.empty:
        return None

    bits: List[str] = []
    cols = set(sub.columns)

    for _, row in sub.head(2).iterrows():
        parts: List[str] = []
        if "Job Zone" in cols:
            parts.append(f"Job Zone {row.get('Job Zone', '')}")
        if "Education" in cols:
            parts.append(f"Education: {row.get('Education', '')}")
        if "Related Experience" in cols:
            parts.append(f"Experience: {row.get('Related Experience', '')}")
        if "Job Training" in cols:
            parts.append(f"Training: {row.get('Job Training', '')}")

        joined = "; ".join([_sanitize_text(x, 180) for x in parts if str(x).strip()])
        if joined:
            bits.append(joined)

    return " | ".join(bits) if bits else None


def retrieve_target_evidence(
    role_title: str,
    candidate_terms: List[str],
    *,
    data_dir: str = "data/onet_raw",
    max_tasks: int = 8,
    max_tech: int = 6,
    max_activities: int = 6,
) -> RetrievedEvidence:
    """Retrieve target-role evidence from O*NET tables.

    This is the retrieval stage of the RAG-style workflow.
    """
    tables = _load_onet_tables(data_dir)
    soc_codes = _resolve_soc_codes_for_title(role_title, data_dir=data_dir)

    items: List[EvidenceItem] = []
    next_id = 1

    # ---------------------------
    # Task statements
    # ---------------------------
    tasks = tables["task_statements"]
    if not tasks.empty and "O*NET-SOC Code" in tasks.columns:
        sub = tasks[tasks["O*NET-SOC Code"].astype(str).isin(soc_codes)].copy()
        task_col = "Task" if "Task" in sub.columns else None
        if task_col is not None:
            sub["__text__"] = sub[task_col].astype(str).map(lambda x: _sanitize_text(x, 250))
            sub["__score__"] = sub["__text__"].map(lambda t: _score_overlap(t, candidate_terms))
            sub = sub.sort_values(["__score__", "__text__"], ascending=[False, True]).head(max_tasks)

            for _, row in sub.iterrows():
                items.append(
                    EvidenceItem(
                        evidence_id=f"E{next_id}",
                        kind="task",
                        text=row["__text__"],
                        source_title=role_title,
                    )
                )
                next_id += 1

    # ---------------------------
    # Technology skills
    # ---------------------------
    tech = tables["technology_skills"]
    if not tech.empty and "O*NET-SOC Code" in tech.columns:
        sub = tech[tech["O*NET-SOC Code"].astype(str).isin(soc_codes)].copy()
        possible_cols = [c for c in ["Example", "Commodity Title", "Hot Technology"] if c in sub.columns]
        if possible_cols:

            def row_to_text(row: pd.Series) -> str:
                parts = []
                for c in possible_cols:
                    v = str(row.get(c, "")).strip()
                    if v and v.lower() != "nan":
                        parts.append(v)
                return _sanitize_text(" | ".join(parts), 220)

            sub["__text__"] = sub.apply(row_to_text, axis=1)
            sub = sub[sub["__text__"].astype(str).str.len() > 0].copy()
            if not sub.empty:
                sub["__score__"] = sub["__text__"].map(lambda t: _score_overlap(t, candidate_terms))
                sub = sub.sort_values(["__score__", "__text__"], ascending=[False, True]).head(max_tech)

                for _, row in sub.iterrows():
                    items.append(
                        EvidenceItem(
                            evidence_id=f"E{next_id}",
                            kind="technology",
                            text=row["__text__"],
                            source_title=role_title,
                        )
                    )
                    next_id += 1

    # ---------------------------
    # Work activities
    # ---------------------------
    wa = tables["work_activities"]
    if not wa.empty and "O*NET-SOC Code" in wa.columns:
        sub = wa[wa["O*NET-SOC Code"].astype(str).isin(soc_codes)].copy()
        if "Scale ID" in sub.columns:
            sub = sub[sub["Scale ID"].astype(str) == "IM"].copy()
        if "Element Name" in sub.columns:
            sub["__dv__"] = pd.to_numeric(sub.get("Data Value", 0.0), errors="coerce").fillna(0.0)
            sub["__text__"] = sub["Element Name"].astype(str).map(lambda x: _sanitize_text(x, 180))
            sub["__score__"] = sub["__text__"].map(lambda t: _score_overlap(t, candidate_terms))
            sub = sub.sort_values(["__score__", "__dv__", "__text__"], ascending=[False, False, True]).head(max_activities)

            for _, row in sub.iterrows():
                items.append(
                    EvidenceItem(
                        evidence_id=f"E{next_id}",
                        kind="work_activity",
                        text=row["__text__"],
                        source_title=role_title,
                    )
                )
                next_id += 1

    # ---------------------------
    # Job zone
    # ---------------------------
    job_zone = _get_job_zone_text(soc_codes, data_dir=data_dir)
    if job_zone:
        items.append(
            EvidenceItem(
                evidence_id=f"E{next_id}",
                kind="job_zone",
                text=job_zone,
                source_title=role_title,
            )
        )

    return RetrievedEvidence(
        target_role=role_title,
        soc_codes=soc_codes,
        items=items,
        job_zone=job_zone,
    )


# ============================================================
# Gap helpers
# ============================================================
_PHYSICAL_SENSORY_PATTERNS = [
    r"\b(static|dynamic|explosive)\s+strength\b",
    r"\bstamina\b",
    r"\bextent\s+flexibility\b",
    r"\bfinger\s+dexterity\b",
    r"\bmanual\s+dexterity\b",
    r"\btrunk\s+strength\b",
    r"\bgross\s+body\s+(coordination|equilibrium)\b",
    r"\bmultilimb\s+coordination\b",
    r"\barm[-\s]?hand\s+steadiness\b",
    r"\brate\s+control\b",
    r"\bresponse\s+orientation\b",
    r"\bcontrol\s+precision\b",
    r"\b(far|near)\s+vision\b",
    r"\bdepth\s+perception\b",
    r"\bperipheral\s+vision\b",
    r"\bvisual\s+color\s+discrimination\b",
    r"\bglare\sensitivity\b",
    r"\bhearing\s+sensitivity\b",
    r"\bsound\s+localization\b",
    r"\bauditory\s+attention\b",
    r"\bspeech\s+recognition\b",
    r"\bspeech\s+clarity\b",
]

_EXCLUSION_RE = re.compile("(" + "|".join(_PHYSICAL_SENSORY_PATTERNS) + ")", flags=re.IGNORECASE)


def top_missing_skills(gap_df: pd.DataFrame, max_n: int = 6) -> List[str]:
    """Return the highest-priority missing skills for prompting and UI."""
    df = gap_df.copy()
    for col in ["gap", "current_importance", "target_importance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["skill"] = df["skill"].astype(str)

    df = df[df["gap"] > 0].copy()
    df = df[~df["skill"].str.contains(_EXCLUSION_RE, na=False)].copy()
    if df.empty:
        return []

    df["priority"] = df["gap"] * df["target_importance"]
    df = df.sort_values(["priority", "gap", "target_importance"], ascending=False)
    return df["skill"].head(int(max_n)).tolist()


def top_transferable_skills(gap_df: pd.DataFrame, max_n: int = 5) -> List[str]:
    """Return strongest overlap skills between current and target profiles."""
    df = gap_df.copy()
    for col in ["current_importance", "target_importance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["skill"] = df["skill"].astype(str)
    df["overlap"] = df[["current_importance", "target_importance"]].min(axis=1)
    df["score"] = df["overlap"] * (df["current_importance"] + df["target_importance"]) / 2.0
    df = df.sort_values(["score", "overlap"], ascending=False)
    return df["skill"].head(int(max_n)).tolist()


# ============================================================
# Offline fallback
# ============================================================
def _offline_strategy_markdown(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
    evidence: RetrievedEvidence,
    route: Optional[Dict[str, Any]],
) -> str:
    """Offline fallback used when OpenAI is unavailable."""
    route_line = "Direct pivot recommended."
    if route and route.get("reachable") and route.get("path"):
        path = route.get("path", [])
        if len(path) >= 3:
            route_line = "Suggested stepping-stone route: " + " → ".join(path)

    ev_lines = [f"- **{item.kind}**: {item.text}" for item in evidence.items[:5]]
    ms = missing_skills[:4] or ["No clear high-priority missing skills found."]
    ts = transfer_skills[:4] or ["No clear transferable anchors found."]

    return (
        "## Career Pivot Strategy\n\n"
        f"**Pivot:** {current_role} → {target_role}\n\n"
        f"**Recommendation:** {route_line}\n\n"
        "### Strong transferable anchors\n"
        + "\n".join([f"- {s}" for s in ts])
        + "\n\n"
        "### High-priority gaps\n"
        + "\n".join([f"- {s}" for s in ms])
        + "\n\n"
        "### Evidence from O*NET target-role data\n"
        + ("\n".join(ev_lines) if ev_lines else "- No target-role evidence retrieved.")
        + "\n\n"
        "### Suggested 90-day approach\n"
        "- Month 1: build foundations in the top two missing skills.\n"
        "- Month 2: create one portfolio artifact tied to target-role tasks.\n"
        "- Month 3: practice interview stories that connect your transfer strengths to target-role evidence.\n"
    )


# ============================================================
# Validation
# ============================================================
def _validate_strategy_json(
    obj: Dict[str, Any],
    *,
    allowed_missing: List[str],
    allowed_transfer: List[str],
    allowed_evidence_ids: List[str],
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Validate and sanitize structured planner JSON."""
    errors: List[str] = []

    required_top = [
        "pivot_summary",
        "prioritized_missing_skills",
        "transferable_strengths",
        "milestones",
        "project_ideas",
    ]
    for key in required_top:
        if key not in obj:
            errors.append(f"Missing top-level key: {key}")

    if errors:
        return False, errors, {}

    cleaned = {
        "pivot_summary": obj.get("pivot_summary", {}),
        "prioritized_missing_skills": [],
        "transferable_strengths": [],
        "milestones": [],
        "project_ideas": [],
        "interview_story_angles": obj.get("interview_story_angles", []),
    }

    for item in obj.get("prioritized_missing_skills", []):
        skill = str(item.get("skill", "")).strip()
        if skill not in allowed_missing:
            errors.append(f"Invalid prioritized missing skill: {skill}")
            continue

        eids = [str(x) for x in item.get("evidence_ids", []) if str(x) in allowed_evidence_ids]
        cleaned["prioritized_missing_skills"].append(
            {
                "skill": skill,
                "why_it_matters": _sanitize_text(item.get("why_it_matters", ""), 240),
                "evidence_ids": eids,
            }
        )

    for item in obj.get("transferable_strengths", []):
        skill = str(item.get("skill", "")).strip()
        if skill not in allowed_transfer:
            errors.append(f"Invalid transferable skill: {skill}")
            continue

        cleaned["transferable_strengths"].append(
            {
                "skill": skill,
                "relevance": _sanitize_text(item.get("relevance", ""), 220),
            }
        )

    for item in obj.get("milestones", []):
        eids = [str(x) for x in item.get("evidence_ids", []) if str(x) in allowed_evidence_ids]
        cleaned["milestones"].append(
            {
                "phase": _sanitize_text(item.get("phase", ""), 60),
                "objective": _sanitize_text(item.get("objective", ""), 220),
                "deliverable": _sanitize_text(item.get("deliverable", ""), 220),
                "evidence_ids": eids,
            }
        )

    for item in obj.get("project_ideas", []):
        skills = [
            str(x)
            for x in item.get("skills", [])
            if str(x) in allowed_missing or str(x) in allowed_transfer
        ]
        eids = [str(x) for x in item.get("evidence_ids", []) if str(x) in allowed_evidence_ids]
        cleaned["project_ideas"].append(
            {
                "title": _sanitize_text(item.get("title", ""), 120),
                "description": _sanitize_text(item.get("description", ""), 300),
                "skills": skills,
                "evidence_ids": eids,
            }
        )

    if not cleaned["prioritized_missing_skills"]:
        errors.append("No valid prioritized_missing_skills survived validation.")
    if len(cleaned["milestones"]) < 2:
        errors.append("Too few valid milestones.")
    if not cleaned["project_ideas"]:
        errors.append("No valid project_ideas survived validation.")

    return len(errors) == 0, errors, cleaned


# ============================================================
# Prompt builders
# ============================================================
def _build_evidence_text(evidence: RetrievedEvidence) -> str:
    """Serialize evidence into prompt-friendly lines."""
    return "\n".join([f"{item.evidence_id} | {item.kind} | {item.text}" for item in evidence.items])


def _planner_prompt(
    *,
    current_role: str,
    target_role: str,
    missing_skills: List[str],
    transfer_skills: List[str],
    evidence: RetrievedEvidence,
    route: Optional[Dict[str, Any]],
) -> str:
    """Prompt for the structured planning step."""
    route_text = "No route computed."
    if route:
        if route.get("reachable") and route.get("path"):
            route_text = "Route: " + " -> ".join([str(x) for x in route.get("path", [])])
        else:
            route_text = str(route.get("notes", "No route found."))

    return f"""
You are designing a career pivot strategy.

Current role: {current_role}
Target role: {target_role}

Allowed missing skills:
{json.dumps(missing_skills, ensure_ascii=False)}

Allowed transferable skills:
{json.dumps(transfer_skills, ensure_ascii=False)}

Stepping-stone route context:
{route_text}

Retrieved O*NET evidence for the target role:
{_build_evidence_text(evidence)}

Return ONLY valid JSON with this exact shape:
{{
  "pivot_summary": {{
    "difficulty": "low|medium|high",
    "route_recommendation": "direct|stepping_stone",
    "reasoning": "..."
  }},
  "prioritized_missing_skills": [
    {{
      "skill": "...",
      "why_it_matters": "...",
      "evidence_ids": ["E1", "E2"]
    }}
  ],
  "transferable_strengths": [
    {{
      "skill": "...",
      "relevance": "..."
    }}
  ],
  "milestones": [
    {{
      "phase": "0-30 days",
      "objective": "...",
      "deliverable": "...",
      "evidence_ids": ["E3"]
    }}
  ],
  "project_ideas": [
    {{
      "title": "...",
      "description": "...",
      "skills": ["..."],
      "evidence_ids": ["E1", "E4"]
    }}
  ],
  "interview_story_angles": [
    "..."
  ]
}}

Hard rules:
- Only use skills from the allowed lists.
- Only cite evidence IDs that exist in the retrieved evidence.
- Keep the reasoning grounded in the retrieved O*NET evidence.
- Make the output concise and implementation-oriented.
""".strip()


def _repair_prompt(original_text: str, errors: List[str]) -> str:
    """Prompt for the repair step if validation fails."""
    return f"""
The previous JSON was invalid.

Validation errors:
{json.dumps(errors, ensure_ascii=False)}

Previous output:
{original_text}

Return ONLY corrected JSON that fixes all validation errors.
""".strip()


def _writer_prompt(
    *,
    current_role: str,
    target_role: str,
    cleaned_strategy: Dict[str, Any],
    evidence: RetrievedEvidence,
) -> str:
    """Prompt that turns validated structured data into user-facing Markdown."""
    return f"""
You are a pragmatic career strategist.

Write a clean Markdown brief for a user.

Pivot:
- Current role: {current_role}
- Target role: {target_role}

Validated strategy JSON:
{json.dumps(cleaned_strategy, ensure_ascii=False, indent=2)}

Available evidence catalog:
{_build_evidence_text(evidence)}

Write:
1. A short verdict
2. Why this pivot is plausible
3. 3 milestones
4. 2 project ideas
5. 3 interview story angles
6. A final recommendation: direct pivot or stepping-stone

Rules:
- Use Markdown
- Be concise and specific
- Do not invent skills outside the validated JSON
- Refer to target-role evidence in plain English, not by raw evidence IDs only
""".strip()


# ============================================================
# Public orchestration API
# ============================================================
def generate_pivot_strategy_bundle(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    route: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    data_dir: str = "data/onet_raw",
) -> Dict[str, Any]:
    """Run the full strategy pipeline and return a rich result bundle.

    This function powers:
      - the final user-facing Markdown
      - the visible LLM trace panel
      - structured downstream logic
    """
    cur = _sanitize_role_name(current_role)
    tgt = _sanitize_role_name(target_role)

    missing_skills = top_missing_skills(gap_df, max_n=6)
    transfer_skills = top_transferable_skills(gap_df, max_n=5)

    candidate_terms = list(dict.fromkeys(missing_skills + transfer_skills + [tgt]))
    evidence = retrieve_target_evidence(
        tgt,
        candidate_terms=candidate_terms,
        data_dir=data_dir,
        max_tasks=8,
        max_tech=6,
        max_activities=6,
    )

    trace: Dict[str, Any] = {
        "mode": "offline",
        "planner_raw": "",
        "repair_raw": "",
        "writer_raw": "",
        "repair_attempted": False,
        "validation_errors": [],
        "retrieved_evidence": [asdict(x) for x in evidence.items],
        "soc_codes": list(evidence.soc_codes),
        "job_zone": evidence.job_zone,
        "missing_skills": missing_skills,
        "transfer_skills": transfer_skills,
        "cleaned_strategy": {},
    }

    offline_md = _offline_strategy_markdown(
        current_role=cur,
        target_role=tgt,
        missing_skills=missing_skills,
        transfer_skills=transfer_skills,
        evidence=evidence,
        route=route,
    )

    if not prefer_online:
        return {
            "markdown": offline_md,
            "source": "Offline evidence",
            "trace": trace,
            "cleaned_strategy": {},
            "evidence": evidence,
        }

    api_key = _get_api_key_optional()
    if not api_key:
        return {
            "markdown": offline_md,
            "source": "Offline evidence",
            "trace": trace,
            "cleaned_strategy": {},
            "evidence": evidence,
        }

    try:
        from openai import OpenAI
    except Exception:
        return {
            "markdown": offline_md,
            "source": "Offline evidence",
            "trace": trace,
            "cleaned_strategy": {},
            "evidence": evidence,
        }

    allowed_evidence_ids = [item.evidence_id for item in evidence.items]

    try:
        client = OpenAI(api_key=api_key)

        planner_resp = client.responses.create(
            model=model,
            instructions="Return strictly valid JSON and nothing else.",
            input=_planner_prompt(
                current_role=cur,
                target_role=tgt,
                missing_skills=missing_skills,
                transfer_skills=transfer_skills,
                evidence=evidence,
                route=route,
            ),
        )
        planner_text = (planner_resp.output_text or "").strip()
        trace["planner_raw"] = planner_text
        planner_obj = _extract_json_object(planner_text)

        valid, errors, cleaned = _validate_strategy_json(
            planner_obj,
            allowed_missing=missing_skills,
            allowed_transfer=transfer_skills,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        trace["validation_errors"] = list(errors)

        if not valid:
            trace["repair_attempted"] = True
            repair_resp = client.responses.create(
                model=model,
                instructions="Return strictly valid JSON and nothing else.",
                input=_repair_prompt(planner_text, errors),
            )
            repair_text = (repair_resp.output_text or "").strip()
            trace["repair_raw"] = repair_text
            repaired_obj = _extract_json_object(repair_text)

            valid, errors, cleaned = _validate_strategy_json(
                repaired_obj,
                allowed_missing=missing_skills,
                allowed_transfer=transfer_skills,
                allowed_evidence_ids=allowed_evidence_ids,
            )
            trace["validation_errors"] = list(errors)

            if not valid:
                return {
                    "markdown": offline_md,
                    "source": "Offline evidence",
                    "trace": trace,
                    "cleaned_strategy": {},
                    "evidence": evidence,
                }

        trace["cleaned_strategy"] = cleaned
        trace["mode"] = "online"

        writer_resp = client.responses.create(
            model=model,
            instructions="Write concise Markdown for a Streamlit UI.",
            input=_writer_prompt(
                current_role=cur,
                target_role=tgt,
                cleaned_strategy=cleaned,
                evidence=evidence,
            ),
        )
        writer_text = (writer_resp.output_text or "").strip()
        trace["writer_raw"] = writer_text

        if not writer_text:
            return {
                "markdown": offline_md,
                "source": "Offline evidence",
                "trace": trace,
                "cleaned_strategy": cleaned,
                "evidence": evidence,
            }

        return {
            "markdown": "🤖 " + writer_text,
            "source": "OpenAI multi-step",
            "trace": trace,
            "cleaned_strategy": cleaned,
            "evidence": evidence,
        }

    except Exception as e:
        trace["validation_errors"] = [f"Exception: {repr(e)}"]
        return {
            "markdown": offline_md,
            "source": "Offline evidence",
            "trace": trace,
            "cleaned_strategy": {},
            "evidence": evidence,
        }


def generate_pivot_strategy_markdown(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    route: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    data_dir: str = "data/onet_raw",
) -> str:
    """Backward-compatible wrapper returning only Markdown."""
    bundle = generate_pivot_strategy_bundle(
        current_role=current_role,
        target_role=target_role,
        gap_df=gap_df,
        route=route,
        model=model,
        prefer_online=prefer_online,
        data_dir=data_dir,
    )
    return str(bundle["markdown"])