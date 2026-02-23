from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd


def _get_api_key_optional() -> str:
    """
    Returns API key or empty string (never raises).
    Online mode requires a non-empty key.
    """
    return os.getenv("OPENAI_API_KEY", "").strip()


def _sanitize_role_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:120]


# Physical / sensory / psychomotor signals in O*NET skills/abilities that are usually NOT actionable
# for typical white-collar/knowledge-worker pivots (and produce absurd learning plans).
_PHYSICAL_SENSORY_PATTERNS = [
    # Strength / stamina / flexibility
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
    # Vision / hearing / perception
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
    # Split-attention style abilities that often appear but are not great learning-plan items
    r"\btime\s+sharing\b",
    r"\breaction\s+time\b",
    r"\bselective\s+attention\b",
]


def _compile_exclusion_regex() -> re.Pattern:
    joined = "(" + "|".join(_PHYSICAL_SENSORY_PATTERNS) + ")"
    return re.compile(joined, flags=re.IGNORECASE)


_EXCLUSION_RE = _compile_exclusion_regex()


def _pick_missing_skills_for_llm(gap_df: pd.DataFrame, *, max_n: int = 6) -> pd.DataFrame:
    """
    Selects a small, high-signal set of missing skills for the prompt.
    IMPORTANT: O*NET IM is 1..5. Goal is practical plan, not physiology checklist.

    Returns dataframe with columns:
      skill, gap, current_importance, target_importance, priority
    """
    required = {"skill", "gap", "current_importance", "target_importance"}
    missing_cols = required - set(gap_df.columns)
    if missing_cols:
        raise ValueError(f"gap_df missing required columns: {sorted(missing_cols)}")

    df = gap_df.copy()
    df["skill"] = df["skill"].astype(str)

    for col in ["gap", "current_importance", "target_importance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df["gap"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["skill", "gap", "current_importance", "target_importance", "priority"])

    # Exclude physical/sensory patterns
    df = df[~df["skill"].str.contains(_EXCLUSION_RE, na=False)].copy()

    def apply_thresholds(d: pd.DataFrame, *, min_tgt: float, min_gap: float) -> pd.DataFrame:
        out = d[(d["target_importance"] >= float(min_tgt)) & (d["gap"] >= float(min_gap))].copy()
        out["priority"] = (out["gap"] * out["target_importance"]).astype(float)
        out = out.sort_values(["priority", "gap", "target_importance"], ascending=False)
        return out

    cand = apply_thresholds(df, min_tgt=3.0, min_gap=0.8)

    if len(cand) < max_n:
        cand2 = apply_thresholds(df, min_tgt=3.0, min_gap=0.5)
        cand = pd.concat([cand, cand2], ignore_index=True).drop_duplicates(subset=["skill"])

    if len(cand) < max_n:
        cand3 = apply_thresholds(df, min_tgt=2.8, min_gap=0.5)
        cand = pd.concat([cand, cand3], ignore_index=True).drop_duplicates(subset=["skill"])

    if cand.empty:
        df["priority"] = (df["gap"] * df["target_importance"]).astype(float)
        cand = df.sort_values(["priority", "gap", "target_importance"], ascending=False)

    return cand.head(int(max_n)).reset_index(drop=True)


def _offline_learning_plan_markdown(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    language: str,
    max_missing: int,
) -> str:
    """
    Deterministic offline fallback.
    English-only by project requirement.
    Never calls external services.
    """
    cur = _sanitize_role_name(current_role)
    tgt = _sanitize_role_name(target_role)

    miss = _pick_missing_skills_for_llm(gap_df, max_n=max_missing)

    if miss.empty:
        return (
            "🧰 **Learning Plan (Offline fallback, deterministic):**\n\n"
            "✅ After filtering, there are no clear high-priority skill gaps for this pivot.\n\n"
            "**Focus (8–14 days):**\n"
            "- Build 1 portfolio artifact that makes the pivot plausible (repo + short demo).\n"
            "- Rewrite your resume/LinkedIn using 3 transferable stories (problem → action → impact).\n"
            "- Practice 6 target-role interview questions (STAR + metrics).\n"
        )

    skills = miss["skill"].astype(str).tolist()

    # Deterministic 3-phase split
    n = len(skills)
    a = max(1, n // 3)
    b = max(1, n // 3)

    foundations = skills[:a]
    intermediate = skills[a : a + b]
    advanced = skills[a + b :]

    def _bullets(items: list[str], flavor: str) -> str:
        if not items:
            return "- (No additional priority skills detected in this dataset.)"
        if flavor == "foundation":
            return "\n".join([f"- **{s}** → 2 practice blocks/week + 1 measurable output" for s in items])
        if flavor == "intermediate":
            return "\n".join(
                [f"- **{s}** → applied practice + 1 deliverable (write-up, dashboard, case study)" for s in items]
            )
        return "\n".join([f"- **{s}** → realistic constraints + a quality bar (tests, review, iteration)" for s in items])

    return (
        "🧰 **Learning Plan (Offline fallback, deterministic):**\n"
        "_OpenAI unavailable (no key/quota/network/dependency) — generated locally._\n\n"
        f"**Pivot:** {cur} → {tgt}\n\n"
        "## 1) Foundations (2–3 weeks)\n"
        f"{_bullets(foundations, 'foundation')}\n\n"
        "**Mini-project:** One small artifact demonstrating **2 foundation skills** (repo + README + demo).\n\n"
        "## 2) Intermediate (3–6 weeks)\n"
        f"{_bullets(intermediate, 'intermediate')}\n\n"
        "**Mini-project:** One case study (problem → approach → outcome) as a short write-up.\n\n"
        "## 3) Advanced (6–10 weeks)\n"
        f"{_bullets(advanced, 'advanced')}\n\n"
        "**Mini-project:** One realistic project (constraints, evaluation, iteration) + 5-slide summary.\n\n"
        "## Interview questions (target role)\n"
        f"- What are the 2–3 core problems a **{tgt}** solves day-to-day? (Give examples.)\n"
        "- Tell me about a project with ambiguous requirements — how did you proceed?\n"
        "- How do you measure quality/success (metrics, tests, feedback loops)?\n"
        "- Where do you intentionally make trade-offs (time vs quality vs scope)?\n"
        "- How do you handle stakeholders/conflict/scope creep?\n"
        "- What was your biggest learning jump — and how did you achieve it?\n\n"
        "## 3 common pivot mistakes\n"
        "- Consuming courses without shipping outputs — **portfolio beats certificates**.\n"
        "- Learning skills in isolation instead of around problems — **problem-first**.\n"
        "- Unclear narrative — you need 3 crisp stories (transfer, motivation, evidence).\n"
    )


def generate_learning_plan_markdown(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    language: str = "en",
    max_missing: int = 6,
    prefer_online: bool = True,
) -> str:
    """
    Single entry-point:
    - tries OpenAI if possible
    - otherwise deterministic offline fallback
    ENGLISH ONLY
    """

    fallback = _offline_learning_plan_markdown(
        current_role=current_role,
        target_role=target_role,
        gap_df=gap_df,
        language=language,
        max_missing=max_missing,
    )

    if not prefer_online:
        return fallback

    key = _get_api_key_optional()
    if not key:
        return fallback

    try:
        from openai import OpenAI
    except Exception:
        return fallback

    cur = _sanitize_role_name(current_role)
    tgt = _sanitize_role_name(target_role)

    miss = _pick_missing_skills_for_llm(gap_df, max_n=max_missing)
    if miss.empty:
        return fallback

    bullets = []
    for _, r in miss.iterrows():
        bullets.append(
            f"- {r['skill']} (gap={float(r['gap']):.2f}, target={float(r['target_importance']):.1f}, current={float(r['current_importance']):.1f})"
        )

    # ===== PROMPT (ENGLISH ONLY) =====
    instructions = (
        "You are a pragmatic career coach. "
        "No fluff. No exaggeration. "
        "Only actionable, job-relevant steps. "
        "Return clean Markdown. "
        "Respond strictly in English."
    )

    user_text = (
        f"Current role: {cur}\n"
        f"Target role: {tgt}\n\n"
        "Top Missing Skills:\n"
        + "\n".join(bullets)
        + "\n\n"
        "Create:\n"
        "1) A 3-phase plan (Foundations / Intermediate / Advanced)\n"
        "2) One mini-project per phase\n"
        "3) 6 interview questions + 1 guidance sentence each\n"
        "4) 3 common pivot mistakes\n"
    )

    try:
        client = OpenAI(api_key=key)
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_text,
        )

        text = (resp.output_text or "").strip()
        if not text:
            return fallback

        return "🤖 **AI Coach (OpenAI):**\n\n" + text

    except Exception:
        return fallback