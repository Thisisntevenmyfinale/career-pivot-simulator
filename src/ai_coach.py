from __future__ import annotations

import os
import re

import pandas as pd
from openai import OpenAI


def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it locally: export OPENAI_API_KEY='...'\n"
            "On Streamlit Cloud: App → Settings → Secrets."
        )
    return key


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
    r"\bglare\s+sensitivity\b",
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
    Selects a small, high-signal set of missing skills for the LLM prompt.
    IMPORTANT: O*NET IM is 1..5. Our goal is a practical plan, not a physiology checklist.

    Rules:
    1) Start strict:
       - gap > 0
       - target_importance >= 3.0
       - gap >= 0.8
    2) Exclude physical/sensory/psychomotor patterns (see regex).
    3) If too few remain, relax thresholds gradually (but keep exclusions).
    4) Rank by priority = gap * target_importance (classic), and return top max_n.

    Returns dataframe with columns:
      skill, gap, current_importance, target_importance, priority
    """
    required = {"skill", "gap", "current_importance", "target_importance"}
    missing_cols = required - set(gap_df.columns)
    if missing_cols:
        raise ValueError(f"gap_df missing required columns: {sorted(missing_cols)}")

    df = gap_df.copy()
    df["skill"] = df["skill"].astype(str)

    # numeric safety
    for col in ["gap", "current_importance", "target_importance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Only missing
    df = df[df["gap"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["skill", "gap", "current_importance", "target_importance", "priority"])

    # Exclude physical/sensory patterns
    df = df[~df["skill"].str.contains(_EXCLUSION_RE, na=False)].copy()

    # helper to apply thresholds
    def apply_thresholds(d: pd.DataFrame, *, min_tgt: float, min_gap: float) -> pd.DataFrame:
        out = d[(d["target_importance"] >= float(min_tgt)) & (d["gap"] >= float(min_gap))].copy()
        out["priority"] = (out["gap"] * out["target_importance"]).astype(float)
        out = out.sort_values(["priority", "gap", "target_importance"], ascending=False)
        return out

    # Stage 1: strict
    cand = apply_thresholds(df, min_tgt=3.0, min_gap=0.8)

    # Stage 2: relax gap
    if len(cand) < max_n:
        cand2 = apply_thresholds(df, min_tgt=3.0, min_gap=0.5)
        cand = pd.concat([cand, cand2], ignore_index=True).drop_duplicates(subset=["skill"])

    # Stage 3: relax target importance slightly (still meaningful on 1..5 scale)
    if len(cand) < max_n:
        cand3 = apply_thresholds(df, min_tgt=2.8, min_gap=0.5)
        cand = pd.concat([cand, cand3], ignore_index=True).drop_duplicates(subset=["skill"])

    # Final fallback: just take top missing by priority after exclusions
    if cand.empty:
        df["priority"] = (df["gap"] * df["target_importance"]).astype(float)
        cand = df.sort_values(["priority", "gap", "target_importance"], ascending=False)

    return cand.head(int(max_n)).reset_index(drop=True)


def generate_ai_learning_plan_markdown(
    *,
    current_role: str,
    target_role: str,
    gap_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    language: str = "de",
    max_missing: int = 6,
) -> str:
    """
    Uses OpenAI to translate quantified gaps into a realistic learning plan (Markdown).
    The LLM is used for *wording + structuring*, while skill selection remains deterministic.
    """
    _get_api_key()
    client = OpenAI()

    cur = _sanitize_role_name(current_role)
    tgt = _sanitize_role_name(target_role)

    miss = _pick_missing_skills_for_llm(gap_df, max_n=max_missing)

    if miss.empty:
        return (
            "✅ **AI Coach:** Nach Filtern (O*NET IM 1–5) gibt es hier keine klaren prioritären Skill-Gaps. "
            "Fokus: 1 Portfolio-Projekt + Bewerbungsunterlagen + Interview-Stories."
        )

    bullets = []
    for _, r in miss.iterrows():
        bullets.append(
            f"- {r['skill']} (gap={float(r['gap']):.2f}, target={float(r['target_importance']):.1f}, current={float(r['current_importance']):.1f})"
        )

    if language.lower().startswith("de"):
        instructions = (
            "Du bist ein pragmatischer Career-Coach. "
            "Du darfst NICHT übertreiben oder unnötige Themen erfinden. "
            "Gib nur job-relevante, umsetzbare Schritte. "
            "Wenn ein Skill abstrakt ist, übersetze ihn in ein konkretes Verhalten/Übungsformat. "
            "Antworte als Markdown, klar strukturiert."
        )
        user_text = (
            f"Aktueller Job: {cur}\n"
            f"Zieljob: {tgt}\n\n"
            "Top Missing Skills (aus O*NET, Importance 1..5, bereits gefiltert für Career-Actionability):\n"
            + "\n".join(bullets)
            + "\n\n"
            "Erstelle:\n"
            "1) Einen 3-Phasen Lernplan (Foundations / Intermediate / Advanced), max 5 bullets pro Phase.\n"
            "2) Pro Phase: 1 Mini-Projekt mit konkretem Output (Repo, Demo, Blogpost, Case Study).\n"
            "3) Danach: 6 Interviewfragen (für den Zieljob) + je 1 Satz, worauf man achten sollte.\n"
            "4) Optional: 3 typische Fehler/Illusionen von Career-Changern.\n"
            "Kurz, praktisch, keine Floskeln."
        )
    else:
        instructions = (
            "You are a pragmatic career coach. Do not over-recommend or invent unnecessary topics. "
            "Only give job-relevant, actionable steps. Return Markdown."
        )
        user_text = (
            f"Current role: {cur}\n"
            f"Target role: {tgt}\n\n"
            "Top Missing Skills (O*NET 1..5, filtered for career-actionability):\n"
            + "\n".join(bullets)
            + "\n\n"
            "Create:\n"
            "1) A 3-phase plan (Foundations/Intermediate/Advanced), max 5 bullets each.\n"
            "2) One mini-project per phase with a concrete output.\n"
            "3) Then: 6 interview questions + 1 sentence guidance each.\n"
            "4) Optional: 3 common mistakes career changers make.\n"
            "Short, practical, no fluff."
        )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_text,
    )

    text = (resp.output_text or "").strip()
    return text if text else "⚠️ AI Coach: Empty response. Try again."