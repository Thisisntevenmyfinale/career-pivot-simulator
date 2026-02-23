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
    Uses the same missing-skill selection logic as the online prompt.
    Never calls external services.
    """
    cur = _sanitize_role_name(current_role)
    tgt = _sanitize_role_name(target_role)

    miss = _pick_missing_skills_for_llm(gap_df, max_n=max_missing)

    if miss.empty:
        if language.lower().startswith("en"):
            return (
                "🧰 **Learning Plan (Offline-Fallback):**\n\n"
                "✅ In dieser Pivot-Konfiguration gibt es nach Filtern keine klar prioritären Skill-Gaps.\n\n"
                "**Empfohlener Fokus (8–14 Tage):**\n"
                "- 1 Portfolio-Projekt, das deinen Pivot plausibel macht (Output: Repo + kurze Demo).\n"
                "- CV/LinkedIn: 3 transferierbare Stories (Problem → Aktion → Ergebnis).\n"
                "- 6 Zielrollen-Interviewfragen üben (STAR + Zahlen/Impact).\n"
            )
        return (
            "🧰 **Learning Plan (Offline fallback):**\n\n"
            "✅ After filtering, there are no clear priority skill gaps for this pivot.\n\n"
            "**Focus (8–14 days):**\n"
            "- One portfolio project (repo + short demo).\n"
            "- Resume/LinkedIn: 3 transferable stories (problem → action → impact).\n"
            "- Practice 6 target-role interview questions (STAR + metrics).\n"
        )

    skills = miss["skill"].astype(str).tolist()

    # simple 3-phase split (deterministic, stable)
    n = len(skills)
    a = max(1, n // 3)
    b = max(1, n // 3)
    foundations = skills[:a]
    intermediate = skills[a : a + b]
    advanced = skills[a + b :]

    def bullets(phase_skills: list[str]) -> str:
        return "\n".join([f"- **{s}** → 2×/Woche Übungsblock + 1 messbarer Output" for s in phase_skills]) if phase_skills else "- (keine weiteren prioritären Skills)"

    if language.lower().startswith("en"):
        return (
            "🧰 **Learning Plan (Offline-Fallback, deterministisch):**\n"
            "_OpenAI nicht verfügbar (kein Key/Quota/Netz/Dependency) – Plan wird lokal erzeugt._\n\n"
            f"**Pivot:** {cur} → {tgt}\n\n"
            "## 1) Foundations (2–3 Wochen)\n"
            f"{bullets(foundations)}\n\n"
            "**Mini-Projekt:** 1 kleines Artefakt, das **2 Foundations-Skills** sichtbar macht (Repo + Readme + 1 Demo-Screenshot).\n\n"
            "## 2) Intermediate (3–6 Wochen)\n"
            f"{bullets(intermediate)}\n\n"
            "**Mini-Projekt:** 1 Case-Study (Problem → Ansatz → Ergebnis) als Blogpost/Notion/README.\n\n"
            "## 3) Advanced (6–10 Wochen)\n"
            f"{bullets(advanced)}\n\n"
            "**Mini-Projekt:** 1 “realistischeres” Projekt (mehr Constraints, Tests/Quality, kurze Präsentation).\n\n"
            "## Interview-Fragen (Zielrolle)\n"
            f"- Welche 2–3 Kernprobleme löst ein(e) **{tgt}** im Alltag?\n"
            "- Erzähle von einem Projekt, das unklare Anforderungen hatte – wie bist du vorgegangen?\n"
            "- Wie misst du Qualität/Erfolg (Metriken, Tests, Feedback-Loops)?\n"
            "- Wo gehst du bewusst Trade-offs ein (Zeit vs Qualität vs Scope)?\n"
            "- Wie gehst du mit Stakeholdern/Konflikten/Scope-Creep um?\n"
            "- Was war dein größter Lernsprung – und wie hast du ihn erreicht?\n\n"
            "## 3 typische Fehler beim Pivot\n"
            "- Zu viel “Konsum” (Kurse) ohne Output – **Portfolio schlägt Zertifikate**.\n"
            "- Skills isoliert lernen statt entlang eines Problems – **Problem-First**.\n"
            "- Unklare Narrative – du brauchst 3 klare Stories (Transfer, Motivation, Beleg).\n"
        )

    return (
        "🧰 **Learning Plan (Offline fallback, deterministic):**\n"
        "_OpenAI unavailable (no key/quota/network/dependency) – generated locally._\n\n"
        f"**Pivot:** {cur} → {tgt}\n\n"
        "## 1) Foundations (2–3 weeks)\n"
        + "\n".join([f"- **{s}** → 2 practice blocks/week + one measurable output" for s in foundations])
        + "\n\n"
        "**Mini-project:** one small artifact demonstrating **2 foundations skills** (repo + README + demo).\n\n"
        "## 2) Intermediate (3–6 weeks)\n"
        + "\n".join([f"- **{s}** → applied practice + one deliverable" for s in intermediate])
        + "\n\n"
        "**Mini-project:** one case study (problem → approach → outcome) as a short write-up.\n\n"
        "## 3) Advanced (6–10 weeks)\n"
        + "\n".join([f"- **{s}** → realistic constraints + quality bar" for s in advanced])
        + "\n\n"
        "**Mini-project:** one more realistic project (constraints, tests/quality, short presentation).\n\n"
        "## Interview questions (target role)\n"
        f"- What are the 2–3 core problems a **{tgt}** solves day-to-day?\n"
        "- Tell me about a project with ambiguous requirements — how did you proceed?\n"
        "- How do you measure quality/success (metrics, tests, feedback loops)?\n"
        "- Where do you intentionally make trade-offs (time vs quality vs scope)?\n"
        "- How do you handle stakeholders/conflict/scope creep?\n"
        "- What was your biggest learning jump — and how did you achieve it?\n\n"
        "## 3 common pivot mistakes\n"
        "- Too much consumption (courses) without output — **portfolio beats certificates**.\n"
        "- Learning skills in isolation instead of around a problem — **problem-first**.\n"
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
    ALWAYS English unless language starts with 'de'
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

    # ===== LANGUAGE SWITCH (CLEAN) =====
    if language.lower().startswith("de"):

        instructions = (
            "Du bist ein pragmatischer Career-Coach. "
            "Keine Floskeln. Keine Übertreibung. "
            "Nur umsetzbare, job-relevante Schritte. "
            "Markdown-Struktur verwenden."
        )

        user_text = (
            f"Aktueller Job: {cur}\n"
            f"Zieljob: {tgt}\n\n"
            "Top Missing Skills:\n"
            + "\n".join(bullets)
            + "\n\n"
            "Erstelle:\n"
            "1) 3-Phasen Lernplan (Foundations / Intermediate / Advanced)\n"
            "2) Pro Phase 1 Mini-Projekt\n"
            "3) 6 Interviewfragen + 1 Satz Hinweis\n"
            "4) 3 typische Fehler\n"
        )

    else:

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