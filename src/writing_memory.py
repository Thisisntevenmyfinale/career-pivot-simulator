"""
Application Ghost-Writer Memory
=================================
Every approved application makes the next one better.

The Ghost-Writer Memory stores the strongest phrases, sentences,
and structural patterns from past applications that passed the
quality gate (score ≥ threshold). When generating new applications,
these phrases are injected as context so the model can build on
proven language — not start from scratch every time.

This creates a personal writing asset that compounds over sessions:
  Month 1: Generic, good-quality output
  Month 2: Output flavored by your best past language
  Month 3: Output that sounds uniquely, powerfully like you

Architecture:
  Pure Python. No LLM. No API calls.
  Phrases stored in session_state.writing_memory (persisted).
  Phrase extraction: regex-based sentence splitting with quality filter.
  Relevance scoring: keyword overlap between stored context and new context.

Memory structure per entry:
  text:       the phrase or sentence
  context:    what artifact type and role it came from
  score:      quality gate score when it was generated
  n_uses:     how many times it's been reused
  added_at:   ISO date
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Phrase extraction
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_MIN_PHRASE_LEN = 40    # characters — short snippets aren't useful
_MAX_PHRASE_LEN = 280   # characters — don't store whole paragraphs

# Patterns that indicate a strong, specific phrase worth keeping
_STRONG_SIGNALS = [
    r'\d+%',                          # quantified impact
    r'\d+ year',                      # time context
    r'led|built|designed|delivered',  # strong verbs
    r'directly|specifically|resulting in',
    r'cross-functional|stakeholder',  # specificity signals
    r'revenue|growth|efficiency|impact',
]
_STRONG_RE = re.compile('|'.join(_STRONG_SIGNALS), re.IGNORECASE)

# Patterns to exclude — generic fluff
_FLUFF_PATTERNS = [
    r'^I am excited',
    r'^I am passionate',
    r'^Thank you for',
    r'please find attached',
    r'look forward to hearing',
    r'best regards',
    r'^sincerely',
]
_FLUFF_RE = re.compile('|'.join(_FLUFF_PATTERNS), re.IGNORECASE)


def extract_strong_phrases(text: str, score: int = 70) -> List[str]:
    """
    Extract the strongest, most reusable phrases from a generated text.
    Only called when score ≥ 65 (quality gate passed).
    """
    if not text or score < 65:
        return []

    sentences = _SENTENCE_SPLIT.split(text.strip())
    strong = []
    for sent in sentences:
        sent = sent.strip().rstrip('.,;')
        if len(sent) < _MIN_PHRASE_LEN or len(sent) > _MAX_PHRASE_LEN:
            continue
        if _FLUFF_RE.search(sent):
            continue
        if _STRONG_RE.search(sent):
            strong.append(sent)

    # If no flagged-strong sentences, take the top sentences by length
    if not strong:
        candidates = [
            s.strip() for s in sentences
            if _MIN_PHRASE_LEN <= len(s.strip()) <= _MAX_PHRASE_LEN
            and not _FLUFF_RE.search(s.strip())
        ]
        strong = sorted(candidates, key=len, reverse=True)[:3]

    return strong[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Memory operations
# ─────────────────────────────────────────────────────────────────────────────

def add_to_memory(
    state: Any,
    *,
    text: str,
    artifact: str,
    role_context: str,
    score: int,
) -> None:
    """
    Extract phrases from text and add to writing memory.

    artifact:     "Cover Letter" | "CV Bullets" | "LinkedIn InMail" etc.
    role_context: target role / company for relevance matching later
    score:        quality gate score (only stores if ≥ 65)
    """
    phrases = extract_strong_phrases(text, score)
    if not phrases:
        return

    mem: List[Dict] = state.get("writing_memory") or []
    today = date.today().isoformat()

    for phrase in phrases:
        # Deduplicate: skip if very similar phrase already exists
        if any(_similarity(phrase, entry["text"]) > 0.8 for entry in mem):
            continue
        mem.append({
            "text":       phrase,
            "artifact":   artifact,
            "context":    role_context,
            "score":      score,
            "n_uses":     0,
            "added_at":   today,
        })

    # Keep memory bounded: max 80 phrases, prefer highest scored
    if len(mem) > 80:
        mem.sort(key=lambda e: (e.get("score", 0), e.get("n_uses", 0)), reverse=True)
        mem = mem[:80]

    state["writing_memory"] = mem


def get_relevant_phrases(
    state: Any,
    *,
    artifact: str,
    role_context: str,
    n: int = 5,
) -> List[Dict]:
    """
    Return the n most relevant phrases for a new generation task.
    Relevance = artifact match × keyword overlap × score weight.
    """
    mem: List[Dict] = state.get("writing_memory") or []
    if not mem:
        return []

    context_words = set(_tokenize(role_context + " " + artifact))
    scored = []
    for entry in mem:
        # Artifact type match bonus
        art_match = 1.5 if entry.get("artifact") == artifact else 1.0
        # Keyword overlap with current context
        entry_words = set(_tokenize(entry.get("context", "") + " " + entry.get("text", "")))
        overlap = len(context_words & entry_words) / max(1, len(context_words))
        # Quality score weight
        quality_weight = (entry.get("score", 65) / 100)
        relevance = art_match * (0.5 + overlap) * quality_weight
        scored.append((relevance, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [entry for _, entry in scored[:n]]

    # Mark as used
    for entry in top:
        entry["n_uses"] = entry.get("n_uses", 0) + 1

    return top


def format_memory_injection(phrases: List[Dict], artifact: str) -> str:
    """
    Format retrieved phrases as a context injection for the LLM prompt.
    """
    if not phrases:
        return ""

    lines = [
        f"WRITING MEMORY — reuse and adapt these proven phrases from past high-scoring {artifact}s "
        f"(do NOT copy verbatim — integrate naturally):\n"
    ]
    for i, p in enumerate(phrases, 1):
        lines.append(f"{i}. \"{p['text']}\"  [score: {p.get('score','?')}/100]")

    return "\n".join(lines)


def get_memory_stats(state: Any) -> Dict[str, Any]:
    """Return stats about the writing memory for display."""
    mem: List[Dict] = state.get("writing_memory") or []
    if not mem:
        return {"total": 0, "avg_score": 0, "artifacts": {}, "total_uses": 0}

    by_artifact: Dict[str, int] = {}
    for e in mem:
        art = e.get("artifact", "Other")
        by_artifact[art] = by_artifact.get(art, 0) + 1

    scores = [e.get("score", 65) for e in mem]
    total_uses = sum(e.get("n_uses", 0) for e in mem)

    return {
        "total":       len(mem),
        "avg_score":   round(sum(scores) / len(scores)) if scores else 0,
        "artifacts":   by_artifact,
        "total_uses":  total_uses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z]{3,}', text.lower())


def _similarity(a: str, b: str) -> float:
    """Rough character-level similarity (Jaccard on trigrams)."""
    def trigrams(s: str):
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)}
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
