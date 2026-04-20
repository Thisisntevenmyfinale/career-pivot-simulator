"""
LinkedIn Connections CSV → Smart Warm Intro Engine
====================================================
Cold applications: 2-3% response rate.
Warm referrals via network: 35-50% response rate.

This engine upgrades warm intro from "text box entry" to:
  1. Upload your actual LinkedIn connections export (CSV)
  2. System auto-matches against your target companies
  3. Scores each connection by seniority + PM-adjacency
  4. Generates personalised 1:1 DMs for the top matches

Why this beats manual entry:
  - You have 200-2000 connections you've forgotten about
  - The system finds matches you'd never think to search for
  - Every message is personalised to that person's actual role
  - It covers all target companies simultaneously, not one at a time

LinkedIn CSV export:
  Settings → Data Privacy → Get a copy of your data → Connections.csv
  Fields: First Name, Last Name, Email Address, Company, Position, Connected On

Architecture:
  parse_connections_csv()     → normalise the messy LinkedIn export
  score_connections()         → seniority × PM-adjacency heuristic (pure Python)
  find_warm_intros()          → fuzzy match company names against target list
  draft_dm()                  → gpt-4o-mini, temp=0.6 for natural voice variation
  bulk_draft_dms()            → parallel generation for top-N matches

Model choice: gpt-4o-mini at temp=0.6 for DMs.
  Why not gpt-4o? DMs are short (4 sentences) — the marginal quality gain
  from gpt-4o is negligible, while the latency cost for 5 parallel DMs
  would make the UX painful (4s vs 0.8s per message).
  Why temp=0.6? Low enough for coherent sentences, high enough that
  5 DMs to 5 different people don't read identically.
"""

from __future__ import annotations

import csv
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Seniority + PM-adjacency scoring (pure Python, no API)
# ─────────────────────────────────────────────────────────────────────────────

_SENIORITY: List[Tuple[str, int]] = [
    ("c-suite", 7), ("chief", 7), ("cpo", 7), ("cto", 7), ("ceo", 7),
    ("vp", 6), ("vice president", 6), ("director", 5), ("head of", 5),
    ("principal", 4), ("lead", 3), ("senior", 3), ("staff", 4),
    ("manager", 3), ("product manager", 4), ("pm ", 3),
    ("associate", 2), ("junior", 1), ("intern", 0), ("student", 0),
]

_PM_ADJACENT: List[str] = [
    "product", "growth", "strategy", "analytics", "data", "design",
    "ux", "research", "engineering", "tech", "platform", "operations",
    "chief of staff", "business development", "partnerships",
]

_COMPANY_NOISE = re.compile(
    r"\b(gmbh|ag|se|inc|llc|ltd|corp|corporation|company|co\.?|group|holding|pty|plc)\b",
    re.IGNORECASE,
)


def _normalise_company(name: str) -> str:
    name = _COMPANY_NOISE.sub("", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def _score_position(position: str) -> Tuple[int, int]:
    """Returns (seniority_score, pm_adjacency_score)."""
    pos = position.lower()
    seniority = 2  # default: individual contributor
    for kw, val in _SENIORITY:
        if kw in pos:
            seniority = max(seniority, val)
    pm_adj = sum(1 for kw in _PM_ADJACENT if kw in pos)
    return seniority, pm_adj


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_connections_csv(csv_text: str) -> List[Dict[str, str]]:
    """
    Parse LinkedIn connections export CSV.
    LinkedIn wraps the actual CSV in a few header lines — we find the real header.
    Returns list of normalised connection dicts.
    """
    if not csv_text or not csv_text.strip():
        return []

    lines = csv_text.strip().splitlines()

    # Find the row that contains the real column headers
    header_idx = 0
    for i, line in enumerate(lines):
        if "First Name" in line or "first name" in line.lower():
            header_idx = i
            break

    reader = csv.DictReader(lines[header_idx:])
    connections = []

    for row in reader:
        # Normalise field names (LinkedIn sometimes uses different casing)
        def _get(*keys: str) -> str:
            for k in keys:
                v = row.get(k) or row.get(k.title()) or row.get(k.upper()) or ""
                if v.strip():
                    return v.strip()
            return ""

        first = _get("First Name", "first_name", "firstname")
        last  = _get("Last Name", "last_name", "lastname")
        email = _get("Email Address", "email_address", "email")
        co    = _get("Company", "company")
        pos   = _get("Position", "position")
        conn  = _get("Connected On", "connected_on")

        if not first and not co:
            continue  # skip blank rows

        seniority, pm_adj = _score_position(pos)
        connections.append({
            "first_name":      first,
            "last_name":       last,
            "full_name":       f"{first} {last}".strip(),
            "email":           email,
            "company":         co,
            "company_norm":    _normalise_company(co),
            "position":        pos,
            "connected_on":    conn,
            "seniority_score": seniority,
            "pm_adj_score":    pm_adj,
            "composite_score": seniority * 2 + pm_adj,
        })

    return connections


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def find_warm_intros(
    connections: List[Dict[str, str]],
    target_companies: List[str],
    *,
    max_per_company: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fuzzy-match connections against target company list.
    Returns matches sorted by composite_score (seniority + PM-adjacency).
    """
    if not connections or not target_companies:
        return []

    norm_targets = {_normalise_company(c): c for c in target_companies}
    matches: List[Dict] = []

    for conn in connections:
        cn = conn["company_norm"]
        if not cn:
            continue
        matched = None
        for norm_t, orig_t in norm_targets.items():
            if cn == norm_t:
                matched = orig_t
                break
            if norm_t in cn or cn in norm_t:
                matched = orig_t
                break
            # Token overlap: share ≥1 non-trivial word
            t_words = set(norm_t.split())
            c_words = set(cn.split())
            shared = t_words & c_words - {"of", "the", "and", "for", "at", "in"}
            if len(shared) >= 1 and max(len(t_words), len(c_words)) <= 4:
                matched = orig_t
                break
        if matched:
            matches.append({**conn, "target_company": matched})

    # Sort by composite_score DESC
    matches.sort(key=lambda x: x["composite_score"], reverse=True)

    # Cap per company
    company_counts: Dict[str, int] = {}
    filtered = []
    for m in matches:
        tc = m["target_company"]
        company_counts[tc] = company_counts.get(tc, 0)
        if company_counts[tc] < max_per_company:
            filtered.append(m)
            company_counts[tc] += 1

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# DM generation
# ─────────────────────────────────────────────────────────────────────────────

_DM_SYSTEM = """You are writing a LinkedIn DM for a career changer.
The message must feel personal and non-templated.

RULES (strict):
- Max 4 sentences. No exceptions.
- Start with the recipient's first name
- Reference their SPECIFIC role or something specific about their company — not generic praise
- The ask: "15-min coffee chat" — NOT a referral, NOT "can you refer me", NOT "I'm applying"
- Never use phrases like: "I hope this finds you well", "I'm reaching out because", "I was wondering if"
- End with a simple yes/no question
- Tone: warm, direct, slightly informal — like a peer, not a fan
- Do NOT say you're "trying to break into PM" — show PM thinking instead

Output JSON only:
{
  "subject_line": "First line of the DM (10-15 words, hooks attention)",
  "message": "The full 4-sentence DM — no headers, no bullets, just text",
  "follow_up": "1 follow-up message if no response in 1 week (2-3 sentences max)"
}"""


def draft_dm(
    oai_key: str,
    *,
    connection: Dict[str, str],
    target_role: str,
    pivot_dna: Optional[Dict] = None,
    cv_profile: Optional[Dict] = None,
) -> Optional[Dict[str, str]]:
    """Generate a personalised DM for a single connection."""
    if not oai_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        context = {
            "recipient_first_name": connection.get("first_name", ""),
            "recipient_role":       connection.get("position", ""),
            "recipient_company":    connection.get("company", ""),
            "sender_current_role":  (cv_profile or {}).get("extracted_role", ""),
            "sender_target_role":   target_role,
            "sender_pivot_hook":    (pivot_dna or {}).get("pivot_hook", ""),
            "sender_unfair_advantage": (pivot_dna or {}).get("unfair_advantage", ""),
            "three_word_brand":     (pivot_dna or {}).get("three_word_brand", ""),
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": _DM_SYSTEM},
                {"role": "user",   "content": json.dumps(context, indent=2)},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def bulk_draft_dms(
    oai_key: str,
    matches: List[Dict[str, Any]],
    *,
    target_role: str,
    pivot_dna: Optional[Dict] = None,
    cv_profile: Optional[Dict] = None,
    max_n: int = 6,
) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Generate DMs for top-N matches in parallel using ThreadPoolExecutor.
    Returns dict: connection full_name → DM dict (or None on failure).

    Parallelism here matters: 6 sequential calls ≈ 5s; 6 parallel ≈ 1s.
    """
    top = matches[:max_n]
    results: Dict[str, Optional[Dict]] = {}

    def _generate(conn: Dict) -> Tuple[str, Optional[Dict]]:
        key = f"{conn.get('full_name','')}_{conn.get('company','')}"
        dm = draft_dm(
            oai_key,
            connection=conn,
            target_role=target_role,
            pivot_dna=pivot_dna,
            cv_profile=cv_profile,
        )
        return key, dm

    with ThreadPoolExecutor(max_workers=min(6, len(top))) as executor:
        futures = {executor.submit(_generate, conn): conn for conn in top}
        for future in as_completed(futures):
            key, dm = future.result()
            results[key] = dm

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def seniority_label(score: int) -> str:
    return {7: "C-Suite", 6: "VP", 5: "Director", 4: "Principal/PM", 3: "Senior/Lead",
            2: "Mid-level", 1: "Junior", 0: "Entry/Intern"}.get(score, "Unknown")


def seniority_color(score: int) -> str:
    if score >= 5: return "#057642"
    if score >= 3: return "#0A66C2"
    return "#5F6B7A"
