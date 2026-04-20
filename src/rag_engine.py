"""
Career RAG Engine
==================
Retrieval-Augmented Generation for the Pivot-Zwilling.

Upgrades the Zwilling from a context-injected chatbot to a true
RAG system: every user query retrieves the most relevant chunks
from the user's actual career documents before generating a response.

Why this matters:
  Without RAG → The Zwilling knows Alex's profile in broad strokes
                from the system prompt, but can't cite specifics.
  With RAG    → "What's blocking me at Contentful?" retrieves the
                actual rejection log entry. "What's my top gap?"
                retrieves the exact O*NET gap data. Responses are
                grounded in facts, not hallucinated summaries.

Document corpus (built from session state):
  - CV text chunks  (200-word overlapping windows)
  - Pipeline jobs   (one chunk per job: title + company + status + notes)
  - Gap analysis    (top gaps with percentile data)
  - Pivot DNA       (transferable arguments + voice)
  - Mock interview  (scores + feedback per dimension)
  - Outcome log     (each rejection with stage + feedback)
  - Cohort data     (what worked / what failed for this pivot type)

Architecture:
  Embedding model: text-embedding-3-small (1536-dim, cheapest OpenAI model)
  Storage:         numpy float32 arrays in session_state (no external DB)
  Retrieval:       cosine similarity, top-k chunks
  Integration:     injected as a "RETRIEVED CONTEXT" block before user message
  Rebuild trigger: any time source documents change (tracked via content hash)

Cost estimate: ~$0.0001 per query (embedding is nearly free)
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> List[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += max_words - overlap
    return chunks


def build_corpus(state: Any) -> List[Dict[str, str]]:
    """
    Build the full document corpus from session state.
    Returns list of {text, source, label} dicts.
    Each will be embedded and stored for retrieval.
    """
    docs: List[Dict[str, str]] = []

    # ── CV text ──────────────────────────────────────────────────────────
    cv_text = (state.get("cv_text") or "").strip()
    if cv_text:
        for i, chunk in enumerate(_chunk_text(cv_text)):
            docs.append({
                "text":   chunk,
                "source": "cv",
                "label":  f"CV (part {i+1})",
            })

    # ── CV profile structured ─────────────────────────────────────────────
    cv_profile = state.get("cv_profile") or {}
    if cv_profile:
        skills = cv_profile.get("top_skills") or []
        yoe = cv_profile.get("years_experience", "?")
        achievements = cv_profile.get("key_achievements") or []
        profile_text = (
            f"Candidate profile: {cv_profile.get('extracted_role','?')} with {yoe} years experience. "
            f"Top skills: {', '.join(skills[:10])}. "
            f"Key achievements: {' '.join(achievements[:3])}."
        )
        docs.append({"text": profile_text, "source": "cv_profile", "label": "CV Profile Summary"})

    # ── Pivot DNA ─────────────────────────────────────────────────────────
    pivot_dna = state.get("pivot_dna") or {}
    if pivot_dna:
        dna_parts = []
        if pivot_dna.get("pivot_hook"):
            dna_parts.append(f"Pivot hook: {pivot_dna['pivot_hook']}")
        if pivot_dna.get("strongest_transferable_argument"):
            dna_parts.append(f"Strongest argument: {pivot_dna['strongest_transferable_argument']}")
        if pivot_dna.get("unfair_advantage"):
            dna_parts.append(f"Unfair advantage: {pivot_dna['unfair_advantage']}")
        if pivot_dna.get("pivot_risk"):
            dna_parts.append(f"Main risk: {pivot_dna['pivot_risk']}")
        if pivot_dna.get("mitigation"):
            dna_parts.append(f"Mitigation: {pivot_dna['mitigation']}")
        if pivot_dna.get("target_companies"):
            dna_parts.append(f"Target companies: {', '.join(pivot_dna['target_companies'][:5])}")
        if dna_parts:
            docs.append({
                "text":   "\n".join(dna_parts),
                "source": "pivot_dna",
                "label":  "Pivot DNA & Strategy",
            })

    # ── Skill gap analysis ────────────────────────────────────────────────
    skill_gap = state.get("skill_gap_results") or {}
    if skill_gap:
        gaps = skill_gap.get("gaps") or []
        gap_texts = []
        for g in gaps[:6]:
            skill = g.get("skill", "")
            sev = g.get("severity", "")
            importance = g.get("importance_in_target", 0)
            current_level = g.get("current_level", 0)
            gap_texts.append(
                f"Gap: {skill} — severity: {sev}, target importance: {importance:.2f}, "
                f"current level: {current_level:.2f}"
            )
        fit_pct = skill_gap.get("fit_percentile", 0)
        gap_summary = (
            f"O*NET skill fit: {fit_pct}th percentile. "
            f"Top skill gaps: {'; '.join(gap_texts)}"
        )
        docs.append({"text": gap_summary, "source": "skill_gap", "label": "Skill Gap Analysis"})

    # ── Pipeline jobs ─────────────────────────────────────────────────────
    pipeline = state.get("pipeline_jobs") or []
    for job in pipeline:
        job_text = (
            f"Application: {job.get('title','?')} at {job.get('company','?')}. "
            f"Status: {job.get('status','?')}. "
            f"Applied: {job.get('date_added','?')}. "
            f"Last update: {job.get('date_updated','?')}. "
            + (f"Notes: {job.get('notes','')}" if job.get("notes") else "")
        )
        docs.append({
            "text":   job_text,
            "source": "pipeline",
            "label":  f"Pipeline: {job.get('title','?')} @ {job.get('company','?')}",
        })

    # ── Outcome log ───────────────────────────────────────────────────────
    outcome_log = state.get("outcome_log") or []
    for outcome in outcome_log:
        outcome_text = (
            f"Outcome: {outcome.get('job_title','?')} at {outcome.get('company','?')} — "
            f"reached {outcome.get('actual_stage','?')}. "
            f"Date: {outcome.get('date','?')}. "
            + (f"Notes: {outcome.get('notes','')}" if outcome.get("notes") else "")
        )
        docs.append({
            "text":   outcome_text,
            "source": "outcome",
            "label":  f"Outcome: {outcome.get('company','?')}",
        })

    # ── Mock interview ────────────────────────────────────────────────────
    mock = state.get("mock_interview_report") or {}
    if mock:
        dims = mock.get("dimension_scores") or {}
        dim_text = ", ".join(f"{k}: {v}/100" for k, v in dims.items())
        mock_text = (
            f"Mock interview overall: {mock.get('overall_score','?')}/100. "
            f"Verdict: {mock.get('hire_recommendation','')}. "
            f"One-line: {mock.get('one_line_verdict','')}. "
            f"Dimensions — {dim_text}. "
            f"Top improvements: {'; '.join(mock.get('top_improvements',[])[:3])}."
        )
        docs.append({"text": mock_text, "source": "mock_interview", "label": "Mock Interview Report"})

    # ── Cohort intelligence ───────────────────────────────────────────────
    cohort = state.get("cohort_intelligence") or {}
    if cohort:
        cohort_text = (
            f"Cohort data for {cohort.get('pivot_description','this pivot')}: "
            f"median {cohort.get('median_timeline_weeks','?')} weeks, "
            f"{cohort.get('median_applications','?')} applications needed. "
            f"What worked: {cohort.get('what_worked','')[:300]}. "
            f"What failed: {cohort.get('what_failed','')[:200]}."
        )
        docs.append({"text": cohort_text, "source": "cohort", "label": "Cohort Intelligence"})

    # ── Calibration data ──────────────────────────────────────────────────
    cal = state.get("calibration_data") or {}
    if cal.get("calibrated"):
        cal_text = (
            f"Personal calibration: response rate {cal.get('personal_response_rate',0)*100:.0f}%, "
            f"adjustment factor {cal.get('adjustment_factor',1):.2f}x. "
            f"Dominant rejection stage: {cal.get('dominant_rejection_stage','?')}. "
            f"Insight: {cal.get('insight','')}"
        )
        docs.append({"text": cal_text, "source": "calibration", "label": "ROI Calibration"})

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings(texts: List[str], api_key: str) -> Optional[np.ndarray]:
    """
    Embed a list of texts using text-embedding-3-small.
    Returns float32 numpy array of shape [n_texts, 1536].
    """
    if not texts or not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Process in batches of 100 (API limit is higher but safe)
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            batch_embs = [e.embedding for e in resp.data]
            all_embeddings.extend(batch_embs)
        return np.array(all_embeddings, dtype=np.float32)
    except Exception:
        return None


def _corpus_hash(docs: List[Dict[str, str]]) -> str:
    """Cheap content hash to detect when corpus needs rebuilding."""
    content = json.dumps([d["text"][:50] for d in docs], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Index management
# ─────────────────────────────────────────────────────────────────────────────

def build_index(state: Any, api_key: str, force: bool = False) -> bool:
    """
    Build (or rebuild) the RAG index from current session state.

    Stores in state:
      rag_docs:         list of {text, source, label}
      rag_embeddings:   list of lists (JSON-serializable float32 → list)
      rag_corpus_hash:  str — fingerprint for change detection

    Returns True if index was built successfully.
    """
    docs = build_corpus(state)
    if not docs:
        return False

    current_hash = _corpus_hash(docs)
    existing_hash = state.get("rag_corpus_hash", "")

    if not force and current_hash == existing_hash and state.get("rag_embeddings"):
        return True  # Already up to date

    texts = [d["text"] for d in docs]
    embeddings = _get_embeddings(texts, api_key)
    if embeddings is None:
        return False

    # Store as plain Python lists for JSON serialization
    state["rag_docs"] = docs
    state["rag_embeddings"] = embeddings.tolist()
    state["rag_corpus_hash"] = current_hash
    return True


def retrieve(
    query: str,
    state: Any,
    api_key: str,
    k: int = 4,
) -> List[Dict[str, str]]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Returns list of {text, source, label, score} dicts,
    or empty list if index not ready.
    """
    raw_embs = state.get("rag_embeddings")
    docs = state.get("rag_docs") or []

    if not raw_embs or not docs:
        return []

    # Embed the query
    q_embs = _get_embeddings([query], api_key)
    if q_embs is None:
        return []

    emb_matrix = np.array(raw_embs, dtype=np.float32)
    q_vec = q_embs[0]

    # Cosine similarity (both already normalised by the API)
    norms_m = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
    norms_q = np.linalg.norm(q_vec) + 1e-9
    similarities = (emb_matrix / norms_m) @ (q_vec / norms_q)

    top_k_idx = np.argsort(similarities)[::-1][:k]
    results = []
    for idx in top_k_idx:
        if idx < len(docs):
            results.append({
                **docs[idx],
                "score": float(similarities[idx]),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Context formatting for prompt injection
# ─────────────────────────────────────────────────────────────────────────────

def format_retrieved_context(chunks: List[Dict[str, str]]) -> str:
    """
    Format retrieved chunks as a structured context block for the LLM.
    This is injected into the Zwilling conversation before each user message.
    """
    if not chunks:
        return ""

    lines = ["=== RETRIEVED CAREER DOCUMENTS (most relevant to this query) ==="]
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"\n[{i}] {chunk.get('label','Document')} "
            f"(relevance: {chunk.get('score',0):.2f})\n"
            f"{chunk['text']}"
        )
    lines.append("\n=== END RETRIEVED CONTEXT ===")
    lines.append("Use the above facts. Do not invent details not present in the context.")
    return "\n".join(lines)


def index_ready(state: Any) -> bool:
    """Return True if the RAG index exists and has content."""
    return bool(state.get("rag_embeddings") and state.get("rag_docs"))


def index_size(state: Any) -> int:
    """Return number of indexed chunks."""
    return len(state.get("rag_docs") or [])
