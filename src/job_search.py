"""
Real Job Search
===============
Searches for live job listings via SerpAPI (Google Jobs engine), which aggregates
postings from LinkedIn, Indeed, Glassdoor, and hundreds of other boards.

This means every "Apply" button links to an actual job on an actual platform.
The full job description is captured and fed into `generate_application_package()`
so that cover letters are tailored to the SPECIFIC job posting, not just the role.

Setup
-----
Add `SERP_API_KEY` to your Streamlit secrets (https://serpapi.com — free tier:
100 searches/month). Without it, the tool falls back to LLM-generated job listings.

Architecture
------------
1. search_real_jobs() — calls SerpAPI, returns List[RealJobResult]
2. real_job_to_listing() — converts SerpAPI result → JobListing dataclass
   (so it plugs into the existing Smart Apply pipeline unchanged)
3. extract_cv_text() — extracts text from uploaded PDF/DOCX file (for drag-and-drop CV)
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# CV file text extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_cv_text(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF, DOCX, or TXT file.
    Called with a Streamlit UploadedFile object.
    """
    fname = getattr(uploaded_file, "name", "").lower()
    content = uploaded_file.read()

    # PDF
    if fname.endswith(".pdf") or getattr(uploaded_file, "type", "") == "application/pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = "\n\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass
        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            out = io.StringIO()
            extract_text_to_fp(io.BytesIO(content), out, laparams=LAParams())
            text = out.getvalue().strip()
            if text:
                return text
        except Exception:
            pass
        return "[Could not extract PDF text. Try copying and pasting your CV text instead.]"

    # DOCX
    if fname.endswith(".docx") or "word" in getattr(uploaded_file, "type", ""):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            return f"[Could not extract DOCX text: {e}]"

    # DOC (old Word format) or TXT
    try:
        return content.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "[Could not extract file text.]"


# ──────────────────────────────────────────────────────────────────────────────
# Real job search via SerpAPI Google Jobs
# ──────────────────────────────────────────────────────────────────────────────

def search_real_jobs(
    target_role: str,
    location: str = "United States",
    n_jobs: int = 5,
    serp_api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for real job listings via SerpAPI (Google Jobs aggregator).

    Returns a list of raw job dicts. Convert to JobListing via
    real_job_to_listing() for use in the Smart Apply pipeline.

    Each result includes:
      title, company, location, via (source), description (full text),
      apply_link, posted_at, salary, job_type, is_real=True
    """
    if not serp_api_key:
        return []

    try:
        import requests
    except ImportError:
        return []

    query = f"{target_role} {location}"
    params = {
        "engine": "google_jobs",
        "q": query,
        "hl": "en",
        "api_key": serp_api_key,
    }

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": repr(e)}]

    results = []
    for job in data.get("jobs_results", [])[:n_jobs]:
        ext = job.get("detected_extensions", {})
        # Try to get the best apply link
        apply_links = job.get("apply_options", []) or job.get("related_links", [])
        apply_link = ""
        apply_source = job.get("via", "")
        for al in apply_links:
            link = al.get("link", "")
            title = al.get("title", "") or al.get("text", "")
            if link:
                apply_link = link
                apply_source = title
                break

        results.append({
            "title": str(job.get("title", target_role)),
            "company": str(job.get("company_name", "Unknown")),
            "location": str(job.get("location", location)),
            "via": str(job.get("via", "Google Jobs")),
            "description": str(job.get("description", "")),
            "apply_link": apply_link,
            "apply_source": apply_source,
            "posted_at": str(ext.get("posted_at", "Recently")),
            "salary": str(ext.get("salary", "Not specified")),
            "job_type": str(ext.get("schedule_type", "Full-time")),
            "is_real": True,
        })

    return results


def real_job_to_listing(raw: Dict[str, Any], idx: int, match_score: int = 65):
    """
    Convert a raw SerpAPI job result into a JobListing dataclass for the Smart Apply pipeline.
    Imports JobListing lazily to avoid circular imports.
    """
    from src.smart_apply import JobListing

    # Determine company emoji from job type / company name heuristic
    via = raw.get("via", "").lower()
    emoji = "💼"
    if "linkedin" in via:
        emoji = "🔗"
    elif "indeed" in via:
        emoji = "🏢"
    elif "glassdoor" in via:
        emoji = "🌐"
    elif "startup" in raw.get("company", "").lower():
        emoji = "🚀"

    # Source badge
    source_note = raw.get("via", "Google Jobs")

    return JobListing(
        id=f"real_{idx}",
        title=raw["title"],
        company=raw["company"],
        company_emoji=emoji,
        location=raw["location"],
        job_type=raw.get("job_type", "Full-time"),
        salary_range=raw.get("salary", "See posting"),
        posted_ago=raw.get("posted_at", "Recently"),
        applicant_count=0,          # not available from SerpAPI
        is_easy_apply=False,
        match_score=match_score,
        key_requirements=[],         # extracted from description by application package generator
        description_preview=raw["description"][:300] + ("…" if len(raw["description"]) > 300 else ""),
        hiring_manager_name="Hiring Team",
        hiring_manager_title="",
        network_connections=0,
        seniority="",
        # Attach extra fields for downstream use
        apply_link=raw.get("apply_link", ""),
        apply_source=source_note,
        full_description=raw.get("description", ""),
        is_real_job=True,
    )
