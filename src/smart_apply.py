"""
Smart Apply Engine
==================
Takes the pipeline from decision support → execution support.

Given a target role + personal CV profile, generates:
  1. 4 realistic job listings (LinkedIn-style: company, salary, requirements, match %)
  2. A complete application package for any chosen job:
       • Tailored cover letter
       • 3 CV bullet point rewrites (before → after, STAR format)
       • LinkedIn InMail to the hiring manager
       • Interview preparation guide (5 questions + model answers)
  3. "Pivot Peers" — 3 anonymised success stories of people who made
     a similar transition (LLM-generated social proof, grounded in skill path)

Architecture
------------
Pass 1 (job generation)   — gpt-4o-mini, JSON, 1 call
Pass 2 (package gen)      — gpt-4o, JSON, 1 call per selected job (on-demand)
Pass 3 (pivot peers)      — gpt-4o-mini, JSON, 1 call

All output is grounded in the user's actual skill vector (cv_parser),
gap analysis (top transferable + missing skills), and match score.

The LLM does NOT have access to real job boards. Job listings are plausible
but synthetic. This is clearly a simulation / prototype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JobListing:
    id: str
    title: str
    company: str
    company_emoji: str
    location: str
    job_type: str               # "Full-time" | "Contract" | "Hybrid"
    salary_range: str
    posted_ago: str             # "2 days ago"
    applicant_count: int
    is_easy_apply: bool
    match_score: int            # 0-100 (LLM estimated vs user profile)
    key_requirements: List[str] # top 4 requirements
    description_preview: str
    hiring_manager_name: str
    hiring_manager_title: str
    network_connections: int    # "X people you know work here"
    seniority: str              # "Mid-Senior level"
    # Real job fields (populated when sourced from SerpAPI)
    apply_link: str = ""
    apply_source: str = ""
    full_description: str = ""   # full job posting text → feeds into tailored cover letter
    is_real_job: bool = False    # True = sourced from real job board via SerpAPI


@dataclass
class CVBulletRewrite:
    skill_highlighted: str
    original: str
    rewritten: str
    why: str                    # why this rewrite strengthens the application


@dataclass
class ApplicationPackage:
    job_id: str
    job_title: str
    company: str
    cover_letter: str
    cv_bullet_rewrites: List[CVBulletRewrite]
    linkedin_inmail: str
    interview_prep: List[Dict]  # [{question, model_answer, why_asked}]
    positioning_statement: str  # 1-sentence pitch for this specific job
    source: str = "online"


@dataclass
class PivotPeer:
    name: str                   # "Sarah K."
    initials: str               # "SK"
    avatar_color: str           # hex background for avatar
    previous_role: str
    current_role: str
    company_now: str
    months_to_pivot: int
    key_milestone: str          # "Completed AWS certification in month 3"
    testimonial: str            # short quote
    connection_degree: int      # 1 or 2


# ──────────────────────────────────────────────────────────────────────────────
# Offline fallbacks
# ──────────────────────────────────────────────────────────────────────────────

def _offline_job_listings(target_role: str, match_score: float) -> List[JobListing]:
    base = max(int(match_score) - 10, 40)
    companies = [
        ("Horizon Technologies", "🚀", "San Francisco, CA"),
        ("Meridian Group", "🏢", "New York, NY"),
        ("Apex Digital", "💡", "Austin, TX"),
        ("Crestline Partners", "🌐", "Remote"),
    ]
    titles = [
        f"Senior {target_role}",
        target_role,
        f"Lead {target_role}",
        f"Associate {target_role}",
    ]
    results = []
    for i, ((company, emoji, location), title) in enumerate(zip(companies, titles)):
        results.append(JobListing(
            id=f"job_{i}",
            title=title,
            company=company,
            company_emoji=emoji,
            location=location,
            job_type="Full-time" if i < 3 else "Hybrid",
            salary_range=f"${90 + i * 15}k – ${120 + i * 15}k",
            posted_ago=f"{i + 1} day{'s' if i > 0 else ''} ago",
            applicant_count=80 + i * 35,
            is_easy_apply=i % 2 == 0,
            match_score=min(base + (3 - i) * 5, 95),
            key_requirements=[
                "3+ years relevant experience",
                "Strong communication skills",
                "Team collaboration",
                "Domain expertise",
            ],
            description_preview=f"We are looking for a talented {title} to join our growing team...",
            hiring_manager_name=["Alex Chen", "Jordan Smith", "Maya Patel", "Chris Liu"][i],
            hiring_manager_title=["VP Engineering", "Director of Operations", "Head of Product", "Team Lead"][i],
            network_connections=i + 1,
            seniority=["Mid-Senior level", "Entry level", "Director", "Mid-Senior level"][i],
        ))
    return results


def _offline_application_package(job: JobListing, current_role: str, target_role: str) -> ApplicationPackage:
    return ApplicationPackage(
        job_id=job.id,
        job_title=job.title,
        company=job.company,
        cover_letter=(
            f"Dear {job.hiring_manager_name},\n\n"
            f"I am writing to express my interest in the {job.title} role at {job.company}. "
            f"Coming from a background in {current_role}, I bring transferable skills that directly support "
            f"the demands of {target_role}. I am excited to contribute to {job.company}'s mission and would "
            f"welcome the opportunity to discuss how my background aligns with your team's needs.\n\n"
            f"Best regards"
        ),
        cv_bullet_rewrites=[
            CVBulletRewrite(
                skill_highlighted="Leadership",
                original="Led team projects and coordinated deliverables.",
                rewritten="Directed cross-functional team of 6 to deliver 3 projects ahead of schedule, reducing time-to-completion by 18%.",
                why="Quantifies impact and introduces leadership vocabulary relevant to " + target_role,
            )
        ],
        linkedin_inmail=(
            f"Hi {job.hiring_manager_name},\n\n"
            f"I came across the {job.title} role at {job.company} and was immediately drawn to it. "
            f"I'm currently transitioning from {current_role} to {target_role} and believe my background "
            f"would bring a unique perspective to your team. Would you be open to a brief conversation?\n\n"
            f"Best, [Your Name]"
        ),
        interview_prep=[
            {
                "question": "Walk me through your transition from " + current_role + " to " + target_role,
                "model_answer": "Frame your pivot as deliberate, not reactive. Emphasise the skills that transfer and the concrete steps you've taken to close gaps.",
                "why_asked": "Hiring managers want to know this is a reasoned decision, not a desperate career change.",
            }
        ],
        positioning_statement=f"A {current_role} with a deliberate transition strategy to {target_role}, backed by targeted upskilling.",
        source="offline",
    )


def _offline_pivot_peers(current_role: str, target_role: str) -> List[PivotPeer]:
    colors = ["#0A66C2", "#117A37", "#8B45D4"]
    peers_data = [
        ("Alex M.", "AM", f"{current_role}", f"{target_role}", "Innovatech", 8,
         "Completed a professional certification in month 3, which unlocked the first interview.",
         "The hardest part was convincing myself it was possible. The skill overlap was real — I just needed evidence.", 1),
        ("Jordan L.", "JL", f"{current_role}", f"{target_role}", "NextGen Co.", 11,
         "Built two portfolio projects that demonstrated target-role skills to sceptical recruiters.",
         "Start networking before you feel ready. Half my opportunities came from conversations, not applications.", 2),
        ("Priya S.", "PS", f"{current_role}", f"{target_role}", "Vertex Labs", 7,
         "Took a hybrid contract role first — a stepping stone that gave them direct-path experience.",
         "The stepping stone wasn't a step back. It was the bridge that made the final jump possible.", 1),
    ]
    return [
        PivotPeer(
            name=name, initials=initials, avatar_color=colors[i],
            previous_role=prev, current_role=curr, company_now=company,
            months_to_pivot=months, key_milestone=milestone,
            testimonial=quote, connection_degree=degree,
        )
        for i, (name, initials, prev, curr, company, months, milestone, quote, degree) in enumerate(peers_data)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Online generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_job_listings(
    target_role: str,
    current_role: str,
    match_score: float,
    top_transfer: List[str],
    top_missing: List[str],
    cv_profile: Optional[Dict] = None,
    n_jobs: int = 4,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> List[JobListing]:
    """Generate n realistic LinkedIn-style job listings for the target role."""
    if not prefer_online:
        return _offline_job_listings(target_role, match_score)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_job_listings(target_role, match_score)

    cv_context = ""
    if cv_profile and cv_profile.get("extracted_role"):
        p = cv_profile
        cv_context = (
            f"\nCandidate: {p.get('extracted_role')}, {p.get('years_experience', 0):.0f} yrs, "
            f"{p.get('education_level', '')}. "
            f"Top skills: {', '.join(p.get('top_skills', [])[:5])}"
        )

    prompt = f"""Generate {n_jobs} realistic LinkedIn job listings for the role: {target_role}

CONTEXT:
- Candidate is transitioning FROM: {current_role}
- Overall skill match: {match_score:.0f}/100
- Transferable skills: {', '.join(top_transfer[:4])}
- Skills to develop: {', '.join(top_missing[:3])}
{cv_context}

Make the listings REALISTIC and VARIED:
- Mix of company sizes (startup, mid-market, enterprise)
- Mix of seniority (1 entry-level, 2 mid-senior, 1 senior/lead)
- Realistic salary ranges for the role and location
- Requirements that reflect the actual skill gaps identified
- Each job should feel distinctly different (different company culture, emphasis)

Respond ONLY with valid JSON:
{{
  "jobs": [
    {{
      "title": "Senior {target_role}",
      "company": "Company Name",
      "company_emoji": "🚀",
      "location": "City, State or Remote",
      "job_type": "Full-time",
      "salary_range": "$95k – $130k",
      "posted_ago": "2 days ago",
      "applicant_count": 127,
      "is_easy_apply": true,
      "match_score": 74,
      "key_requirements": ["Requirement 1", "Requirement 2", "Requirement 3", "Requirement 4"],
      "description_preview": "We are seeking a talented... (2-3 sentences, make it feel real)",
      "hiring_manager_name": "Alex Chen",
      "hiring_manager_title": "VP of Engineering",
      "network_connections": 2,
      "seniority": "Mid-Senior level"
    }}
  ]
}}

For match_score: estimate how well the candidate (given their background) matches each specific job.
Range: 50-92. Vary them realistically — not all jobs are equal matches.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=1200,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        raw_jobs = data.get("jobs", [])

        listings = []
        for i, j in enumerate(raw_jobs[:n_jobs]):
            listings.append(JobListing(
                id=f"job_{i}",
                title=str(j.get("title", target_role)),
                company=str(j.get("company", "Unknown")),
                company_emoji=str(j.get("company_emoji", "🏢")),
                location=str(j.get("location", "Remote")),
                job_type=str(j.get("job_type", "Full-time")),
                salary_range=str(j.get("salary_range", "Competitive")),
                posted_ago=str(j.get("posted_ago", "Today")),
                applicant_count=int(j.get("applicant_count", 100)),
                is_easy_apply=bool(j.get("is_easy_apply", False)),
                match_score=int(min(max(int(j.get("match_score", 65)), 0), 100)),
                key_requirements=[str(r) for r in j.get("key_requirements", [])[:4]],
                description_preview=str(j.get("description_preview", "")),
                hiring_manager_name=str(j.get("hiring_manager_name", "Hiring Manager")),
                hiring_manager_title=str(j.get("hiring_manager_title", "Team Lead")),
                network_connections=int(j.get("network_connections", 0)),
                seniority=str(j.get("seniority", "Mid-Senior level")),
            ))
        return listings if listings else _offline_job_listings(target_role, match_score)

    except Exception as e:
        return _offline_job_listings(target_role, match_score)


def generate_application_package(
    job: JobListing,
    current_role: str,
    target_role: str,
    cv_profile: Optional[Dict] = None,
    top_transfer: Optional[List[str]] = None,
    top_missing: Optional[List[str]] = None,
    model: str = "gpt-4o",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> ApplicationPackage:
    """Generate a complete, personalised application package for a specific job."""
    if not prefer_online:
        return _offline_application_package(job, current_role, target_role)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_application_package(job, current_role, target_role)

    cv_context = ""
    if cv_profile and cv_profile.get("extracted_role"):
        p = cv_profile
        cv_context = (
            f"\nCANDIDATE PROFILE (from CV):\n"
            f"- Role: {p.get('extracted_role')}\n"
            f"- Experience: {p.get('years_experience', 0):.0f} years\n"
            f"- Education: {p.get('education_level', 'Not specified')}\n"
            f"- Top skills: {', '.join(p.get('top_skills', [])[:6])}\n"
            f"Write about THIS SPECIFIC PERSON, not a generic candidate.\n"
        )

    transfer_str = ", ".join((top_transfer or [])[:5])
    missing_str = ", ".join((top_missing or [])[:3])

    # Use full job description if available (real job from SerpAPI), else preview
    _job_desc = (job.full_description or job.description_preview or "")[:3000]
    _real_note = (
        f"\nNOTE: This is a REAL job posting sourced from {job.apply_source or 'a live job board'}. "
        "Reference specific details from the job description in the cover letter and InMail.\n"
        if job.is_real_job else ""
    )

    prompt = f"""You are a senior career coach. Create a complete application package for this candidate.

JOB:
- Title: {job.title}
- Company: {job.company}
- Seniority: {job.seniority or "Not specified"}
- Location: {job.location}
- Job type: {job.job_type}
- Full job description:
{_job_desc}
{_real_note}

PIVOT:
- From: {current_role}
- To: {target_role}
- Match score: {job.match_score}/100
- Transferable strengths: {transfer_str}
- Gaps to address: {missing_str}
{cv_context}
Respond ONLY with valid JSON:
{{
  "positioning_statement": "One sentence: how to position this candidate for THIS specific job",
  "cover_letter": "4-5 paragraphs. Professional, specific, human. Reference the actual job requirements and the candidate's real background. Acknowledge the pivot honestly. Show the strategic nature of the transition. Do NOT use generic filler.",
  "cv_bullet_rewrites": [
    {{
      "skill_highlighted": "name of skill this targets",
      "original": "Original CV bullet (write a realistic one based on their background)",
      "rewritten": "Rewritten STAR-format bullet optimised for this role (quantify impact where possible)",
      "why": "One sentence: why this rewrite helps for this specific role"
    }}
  ],
  "linkedin_inmail": "A warm, specific 4-6 sentence InMail to {job.hiring_manager_name}. Reference specific details about the role and company. Show genuine interest and a clear ask. NOT a cold pitch.",
  "interview_prep": [
    {{
      "question": "Interview question they will likely face",
      "model_answer": "Concrete answer strategy (2-3 sentences). Be specific about what to say.",
      "why_asked": "Why this question matters for this specific role/company"
    }}
  ]
}}

Requirements:
- cv_bullet_rewrites: exactly 3 items
- interview_prep: exactly 5 items
- Be specific — mention actual skills, actual job requirements
- The cover letter should feel like it was written by a human for THIS job, not copy-pasted
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000,
        )
        data = json.loads(resp.choices[0].message.content or "{}")

        rewrites = []
        for r in data.get("cv_bullet_rewrites", [])[:3]:
            rewrites.append(CVBulletRewrite(
                skill_highlighted=str(r.get("skill_highlighted", "")),
                original=str(r.get("original", "")),
                rewritten=str(r.get("rewritten", "")),
                why=str(r.get("why", "")),
            ))

        return ApplicationPackage(
            job_id=job.id,
            job_title=job.title,
            company=job.company,
            cover_letter=str(data.get("cover_letter", "")),
            cv_bullet_rewrites=rewrites,
            linkedin_inmail=str(data.get("linkedin_inmail", "")),
            interview_prep=list(data.get("interview_prep", []))[:5],
            positioning_statement=str(data.get("positioning_statement", "")),
            source="online",
        )

    except Exception as e:
        pkg = _offline_application_package(job, current_role, target_role)
        pkg.source = f"offline (error: {repr(e)[:80]})"
        return pkg


def generate_pivot_peers(
    current_role: str,
    target_role: str,
    match_score: float,
    route_steps: Optional[List[str]] = None,
    n_peers: int = 3,
    model: str = "gpt-4o-mini",
    prefer_online: bool = True,
    api_key: Optional[str] = None,
) -> List[PivotPeer]:
    """Generate n anonymised 'pivot peer' success stories for this transition."""
    if not prefer_online:
        return _offline_pivot_peers(current_role, target_role)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _offline_pivot_peers(current_role, target_role)

    route_context = ""
    if route_steps and len(route_steps) > 2:
        route_context = f"\nA known stepping-stone path: {' → '.join(route_steps)}"

    prompt = f"""Generate {n_peers} realistic, anonymised career pivot success stories.

PIVOT: {current_role} → {target_role}
Skill overlap: {match_score:.0f}/100{route_context}

Make each story GENUINELY DIFFERENT:
- Different timelines (6-14 months)
- Different strategies (direct pivot, stepping stone, education, portfolio)
- Different barriers they overcame
- Different current companies (realistic names for the target role)

The testimonial quotes should feel authentic and specific — not generic motivation clichés.

Respond ONLY with valid JSON:
{{
  "peers": [
    {{
      "name": "Alex M.",
      "initials": "AM",
      "avatar_color": "#0A66C2",
      "previous_role": "{current_role}",
      "current_role": "{target_role}",
      "company_now": "Realistic company name",
      "months_to_pivot": 9,
      "key_milestone": "Specific action that unlocked the transition",
      "testimonial": "Authentic quote, 1-2 sentences, specific not generic",
      "connection_degree": 2
    }}
  ]
}}

avatar_color options: "#0A66C2", "#117A37", "#8B45D4", "#C37D16", "#B71C1C"
Vary the colours across the {n_peers} peers.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=800,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        raw = data.get("peers", [])

        peers = []
        for p in raw[:n_peers]:
            peers.append(PivotPeer(
                name=str(p.get("name", "A.N.")),
                initials=str(p.get("initials", "AN")),
                avatar_color=str(p.get("avatar_color", "#0A66C2")),
                previous_role=str(p.get("previous_role", current_role)),
                current_role=str(p.get("current_role", target_role)),
                company_now=str(p.get("company_now", "Tech Co.")),
                months_to_pivot=int(p.get("months_to_pivot", 9)),
                key_milestone=str(p.get("key_milestone", "")),
                testimonial=str(p.get("testimonial", "")),
                connection_degree=int(p.get("connection_degree", 2)),
            ))
        return peers if peers else _offline_pivot_peers(current_role, target_role)

    except Exception:
        return _offline_pivot_peers(current_role, target_role)
