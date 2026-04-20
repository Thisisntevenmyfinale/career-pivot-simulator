"""
Demo Profile — Alex Müller
===========================
A fully pre-baked career pivot profile for testers and demos.

Persona: Senior Digital Marketing Manager → Product Manager (SaaS)

This is one of the most common, credible, and relatable pivots:
  - Strong data + analytics background maps directly to PM work
  - Cross-functional project leadership is transferable
  - User research and customer insights overlap heavily
  - Gap: no formal roadmap ownership, no engineering collaboration title

The profile is intentionally realistic, not perfect:
  - 74th percentile O*NET fit (not a slam dunk — gap exists and is interesting)
  - One clear skill gap (technical product management / engineering basics)
  - Mock interview score around 68 — room to improve
  - 8 applications, below cohort median — pipeline needs work

Why this persona works for demos:
  - Every single feature in PivotOS produces meaningful, specific output
  - The pivot argument is strong but not obvious — debate tools are interesting
  - The rejection pattern (phone screen rejections) triggers the Interpreter
  - OPS lands around 45-52% — shows real signal without being discouraging
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────────────
# CV Text — full realistic document
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CV_TEXT = """Alex Müller
Senior Digital Marketing Manager
alex.mueller@email.com | LinkedIn: linkedin.com/in/alexmuller | Berlin, Germany

PROFESSIONAL SUMMARY
Results-driven Senior Digital Marketing Manager with 7 years of experience at B2B SaaS companies.
Specialised in data-driven growth marketing, product-led growth campaigns, and cross-functional
initiative leadership. Consistently delivered measurable pipeline impact through disciplined
experimentation — 47 A/B tests shipped in 2023, 12% average conversion lift. Now transitioning
to Product Management to build the products I've been marketing.

EXPERIENCE

Senior Digital Marketing Manager | ScaleOps GmbH (Berlin, Series C SaaS) | 2021 – Present
- Owned the entire demand generation pipeline: €4.2M annual budget, 340% YoY pipeline growth
- Defined and prioritised the marketing feature roadmap in collaboration with the Product team;
  wrote 8 detailed PRDs for CRM integration features based on customer interview insights
- Led cross-functional growth squad (6 engineers, 2 designers, 3 marketers) to ship a
  self-serve onboarding flow — reduced time-to-first-value from 14 days to 3 days
- Conducted 60+ customer discovery interviews; synthesised findings into product positioning
  docs adopted directly by the PM team for roadmap prioritisation
- Managed 3 external development agencies; defined acceptance criteria, ran sprint reviews,
  approved technical deliverables

Digital Marketing Manager | Zendesk (Dublin Office) | 2019 – 2021
- Built Zendesk's EMEA demand generation function from scratch; grew MQL volume 280% in 18 months
- Launched 4 product features to market in collaboration with Product and Engineering
- Owned NPS survey programme: 2,400 responses/quarter, presented insights to C-suite quarterly
- Created and managed the Marketing API integration with Salesforce; wrote the technical spec,
  coordinated with 2 engineering teams across timezones

Marketing Analyst | HubSpot (Dublin) | 2017 – 2019
- Built the EMEA marketing analytics stack (Amplitude, Mixpanel, Looker); now used by 40+ people
- Led competitive intelligence programme — weekly briefings consumed by Product leadership
- Automated reporting infrastructure: saved 12 hours/week across the marketing team
- Co-designed HubSpot's EMEA Content Strategy with the Product Marketing and PM teams

EDUCATION
MSc Marketing Analytics | Trinity College Dublin | 2017
BSc Business Administration | Goethe University Frankfurt | 2015

SKILLS & TOOLS
Analytics & Data: SQL, Python (pandas, basic), Amplitude, Mixpanel, Looker, Google Analytics 4
Product: JIRA, Confluence, Figma (proficient), UserTesting, Hotjar, Productboard
Marketing: HubSpot, Salesforce, Marketo, Google Ads, LinkedIn Ads, Intercom
Research: Customer interviews (JTBD framework), NPS, survey design, competitive analysis
Process: Agile/Scrum (sprint planning, retrospectives), OKR setting, stakeholder management

CERTIFICATIONS
- Reforge: Product-Led Growth Certificate (2023)
- Google Analytics Certified
- HubSpot Marketing Certification

SIDE PROJECTS
- Built a Chrome extension for marketers to extract UTM parameters (200+ active users, 4.2★)
- Maintains a newsletter on PLG strategy (1,400 subscribers, biweekly)
- Mentor at Women in Tech Berlin (quarterly 1:1 career coaching sessions)
"""


# ─────────────────────────────────────────────────────────────────────────────
# CV Profile — structured data (what parse_cv would return)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CV_PROFILE: Dict[str, Any] = {
    "extracted_role":         "Senior Digital Marketing Manager",
    "years_experience":       7,
    "top_skills": [
        "Data Analysis",
        "Customer Research",
        "Project Management",
        "Product Requirements",
        "A/B Testing",
        "Stakeholder Management",
        "SQL",
        "Agile/Scrum",
        "Cross-functional Leadership",
        "Product Strategy",
    ],
    "skills_mapped_count":    10,
    "education":              "MSc Marketing Analytics, Trinity College Dublin",
    "industries":             ["SaaS", "B2B", "Marketing Technology"],
    "companies":              ["ScaleOps GmbH", "Zendesk", "HubSpot"],
    "languages":              ["English", "German"],
    "certifications":         ["Reforge PLG Certificate", "Google Analytics"],
    "key_achievements": [
        "340% YoY pipeline growth with €4.2M budget",
        "Led cross-functional squad that reduced time-to-first-value from 14 to 3 days",
        "Wrote 8 product PRDs adopted directly by the PM team",
        "Built analytics stack now used by 40+ people",
        "47 A/B tests shipped in 2023, 12% average conversion lift",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Pivot DNA — what build_pivot_dna would return
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PIVOT_DNA: Dict[str, Any] = {
    "pivot_hook":
        "I've spent 7 years building the demand for products I didn't control. "
        "Now I want to build the products.",

    "strongest_transferable_argument":
        "I've written product requirements, led cross-functional squads, and shipped features "
        "in collaboration with engineering teams — the only thing missing from my PM title is "
        "the title itself. I own the outcome, I run the sprint reviews, I prioritise the "
        "roadmap. Product Management is a formalisation of work I'm already doing.",

    "unfair_advantage":
        "I understand user behaviour at a depth most PMs don't. 7 years of analytics, "
        "60+ customer interviews, and direct ownership of NPS programmes means I can "
        "walk into any product decision with quantified user signal — not assumptions.",

    "writing_tone":
        "Direct, data-grounded, without corporate filler. Short sentences. "
        "Specific numbers over vague claims. Confident but not arrogant.",

    "career_narrative":
        "Started as a marketing analyst building data infrastructure, moved to managing "
        "growth at Zendesk and HubSpot, then took ownership of a cross-functional squad "
        "at ScaleOps. The through-line is always: define the outcome, gather the signal, "
        "ship the thing. Product Management is the natural next frame for that skill set.",

    "three_word_brand":     "Data. Cross-functional. Shipped.",
    "pivot_risk":           "No formal PM title — teams may underestimate the depth of product work already done",
    "mitigation":           "Lead with the 8 PRDs, the squad leadership, and the PLG certification. Show, don't tell.",
    "target_companies":     ["Linear", "Notion", "Personio", "Contentful", "GetYourGuide"],
    "target_company_type":  "Series B–D product-led SaaS, ideally B2B, Berlin/remote preferred",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cohort Intelligence — what get_cohort_intelligence would return
# ─────────────────────────────────────────────────────────────────────────────

DEMO_COHORT: Dict[str, Any] = {
    "pivot_description":         "Marketing Manager → Product Manager (B2B SaaS)",
    "median_timeline_weeks":     14,
    "median_applications":       32,
    "median_interviews":         6,
    "typical_first_role_level":  "PM II / Mid-level PM",
    "what_worked": (
        "Candidates who made this pivot successfully typically led with a specific product "
        "decision they owned — not 'I worked with PM'. The PLG certificate helped with "
        "credibility. Being explicit about A/B test design and interpretation resonated "
        "with hiring managers who wanted data-first PMs."
    ),
    "what_failed": (
        "Generic pivots that led with 'I've always been interested in product' "
        "got filtered immediately. Applications that didn't show existing PM-adjacent work "
        "were rejected at the recruiter screen."
    ),
    "salary_expectation": "€70,000–€95,000 base depending on company stage and location",
    "key_differentiator": "Portfolio of real product decisions + measurable user impact data",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline jobs — 8 realistic applications
# ─────────────────────────────────────────────────────────────────────────────

_today = date.today()

DEMO_PIPELINE: list = [
    {
        "id": "demo_1",
        "title": "Product Manager",
        "company": "Linear",
        "status": "phone_screen",
        "date_added": (_today - timedelta(days=18)).isoformat(),
        "date_updated": (_today - timedelta(days=12)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "Recruiter call done. Waiting for decision.",
    },
    {
        "id": "demo_2",
        "title": "Product Manager — Growth",
        "company": "Personio",
        "status": "applied",
        "date_added": (_today - timedelta(days=14)).isoformat(),
        "date_updated": (_today - timedelta(days=14)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "",
    },
    {
        "id": "demo_3",
        "title": "Associate Product Manager",
        "company": "Contentful",
        "status": "rejected",
        "date_added": (_today - timedelta(days=22)).isoformat(),
        "date_updated": (_today - timedelta(days=16)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "Feedback: 'Went with candidate with more direct PM experience'",
    },
    {
        "id": "demo_4",
        "title": "PM — Platform",
        "company": "GetYourGuide",
        "status": "applied",
        "date_added": (_today - timedelta(days=10)).isoformat(),
        "date_updated": (_today - timedelta(days=10)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "",
    },
    {
        "id": "demo_5",
        "title": "Product Manager",
        "company": "SumUp",
        "status": "rejected",
        "date_added": (_today - timedelta(days=28)).isoformat(),
        "date_updated": (_today - timedelta(days=21)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "No feedback provided.",
    },
    {
        "id": "demo_6",
        "title": "Growth PM",
        "company": "Pitch",
        "status": "applied",
        "date_added": (_today - timedelta(days=6)).isoformat(),
        "date_updated": (_today - timedelta(days=6)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "",
    },
    {
        "id": "demo_7",
        "title": "Product Manager",
        "company": "Tier Mobility",
        "status": "viewed",
        "date_added": (_today - timedelta(days=9)).isoformat(),
        "date_updated": (_today - timedelta(days=7)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "Profile viewed on LinkedIn.",
    },
    {
        "id": "demo_8",
        "title": "PM — Analytics",
        "company": "Adjust",
        "status": "first_round",
        "date_added": (_today - timedelta(days=20)).isoformat(),
        "date_updated": (_today - timedelta(days=5)).isoformat(),
        "source": "demo",
        "cover_letter": "",
        "notes": "First round interview scheduled. Prep needed.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Outcome log — 3 closed outcomes (enough to trigger calibration)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_OUTCOME_LOG: list = [
    {
        "id": "demo_3",
        "job_title": "Associate Product Manager",
        "company": "Contentful",
        "predicted_roi": 0.24,
        "actual_stage": "phone_screen",
        "reached_response": True,
        "reached_interview": False,
        "reached_final": False,
        "is_offer": False,
        "notes": "Went with candidate with more direct PM experience",
        "date": (_today - timedelta(days=16)).isoformat(),
    },
    {
        "id": "demo_5",
        "job_title": "Product Manager",
        "company": "SumUp",
        "predicted_roi": 0.19,
        "actual_stage": "no_response",
        "reached_response": False,
        "reached_interview": False,
        "reached_final": False,
        "is_offer": False,
        "notes": "",
        "date": (_today - timedelta(days=21)).isoformat(),
    },
    {
        "id": "demo_9",
        "job_title": "PM — Data",
        "company": "N26",
        "predicted_roi": 0.31,
        "actual_stage": "phone_screen",
        "reached_response": True,
        "reached_interview": False,
        "reached_final": False,
        "is_offer": False,
        "notes": "Strong call but no follow-up",
        "date": (_today - timedelta(days=25)).isoformat(),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Mock interview report — pre-baked, realistic
# ─────────────────────────────────────────────────────────────────────────────

DEMO_MOCK_INTERVIEW: Dict[str, Any] = {
    "overall_score": 68,
    "hire_recommendation": "Maybe — strong background, pivot narrative needs sharpening",
    "one_line_verdict": "Data strength is clear; needs to own the product decisions more assertively",
    "dimension_scores": {
        "STAR structure":          72,
        "Pivot narrative clarity":  61,
        "Technical depth":          58,
        "Stakeholder handling":     78,
        "Cultural fit signals":     71,
    },
    "top_improvements": [
        "Lead with product decisions owned — not collaboration with the PM team",
        "Sharpen the 'why PM now' answer — currently too long and hedging",
        "Add one technical PM example (API design tradeoff, build vs. buy decision)",
    ],
    "strong_answers": [
        "The cross-functional squad leadership story (time-to-value from 14→3 days)",
        "NPS programme ownership and C-suite presentation",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Occupation mappings (used to pre-set the selectboxes)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CURRENT_OCC  = "Marketing Managers"
DEMO_TARGET_OCC   = "Computer and Information Systems Managers"


# ─────────────────────────────────────────────────────────────────────────────
# Full load function — injects everything into session state
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_profile(state: Any) -> None:
    """
    Inject the full demo profile into session_state.
    Safe to call multiple times — idempotent.
    """
    from src.outcome_tracker import compute_calibration

    state["cv_text"]               = DEMO_CV_TEXT
    state["cv_profile"]            = DEMO_CV_PROFILE
    state["pivot_dna"]             = DEMO_PIVOT_DNA
    state["cohort_intelligence"]   = DEMO_COHORT
    state["pipeline_jobs"]         = list(DEMO_PIPELINE)
    state["outcome_log"]           = list(DEMO_OUTCOME_LOG)
    state["mock_interview_report"] = DEMO_MOCK_INTERVIEW
    state["interview_prep_done"]   = True
    state["momentum_streak_days"]  = 5
    state["momentum_last_date"]    = (date.today() - timedelta(days=1)).isoformat()
    state["demo_mode"]             = True
    state["demo_current_occ"]      = DEMO_CURRENT_OCC
    state["demo_target_occ"]       = DEMO_TARGET_OCC

    # Recompute calibration from the pre-baked outcome log
    state["calibration_data"] = compute_calibration(list(DEMO_OUTCOME_LOG))

    # Reset daily brief so it regenerates with demo data
    state["daily_brief_date"]    = ""
    state["daily_brief_content"] = None
    # Reset OPS delta baseline
    state["ops_previous"] = None
