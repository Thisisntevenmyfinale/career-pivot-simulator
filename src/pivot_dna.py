"""
Pivot DNA — Your Persistent Career Identity
============================================
The fundamental problem with every AI career tool:
outputs sound like GPT, not like you.

Pivot DNA solves this by building a persistent profile from your CV:
- Your writing voice (sentence length, vocabulary richness, active vs. passive)
- Your strongest transferable story
- Your 3 most compelling achievement patterns
- Your terminology preferences (words you use vs. words you don't)
- Your natural framing style (quantitative? narrative? conceptual?)

Once built, this profile is injected into EVERY output:
cover letters, outreach, pivot briefs, interview prep.
The result sounds like you wrote it at your best — not like an AI template.

Additionally: Cohort Intelligence.
Based on the pivot path, we surface what people in the same situation
actually did, how long it took, and what made the difference.
This turns LLM estimates into benchmark-anchored guidance.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Voice Analysis (deterministic Python — no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_writing_voice(cv_text: str) -> Dict[str, Any]:
    """
    Deterministic analysis of the user's writing style from their CV.
    No LLM needed — pure text analytics.

    Returns voice profile used to calibrate all AI outputs.
    """
    if not cv_text or len(cv_text.strip()) < 100:
        return _default_voice()

    text = cv_text.strip()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    words_lower = [w.lower() for w in words]

    if not sentences or not words:
        return _default_voice()

    # Sentence length analysis
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 15

    # Vocabulary richness (type-token ratio)
    unique_words = set(words_lower)
    vocab_richness = len(unique_words) / max(len(words_lower), 1)

    # Active vs. passive markers
    passive_markers = ["was", "were", "been", "being", "is", "are", "had been"]
    active_action_verbs = [
        "led", "built", "created", "designed", "launched", "drove", "managed",
        "developed", "delivered", "achieved", "increased", "reduced", "generated",
        "established", "transformed", "spearheaded", "orchestrated", "implemented"
    ]
    passive_count = sum(words_lower.count(p) for p in passive_markers)
    active_count = sum(words_lower.count(v) for v in active_action_verbs)
    voice_lean = "active" if active_count > passive_count * 0.5 else "passive"

    # Quantitative tendency
    numbers_found = re.findall(r'\b\d+[%$]?|\$\d+|\d+[kKmM]?\b', text)
    quantitative_score = min(100, len(numbers_found) * 8)  # scale to 0-100

    # Formality level
    informal_words = ["amazing", "awesome", "passionate", "love", "excited", "keen", "eager"]
    formal_words = ["implemented", "facilitated", "collaborated", "demonstrated", "contributed"]
    informal_count = sum(words_lower.count(w) for w in informal_words)
    formal_count = sum(words_lower.count(w) for w in formal_words)
    formality = "formal" if formal_count >= informal_count else "conversational"

    # Power words extracted from CV (to reuse in outputs)
    power_words = [w for w in active_action_verbs if w in words_lower][:8]

    # Tone fingerprint from first 200 words (opening statement style)
    first_words = " ".join(words[:200])

    # Style profile
    if avg_sent_len <= 12:
        sentence_style = "punchy"
    elif avg_sent_len <= 18:
        sentence_style = "balanced"
    else:
        sentence_style = "detailed"

    return {
        "avg_sentence_length": round(avg_sent_len, 1),
        "sentence_style": sentence_style,      # "punchy" | "balanced" | "detailed"
        "vocabulary_richness": round(vocab_richness, 3),
        "voice_lean": voice_lean,               # "active" | "passive"
        "quantitative_score": quantitative_score,  # 0-100: how data-driven their writing is
        "formality": formality,                 # "formal" | "conversational"
        "power_words": power_words,             # verbs already in their CV
        "opening_style": first_words[:100],    # first words for tone reference
        "word_count": len(words),
    }


def _default_voice() -> Dict[str, Any]:
    return {
        "avg_sentence_length": 15,
        "sentence_style": "balanced",
        "vocabulary_richness": 0.6,
        "voice_lean": "active",
        "quantitative_score": 40,
        "formality": "formal",
        "power_words": ["led", "built", "delivered", "managed"],
        "opening_style": "",
        "word_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pivot DNA Builder (LLM-powered, uses voice profile)
# ─────────────────────────────────────────────────────────────────────────────

def build_pivot_dna(
    cv_text: str,
    current_role: str,
    target_role: str,
    years_experience: float = 0,
    top_skills: Optional[List[str]] = None,
    voice_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Build the user's Pivot DNA — their persistent career identity profile.

    This is called once per user and stored in session state.
    All subsequent AI outputs inject this DNA for voice consistency.

    Returns:
      {
        "strongest_transferable_argument": str,   # THE single best bridge argument
        "top_3_achievement_patterns": List[str],  # recurring strength patterns in CV
        "signature_phrases": List[str],           # phrases to reuse (from their CV)
        "avoid_phrases": List[str],               # generic phrases NOT to use
        "voice_instructions": str,                # one-paragraph style guide for LLM
        "pivot_hook": str,                        # 2-sentence "why this pivot" hook
        "unfair_advantage": str,                  # what native hires can't bring
        "writing_persona": str,                   # "data-driven analyst" / "systems thinker" etc
        "source": str,
      }
    """
    voice = voice_profile or analyze_writing_voice(cv_text)

    _fallback = {
        "strongest_transferable_argument": f"Deep {current_role} expertise creates a distinct advantage in {target_role} roles.",
        "top_3_achievement_patterns": ["Cross-functional leadership", "Quantitative problem-solving", "Stakeholder alignment"],
        "signature_phrases": voice.get("power_words", ["led", "built", "delivered"]),
        "avoid_phrases": ["passionate about", "team player", "results-driven"],
        "voice_instructions": f"Write in {voice.get('formality','formal')} tone with {voice.get('sentence_style','balanced')} sentences. Use active voice. Be quantitative where possible.",
        "pivot_hook": f"Experienced {current_role} making a deliberate move into {target_role}.",
        "unfair_advantage": f"Domain expertise from {current_role} that most {target_role} candidates lack.",
        "writing_persona": "experienced professional",
        "source": "offline",
    }

    if not prefer_online:
        return {**_fallback, **{"voice_profile": voice}}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return {**_fallback, **{"voice_profile": voice}}

    # Build voice style description for prompt
    style_desc = (
        f"Sentence style: {voice['sentence_style']} (avg {voice['avg_sentence_length']} words/sentence). "
        f"Voice: {voice['voice_lean']}. "
        f"Tone: {voice['formality']}. "
        f"Quantitative tendency: {voice['quantitative_score']}/100. "
        f"Power words found in CV: {', '.join(voice['power_words'][:5])}."
    )

    skills_str = ", ".join(top_skills[:8]) if top_skills else "not specified"
    cv_excerpt = cv_text[:2000] if cv_text else "not provided"

    prompt = f"""You are a master career narrative strategist. Analyze this person's CV and pivot to extract their unique Pivot DNA.

PIVOT: {current_role} ({years_experience:.0f} years experience) → {target_role}
TOP SKILLS: {skills_str}
WRITING VOICE ANALYSIS: {style_desc}

CV (excerpt):
{cv_excerpt}

Your job:
1. Find their STRONGEST transferable argument — the one insight that makes this pivot obvious and compelling, not apologetic
2. Identify 3 recurring achievement PATTERNS in their CV (not just skills — patterns like "sees systemic problems before others do" or "turns ambiguous situations into structured plans")
3. Extract 5 SIGNATURE PHRASES from their actual CV text — words and phrases that sound like them, not like a generic template
4. List 5 PHRASES TO AVOID — the generic filler language that weakens their narrative
5. Write VOICE INSTRUCTIONS: a one-paragraph style guide for any AI generating content for this person
6. Craft their PIVOT HOOK: 2 sentences that open their story — specific, confident, not apologetic
7. Define their UNFAIR ADVANTAGE: what they uniquely bring that a "native hire" with direct experience cannot

Be specific. Reference actual things from their CV. Make this feel like a real person, not a template.

Respond ONLY with valid JSON:
{{
  "strongest_transferable_argument": "...",
  "top_3_achievement_patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "signature_phrases": ["phrase 1", "phrase 2", "phrase 3", "phrase 4", "phrase 5"],
  "avoid_phrases": ["passionate about", "team player", "results-driven", "dynamic", "synergy"],
  "voice_instructions": "Write in a direct, data-first style. Sentences should be concise (under 18 words). Lead with outcomes before context. Use active verbs. Reference specific numbers, percentages, and timelines wherever they exist in the background. Avoid adjectives that can't be proven.",
  "pivot_hook": "Six years of turning financial complexity into strategic clarity taught me that the highest-leverage version of analytical thinking isn't in a model — it's in the product decisions those models inform. I'm moving into product management to operate at that leverage point directly.",
  "unfair_advantage": "You bring quantitative fluency and stakeholder credibility that most product managers spend years trying to fake. In rooms with data teams, finance leaders, and C-suite, you speak the language natively — which removes the single biggest friction point in cross-functional product leadership.",
  "writing_persona": "systems-minded analyst"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        result["voice_profile"] = voice
        return result
    except Exception as e:
        return {**_fallback, "voice_profile": voice, "source": "online_error", "error": str(e)}


def get_voice_injection_prompt(dna: Dict[str, Any]) -> str:
    """
    Returns a system-level style instruction to inject into any AI prompt
    to make outputs sound like the user, not like GPT.
    """
    if not dna:
        return ""

    parts = []
    if dna.get("voice_instructions"):
        parts.append(f"VOICE STYLE: {dna['voice_instructions']}")
    if dna.get("signature_phrases"):
        parts.append(f"USE THESE PHRASES (from candidate's own writing): {', '.join(dna['signature_phrases'][:4])}")
    if dna.get("avoid_phrases"):
        parts.append(f"AVOID THESE PHRASES (too generic): {', '.join(dna['avoid_phrases'][:4])}")
    if dna.get("writing_persona"):
        parts.append(f"WRITE AS: a {dna['writing_persona']}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Cohort Intelligence — What works for YOUR specific pivot
# ─────────────────────────────────────────────────────────────────────────────

def get_cohort_intelligence(
    current_role: str,
    target_role: str,
    years_experience: float = 0,
    location: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Cohort Intelligence: what people in your exact pivot situation actually experienced.

    Draws on LLM knowledge of career transition patterns, LinkedIn case studies,
    industry forums, and bootcamp/community outcome data.

    Returns:
      {
        "cohort_size_estimate": str,    # "~200-500 documented cases" or similar
        "median_timeline_weeks": int,   # typical weeks from start to first interview
        "median_applications": int,     # typical number of applications needed
        "warm_intro_rate": int,         # % who got in via referral vs. cold
        "most_common_entry_companies": List[str],  # companies that hire this pivot most
        "what_worked": List[str],       # top 3 things that distinguished success
        "what_failed": List[str],       # top 3 things that correlated with failure
        "biggest_misconception": str,   # most common wrong belief about this pivot
        "fastest_path": str,            # the quickest route to first interview
        "benchmark_context": str,       # 2-sentence "here's where you stand" read
        "your_week_estimate": str,      # personalised timeline estimate
        "confidence": str,              # "high" | "medium" | "low"
        "source": str,
      }
    """
    _fallback = {
        "cohort_size_estimate": "~100-300 documented cases",
        "median_timeline_weeks": 12,
        "median_applications": 20,
        "warm_intro_rate": 35,
        "most_common_entry_companies": [],
        "what_worked": ["Building a proof-of-skill portfolio", "Targeting warm intros over cold apply", "Narrowing to 5 target companies"],
        "what_failed": ["Sending 50+ cold applications without personalisation", "Waiting to be 'fully ready' before applying", "Targeting companies where the pivot is unusual"],
        "biggest_misconception": "That you need a certification or degree to be taken seriously in the new field.",
        "fastest_path": "Warm intro to a company where your background is a genuine asset, not a risk.",
        "benchmark_context": "Add API key for cohort-based benchmarks.",
        "your_week_estimate": "10-16 weeks is typical for this pivot path.",
        "confidence": "low",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    exp_bracket = (
        "junior (0-2 years)" if years_experience < 3 else
        "mid-level (3-6 years)" if years_experience < 7 else
        "senior (7+ years)"
    )

    prompt = f"""You are a career transition researcher with deep knowledge of how people successfully change careers.

PIVOT: {current_role} → {target_role}
EXPERIENCE: {exp_bracket}
LOCATION: {location or "US/global"}

Based on everything you know about this specific career transition — LinkedIn case studies, career community discussions, bootcamp outcomes, industry reports, hiring manager perspectives — provide realistic cohort benchmarks.

Be specific to THIS pivot path, not generic career change advice.
Flag when you have high vs. low confidence in specific numbers.
Name real companies where this pivot is known to happen successfully.

Respond ONLY with valid JSON:
{{
  "cohort_size_estimate": "~300-600 documented transitions in the last 3 years",
  "median_timeline_weeks": 14,
  "median_applications": 22,
  "warm_intro_rate": 40,
  "most_common_entry_companies": [
    "Mid-market SaaS (Series B-D) — most open to pivot candidates",
    "Fintech startups — value domain expertise over PM pedigree",
    "B2B analytics tools — where domain knowledge is the product"
  ],
  "what_worked": [
    "Shipping a public side project in the target domain within 60 days of starting the search",
    "Getting a warm intro vs. cold applying (3x higher conversion rate)",
    "Targeting companies where your domain background gives you a buyer-side advantage"
  ],
  "what_failed": [
    "Spending 3+ months getting certifications before applying anywhere",
    "Targeting FAANG or large-cap companies as first role (they rarely take career pivots)",
    "Generic cover letters that don't address the pivot risk directly"
  ],
  "biggest_misconception": "That you need 2 years of direct PM experience before anyone will hire you. The actual filter is whether you can demonstrate product thinking — which is demonstrable via side projects, case studies, and how you talk about past work.",
  "fastest_path": "Identify 5 companies where your domain background is a direct asset. Get one warm intro per week. Be prepared to do a take-home case. This path typically yields a first interview in 6-8 weeks.",
  "benchmark_context": "At the {exp_bracket} level, this pivot typically takes 12-18 weeks and 15-25 applications. You're in the median range if you've been searching 8+ weeks with 10+ applications and haven't had a first round yet — not behind schedule.",
  "your_week_estimate": "10-16 weeks to first offer, assuming 8-12 quality applications per month and at least one warm intro pathway active.",
  "confidence": "medium"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online (LLM cohort analysis)"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Proof-of-Skill Generator — Turn gaps into portfolio artifacts
# ─────────────────────────────────────────────────────────────────────────────

def generate_skill_proof(
    skill_name: str,
    skill_gap_level: float,       # current level 0-5
    skill_target_level: float,    # required level 0-5
    target_role: str,
    current_role: str = "",
    time_available_hours: int = 4,  # how many hours the user can spend
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate a concrete proof-of-skill artifact for a skill gap.

    Not "take a course" — a specific project that produces a
    portfolio artifact demonstrable in an interview within hours.

    Returns:
      {
        "project_title": str,
        "time_estimate_hours": int,
        "what_you_build": str,          # concrete deliverable
        "step_by_step": List[str],      # 4-6 specific steps
        "tools_needed": List[str],
        "artifact_format": str,         # "GitHub repo" | "Notion doc" | "Google Sheet" | etc
        "how_to_use_in_interview": str, # exact phrasing to reference in interview
        "linkedin_post_hook": str,      # 2-sentence hook for a LinkedIn post about it
        "cv_bullet": str,               # ready-to-paste CV bullet for this project
        "difficulty": str,              # "beginner" | "intermediate" | "advanced"
        "source": str,
      }
    """
    _fallback = {
        "project_title": f"Hands-on {skill_name} Project",
        "time_estimate_hours": time_available_hours,
        "what_you_build": f"A practical demonstration of {skill_name} competency.",
        "step_by_step": [
            f"Find a free {skill_name} tutorial or dataset",
            "Follow through the core concepts",
            "Apply to a real problem",
            "Document your work",
        ],
        "tools_needed": ["Free online tools"],
        "artifact_format": "GitHub repository",
        "how_to_use_in_interview": f"I recently built a project demonstrating {skill_name} — happy to walk you through it.",
        "linkedin_post_hook": f"Just shipped a {skill_name} project. Here's what I learned.",
        "cv_bullet": f"Built a hands-on {skill_name} project demonstrating practical application.",
        "difficulty": "intermediate",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    gap_desc = f"Current level: {skill_gap_level:.1f}/5 → Target: {skill_target_level:.1f}/5 (gap: {skill_target_level - skill_gap_level:.1f})"

    prompt = f"""You are a senior hiring manager and career coach. Design a proof-of-skill project that demonstrates {skill_name} competency for a {target_role} role.

SKILL: {skill_name}
GAP: {gap_desc}
CANDIDATE BACKGROUND: {current_role}
TIME AVAILABLE: {time_available_hours} hours
TARGET ROLE: {target_role}

Rules:
- The project must be completable in under {time_available_hours + 2} hours
- It must produce a CONCRETE ARTIFACT (not just "understanding")
- The artifact must be linkable or showable in an interview
- Use free tools only (no paid subscriptions required)
- The project should be specifically relevant to {target_role} work
- Be SPECIFIC: name exact datasets, tools, templates, frameworks

The output must be something a hiring manager would find genuinely impressive, not trivially easy.

Respond ONLY with valid JSON:
{{
  "project_title": "SQL Funnel Analysis: User Drop-off in a SaaS Product",
  "time_estimate_hours": 3,
  "what_you_build": "A 5-query SQL analysis of a public e-commerce dataset that identifies user drop-off points in the purchase funnel, with a 1-page executive summary of findings and 3 recommended product interventions.",
  "step_by_step": [
    "Download the public Olist e-commerce dataset from Kaggle (free, 100k rows, real transaction data)",
    "Load into DBeaver (free) or Google BigQuery sandbox (free tier)",
    "Write 5 queries: funnel conversion by stage, drop-off by device type, time-to-purchase distribution, repeat purchase rate by category, revenue concentration by customer decile",
    "Export results to a Google Sheet with clean visualizations (no code required)",
    "Write a 300-word executive summary: what you found, why it matters, what you'd do about it as a PM",
    "Publish the Google Sheet + summary as a public Notion page"
  ],
  "tools_needed": ["Kaggle (free dataset)", "DBeaver (free SQL client) or BigQuery", "Google Sheets", "Notion (free)"],
  "artifact_format": "Notion page with embedded Google Sheet + written summary",
  "how_to_use_in_interview": "I recently ran a funnel analysis on a real e-commerce dataset — I found a 34% drop-off at the payment step that correlated with mobile device type. I can walk you through the analysis and the product recommendations I drew from it. Here's the link.",
  "linkedin_post_hook": "I spent 3 hours turning a public dataset into a product insight. Here's the funnel drop-off I found — and what I'd do about it as a PM.",
  "cv_bullet": "Built SQL funnel analysis (5 queries, 100k-row dataset) identifying 34% payment drop-off; produced PM-ready executive summary with 3 data-backed product interventions",
  "difficulty": "intermediate"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Hiring Manager Dossier — Person-level intel
# ─────────────────────────────────────────────────────────────────────────────

def generate_hiring_manager_dossier(
    manager_name: str,
    company_name: str,
    role_title: str,
    manager_linkedin_url: str = "",
    manager_background_notes: str = "",
    current_role: str = "",
    target_role: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Person-level hiring intelligence: what does THIS person care about?

    Returns:
      {
        "background_summary": str,
        "likely_hiring_philosophy": str,  # inferred from background/writing
        "what_they_value": List[str],     # 3-4 specific things
        "language_to_use": List[str],     # phrases/framing that will resonate
        "language_to_avoid": List[str],   # framing that will turn them off
        "cover_letter_hook": str,         # one specific thing to open with
        "linkedin_comment_angle": str,    # how to engage their content first
        "interview_likely_focus": List[str],  # what they'll probe
        "one_line_read": str,
        "confidence": str,
        "source": str,
      }
    """
    _fallback = {
        "background_summary": f"Research {manager_name} on LinkedIn to build a complete picture.",
        "likely_hiring_philosophy": "Unable to determine without background information.",
        "what_they_value": ["Structured thinking", "Domain expertise", "Communication clarity"],
        "language_to_use": ["First-principles", "Impact", "Systems thinking"],
        "language_to_avoid": ["Passionate", "Team player", "Results-driven"],
        "cover_letter_hook": f"Reference something specific from {manager_name}'s public work.",
        "linkedin_comment_angle": "Engage with their most recent post before reaching out.",
        "interview_likely_focus": ["Problem-solving approach", "Cross-functional experience", "Specific domain knowledge"],
        "one_line_read": "Add API key for hiring manager analysis.",
        "confidence": "low",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    bg_context = manager_background_notes or f"Hiring manager at {company_name} for {role_title}"

    prompt = f"""You are an executive recruiter and people intelligence specialist.

HIRING MANAGER: {manager_name}
COMPANY: {company_name}
ROLE THEY'RE HIRING FOR: {role_title}
BACKGROUND INFO: {bg_context}
LINKEDIN: {manager_linkedin_url or "not provided"}
CANDIDATE PIVOT: {current_role} → {target_role}

Based on everything you know or can reasonably infer about this person's background, career, and public presence:
1. Describe their likely hiring philosophy — what kind of people do they hire and why
2. Identify what they personally value in a candidate (not generic — infer from their background)
3. What language/framing will resonate vs. fall flat
4. What is the single best hook to open a cover letter or message to them

Be honest about confidence level. Use "likely" and "probably" where inferring.
Flag if you have no specific knowledge of this person.

Respond ONLY with valid JSON:
{{
  "background_summary": "Engineering background → PM career suggests they value technical credibility. Has built 0-1 products, not just scaled existing ones. Published writing suggests they think in systems and trade-offs, not features.",
  "likely_hiring_philosophy": "Probably values structured thinking over intuition, believes in showing your work, and hires for potential to grow rather than perfect pattern-matching to the job description. Likely skeptical of candidates who can't get specific about past decisions.",
  "what_they_value": [
    "Structured thinking — shows your reasoning, not just conclusions",
    "Technical fluency — comfortable in engineering conversations",
    "First-principles approach — questions assumptions before building",
    "Speed of learning — how fast can you get up to speed in new domains"
  ],
  "language_to_use": [
    "Trade-offs and constraints",
    "Why I made this decision",
    "What I learned from what didn't work",
    "Systems thinking / second-order effects"
  ],
  "language_to_avoid": [
    "Passionate about user experience",
    "Cross-functional collaboration (too generic)",
    "Delivering results (says nothing)",
    "Excited about the opportunity"
  ],
  "cover_letter_hook": "Reference their public writing about [specific topic] — something like 'Your piece on building B2B products without a sales team changed how I think about distribution. That lens is directly what I'd bring to the roadmap for [role].'",
  "linkedin_comment_angle": "Engage with their most recent post on product strategy with a specific, substantive comment — not praise, a genuine addition to the argument. This establishes you as a peer, not a supplicant.",
  "interview_likely_focus": [
    "Walk me through a specific decision you made and why",
    "Tell me about a time you disagreed with leadership and what happened",
    "How do you decide what NOT to build"
  ],
  "one_line_read": "Lead with structure and specificity. This person is not moved by enthusiasm — they're moved by evidence of rigorous thinking.",
  "confidence": "medium"
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=900,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online (gpt-4o)"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}
