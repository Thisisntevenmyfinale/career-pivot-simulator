"""
Pivot Brief — The Shareable Career Pivot Dossier
=================================================
A one-page narrative document that synthesizes your entire pivot story.

Why this matters:
- Most career changers can't articulate their story in under 3 minutes
- Hiring managers make "pivot risk" decisions in the first 60 seconds
- A structured brief pre-empts every objection before the interview

The Pivot Brief includes:
1. The Pivot Narrative — your "why this, why now" in 3 sentences
2. O*NET Profile — your fit score + strongest transferable skills
3. Skill Gap Closure — what you've done / are doing to close gaps
4. Target Role Context — why you're targeting this role specifically
5. Market Timing — why now is a good time for this pivot
6. Your Unfair Advantage — what you bring that a native hire doesn't
7. 30-60-90 Day Plan — what you'd do in the first 90 days

Outputs: clean HTML/Markdown document ready to share or attach to applications.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_pivot_brief(
    current_role: str,
    target_role: str,
    years_experience: float = 0,
    top_transferable_skills: Optional[List[str]] = None,
    top_skill_gaps: Optional[List[str]] = None,
    fit_score: Optional[float] = None,
    candidate_name: str = "",
    upskilling_in_progress: Optional[List[str]] = None,
    key_achievements: Optional[List[str]] = None,
    target_companies: Optional[List[str]] = None,
    pivot_motivation: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Generate the complete Pivot Brief narrative document.

    Returns:
      {
        "pivot_narrative": str,         # 3 sentences: hook, bridge, forward
        "unfair_advantage": str,        # what pivot candidates uniquely bring
        "transferable_story": str,      # how your background maps to target role
        "gap_closure_plan": str,        # what you're doing to close the gaps
        "target_role_thesis": str,      # why THIS role, why THIS company type
        "thirty_sixty_ninety": {        # 30-60-90 day plan
          "day_30": str,
          "day_60": str,
          "day_90": str,
        },
        "elevator_pitch": str,          # 30-second verbal pitch
        "linkedin_headline": str,       # optimized LinkedIn headline for pivot
        "objection_handlers": List[{   # pre-empt common pivot objections
          "objection": str,
          "response": str,
        }],
        "one_line_brand": str,          # your professional brand statement
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "pivot_narrative": f"Experienced {current_role} transitioning to {target_role}, bringing a unique cross-domain perspective.",
        "unfair_advantage": "Add API key for personalized pivot brief generation.",
        "transferable_story": "",
        "gap_closure_plan": "",
        "target_role_thesis": "",
        "thirty_sixty_ninety": {
            "day_30": "Orient to team, stakeholders, and current priorities.",
            "day_60": "Deliver first independent contribution; identify highest-impact opportunity.",
            "day_90": "Own a meaningful workstream; propose a process or product improvement.",
        },
        "elevator_pitch": f"I'm a {current_role} moving into {target_role}.",
        "linkedin_headline": f"{current_role} → {target_role} | Career Transition",
        "objection_handlers": [],
        "one_line_brand": "Building bridges between domain expertise and new opportunities.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    transferable = ", ".join(top_transferable_skills[:6]) if top_transferable_skills else "not specified"
    gaps = ", ".join(top_skill_gaps[:4]) if top_skill_gaps else "not specified"
    upskilling = ", ".join(upskilling_in_progress[:3]) if upskilling_in_progress else "none specified"
    achievements = "\n".join([f"- {a}" for a in (key_achievements or [])[:4]]) or "not specified"
    companies = ", ".join(target_companies[:4]) if target_companies else "not specified"

    prompt = f"""You are a senior executive coach and career narrative specialist. Generate a complete Pivot Brief for this career changer.

CANDIDATE: {candidate_name or "the candidate"}
PIVOT: {current_role} ({years_experience:.0f} years) → {target_role}
FIT SCORE: {"%.0f" % fit_score + "th percentile on O*NET match" if fit_score else "not calculated"}
TOP TRANSFERABLE SKILLS: {transferable}
SKILL GAPS TO CLOSE: {gaps}
CURRENTLY UPSKILLING IN: {upskilling}
KEY CAREER ACHIEVEMENTS:
{achievements}
TARGET COMPANIES: {companies}
MOTIVATION (if given): {pivot_motivation or "not specified"}

Craft each section as if you're coaching a real person who needs to own their narrative in an interview.

The pivot narrative should have three parts:
1. HOOK — the insight that drove the pivot (not "I've always loved X")
2. BRIDGE — how the past makes them better at the target role (not just "transferable skills")
3. FORWARD — what specifically they want to build/achieve in the new role

The unfair advantage is the KEY differentiator — what a native hire CAN'T bring that this person can.

Objections should cover the 3 most common pivot objections for this specific transition.

Respond ONLY with valid JSON:
{{
  "pivot_narrative": "After 6 years building financial models that drove $200M in resource allocation decisions, I realized the highest-leverage version of my skills isn't in a spreadsheet — it's in a product that helps thousands of teams make better decisions at once. That insight pulled me toward product management. My background in quantitative reasoning and stakeholder communication isn't a workaround for my lack of 'traditional' PM experience — it's the part most MBAs who go straight into PM are still learning.",
  "unfair_advantage": "You bring a level of analytical rigor and domain credibility that most PMs have to fake. In rooms with finance leaders, data scientists, and executive stakeholders, you speak the language natively — and that removes a major friction point that slows most product decisions down.",
  "transferable_story": "Building financial models IS product work: you define requirements (what decision needs to be made?), design the interface (what inputs/outputs matter?), validate with users (does this answer the question they actually had?), and iterate based on feedback. The medium changes. The cognitive work is the same.",
  "gap_closure_plan": "Currently completing Google PM certification (completing April 2025). Shipped a personal project (budgeting tool, 200 users) to demonstrate product execution. Targeting roles with strong analytical components where the PM/Finance overlap is highest.",
  "target_role_thesis": "Fintech and data-tool companies need PMs who understand what their buyers actually do — not just user research proxies. My target is B2B SaaS with a finance or analytics buyer persona, where my domain experience is a feature, not a workaround.",
  "thirty_sixty_ninety": {{
    "day_30": "Shadow 3 customer calls per week. Map the current roadmap to customer pain. Deliver a structured read-out to the team on gaps I observe.",
    "day_60": "Own one active workstream end-to-end. Introduce one analytical framework that the team hasn't been using. Build relationships with 3 key engineering leads.",
    "day_90": "Propose a data-driven reprioritization of Q3 roadmap items. Ship one feature I own from spec to launch. Present results to VP Product."
  }},
  "elevator_pitch": "I spent 6 years as a financial analyst deciding where capital should go — now I'm building the products that help other people make those decisions. I'm moving into product management because it's the highest-leverage place to apply systems thinking at scale. I'm targeting fintech and analytics tools, where my domain experience is a direct asset.",
  "linkedin_headline": "Financial Analyst → Product Manager | Fintech & Data Tools | Systems Thinker | Google PM Certified",
  "objection_handlers": [
    {{
      "objection": "You don't have any PM experience on your CV.",
      "response": "I've been doing the cognitive work of product management for 6 years — defining requirements, iterating on feedback, aligning stakeholders on tradeoffs. I'm seeking my first formal PM title, not my first experience managing a product problem. Here's a specific example: [brief story about a decision that was essentially PM work]."
    }},
    {{
      "objection": "Why should we hire you over someone with 5 years of direct PM experience?",
      "response": "Because your buyers are finance teams, and I am your buyer. Most PMs spend 6 months building proxy understanding of what a CFO actually cares about. I don't need that ramp."
    }},
    {{
      "objection": "This is a big career change — what if it doesn't work out?",
      "response": "The risk is asymmetric. If I fail as a PM, I still have rare analytical skills to fall back on. If I stay in finance and discover at 40 that I should have pivoted, the risk is real. I've made a deliberate, data-driven decision — which is exactly the skill you're hiring for."
    }}
  ],
  "one_line_brand": "Financial analyst who builds products — not just models — that help teams make smarter decisions at scale."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1400,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online (gpt-4o)"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}


def format_pivot_brief_as_markdown(
    brief: Dict[str, Any],
    current_role: str,
    target_role: str,
    candidate_name: str = "",
    fit_score: Optional[float] = None,
) -> str:
    """Convert pivot brief dict to clean, shareable Markdown."""
    today = datetime.now().strftime("%B %Y")
    name_line = f"**{candidate_name}**  |  " if candidate_name else ""

    lines = [
        f"# The Pivot Brief",
        f"{name_line}{current_role} → {target_role}  |  *Generated {today}*",
        "",
        "---",
        "",
        f"## My Brand",
        f"> {brief.get('one_line_brand', '')}",
        "",
        f"## The Pivot Story",
        brief.get("pivot_narrative", ""),
        "",
        f"## My Unfair Advantage",
        brief.get("unfair_advantage", ""),
        "",
        f"## How My Background Maps",
        brief.get("transferable_story", ""),
        "",
        f"## How I'm Closing the Gaps",
        brief.get("gap_closure_plan", ""),
        "",
        f"## Why This Role, Now",
        brief.get("target_role_thesis", ""),
        "",
        "## 30-60-90 Day Plan",
    ]

    t = brief.get("thirty_sixty_ninety", {})
    if t.get("day_30"):
        lines.append(f"**Day 30:** {t['day_30']}")
    if t.get("day_60"):
        lines.append(f"\n**Day 60:** {t['day_60']}")
    if t.get("day_90"):
        lines.append(f"\n**Day 90:** {t['day_90']}")

    lines += [
        "",
        "## Elevator Pitch (30 seconds)",
        f"> {brief.get('elevator_pitch', '')}",
        "",
        f"## LinkedIn Headline",
        f"`{brief.get('linkedin_headline', '')}`",
    ]

    handlers = brief.get("objection_handlers", [])
    if handlers:
        lines += ["", "## Handling the Tough Questions"]
        for i, h in enumerate(handlers, 1):
            lines.append(f"\n**Q{i}: {h.get('objection', '')}**")
            lines.append(f"{h.get('response', '')}")

    if fit_score:
        lines += ["", "---", f"*O\\*NET Fit Score: {fit_score:.0f}th percentile*  |  *Source: {brief.get('source', '')}*"]

    return "\n".join(lines)


def generate_interview_war_room(
    company_name: str,
    role_title: str,
    interview_round: str = "first round",
    job_description: str = "",
    current_role: str = "",
    target_role: str = "",
    company_brief: Optional[Dict[str, Any]] = None,
    cv_summary: str = "",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    prefer_online: bool = True,
) -> Dict[str, Any]:
    """
    Interview War Room — complete pre-interview briefing document.

    Combines company intelligence + likely questions + prepared STAR answers
    + questions to ask + salary anchor into one battle-ready briefing.

    Returns:
      {
        "interview_brief": str,           # 3-sentence context summary
        "must_know_facts": List[str],     # 5 things about the company to know cold
        "likely_questions": List[{        # predicted questions with STAR answer frameworks
          "question": str,
          "why_asked": str,
          "star_framework": str,          # how to structure the answer
          "answer_hint": str,             # what to emphasize given their background
        }],
        "questions_to_ask": List[str],    # smart questions to ask the interviewer
        "salary_anchor_strategy": str,    # how to handle comp discussion
        "red_flags_to_probe": List[str],  # things to investigate during the interview
        "opening_statement": str,         # how to answer "tell me about yourself"
        "closing_statement": str,         # how to close the interview strongly
        "source": str,
      }
    """
    _fallback: Dict[str, Any] = {
        "interview_brief": f"Preparing for {interview_round} interview at {company_name} for {role_title}.",
        "must_know_facts": ["Research the company's latest news", "Know their product/service well", "Understand the role requirements"],
        "likely_questions": [],
        "questions_to_ask": ["What does success look like in the first 90 days?", "What are the biggest challenges the team is facing?"],
        "salary_anchor_strategy": "Add API key for salary anchor strategy.",
        "red_flags_to_probe": [],
        "opening_statement": "Add API key for personalized opening statement.",
        "closing_statement": "I'm genuinely excited about this opportunity and would love to continue the conversation.",
        "source": "offline",
    }

    if not prefer_online:
        return _fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception:
        return _fallback

    company_ctx = ""
    if company_brief:
        company_ctx = f"""
COMPANY INTELLIGENCE (pre-gathered):
- Stage: {company_brief.get('stage', 'unknown')}
- Hiring Signal: {company_brief.get('hiring_signal', 'unknown')}
- Culture: {company_brief.get('culture_snapshot', '')}
- Interview Process: {company_brief.get('interview_process', '')}
- Red Flags: {'; '.join(company_brief.get('red_flags', []))}
- Green Flags: {'; '.join(company_brief.get('green_flags', []))}
"""

    prompt = f"""You are an elite interview coach preparing a candidate for a high-stakes interview.

INTERVIEW: {interview_round.upper()} at {company_name} for {role_title}
CANDIDATE: {current_role} → {target_role}
CV SUMMARY: {cv_summary or "not provided"}
JOB DESCRIPTION: {job_description[:1000] if job_description else "not provided"}
{company_ctx}

Build a complete pre-interview war room briefing. Be specific. Reference real things about the company.
The questions must be role and company-specific, not generic.
The STAR frameworks should account for the pivot background — help them bridge their old experience to new context.

Respond ONLY with valid JSON:
{{
  "interview_brief": "You're walking into a first-round at {company_name} — likely a 45-minute screen with the hiring manager. This round filters for basic fit and communication. Your pivot background will come up immediately — lead with the narrative, not the apology.",
  "must_know_facts": [
    "Their last funding round and what it means for headcount",
    "The specific product/team you'd join and recent launches",
    "Their core metric (NPS, ARR, DAU) and public trajectory",
    "The hiring manager's background if visible on LinkedIn",
    "One thing in the JD that's unusual — ask about it"
  ],
  "likely_questions": [
    {{
      "question": "Walk me through your background and why you're making this pivot.",
      "why_asked": "Pivot candidates get this question in the first 3 minutes. They're assessing narrative coherence and risk.",
      "star_framework": "Don't use STAR here — use the Pivot Arc: Hook (insight that drove the change) → Bridge (how your past makes you better here) → Forward (what you want to build).",
      "answer_hint": "Lead with impact from your current role, then bridge to why {role_title} is the natural next step. Don't over-explain or apologize."
    }},
    {{
      "question": "Give me an example of a time you had to influence without authority.",
      "why_asked": "Cross-functional influence is critical for this role. They're testing whether you can operate without positional power.",
      "star_framework": "S: stakeholder landscape + competing priorities. T: what alignment was required. A: the specific approach you used. R: the outcome.",
      "answer_hint": "Use a story where you built consensus among people who had different incentives. Quantify the impact."
    }}
  ],
  "questions_to_ask": [
    "What's the single biggest thing you'd want the person in this role to accomplish in their first 90 days?",
    "What separates the people who thrive here from those who don't?",
    "What does the team find hardest about this problem space right now?",
    "How does the team typically handle disagreements about prioritization?"
  ],
  "salary_anchor_strategy": "Don't give a number first. If pressed, say: 'I've done research on market rates and I'm targeting the range of $X-Y based on the scope and location — I'd want to understand the full comp structure before committing to a number.' Then ask: 'What's the budgeted range for this role?'",
  "red_flags_to_probe": [
    "Ask about the last person in this role — why did they leave?",
    "Ask about team headcount changes in the last 12 months",
    "Ask what success looked like for the last person in the role"
  ],
  "opening_statement": "I spent 6 years in [current field] doing [core work]. The insight that drove my pivot was [specific insight]. I've spent the last [X months] preparing specifically for [target role] — I've [specific preparation action]. I'm targeting companies like {company_name} because [specific reason tied to company]. Here's what I'd do in my first 90 days...",
  "closing_statement": "Based on what we've discussed today, I'm genuinely excited about this role — specifically [X you learned]. I'd love to know what the next step looks like from your end, and what I could do to be the strongest candidate in your consideration set."
}}"""

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=1400,
        )
        result = json.loads(r.choices[0].message.content or "{}")
        result["source"] = "online (gpt-4o)"
        return result
    except Exception as e:
        return {**_fallback, "source": "online_error", "error": str(e)}
