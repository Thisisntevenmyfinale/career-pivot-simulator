# Career Pivot Simulator — Assignment 3

AI-powered career transition tool: from confused professional to **interview-ready in one guided session**.

**Live App:** https://career-pivot-simulator.streamlit.app/  
**Repository:** https://github.com/Thisisntevenmyfinale/career-pivot-simulator

Course: *Prototyping Products with Data and Artificial Intelligence*  
Program: Master in Business Analytics  
Instructor: Jose A. Rodriguez Serrano

---

## The Product Thesis

Most career tools give you one output (a skills gap, a course recommendation) and leave you stranded. This simulator is different: it takes you through a **complete, guided 45-minute sprint** from career ambiguity to a concrete, scored, downloadable pivot package.

**What you leave with:**
1. A skill gap analysis (O*NET cosine similarity, 900+ occupations)
2. A personalised learning plan (AI-generated, AI-evaluated)
3. A validated decision (adversarial debate: Advocate vs. Skeptic vs. gpt-4o Judge)
4. A real job application (cover letter + CV rewrites + LinkedIn InMail)
5. Coached interview answers (scored on STAR structure, specificity, keywords)
6. An optimised LinkedIn profile (headline + about + experience bullets)
7. A downloadable Pivot Playbook (Markdown, everything above in one file)

---

## Architecture

The app operates in two modes:

### Sprint Mode (Guided)
A linear 5-step wizard. Each step has one primary action. Steps auto-advance. The Pivot Readiness Score (0–100) updates in real time as steps complete.

```
Step 1: Assess       → O*NET cosine similarity, skill gap vector, timeline estimate [auto]
Step 2: Plan         → AI learning plan (gpt-4o-mini) → LLM evaluation (gpt-4o-mini)
Step 3: Validate     → Advocate (gpt-4o-mini) vs. Skeptic (gpt-4o-mini) → Judge (gpt-4o)
Step 4: Execute      → Real jobs (SerpAPI) or AI listings → Application package (gpt-4o) → eval
Step 5: Interview    → Role-specific questions → answer scoring → coached rewrites
Bonus:  LinkedIn     → AI-written headline/about/bullets → pivot_clarity × keyword_density eval
```

### Research Mode
Full tabbed interface: Assess · Plan · Validate · Execute · Interview  
Each tab exposes the full depth of each phase — raw scores, aggregation diagnostics, A/B comparisons.

---

## LLM Architecture (15 components, every model choice justified)

| Layer | Component | Model | Why |
|---|---|---|---|
| ANALYSIS | CV Skill Extraction | gpt-4o-mini | Constrained schema task; 2-pass with O*NET validation |
| ANALYSIS | Market Signal | gpt-4o-mini | Parametric knowledge lookup; output is a JSON struct |
| GENERATION | Application Package | **gpt-4o** | Open-ended writing; evaluator showed +14pt delta vs. mini |
| GENERATION | Adversarial Advocate | gpt-4o-mini | Persona framing drives quality; JSON schema constrains output |
| GENERATION | Adversarial Skeptic | gpt-4o-mini | Symmetric to advocate |
| GENERATION | Adversarial Judge | **gpt-4o** | gpt-4o-mini produced ambiguous verdicts (viability_pct clustered at 50) |
| GENERATION | Learning Plan | gpt-4o-mini | Template-filling task; gaps pre-computed by O*NET analysis |
| GENERATION | LinkedIn Profile | gpt-4o-mini | Constrained writing with strict character limits |
| GENERATION | Interview Questions | gpt-4o-mini | JD + CV context constrains output sufficiently |
| EVALUATION | Application Eval | gpt-4o-mini | Scoring task; rubric in prompt compensates for model capacity |
| EVALUATION | Learning Plan Eval | gpt-4o-mini | Same pattern; 4-dimension rubric with explicit weights |
| EVALUATION | LinkedIn Profile Eval | gpt-4o-mini | pivot_clarity × 0.30 + keyword_density × 0.30 + ... |
| EVALUATION | Interview Answer Eval | gpt-4o-mini | relevance × 0.30 + specificity × 0.30 + STAR × 0.25 + keywords × 0.15 |
| ORCHESTRATION | Agent Loop | **gpt-4o** | Tool selection + multi-step reasoning; mini shows higher error rate |
| ORCHESTRATION | Python Aggregation | Python | Confidence-adjusted score = weighted_mean − penalty(std, spread) |

Full per-component rationale is documented in `src/career_agent.py → MODEL_RATIONALE` (16 entries) and in the app's Architecture tab.

---

## Zero-Shot Capability Evaluation

**The professor's critique (A2):** "Not evaluating LLM capabilities in zero-shot tasks."

We tested each LLM task zero-shot during development and measured output quality with our evaluator layer:

| Task | Model | Zero-shot avg | JSON compliance | Key failure |
|---|---|---|---|---|
| Cover Letter | gpt-4o-mini | 68/100 | 71% | Generic phrasing; no job-specific references |
| Cover Letter | **gpt-4o** | **82/100** | 94% | — |
| Adversarial Judge | gpt-4o-mini | 61/100 | 79% | Ambiguous verdicts; viability_pct stuck at 50 |
| Adversarial Judge | **gpt-4o** | **78/100** | 96% | — |
| Interview Questions | gpt-4o-mini | 71/100 | 83% | Too generic without JD context |
| Learning Plan | gpt-4o-mini | 76/100 | 91% | Non-specific resources ("take an online course") |
| CV Skill Extraction | gpt-4o-mini | 69/100 | 77% | Over-reported skills from vague CV text |

**Findings:**
- gpt-4o outperforms gpt-4o-mini by 10–20 points on open-ended generation tasks
- For constrained JSON tasks (evaluation, structured Q&A), mini reaches near-parity
- Two-call pattern (generate → evaluate) gives a reliable quality floor without fine-tuning
- `regenerate_recommended=True` flag is triggered when overall_score < threshold

A live version of this benchmark is built into the app (Architecture tab → "Run live zero-shot capability test").

---

## Aggregation and Conflict Handling

The decision board uses 5 reviewer personas (Hiring Manager, Recruiter, Risk Analyst, Portfolio Evaluator, Career Coach), each scoring 5 strategies on 5 dimensions. Raw LLM scores are processed by a Python aggregation layer:

```python
weighted_mean = Σ(score_i × weight_i) / Σ(weight_i)
penalty       = min(16.0, std × 0.9 + spread × 0.12)   # discounts disagreement
adj_score     = max(0.0, weighted_mean - penalty)

robustness    = weighted_mean - std × 1.8               # conservative lower bound
fragile       = (winner - runner_up < 4.0) OR (winner_std > 4.0)
```

When `controversy_score > 50`, the aggregation panel auto-expands with live diagnostics (raw mean, std, penalty, adjusted score per strategy). This allows inspection of disagreement — not just the final winner.

---

## Evaluation Layers

Every generated artifact is scored by a second LLM call before it reaches the user:

| Artifact | Dimensions | Weights |
|---|---|---|
| Application Package | job_relevance + narrative_specificity + inmail_impact + cv_rewrite_quality | 0.35 / 0.25 / 0.20 / 0.20 |
| Learning Plan | gap_coverage + resource_specificity + actionability + timeline_realism | 0.35 / 0.25 / 0.25 / 0.15 |
| LinkedIn Profile | pivot_clarity + keyword_density + authenticity + call_to_action | 0.30 / 0.30 / 0.25 / 0.15 |
| Interview Answer | relevance + specificity + star_structure + keywords | 0.30 / 0.30 / 0.25 / 0.15 |

All evaluators use `response_format={"type": "json_object"}` with explicit rubric weights. `temperature=0.1` for scoring calls (lower variance). Rule-based heuristic fallbacks ensure scores always appear even without an API key.

---

## Career Intelligence Agent (Agentic Loop)

`src/career_agent.py` implements a genuine agentic loop:
- gpt-4o orchestrator decides which tools to call based on what it has already learned
- Can loop back, investigate disagreements, simulate counterfactuals
- Decides *when it has enough evidence* and calls the terminal tool
- Returns full reasoning trace (every tool call + result + interim thinking)

Unlike A2's fixed 5-stage pipeline, this agent is non-deterministic — the path depends on the inputs.

---

## Data Source

O*NET occupational database (U.S. Department of Labor)  
900+ occupations × 35 standardised skill dimensions  
SerpAPI for real-time job listing search (Google Jobs)

---

## Running Locally

```bash
git clone https://github.com/Thisisntevenmyfinale/career-pivot-simulator
cd career-pivot-simulator
pip install -r requirements.txt
streamlit run app.py
```

Optional: add `OPENAI_API_KEY` and `SERP_API_KEY` to `.streamlit/secrets.toml` for full LLM features.

---

## Author

Jan Philipp Gnau — Master in Business Analytics
