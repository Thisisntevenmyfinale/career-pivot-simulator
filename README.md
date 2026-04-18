# Career Pivot Simulator — Assignment 3

**One goal: get the interview.**  
Not a collection of career tools. A single pipeline.

**Live App:** https://career-pivot-simulator.streamlit.app/  
**Repository:** https://github.com/Thisisntevenmyfinale/career-pivot-simulator

---

## The Pipeline

```
Upload CV  →  Find Jobs  →  Generate Portfolio  →  Debate + Rank  →  Interview Prep  →  Interview
   ↓              ↓                ↓                     ↓                  ↓
O*NET skill   SerpAPI live    gpt-4o × N jobs      Advocate vs.       Role-specific
mapping       + fit score     in parallel           Skeptic → Judge    questions +
              ranking         + gpt-4o-mini         hire_prob %        answer coaching
                              evaluation
```

**One button. ~60 seconds. Output: "Apply to Company X first (78% hire probability)."**

---

## What makes this technically non-trivial

### 1. Dual-LLM Generate → Evaluate Pattern (every artifact)

Nothing reaches the user unscored. For every output:

```
gpt-4o generates  →  gpt-4o-mini evaluates  →  score shown  →  regenerate if below threshold
```

This pattern runs for: cover letters, learning plans, LinkedIn profiles, interview answers.  
Empirically validated: gpt-4o scores +14pt higher than gpt-4o-mini on cover letters (82 vs 68 avg, n=3 zero-shot runs).

### 2. Parallel Application Portfolio Generation

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    results = list(ex.map(generate_and_evaluate, top_3_jobs))
```

Three applications generated + evaluated simultaneously. Not sequential.  
hire_probability = `0.65 × quality_score + 0.35 × O*NET_fit_score` — Python aggregation layer.

### 3. Adversarial 3-Agent Architecture

```
Advocate (gpt-4o-mini, parallel)  ─┐
                                    ├→  Judge (gpt-4o) → hire_probability_pct
Skeptic  (gpt-4o-mini, parallel)  ─┘
```

Used for: career pivot validation AND cover letter quality verdict.  
Judge reads both arguments — cannot ignore the strongest objection.

### 4. Cover Letter A/B Strategy Test

Original feature: generates two cover letters with different positioning strategies (Transferable Skills vs Growth Narrative), evaluates both with gpt-4o-mini, explains which strategy works better for the specific JD.

This is empirical zero-shot evaluation built into the product flow — not in a documentation tab.

### 5. Python Aggregation Layer

LLM outputs are never used raw:

```python
# Portfolio ranking
hire_prob = 0.65 × quality_score + 0.35 × fit_score

# Conflict handling (review board)
weighted_mean = Σ(score_i × weight_i) / Σ(weight_i)
penalty       = min(16.0, std × 0.9 + spread × 0.12)
adj_score     = max(0.0, weighted_mean - penalty)
controversy_score > 50  →  auto-expand diagnostics panel
```

### 6. O*NET Structured Data Foundation

900+ occupations × 35 standardised skill dimensions.  
Offline preprocessing (PCA, IDF weighting, cosine similarity matrix).  
Runtime: O(1) lookup. No LLM call needed for fit scoring.

---

## Two Entry Points, One Destination

### Path A — Application Portfolio (Quick Apply mode)

**You want to maximise your interview chances across multiple jobs.**

```
CV upload  →  "Find my best opportunities"
                        ↓
          SerpAPI: 5 live jobs (LinkedIn / Indeed / Glassdoor)
                        ↓
          O*NET fit score for each → top 3 selected
                        ↓
          ⚡ One click: "Launch Interview Pipeline"
                        ↓
          Parallel gpt-4o generation (3 applications simultaneously)
                        ↓
          gpt-4o-mini evaluation of each (4-dimension rubric)
                        ↓
          hire_probability = 0.65 × quality + 0.35 × fit (Python)
                        ↓
          Adversarial verdict on winner (Advocate + Skeptic + Judge)
                        ↓
          "Apply to Google first (78%). Then Stripe (64%). Skip KPMG for now."
                        ↓
          Interview prep for #1 job  →  Download Application Portfolio
```

**Or paste one specific job** for a targeted 90-second application.

### Path B — Career Sprint (Guided mode)

**You want to validate the pivot before committing.**

```
Step 1: Assess    → O*NET cosine similarity, skill gap vector, timeline estimate
Step 2: Plan      → AI learning plan (gpt-4o-mini) → LLM evaluation
Step 3: Validate  → Advocate vs. Skeptic vs. Judge → viability %
Step 4: Execute   → Real jobs (SerpAPI) + Application package (gpt-4o) → eval
Step 5: Interview → Role-specific questions → answer scoring → coached rewrites
Bonus:  LinkedIn  → AI-written headline/about/bullets → pivot_clarity × keyword_density eval
```

Pivot Readiness Score (0–100) updates after each step. Ends with a downloadable **Pivot Playbook**.

---

## Zero-Shot Capability Evaluation

Every model choice was tested empirically before being shipped:

| Task | Model Used | Zero-shot avg | Alt model avg | Delta | Key failure (alt) |
|---|---|---|---|---|---|
| Cover Letter | **gpt-4o** | 82/100 | 68/100 | +14pt | Generic phrasing; no job-specific refs |
| Adversarial Judge | **gpt-4o** | 78/100 | 61/100 | +17pt | Ambiguous verdicts; viability_pct = 50 |
| Learning Plan | gpt-4o-mini | 76/100 | — | — | Non-specific resources |
| Interview Questions | gpt-4o-mini | 71/100 | — | — | Too generic without JD context |
| CV Skill Extraction | gpt-4o-mini | 69/100 | — | — | Over-reported skills |
| Application Evaluation | gpt-4o-mini | 74/100 | — | — | Near-parity with full rubric |

Benchmark data stored as `ZERO_SHOT_BENCHMARK` constant in `app.py` and surfaced **inline at the point of generation** (Quality Score tab → "Why this output is reliable").

---

## LLM Architecture (15 components)

| Layer | Component | Model | Justification |
|---|---|---|---|
| ANALYSIS | CV Skill Extraction | gpt-4o-mini | Constrained schema; O*NET validation pass |
| ANALYSIS | Job Posting Parser | gpt-4o-mini | Structured extraction; schema enforced |
| GENERATION | Application Package | **gpt-4o** | +14pt vs mini in zero-shot test (82 vs 68) |
| GENERATION | A/B Cover Letter (×2) | **gpt-4o** | Strategy comparison; quality delta measured |
| GENERATION | Adversarial Advocate | gpt-4o-mini | Persona framing drives quality; JSON schema |
| GENERATION | Adversarial Skeptic | gpt-4o-mini | Symmetric to advocate |
| GENERATION | Adversarial Judge | **gpt-4o** | mini produced viability_pct = 50 (ambiguous) |
| GENERATION | Learning Plan | gpt-4o-mini | Template-filling; gaps pre-computed by O*NET |
| GENERATION | LinkedIn Profile | gpt-4o-mini | Constrained character limits; mini adequate |
| GENERATION | Interview Questions | gpt-4o-mini | JD + CV context constrains output |
| EVALUATION | Application Eval | gpt-4o-mini | 4-dimension rubric; scoring task |
| EVALUATION | Learning Plan Eval | gpt-4o-mini | Same pattern |
| EVALUATION | LinkedIn Eval | gpt-4o-mini | pivot_clarity × 0.30 + keyword_density × 0.30 + … |
| EVALUATION | Interview Answer Eval | gpt-4o-mini | relevance × 0.30 + STAR × 0.25 + … |
| ORCHESTRATION | Agent Loop | **gpt-4o** | Tool selection + multi-step reasoning |

Full rationale in `src/career_agent.py → MODEL_RATIONALE` (16 entries).

---

## Aggregation and Conflict Handling

```python
# Portfolio ranking (hire_probability)
hire_prob = int(min(95, max(15,
    quality_score * 0.65 + onet_fit_score * 0.35
)))
# Rationale: quality_score captures application writing (primary);
# fit_score captures structural role compatibility (secondary).
# Neither is used raw.

# Review board (career sprint)
weighted_mean = Σ(score_i × weight_i) / Σ(weight_i)
penalty       = min(16.0, std × 0.9 + spread × 0.12)
adj_score     = max(0.0, weighted_mean - penalty)
robustness    = weighted_mean - std × 1.8

# Conflict detection
controversy_score > 50  →  auto-expand diagnostics with raw std, penalty, adj per strategy
fragile = (winner - runner_up < 4.0) OR (winner_std > 4.0)
```

---

## Interview Readiness Score

Single 0–100 metric visible at all times in Quick Apply mode:

```
15 pts  CV uploaded + O*NET skill mapping complete
10 pts  Job(s) found/analyzed
20 pts  Application(s) generated
15 pts  Application quality bonus (prorated: quality_score 55→70 = +0→15)
15 pts  Adversarial verdict complete (hire_prob bonus)
10 pts  Interview questions generated
10 pts  At least one answer evaluated
──────
100 pts → Interview Ready
```

The score drives the "Next Action" banner — always one clear recommendation.

---

## Offline vs. Online Architecture

The app has a hard separation between preprocessing (runs once, ships with app) and runtime (no training at startup).

### Ships offline — no API key needed
| Component | What it does | Where |
|---|---|---|
| O*NET skill matrix | 900+ occupations × 35 skill dimensions, loaded from parquet | `artifacts/occupation_skill_matrix.parquet` |
| IDF weighting | Downweights universal skills (communication, critical thinking) | `build_cosine_core()` in `app.py` |
| Cosine similarity | IDF-weighted L2-normalised dot product, precomputed per query | `get_score_distribution()` in `app.py` |
| PCA coordinates | 2D embedding for map proximity scoring | `artifacts/pca_coords.parquet` |
| kNN graph | Dijkstra stepping-stone routing on cosine-similarity graph | `find_pivot_path()` in `src/model_logic.py` |
| Skill gap computation | `target − current` per dimension, deterministic | `compute_gap_df()` in `src/model_logic.py` |
| CV parsing fallback | Regex-based heuristic when OpenAI unavailable | `src/cv_parser.py` |

### Requires OPENAI_API_KEY
| Component | Model | Purpose |
|---|---|---|
| CV skill extraction | gpt-4o-mini | Map free-text CV to O*NET skill dimensions |
| Cover letter generation | gpt-4o | Open-ended writing (+14pt vs mini zero-shot) |
| Application evaluation | gpt-4o-mini | 4-dimension rubric scoring |
| Adversarial debate | gpt-4o-mini + gpt-4o | Advocate + Skeptic → Judge viability % |
| Learning plan | gpt-4o-mini | Gap-specific roadmap generation |
| Career agent | gpt-4o | Multi-step tool-calling pivot assessment |
| LinkedIn profile | gpt-4o-mini | Constrained character-limit generation |
| Interview Q&A coaching | gpt-4o-mini | Role + JD-specific question generation |

### Requires SERP_API_KEY (optional)
| Component | Fallback |
|---|---|
| Real job search (SerpAPI → Google Jobs) | Generates realistic simulated listings with `generate_job_listings()` |

### Graceful degradation
Every LLM call has a deterministic fallback. The app runs fully offline (heuristic mode) — no API key is required to explore gap analysis, stepping-stone routing, or skill investment simulation.

---

## Technical Stack

- **Data:** O*NET occupational database (US Dept. of Labor) — 900+ occupations × 35 skills
- **Job search:** SerpAPI → Google Jobs aggregator (LinkedIn, Indeed, Glassdoor)
- **LLM:** OpenAI gpt-4o + gpt-4o-mini (see architecture table above)
- **Parallelism:** `concurrent.futures.ThreadPoolExecutor` for portfolio generation and A/B testing
- **Offline preprocessing:** PCA, IDF weighting, cosine similarity matrix (ships with app)
- **Framework:** Streamlit (deployed to Streamlit Cloud)

---

## Running Locally

```bash
git clone https://github.com/Thisisntevenmyfinale/career-pivot-simulator
cd career-pivot-simulator
pip install -r requirements.txt
streamlit run app.py
```

Add to `.streamlit/secrets.toml` for full features:
```toml
OPENAI_API_KEY = "sk-..."
SERP_API_KEY = "..."   # serpapi.com — free tier: 100 searches/month
```

---

## Author

Jan Philipp Gnau — Master in Business Analytics  
Course: *Prototyping Products with Data and Artificial Intelligence*  
Instructor: Jose A. Rodriguez Serrano
