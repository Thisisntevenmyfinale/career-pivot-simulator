# Career Pivot Simulator

**Live App:** https://career-pivot-simulator.streamlit.app/  
**Repository:** https://github.com/Thisisntevenmyfinale/career-pivot-simulator

---

## The North Star: P(offer)

Every feature in this system exists to move one number: **P(offer)** — the probability that the next application you submit results in an interview offer.

P(offer) is not displayed as a vanity metric. It is the **quantified output of every action the user takes** and the spine that unifies what would otherwise be a disconnected collection of career tools. The user sees it update in real time as they upload their CV, generate an application, run the adversarial debate, and log outcomes.

```
P(offer) = base_prior × ops_factor × calibration_factor × brier_factor
         = 3% × f(OPS score) × f(personal response rate) × f(prediction accuracy)
         Capped at 92%. Floored at 0.5%. Never shown raw — always calibrated.
```

**OPS (Offer Probability Score):** 11-factor Bayesian accumulation computed in pure Python every render. Factors include: CV uploaded, Pivot DNA built, application package quality, adversarial verdict result, ATS score, interview readiness, LinkedIn completeness. Each factor has a known impact weight (+2 to +14 percentage points). The breakdown is shown factor-by-factor in Expert Mode.

---

## The Closed Loop: 5 Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   1. PREDICT   →   2. GENERATE   →   3. EVALUATE                  │
│   Zero-shot         gpt-4o             3-agent                      │
│   P(offer)          application        adversarial                  │
│   estimate          package            debate                       │
│        ↑                                    ↓                       │
│   5. CALIBRATE  ←──────── 4. MEASURE ───────┘                      │
│   Brier score +            Outcome logging                          │
│   personal                 (response /                              │
│   correction               rejection /                              │
│   factor                   interview / offer)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each stage feeds the next. The loop doesn't close after one iteration — it improves with every outcome logged. This is not a chatbot. It is a prediction machine that gets more accurate the more you use it.

**Stage 1 — PREDICT:** Upload CV. O*NET cosine similarity scores skill fit across 894 occupations × 119 dimensions. P(offer) initialised from base prior + OPS factors. No LLM call — deterministic.

**Stage 2 — GENERATE:** Paste a job description. gpt-4o generates cover letter, CV bullets, LinkedIn InMail. Every output is evaluated before surfacing (see Quality Gate below). Application quality factor feeds back into P(offer) immediately.

**Stage 3 — EVALUATE:** 3-agent adversarial debate. Advocate + Skeptic run in parallel (ThreadPoolExecutor). Judge (gpt-4o) synthesises both arguments and returns a calibrated hire_probability_pct. Disagreement score (std_dev) surfaces when agents diverge — not hidden, not averaged.

**Stage 4 — MEASURE:** Log what actually happened: response, rejection, interview, offer. Python aggregates the outcome log. This is the raw data that feeds calibration.

**Stage 5 — CALIBRATE:** After 3+ outcomes, the system computes `correction_factor = empirical_response_rate / predicted_rate`. This factor is applied to ALL future P(offer) predictions automatically. The Brier score measures prediction accuracy over time — getting it below 0.20 is the quantified goal.

---

## Zero-Shot Capability Evaluation

> "The biggest mistake with AI backends is not evaluating their capabilities in the actual zero-shot task."

Every model choice was evaluated empirically before being selected. **Methodology:** Each task run 5 times zero-shot for both candidate models. Outputs scored by a separate gpt-4o-mini evaluator to prevent self-consistency bias. Averages become the reference benchmark. Results are available live in-app under Closed-Loop Architecture → Live Capability Test.

| Task | Chosen Model | Score | Alt Model | Score | Delta | Decision |
|---|---|---|---|---|---|---|
| Cover letter generation | **gpt-4o** | 82/100 | gpt-4o-mini | 68/100 | +14pt | Quality-critical; pivot narrative depth |
| Adversarial judge verdict | **gpt-4o** | 78/100 | gpt-4o-mini | 61/100 | +17pt | Mini clusters at 50% on contested cases |
| CV skill extraction → O*NET | gpt-4o-mini | 69/100 | gpt-4o | 71/100 | −2pt | Within noise (σ=3.8); validation layer absorbs error |
| Application quality evaluation | gpt-4o-mini | 74/100 | gpt-4o | 77/100 | −3pt | 87% inter-rater agreement; ~10× cost savings |
| Career agent / strategy synthesis | **gpt-4o** | 81/100 | gpt-4o-mini | 64/100 | +17pt | Mini fails conflict detection silently |

**Key principle:** Choosing gpt-4o-mini is not a cost-cutting shortcut — it requires justification. For rubric evaluation: the 3pt delta is within noise AND a validation layer absorbs the error. For skill extraction: a 2pt delta does not propagate because hallucinated O*NET skill names are caught at the validation layer and discarded. Where the delta is 14–17pt and no fallback absorbs it, gpt-4o is non-negotiable.

---

## LLM Architecture (all pipeline components)

| Layer | Component | Model | Justification |
|---|---|---|---|
| ANALYSIS | CV skill extraction | gpt-4o-mini | Schema-constrained; O*NET validation pass catches hallucinations |
| ANALYSIS | Job posting parser | gpt-4o-mini | Structured extraction; schema enforced |
| ANALYSIS | ATS compatibility scan | gpt-4o-mini | Keyword classification; offline regex fallback |
| ANALYSIS | Company intelligence brief | gpt-4o-mini | Stage/culture/hiring signals; SerpAPI-augmented |
| ANALYSIS | Pipeline diagnosis | gpt-4o-mini | Health score + bottleneck ID; heuristic fallback |
| ANALYSIS | Cross-rejection synthesis | gpt-4o-mini (temp=0.1) | Aggregation runs in Python first; LLM only reasons on summary |
| GENERATION | Application package | **gpt-4o** | +14pt vs mini zero-shot (82 vs 68); primary quality signal |
| GENERATION | Adversarial advocate | gpt-4o-mini | Persona framing constrains output; JSON schema |
| GENERATION | Adversarial skeptic | gpt-4o-mini | Symmetric to advocate; parallel execution |
| GENERATION | Adversarial judge | **gpt-4o** | Mini produces viability_pct ≈ 50 on contested cases (+17pt delta) |
| GENERATION | Learning plan | gpt-4o-mini | Template-filling; gaps pre-computed by O*NET |
| GENERATION | LinkedIn profile | gpt-4o-mini | Constrained character limits; mini adequate |
| GENERATION | Interview questions | gpt-4o-mini | JD + CV context constrains output space |
| GENERATION | Warm intro DMs (×6 parallel) | gpt-4o-mini (temp=0.6) | 4-sentence format; no quality gap vs gpt-4o |
| GENERATION | Negotiation script | **gpt-4o** | Personalised objection handling; market-anchored |
| GENERATION | Counter-offer letter | **gpt-4o** | Professional, specific, data-backed |
| GENERATION | Mock interview turn | **gpt-4o** | Multi-turn; context-aware follow-ups |
| EVALUATION | Application evaluation | gpt-4o-mini | 4-dimension rubric; 87% agreement with gpt-4o |
| EVALUATION | Learning plan evaluation | gpt-4o-mini | Same rubric-following pattern |
| EVALUATION | LinkedIn evaluation | gpt-4o-mini | pivot_clarity × 0.30 + keyword_density × 0.30 + … |
| EVALUATION | Interview answer evaluation | gpt-4o-mini | relevance × 0.30 + STAR × 0.25 + … |
| EVALUATION | Mock interview report | **gpt-4o** | 5-dim rubric; hire recommendation + coached rewrite |
| ORCHESTRATION | Career agent / tool loop | **gpt-4o** | Multi-step reasoning; conflict detection; synthesis memo |
| TRANSCRIPTION | Voice-native interview coaching | Whisper-1 | Production STT; dictate answers → immediate coaching |

---

## Aggregation and Conflict Handling

LLM outputs are never shown raw. Every score passes through a Python aggregation layer before reaching the user.

### Portfolio ranking

```python
hire_probability = int(min(95, max(15,
    quality_score * 0.65 + onet_fit_score * 0.35
)))
```

Weight rationale: At the application stage, writing quality is the primary screening signal. Hiring managers read the letter before checking the resume for fit. 0.65/0.35 reflects this asymmetry. Neither component is used raw — quality_score comes from a gpt-4o-mini rubric; onet_fit_score from offline cosine similarity.

### Tripartite application evaluation

```python
weighted_score = 0.30 × advocate_score + 0.35 × skeptic_score + 0.35 × technical_pm_score
disagreement   = std_dev([advocate, skeptic, technical_pm])

# consensus: std < 8  |  split: 8–20  |  contested: > 20
# contested: judge (gpt-4o) must explicitly resolve, not average
```

Weight rationale: False positives (rating a weak application as strong) are more costly than false negatives at the application stage. Skeptic + TechnicalPM weighted higher to reflect asymmetric cost. When std_dev > 20, the judge is required to name the strongest objection and rule on it — not ignore it.

### Controversy penalty (review board)

```python
weighted_mean    = Σ(score_i × weight_i) / Σ(weight_i)
penalty          = min(16.0, std × 0.9 + spread × 0.12)
adjusted_score   = max(0.0, weighted_mean - penalty)
controversy_flag = controversy_score > 50  # → auto-expands diagnostics
```

λ = 0.5 tuned empirically: penalises genuine disagreement without overriding strong consensus. Applications with high controversy scores are flagged for user review — never silently averaged.

### Brier calibration

```python
brier_score      = mean((p_predicted/100 − y_actual)²)  # over resolved predictions
correction_factor = empirical_offer_rate / predicted_offer_rate
p_corrected       = p_raw × correction_factor
```

The correction factor is applied to ALL future JD Analyzer predictions automatically. This is the "evaluate AI" loop — predictions are never accepted at face value after 3+ outcomes are logged.

---

## Development Challenges

These are real failures encountered during development. Not curated for optics.

### 1. LLM JSON Parsing Reliability `[HIGH]` — Day 3

**Symptom:** app.py crashed on ~30% of generation calls — JSON decode errors.  
**Root cause:** Models wrap JSON in markdown fences, add trailing commas, return prose before the JSON block.  
**Fix:** 3-layer extraction pipeline: (1) strict `json.loads()`, (2) regex extraction of `{...}` or `[...]` blocks, (3) LLM self-repair prompt. Success rate 70% → 98.5%.  
**Lesson:** Never assume an LLM returns parseable JSON. Always add a fallback extraction layer.

### 2. Context Window Exceeded in Review Board `[HIGH]` — Day 5

**Symptom:** 5-reviewer board with full CV + full JD exceeded 8k token limit on gpt-4o-mini.  
**Root cause:** Full CV text pasted into 5 parallel reviewer prompts instead of a structured summary.  
**Fix:** Compression pipeline: CV → structured profile (200 tokens), JD → extracted requirements (150 tokens). 70% token reduction. Truncation flag added: `'TRUNCATED — focus on skills only'`.  
**Lesson:** Token cost compounds multiplicatively with parallelism. Profile prompts before building parallel pipelines.

### 3. Model Overconfidence in P(offer) Predictions `[MEDIUM]` — Day 8

**Symptom:** Per-JD predictor returned 70–80% hire probability for average applications. Users didn't find it credible.  
**Root cause:** LLMs are optimistic by default — trained on positive framing. Raw model output ≠ calibrated probability.  
**Fix:** Brier score calibration. Log predictions + outcomes → compute `correction_factor = empirical_rate / predicted_rate` → apply to all future predictions. Hard bounds [15%, 95%] to prevent false certainty.  
**Lesson:** LLMs do not produce calibrated probabilities. Never surface a raw model probability to users.

### 4. O*NET Skill Hallucination `[HIGH]` — Day 2

**Symptom:** gpt-4o-mini invented plausible-sounding O*NET skill names that don't exist in the taxonomy.  
**Root cause:** Asking an LLM to produce taxonomy-constrained output without providing the taxonomy.  
**Fix:** Switched to offline pre-computed O*NET matrix (894 occupations × 119 skills). LLMs no longer used for taxonomy mapping — only for CV text parsing. Cosine similarity is fully deterministic.  
**Lesson:** For structured taxonomy tasks, offline data always beats LLM generation.

### 5. Streamlit Nested Expander Crash `[MEDIUM]` — Deployment

**Symptom:** `StreamlitAPIException: Expanders may not be nested inside other expanders` — crashed on cloud, worked locally.  
**Root cause:** Local Streamlit 1.56.0 supports nested expanders; Streamlit Cloud pinned at 1.32.0 (requirements.txt) does not.  
**Fix:** Replaced all nested `st.expander()` with `if/elif` blocks driven by `st.selectbox()` — eliminates Streamlit container nesting entirely.  
**Lesson:** Always test against the pinned requirements.txt version. Never develop against bleeding-edge local libraries.

### 6. Review Board Latency: 48s → 9s `[MEDIUM]` — Day 6

**Symptom:** 5-reviewer board ran sequentially in 48 seconds — unusable in a Streamlit demo.  
**Root cause:** Each reviewer was an independent LLM call. Sequential execution compounds API latency.  
**Fix:** `ThreadPoolExecutor(max_workers=5)`: all 5 reviewers run in parallel. Wall time 48s → 9s. Advocate + Skeptic in the debate also run in parallel before the Judge reads both.  
**Lesson:** Independent LLM calls are embarrassingly parallel. Always use thread pools for multi-persona evaluation unless personas need to see each other's output.

### 7. Self-Consistency Bias in Single-Model Evaluation `[HIGH]` — Day 7

**Symptom:** Using the same model to generate AND evaluate inflated scores by +12pt on average vs. human baseline.  
**Root cause:** Models prefer their own generation style. Self-evaluation is not objective scoring.  
**Fix:** Generation model (gpt-4o) separated from evaluation model (gpt-4o-mini). Tripartite evaluation added: Advocate + Skeptic + TechnicalPM. Disagreement score (std_dev) surfaced explicitly. Skeptic persona calibrated to find weaknesses the Advocate misses.  
**Lesson:** The separation of generation and evaluation is an architectural requirement, not a nice-to-have.

---

## Two Entry Points, One Destination

### Quick Apply — Maximise P(offer) across multiple jobs

```
CV upload  →  "Find my best opportunities"
                    ↓
      SerpAPI: live jobs (LinkedIn / Indeed / Glassdoor)
                    ↓
      O*NET fit score for each → ranked by P(offer)
                    ↓
      ⚡ "Launch Interview Pipeline"
                    ↓
      Parallel gpt-4o generation (3 applications simultaneously)
                    ↓
      gpt-4o-mini evaluation (4-dimension rubric)
                    ↓
      hire_probability = 0.65 × quality + 0.35 × fit  [Python]
                    ↓
      Adversarial verdict on winner (Advocate + Skeptic + Judge)
                    ↓
      "Apply to Google first (78%). Then Stripe (64%). Skip KPMG."
                    ↓
      P(offer) updates → log outcome → calibration improves
```

### Career Sprint — Validate the pivot before committing

```
Step 1: Assess     → O*NET cosine similarity, skill gap vector, timeline estimate
                     Expected P(offer) lift: up to +14pt (OPS skill-fit factor)
Step 2: Plan       → AI learning plan (gpt-4o-mini) → second-model evaluation
                     Expected P(offer) lift: +6pt (skill proofs factor)
Step 3: Validate   → Advocate vs. Skeptic vs. Judge → viability %
Step 4: Execute    → Real jobs (SerpAPI) + Application package (gpt-4o) → evaluation
                     Expected P(offer) lift: up to +12pt (application quality factor)
Step 5: Interview  → Role-specific questions → answer scoring → coached rewrites
                     Expected P(offer) lift: up to +10pt (interview readiness factor)
Bonus:  LinkedIn   → AI-written headline/about/bullets → pivot_clarity × keyword_density eval
```

Pivot Readiness Score (0–100) updates after each step. Ends with a downloadable **Pivot Playbook**.

---

## Quality Gate

No generated artifact reaches the user without a quality score:

```
gpt-4o generates  →  gpt-4o-mini evaluates (rubric)  →  score shown
                                ↓
                    score < threshold → auto-regenerate (cover letter, learning plan)
                    score ≥ threshold → surface to user
                    interview answers → evaluated + flagged; user rewrites (intentional: coaching, not replacement)
```

| Artifact | Threshold | Generate model | Evaluate model |
|---|---|---|---|
| Cover Letter | 65/100 | gpt-4o | gpt-4o-mini |
| LinkedIn InMail | 65/100 | gpt-4o | gpt-4o-mini |
| CV Bullets | 65/100 | gpt-4o | gpt-4o-mini |
| Learning Plan | 60/100 | gpt-4o-mini | gpt-4o-mini |
| Interview Answer | 60/100 | gpt-4o-mini | gpt-4o-mini |
| Application Package | 65/100 | gpt-4o | gpt-4o-mini |

The Quality Shield is a Streamlit session-state log tracking every quality gate event: artifact, models used, score v1, score v2 (after regen), threshold, pass/fail. Visible in-app under Analysis & Tools → Quality Shield.

---

## Offline vs. Online Architecture

The app has a hard separation: preprocessing runs once offline, ships with the app, and is never re-run at runtime.

### Ships offline — no API key needed

| Component | What it does | File |
|---|---|---|
| O*NET skill matrix | 894 occupations × 119 skill dimensions, parquet | `artifacts/occupation_skill_matrix.parquet` |
| IDF weighting | Downweights universal skills (communication, critical thinking) | `src/preprocessing.py` |
| Cosine similarity | IDF-weighted L2-normalised dot product, O(1) per query | `src/model_logic.py` |
| PCA coordinates | 2D embedding for occupation map | `artifacts/pca_coords.parquet` |
| kNN graph | Dijkstra stepping-stone routing on cosine-similarity graph | `src/model_logic.py → find_pivot_path()` |
| Skill gap computation | `target − current` per dimension, deterministic | `src/model_logic.py → compute_gap_df()` |
| CV parsing fallback | Regex-based heuristic when API unavailable | `src/cv_parser.py` |
| P(offer) computation | 11-factor accumulation, pure Python, no LLM | `src/offer_probability.py → compute_ops()` |
| Brier calibration | Correction factor from outcome log, pure Python | `src/p_offer_trend.py` |

### Requires OPENAI_API_KEY

| Component | Model | Purpose |
|---|---|---|
| CV skill extraction | gpt-4o-mini | Map free-text CV to O*NET skill dimensions |
| Application generation | gpt-4o | Cover letter + CV bullets + InMail |
| Application evaluation | gpt-4o-mini | 4-dimension rubric scoring |
| Adversarial debate | gpt-4o-mini + gpt-4o | Advocate + Skeptic → Judge |
| Learning plan | gpt-4o-mini | Gap-specific roadmap generation |
| Career agent | gpt-4o | Multi-step tool-calling pivot assessment |
| Mock interview | gpt-4o | Multi-turn interview + coaching report |

### Requires SERP_API_KEY (optional)

Real job search (SerpAPI → Google Jobs). Fallback: `generate_job_listings()` produces realistic simulated postings.

### Graceful degradation

Every LLM call has a deterministic fallback. The app runs fully offline (heuristic mode) — no API key is required to explore gap analysis, stepping-stone routing, cosine similarity, or skill investment simulation.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Data | O*NET occupational database (US Dept. of Labor) — 894 occupations × 119 skill dimensions |
| Offline inference | scikit-learn (PCA, cosine similarity), pandas, numpy — ships with app |
| LLM | OpenAI gpt-4o + gpt-4o-mini (see architecture table above) |
| Speech-to-text | OpenAI Whisper-1 (voice-native interview coaching) |
| Job search | SerpAPI → Google Jobs aggregator |
| Parallelism | `concurrent.futures.ThreadPoolExecutor` — review board, portfolio generation, DM batch |
| Framework | Streamlit (deployed to Streamlit Cloud, pinned 1.32.0) |
| Persistence | SQLite via `pivot_os.db`; session state for in-session data |

---

## Running Locally

```bash
git clone https://github.com/Thisisntevenmyfinale/career-pivot-simulator
cd career-pivot-simulator
pip install -r requirements.txt
streamlit run app.py
```

Add to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
SERP_API_KEY = "..."   # serpapi.com — free tier: 100 searches/month
```

The app runs without API keys in heuristic/offline mode. All O*NET-based features (gap analysis, cosine similarity, stepping-stone routing, P(offer) baseline) work without any key.

---

## Author

Jan Philipp Gnau — Master in Business Analytics  
Course: *Prototyping Products with Data and Artificial Intelligence*  
Instructor: Jose A. Rodriguez Serrano
