from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.model_logic import (
    load_runtime_artifacts,
    compute_gap_df,
    compute_confidence_score,
    find_pivot_path,
)
from src.skill_investment_simulator import (
    simulate_skill_investment,
    suggest_best_investment_skills,
)
from src.ai_coach import generate_learning_plan_markdown
from src.llm_review_board import (
    generate_all_strategies,
    evaluate_strategies_by_reviewers,
)
from src.review_aggregation import (
    compute_consensus,
    generate_judge_memo,
    rerank_after_skill_investment,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")


# ============================================================
# Small helpers
# ============================================================
def _has_openai_secret() -> bool:
    try:
        return bool(str(st.secrets["OPENAI_API_KEY"]).strip())
    except Exception:
        return False


def _render_table_card(
    df: pd.DataFrame,
    columns: List[str],
    headers: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> None:
    if df is None or df.empty:
        st.info("No data.")
        return

    view = df[columns].copy()
    headers = headers or columns
    numeric_cols_set = set(numeric_cols or [])

    for c in columns:
        if c in numeric_cols_set:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
            )

    th = "".join([f"<th>{h}</th>" for h in headers])
    rows_html = []
    for _, r in view.iterrows():
        tds = []
        for c in columns:
            val = r[c]
            cls = "num" if c in numeric_cols_set else ""
            safe = str(val).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            tds.append(f"<td class='{cls}'>{safe}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    html = f"""
<div class="li-table-wrap">
  <table class="li-table">
    <thead><tr>{th}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _learning_plan_source_label(md: str) -> str:
    text = (md or "").strip()
    if not text:
        return "—"
    if text.startswith("🤖"):
        return "OpenAI"
    if text.startswith("⚠️ ONLINE_ERROR"):
        return "OpenAI error → fallback"
    if text.startswith("ERROR::"):
        return "Secret / runtime error"
    return "Offline"


def _extract_review_trace_status(trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": trace.get("mode", ""),
        "model": trace.get("model", ""),
        "errors": trace.get("errors", []),
    }


def _render_bullet_list(title: str, items: List[str], empty_text: str = "No items.") -> None:
    st.markdown(f"**{title}**")
    clean = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not clean:
        st.caption(empty_text)
        return
    for item in clean[:6]:
        st.markdown(f"- {item}")
        


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
:root{
  --li-blue: #0A66C2;
  --li-blue-dark: #004182;
  --li-bg: #F3F2EF;
  --li-card: #FFFFFF;
  --li-border-soft: rgba(0,0,0,0.08);
  --li-text: rgba(0,0,0,0.90);
  --li-subtext: rgba(0,0,0,0.62);
  --radius: 10px;
  --shadow: 0 1px 1px rgba(0,0,0,0.04);
}

[data-testid="stHeader"]{ height:0px !important; background:transparent !important; }
[data-testid="stToolbar"]{ display:none !important; }
#MainMenu{ visibility:hidden; }
footer{ visibility:hidden; }

html, body, .stApp{
  background: var(--li-bg) !important;
  color: var(--li-text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.block-container{
  max-width: 1128px !important;
  padding-top: 18px !important;
  padding-bottom: 28px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--li-card) !important;
  border: 1px solid var(--li-border-soft) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  padding: 14px !important;
}
section[data-testid="stSidebar"]{
  background: var(--li-bg) !important;
  border-right: 1px solid var(--li-border-soft) !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]{
  padding: 12px !important;
}

:root{
  --primary-color: #0A66C2 !important;
  --primaryColor: #0A66C2 !important;
  --primary-color-hover: #004182 !important;
  --primaryColorHover: #004182 !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  --primary-color: #0A66C2 !important;
  --primaryColor: #0A66C2 !important;
  --primary-color-hover: #004182 !important;
  --primaryColorHover: #004182 !important;
}

[data-baseweb="radio"] svg{
  color: var(--li-blue) !important;
  fill: var(--li-blue) !important;
}
[data-baseweb="radio"] [aria-checked="true"] svg{
  color: var(--li-blue) !important;
  fill: var(--li-blue) !important;
}
[data-baseweb="radio"] [aria-checked="true"]{
  border-color: var(--li-blue) !important;
}

[data-baseweb="switch"] [role="switch"][aria-checked="true"]{
  background-color: rgba(10,102,194,0.35) !important;
  border-color: rgba(10,102,194,0.35) !important;
}
[data-baseweb="switch"] [role="switch"][aria-checked="true"] > div{
  background: var(--li-blue) !important;
}

.stButton > div,
.stDownloadButton > div{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-baseweb="button"]{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-baseweb="button"] > div{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.stButton > button, .stDownloadButton > button{
  background: var(--li-blue) !important;
  border: 1px solid var(--li-blue) !important;
  color: #fff !important;
  border-radius: 999px !important;
  height: 40px !important;
  padding: 0 18px !important;
  font-weight: 800 !important;
  font-size: 14px !important;
  box-shadow: none !important;
  outline: none !important;
  white-space: nowrap !important;
  width: auto !important;
  max-width: 100% !important;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  background: var(--li-blue-dark) !important;
  border: 1px solid var(--li-blue-dark) !important;
}

.li-table-wrap{
  border: 1px solid rgba(0,0,0,0.05) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  background: #fff !important;
  margin: 10px 0 14px 0 !important;
}
.li-table tbody tr:last-child td{ border-bottom: none !important; }

table.li-table{
  width: 100% !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  font-size: 13px !important;
}
.li-table thead th{
  padding: 10px 12px !important;
  background: #FBFBFC !important;
  color: rgba(0,0,0,0.70) !important;
  border-bottom: 1px solid rgba(0,0,0,0.06) !important;
  font-weight: 800 !important;
}
.li-table tbody td{
  padding: 10px 12px !important;
  border-bottom: 1px solid rgba(0,0,0,0.04) !important;
  color: rgba(0,0,0,0.88) !important;
}
.li-table .num{
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
}

.li-subtitle{
  margin-top: -4px;
  margin-bottom: 6px;
  color: var(--li-subtext);
  font-size: 13px;
}

.status-pill{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  margin-right:8px;
}
.status-ok{
  background:#E7F6EC;
  color:#117A37;
  border:1px solid #CBEAD5;
}
.status-warn{
  background:#FFF4E5;
  color:#A05A00;
  border:1px solid #F3D7A5;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================
with st.container(border=True):
    st.markdown("## 🧭 Career Pivot Simulator")
    st.markdown(
        '<div class="li-subtitle">Career pivot decision support with O*NET skill matching, route analysis, learning plan generation, and an adversarial LLM review board.</div>',
        unsafe_allow_html=True,
    )
    llm_html = (
        '<span class="status-pill status-ok">LLM ready: Streamlit secret found</span>'
        if _has_openai_secret()
        else '<span class="status-pill status-warn">LLM fallback mode: no Streamlit secret found</span>'
    )
    st.markdown(llm_html, unsafe_allow_html=True)


# ============================================================
# Load artifacts
# ============================================================
@st.cache_data(show_spinner=False)
def load_artifacts_cached() -> Any:
    return load_runtime_artifacts("artifacts")


try:
    art = load_artifacts_cached()
except Exception as e:
    st.error("Missing or invalid runtime artifacts.")
    st.info("Run: `python scripts/preprocess_onet.py` to generate artifacts.")
    st.exception(e)
    st.stop()

mat: pd.DataFrame = art.matrix
occupations: list[str] = mat.index.astype(str).tolist()
X_BASE: np.ndarray = np.asarray(mat.to_numpy(), dtype=np.float32, order="C")
OCCS: list[str] = occupations
OCC_TO_IDX: dict[str, int] = {o: i for i, o in enumerate(OCCS)}


# ============================================================
# Session state
# ============================================================
DEFAULT_STATE = {
    "has_run": False,
    "target_override": None,
    "route_result": None,
    "route_config": {"k_neighbors": 10, "max_steps": 6},
    "learning_plan_md": "",
    "learning_plan_source": "—",
    "sim_result": None,
    "review_board_strategies": None,
    "review_board_evaluations": None,
    "review_board_consensus": None,
    "review_board_judge_memo": None,
    "review_board_trace": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Scoring core
# ============================================================
@st.cache_resource(show_spinner=False)
def build_cosine_core(use_idf: bool) -> Dict[str, Any]:
    X = X_BASE
    n, d = X.shape

    if use_idf:
        df = np.sum(X > 0.0, axis=0).astype(np.float32, copy=False)
        idf = np.log((np.float32(n) + 1.0) / (1.0 + df)) + 1.0
        idf = np.clip(idf, 1.0, None).astype(np.float32, copy=False)
        Xw = X * idf[None, :]
    else:
        idf = np.ones(d, dtype=np.float32)
        Xw = X

    norms = np.linalg.norm(Xw, axis=1)
    norms_safe = np.where(norms == 0.0, 1.0, norms).astype(np.float32, copy=False)
    Xn = (Xw / norms_safe[:, None]).astype(np.float32, copy=False)

    return {"occs": OCCS, "occ_to_idx": OCC_TO_IDX, "Xn": Xn, "Xn_T": Xn.T}


def _percentile_from_sorted(sorted_vals: np.ndarray, x: float) -> float:
    if sorted_vals.size == 0 or not np.isfinite(x):
        return 0.0
    left = int(np.searchsorted(sorted_vals, x, side="left"))
    right = int(np.searchsorted(sorted_vals, x, side="right"))
    eq = right - left
    pct = 100.0 * (left + 0.5 * eq) / float(sorted_vals.size)
    return float(np.clip(pct, 0.0, 100.0))


def _midrank_percentiles(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    n = v.size
    if n == 0:
        return np.asarray([], dtype=np.float32)

    order = np.argsort(v, kind="mergesort")
    vs = v[order]

    pct_sorted = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        j = i + 1
        while j < n and vs[j] == vs[i]:
            j += 1
        less = i
        eq = j - i
        pct_sorted[i:j] = np.float32(100.0 * (less + 0.5 * eq) / float(n))
        i = j

    pct = np.empty(n, dtype=np.float32)
    pct[order] = pct_sorted
    return pct


def get_score_distribution(use_idf: bool, current_occ: str) -> Dict[str, Any]:
    core = build_cosine_core(bool(use_idf))
    Xn: np.ndarray = core["Xn"]
    Xn_T: np.ndarray = core["Xn_T"]
    occ_to_idx: dict[str, int] = core["occ_to_idx"]

    i = occ_to_idx.get(current_occ, -1)
    if i < 0:
        return {
            "scores": np.asarray([], dtype=np.float32),
            "scores_sorted": np.asarray([], dtype=np.float32),
            "raw_scores_all": np.asarray([], dtype=np.float32),
            "mask_other": np.asarray([], dtype=bool),
        }

    sims = Xn[i] @ Xn_T
    sims = np.clip(sims, -1.0, 1.0)
    raw_scores_all = np.maximum(sims, 0.0) * 100.0
    raw_scores_all = np.clip(raw_scores_all, 0.0, 100.0).astype(np.float32, copy=False)

    mask_other = np.ones(raw_scores_all.shape[0], dtype=bool)
    mask_other[i] = False

    return {
        "scores": raw_scores_all[mask_other],
        "scores_sorted": np.sort(raw_scores_all[mask_other]),
        "raw_scores_all": raw_scores_all,
        "mask_other": mask_other,
    }


def recommend_neighbors(use_idf: bool, current_occ: str, top_k: int = 10) -> pd.DataFrame:
    dist = get_score_distribution(bool(use_idf), str(current_occ))
    scores_other = dist["scores"]
    raw_all = dist["raw_scores_all"]
    mask_other = dist["mask_other"]

    if scores_other.size == 0 or raw_all.size == 0:
        return pd.DataFrame(columns=["occupation", "match_raw", "match_percentile"])

    pct_other = _midrank_percentiles(scores_other)
    occ_other = np.asarray(OCCS, dtype=object)[mask_other]
    raw_other = raw_all[mask_other]

    k = int(min(max(int(top_k), 0), raw_other.size))
    if k == 0:
        return pd.DataFrame(columns=["occupation", "match_raw", "match_percentile"])

    idx_part = np.argpartition(-raw_other, kth=k - 1)[:k]
    idx_sorted = idx_part[np.argsort(-raw_other[idx_part], kind="mergesort")]

    return pd.DataFrame(
        {
            "occupation": occ_other[idx_sorted],
            "match_raw": raw_other[idx_sorted].astype(float),
            "match_percentile": pct_other[idx_sorted].astype(float),
        }
    ).reset_index(drop=True)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    with st.container(border=True):
        st.subheader("Controls")

        mode = st.radio("Mode", options=["Guided", "Research"], index=0)
        guided = mode == "Guided"

        st.divider()

        with st.expander("Choose your pivot", expanded=True):
            current = st.selectbox("Current occupation", options=occupations, index=0)
            default_target_idx = 1 if len(occupations) > 1 else 0
            selected_target = st.selectbox("Target occupation", options=occupations, index=default_target_idx)

            target = st.session_state.target_override or selected_target

            if current == target:
                st.warning("Pick a different target.")

        with st.expander("Scoring", expanded=True):
            use_idf = st.toggle("Downweight common skills (IDF)", value=True)
            score_mode = st.radio(
                "Overview score",
                options=["Percentile (recommended)", "Raw similarity (transparent)"],
                index=0,
            )
            st.caption("Percentile answers: how strong is this target versus all other options from the current role?")

        if not guided:
            with st.expander("Research knobs", expanded=False):
                k_neighbors = st.slider("kNN neighbors", 2, 20, int(st.session_state.route_config["k_neighbors"]), 1)
                max_steps = st.slider("Max steps", 2, 10, int(st.session_state.route_config["max_steps"]), 1)
                st.session_state.route_config = {"k_neighbors": int(k_neighbors), "max_steps": int(max_steps)}

        st.divider()
        if st.button("🚀 Run pivot analysis", use_container_width=True):
            st.session_state.has_run = True

    with st.container(border=True):
        st.subheader("Dataset snapshot")
        st.metric("Occupations", mat.shape[0])
        st.metric("Skills", mat.shape[1])


# ============================================================
# Empty state
# ============================================================
if not st.session_state.has_run:
    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        with st.container():
            st.subheader("What this prototype does")
            st.markdown(
                """
- Compare a current role and target role using skill-profile similarity.
- Show what transfers and what blocks the pivot.
- Suggest stepping-stone roles.
- Simulate targeted skill investment.
- Generate an LLM learning plan.
- Run an adversarial review board with competing strategies and reviewer personas.
                """
            )

    with c2:
        with st.container():
            st.subheader("Fast path")
            st.markdown(
                """
1. Pick current and target roles  
2. Click **Run pivot analysis**  
3. Review match + gaps  
4. Generate the learning plan  
5. Run the **Decision Board**  
                """
            )
    st.stop()


# ============================================================
# Core computations
# ============================================================
dist = get_score_distribution(bool(use_idf), str(current))
scores_all_sorted = dist["scores_sorted"]
raw_all = dist["raw_scores_all"]

target_idx = OCC_TO_IDX.get(str(target), -1)
raw_target = float(raw_all[target_idx]) if target_idx >= 0 and raw_all.size else 0.0
pct_target = _percentile_from_sorted(scores_all_sorted, float(raw_target))

show_percentile = score_mode.startswith("Percentile")
match_score_display = float(pct_target if show_percentile else raw_target)

gap_df = compute_gap_df(mat, str(current), str(target))
conf = compute_confidence_score(mat, art.pca_meta, str(current), str(target))
neighbors_df = recommend_neighbors(bool(use_idf), str(current), top_k=10)


# ============================================================
# Overview
# ============================================================
with st.container(border=True):
    st.subheader("Overview")

    m1, m2, m3 = st.columns([1, 1, 1], gap="large")
    m1.metric("Match", f"{match_score_display:.0f}/100")
    m2.metric("Confidence", f"{conf['confidence_score']:.0f}/100")
    m3.metric("Scoring", "IDF cosine" if use_idf else "Raw cosine")

    if match_score_display >= 70:
        st.success("Strong candidate: validate the story, build evidence, and compare strategies.")
    elif match_score_display >= 45:
        st.info("Promising with gaps: a stepping-stone or hybrid strategy may outperform a direct pivot.")
    else:
        st.warning("Hard pivot: use route analysis, skill investment, and the review board before choosing a strategy.")


# ============================================================
# Main layout
# ============================================================
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    with st.container():
        st.subheader("Career neighborhood")
        st.caption("Closest roles to your current occupation. Useful stepping-stone candidates.")

        show_df = neighbors_df.copy()
        show_df["match_raw"] = show_df["match_raw"].round(2)
        show_df["match_percentile"] = show_df["match_percentile"].round(2)

        _render_table_card(
            show_df,
            columns=["occupation", "match_percentile", "match_raw"],
            headers=["Occupation", "Match (pct)", "Match (raw)"],
            numeric_cols=["match_percentile", "match_raw"],
        )

        with st.container(border=True):
            st.markdown("**Try a stepping-stone target**")
            if show_df.empty:
                st.info("No recommendations available.")
            else:
                label_to_occ: Dict[str, str] = {}
                options = []
                for _, r in show_df.head(8).iterrows():
                    occ = str(r["occupation"])
                    label = f"{occ} — pct {float(r['match_percentile']):.0f}/100 • raw {float(r['match_raw']):.0f}/100"
                    options.append(label)
                    label_to_occ[label] = occ

                pick = st.selectbox("Recommended targets", options=options, index=0, label_visibility="collapsed")
                if st.button("Use as target", use_container_width=True):
                    st.session_state.target_override = label_to_occ[pick]
                    st.session_state.has_run = True
                    st.session_state.route_result = None
                    st.session_state.review_board_strategies = None
                    st.session_state.review_board_evaluations = None
                    st.session_state.review_board_consensus = None
                    st.session_state.review_board_judge_memo = None
                    st.rerun()

with right:
    with st.container():
        st.subheader("Route + Learning")
        st.caption("Operational support before the strategy board.")

        st.markdown("### Stepping-stone route")
        if guided:
            st.caption("Optional: find intermediate roles that make the pivot more realistic.")
            col_a, col_b = st.columns([1, 1])

            with col_a:
                if st.button("Find route", use_container_width=True):
                    with st.spinner("Finding a route..."):
                        st.session_state.route_result = find_pivot_path(
                            mat,
                            start_occ=str(current),
                            target_occ=str(target),
                            k_neighbors=12,
                            max_steps=6,
                        )

            with col_b:
                if st.button("Reset route", use_container_width=True):
                    st.session_state.route_result = None
        else:
            st.caption("Research mode: use custom graph settings from the sidebar.")
            col_a, col_b = st.columns([1, 1])

            with col_a:
                if st.button("Find route (research)", use_container_width=True):
                    cfg = st.session_state.route_config
                    with st.spinner("Finding a route..."):
                        st.session_state.route_result = find_pivot_path(
                            mat,
                            start_occ=str(current),
                            target_occ=str(target),
                            k_neighbors=int(cfg["k_neighbors"]),
                            max_steps=int(cfg["max_steps"]),
                        )

            with col_b:
                if st.button("Reset route", use_container_width=True):
                    st.session_state.route_result = None

        route = st.session_state.route_result
        if not route:
            st.info("Route not computed yet.")
        elif not route.get("reachable"):
            st.warning("No route found with the current assumptions.")
        else:
            path = route.get("path", [])
            st.success(" → ".join([str(p) for p in path]) if path else "Route computed.")

        st.divider()

        st.markdown("### Skill investment simulator")
        st.caption("Counterfactual: if you improve selected skills, how much does the match move?")

        sim_candidates_df = suggest_best_investment_skills(gap_df, top_k=8)

        if sim_candidates_df.empty:
            st.info("No positive skill gaps available for simulation.")
        else:
            skill_options = sim_candidates_df["skill"].astype(str).tolist()
            default_pick = skill_options[: min(3, len(skill_options))]

            selected_sim_skills = st.multiselect(
                "Choose skills to improve",
                options=skill_options,
                default=default_pick,
            )

            uplift_ratio = st.slider(
                "How much of each gap do you close?",
                min_value=0.10,
                max_value=1.00,
                value=0.50,
                step=0.05,
            )

            q1, q2 = st.columns([1, 1])

            with q1:
                if st.button("Run skill simulation", use_container_width=True):
                    st.session_state.sim_result = simulate_skill_investment(
                        mat,
                        current_role=str(current),
                        target_role=str(target),
                        selected_skills=selected_sim_skills,
                        uplift_ratio=float(uplift_ratio),
                    )

            with q2:
                if st.button("Clear simulation", use_container_width=True):
                    st.session_state.sim_result = None

        st.divider()

        st.markdown("### Learning plan")
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("Generate plan", use_container_width=True):
                with st.spinner("Generating plan..."):
                    md = generate_learning_plan_markdown(
                        current_role=str(current),
                        target_role=str(target),
                        gap_df=gap_df,
                        language="en",
                        model="gpt-4o-mini",
                        max_missing=6,
                        prefer_online=True,
                    )
                    st.session_state.learning_plan_md = md
                    st.session_state.learning_plan_source = _learning_plan_source_label(md)

        with c2:
            if st.button("Clear plan", use_container_width=True):
                st.session_state.learning_plan_md = ""
                st.session_state.learning_plan_source = "—"


# ============================================================
# Learning plan preview
# ============================================================
plan_md = (st.session_state.learning_plan_md or "").strip()
if plan_md:
    with st.container(border=True):
        st.subheader("Learning plan preview")
        st.caption(f"Source: {st.session_state.learning_plan_source} • Output is Markdown.")
        st.markdown(plan_md)



# ============================================================
# Decision Board (Hero Feature)
# ============================================================
with st.container(border=True):
    st.subheader("⚖️ Career Pivot Decision Engine")
    st.caption(
        "Generate competing pivot strategies, pressure-test them with multiple expert personas, aggregate disagreement in Python, and re-rank the recommendation under skill-investment counterfactuals."
    )

    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        if st.button("1) Generate strategies", use_container_width=True):
            with st.spinner("Generating differentiated strategy archetypes..."):
                strategies_bundle = generate_all_strategies(
                    current_role=str(current),
                    target_role=str(target),
                    gap_df=gap_df,
                    model="gpt-4o-mini",
                    prefer_online=True,
                )
                st.session_state.review_board_strategies = strategies_bundle.get("strategies")
                st.session_state.review_board_trace["strategies_bundle"] = strategies_bundle.get("trace", {})
                st.session_state.review_board_evaluations = None
                st.session_state.review_board_consensus = None
                st.session_state.review_board_judge_memo = None
                st.session_state.review_board_trace.pop("counterfactual_consensus", None)

    with b2:
        if st.button("2) Get expert evals", use_container_width=True):
            if not st.session_state.review_board_strategies:
                st.error("Generate strategies first.")
            else:
                with st.spinner("Running multi-persona review pass..."):
                    evals_bundle = evaluate_strategies_by_reviewers(
                        strategies=st.session_state.review_board_strategies,
                        current_role=str(current),
                        target_role=str(target),
                        model="gpt-4o-mini",
                        prefer_online=True,
                    )
                    st.session_state.review_board_evaluations = evals_bundle.get("evaluations")
                    st.session_state.review_board_trace["evaluations_bundle"] = evals_bundle.get("trace", {})
                    st.session_state.review_board_consensus = None
                    st.session_state.review_board_judge_memo = None
                    st.session_state.review_board_trace.pop("counterfactual_consensus", None)

    with b3:
        if st.button("3) Compute consensus", use_container_width=True):
            if not st.session_state.review_board_evaluations:
                st.error("Get expert evals first.")
            else:
                with st.spinner("Computing confidence-adjusted consensus..."):
                    st.session_state.review_board_consensus = compute_consensus(
                        st.session_state.review_board_evaluations
                    )
                    st.session_state.review_board_judge_memo = None

    strategies = st.session_state.review_board_strategies
    evaluations = st.session_state.review_board_evaluations
    consensus = st.session_state.review_board_consensus
    judge_memo = st.session_state.review_board_judge_memo

    if strategies:
        st.divider()
        st.markdown("### Competing pivot strategies")

        strategies_trace = st.session_state.review_board_trace.get("strategies_bundle", {})
        diversity_warnings = strategies_trace.get("diversity_warnings", [])
        if diversity_warnings:
            with st.expander("Strategy diversity diagnostics", expanded=False):
                for msg in diversity_warnings:
                    st.warning(str(msg))

        for i, strat in enumerate(strategies):
            with st.container(border=True):
                top_row_left, top_row_right = st.columns([2.0, 1.0])

                with top_row_left:
                    st.markdown(f"**{i + 1}. {strat.archetype.name}**")
                    st.markdown(str(strat.summary))

                with top_row_right:
                    st.markdown(f"**Code:** `{strat.archetype.code}`")
                    st.markdown(f"**Risk:** {strat.archetype.risk_level}")
                    st.markdown(f"**Estimated days:** {strat.archetype.estimated_days}")

                d1, d2, d3, d4, d5 = st.columns(5)
                d1.metric("Speed", f"{float(getattr(strat, 'speed_bias', 5.0)):.1f}/10")
                d2.metric("Risk control", f"{float(getattr(strat, 'risk_bias', 5.0)):.1f}/10")
                d3.metric("Evidence", f"{float(getattr(strat, 'evidence_burden', 5.0)):.1f}/10")
                d4.metric("Market signal", f"{float(getattr(strat, 'market_signal_strength', 5.0)):.1f}/10")
                d5.metric("Gap focus", f"{float(getattr(strat, 'skill_gap_focus', 5.0)):.1f}/10")

                info_left, info_right = st.columns(2)
                with info_left:
                    st.markdown(f"**Best for:** {str(getattr(strat, 'best_for_profile', '') or '—')}")
                    st.markdown(f"**Evidence strategy:** {str(getattr(strat, 'evidence_strategy', '') or '—')}")
                with info_right:
                    st.markdown(f"**Key trade-off:** {str(getattr(strat, 'key_tradeoff', '') or '—')}")
                    st.markdown(f"**Confidence rationale:** {str(getattr(strat, 'confidence_rationale', '') or '—')}")

                with st.expander("Strategy details", expanded=False):
                    phases_rows = []
                    for phase in strat.phases:
                        phases_rows.append(
                            {
                                "phase": phase.phase,
                                "objective": phase.objective,
                            }
                        )
                    phases_df = pd.DataFrame(phases_rows)
                    if not phases_df.empty:
                        _render_table_card(
                            phases_df,
                            columns=["phase", "objective"],
                            headers=["Phase", "Objective"],
                            numeric_cols=[],
                        )

                    c_a, c_b = st.columns(2)
                    with c_a:
                        _render_bullet_list("Key missing skills addressed", getattr(strat, "key_missing_skills", []))
                        _render_bullet_list("Success criteria", getattr(strat, "success_criteria", []))
                    with c_b:
                        _render_bullet_list("Transferable anchors", getattr(strat, "transferable_anchors", []))
                        _render_bullet_list("Potential risks", getattr(strat, "potential_risks", []))

    if evaluations:
        st.divider()
        st.markdown("### Reviewer coverage and disagreement")

        review_rows = []
        detail_rows = []

        for ev in evaluations:
            review_rows.append(
                {
                    "reviewer_persona": ev.reviewer_persona,
                    "strongest_strategy": ev.strongest_strategy,
                    "weakest_strategy": ev.weakest_strategy,
                    "reviewer_weight": float(getattr(ev, "reviewer_weight", 1.0)),
                }
            )

            for s in ev.strategy_scores:
                detail_rows.append(
                    {
                        "reviewer_persona": ev.reviewer_persona,
                        "strategy_code": s.strategy_code,
                        "overall_score": float(s.overall_score),
                        "alignment_with_role": float(s.alignment_with_role),
                        "market_feasibility": float(s.market_feasibility),
                        "time_efficiency": float(s.time_efficiency),
                        "risk_assessment": float(s.risk_assessment),
                        "narrative_strength": float(s.narrative_strength),
                    }
                )

        review_df = pd.DataFrame(review_rows)
        if not review_df.empty:
            _render_table_card(
                review_df,
                columns=["reviewer_persona", "strongest_strategy", "weakest_strategy", "reviewer_weight"],
                headers=["Reviewer", "Strongest", "Weakest", "Weight"],
                numeric_cols=["reviewer_weight"],
            )

        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            with st.expander("Reviewer score matrix", expanded=False):
                _render_table_card(
                    detail_df,
                    columns=[
                        "reviewer_persona",
                        "strategy_code",
                        "overall_score",
                        "alignment_with_role",
                        "market_feasibility",
                        "time_efficiency",
                        "risk_assessment",
                        "narrative_strength",
                    ],
                    headers=[
                        "Reviewer",
                        "Strategy",
                        "Overall",
                        "Role Fit",
                        "Market",
                        "Time",
                        "Risk",
                        "Narrative",
                    ],
                    numeric_cols=[
                        "overall_score",
                        "alignment_with_role",
                        "market_feasibility",
                        "time_efficiency",
                        "risk_assessment",
                        "narrative_strength",
                    ],
                )

        with st.expander("Reviewer arguments", expanded=False):
            for ev in evaluations:
                st.markdown(f"**{ev.reviewer_persona}** — {ev.overall_recommendation}")
                for s in ev.strategy_scores:
                    with st.container(border=True):
                        st.markdown(f"**{s.strategy_code}** · overall {float(s.overall_score):.1f}/100")
                        st.markdown(str(s.justification))
                        left_col, right_col = st.columns(2)
                        with left_col:
                            st.markdown(f"**Best strength:** {str(getattr(s, 'best_strength', '') or '—')}")
                            st.markdown(f"**Biggest risk:** {str(getattr(s, 'biggest_risk', '') or '—')}")
                            st.markdown(f"**Success condition:** {str(getattr(s, 'success_condition', '') or '—')}")
                        with right_col:
                            st.markdown(f"**Killer objection:** {str(getattr(s, 'killer_objection', '') or '—')}")
                            st.markdown(f"**Best candidate fit:** {str(getattr(s, 'best_candidate_fit', '') or '—')}")
                            concerns = getattr(s, "concerns", []) or []
                            if concerns:
                                st.markdown("**Concerns**")
                                for c in concerns[:4]:
                                    st.markdown(f"- {c}")

    if consensus:
        st.divider()
        st.markdown("### Consensus result")

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        c1.metric("Winner", consensus.winner_strategy)
        c2.metric("Winner score", f"{consensus.winner_score:.1f}/100")
        c3.metric("Consensus strength", f"{consensus.consensus_strength:.0f}/100")
        c4.metric("Robustness", f"{float(getattr(consensus, 'robustness_score', 0.0)):.0f}/100")

        c5, c6 = st.columns(2)
        with c5:
            st.metric("Controversy", f"{float(getattr(consensus, 'controversy_score', 0.0)):.0f}/100")
        with c6:
            st.metric("Fragile winner", "Yes" if bool(getattr(consensus, "fragile_winner", False)) else "No")


        ranking_df = pd.DataFrame(consensus.strategy_rankings, columns=["strategy_code", "confidence_adjusted_score"])
        _render_table_card(
            ranking_df,
            columns=["strategy_code", "confidence_adjusted_score"],
            headers=["Strategy", "Confidence-Adjusted Score"],
            numeric_cols=["confidence_adjusted_score"],
        )

        st.markdown("### Why this recommendation currently wins")
        st.info(str(getattr(consensus, "winner_reason", "") or "No winner rationale available."))

        x1, x2 = st.columns(2)
        with x1:
            st.markdown("**Why the runner-up does not win yet**")
            st.markdown(str(getattr(consensus, "runner_up_reason", "") or "—"))
        with x2:
            st.markdown("**Main vulnerability of the winner**")
            st.markdown(str(getattr(consensus, "winner_vulnerability", "") or "—"))

        decision_levers = getattr(consensus, "decision_levers", []) or []
        if decision_levers:
            _render_bullet_list("Decision levers", decision_levers)

        switch_conditions = getattr(consensus, "switch_conditions", []) or []
        if switch_conditions:
            _render_bullet_list("What could flip the recommendation", switch_conditions)

        diagnostics = getattr(consensus, "strategy_diagnostics", []) or []
        


        diagnostics = getattr(consensus, "strategy_diagnostics", []) or []
        if diagnostics:
            with st.expander("Consensus diagnostics", expanded=False):
                diagnostic_df = pd.DataFrame(diagnostics)
                if not diagnostic_df.empty:
                    _render_table_card(
                        diagnostic_df,
                        columns=[
                            "strategy",
                            "confidence_adjusted_score",
                            "raw_mean_score",
                            "disagreement_penalty",
                            "std_dev",
                            "robustness_score",
                        ],
                        headers=[
                            "Strategy",
                            "Adj. Score",
                            "Raw Mean",
                            "Penalty",
                            "Std Dev",
                            "Robustness",
                        ],
                        numeric_cols=[
                            "confidence_adjusted_score",
                            "raw_mean_score",
                            "disagreement_penalty",
                            "std_dev",
                            "robustness_score",
                        ],
                    )

        if consensus.major_disagreements:
            with st.expander("Major disagreements", expanded=False):
                disagreement_df = pd.DataFrame(consensus.major_disagreements)
                if not disagreement_df.empty:
                    _render_table_card(
                        disagreement_df,
                        columns=["strategy", "strongest_advocate", "strongest_critic", "spread", "std_dev"],
                        headers=["Strategy", "Advocate", "Critic", "Spread", "Std Dev"],
                        numeric_cols=["spread", "std_dev"],
                    )

        alignment_summary = getattr(consensus, "reviewer_alignment_summary", []) or []
        if alignment_summary:
            with st.expander("Reviewer alignment summary", expanded=False):
                align_df = pd.DataFrame(alignment_summary)
                if not align_df.empty:
                    _render_table_card(
                        align_df,
                        columns=["reviewer_persona", "preferred_strategy", "least_preferred_strategy", "reviewer_weight"],
                        headers=["Reviewer", "Preferred", "Least Preferred", "Weight"],
                        numeric_cols=["reviewer_weight"],
                    )

        if st.button("4) Generate judge memo", use_container_width=True):
            with st.spinner("Generating final judge recommendation..."):
                missing = gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False)["skill"].head(4).tolist()
                transfer = gap_df.copy()
                transfer["overlap"] = np.minimum(transfer["current_importance"], transfer["target_importance"])
                top_transfer = transfer.sort_values("overlap", ascending=False)["skill"].head(3).tolist()

                gap_summary = (
                    f"Top missing skills: {', '.join(missing) if missing else 'none'}. "
                    f"Top transferable anchors: {', '.join(top_transfer) if top_transfer else 'none'}."
                )

                judge_bundle = generate_judge_memo(
                    current_role=str(current),
                    target_role=str(target),
                    consensus_result=consensus,
                    evaluations=evaluations or [],
                    gap_summary=gap_summary,
                    model="gpt-4o-mini",
                    prefer_online=True,
                )
                st.session_state.review_board_judge_memo = judge_bundle.get("memo")
                st.session_state.review_board_trace["judge_bundle"] = judge_bundle.get("trace", {})

    judge_memo = st.session_state.review_board_judge_memo
    if judge_memo:
        st.divider()
        st.markdown("### Final judge recommendation")

        j1, j2, j3 = st.columns([1, 1, 1])
        j1.metric("Verdict", str(judge_memo.verdict))
        j2.metric("Recommended strategy", str(judge_memo.recommended_strategy))
        j3.metric("Timeline", str(judge_memo.success_timeline))

        st.markdown("**Executive summary**")
        st.markdown(str(judge_memo.executive_summary))

        col_sf, col_cr = st.columns(2)

        with col_sf:
            _render_bullet_list("Key success factors", getattr(judge_memo, "key_success_factors", []))

        with col_cr:
            _render_bullet_list("Critical risks", getattr(judge_memo, "critical_risks", []))

        st.markdown("**First 30-day actions**")
        for action in getattr(judge_memo, "first_30_day_actions", [])[:5]:
            st.markdown(f"- {action}")

        st.markdown("**Interview narrative**")
        st.info(str(judge_memo.interview_narrative))

        networking_targets = getattr(consensus, "networking_targets", []) or []
        if networking_targets:
            st.markdown("**Suggested networking targets**")
            for row in networking_targets[:4]:
                with st.container(border=True):
                    st.markdown(f"**Target:** {row.get('target', '—')}")
                    st.markdown(f"**Why:** {row.get('why', '—')}")
                    st.markdown(f"**Question to ask:** {row.get('ask', '—')}")

    if evaluations and consensus:
        st.divider()
        with st.expander("Counterfactual: re-rank after skill investment", expanded=False):
            sim_cands = suggest_best_investment_skills(gap_df, top_k=8)
            if sim_cands.empty:
                st.info("No candidate skills available.")
            else:
                skill_options = sim_cands["skill"].astype(str).tolist()
                selected_skills = st.multiselect(
                    "Which skills would you improve?",
                    options=skill_options,
                    default=skill_options[: min(2, len(skill_options))],
                    key="counterfactual_skills",
                )
                uplift_ratio = st.slider(
                    "How close to target level?",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.50,
                    step=0.05,
                    key="counterfactual_uplift",
                )

                if st.button("Re-evaluate strategies", use_container_width=True, key="rerank_board"):
                    with st.spinner("Re-ranking strategies under counterfactual skill investment..."):
                        new_consensus = rerank_after_skill_investment(
                            evaluations=evaluations,
                            invested_skills=selected_skills,
                            uplift_ratio=float(uplift_ratio),
                        )
                        st.session_state.review_board_trace["counterfactual_consensus"] = new_consensus

                new_consensus = st.session_state.review_board_trace.get("counterfactual_consensus")
                if new_consensus:
                    cf1, cf2, cf3 = st.columns(3)
                    cf1.metric("Counterfactual winner", new_consensus.winner_strategy)
                    cf2.metric("New winner score", f"{new_consensus.winner_score:.1f}/100")
                    cf3.metric("New robustness", f"{float(getattr(new_consensus, 'robustness_score', 0.0)):.0f}/100")

                    ranking_cf_df = pd.DataFrame(
                        new_consensus.strategy_rankings,
                        columns=["strategy_code", "avg_score"],
                    )
                    _render_table_card(
                        ranking_cf_df,
                        columns=["strategy_code", "avg_score"],
                        headers=["Strategy", "Counterfactual Score"],
                        numeric_cols=["avg_score"],
                    )

                    if new_consensus.winner_strategy != consensus.winner_strategy:
                        st.success(
                            f"Recommendation flipped: {consensus.winner_strategy} → {new_consensus.winner_strategy}"
                        )
                    else:
                        st.info(f"{new_consensus.winner_strategy} remains the best strategy after skill investment.")


# ============================================================
# Explainability
# ============================================================
with st.container(border=True):
    st.subheader("Explainability")
    st.caption("High-signal view: what transfers versus what blocks this pivot.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Top transferable skills**")
        top_transfer = gap_df.copy()
        top_transfer["overlap"] = np.minimum(top_transfer["current_importance"], top_transfer["target_importance"])
        top_transfer = top_transfer.sort_values("overlap", ascending=False).head(10)

        _render_table_card(
            top_transfer,
            columns=["skill", "current_importance", "target_importance", "overlap"],
            headers=["Skill", "Current", "Target", "Overlap"],
            numeric_cols=["current_importance", "target_importance", "overlap"],
        )

    with c2:
        st.markdown("**Top missing skills**")
        top_missing = gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False).head(10)
        if top_missing.empty:
            st.success("No missing skills detected.")
        else:
            _render_table_card(
                top_missing,
                columns=["skill", "current_importance", "target_importance", "gap"],
                headers=["Skill", "Current", "Target", "Gap"],
                numeric_cols=["current_importance", "target_importance", "gap"],
            )


# ============================================================
# Research extras
# ============================================================
if not guided:
    with st.expander("Research notes", expanded=False):
        st.markdown(
            """
**Confidence** is a heuristic coverage score, not a probability.  
It combines overlap, dataset density, and embedding support.
            """
        )

    with st.expander("Export CSV", expanded=False):
        export_df = gap_df.copy()
        export_df.insert(0, "current_occupation", str(current))
        export_df.insert(1, "target_occupation", str(target))
        export_df.insert(2, "match_score_display", round(match_score_display, 2))
        export_df.insert(3, "score_mode", "percentile" if show_percentile else "raw")
        export_df.insert(4, "use_idf", bool(use_idf))
        export_df.insert(5, "confidence_score", round(float(conf["confidence_score"]), 2))

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download pivot CSV",
            data=csv_bytes,
            file_name=f"pivot_{str(current)}_to_{str(target)}.csv".replace(" ", "_").lower(),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# LLM system trace
# ============================================================
with st.expander("LLM system trace (advanced)", expanded=False):
    st.markdown("### Learning plan source")
    st.write(
        {
            "learning_plan_source": st.session_state.learning_plan_source,
            "streamlit_secret_present": _has_openai_secret(),
        }
    )

    st.markdown("### Review Board trace")
    trace = st.session_state.review_board_trace or {}
    st.write(
        {
            "strategies_bundle": _extract_review_trace_status(trace.get("strategies_bundle", {})),
            "evaluations_bundle": _extract_review_trace_status(trace.get("evaluations_bundle", {})),
            "judge_bundle": trace.get("judge_bundle", {}),
        }
    )