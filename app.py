# app.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.model_logic import (
    load_runtime_artifacts,
    compute_gap_df,
    compute_skill_contributions,
    compute_confidence_score,
    find_pivot_path,
)
from src.ai_coach import generate_learning_plan_markdown


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")





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

/* remove streamlit chrome */
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

/* cards */
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

/* ====== FIX 1: force BLUE radio + switch (BaseWeb + Streamlit tokens) ====== */

/* Force Streamlit theme tokens (some versions drive red from these) */
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

/* RADIO: hit BaseWeb SVG + checked state */
[data-baseweb="radio"] svg{
  color: var(--li-blue) !important;
  fill: var(--li-blue) !important;
}
[data-baseweb="radio"] [aria-checked="true"] svg{
  color: var(--li-blue) !important;
  fill: var(--li-blue) !important;
}
/* Some builds use pseudo elements / borders instead of accent-color */
[data-baseweb="radio"] [aria-checked="true"]{
  border-color: var(--li-blue) !important;
}

/* SWITCH: hit track + knob (BaseWeb) */
[data-baseweb="switch"] [role="switch"][aria-checked="true"]{
  background-color: rgba(10,102,194,0.35) !important;
  border-color: rgba(10,102,194,0.35) !important;
}
[data-baseweb="switch"] [role="switch"][aria-checked="true"] > div{
  background: var(--li-blue) !important;
}

/* ====== FIX 2: buttons - remove outer boxes everywhere ====== */
/* remove outer Streamlit wrapper box but keep layout intact */
.stButton > div,
.stDownloadButton > div{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* BaseWeb button root wrappers */
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

/* actual button */
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
  white-space: nowrap !important;      /* <-- FIX generate button wrap */
  width: auto !important;              /* <-- FIX huge button */
  max-width: 100% !important;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  background: var(--li-blue-dark) !important;
  border: 1px solid var(--li-blue-dark) !important;
}
.stButton > button:focus, .stDownloadButton > button:focus{
  box-shadow: none !important;
  outline: none !important;
}

/* ====== FIX 3: table - remove double bottom line & make lighter ====== */
.li-table-wrap{
  border: 1px solid rgba(0,0,0,0.05) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  background: #fff !important;
  margin: 10px 0 14px 0 !important;   /* <-- IMPORTANT: bottom whitespace */
}
/* ensure last row doesn't create a "double" bottom edge */
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
.li-table tbody tr:last-child td{
  border-bottom: none !important;  /* <-- kills the second line */
}
.li-table .num{
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
}

/* small subtitle */
.li-subtitle{
  margin-top: -4px;
  margin-bottom: 6px;
  color: var(--li-subtext);
  font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True,
)






def _render_table_card(
    df: pd.DataFrame,
    columns: List[str],
    headers: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> None:
    """Crisp HTML table (display only)."""
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


# ============================================================
# Header
# ============================================================
with st.container(border=True):
    st.markdown("## 🧭 Career Pivot Simulator")
    st.markdown(
        '<div class="li-subtitle">Data + AI prototype to explore career pivots: match scoring • explainability • stepping-stone route • AI learning plan.</div>',
        unsafe_allow_html=True,
    )


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
coords_pca: pd.DataFrame = art.coords
occupations: list[str] = mat.index.astype(str).tolist()

X_BASE: np.ndarray = np.asarray(mat.to_numpy(), dtype=np.float32, order="C")
OCCS: list[str] = occupations
OCC_TO_IDX: dict[str, int] = {o: i for i, o in enumerate(OCCS)}
N_OCC: int = X_BASE.shape[0]


# ============================================================
# Session state
# ============================================================
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "target_override" not in st.session_state:
    st.session_state.target_override = None
if "learning_plan_md" not in st.session_state:
    st.session_state.learning_plan_md = ""
if "learning_plan_source" not in st.session_state:
    st.session_state.learning_plan_source = "—"
if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "route_config" not in st.session_state:
    st.session_state.route_config = {"k_neighbors": 10, "max_steps": 6}


# ============================================================
# Scoring core (unchanged)
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

    return {"occs": OCCS, "occ_to_idx": OCC_TO_IDX, "Xn": Xn, "Xn_T": Xn.T, "idf": idf, "n": n}


def _percentile_from_sorted(sorted_vals: np.ndarray, x: float) -> float:
    v = sorted_vals
    if v.size == 0 or not np.isfinite(x):
        return 0.0
    left = int(np.searchsorted(v, x, side="left"))
    right = int(np.searchsorted(v, x, side="right"))
    eq = right - left
    pct = 100.0 * (left + 0.5 * eq) / float(v.size)
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
        pct = 100.0 * (less + 0.5 * eq) / float(n)
        pct_sorted[i:j] = np.float32(pct)
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

    scores = raw_scores_all[mask_other]
    scores_sorted = np.sort(scores)

    return {"scores": scores, "scores_sorted": scores_sorted, "raw_scores_all": raw_scores_all, "mask_other": mask_other}


def recommend_neighbors(use_idf: bool, current_occ: str, top_k: int = 10) -> pd.DataFrame:
    dist = get_score_distribution(bool(use_idf), str(current_occ))
    scores_other: np.ndarray = dist["scores"]
    raw_all: np.ndarray = dist["raw_scores_all"]
    mask_other: np.ndarray = dist["mask_other"]

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

    df = pd.DataFrame(
        {"occupation": occ_other[idx_sorted], "match_raw": raw_other[idx_sorted].astype(float), "match_percentile": pct_other[idx_sorted].astype(float)}
    ).reset_index(drop=True)
    return df


# ============================================================
# Sidebar (2 clean cards)
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
            target = st.selectbox("Target occupation", options=occupations, index=default_target_idx)

            if st.session_state.target_override:
                target = st.session_state.target_override

            if current == target:
                st.warning("Pick a different target (current and target are identical).")

        with st.expander("Scoring", expanded=True):
            use_idf = st.toggle("Downweight common skills (IDF)", value=True)
            score_mode = st.radio(
                "Overview score",
                options=["Percentile (recommended)", "Raw similarity (transparent)"],
                index=0,
            )
            st.caption("Tip: Percentile answers: “How strong is this target vs other options from my current role?”")

        if not guided:
            with st.expander("Research knobs (optional)", expanded=False):
                st.subheader("Stepping-stone route")
                k_neighbors = st.slider("kNN neighbors", 2, 20, int(st.session_state.route_config["k_neighbors"]), 1)
                max_steps = st.slider("Max steps", 2, 10, int(st.session_state.route_config["max_steps"]), 1)
                st.session_state.route_config = {"k_neighbors": int(k_neighbors), "max_steps": int(max_steps)}

        st.divider()
        run = st.button("🚀 Run pivot analysis", use_container_width=True)
        if run:
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
            st.subheader("What you can do here")
            st.markdown(
                """
- Compare a **current** vs **target** occupation using skill-profile similarity.
- See **what transfers** and **what’s missing** (explainability).
- Try a **stepping-stone route** (intermediate roles).
- Generate a **3-phase learning plan** with OpenAI (cached in the session once generated).
                """
            )

    with c2:
        with st.container():
            st.subheader("How to use it (fast)")
            st.markdown(
                """
1) Pick current + target  
2) Click **Run pivot analysis**  
3) Generate the learning plan (optional)  
4) If it’s a hard pivot, choose a stepping-stone suggestion
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
contrib = compute_skill_contributions(gap_df)
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
        st.success("Next action: Strong candidate — generate a learning plan + build 1 portfolio artifact.")
    elif match_score_display >= 45:
        st.info("Next action: Promising but with gaps — consider a stepping-stone role and follow the plan.")
    else:
        st.warning("Next action: Hard pivot in this dataset — start with a stepping-stone role, then re-evaluate.")


# ============================================================
# Main layout
# ============================================================
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    with st.container():
        st.subheader("Career neighborhood")
        st.caption("Closest roles to your current occupation. Great stepping-stone candidates.")

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
            st.markdown("**Try a stepping-stone target:**")
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
                    st.rerun()

with right:
    with st.container():
        st.subheader("Route + Learning plan")
        st.caption("Keep it simple: route is optional. Learning plan is generated on demand.")

        st.markdown("### Stepping-stone route")
        with st.container():
            if guided:
                st.caption("Optional: find intermediate roles that make the pivot more realistic.")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("Find route", use_container_width=True):
                        with st.spinner("Finding a route…"):
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
                st.caption("Research mode: adjust kNN neighbors and max steps in the sidebar.")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Find route (research)", use_container_width=True):
                        cfg = st.session_state.route_config
                        with st.spinner("Finding a route…"):
                            st.session_state.route_result = find_pivot_path(
                                mat,
                                start_occ=str(current),
                                target_occ=str(target),
                                k_neighbors=int(cfg["k_neighbors"]),
                                max_steps=int(cfg["max_steps"]),
                            )
                with c2:
                    if st.button("Reset route", use_container_width=True):
                        st.session_state.route_result = None

            route = st.session_state.route_result
            if not route:
                st.info("Route not computed yet.")
            else:
                if not route.get("reachable"):
                    st.warning("No route found with the current assumptions. Try again.")
                else:
                    path = route.get("path", [])
                    if path:
                        st.write(" → ".join([f"**{p}**" for p in path]))
                    else:
                        st.info("Route computed, but path is empty (unexpected).")

        st.divider()

        st.markdown("### Learning plan (3 phases)")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Generate plan", use_container_width=True):
                with st.spinner("Generating plan (OpenAI if available)…"):
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
                    st.session_state.learning_plan_source = "OpenAI" if md.startswith("🤖") else "Offline"

        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.learning_plan_md = ""
                st.session_state.learning_plan_source = "—"

        st.caption("Output is shown full-width below (no PDF).")


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
# Explainability
# ============================================================
with st.container(border=True):
    st.subheader("Explain (compact)")
    st.caption("High-signal view: what transfers vs what blocks this pivot.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Top transferable skills (overlap)**")
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
        st.markdown("**Top missing skills (gaps)**")
        top_missing = gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False).head(10)
        if top_missing.empty:
            st.success("No missing skills detected in this dataset.")
        else:
            _render_table_card(
                top_missing,
                columns=["skill", "current_importance", "target_importance", "gap"],
                headers=["Skill", "Current", "Target", "Gap"],
                numeric_cols=["current_importance", "target_importance", "gap"],
            )


# ============================================================
# Research-only extras
# ============================================================
if not guided:
    with st.expander("Research: scoring notes (optional)", expanded=False):
        st.markdown(
            """
**What does “Confidence” mean here?**  
It’s a heuristic “coverage” score (not a probability). It combines:
- how much the skill-support overlaps (binary coverage),
- how dense the dataset is,
- and how much variance the 2D embedding explains (from PCA metadata).
            """
        )

    with st.expander("Export (CSV)", expanded=False):
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