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
from src.career_agent import (
    run_career_agent,
    AgentResult,
    AgentStep,
    MODEL_RATIONALE,
)
from src.cv_parser import parse_cv, compute_personal_gap_df
from src.cover_letter import generate_pivot_narrative
from src.job_analyzer import analyze_job_posting
from src.pivot_debate import run_pivot_debate, DebateRound, DebateVerdict
from src.smart_apply import (
    generate_job_listings, generate_application_package, generate_pivot_peers,
    JobListing, ApplicationPackage, PivotPeer,
)
from src.salary_estimator import estimate_salary_impact
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")

st.markdown("""
<style>
/* ── LinkedIn Navbar ─────────────────────────────────────── */
.li-topnav{
  position:sticky;top:0;z-index:9999;
  background:#ffffff;
  border-bottom:1px solid rgba(0,0,0,0.12);
  height:52px;
  display:flex;align-items:center;justify-content:space-between;
  width:calc(100% + 80px);margin-left:-40px;margin-top:-20px;margin-bottom:20px;
  padding:0 clamp(12px,3vw,40px);
  box-shadow:0 1px 3px rgba(0,0,0,0.06);
  box-sizing:border-box;
}
.li-nav-left{display:flex;align-items:center;gap:8px;}
.li-nav-logo{
  background:#0A66C2;color:#fff;font-weight:900;font-size:17px;
  width:32px;height:32px;display:flex;align-items:center;justify-content:center;
  border-radius:4px;flex-shrink:0;letter-spacing:-0.5px;
}
.li-nav-search{
  display:flex;align-items:center;gap:6px;
  background:#EEF3FB;border-radius:4px;
  padding:7px 12px;min-width:220px;
  font-size:13px;color:rgba(0,0,0,0.55);cursor:text;
}
.li-nav-search svg{flex-shrink:0;opacity:0.6;}
.li-nav-center{display:flex;align-items:stretch;gap:0;}
.li-nav-item{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:0 16px;gap:3px;font-size:11px;font-weight:600;color:rgba(0,0,0,0.55);
  cursor:pointer;min-width:56px;height:52px;border-bottom:2px solid transparent;
  transition:color 0.1s;text-decoration:none;
}
.li-nav-item:hover{color:rgba(0,0,0,0.88);}
.li-nav-item.active{color:rgba(0,0,0,0.88);border-bottom:2px solid rgba(0,0,0,0.88);}
.li-nav-item svg{opacity:0.75;}
.li-nav-item.active svg{opacity:1;}
.li-nav-right{display:flex;align-items:center;gap:12px;}
.li-nav-avatar-wrap{display:flex;flex-direction:column;align-items:center;gap:2px;cursor:pointer;}
.li-nav-avatar{
  width:24px;height:24px;border-radius:50%;background:#0A66C2;
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:10px;font-weight:700;
}
.li-nav-me-label{font-size:11px;font-weight:600;color:rgba(0,0,0,0.55);}
.li-nav-vdivider{width:1px;height:28px;background:rgba(0,0,0,0.12);}
.li-nav-premium{
  font-size:12px;font-weight:600;
  background:linear-gradient(135deg,#C37D16,#E6A817);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;cursor:pointer;white-space:nowrap;
}
/* ── Compensate for navbar height ── */
.block-container{padding-top:0!important;}
/* ── LinkedIn page-title breadcrumb ── */
.li-page-crumb{
  display:flex;align-items:center;gap:6px;
  font-size:12px;color:rgba(0,0,0,0.5);margin-bottom:16px;
  padding-top:4px;
}
.li-page-crumb-active{font-weight:700;color:rgba(0,0,0,0.75);}
/* ── Section flow separators (no box) ── */
.li-section{
  background:#ffffff;border-radius:10px;
  border:1px solid rgba(0,0,0,0.08);
  margin-bottom:12px;overflow:hidden;
}
.li-section-head{
  padding:16px 20px 0 20px;
  font-size:16px;font-weight:800;color:rgba(0,0,0,0.9);
}
.li-section-sub{
  padding:4px 20px 12px 20px;
  font-size:13px;color:rgba(0,0,0,0.55);
}
/* ── Job Cards ── */
.li-job-card{
  background:#fff;border:1px solid rgba(0,0,0,0.1);
  border-radius:10px;padding:18px 20px;
  transition:box-shadow 0.15s, border-color 0.15s;
  margin-bottom:10px;
}
.li-job-card:hover{box-shadow:0 4px 16px rgba(0,0,0,0.08);border-color:rgba(0,0,0,0.18);}
.li-job-header{display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;}
.li-job-logo{
  width:48px;height:48px;border-radius:6px;
  background:#EEF3FB;border:1px solid rgba(0,0,0,0.08);
  display:flex;align-items:center;justify-content:center;
  font-size:22px;flex-shrink:0;
}
.li-job-meta{flex:1;}
.li-job-title{font-size:15px;font-weight:700;color:#0A66C2;line-height:1.3;margin-bottom:2px;}
.li-job-company{font-size:13px;font-weight:600;color:rgba(0,0,0,0.8);margin-bottom:2px;}
.li-job-detail{font-size:12px;color:rgba(0,0,0,0.55);margin-bottom:2px;}
.li-job-tags{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 8px 0;}
.li-job-tag{
  font-size:11px;padding:3px 8px;border-radius:12px;
  background:#F3F2EF;color:rgba(0,0,0,0.65);font-weight:500;
}
.li-job-tag-easy{background:#E7F6EC;color:#117A37;font-weight:700;}
.li-match-bar-wrap{margin:8px 0;}
.li-match-bar-label{font-size:11px;font-weight:700;color:rgba(0,0,0,0.55);margin-bottom:3px;display:flex;justify-content:space-between;}
.li-match-bar-bg{height:6px;background:rgba(0,0,0,0.07);border-radius:3px;overflow:hidden;}
.li-match-bar-fill{height:6px;border-radius:3px;transition:width 0.6s;}
.li-job-footer{font-size:11px;color:rgba(0,0,0,0.42);margin-top:8px;display:flex;gap:16px;}
.li-network-note{color:#0A66C2;font-weight:600;}
/* ── Application Package ── */
.li-pkg-section{
  border-left:3px solid #0A66C2;
  padding:12px 16px;margin:12px 0;
  background:#F8FAFF;border-radius:0 8px 8px 0;
}
.li-pkg-label{
  font-size:10px;font-weight:800;letter-spacing:0.08em;
  text-transform:uppercase;color:#0A66C2;margin-bottom:6px;
}
.li-cv-rewrite{
  display:grid;grid-template-columns:1fr 1fr;gap:12px;
  margin:8px 0;
}
.li-cv-before{
  background:#FFF4E5;border-radius:6px;padding:10px 12px;
  font-size:12px;color:rgba(0,0,0,0.65);border:1px solid #F3D7A5;
}
.li-cv-after{
  background:#E7F6EC;border-radius:6px;padding:10px 12px;
  font-size:12px;color:rgba(0,0,0,0.8);border:1px solid #CBEAD5;
  font-weight:500;
}
.li-cv-arrow{font-size:16px;color:#0A66C2;text-align:center;align-self:center;}
/* ── Pivot Peers ── */
.li-peers-wrap{display:flex;flex-direction:column;gap:14px;}
.li-peer{
  display:flex;gap:12px;align-items:flex-start;
  padding:14px 16px;background:#fff;
  border-radius:10px;border:1px solid rgba(0,0,0,0.08);
}
.li-peer-avatar{
  width:44px;height:44px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:15px;font-weight:800;flex-shrink:0;
}
.li-peer-body{flex:1;}
.li-peer-name{font-size:14px;font-weight:700;color:rgba(0,0,0,0.88);}
.li-peer-path{font-size:12px;color:rgba(0,0,0,0.55);margin:1px 0 4px 0;}
.li-peer-company{font-size:12px;font-weight:600;color:#0A66C2;margin-bottom:6px;}
.li-peer-milestone{
  font-size:12px;color:rgba(0,0,0,0.65);background:#EEF3FB;
  border-radius:6px;padding:6px 10px;margin-bottom:6px;
}
.li-peer-quote{font-size:12px;color:rgba(0,0,0,0.6);font-style:italic;line-height:1.5;}
.li-peer-timing{
  font-size:11px;font-weight:700;color:#117A37;
  background:#E7F6EC;border-radius:12px;padding:2px 8px;
  white-space:nowrap;
}
.li-degree{font-size:10px;color:rgba(0,0,0,0.45);margin-top:4px;}
</style>
<nav class="li-topnav">
  <div class="li-nav-left">
    <div class="li-nav-logo">in</div>
    <div class="li-nav-search">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
        <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" stroke-width="1.6"/>
        <path d="M12 12L10 10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
      </svg>
      Search jobs, skills, people…
    </div>
  </div>
  <div class="li-nav-center">
    <a class="li-nav-item" href="#">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
      Home
    </a>
    <a class="li-nav-item" href="#">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>
      My Network
    </a>
    <a class="li-nav-item active" href="#">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20 6H4V4H2v18h2v-2h18v2h2V4h-2v2zM4 16V8h16v8H4zm2-6h2v4H6zm4 0h2v4h-2zm4 0h2v4h-2z"/></svg>
      Jobs
    </a>
    <a class="li-nav-item" href="#">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/></svg>
      Messaging
    </a>
    <a class="li-nav-item" href="#">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>
      Notifications
    </a>
  </div>
  <div class="li-nav-right">
    <div class="li-nav-avatar-wrap">
      <div class="li-nav-avatar">JP</div>
      <span class="li-nav-me-label">Me ▾</span>
    </div>
    <div class="li-nav-vdivider"></div>
    <span class="li-nav-premium">✦ Try Premium</span>
  </div>
</nav>
""", unsafe_allow_html=True)

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
  --shadow: 0 2px 8px rgba(0,0,0,0.06);
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

/* ── Top-level section cards ── */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--li-card) !important;
  border: 1px solid var(--li-border-soft) !important;
  border-radius: var(--radius) !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  padding: 20px 24px !important;
}
/* ── Nested containers: strip all chrome ── */
div[data-testid="stVerticalBlockBorderWrapper"]
  div[data-testid="stVerticalBlockBorderWrapper"]{
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"]{
  background: var(--li-bg) !important;
  border-right: 1px solid var(--li-border-soft) !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
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

/* ── Buttons: strip Streamlit wrapper chrome ── */
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

/* ── Primary buttons: LinkedIn blue ── */
.stButton > button,
.stDownloadButton > button{
  background: #0A66C2 !important;
  border: none !important;
  color: #ffffff !important;
  border-radius: 20px !important;
  height: 36px !important;
  padding: 0 20px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  box-shadow: none !important;
  outline: none !important;
  white-space: nowrap !important;
  width: auto !important;
  max-width: 100% !important;
  transition: background 0.15s !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover{
  background: #004182 !important;
}
.stButton > button:disabled{
  background: rgba(0,0,0,0.12) !important;
  color: rgba(0,0,0,0.35) !important;
}

/* ── Secondary / ghost buttons (Streamlit type="secondary") ── */
.stButton > button[kind="secondary"]{
  background: transparent !important;
  border: 1px solid rgba(0,0,0,0.22) !important;
  color: rgba(0,0,0,0.55) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
}
.stButton > button[kind="secondary"]:hover{
  background: rgba(0,0,0,0.04) !important;
  border-color: rgba(0,0,0,0.35) !important;
  color: rgba(0,0,0,0.80) !important;
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
.status-challenge{
  background:#FDECEA;
  color:#B71C1C;
  border:1px solid #F5C6C3;
}

/* ── Metric cards — minimal, no heavy box ── */
[data-testid="stMetric"]{
  background: transparent !important;
  border: none !important;
  border-left: 3px solid var(--li-blue) !important;
  border-radius: 0 !important;
  padding: 8px 16px !important;
  box-shadow: none !important;
}
[data-testid="stMetricLabel"]{
  font-size: 11px !important;
  font-weight: 700 !important;
  color: var(--li-subtext) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"]{
  font-size: 24px !important;
  font-weight: 800 !important;
  color: var(--li-text) !important;
  line-height: 1.15 !important;
}
[data-testid="stMetricDelta"]{
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"]{
  background: transparent !important;
  border-bottom: 2px solid var(--li-border-soft) !important;
  gap: 4px !important;
  padding-bottom: 0 !important;
}
[data-baseweb="tab"]{
  background: transparent !important;
  border: none !important;
  border-radius: 6px 6px 0 0 !important;
  padding: 8px 18px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--li-subtext) !important;
}
[data-baseweb="tab"][aria-selected="true"]{
  background: var(--li-card) !important;
  color: var(--li-blue) !important;
  border-bottom: 2px solid var(--li-blue) !important;
}
[data-baseweb="tab-panel"]{
  padding-top: 20px !important;
}



/* ── Trace tool cards ── */
.tool-card{
  background: var(--li-card);
  border: 1px solid var(--li-border-soft);
  border-radius: 10px;
  padding: 14px 18px;
  margin-bottom: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.tool-card-header{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.tool-badge{
  display: inline-block;
  background: #EEF3FB;
  color: var(--li-blue);
  border: 1px solid #C7D9F5;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 700;
  font-family: monospace;
}
.tool-timer{
  font-size: 11px;
  color: var(--li-subtext);
}
.thinking-block{
  background: #F8F9FA;
  border-left: 3px solid var(--li-blue);
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin-bottom: 10px;
  font-size: 13px;
  color: var(--li-text);
  line-height: 1.6;
}

/* ── Agent final result hero ── */
.agent-verdict-hero{
  background: linear-gradient(135deg, #EEF3FB 0%, #F8FAFF 100%);
  border: 1px solid #C7D9F5;
  border-radius: 12px;
  padding: 20px 24px;
  margin: 16px 0;
}
.agent-verdict-title{
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--li-subtext);
  margin-bottom: 4px;
}
.agent-verdict-summary{
  font-size: 15px;
  font-weight: 500;
  color: var(--li-text);
  line-height: 1.65;
  margin-top: 10px;
}

/* ── Selectbox ── */
[data-baseweb="select"]{
  border-radius: 8px !important;
}

/* ── Subheaders — tighter, cleaner ── */
h3{
  font-size: 16px !important;
  font-weight: 800 !important;
  color: var(--li-text) !important;
  margin-bottom: 2px !important;
}
h2{
  font-size: 20px !important;
  font-weight: 800 !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"]{
  font-size: 12px !important;
  color: var(--li-subtext) !important;
  margin-top: 2px !important;
}

/* ── Expander ── */
[data-testid="stExpander"]{
  border: 1px solid var(--li-border-soft) !important;
  border-radius: 8px !important;
  background: var(--li-card) !important;
}

/* ── Divider ── */
hr{
  border-color: var(--li-border-soft) !important;
  margin: 20px 0 !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"]{
  border-radius: 8px !important;
  font-size: 13px !important;
}
</style>
""",
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
    # A3 — Career Intelligence Agent
    "agent_result": None,
    "agent_steps": [],
    "agent_running": False,
    # CV — Personal Profile
    "cv_text": "",
    "cv_profile": None,     # dict from parse_cv
    "cv_gap_df": None,      # personal gap df
    # Pivot Narrative
    "pivot_narrative": None,
    # Job Posting Analyzer
    "job_posting_text": "",
    "job_analysis": None,
    # Adversarial Debate
    "debate_result": None,
    # Smart Apply
    "smart_apply_jobs": None,
    "smart_apply_selected_idx": None,
    "smart_apply_package": None,
    # Pivot Peers
    "pivot_peers": None,
    # Salary Estimator
    "salary_result": None,
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
    # LinkedIn profile card header
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0A66C2,#004182);'
        'border-radius:8px 8px 0 0;height:52px;margin:-8px -8px 0 -8px;"></div>'
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'margin-top:-28px;margin-bottom:12px;">'
        '<div style="width:56px;height:56px;border-radius:50%;background:#fff;'
        'border:3px solid #fff;display:flex;align-items:center;justify-content:center;'
        'font-size:22px;font-weight:900;color:#0A66C2;box-shadow:0 2px 8px rgba(0,0,0,0.15)">JP</div>'
        '<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.88);margin-top:6px">Career Pivot Planner</div>'
        '<div style="font-size:11px;color:rgba(0,0,0,0.5);margin-top:1px">Career Intelligence · Jobs</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("Mode", options=["Guided", "Research"], index=0, horizontal=True)
    guided = mode == "Guided"

    st.divider()

    st.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;'
        'color:rgba(0,0,0,0.45);margin-bottom:6px">Your Pivot</div>',
        unsafe_allow_html=True,
    )
    current = st.selectbox("Current occupation", options=occupations, index=0, label_visibility="collapsed")
    st.caption("↑ Current occupation")
    default_target_idx = 1 if len(occupations) > 1 else 0
    selected_target = st.selectbox("Target occupation", options=occupations, index=default_target_idx, label_visibility="collapsed")
    st.caption("↑ Target occupation")

    target = st.session_state.target_override or selected_target

    if current == target:
        st.warning("Pick a different target.")

    st.divider()

    st.markdown("**Scoring**")
    use_idf = st.toggle("Downweight common skills (IDF)", value=True)
    score_mode = st.radio(
        "Score display",
        options=["Percentile", "Raw similarity"],
        index=0,
    )

    if not guided:
        st.divider()
        st.markdown("**Research knobs**")
        k_neighbors = st.slider("kNN neighbors", 2, 20, int(st.session_state.route_config["k_neighbors"]), 1)
        max_steps = st.slider("Max steps", 2, 10, int(st.session_state.route_config["max_steps"]), 1)
        st.session_state.route_config = {"k_neighbors": int(k_neighbors), "max_steps": int(max_steps)}

    st.divider()
    # ── CV Upload ─────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:rgba(0,0,0,0.45);margin-bottom:6px">Your Profile (optional)</div>',
        unsafe_allow_html=True,
    )

    cv_text_input = st.text_area(
        "Paste your CV / résumé",
        value=st.session_state.cv_text,
        height=120,
        placeholder="Paste your CV text here to personalise the analysis...",
        label_visibility="collapsed",
    )

    cv_col_a, cv_col_b = st.columns([2, 1])
    with cv_col_a:
        if st.button("Extract my skills", use_container_width=True):
            if cv_text_input.strip():
                with st.spinner("Analysing CV..."):
                    st.session_state.cv_text = cv_text_input
                    api_key_for_cv = ""
                    try:
                        api_key_for_cv = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                    except Exception:
                        pass
                    result = parse_cv(
                        cv_text=cv_text_input,
                        skill_columns=list(mat.columns),
                        model="gpt-4o-mini",
                        prefer_online=_has_openai_secret(),
                        api_key=api_key_for_cv or None,
                    )
                    st.session_state.cv_profile = result
                    # Recompute personal gap df using current target
                    if "skill_vector" in result:
                        st.session_state.cv_gap_df = compute_personal_gap_df(
                            result["skill_vector"], str(selected_target), mat
                        )
                st.rerun()
            else:
                st.warning("Paste your CV text first.")
    with cv_col_b:
        if st.session_state.cv_profile:
            if st.button("Clear", use_container_width=True, key="clear_cv", type="secondary"):
                st.session_state.cv_text = ""
                st.session_state.cv_profile = None
                st.session_state.cv_gap_df = None
                st.rerun()

    if st.session_state.cv_profile:
        p = st.session_state.cv_profile
        st.markdown(
            f'<span class="status-pill status-ok">✓ Profile loaded · {p.get("skills_mapped_count", 0)} skills mapped</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:rgba(0,0,0,0.45);margin-bottom:6px">Run Analysis</div>',
        unsafe_allow_html=True,
    )
    if st.button("🚀 Run pivot analysis", use_container_width=True, type="primary"):
        st.session_state.has_run = True

    st.caption(f"Dataset · {mat.shape[0]} occupations · {mat.shape[1]} skills")
# ============================================================
# Empty state
# ============================================================
if not st.session_state.has_run:
    st.markdown(
        '<div style="background:#fff;border-radius:10px;border:1px solid rgba(0,0,0,0.08);'
        'padding:32px 36px;margin-bottom:16px;">'
        '<div style="font-size:22px;font-weight:800;color:rgba(0,0,0,0.88);margin-bottom:6px">'
        'Find your next role with confidence</div>'
        '<div style="font-size:14px;color:rgba(0,0,0,0.55);margin-bottom:24px;line-height:1.6">'
        'AI-powered career pivot analysis — skill matching, gap analysis, smart job discovery, '
        'and ready-to-send application packages. Powered by O*NET data + GPT-4.</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
        '<div style="background:#EEF3FB;border-radius:8px;padding:16px 18px;">'
        '<div style="font-size:20px;margin-bottom:6px">🎯</div>'
        '<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.85);margin-bottom:3px">Skill Match Analysis</div>'
        '<div style="font-size:12px;color:rgba(0,0,0,0.55)">Cosine similarity across 161 O*NET skill dimensions. Percentile-ranked against 900+ occupations.</div>'
        '</div>'
        '<div style="background:#EEF3FB;border-radius:8px;padding:16px 18px;">'
        '<div style="font-size:20px;margin-bottom:6px">🤖</div>'
        '<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.85);margin-bottom:3px">9-Tool AI Agent</div>'
        '<div style="font-size:12px;color:rgba(0,0,0,0.55)">Autonomous agentic loop: market signal, route finding, adversarial debate, skill simulation.</div>'
        '</div>'
        '<div style="background:#EEF3FB;border-radius:8px;padding:16px 18px;">'
        '<div style="font-size:20px;margin-bottom:6px">📄</div>'
        '<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.85);margin-bottom:3px">Smart Apply</div>'
        '<div style="font-size:12px;color:rgba(0,0,0,0.55)">AI-curated job matches + instant application kit: cover letter, CV rewrites, InMail, interview prep.</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:20px;font-size:13px;color:rgba(0,0,0,0.5)">'
        '← Select your current and target occupation in the sidebar, then click <strong>Run pivot analysis</strong></div>'
        '</div>',
        unsafe_allow_html=True,
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

show_percentile = score_mode == "Percentile"
match_score_display = float(pct_target if show_percentile else raw_target)

_base_gap_df = compute_gap_df(mat, str(current), str(target))
# Use personal gap if CV is loaded and target matches; otherwise fall back to role-based
_cv_profile = st.session_state.cv_profile
_cv_gap_df = st.session_state.cv_gap_df
if _cv_profile and _cv_gap_df is not None and not _cv_gap_df.empty:
    gap_df = _cv_gap_df
    _personal_mode = True
else:
    gap_df = _base_gap_df
    _personal_mode = False

conf = compute_confidence_score(mat, art.pca_meta, str(current), str(target))
neighbors_df = recommend_neighbors(bool(use_idf), str(current), top_k=10)


# ============================================================
# Overview
# ============================================================
with st.container(border=True):
    _llm_badge = (
        '<span class="status-pill status-ok" style="font-size:11px">● LLM online</span>'
        if _has_openai_secret()
        else '<span class="status-pill status-warn" style="font-size:11px">○ LLM offline</span>'
    )
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'<div style="font-size:12px;color:rgba(0,0,0,0.45);font-weight:600;letter-spacing:0.04em;text-transform:uppercase">Career Intelligence · Jobs</div>'
            f'{_llm_badge}</div>'
            f'<div style="font-size:20px;font-weight:800;color:rgba(0,0,0,0.90);margin-bottom:14px">'
            f'{current} <span style="color:#0A66C2;font-size:18px">→</span> {target}</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        if _personal_mode and _cv_profile:
            p = _cv_profile
            st.markdown(
                f'<div style="background:#EEF3FB;border-radius:8px;padding:10px 14px;text-align:right">'
                f'<div style="font-size:10px;font-weight:800;letter-spacing:0.05em;text-transform:uppercase;color:#0A66C2;margin-bottom:2px">Personal Mode</div>'
                f'<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.85)">{p.get("extracted_role","") or "CV loaded"}</div>'
                f'<div style="font-size:11px;color:rgba(0,0,0,0.5)">{p.get("years_experience",0):.0f} yrs exp · {p.get("skills_mapped_count",0)} skills mapped</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Pivot Readiness Score (synthesises match + gap + CV quality) ──
    _n_gaps = int((gap_df["gap"] > 0).sum()) if not gap_df.empty else 0
    _n_total = len(gap_df) if not gap_df.empty else 1
    _gap_ratio = _n_gaps / max(_n_total, 1)
    _cv_score = min(float(_cv_profile.get("skills_mapped_count", 0)) / 40.0, 1.0) if _cv_profile else 0.5
    _readiness = int(
        0.45 * (match_score_display / 100)
        + 0.30 * (1 - _gap_ratio)
        + 0.25 * _cv_score
        * 100
    )
    _readiness = max(5, min(_readiness, 97))  # keep in [5, 97] — never claim 0 or 100
    _r_color = "#117A37" if _readiness >= 65 else ("#A05A00" if _readiness >= 40 else "#B71C1C")
    _r_label = "Strong" if _readiness >= 65 else ("Promising" if _readiness >= 40 else "Early Stage")
    _weeks = max(4, int((_n_gaps * 3.5) * (1 - match_score_display / 200)))  # rough weeks estimate

    m1, m2, m3, m4 = st.columns(4, gap="large")
    m1.metric("Match Score", f"{match_score_display:.0f} / 100")
    m2.metric("Confidence", f"{conf['confidence_score']:.0f} / 100")
    m3.metric("Skill Gaps", f"{_n_gaps} to close")
    m4.metric("Est. Readiness", f"~{_weeks}w", help="Rough estimate of weeks to be apply-ready based on gap count and match score")

    st.markdown(
        f'<div style="margin:14px 0 8px 0;display:flex;align-items:center;gap:12px;">'
        f'<div style="flex:1;background:rgba(0,0,0,0.07);border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="width:{_readiness}%;height:8px;background:{_r_color};border-radius:4px;transition:width 0.6s;"></div></div>'
        f'<div style="font-size:13px;font-weight:800;color:{_r_color};white-space:nowrap">'
        f'Pivot Readiness: {_readiness}/100 · {_r_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Match score distribution sparkline ──────────────────────
    if scores_all_sorted.size > 10:
        _hist_counts, _hist_edges = np.histogram(scores_all_sorted, bins=40)
        _bin_centers = (_hist_edges[:-1] + _hist_edges[1:]) / 2
        _fig_dist = go.Figure()
        _fig_dist.add_trace(go.Bar(
            x=_bin_centers, y=_hist_counts,
            marker_color=["#0A66C2" if abs(bc - raw_target) < 2.5 else "rgba(10,102,194,0.18)"
                          for bc in _bin_centers],
            hovertemplate="%{x:.0f} score · %{y} roles<extra></extra>",
        ))
        _fig_dist.add_vline(
            x=raw_target, line_color="#0A66C2", line_width=2, line_dash="solid",
            annotation_text=f"  {target[:25]}… ({raw_target:.0f})",
            annotation_font_size=11, annotation_font_color="#0A66C2",
        )
        _fig_dist.update_layout(
            margin=dict(l=0, r=0, t=24, b=0), height=110,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Where this pivot sits among all possible pivots from your current role",
                       font_size=11, font_color="rgba(0,0,0,0.45)", x=0),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bargap=0.05,
        )
        st.plotly_chart(_fig_dist, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
    if _personal_mode:
        st.info("Personal mode active — skill gaps reflect YOUR CV profile vs. the target role, not the O*NET role average.")
    if match_score_display >= 70:
        st.success("Strong overlap — validate the story, build evidence, and compare strategies.")
    elif match_score_display >= 45:
        st.info("Promising with gaps — a stepping-stone or hybrid strategy may outperform a direct pivot.")
    else:
        st.warning("Hard pivot — use route analysis, skill investment, and the review board before deciding.")


# ============================================================
# Smart Apply — AI Job Matching + Application Package Generator
# ============================================================
st.markdown(
    '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;'
    'color:rgba(0,0,0,0.35);margin:8px 0 10px 2px">Jobs · Recommended for you</div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        '<div style="background:#0A66C2;color:#fff;font-weight:900;font-size:13px;'
        'width:24px;height:24px;border-radius:3px;display:flex;align-items:center;'
        'justify-content:center">in</div>'
        '<span style="font-size:17px;font-weight:800;color:rgba(0,0,0,0.9)">Smart Apply</span>'
        '<span style="font-size:12px;color:#fff;background:#0A66C2;border-radius:12px;'
        'padding:2px 8px;font-weight:700">NEW</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    _sa_personal = bool(st.session_state.cv_profile and st.session_state.cv_profile.get("extracted_role"))
    st.caption(
        "AI-curated job matches for your target role + one-click application package: tailored cover letter, "
        "CV bullet rewrites, LinkedIn InMail to the hiring manager, and interview prep guide."
        + (" Personalised to your CV." if _sa_personal else " Upload your CV for personalised output.")
    )
    if _sa_personal:
        st.markdown(
            '<span class="status-pill status-ok">✓ Personalised to your CV profile</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

    _top_transfer_sa = (
        gap_df.assign(overlap=lambda d: np.minimum(d["current_importance"], d["target_importance"]))
        .sort_values("overlap", ascending=False)["skill"].head(5).tolist()
    )
    _top_missing_sa = (
        gap_df[gap_df["gap"] > 0]
        .sort_values(["gap", "target_importance"], ascending=False)["skill"].head(4).tolist()
    )

    sa_col1, sa_col2 = st.columns([2, 1], gap="small")
    with sa_col1:
        if st.button("🔍 Find matching jobs", use_container_width=True, key="sa_find_jobs"):
            with st.spinner("AI is curating job matches for you…"):
                _sa_key = ""
                try:
                    _sa_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass
                st.session_state.smart_apply_jobs = generate_job_listings(
                    target_role=str(target),
                    current_role=str(current),
                    match_score=match_score_display,
                    top_transfer=_top_transfer_sa,
                    top_missing=_top_missing_sa,
                    cv_profile=st.session_state.cv_profile,
                    n_jobs=4,
                    model="gpt-4o-mini",
                    prefer_online=_has_openai_secret(),
                    api_key=_sa_key or None,
                )
                st.session_state.smart_apply_selected_idx = None
                st.session_state.smart_apply_package = None
            st.rerun()
    with sa_col2:
        if st.session_state.smart_apply_jobs:
            if st.button("Clear", key="clear_smart_apply", type="secondary", use_container_width=True):
                st.session_state.smart_apply_jobs = None
                st.session_state.smart_apply_selected_idx = None
                st.session_state.smart_apply_package = None

    # ── Job Cards ──────────────────────────────────────────────
    sa_jobs: Optional[List[JobListing]] = st.session_state.smart_apply_jobs
    if sa_jobs:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.55);margin:16px 0 10px 0">'
            f'{len(sa_jobs)} jobs matched for you · {target}</div>',
            unsafe_allow_html=True,
        )

        _sa_api_key = ""
        try:
            _sa_api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        except Exception:
            pass

        for i, job in enumerate(sa_jobs):
            _match_color = "#117A37" if job.match_score >= 72 else ("#A05A00" if job.match_score >= 52 else "#0A66C2")
            _match_bar_w = job.match_score
            _easy = '<span class="li-job-tag li-job-tag-easy">⚡ Easy Apply</span>' if job.is_easy_apply else ""
            _net = (f'<span class="li-network-note">👥 {job.network_connections} connection{"s" if job.network_connections > 1 else ""} work here</span>'
                    if job.network_connections > 0 else "")

            st.markdown(
                f'<div class="li-job-card">'
                f'  <div class="li-job-header">'
                f'    <div class="li-job-logo">{job.company_emoji}</div>'
                f'    <div class="li-job-meta">'
                f'      <div class="li-job-title">{job.title}</div>'
                f'      <div class="li-job-company">{job.company}</div>'
                f'      <div class="li-job-detail">{job.location} · {job.job_type} · {job.salary_range}</div>'
                f'      <div class="li-job-detail">{job.seniority}</div>'
                f'    </div>'
                f'  </div>'
                f'  <div class="li-job-tags">'
                + "".join([f'<span class="li-job-tag">{r}</span>' for r in job.key_requirements[:3]])
                + _easy
                + f'  </div>'
                f'  <div class="li-match-bar-wrap">'
                f'    <div class="li-match-bar-label">'
                f'      <span>Profile match</span>'
                f'      <span style="color:{_match_color};font-weight:800">{job.match_score}%</span>'
                f'    </div>'
                f'    <div class="li-match-bar-bg">'
                f'      <div class="li-match-bar-fill" style="width:{_match_bar_w}%;background:{_match_color}"></div>'
                f'    </div>'
                f'  </div>'
                f'  <div class="li-job-footer">'
                f'    <span>Posted {job.posted_ago}</span>'
                f'    <span>{job.applicant_count} applicants</span>'
                + (_net if _net else "")
                + f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Apply button per job
            apply_col_a, apply_col_b = st.columns([2, 3])
            with apply_col_a:
                if st.button(
                    f"{'⚡ Easy Apply' if job.is_easy_apply else '📄 Generate Application Package'}",
                    key=f"sa_apply_{i}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Generating your personalised package for {job.company}…"):
                        st.session_state.smart_apply_selected_idx = i
                        st.session_state.smart_apply_package = generate_application_package(
                            job=job,
                            current_role=str(current),
                            target_role=str(target),
                            cv_profile=st.session_state.cv_profile,
                            top_transfer=_top_transfer_sa,
                            top_missing=_top_missing_sa,
                            model="gpt-4o",
                            prefer_online=_has_openai_secret(),
                            api_key=_sa_api_key or None,
                        )
                    st.rerun()

            # Show package if this is the selected job
            pkg: Optional[ApplicationPackage] = st.session_state.smart_apply_package
            if pkg and st.session_state.smart_apply_selected_idx == i:
                st.markdown(
                    f'<div style="background:#EEF3FB;border-radius:10px;padding:16px 20px;margin:8px 0 16px 0;">'
                    f'<div style="font-size:10px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;'
                    f'color:#0A66C2;margin-bottom:4px">Application Package · {pkg.job_title} @ {pkg.company}</div>'
                    f'<div style="font-size:13px;font-weight:600;color:rgba(0,0,0,0.75);font-style:italic;line-height:1.5">'
                    f'"{pkg.positioning_statement}"</div></div>',
                    unsafe_allow_html=True,
                )

                pkg_tab1, pkg_tab2, pkg_tab3, pkg_tab4 = st.tabs(
                    ["📄 Cover Letter", "✏️ CV Rewrites", "💬 LinkedIn InMail", "🎯 Interview Prep"]
                )

                with pkg_tab1:
                    st.markdown(
                        f'<div class="li-pkg-section"><div class="li-pkg-label">Cover Letter</div>'
                        f'<div style="font-size:13px;line-height:1.75;white-space:pre-wrap;color:rgba(0,0,0,0.8)">'
                        f'{pkg.cover_letter}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "Download cover letter",
                        data=pkg.cover_letter,
                        file_name=f"cover_letter_{job.company.replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_cover_{i}",
                    )

                with pkg_tab2:
                    st.markdown(
                        '<div class="li-pkg-section"><div class="li-pkg-label">CV Bullet Rewrites</div>'
                        '<div style="font-size:12px;color:rgba(0,0,0,0.5);margin-bottom:12px">'
                        'Before → After · Optimised for this specific role</div></div>',
                        unsafe_allow_html=True,
                    )
                    for rewrite in pkg.cv_bullet_rewrites:
                        st.markdown(
                            f'<div style="margin-bottom:16px">'
                            f'<div style="font-size:11px;font-weight:700;color:#0A66C2;text-transform:uppercase;'
                            f'letter-spacing:0.05em;margin-bottom:6px">Skill targeted: {rewrite.skill_highlighted}</div>'
                            f'<div class="li-cv-rewrite">'
                            f'<div class="li-cv-before"><strong style="font-size:10px;text-transform:uppercase;color:rgba(0,0,0,0.45)">Before</strong><br>{rewrite.original}</div>'
                            f'<div class="li-cv-after"><strong style="font-size:10px;text-transform:uppercase;color:#117A37">After</strong><br>{rewrite.rewritten}</div>'
                            f'</div>'
                            f'<div style="font-size:11px;color:rgba(0,0,0,0.5);margin-top:4px;font-style:italic">💡 {rewrite.why}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with pkg_tab3:
                    st.markdown(
                        f'<div class="li-pkg-section"><div class="li-pkg-label">LinkedIn InMail to {job.hiring_manager_name}, {job.hiring_manager_title}</div>'
                        f'<div style="font-size:13px;line-height:1.75;white-space:pre-wrap;color:rgba(0,0,0,0.8)">'
                        f'{pkg.linkedin_inmail}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "Download InMail",
                        data=pkg.linkedin_inmail,
                        file_name=f"inmail_{job.hiring_manager_name.replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"dl_inmail_{i}",
                    )

                with pkg_tab4:
                    for qi, q in enumerate(pkg.interview_prep, 1):
                        with st.expander(f"Q{qi}: {q.get('question', '')}", expanded=qi == 1):
                            st.markdown(
                                f'<div class="li-pkg-section">'
                                f'<div class="li-pkg-label">Model Answer Strategy</div>'
                                f'<div style="font-size:13px;line-height:1.65;color:rgba(0,0,0,0.8)">{q.get("model_answer","")}</div>'
                                f'</div>'
                                f'<div style="font-size:12px;color:rgba(0,0,0,0.5);margin-top:8px;font-style:italic">'
                                f'Why asked: {q.get("why_asked","")}</div>',
                                unsafe_allow_html=True,
                            )

                st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── Pivot Peers — social proof ──────────────────────────────
    st.divider()
    st.markdown(
        '<div style="font-size:16px;font-weight:800;color:rgba(0,0,0,0.88);margin-bottom:4px">'
        '👥 People who made this pivot</div>'
        '<div style="font-size:13px;color:rgba(0,0,0,0.55);margin-bottom:14px">'
        'Anonymised success stories from professionals who transitioned from '
        f'{current} → {target}</div>',
        unsafe_allow_html=True,
    )
    pp_col1, pp_col2 = st.columns([2, 1], gap="small")
    with pp_col1:
        if st.button("Show pivot peers", use_container_width=True, key="sa_pivot_peers"):
            with st.spinner("Finding peer success stories…"):
                _pp_key = ""
                try:
                    _pp_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass
                _route_steps = None
                if st.session_state.route_result and st.session_state.route_result.get("reachable"):
                    _route_steps = st.session_state.route_result.get("path", [])
                st.session_state.pivot_peers = generate_pivot_peers(
                    current_role=str(current),
                    target_role=str(target),
                    match_score=match_score_display,
                    route_steps=_route_steps,
                    n_peers=3,
                    model="gpt-4o-mini",
                    prefer_online=_has_openai_secret(),
                    api_key=_pp_key or None,
                )
            st.rerun()
    with pp_col2:
        if st.session_state.pivot_peers:
            if st.button("Clear", key="clear_peers", type="secondary", use_container_width=True):
                st.session_state.pivot_peers = None

    peers_list: Optional[List[PivotPeer]] = st.session_state.pivot_peers
    if peers_list:
        peers_html = '<div class="li-peers-wrap">'
        for peer in peers_list:
            degree_label = f"• {peer.connection_degree}nd connection" if peer.connection_degree == 2 else "• 1st connection"
            peers_html += (
                f'<div class="li-peer">'
                f'  <div class="li-peer-avatar" style="background:{peer.avatar_color}">{peer.initials}</div>'
                f'  <div class="li-peer-body">'
                f'    <div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'      <div>'
                f'        <div class="li-peer-name">{peer.name}</div>'
                f'        <div class="li-peer-path">{peer.previous_role} → {peer.current_role}</div>'
                f'        <div class="li-peer-company">{peer.current_role} at {peer.company_now}</div>'
                f'      </div>'
                f'      <span class="li-peer-timing">{peer.months_to_pivot} months</span>'
                f'    </div>'
                f'    <div class="li-peer-milestone">🔑 {peer.key_milestone}</div>'
                f'    <div class="li-peer-quote">"{peer.testimonial}"</div>'
                f'    <div class="li-degree">{degree_label}</div>'
                f'  </div>'
                f'</div>'
            )
        peers_html += '</div>'
        st.markdown(peers_html, unsafe_allow_html=True)


st.markdown(
    '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;'
    'color:rgba(0,0,0,0.35);margin:16px 0 10px 2px">Skill Analysis · Your pivot profile</div>',
    unsafe_allow_html=True,
)
# ============================================================
# Main layout — tabbed
# ============================================================
with st.container(border=True):
    _tab_labels = ["Career Neighborhood", "Route & Simulation", "Skill Gaps"]
    if _personal_mode:
        _tab_labels.append("My Profile")
    _tabs = st.tabs(_tab_labels)
    main_tab_nbhd = _tabs[0]
    main_tab_route = _tabs[1]
    main_tab_explain = _tabs[2]
    main_tab_profile = _tabs[3] if _personal_mode else None

    # ── Tab 1: Career Neighborhood ─────────────────────────────
    with main_tab_nbhd:
        st.caption("Closest roles to your current occupation — useful stepping-stone candidates.")

        show_df = neighbors_df.copy()
        show_df["match_raw"] = show_df["match_raw"].round(2)
        show_df["match_percentile"] = show_df["match_percentile"].round(2)

        # Summary chips above table
        if not show_df.empty:
            top_match = show_df.iloc[0]
            st.markdown(
                f'<span class="status-pill status-ok">Top match: {str(top_match["occupation"])[:35]} ({float(top_match["match_percentile"]):.0f}th pct)</span>'
                f'<span class="status-pill status-warn">{len(show_df)} neighbors found</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        _render_table_card(
            show_df,
            columns=["occupation", "match_percentile", "match_raw"],
            headers=["Occupation", "Match (pct)", "Match (raw)"],
            numeric_cols=["match_percentile", "match_raw"],
        )

        st.markdown("**Set a stepping-stone as target**")
        if show_df.empty:
            st.info("No recommendations available.")
        else:
            label_to_occ: Dict[str, str] = {}
            options = []
            for _, r in show_df.head(8).iterrows():
                occ = str(r["occupation"])
                label = f"{occ} — {float(r['match_percentile']):.0f} pct"
                options.append(label)
                label_to_occ[label] = occ

            pick = st.selectbox("Recommended targets", options=options, index=0, label_visibility="collapsed")
            if st.button("Use as target →", use_container_width=True):
                st.session_state.target_override = label_to_occ[pick]
                st.session_state.has_run = True
                st.session_state.route_result = None
                st.session_state.review_board_strategies = None
                st.session_state.review_board_evaluations = None
                st.session_state.review_board_consensus = None
                st.session_state.review_board_judge_memo = None
                st.rerun()

    # ── Tab 2: Route & Simulation ──────────────────────────────
    with main_tab_route:
        route_col, sim_col = st.columns([1, 1], gap="large")

        with route_col:
            st.markdown("**Stepping-stone route**")
            if guided:
                st.caption("Find intermediate roles that make the pivot more realistic.")
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
                    if st.button("Reset", use_container_width=True, key="reset_route_guided", type="secondary"):
                        st.session_state.route_result = None
            else:
                st.caption("Research mode: custom graph settings from the sidebar.")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    if st.button("Find route", use_container_width=True):
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
                    if st.button("Reset", use_container_width=True, key="reset_route_research", type="secondary"):
                        st.session_state.route_result = None

            route = st.session_state.route_result
            if not route:
                st.info("Route not computed yet.")
            elif not route.get("reachable"):
                st.warning("No route found with the current assumptions.")
            else:
                path = route.get("path", [])
                if path:
                    # Visual path display
                    path_html = " <span style='color:#0A66C2;font-weight:800'>→</span> ".join(
                        [f"<span style='font-weight:600'>{p}</span>" for p in path]
                    )
                    st.markdown(
                        f'<div style="background:#EEF3FB;border-radius:8px;padding:12px 16px;font-size:13px;margin-top:8px">{path_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{len(path)} steps · {len(path)-1} hops")
                else:
                    st.success("Route computed.")

        with sim_col:
            st.markdown("**Skill investment simulator**")
            st.caption("Counterfactual: how much does the match improve if you close selected gaps?")

            sim_candidates_df = suggest_best_investment_skills(gap_df, top_k=8)

            if sim_candidates_df.empty:
                st.info("No positive skill gaps available for simulation.")
            else:
                skill_options = sim_candidates_df["skill"].astype(str).tolist()
                default_pick = skill_options[: min(3, len(skill_options))]

                selected_sim_skills = st.multiselect(
                    "Skills to improve",
                    options=skill_options,
                    default=default_pick,
                    label_visibility="collapsed",
                )

                uplift_ratio = st.slider(
                    "Gap closure %",
                    min_value=0.10,
                    max_value=1.00,
                    value=0.50,
                    step=0.05,
                    format="%.0f%%",
                    help="How much of each skill gap you close",
                )

                q1, q2 = st.columns([2, 1])
                with q1:
                    if st.button("Run simulation", use_container_width=True):
                        st.session_state.sim_result = simulate_skill_investment(
                            mat,
                            current_role=str(current),
                            target_role=str(target),
                            selected_skills=selected_sim_skills,
                            uplift_ratio=float(uplift_ratio),
                        )
                with q2:
                    if st.button("Clear", use_container_width=True, key="clear_sim", type="secondary"):
                        st.session_state.sim_result = None

            sim_result = st.session_state.sim_result
            if sim_result:
                st.divider()
                before = float(sim_result.get("similarity_before", 0))
                after = float(sim_result.get("similarity_after", 0))
                delta = after - before
                s1, s2, s3 = st.columns(3)
                s1.metric("Before", f"{before:.1f}")
                s2.metric("After", f"{after:.1f}", delta=f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}")
                s3.metric("Skills invested", len(sim_result.get("invested_skills", [])))

    # ── Tab 3: Skill Gaps ──────────────────────────────────────
    with main_tab_explain:
        st.caption("High-signal view: what transfers versus what blocks this pivot.")

        n_missing = int((gap_df["gap"] > 0).sum()) if not gap_df.empty else 0
        n_transfer = len(gap_df) - n_missing if not gap_df.empty else 0
        st.markdown(
            f'<span class="status-pill status-ok">{n_transfer} transferable skills</span>'
            f'<span class="status-pill status-challenge">{n_missing} skill gaps</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

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

        # ── Skill gap visual chart ────────────────────────────────
        if not gap_df.empty:
            _chart_df = gap_df.copy()
            _chart_df["overlap"] = np.minimum(
                _chart_df["current_importance"], _chart_df["target_importance"]
            )
            _top_skills = (
                _chart_df.assign(
                    abs_target=_chart_df["target_importance"]
                ).sort_values("abs_target", ascending=False).head(14)
            )
            _fig_gap = go.Figure()
            _fig_gap.add_trace(go.Bar(
                name="You have",
                x=_top_skills["skill"],
                y=_top_skills["current_importance"],
                marker_color="#0A66C2",
                hovertemplate="%{x}<br>Your level: %{y:.1f}<extra></extra>",
            ))
            _fig_gap.add_trace(go.Bar(
                name="Role requires",
                x=_top_skills["skill"],
                y=(_top_skills["target_importance"] - _top_skills["current_importance"]).clip(lower=0),
                base=_top_skills["current_importance"],
                marker_color="rgba(183,28,28,0.35)",
                hovertemplate="%{x}<br>Gap: %{y:.1f}<extra></extra>",
            ))
            _fig_gap.update_layout(
                barmode="stack",
                height=240,
                margin=dict(l=0, r=0, t=28, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(
                    text="Skill profile: your level (blue) vs. gap to close (red) · Top 14 skills by target importance",
                    font_size=11, font_color="rgba(0,0,0,0.45)", x=0,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font_size=11),
                xaxis=dict(tickfont_size=10, tickangle=-30, showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                           title=dict(text="O*NET level (0–7)", font_size=10)),
            )
            st.plotly_chart(_fig_gap, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 4: My Profile (only when CV loaded) ────────────────
    if _personal_mode and main_tab_profile is not None and _cv_profile:
        with main_tab_profile:
            p = _cv_profile
            st.caption("Your extracted skill profile, mapped to the O*NET skill space.")

            # Header metrics
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Role (from CV)", str(p.get("extracted_role", "—") or "—")[:25])
            pm2.metric("Experience", f"{p.get('years_experience', 0):.0f} years")
            pm3.metric("Education", str(p.get("education_level", "—") or "—")[:20])
            pm4.metric("Skills mapped", f"{p.get('skills_mapped_count', 0)} / {len(mat.columns)}")

            conf_val = float(p.get("confidence", 0))
            conf_cls = "status-ok" if conf_val >= 0.7 else ("status-warn" if conf_val >= 0.4 else "status-challenge")
            st.markdown(
                f'<span class="status-pill {conf_cls}">Extraction confidence: {conf_val:.0%}</span>'
                f'<span class="status-pill status-ok">Source: {p.get("source","?")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

            # Top skills
            top_skills = p.get("top_skills", [])
            if top_skills:
                st.markdown("**Your strongest skills (O*NET mapped)**")
                vec = p.get("skill_vector")
                if vec is not None:
                    top_df = pd.DataFrame({
                        "skill": top_skills,
                        "your_level": [round(float(vec[s]), 2) if s in vec.index else 0.0 for s in top_skills],
                        "target_level": [
                            round(float(mat.loc[str(target), s]), 2)
                            if str(target) in mat.index and s in mat.columns else 0.0
                            for s in top_skills
                        ],
                    })
                    top_df["gap"] = (top_df["target_level"] - top_df["your_level"]).clip(lower=0).round(2)
                    _render_table_card(
                        top_df,
                        columns=["skill", "your_level", "target_level", "gap"],
                        headers=["Skill", "Your Level", "Target Level", "Gap"],
                        numeric_cols=["your_level", "target_level", "gap"],
                    )

            # Raw extracted skills
            raw = p.get("extracted_skills_raw", [])
            if raw:
                with st.expander("Raw extracted skills from CV", expanded=False):
                    raw_df = pd.DataFrame(raw)
                    if not raw_df.empty and "skill" in raw_df.columns:
                        _render_table_card(
                            raw_df,
                            columns=[c for c in ["skill", "level", "evidence"] if c in raw_df.columns],
                            headers=[c.title() for c in ["skill", "level", "evidence"] if c in raw_df.columns],
                        )




st.markdown(
    '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;'
    'color:rgba(0,0,0,0.35);margin:16px 0 10px 2px">Prepare · Close your skill gaps</div>',
    unsafe_allow_html=True,
)
# ============================================================
# LLM Learning Plan
# ============================================================
with st.container(border=True):
    st.subheader("🧠 AI Learning Plan")
    st.caption("LLM-generated upskilling roadmap based on your skill gaps.")

    lp1, lp2 = st.columns([2, 1], gap="small")

    with lp1:
        if st.button("Generate learning plan", use_container_width=True):
            with st.spinner("Generating..."):
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

    with lp2:
        if st.button("Clear", use_container_width=True, key="clear_learning_plan", type="secondary"):
            st.session_state.learning_plan_md = ""
            st.session_state.learning_plan_source = "—"

    plan_md = (st.session_state.learning_plan_md or "").strip()
    if plan_md:
        st.divider()
        st.caption(f"Source: {st.session_state.learning_plan_source}")
        st.markdown(plan_md)

# ============================================================
# Salary Impact Estimator
# ============================================================
with st.container(border=True):
    st.subheader("💰 Salary Impact Estimator")
    _si_personal = bool(_cv_profile and _cv_profile.get("years_experience", 0) > 0)
    st.caption(
        "LLM-estimated compensation trajectory for this pivot — current salary vs. "
        "target entry vs. target senior level, with break-even timeline."
        + (" Personalised to your CV." if _si_personal else "")
    )
    st.markdown(
        '<span style="font-size:10px;color:rgba(0,0,0,0.4);font-style:italic">'
        'Figures are AI-estimated based on US labour market data — use as directional guidance only.</span>',
        unsafe_allow_html=True,
    )

    si_col1, si_col2 = st.columns([2, 1], gap="small")
    with si_col1:
        if st.button("Estimate salary impact", use_container_width=True, key="run_salary"):
            with st.spinner("Modelling compensation trajectory…"):
                _si_key = ""
                try:
                    _si_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass
                _yrs = float(_cv_profile.get("years_experience", 0)) if _cv_profile else 0.0
                st.session_state.salary_result = estimate_salary_impact(
                    current_role=str(current),
                    target_role=str(target),
                    match_score=match_score_display,
                    years_experience=_yrs,
                    model="gpt-4o-mini",
                    prefer_online=_has_openai_secret(),
                    api_key=_si_key or None,
                )
            st.rerun()
    with si_col2:
        if st.session_state.salary_result:
            if st.button("Clear", key="clear_salary", type="secondary", use_container_width=True):
                st.session_state.salary_result = None

    _sr = st.session_state.salary_result
    if _sr:
        st.divider()
        _s1, _s2, _s3, _s4 = st.columns(4, gap="large")
        _s1.metric(
            "Current (median)",
            f"${_sr['current_median']:,.0f}",
            help=f"Range: ${_sr['current_range'][0]:,.0f} – ${_sr['current_range'][1]:,.0f}",
        )
        _entry_delta = _sr["entry_delta_pct"]
        _s2.metric(
            "Target entry (median)",
            f"${_sr['target_entry_median']:,.0f}",
            delta=f"{_entry_delta:+.1f}%",
            delta_color="inverse" if _entry_delta < 0 else "normal",
        )
        _ceiling_delta = _sr["ceiling_delta_pct"]
        _s3.metric(
            "Target senior (median)",
            f"${_sr['target_senior_median']:,.0f}",
            delta=f"{_ceiling_delta:+.1f}%",
            delta_color="normal",
        )
        _s4.metric(
            "Break-even",
            f"{_sr['months_to_breakeven']} months",
            help="Months from entering target role until salary exceeds current",
        )

        # ── Salary trajectory chart ──────────────────────────────
        _traj = _sr.get("trajectory", [])
        if _traj:
            _months = [p["month"] for p in _traj]
            _salaries = [p["salary"] for p in _traj]
            _phases = [p.get("phase", "") for p in _traj]
            _current_line = [_sr["current_median"]] * len(_months)

            _fig_sal = go.Figure()
            # Current role flat line
            _fig_sal.add_trace(go.Scatter(
                x=_months, y=_current_line,
                mode="lines",
                name="Current role (stay)",
                line=dict(color="rgba(0,0,0,0.25)", width=1.5, dash="dot"),
                hovertemplate="Month %{x}: $%{y:,.0f} (stay)<extra></extra>",
            ))
            # Target trajectory
            _point_colors = ["#0A66C2" if s == "Growth" else "#A05A00" for s in _phases]
            _fig_sal.add_trace(go.Scatter(
                x=_months, y=_salaries,
                mode="lines+markers",
                name="Pivot trajectory",
                line=dict(color="#0A66C2", width=2.5),
                marker=dict(size=8, color=_point_colors, line=dict(color="#fff", width=1.5)),
                hovertemplate="Month %{x}: $%{y:,.0f}<extra></extra>",
                fill="tonexty",
                fillcolor="rgba(10,102,194,0.06)",
            ))
            # Break-even line
            if _sr["months_to_breakeven"] <= 36:
                _fig_sal.add_vline(
                    x=_sr["months_to_breakeven"],
                    line_color="#117A37", line_width=1.5, line_dash="dash",
                    annotation_text=f"  Break-even: month {_sr['months_to_breakeven']}",
                    annotation_font_size=10, annotation_font_color="#117A37",
                )

            _fig_sal.update_layout(
                height=240,
                margin=dict(l=0, r=0, t=28, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(
                    text="Estimated 36-month salary trajectory · AI simulation based on US market data",
                    font_size=11, font_color="rgba(0,0,0,0.45)", x=0,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font_size=11),
                xaxis=dict(
                    title=dict(text="Months after pivot start", font_size=10),
                    showgrid=False, zeroline=False,
                    tickvals=[0, 6, 12, 18, 24, 36],
                    ticktext=["Now", "6m", "12m", "18m", "24m", "36m"],
                ),
                yaxis=dict(
                    title=dict(text="Annual salary (USD)", font_size=10),
                    showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                    tickformat="$,.0f",
                ),
            )
            st.plotly_chart(_fig_sal, use_container_width=True, config={"displayModeBar": False})

        # Insights
        for insight in _sr.get("insights", []):
            st.markdown(
                f'<div style="display:flex;gap:8px;margin-bottom:6px;font-size:13px;'
                f'color:rgba(0,0,0,0.7);line-height:1.5">'
                f'<span style="color:#0A66C2;flex-shrink:0">›</span>{insight}</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# Pivot Narrative Generator
# ============================================================
with st.container(border=True):
    st.subheader("✍️ Pivot Narrative Generator")
    _pn_personal = bool(st.session_state.cv_profile and st.session_state.cv_profile.get("extracted_role"))
    st.caption(
        ("Personalised to your CV — cover letter, elevator pitch, LinkedIn About, and interview talking points." if _pn_personal
         else "Generate application materials for this pivot. Upload your CV in the sidebar to personalise the output.")
    )
    if _pn_personal:
        st.markdown(
            '<span class="status-pill status-ok">✓ Personal mode — output tailored to your CV</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    pn1, pn2 = st.columns([2, 1], gap="small")
    with pn1:
        if st.button("Generate pivot narrative", use_container_width=True):
            with st.spinner("Generating your personalised pivot materials..."):
                _top_transfer_pn = (
                    gap_df.assign(overlap=lambda d: np.minimum(d["current_importance"], d["target_importance"]))
                    .sort_values("overlap", ascending=False)["skill"].head(5).tolist()
                )
                _top_missing_pn = (
                    gap_df[gap_df["gap"] > 0]
                    .sort_values(["gap", "target_importance"], ascending=False)["skill"].head(4).tolist()
                )
                _agent_summary = None
                if st.session_state.agent_result:
                    _agent_summary = st.session_state.agent_result.executive_summary
                _api_key_pn = ""
                try:
                    _api_key_pn = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass
                narrative = generate_pivot_narrative(
                    current_role=str(current),
                    target_role=str(target),
                    recommended_strategy=(
                        st.session_state.agent_result.recommended_strategy
                        if st.session_state.agent_result else
                        (st.session_state.review_board_consensus.winner_strategy
                         if st.session_state.review_board_consensus else "HYBRID")
                    ),
                    top_transfer=_top_transfer_pn,
                    top_missing=_top_missing_pn,
                    match_score=match_score_display,
                    verdict=(
                        st.session_state.agent_result.verdict
                        if st.session_state.agent_result else "Feasible with Conditions"
                    ),
                    cv_profile=st.session_state.cv_profile,
                    agent_executive_summary=_agent_summary,
                    model="gpt-4o-mini",
                    prefer_online=_has_openai_secret(),
                    api_key=_api_key_pn or None,
                )
                st.session_state.pivot_narrative = narrative
    with pn2:
        if st.session_state.pivot_narrative:
            if st.button("Clear", key="clear_narrative", type="secondary", use_container_width=True):
                st.session_state.pivot_narrative = None

    pn = st.session_state.pivot_narrative
    if pn:
        st.divider()
        src_label = "Personalised (CV + analysis)" if pn.get("personalized") else "Role-based (O*NET)"
        st.caption(f"Source: {pn.get('source','?')} · {src_label}")

        pn_tab1, pn_tab2, pn_tab3, pn_tab4 = st.tabs(
            ["Cover Letter", "Elevator Pitch", "LinkedIn About", "Talking Points"]
        )

        with pn_tab1:
            cl_text = pn.get("cover_letter", "")
            if cl_text:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:10px;padding:24px 28px;font-size:14px;line-height:1.75;white-space:pre-wrap">{cl_text}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download cover letter",
                    data=cl_text,
                    file_name=f"cover_letter_{str(target).replace(' ', '_').lower()}.txt",
                    mime="text/plain",
                )

        with pn_tab2:
            pitch = pn.get("elevator_pitch", "")
            if pitch:
                st.markdown(
                    f'<div class="agent-verdict-hero"><div class="agent-verdict-title">Elevator Pitch</div>'
                    f'<div class="agent-verdict-summary">{pitch}</div></div>',
                    unsafe_allow_html=True,
                )

        with pn_tab3:
            about = pn.get("linkedin_about", "")
            if about:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:10px;padding:20px 24px;font-size:14px;line-height:1.75;white-space:pre-wrap">{about}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Copy LinkedIn About",
                    data=about,
                    file_name="linkedin_about.txt",
                    mime="text/plain",
                    key="dl_linkedin",
                )

        with pn_tab4:
            points = pn.get("talking_points", [])
            for i, pt in enumerate(points, 1):
                st.markdown(
                    f'<div style="padding:12px 16px;border-left:3px solid #0A66C2;background:#F8FAFF;border-radius:0 8px 8px 0;margin-bottom:10px;font-size:14px">'
                    f'<span style="font-weight:700;color:#0A66C2;margin-right:8px">{i}.</span>{pt}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# Job Posting Analyzer
# ============================================================
with st.container(border=True):
    st.subheader("🎯 Job Posting Analyzer")
    _jp_personal = bool(st.session_state.cv_profile and st.session_state.cv_profile.get("extracted_role"))
    st.caption(
        "Paste a real job posting — get an instant match score, advantage/gap breakdown, and application readiness verdict."
        + (" Personalised to your CV." if _jp_personal else " Upload your CV for personalised results.")
    )

    job_text_input = st.text_area(
        "Job posting",
        value=st.session_state.job_posting_text,
        height=180,
        placeholder="Paste the full job description here...\n\nWorks best with the full posting including requirements, responsibilities, and qualifications.",
        label_visibility="collapsed",
    )

    ja_col1, ja_col2 = st.columns([2, 1], gap="small")
    with ja_col1:
        if st.button("Analyse this job posting", use_container_width=True):
            if job_text_input.strip():
                with st.spinner("Extracting requirements and computing match..."):
                    st.session_state.job_posting_text = job_text_input
                    _ja_key = ""
                    try:
                        _ja_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                    except Exception:
                        pass
                    ja_result = analyze_job_posting(
                        job_text=job_text_input,
                        skill_columns=list(mat.columns),
                        matrix=mat,
                        current_role=str(current),
                        target_role=str(target),
                        cv_profile=st.session_state.cv_profile,
                        model="gpt-4o-mini",
                        prefer_online=_has_openai_secret(),
                        api_key=_ja_key or None,
                    )
                    st.session_state.job_analysis = ja_result
                st.rerun()
            else:
                st.warning("Paste a job posting first.")
    with ja_col2:
        if st.session_state.job_analysis:
            if st.button("Clear", key="clear_job", type="secondary", use_container_width=True):
                st.session_state.job_analysis = None
                st.session_state.job_posting_text = ""

    ja = st.session_state.job_analysis
    if ja and not ja.get("error"):
        st.divider()

        # Header row
        readiness = ja.get("application_readiness", "?")
        r_cls = {"Strong": "status-ok", "Moderate": "status-warn", "Stretch": "status-challenge"}.get(readiness, "status-warn")
        st.markdown(
            f'<div style="font-size:20px;font-weight:800;margin-bottom:4px">{ja.get("role_title","?")} '
            f'<span style="font-size:14px;font-weight:400;color:rgba(0,0,0,0.5)">at {ja.get("company","?")}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="status-pill {r_cls}">Application readiness: {readiness}</span>'
            f'<span class="status-pill status-ok">Match: {ja.get("match_score",0):.0f}/100</span>'
            + (f'<span class="status-pill status-warn">vs. {ja.get("user_label","profile")}</span>' if _jp_personal else ""),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
        st.caption(ja.get("readiness_rationale", ""))

        # Columns: matches vs gaps
        jc1, jc2 = st.columns(2, gap="large")
        with jc1:
            st.markdown("**Your advantages for this role**")
            matches = ja.get("top_matches", [])
            if matches:
                match_df = pd.DataFrame(matches)
                _render_table_card(
                    match_df,
                    columns=["skill", "your_level", "job_requires"],
                    headers=["Skill", "Your Level", "Job Needs"],
                    numeric_cols=["your_level", "job_requires"],
                )
            else:
                st.info("No strong skill matches detected.")

        with jc2:
            st.markdown("**Gaps to address before applying**")
            gaps = ja.get("top_gaps", [])
            if gaps:
                gap_df_ja = pd.DataFrame(gaps)
                _render_table_card(
                    gap_df_ja,
                    columns=["skill", "your_level", "job_requires", "gap"],
                    headers=["Skill", "You", "Needed", "Gap"],
                    numeric_cols=["your_level", "job_requires", "gap"],
                )
            else:
                st.success("No significant gaps detected.")

        # Key responsibilities + required skills
        resp_col, skill_col = st.columns(2, gap="large")
        with resp_col:
            resps = ja.get("key_responsibilities", [])
            if resps:
                st.markdown("**Key responsibilities**")
                for r in resps[:4]:
                    st.markdown(f"- {r}")
        with skill_col:
            req_skills = ja.get("required_skills_raw", [])
            if req_skills:
                st.markdown("**Required skills (from posting)**")
                pills = " ".join([f'<span class="status-pill status-warn">{s}</span>' for s in req_skills[:8]])
                st.markdown(pills, unsafe_allow_html=True)


st.markdown(
    '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;'
    'color:rgba(0,0,0,0.35);margin:16px 0 10px 2px">Validate · Pressure-test your decision</div>',
    unsafe_allow_html=True,
)
# ============================================================
# Adversarial Pivot Debate
# ============================================================
with st.container(border=True):
    st.subheader("⚔️ Adversarial Pivot Debate")
    st.caption(
        "Three-agent debate: an Advocate argues FOR the pivot, a Skeptic argues AGAINST, "
        "a Judge weighs both and delivers a probability-style verdict. "
        "Architecturally distinct from the review board — adversarial and sequential, not parallel."
    )

    db_col1, db_col2 = st.columns([2, 1], gap="small")
    with db_col1:
        if st.button("Run adversarial debate", use_container_width=True):
            with st.spinner("Advocate and Skeptic building arguments... Judge deliberating..."):
                _db_key = ""
                try:
                    _db_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass

                _gap_summary_db = (
                    gap_df[gap_df["gap"] > 0].sort_values("gap", ascending=False)["skill"].head(4).tolist()
                )
                _gap_str_db = f"{len(_gap_summary_db)} skill gaps. Top: {', '.join(_gap_summary_db[:3])}"

                # Gather context from previous analyses if available
                _market_sig = None
                if st.session_state.agent_steps:
                    for step in st.session_state.agent_steps:
                        if step.kind == "tool_result" and step.tool_name == "get_market_signal":
                            _market_sig = step.tool_result
                            break

                debate_result = run_pivot_debate(
                    current_role=str(current),
                    target_role=str(target),
                    match_score=match_score_display,
                    gap_summary=_gap_str_db,
                    market_signal=_market_sig,
                    agent_summary=st.session_state.agent_result.executive_summary if st.session_state.agent_result else None,
                    consensus_winner=st.session_state.review_board_consensus.winner_strategy if st.session_state.review_board_consensus else None,
                    cv_profile=st.session_state.cv_profile,
                    model_debate="gpt-4o-mini",
                    model_judge="gpt-4o",
                    prefer_online=_has_openai_secret(),
                    api_key=_db_key or None,
                )
                st.session_state.debate_result = debate_result
            st.rerun()
    with db_col2:
        if st.session_state.debate_result:
            if st.button("Clear", key="clear_debate", type="secondary", use_container_width=True):
                st.session_state.debate_result = None

    dr = st.session_state.debate_result
    if dr:
        advocate: DebateRound = dr["advocate"]
        skeptic: DebateRound = dr["skeptic"]
        verdict: DebateVerdict = dr["verdict"]

        st.divider()

        # Verdict hero
        viability = verdict.pivot_viability_pct
        v_color = "#117A37" if viability >= 70 else ("#A05A00" if viability >= 45 else "#B71C1C")
        v_bg = "#EEF3FB" if viability >= 70 else ("#FFF4E5" if viability >= 45 else "#FDECEA")
        st.markdown(
            f'<div style="background:{v_bg};border-radius:12px;padding:20px 24px;margin-bottom:16px">'
            f'<div style="font-size:11px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:rgba(0,0,0,0.45);margin-bottom:6px">Judge Verdict</div>'
            f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:8px">'
            f'<div style="font-size:40px;font-weight:900;color:{v_color}">{viability}%</div>'
            f'<div style="font-size:18px;font-weight:700;color:rgba(0,0,0,0.8)">{verdict.verdict_label}</div>'
            f'</div>'
            f'<div style="font-size:13px;color:rgba(0,0,0,0.65);line-height:1.6">{verdict.judge_reasoning}</div>'
            f'<div style="margin-top:10px;font-size:13px"><strong>Decisive factor:</strong> {verdict.decisive_factor}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Two-column debate
        adv_col, skp_col = st.columns(2, gap="large")

        with adv_col:
            st.markdown(
                '<div style="border-left:4px solid #117A37;padding-left:12px;margin-bottom:12px">'
                '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#117A37">Advocate — For the pivot</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{advocate.main_argument}*")
            st.markdown("**Evidence:**")
            for ev in advocate.strongest_evidence[:3]:
                st.markdown(f"+ {ev}")
            if advocate.closing_statement:
                st.caption(f"→ {advocate.closing_statement}")
            st.markdown(
                f'<div style="background:#F0FFF4;border-radius:8px;padding:10px 14px;margin-top:10px;font-size:13px">'
                f'<strong>Judge accepted:</strong> {verdict.strongest_pro_argument}</div>',
                unsafe_allow_html=True,
            )

        with skp_col:
            st.markdown(
                '<div style="border-left:4px solid #B71C1C;padding-left:12px;margin-bottom:12px">'
                '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#B71C1C">Skeptic — Against the pivot</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{skeptic.main_argument}*")
            st.markdown("**Evidence:**")
            for ev in skeptic.strongest_evidence[:3]:
                st.markdown(f"− {ev}")
            if skeptic.closing_statement:
                st.caption(f"→ {skeptic.closing_statement}")
            st.markdown(
                f'<div style="background:#FFF5F5;border-radius:8px;padding:10px 14px;margin-top:10px;font-size:13px">'
                f'<strong>Judge accepted:</strong> {verdict.strongest_con_argument}</div>',
                unsafe_allow_html=True,
            )

        # Conditions
        st.divider()
        cond_a, cond_b = st.columns(2, gap="large")
        with cond_a:
            if verdict.conditions_for_success:
                st.markdown("**This pivot succeeds if:**")
                for c in verdict.conditions_for_success:
                    st.markdown(f"✓ {c}")
        with cond_b:
            if verdict.conditions_for_failure:
                st.markdown("**This pivot fails if:**")
                for c in verdict.conditions_for_failure:
                    st.markdown(f"✗ {c}")

        if verdict.recommended_next_action:
            st.markdown(
                f'<div style="background:#EEF3FB;border-radius:8px;padding:14px 18px;margin-top:10px">'
                f'<span style="font-size:11px;font-weight:800;letter-spacing:0.05em;text-transform:uppercase;color:#0A66C2">Next action</span>'
                f'<div style="font-size:14px;font-weight:600;margin-top:4px">{verdict.recommended_next_action}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Source: {verdict.source}")


# ============================================================
# Decision Board (Hero Feature)
# ============================================================
with st.container(border=True):
    st.subheader("Decision Board")
    st.caption(
        "Generate competing pivot strategies, pressure-test with expert personas, aggregate disagreement, and identify what could flip the recommendation."
    )

    def _step_pill(label: str, done: bool) -> str:
        cls = "status-ok" if done else "status-warn"
        icon = "✓" if done else "○"
        return f'<span class="status-pill {cls}">{icon} {label}</span>'

    # Progress tracker at top
    st.markdown(
        _step_pill("Strategies", bool(st.session_state.review_board_strategies))
        + _step_pill("Evaluations", bool(st.session_state.review_board_evaluations))
        + _step_pill("Consensus", bool(st.session_state.review_board_consensus))
        + _step_pill("Judge memo", bool(st.session_state.review_board_judge_memo)),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3, gap="small")

    with b1:
        if st.button("① Generate strategies", use_container_width=True):
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
                st.rerun()

    with b2:
        if st.button("② Expert evaluations", use_container_width=True):
            if not st.session_state.review_board_strategies:
                st.error("Generate strategies first.")
            else:
                with st.spinner("Evaluating strategies across reviewer personas..."):
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
                    st.rerun()

    with b3:
        if st.button("③ Compute consensus", use_container_width=True):
            if not st.session_state.review_board_evaluations:
                st.error("Get expert evals first.")
            else:
                with st.spinner("Computing confidence-adjusted consensus..."):
                    st.session_state.review_board_consensus = compute_consensus(
                        st.session_state.review_board_evaluations
                    )
                    st.session_state.review_board_judge_memo = None
                    st.rerun()

    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)
    strategies = st.session_state.review_board_strategies
    evaluations = st.session_state.review_board_evaluations
    consensus = st.session_state.review_board_consensus
    judge_memo = st.session_state.review_board_judge_memo

    # --------------------------------------------------------
    # Strategies
    # --------------------------------------------------------
    if strategies:
        st.divider()
        st.markdown("**① Competing pivot strategies**")
        st.caption("Five differentiated archetypes generated by the strategy engine.")

        strategies_trace = st.session_state.review_board_trace.get("strategies_bundle", {})
        diversity_warnings = strategies_trace.get("diversity_warnings", [])
        if diversity_warnings:
            with st.expander("Strategy diversity diagnostics", expanded=False):
                for msg in diversity_warnings:
                    st.warning(str(msg))

        strategy_tabs = st.tabs([f"{s.archetype.code}" for s in strategies])

        for tab, strat in zip(strategy_tabs, strategies):
            with tab:
                top_left, top_right = st.columns([2.0, 1.0])

                with top_left:
                    st.markdown(f"### {strat.archetype.name}")
                    st.markdown(str(strat.summary))

                with top_right:
                    st.metric("Risk", str(strat.archetype.risk_level).title())
                    st.metric("Estimated days", int(strat.archetype.estimated_days))

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Speed", f"{float(getattr(strat, 'speed_bias', 5.0)):.1f}/10")
                m2.metric("Risk control", f"{float(getattr(strat, 'risk_bias', 5.0)):.1f}/10")
                m3.metric("Evidence", f"{float(getattr(strat, 'evidence_burden', 5.0)):.1f}/10")
                m4.metric("Market signal", f"{float(getattr(strat, 'market_signal_strength', 5.0)):.1f}/10")
                m5.metric("Gap focus", f"{float(getattr(strat, 'skill_gap_focus', 5.0)):.1f}/10")

                t1, t2 = st.columns(2)
                with t1:
                    st.markdown(f"**Best for:** {str(getattr(strat, 'best_for_profile', '') or '—')}")
                    st.markdown(f"**Evidence strategy:** {str(getattr(strat, 'evidence_strategy', '') or '—')}")
                with t2:
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

                    a1, a2 = st.columns(2)
                    with a1:
                        _render_bullet_list("Key missing skills addressed", getattr(strat, "key_missing_skills", []))
                        _render_bullet_list("Success criteria", getattr(strat, "success_criteria", []))
                    with a2:
                        _render_bullet_list("Transferable anchors", getattr(strat, "transferable_anchors", []))
                        _render_bullet_list("Potential risks", getattr(strat, "potential_risks", []))

    # --------------------------------------------------------
    # Reviewer coverage
    # --------------------------------------------------------
    if evaluations:
        st.divider()
        st.markdown("**② Reviewer coverage and disagreement**")
        st.caption("Five expert personas evaluate each strategy independently.")

        # Summary chips
        n_eval = len(evaluations)
        personas = [ev.reviewer_persona for ev in evaluations]
        strongest_counts: Dict[str, int] = {}
        for ev in evaluations:
            s = ev.strongest_strategy
            strongest_counts[s] = strongest_counts.get(s, 0) + 1
        top_pick = max(strongest_counts, key=strongest_counts.get) if strongest_counts else "?"
        st.markdown(
            f'<span class="status-pill status-ok">{n_eval} reviewers</span>'
            f'<span class="status-pill status-warn">Most favored: {top_pick}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

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

        reviewer_tabs = st.tabs([ev.reviewer_persona for ev in evaluations])
        for tab, ev in zip(reviewer_tabs, evaluations):
            with tab:
                st.caption(ev.overall_recommendation)
                for i, s in enumerate(ev.strategy_scores):
                    if i > 0:
                        st.divider()
                    score_label = f"**{s.strategy_code}**"
                    score_val = f"{float(s.overall_score):.0f} / 100"
                    st.markdown(
                        f'{score_label} <span style="color:rgba(0,0,0,0.45);font-size:13px;margin-left:8px">{score_val}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(str(s.justification))
                    r1, r2 = st.columns(2)
                    with r1:
                        if getattr(s, "best_strength", ""):
                            st.markdown(f"↑ {s.best_strength}")
                        if getattr(s, "success_condition", ""):
                            st.caption(f"Success condition: {s.success_condition}")
                    with r2:
                        if getattr(s, "killer_objection", ""):
                            st.markdown(f"↓ {s.killer_objection}")
                        if getattr(s, "biggest_risk", ""):
                            st.caption(f"Biggest risk: {s.biggest_risk}")

    # --------------------------------------------------------
    # Consensus
    # --------------------------------------------------------
    if consensus:
        st.divider()
        st.markdown("**③ Consensus result**")
        st.caption("Confidence-adjusted aggregation across all reviewers with disagreement penalty.")

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Winner", consensus.winner_strategy)
        c2.metric("Runner-up", consensus.runner_up_strategy)
        c3.metric("Winner score", f"{consensus.winner_score:.0f}")
        c4.metric("Consensus", f"{consensus.consensus_strength:.0f}")
        c5.metric("Robustness", f"{float(getattr(consensus, 'robustness_score', 0.0)):.0f}")
        c6.metric("Controversy", f"{float(getattr(consensus, 'controversy_score', 0.0)):.0f}")
        c7.metric("Fragile", "Yes" if bool(getattr(consensus, "fragile_winner", False)) else "No")

        ranking_df = pd.DataFrame(consensus.strategy_rankings, columns=["strategy_code", "confidence_adjusted_score"])
        _render_table_card(
            ranking_df,
            columns=["strategy_code", "confidence_adjusted_score"],
            headers=["Strategy", "Confidence-Adjusted Score"],
            numeric_cols=["confidence_adjusted_score"],
        )

        st.markdown("**Why this recommendation wins**")
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

        if st.button("④ Generate judge memo", use_container_width=True):
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

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------
    judge_memo = st.session_state.review_board_judge_memo
    if judge_memo:
        st.divider()        
        st.markdown("**④ Final judge recommendation**")

        jv_cls = "status-ok" if judge_memo.verdict == "Highly Feasible" else ("status-challenge" if judge_memo.verdict == "Challenging" else "status-warn")
        st.markdown(
            f'<span class="status-pill {jv_cls}">{judge_memo.verdict}</span>'
            f'<span class="status-pill status-ok">{judge_memo.recommended_strategy}</span>'
            f'<span class="status-pill status-ok">{judge_memo.success_timeline}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

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
            nt_tabs = st.tabs([f"Target {i+1}" for i in range(min(4, len(networking_targets)))])
            for tab, row in zip(nt_tabs, networking_targets[:4]):
                with tab:
                    st.markdown(f"**Target:** {row.get('target', '—')}")
                    st.markdown(f"**Why:** {row.get('why', '—')}")
                    st.markdown(f"**Question to ask:** {row.get('ask', '—')}")

    # --------------------------------------------------------
    # Counterfactual
    # --------------------------------------------------------
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

    # ── Aggregation Documentation (always visible when consensus exists) ──────
    if consensus:
        st.divider()
        with st.expander("⚙️ How the aggregation works — formula & conflict handling", expanded=False):
            st.markdown(
                "This section documents the Python aggregation layer that processes raw LLM scores "
                "into the final recommendation. LLM outputs are **never used raw** — they are "
                "always passed through this deterministic pipeline."
            )

            col_formula, col_why = st.columns([3, 2], gap="large")

            with col_formula:
                st.markdown("**The confidence-adjusted score formula**")
                st.code(
                    "# For each strategy:\n"
                    "weighted_mean = Σ(score_i × weight_i) / Σ(weight_i)\n"
                    "std           = standard deviation across reviewers\n"
                    "spread        = max_score - min_score\n\n"
                    "penalty       = min(16.0,  std × 0.9  +  spread × 0.12)\n"
                    "adj_score     = max(0.0,  weighted_mean - penalty)\n\n"
                    "# Robustness (stability under worst-case disagreement):\n"
                    "robustness    = weighted_mean - std × 1.8\n\n"
                    "# Fragile winner flag:\n"
                    "fragile       = (winner - runner_up < 4.0) OR (winner_std > 4.0)",
                    language="python",
                )

            with col_why:
                st.markdown("**Why this formula?**")
                st.markdown(
                    "- **Weighted mean** rewards reviewers with higher expertise weight — not all reviewers count equally\n"
                    "- **Penalty for std** captures *spread of opinion* — if reviewers disagree, the score should be discounted\n"
                    "- **Penalty for spread** catches bimodal disagreement (one champion, one critic) even when std is moderate\n"
                    "- **Cap at 16** prevents extreme outliers from zeroing out an otherwise strong strategy\n"
                    "- **Robustness** is a conservative lower bound — useful when the winner margin is narrow"
                )

            # Live numbers from current run
            if hasattr(consensus, "strategy_diagnostics") and consensus.strategy_diagnostics:
                st.markdown("**Live numbers from this run**")
                diag_data = []
                for d in consensus.strategy_diagnostics:
                    penalty = d.get("disagreement_penalty", 0)
                    raw = d.get("raw_mean_score", 0)
                    adj = d.get("confidence_adjusted_score", 0)
                    std = d.get("std_dev", 0)
                    rob = d.get("robustness_score", 0)
                    is_winner = d["strategy"] == consensus.winner_strategy
                    diag_data.append({
                        "strategy": ("★ " if is_winner else "  ") + d["strategy"],
                        "raw_mean": round(raw, 1),
                        "std": round(std, 2),
                        "penalty": round(penalty, 2),
                        "adj_score": round(adj, 1),
                        "robustness": round(rob, 1),
                    })
                diag_df = pd.DataFrame(diag_data)
                _render_table_card(
                    diag_df,
                    columns=["strategy", "raw_mean", "std", "penalty", "adj_score", "robustness"],
                    headers=["Strategy", "Raw Mean", "Std Dev", "Penalty", "Adj. Score", "Robustness"],
                    numeric_cols=["raw_mean", "std", "penalty", "adj_score", "robustness"],
                )
                st.caption(
                    f"★ Winner: {consensus.winner_strategy} · "
                    f"Fragile: {'Yes — margin < 4 pts or high variance' if consensus.fragile_winner else 'No'} · "
                    f"Controversy: {consensus.controversy_score:.0f}/100"
                )

            st.markdown("**Conflict detection & investigation**")
            st.markdown(
                "A strategy is flagged for conflict investigation when `std ≥ 2.5` **or** `spread ≥ 10`. "
                "The agent's `investigate_disagreement` tool then surfaces *which* reviewers disagree, "
                "*on which dimension* (role fit, market, risk, narrative), and *what evidence would close the gap*. "
                "This makes conflict-handling a transparent, traceable operation — not a silent score reduction."
            )
            if consensus.major_disagreements:
                for d in consensus.major_disagreements[:3]:
                    sev_cls = "status-challenge" if float(d.get("std_dev", 0)) >= 3.5 else "status-warn"
                    st.markdown(
                        f'<span class="status-pill {sev_cls}">{d["strategy"]}</span>'
                        f'<span style="font-size:13px;color:rgba(0,0,0,0.65)"> '
                        f'Spread {d["spread"]:.0f} pts · {d["strongest_advocate"]} (advocate) vs {d["strongest_critic"]} (critic)</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No major disagreements detected — reviewers are broadly aligned.")


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


# ============================================================
# Download Full Report
# ============================================================
with st.container(border=True):
    _rpt_sections = []
    _rpt_sections.append(f"# Career Pivot Report\n\n**{current} → {target}**\n")
    _rpt_sections.append(f"## Overview\n- Match Score: {match_score_display:.0f}/100\n- Confidence: {conf['confidence_score']:.0f}/100\n- Skill Gaps: {_n_gaps}\n- Pivot Readiness: {_readiness}/100 ({_r_label})\n")

    if st.session_state.salary_result:
        _sr2 = st.session_state.salary_result
        _rpt_sections.append(
            f"## Salary Impact\n"
            f"- Current median: ${_sr2['current_median']:,.0f}\n"
            f"- Target entry: ${_sr2['target_entry_median']:,.0f} ({_sr2['entry_delta_pct']:+.1f}%)\n"
            f"- Target senior: ${_sr2['target_senior_median']:,.0f} ({_sr2['ceiling_delta_pct']:+.1f}%)\n"
            f"- Break-even: {_sr2['months_to_breakeven']} months\n\n"
            + "\n".join(f"- {i}" for i in _sr2.get("insights", []))
        )

    if not gap_df.empty:
        _top_t = (gap_df.assign(ov=lambda d: np.minimum(d["current_importance"], d["target_importance"]))
                  .sort_values("ov", ascending=False).head(5)["skill"].tolist())
        _top_m = (gap_df[gap_df["gap"] > 0].sort_values(["gap","target_importance"],ascending=False)
                  .head(5)["skill"].tolist())
        _rpt_sections.append(f"## Skill Profile\n**Transferable:** {', '.join(_top_t)}\n**To develop:** {', '.join(_top_m)}\n")

    if st.session_state.learning_plan_md:
        _rpt_sections.append(f"## AI Learning Plan\n{st.session_state.learning_plan_md}\n")

    if st.session_state.pivot_narrative:
        _pn2 = st.session_state.pivot_narrative
        _rpt_sections.append(f"## Cover Letter\n{_pn2.get('cover_letter','')}\n")
        _rpt_sections.append(f"## Elevator Pitch\n{_pn2.get('elevator_pitch','')}\n")

    if st.session_state.debate_result:
        _v2 = st.session_state.debate_result.get("verdict")
        if _v2:
            _rpt_sections.append(
                f"## Adversarial Debate Verdict\n"
                f"- Viability: {_v2.pivot_viability_pct}% — {_v2.verdict_label}\n"
                f"- Decisive factor: {_v2.decisive_factor}\n"
                f"- Recommended action: {_v2.recommended_next_action}\n"
            )

    if st.session_state.agent_result:
        _ag2 = st.session_state.agent_result
        _rpt_sections.append(f"## AI Agent Summary\n{_ag2.executive_summary}\n")

    _full_report = "\n\n---\n\n".join(_rpt_sections)
    _full_report += f"\n\n---\n*Generated by Career Pivot Simulator · {current} → {target}*\n"

    dl_col, info_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="📥 Download Pivot Report",
            data=_full_report,
            file_name=f"pivot_report_{current[:15].replace(' ','_')}_{target[:15].replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with info_col:
        _report_sections_done = sum([
            bool(st.session_state.salary_result),
            bool(st.session_state.learning_plan_md),
            bool(st.session_state.pivot_narrative),
            bool(st.session_state.debate_result),
            bool(st.session_state.agent_result),
        ])
        st.markdown(
            f'<div style="font-size:12px;color:rgba(0,0,0,0.55);padding-top:6px">'
            f'Report includes {_report_sections_done + 2}/7 sections completed. '
            f'Run more analyses above to enrich the report.</div>',
            unsafe_allow_html=True,
        )


st.markdown(
    '<div style="font-size:11px;font-weight:700;letter-spacing:0.10em;text-transform:uppercase;'
    'color:rgba(0,0,0,0.35);margin:16px 0 10px 2px">AI Advisor · Autonomous deep analysis</div>',
    unsafe_allow_html=True,
)
# ============================================================
# Career Intelligence Agent (A3)
# ============================================================
with st.container(border=True):
    st.subheader("🤖 Career Intelligence Agent")
    st.markdown(
        '<div class="li-subtitle">'
        "Autonomous reasoning agent with 8 tools. Unlike the fixed A2 pipeline, this agent "
        "decides which tools to call, in what order, and when it has enough evidence — "
        "including explicit conflict investigation when reviewers disagree."
        "</div>",
        unsafe_allow_html=True,
    )

    # Tabs for context vs action
    agent_tab_run, agent_tab_arch, agent_tab_compare = st.tabs(
        ["Run Agent", "Model Rationale", "A2 vs A3"]
    )

    with agent_tab_arch:
        for role, info in MODEL_RATIONALE.items():
            col_label, col_model = st.columns([3, 1])
            col_label.markdown(f"**{role.replace('_', ' ').title()}**")
            col_model.code(info["model"])
            st.markdown(f"{info['why']}")
            st.caption(f"Alternative considered: {info['alternative_considered']}")
            if "cost_note" in info:
                st.caption(info["cost_note"])
            st.divider()

    with agent_tab_compare:
        c_a, c_b = st.columns(2, gap="large")
        with c_a:
            st.markdown("**A2 — Fixed 5-stage pipeline**")
            st.code(
                "Stage 1: Strategy Generation\n"
                "    ↓ (always)\n"
                "Stage 2: Persona Evaluation × 5\n"
                "    ↓ (always)\n"
                "Stage 3: Structured Parsing\n"
                "    ↓ (always)\n"
                "Stage 4: Aggregation + Penalty\n"
                "    ↓ (always)\n"
                "Stage 5: Final Synthesis",
                language=None,
            )
            st.caption("Every run executes all stages. Disagreements are penalised silently.")
        with c_b:
            st.markdown("**A3 — Agentic loop**")
            st.code(
                "Agent sees: current_role → target_role [+ personal CV]\n"
                "    ↓\n"
                "→ get_occupation_similarity\n"
                "→ analyze_skill_gap         (personal if CV loaded)\n"
                "→ get_market_signal         (NEW: job demand + hot skills)\n"
                "→ find_stepping_stone_route (if low sim)\n"
                "→ retrieve_role_evidence    (optional)\n"
                "→ run_strategy_evaluation\n"
                "→ investigate_disagreement  (if conflict)\n"
                "→ simulate_skill_investment (if gaps high)\n"
                "→ finalize_recommendation",
                language=None,
            )
            st.caption("9 tools. Tool selection by LLM each iteration. Personalised when CV is loaded.")

    with agent_tab_run:
        # ── Action area: button + secondary link ──────────────
        run_agent_btn = st.button(
            "🚀 Run Career Intelligence Agent",
            disabled=st.session_state.agent_running,
        )
        if st.session_state.agent_result or st.session_state.agent_steps:
            if st.button("Clear results", key="clear_agent", type="secondary"):
                st.session_state.agent_result = None
                st.session_state.agent_steps = []
                st.rerun()

        # Run the agent
        if run_agent_btn and current != target:
            st.session_state.agent_running = True
            st.session_state.agent_result = None
            st.session_state.agent_steps = []

            progress_bar = st.progress(0, text="Agent starting...")
            collected_steps: List[AgentStep] = []
            agent_result_holder: List[AgentResult] = []

            # Build CV context string for the agent
            _agent_cv_context: Optional[str] = None
            if st.session_state.cv_profile:
                p = st.session_state.cv_profile
                top = ", ".join(p.get("top_skills", [])[:8])
                _agent_cv_context = (
                    f"Role: {p.get('extracted_role', 'Unknown')}. "
                    f"Experience: {p.get('years_experience', 0):.0f} years. "
                    f"Education: {p.get('education_level', 'Unknown')}. "
                    f"Top skills from CV: {top}. "
                    f"Skills mapped to O*NET: {p.get('skills_mapped_count', 0)}."
                )

            gen = run_career_agent(
                current_role=str(current),
                target_role=str(target),
                matrix=mat,
                coords=art.coords,
                model="gpt-4o",
                max_iterations=10,
                prefer_online=True,
                cv_context=_agent_cv_context,
            )

            step_count = 0
            max_expected = 20

            try:
                while True:
                    step = next(gen)
                    collected_steps.append(step)
                    step_count += 1
                    progress_bar.progress(
                        min(step_count / max_expected, 0.95),
                        text=f"Step {step_count}: {step.tool_name or step.kind}",
                    )
            except StopIteration as e:
                agent_result_holder.append(e.value)
            except Exception as exc:
                st.error(f"Agent error: {exc}")

            progress_bar.progress(1.0, text="Done.")
            st.session_state.agent_steps = collected_steps
            if agent_result_holder:
                st.session_state.agent_result = agent_result_holder[0]
            st.session_state.agent_running = False
            st.rerun()

        # Final recommendation — shown prominently at top
        agent_result: Optional[AgentResult] = st.session_state.agent_result
        if agent_result:
            st.divider()

            # Verdict badge row
            v = agent_result.verdict
            v_cls = "status-ok" if v == "Highly Feasible" else ("status-challenge" if v == "Challenging" else "status-warn")
            cl_cls = "status-ok" if agent_result.confidence_level == "High" else ("status-warn" if agent_result.confidence_level == "Medium" else "status-challenge")
            st.markdown(
                f'<span class="status-pill {v_cls}">{v}</span>'
                f'<span class="status-pill status-ok">Strategy: {agent_result.recommended_strategy}</span>'
                f'<span class="status-pill {cl_cls}">Confidence: {agent_result.confidence_level}</span>',
                unsafe_allow_html=True,
            )

            # Hero summary box
            st.markdown(
                f'<div class="agent-verdict-hero">'
                f'<div class="agent-verdict-title">Executive Summary</div>'
                f'<div class="agent-verdict-summary">{agent_result.executive_summary}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            fr1, fr2, fr3 = st.columns(3, gap="large")
            with fr1:
                _render_bullet_list("Key Insights", agent_result.key_insights)
            with fr2:
                _render_bullet_list("Critical Risks", agent_result.critical_risks)
            with fr3:
                _render_bullet_list("First 30 Days", agent_result.first_30_day_actions)

            st.caption(
                f"Model: gpt-4o · "
                f"{agent_result.iterations_used} iterations · "
                f"{len(agent_result.tools_called)} tool calls · "
                f"Source: {agent_result.source}"
            )

        # Reasoning trace
        agent_steps: List[AgentStep] = st.session_state.agent_steps or []
        if agent_steps:
            st.divider()
            st.markdown("**Agent reasoning trace**")
            st.caption("Every tool call and result — fully transparent.")

            tool_steps = [s for s in agent_steps if s.kind in ("tool_call", "tool_result", "thinking", "error")]

            TOOL_ICONS = {
                "get_occupation_similarity": "🔍",
                "analyze_skill_gap": "📊",
                "find_stepping_stone_route": "🗺️",
                "retrieve_role_evidence": "📋",
                "run_strategy_evaluation": "⚖️",
                "investigate_disagreement": "🔬",
                "simulate_skill_investment": "🧪",
                "get_market_signal": "📈",
                "finalize_recommendation": "✅",
            }

            for s in tool_steps:
                if s.kind == "thinking" and s.thinking_text:
                    st.markdown(
                        f'<div class="thinking-block">💭 {s.thinking_text}</div>',
                        unsafe_allow_html=True,
                    )

                elif s.kind == "error":
                    st.error(s.thinking_text or "Unknown error")

                elif s.kind == "tool_result" and s.tool_result:
                    tool_name = s.tool_name or ""
                    result_data = s.tool_result
                    icon = TOOL_ICONS.get(tool_name, "🔧")
                    timer = f'<span class="tool-timer">⏱ {s.elapsed_ms:.0f} ms</span>' if s.elapsed_ms else ""

                    st.markdown(
                        f'<div class="tool-card-header">'
                        f'<span class="tool-badge">{icon} {tool_name}</span>'
                        f'{timer}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if "error" in result_data:
                        st.error(f"Tool error: {result_data['error']}")
                    else:
                        if tool_name == "get_occupation_similarity":
                            cs = result_data.get("cosine_similarity_score", 0)
                            hy = result_data.get("hybrid_score", 0)
                            m1, m2 = st.columns(2)
                            m1.metric("Cosine similarity", f"{cs:.0f} / 100")
                            m2.metric("Hybrid score", f"{hy:.0f} / 100")
                            st.caption(result_data.get("interpretation", ""))

                        elif tool_name == "analyze_skill_gap":
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Missing skills", result_data.get("total_missing_skills", 0))
                            m2.metric("High-signal gaps", result_data.get("high_signal_missing_count", 0))
                            m3.metric("Avg gap", f"{result_data.get('average_gap_magnitude', 0):.2f}")
                            st.caption(result_data.get("gap_summary", ""))

                        elif tool_name == "find_stepping_stone_route":
                            if result_data.get("reachable") and result_data.get("path"):
                                st.success(" → ".join(result_data["path"]))
                            else:
                                st.warning(result_data.get("recommendation", "No route found."))

                        elif tool_name == "retrieve_role_evidence":
                            tasks = result_data.get("tasks", [])
                            st.caption(f"{result_data.get('evidence_count', 0)} evidence items · {result_data.get('target_role', '')}")
                            if tasks:
                                for t in tasks[:3]:
                                    st.markdown(f"- {t}")

                        elif tool_name == "run_strategy_evaluation":
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Winner", result_data.get("winner_strategy", "?"), f"{result_data.get('winner_score', 0):.0f}/100")
                            m2.metric("Runner-up", result_data.get("runner_up_strategy", "?"), f"{result_data.get('runner_up_score', 0):.0f}/100")
                            m3.metric("Controversy", f"{result_data.get('controversy_score', 0):.0f}/100")
                            for d in result_data.get("major_disagreements", []):
                                st.warning(f"⚠️ {d['strategy']}: {d['strongest_advocate']} vs {d['strongest_critic']} (range {d['score_range']})")

                        elif tool_name == "investigate_disagreement":
                            ad = result_data.get("strongest_advocate", {})
                            cr = result_data.get("strongest_critic", {})
                            ic1, ic2 = st.columns(2)
                            with ic1:
                                st.markdown(f"✅ **{ad.get('reviewer', '?')}** — {ad.get('score', 0):.0f}/100")
                                st.caption((ad.get("key_reason") or "")[:150])
                            with ic2:
                                st.markdown(f"❌ **{cr.get('reviewer', '?')}** — {cr.get('score', 0):.0f}/100")
                                st.caption((cr.get("killer_objection") or "")[:150])
                            for cond in result_data.get("resolution_conditions", []):
                                st.markdown(f"→ {cond}")
                            st.caption(result_data.get("impact_on_recommendation", ""))

                        elif tool_name == "simulate_skill_investment":
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Before", f"{result_data.get('similarity_before', 0):.0f}/100")
                            m2.metric("After", f"{result_data.get('similarity_after', 0):.0f}/100")
                            m3.metric("Improvement", f"+{result_data.get('improvement', 0):.1f}")
                            if result_data.get("reranked_winner"):
                                st.success(f"Reranked winner: {result_data['reranked_winner']} ({result_data.get('reranked_winner_score', 0):.0f}/100)")

                        elif tool_name == "get_market_signal":
                            demand = result_data.get("job_demand", "?")
                            competition = result_data.get("competition_level", "?")
                            outlook = result_data.get("growth_outlook", "?")
                            d_cls = "status-ok" if demand == "High" else ("status-warn" if demand == "Medium" else "status-challenge")
                            o_cls = "status-ok" if outlook == "Growing" else ("status-warn" if outlook == "Stable" else "status-challenge")
                            st.markdown(
                                f'<span class="status-pill {d_cls}">Demand: {demand}</span>'
                                f'<span class="status-pill status-warn">Competition: {competition}</span>'
                                f'<span class="status-pill {o_cls}">{outlook}</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)
                            hot_skills = result_data.get("top_employer_skills", [])
                            if hot_skills:
                                st.markdown(f"**Top employer skills:** {', '.join(hot_skills)}")
                            sr = result_data.get("salary_range_usd", {})
                            if sr and sr.get("low") and sr.get("high"):
                                st.caption(f"Salary range: ${sr['low']:,} – ${sr['high']:,} · Timeline: {result_data.get('typical_hiring_timeline_weeks', '?')} weeks to hire")
                            if result_data.get("pivot_market_fit"):
                                st.caption(result_data["pivot_market_fit"])
                            if result_data.get("source") == "llm_simulated":
                                st.caption("⚠️ LLM-simulated market signal — directional only, not live job board data.")

                        elif tool_name == "finalize_recommendation":
                            st.success(f"Verdict: {result_data.get('verdict', '?')} · Strategy: {result_data.get('recommended_strategy', '?')}")

                    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)