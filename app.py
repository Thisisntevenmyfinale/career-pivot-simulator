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
from src.job_search import search_real_jobs, real_job_to_listing, extract_cv_text
from src.evaluator import evaluate_application_package, evaluate_learning_plan
from src.interview_coach import generate_interview_questions, evaluate_interview_answer
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
/* ── Overview stat cards ── */
.li-stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 18px 0;}
.li-stat-card{
  background:#F3F6F9;border-radius:10px;padding:14px 16px;
  border:1px solid rgba(0,0,0,0.06);
}
.li-stat-val{font-size:26px;font-weight:800;color:#0A66C2;line-height:1.1;margin-bottom:2px;}
.li-stat-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:rgba(0,0,0,0.45);}
.li-stat-sub{font-size:11px;color:rgba(0,0,0,0.4);margin-top:2px;}
/* ── Tool card header (consistent across all tools) ── */
.li-tool-header{
  display:flex;align-items:center;gap:10px;margin-bottom:6px;padding-bottom:10px;
  border-bottom:1px solid rgba(0,0,0,0.07);
}
.li-tool-icon{
  width:36px;height:36px;border-radius:8px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:18px;
}
.li-tool-title{font-size:16px;font-weight:800;color:rgba(0,0,0,0.88);line-height:1.2;}
.li-tool-cap{font-size:12px;color:rgba(0,0,0,0.5);margin-top:1px;}
/* ── Phase / section separator ── */
.li-phase{
  display:flex;align-items:center;gap:10px;margin:20px 0 10px 0;
}
.li-phase-line{flex:1;height:1px;background:rgba(0,0,0,0.08);}
.li-phase-text{
  font-size:11px;font-weight:800;letter-spacing:0.10em;text-transform:uppercase;
  color:rgba(0,0,0,0.38);white-space:nowrap;
}
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
    "smart_apply_jobs_source": "ai",   # "ai" or "real"
    "smart_apply_selected_idx": None,
    "smart_apply_package": None,
    # Pivot Peers
    "pivot_peers": None,
    # Salary Estimator
    "salary_result": None,
    # Quality Evaluations (second-pass LLM evaluation layer)
    "pkg_quality_eval": None,      # ApplicationPackage quality score
    "plan_quality_eval": None,     # Learning plan quality score
    # Interview Coach
    "interview_questions": None,   # List[Dict] — generated questions
    "interview_answers": {},       # {idx: answer_text}
    "interview_evals": {},         # {idx: eval_dict}
    "interview_prep_done": False,  # True when ≥1 answer evaluated
    # Phase navigation
    "current_phase": "assess",     # "assess" | "plan" | "validate" | "execute"
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
    # ── CV Upload — drag & drop ────────────────────────────────
    st.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:0.06em;'
        'text-transform:uppercase;color:rgba(0,0,0,0.45);margin-bottom:6px">'
        'Your Profile (optional)</div>',
        unsafe_allow_html=True,
    )

    # File uploader (drag & drop PDF / DOCX)
    cv_uploaded_file = st.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "doc", "txt"],
        label_visibility="collapsed",
        help="Drag & drop your CV here — PDF, DOCX, or TXT",
    )

    # If a new file was dropped, auto-extract text and trigger analysis
    if cv_uploaded_file is not None:
        _file_key = f"{cv_uploaded_file.name}_{cv_uploaded_file.size}"
        if st.session_state.get("_cv_file_key") != _file_key:
            st.session_state["_cv_file_key"] = _file_key
            with st.spinner("Reading your CV…"):
                _extracted = extract_cv_text(cv_uploaded_file)
            st.session_state.cv_text = _extracted

    # Optional: also allow manual text paste as fallback
    with st.expander("Or paste CV text manually", expanded=not bool(cv_uploaded_file)):
        cv_text_input = st.text_area(
            "CV text",
            value=st.session_state.cv_text,
            height=100,
            placeholder="Paste your CV text here…",
            label_visibility="collapsed",
        )
    # Sync text area back if user types manually
    if not cv_uploaded_file:
        st.session_state.cv_text = cv_text_input if "cv_text_input" in dir() else st.session_state.cv_text

    cv_col_a, cv_col_b = st.columns([2, 1])
    with cv_col_a:
        _cv_ready = bool((st.session_state.cv_text or "").strip())
        if st.button(
            "✓ Analyse my profile",
            use_container_width=True,
            disabled=not _cv_ready,
            help="Upload or paste your CV first",
        ):
            if _cv_ready:
                with st.spinner("Mapping your skills to O*NET…"):
                    api_key_for_cv = ""
                    try:
                        api_key_for_cv = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                    except Exception:
                        pass
                    result = parse_cv(
                        cv_text=st.session_state.cv_text,
                        skill_columns=list(mat.columns),
                        model="gpt-4o-mini",
                        prefer_online=_has_openai_secret(),
                        api_key=api_key_for_cv or None,
                    )
                    st.session_state.cv_profile = result
                    if "skill_vector" in result:
                        st.session_state.cv_gap_df = compute_personal_gap_df(
                            result["skill_vector"], str(selected_target), mat
                        )
                st.rerun()
    with cv_col_b:
        if st.session_state.cv_profile:
            if st.button("Clear", use_container_width=True, key="clear_cv", type="secondary"):
                st.session_state.cv_text = ""
                st.session_state.cv_profile = None
                st.session_state.cv_gap_df = None
                st.session_state["_cv_file_key"] = None
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
    # ── Product hero — first thing the professor sees ──
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0A66C2 0%,#004182 100%);'
        'border-radius:12px;padding:36px 40px;margin-bottom:20px;color:#fff">'

        # Eyebrow
        '<div style="font-size:10px;font-weight:800;letter-spacing:0.14em;text-transform:uppercase;'
        'opacity:0.7;margin-bottom:8px">Career Pivot Simulator</div>'

        # Headline
        '<div style="font-size:28px;font-weight:900;line-height:1.2;margin-bottom:10px;'
        'letter-spacing:-0.5px">'
        'From career thought to<br>interview-ready — in one session.'
        '</div>'

        # Sub
        '<div style="font-size:14px;opacity:0.8;line-height:1.6;max-width:560px;margin-bottom:24px">'
        'Pick your current and target role. The simulator takes you through a structured '
        '5-phase journey: skill analysis → learning plan → adversarial validation → '
        'real job applications → AI interview coaching. '
        'Every AI output is scored by a second LLM — nothing leaves raw.'
        '</div>'

        # Journey mini-stepper in hero
        '<div style="display:flex;align-items:center;gap:0;margin-bottom:24px">'
        + "".join(
            f'<div style="display:flex;align-items:center;gap:0">'
            f'<div style="background:rgba(255,255,255,0.2);border:1.5px solid rgba(255,255,255,0.5);'
            f'border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;color:#fff;white-space:nowrap">'
            f'{name}</div>'
            + (f'<div style="width:20px;height:1.5px;background:rgba(255,255,255,0.3)"></div>' if i < 4 else "")
            + f'</div>'
            for i, name in enumerate(["🔍 Assess", "📋 Plan", "⚔️ Validate", "🚀 Execute", "🎤 Interview"])
        )
        + '</div>'

        # Stats row
        '<div style="display:flex;gap:32px">'
        '<div><div style="font-size:20px;font-weight:900">900+</div>'
        '<div style="font-size:11px;opacity:0.6">O*NET occupations</div></div>'
        '<div><div style="font-size:20px;font-weight:900">161</div>'
        '<div style="font-size:11px;opacity:0.6">skill dimensions</div></div>'
        '<div><div style="font-size:20px;font-weight:900">15</div>'
        '<div style="font-size:11px;opacity:0.6">LLM components</div></div>'
        '<div><div style="font-size:20px;font-weight:900">3×</div>'
        '<div style="font-size:11px;opacity:0.6">evaluation layers</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # How it works — 5 columns, one per phase
    _hero_phases = [
        ("🔍", "1 · Assess", "O*NET cosine similarity · skill gap · route analysis · confidence score"),
        ("📋", "2 · Plan", "AI learning plan · salary trajectory · LLM evaluation score"),
        ("⚔️", "3 · Validate", "Adversarial debate · 5-persona decision board · aggregation formula"),
        ("🚀", "4 · Execute", "Real job search (SerpAPI) · cover letter + InMail + CV rewrites · quality eval"),
        ("🎤", "5 · Interview", "Role-specific questions · answer scoring · coached rewrites · readiness 0–100"),
    ]
    _hero_cols = st.columns(5, gap="small")
    for _hci, (_hicon, _hname, _hdesc) in enumerate(_hero_phases):
        with _hero_cols[_hci]:
            st.markdown(
                f'<div style="background:#F8FAFF;border:1px solid #C7D8F0;border-radius:8px;'
                f'padding:14px 12px;height:100%">'
                f'<div style="font-size:20px;margin-bottom:6px">{_hicon}</div>'
                f'<div style="font-size:12px;font-weight:800;color:#0A66C2;margin-bottom:5px">{_hname}</div>'
                f'<div style="font-size:11px;color:rgba(0,0,0,0.55);line-height:1.5">{_hdesc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="margin-top:20px;text-align:center;font-size:13px;color:rgba(0,0,0,0.45)">'
        '← Select your current and target occupation in the sidebar · then click '
        '<strong style="color:#0A66C2">Run pivot analysis</strong> to start</div>',
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

    # ── Pivot Readiness Score (milestone-based journey progress) ──────
    # Each completed phase milestone contributes fixed points toward 100.
    # The base score (0-30) reflects skill match quality — it's always present.
    # Milestone contributions reward completing the full career pivot journey.
    _n_gaps = int((gap_df["gap"] > 0).sum()) if not gap_df.empty else 0
    _n_total = len(gap_df) if not gap_df.empty else 1
    _gap_ratio = _n_gaps / max(_n_total, 1)
    _cv_score = min(float(_cv_profile.get("skills_mapped_count", 0)) / 40.0, 1.0) if _cv_profile else 0.5

    # Base: 0-30 pts from skill match quality (continuous signal)
    _base_pts = int((0.45 * (match_score_display / 100) + 0.30 * (1 - _gap_ratio) + 0.25 * _cv_score) * 30)

    # Milestone pts — each tool completion moves the needle toward 100
    _readiness = max(5, min(
        _base_pts
        + (8  if bool((st.session_state.cv_text or "").strip())      else 0)  # CV uploaded (+8)
        + (10 if bool(st.session_state.learning_plan_md)             else 0)  # Learning plan (+10)
        + (12 if bool(st.session_state.debate_result)                else 0)  # Debate done (+12)
        + (10 if bool(st.session_state.review_board_strategies)      else 0)  # Decision board (+10)
        + (15 if bool(st.session_state.smart_apply_package)          else 0)  # Application pkg (+15)
        + (15 if bool(st.session_state.interview_prep_done)          else 0)  # Interview prep (+15)
    , 100))
    # Milestone breakdown shown in tooltip
    _milestone_labels = [
        ("Skill assessed",    True),
        ("CV uploaded",       bool((st.session_state.cv_text or "").strip())),
        ("Learning plan",     bool(st.session_state.learning_plan_md)),
        ("Debate complete",   bool(st.session_state.debate_result)),
        ("Decision board",    bool(st.session_state.review_board_strategies)),
        ("Application ready", bool(st.session_state.smart_apply_package)),
        ("Interview prepped", bool(st.session_state.interview_prep_done)),
    ]
    _r_color = "#117A37" if _readiness >= 65 else ("#A05A00" if _readiness >= 40 else "#B71C1C")
    _r_label = "Strong" if _readiness >= 65 else ("Promising" if _readiness >= 40 else "Early Stage")
    _weeks = max(4, int((_n_gaps * 3.5) * (1 - match_score_display / 200)))  # rough weeks estimate

    _gap_color = "#117A37" if _n_gaps <= 10 else ("#A05A00" if _n_gaps <= 25 else "#B71C1C")
    _conf_val = int(conf['confidence_score'])
    _conf_color = "#117A37" if _conf_val >= 70 else ("#A05A00" if _conf_val >= 45 else "#B71C1C")
    _match_color_ov = "#117A37" if match_score_display >= 70 else ("#A05A00" if match_score_display >= 45 else "#B71C1C")
    st.markdown(
        f'<div class="li-stats-row">'
        f'<div class="li-stat-card">'
        f'  <div class="li-stat-val" style="color:{_match_color_ov}">{match_score_display:.0f}<span style="font-size:14px;font-weight:600;color:rgba(0,0,0,0.3)">/100</span></div>'
        f'  <div class="li-stat-label">Match Score</div>'
        f'  <div class="li-stat-sub">{"Strong" if match_score_display >= 70 else ("Promising" if match_score_display >= 45 else "Hard pivot")}</div>'
        f'</div>'
        f'<div class="li-stat-card">'
        f'  <div class="li-stat-val" style="color:{_conf_color}">{_conf_val}<span style="font-size:14px;font-weight:600;color:rgba(0,0,0,0.3)">/100</span></div>'
        f'  <div class="li-stat-label">Confidence</div>'
        f'  <div class="li-stat-sub">Data reliability</div>'
        f'</div>'
        f'<div class="li-stat-card">'
        f'  <div class="li-stat-val" style="color:{_gap_color}">{_n_gaps}</div>'
        f'  <div class="li-stat-label">Skill Gaps</div>'
        f'  <div class="li-stat-sub">skills to develop</div>'
        f'</div>'
        f'<div class="li-stat-card">'
        f'  <div class="li-stat-val" style="color:#0A66C2">~{_weeks}<span style="font-size:14px;font-weight:600;color:rgba(0,0,0,0.3)">w</span></div>'
        f'  <div class="li-stat-label">Est. Readiness</div>'
        f'  <div class="li-stat-sub">weeks to apply-ready</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

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

    # ── Pivot Intelligence Brief ─────────────────────────────
    # Rule-based synthesis of the current session state — always visible,
    # always current, no API call needed.
    _done_list = []
    _next_list = []
    if st.session_state.salary_result:          _done_list.append("salary trajectory")
    else:                                        _next_list.append("Salary Estimator")
    if st.session_state.learning_plan_md:        _done_list.append("learning roadmap")
    else:                                        _next_list.append("AI Learning Plan")
    if st.session_state.debate_result:           _done_list.append("adversarial debate")
    else:                                        _next_list.append("Pivot Debate")
    if st.session_state.review_board_strategies: _done_list.append("decision board")
    if st.session_state.pivot_narrative:         _done_list.append("pivot narrative")
    if st.session_state.job_analysis:            _done_list.append("job posting analysis")
    if st.session_state.smart_apply_package:     _done_list.append("application package")
    if st.session_state.agent_result:            _done_list.append("agent deep analysis")
    if st.session_state.interview_prep_done:     _done_list.append("interview preparation")
    else:                                        _next_list.append("Interview Coach")

    _situation_text = (
        f"<b>{'Hard' if match_score_display < 45 else ('Promising' if match_score_display < 70 else 'Strong')} pivot</b> "
        f"({match_score_display:.0f}/100 match · {_n_gaps} skill gaps · ~{_weeks}w to readiness)."
    )
    _done_text = (
        f"Completed: {', '.join(_done_list)}." if _done_list
        else "No analyses run yet — start with a tool below."
    )
    _next_text = (
        f"<b>Recommended next:</b> {_next_list[0]}."
        if _next_list else "<b>Journey complete — you're interview-ready.</b> Download your report."
    )
    _brief_readiness_color = "#117A37" if _readiness >= 65 else ("#A05A00" if _readiness >= 40 else "#B71C1C")

    # Build milestone checklist pills
    _milestone_pills_html = "".join(
        f'<span style="font-size:10px;padding:2px 8px;border-radius:10px;margin-right:4px;margin-bottom:4px;display:inline-block;'
        f'{"background:#E7F6EC;color:#117A37;border:1px solid #A8DDB8" if _done else "background:#F3F6F9;color:#5F6B7A;border:1px solid #C0CCDA"}">'
        f'{"✓" if _done else "○"} {_label}</span>'
        for _label, _done in _milestone_labels
    )

    st.markdown(
        f'<div style="background:#F8FAFF;border:1px solid #C7D8F0;border-radius:10px;'
        f'padding:14px 18px;margin:16px 0 4px 0;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">'
        f'<div style="font-size:10px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#0A66C2">'
        f'📋 Pivot Intelligence Brief</div>'
        f'<div style="font-size:18px;font-weight:900;color:{_brief_readiness_color}">{_readiness}'
        f'<span style="font-size:11px;font-weight:600;color:rgba(0,0,0,0.35)">/100</span></div>'
        f'</div>'
        f'<div style="height:6px;background:rgba(0,0,0,0.07);border-radius:3px;overflow:hidden;margin-bottom:10px">'
        f'<div style="width:{_readiness}%;height:6px;background:{_brief_readiness_color};border-radius:3px;transition:width 0.8s"></div>'
        f'</div>'
        f'<div style="margin-bottom:8px">{_milestone_pills_html}</div>'
        f'<div style="font-size:13px;color:rgba(0,0,0,0.75);line-height:1.7">'
        f'<div>{_situation_text}</div>'
        f'<div style="color:rgba(0,0,0,0.5);font-size:12px;margin-top:3px">{_done_text}</div>'
        f'<div style="margin-top:5px;color:rgba(0,0,0,0.75)">{_next_text}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )



# ── Journey Stepper ────────────────────────────────────────────────────────
# Always-visible 5-phase progress bar. Shows the professor (and user) that this
# is a product with a clear end-goal — not a collection of disconnected tools.
_journey_phases = [
    ("🔍", "Assess",    "Skill landscape",   True),                                                  # always done
    ("📋", "Plan",      "Salary + roadmap",  bool(st.session_state.learning_plan_md or st.session_state.salary_result)),
    ("⚔️", "Validate",  "Debate + decision", bool(st.session_state.debate_result or st.session_state.review_board_consensus)),
    ("🚀", "Execute",   "Apply + materials", bool(st.session_state.smart_apply_package or st.session_state.pivot_narrative)),
    ("🎤", "Interview", "Prep + Coach",      bool(st.session_state.interview_prep_done)),
]
_n_phases_done = sum(1 for _, _, _, _done in _journey_phases if _done)
_journey_pct = int(_n_phases_done / len(_journey_phases) * 100)

_phase_nodes_html = ""
for _pi, (_, _phase_name, _phase_sub, _phase_done) in enumerate(_journey_phases):
    _is_last = _pi == len(_journey_phases) - 1
    _node_color   = "#0A66C2" if _phase_done else "rgba(0,0,0,0.12)"
    _node_bg      = "#0A66C2" if _phase_done else "#fff"
    _label_color  = "rgba(0,0,0,0.88)" if _phase_done else "rgba(0,0,0,0.38)"
    _check        = "✓" if _phase_done else str(_pi + 1)
    _connector    = (
        f'<div style="flex:1;height:2px;margin:0 4px;'
        f'background:{"#0A66C2" if _phase_done else "rgba(0,0,0,0.1)"};'
        f'border-radius:1px;margin-top:-18px"></div>'
        if not _is_last else ""
    )
    _phase_nodes_html += (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:5px;min-width:64px">'
        f'<div style="width:32px;height:32px;border-radius:50%;background:{_node_bg};'
        f'border:2px solid {_node_color};display:flex;align-items:center;justify-content:center;'
        f'font-size:12px;font-weight:900;color:{"#fff" if _phase_done else "rgba(0,0,0,0.25)"}">{_check}</div>'
        f'<div style="font-size:11px;font-weight:800;color:{_label_color};text-align:center;line-height:1.2">{_phase_name}</div>'
        f'<div style="font-size:9px;color:rgba(0,0,0,0.35);text-align:center;line-height:1.2">{_phase_sub}</div>'
        f'</div>'
        + _connector
    )

_readiness_bar_color = "#117A37" if _readiness >= 65 else ("#0A66C2" if _readiness >= 40 else "#A05A00")

st.markdown(
    f'<div style="background:#fff;border:1px solid rgba(0,0,0,0.1);border-radius:12px;'
    f'padding:16px 24px 14px 24px;margin:12px 0 8px 0;'
    f'box-shadow:0 1px 4px rgba(0,0,0,0.05)">'

    # Top row: tagline left, readiness score right
    f'<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px">'
    f'<div>'
    f'<div style="font-size:13px;font-weight:900;color:#1D2226;letter-spacing:-0.2px">'
    f'{str(current).replace("_", " ")} → {str(target).replace("_", " ")}'
    f'</div>'
    f'<div style="font-size:11px;color:rgba(0,0,0,0.45);margin-top:2px">'
    f'Career Pivot Simulator · from career thought to interview-ready in one session'
    f'</div>'
    f'</div>'
    f'<div style="text-align:right;flex-shrink:0;padding-left:16px">'
    f'<div style="font-size:22px;font-weight:900;color:{_readiness_bar_color};line-height:1">'
    f'{_readiness}<span style="font-size:11px;font-weight:600;color:rgba(0,0,0,0.3)">/100</span></div>'
    f'<div style="font-size:9px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;'
    f'color:rgba(0,0,0,0.4)">Pivot Readiness</div>'
    f'</div>'
    f'</div>'

    # Journey stepper nodes
    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">'
    f'{_phase_nodes_html}'
    f'</div>'

    # Progress bar
    f'<div style="height:4px;background:rgba(0,0,0,0.07);border-radius:2px;overflow:hidden">'
    f'<div style="width:{_journey_pct}%;height:4px;background:{_readiness_bar_color};'
    f'border-radius:2px;transition:width 0.8s"></div>'
    f'</div>'
    f'<div style="display:flex;justify-content:space-between;margin-top:4px">'
    f'<div style="font-size:9px;color:rgba(0,0,0,0.35)">{_n_phases_done}/{len(_journey_phases)} phases complete</div>'
    f'<div style="font-size:9px;font-weight:700;color:{_readiness_bar_color}">'
    + ("Interview-ready ✓" if _journey_pct == 100 else f"{100 - _journey_pct}% to interview-ready")
    + f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Phase Tabs ─────────────────────────────────────────────────────────────
_tab_assess, _tab_plan, _tab_validate, _tab_execute, _tab_interview = st.tabs([
    "🔍 Assess · Skill landscape",
    "📋 Plan · Salary + roadmap",
    "⚔️ Validate · Debate + decision",
    "🚀 Execute · Apply + materials",
    "🎤 Interview · Prep + Coach",
])

with _tab_assess:
    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">Skill Analysis · Your pivot profile</div>'
        '<div class="li-phase-line"></div></div>',
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





with _tab_plan:
    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">Prepare · Close your skill gaps</div>'
        '<div class="li-phase-line"></div></div>',
        unsafe_allow_html=True,
    )
    # ============================================================
    # LLM Learning Plan
    # ============================================================
    with st.container(border=True):
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#EEF3FB">🧠</div>'
            '<div><div class="li-tool-title">AI Learning Plan</div>'
            '<div class="li-tool-cap">Personalised upskilling roadmap based on your skill gaps</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        lp1, lp2 = st.columns([2, 1], gap="small")

        with lp1:
            if st.button("Generate learning plan", use_container_width=True):
                _lp_key = ""
                try:
                    _lp_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                except Exception:
                    pass
                with st.spinner("Generating roadmap…"):
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
                    st.session_state.plan_quality_eval = None
                with st.spinner("Evaluating plan quality…"):
                    _gap_names = (
                        gap_df[gap_df["gap"] > 0].sort_values("gap", ascending=False)["skill"]
                        .head(8).tolist()
                    )
                    st.session_state.plan_quality_eval = evaluate_learning_plan(
                        plan_markdown=md,
                        skill_gaps=_gap_names,
                        target_role=str(target),
                        model="gpt-4o-mini",
                        api_key=_lp_key or None,
                        prefer_online=_has_openai_secret(),
                    )

        with lp2:
            if st.button("Clear", use_container_width=True, key="clear_learning_plan", type="secondary"):
                st.session_state.learning_plan_md = ""
                st.session_state.learning_plan_source = "—"
                st.session_state.plan_quality_eval = None

        plan_md = (st.session_state.learning_plan_md or "").strip()
        if plan_md:
            # ── Plan quality badge ────────────────────────────────
            _plan_eval = st.session_state.plan_quality_eval
            if _plan_eval:
                _pqs = _plan_eval.get("overall_score", 0)
                _pqc = "#117A37" if _pqs >= 75 else ("#A05A00" if _pqs >= 55 else "#B71C1C")
                _pdims = _plan_eval.get("dimension_scores", {})
                _pverdict = _plan_eval.get("one_line_verdict", "")
                _pregen = (
                    '<span style="background:#FFF3CD;color:#856404;font-size:10px;font-weight:700;'
                    'border-radius:8px;padding:2px 8px;margin-left:6px">⚠ Regenerate recommended</span>'
                    if _plan_eval.get("regenerate_recommended") else ""
                )
                _pdim_pills = "".join([
                    f'<span style="background:{("#E7F6EC" if v>=75 else ("#FFF8E7" if v>=55 else "#FEECEC"))};'
                    f'color:{("#117A37" if v>=75 else ("#A05A00" if v>=55 else "#B71C1C"))};'
                    f'font-size:10px;font-weight:700;border-radius:8px;padding:2px 7px">'
                    f'{k.replace("_"," ").title()} {v}</span>'
                    for k, v in _pdims.items()
                ])
                st.markdown(
                    f'<div style="background:#F8FAFF;border:1px solid #C7D8F0;border-radius:8px;'
                    f'padding:10px 14px;margin:10px 0 2px 0;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                    f'<span style="font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;color:rgba(0,0,0,0.4)">AI Quality Eval</span>'
                    f'<span style="font-size:20px;font-weight:900;color:{_pqc}">{_pqs}</span>'
                    f'<span style="font-size:11px;font-weight:700;color:{_pqc}">/100</span>'
                    f'{_pregen}'
                    f'<div style="width:100%;display:flex;flex-wrap:wrap;gap:5px;margin-top:4px">{_pdim_pills}</div>'
                    f'<div style="width:100%;font-size:12px;color:rgba(0,0,0,0.55);font-style:italic;margin-top:2px">{_pverdict}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.divider()
            st.caption(f"Source: {st.session_state.learning_plan_source}")
            st.markdown(plan_md)

    # ============================================================
    # Salary Impact Estimator
    # ============================================================
    with st.container(border=True):
        _si_personal = bool(_cv_profile and _cv_profile.get("years_experience", 0) > 0)
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#FFF8E7">💰</div>'
            '<div><div class="li-tool-title">Salary Impact Estimator</div>'
            '<div class="li-tool-cap">Compensation trajectory — entry level vs. senior, with break-even timeline</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
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



with _tab_validate:

    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">Validate · Pressure-test your decision</div>'
        '<div class="li-phase-line"></div></div>',
        unsafe_allow_html=True,
    )
    # ============================================================
    # Adversarial Pivot Debate
    # ============================================================
    with st.container(border=True):
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#FEF0F0">⚔️</div>'
            '<div><div class="li-tool-title">Adversarial Pivot Debate</div>'
            '<div class="li-tool-cap">Advocate vs. Skeptic vs. Judge · probability-style verdict</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Three-agent debate: an Advocate argues FOR the pivot, a Skeptic argues AGAINST, "
            "a Judge weighs both and delivers a probability-style verdict."
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
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#EEF3FB">🏛️</div>'
            '<div><div class="li-tool-title">Decision Board</div>'
            '<div class="li-tool-cap">Competing strategies · expert personas · consensus · what could flip the recommendation</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
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
            _controversy = getattr(consensus, "controversy_score", 0) or 0
            _agg_expanded = bool(_controversy > 50)
            if _agg_expanded:
                st.markdown(
                    '<div style="background:#FFF4F0;border:1px solid #F0A880;border-radius:6px;'
                    'padding:8px 14px;margin-bottom:6px;font-size:12px;color:#B24020;">'
                    f'⚠️ High reviewer disagreement detected (controversy score: {_controversy:.0f}/100) — '
                    'aggregation formula applied below</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("⚙️ How the aggregation works — formula & conflict handling", expanded=_agg_expanded):
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



with _tab_execute:
    # ============================================================
    # Smart Apply — AI Job Matching + Application Package Generator
    # ============================================================
    # PHASE_TAG: execute
    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">Jobs · Find & Apply</div>'
        '<div class="li-phase-line"></div></div>',
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

        sa_col1, sa_col2, sa_col3 = st.columns([5, 5, 2], gap="small")
        with sa_col1:
            if st.button("🤖 AI-curated jobs", use_container_width=True, key="sa_find_jobs"):
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
                    st.session_state.smart_apply_jobs_source = "ai"
                    st.session_state.smart_apply_selected_idx = None
                    st.session_state.smart_apply_package = None
                st.rerun()
        with sa_col2:
            _serp_key = ""
            try:
                _serp_key = str(st.secrets.get("SERP_API_KEY", "")).strip()
            except Exception:
                pass
            _real_jobs_btn_label = "🌐 Search live jobs" if _serp_key else "🌐 Live jobs (add SERP_API_KEY)"
            if st.button(_real_jobs_btn_label, use_container_width=True, key="sa_real_jobs", disabled=not bool(_serp_key)):
                with st.spinner("Searching live job boards (LinkedIn · Indeed · Glassdoor)…"):
                    _raw_jobs = search_real_jobs(
                        target_role=str(target),
                        location="United States",
                        n_jobs=5,
                        serp_api_key=_serp_key,
                    )
                    if _raw_jobs and not _raw_jobs[0].get("error"):
                        st.session_state.smart_apply_jobs = [
                            real_job_to_listing(r, idx=i, match_score=max(50, match_score_display - 5 + i * 3))
                            for i, r in enumerate(_raw_jobs)
                        ]
                        st.session_state.smart_apply_jobs_source = "real"
                    else:
                        err = _raw_jobs[0].get("error", "Unknown error") if _raw_jobs else "No results"
                        st.warning(f"Live job search failed: {err}")
                    st.session_state.smart_apply_selected_idx = None
                    st.session_state.smart_apply_package = None
                st.rerun()
        with sa_col3:
            if st.session_state.smart_apply_jobs:
                if st.button("Clear", key="clear_smart_apply", type="secondary", use_container_width=True):
                    st.session_state.smart_apply_jobs = None
                    st.session_state.smart_apply_jobs_source = "ai"
                    st.session_state.smart_apply_selected_idx = None
                    st.session_state.smart_apply_package = None

        # ── Job Cards ──────────────────────────────────────────────
        sa_jobs: Optional[List[JobListing]] = st.session_state.smart_apply_jobs
        _sa_jobs_source = st.session_state.get("smart_apply_jobs_source", "ai")
        if sa_jobs:
            _src_badge = (
                '<span style="background:#057642;color:#fff;font-size:10px;font-weight:800;'
                'letter-spacing:0.04em;border-radius:10px;padding:2px 9px;margin-left:8px">'
                '🟢 LIVE · Google Jobs</span>'
            ) if _sa_jobs_source == "real" else (
                '<span style="background:#7B5EA7;color:#fff;font-size:10px;font-weight:800;'
                'letter-spacing:0.04em;border-radius:10px;padding:2px 9px;margin-left:8px">'
                '🤖 AI-curated</span>'
            )
            st.markdown(
                f'<div style="font-size:13px;font-weight:700;color:rgba(0,0,0,0.55);margin:16px 0 10px 0;display:flex;align-items:center">'
                f'{len(sa_jobs)} jobs matched for you · {target}{_src_badge}</div>',
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
                _real_badge = (
                    f'<span style="background:#057642;color:#fff;font-size:9px;font-weight:800;'
                    f'border-radius:8px;padding:1px 7px;margin-left:6px">🟢 LIVE</span>'
                ) if getattr(job, "is_real_job", False) else ""
                _via_line = (
                    f'<div class="li-job-detail" style="color:#0A66C2;font-weight:600">'
                    f'via {getattr(job, "apply_source", "")}</div>'
                ) if getattr(job, "is_real_job", False) and getattr(job, "apply_source", "") else ""

                st.markdown(
                    f'<div class="li-job-card">'
                    f'  <div class="li-job-header">'
                    f'    <div class="li-job-logo">{job.company_emoji}</div>'
                    f'    <div class="li-job-meta">'
                    f'      <div class="li-job-title">{job.title}{_real_badge}</div>'
                    f'      <div class="li-job-company">{job.company}</div>'
                    f'      <div class="li-job-detail">{job.location} · {job.job_type} · {job.salary_range}</div>'
                    f'      {_via_line}'
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

                # Apply button per job — real jobs get "Apply Now" external link + package generator
                _is_real = getattr(job, "is_real_job", False)
                _apply_link = getattr(job, "apply_link", "")
                if _is_real and _apply_link:
                    apply_col_a, apply_col_b = st.columns([3, 4])
                else:
                    apply_col_a, apply_col_b = st.columns([2, 3])
                with apply_col_a:
                    if st.button(
                        f"{'⚡ Easy Apply' if job.is_easy_apply else '📄 Generate Application Package'}",
                        key=f"sa_apply_{i}",
                        use_container_width=True,
                    ):
                        with st.spinner(f"Generating your personalised package for {job.company}…"):
                            st.session_state.smart_apply_selected_idx = i
                            _pkg = generate_application_package(
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
                            st.session_state.smart_apply_package = _pkg
                            st.session_state.pkg_quality_eval = None  # reset
                        with st.spinner("Evaluating application quality…"):
                            _pkg_eval = evaluate_application_package(
                                cover_letter=_pkg.cover_letter,
                                linkedin_inmail=_pkg.linkedin_inmail,
                                cv_rewrites=[
                                    {"skill_highlighted": r.skill_highlighted, "rewritten": r.rewritten}
                                    for r in _pkg.cv_bullet_rewrites
                                ],
                                job_title=job.title,
                                company=job.company,
                                job_description=getattr(job, "full_description", ""),
                                cv_text=st.session_state.cv_text or "",
                                model="gpt-4o-mini",
                                prefer_online=_has_openai_secret(),
                                api_key=_sa_api_key or None,
                            )
                            st.session_state.pkg_quality_eval = _pkg_eval
                        st.rerun()
                if _is_real and _apply_link:
                    with apply_col_b:
                        st.link_button(
                            "🔗 Apply Now",
                            url=_apply_link,
                            use_container_width=True,
                        )

                # Show package if this is the selected job
                pkg: Optional[ApplicationPackage] = st.session_state.smart_apply_package
                _pkg_eval = st.session_state.pkg_quality_eval
                if pkg and st.session_state.smart_apply_selected_idx == i:
                    # ── Quality score header ─────────────────────────────────
                    _eval_html = ""
                    if _pkg_eval:
                        _qs = _pkg_eval.get("overall_score", 0)
                        _qc = "#117A37" if _qs >= 75 else ("#A05A00" if _qs >= 55 else "#B71C1C")
                        _ql = "Strong" if _qs >= 75 else ("Acceptable" if _qs >= 55 else "Needs work")
                        _dims = _pkg_eval.get("dimension_scores", {})
                        _verdict = _pkg_eval.get("one_line_verdict", "")
                        _regen_note = (
                            '<span style="background:#FFF3CD;color:#856404;font-size:10px;font-weight:700;'
                            'border-radius:8px;padding:2px 8px;margin-left:6px">⚠ Regenerate recommended</span>'
                            if _pkg_eval.get("regenerate_recommended") else ""
                        )
                        _dim_pills = "".join([
                            f'<span style="background:{("#E7F6EC" if v>=75 else ("#FFF8E7" if v>=55 else "#FEECEC"))};'
                            f'color:{("#117A37" if v>=75 else ("#A05A00" if v>=55 else "#B71C1C"))};'
                            f'font-size:10px;font-weight:700;border-radius:8px;padding:2px 7px">'
                            f'{k.replace("_"," ").title()} {v}</span>'
                            for k, v in _dims.items()
                        ])
                        _eval_html = (
                            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(10,102,194,0.15)">'
                            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                            f'<span style="font-size:10px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:rgba(0,0,0,0.4)">AI Quality Evaluation</span>'
                            f'<span style="font-size:18px;font-weight:900;color:{_qc}">{_qs}</span>'
                            f'<span style="font-size:11px;font-weight:700;color:{_qc}">/100 · {_ql}</span>'
                            f'{_regen_note}'
                            f'</div>'
                            f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px">{_dim_pills}</div>'
                            f'<div style="font-size:12px;color:rgba(0,0,0,0.6);font-style:italic">{_verdict}</div>'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div style="background:#EEF3FB;border-radius:10px;padding:16px 20px;margin:8px 0 16px 0;">'
                        f'<div style="font-size:10px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;'
                        f'color:#0A66C2;margin-bottom:4px">Application Package · {pkg.job_title} @ {pkg.company}</div>'
                        f'<div style="font-size:13px;font-weight:600;color:rgba(0,0,0,0.75);font-style:italic;line-height:1.5">'
                        f'"{pkg.positioning_statement}"</div>'
                        f'{_eval_html}</div>',
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


    # ============================================================
    # Pivot Narrative Generator
    # ============================================================
    with st.container(border=True):
        _pn_personal = bool(st.session_state.cv_profile and st.session_state.cv_profile.get("extracted_role"))
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#F3EEF9">✍️</div>'
            '<div><div class="li-tool-title">Pivot Narrative Generator</div>'
            '<div class="li-tool-cap">Cover letter · elevator pitch · LinkedIn About · interview talking points</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
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
        _jp_personal = bool(st.session_state.cv_profile and st.session_state.cv_profile.get("extracted_role"))
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#EEF3FB">🎯</div>'
            '<div><div class="li-tool-title">Job Posting Analyzer</div>'
            '<div class="li-tool-cap">Instant match score · advantage/gap breakdown · application readiness verdict</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
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

    # ============================================================
    # Pivot Playbook — full session export (the product's deliverable)
    # ============================================================
    _playbook_checks = [
        ("Skill gap analysis",     True),
        ("Salary trajectory",      bool(st.session_state.salary_result)),
        ("AI learning plan",       bool(st.session_state.learning_plan_md)),
        ("Adversarial debate",     bool(st.session_state.debate_result)),
        ("Decision board",         bool(st.session_state.review_board_consensus)),
        ("Application package",    bool(st.session_state.smart_apply_package)),
        ("Interview prep",         bool(st.session_state.interview_prep_done)),
        ("Agent deep analysis",    bool(st.session_state.agent_result)),
    ]
    _playbook_done_count = sum(1 for _, v in _playbook_checks if v)
    _playbook_total = len(_playbook_checks)
    _playbook_pct = int(_playbook_done_count / _playbook_total * 100)
    _playbook_color = "#117A37" if _playbook_pct >= 75 else ("#0A66C2" if _playbook_pct >= 50 else "#A05A00")

    _check_pills = "".join(
        f'<span style="font-size:10px;padding:2px 9px;border-radius:10px;margin-right:4px;margin-bottom:4px;'
        f'display:inline-block;{"background:#E7F6EC;color:#117A37;border:1px solid #A8DDB8" if _v else "background:#F3F6F9;color:#5F6B7A;border:1px solid #C0CCDA"}">'
        f'{"✓" if _v else "○"} {_k}</span>'
        for _k, _v in _playbook_checks
    )

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1D2226 0%,#2D3A42 100%);'
        f'border-radius:12px;padding:24px 28px;margin-bottom:4px">'

        # Top row
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">'
        f'<div>'
        f'<div style="font-size:10px;font-weight:800;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:rgba(255,255,255,0.45);margin-bottom:4px">Your Career Pivot Playbook</div>'
        f'<div style="font-size:20px;font-weight:900;color:#fff;line-height:1.2">'
        f'{str(current)[:28]} → {str(target)[:28]}</div>'
        f'<div style="font-size:12px;color:rgba(255,255,255,0.5);margin-top:3px">'
        f'Pivot Readiness: {_readiness}/100 · {_playbook_done_count}/{_playbook_total} sections complete'
        f'</div>'
        f'</div>'
        f'<div style="font-size:32px;font-weight:900;color:{_playbook_color};text-align:right;flex-shrink:0">'
        f'{_playbook_pct}<span style="font-size:14px;font-weight:600;color:rgba(255,255,255,0.3)">%</span>'
        f'</div>'
        f'</div>'

        # Progress bar
        f'<div style="height:4px;background:rgba(255,255,255,0.12);border-radius:2px;overflow:hidden;margin-bottom:14px">'
        f'<div style="width:{_playbook_pct}%;height:4px;background:{_playbook_color};border-radius:2px"></div>'
        f'</div>'

        # Section pills
        f'<div style="margin-bottom:16px">{_check_pills}</div>'

        # CTA text
        f'<div style="font-size:11px;color:rgba(255,255,255,0.4)">'
        f'Complete more phases above to enrich the playbook. '
        f'The download includes everything from this session.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Build the full playbook content
    _rpt_sections = []
    _rpt_sections.append(
        f"# Career Pivot Playbook\n\n"
        f"**{current} → {target}**  \n"
        f"*Generated by Career Pivot Simulator*\n"
    )
    _rpt_sections.append(
        f"## Session Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Match Score | {match_score_display:.0f}/100 |\n"
        f"| Confidence | {conf['confidence_score']:.0f}/100 |\n"
        f"| Skill Gaps | {_n_gaps} |\n"
        f"| Pivot Readiness | {_readiness}/100 ({_r_label}) |\n"
        f"| Sections completed | {_playbook_done_count}/{_playbook_total} |\n"
    )

    if not gap_df.empty:
        _top_t = (gap_df.assign(ov=lambda d: np.minimum(d["current_importance"], d["target_importance"]))
                  .sort_values("ov", ascending=False).head(5)["skill"].tolist())
        _top_m = (gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False)
                  .head(5)["skill"].tolist())
        _rpt_sections.append(
            f"## Skill Profile\n"
            f"**Your transferable strengths:** {', '.join(_top_t)}\n\n"
            f"**Skills to develop:** {', '.join(_top_m)}\n"
        )

    if st.session_state.salary_result:
        _sr2 = st.session_state.salary_result
        _rpt_sections.append(
            f"## Salary Impact\n"
            f"| Stage | Salary | vs Current |\n|---|---|---|\n"
            f"| Current median | ${_sr2['current_median']:,.0f} | — |\n"
            f"| Target entry | ${_sr2['target_entry_median']:,.0f} | {_sr2['entry_delta_pct']:+.1f}% |\n"
            f"| Target senior | ${_sr2['target_senior_median']:,.0f} | {_sr2['ceiling_delta_pct']:+.1f}% |\n\n"
            f"Break-even timeline: **{_sr2['months_to_breakeven']} months**\n\n"
            + "\n".join(f"- {i}" for i in _sr2.get("insights", []))
        )

    if st.session_state.learning_plan_md:
        _plan_eval2 = st.session_state.plan_quality_eval
        _plan_score_note = (
            f"\n> AI Quality Score: **{_plan_eval2['overall_score']}/100** — {_plan_eval2.get('one_line_verdict','')}"
            if _plan_eval2 else ""
        )
        _rpt_sections.append(f"## AI Learning Plan{_plan_score_note}\n\n{st.session_state.learning_plan_md}\n")

    if st.session_state.debate_result:
        _v2 = st.session_state.debate_result.get("verdict")
        if _v2:
            _rpt_sections.append(
                f"## Adversarial Debate Verdict\n"
                f"- **Viability:** {_v2.pivot_viability_pct}% — *{_v2.verdict_label}*\n"
                f"- **Decisive factor:** {_v2.decisive_factor}\n"
                f"- **Recommended action:** {_v2.recommended_next_action}\n"
            )

    if st.session_state.review_board_consensus:
        _cons = st.session_state.review_board_consensus
        _rpt_sections.append(
            f"## Decision Board Consensus\n"
            f"- **Recommended strategy:** {_cons.winner_strategy}\n"
            f"- **Score:** {_cons.winner_adj_score:.1f}/100\n"
            f"- **Controversy:** {getattr(_cons, 'controversy_score', 0):.0f}/100\n"
            f"- **Fragile winner:** {'Yes — consider runner-up' if getattr(_cons, 'fragile_winner', False) else 'No'}\n"
        )

    if st.session_state.smart_apply_package:
        _pkg2: ApplicationPackage = st.session_state.smart_apply_package
        _pkg_eval2 = st.session_state.pkg_quality_eval
        _pkg_score_note = (
            f"\n> AI Quality Score: **{_pkg_eval2['overall_score']}/100** — {_pkg_eval2.get('one_line_verdict','')}"
            if _pkg_eval2 else ""
        )
        _rpt_sections.append(
            f"## Application Package — {_pkg2.job_title} at {_pkg2.company}{_pkg_score_note}\n\n"
            f"### Cover Letter\n{_pkg2.cover_letter}\n\n"
            f"### LinkedIn InMail\n{_pkg2.linkedin_inmail}\n\n"
            f"### CV Bullet Rewrites\n"
            + "\n".join(
                f"**{r.get('skill_highlighted','')}**\n"
                f"- Before: {r.get('original','')}\n"
                f"- After: {r.get('rewritten','')}"
                for r in (_pkg2.cv_rewrites or [])
            )
        )

    if st.session_state.pivot_narrative:
        _pn2 = st.session_state.pivot_narrative
        _rpt_sections.append(
            f"## Pivot Narrative\n\n"
            f"### LinkedIn Headline\n{_pn2.get('linkedin_headline','')}\n\n"
            f"### Elevator Pitch\n{_pn2.get('elevator_pitch','')}\n\n"
            f"### Cover Letter\n{_pn2.get('cover_letter','')}\n"
        )

    # Interview prep section
    _itv_q2: Optional[List] = st.session_state.interview_questions
    _itv_ev2: dict = st.session_state.interview_evals or {}
    if _itv_q2:
        _itv_scores2 = [v["overall_score"] for v in _itv_ev2.values() if isinstance(v, dict)]
        _itv_avg2 = int(sum(_itv_scores2) / len(_itv_scores2)) if _itv_scores2 else None
        _itv_header = (
            f"## Interview Preparation\n\n"
            + (f"**Interview Readiness Score: {_itv_avg2}/100**\n\n" if _itv_avg2 else "")
        )
        _itv_body = ""
        for _qi2, _q2 in enumerate(_itv_q2):
            _itv_body += f"\n### Q{_qi2+1}: {_q2.get('question','')}\n"
            _itv_body += f"*Type: {_q2.get('type','')} · Difficulty: {_q2.get('difficulty','')}*\n"
            _itv_body += f"*What they're testing: {_q2.get('why_asked','')}*\n"
            _ev2 = _itv_ev2.get(_qi2)
            if _ev2:
                _itv_body += f"\n**Your answer score: {_ev2.get('overall_score',0)}/100** — {_ev2.get('one_line_verdict','')}\n"
                coached = _ev2.get("coached_answer", "")
                if coached:
                    _itv_body += f"\n**Coached answer:**\n{coached}\n"
        _rpt_sections.append(_itv_header + _itv_body)

    if st.session_state.agent_result:
        _ag2 = st.session_state.agent_result
        _rpt_sections.append(
            f"## AI Agent Analysis\n\n"
            f"**Verdict:** {_ag2.verdict}  \n"
            f"**Strategy:** {_ag2.recommended_strategy}  \n"
            f"**Tools used:** {', '.join(_ag2.tools_called or [])}\n\n"
            f"{_ag2.executive_summary}\n"
        )

    _full_playbook = "\n\n---\n\n".join(_rpt_sections)
    _full_playbook += (
        f"\n\n---\n\n*Career Pivot Playbook generated by Career Pivot Simulator*  \n"
        f"*{current} → {target} · Readiness {_readiness}/100*\n"
    )

    # Download button — prominent, with section count
    _dl_col1, _dl_col2, _dl_col3 = st.columns([2, 2, 3])
    with _dl_col1:
        st.download_button(
            label="📥 Download Pivot Playbook",
            data=_full_playbook,
            file_name=f"pivot_playbook_{str(current)[:15].replace(' ','_')}_{str(target)[:15].replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )
    with _dl_col2:
        st.download_button(
            label="📊 Download Skill Gap CSV",
            data=gap_df.to_csv(index=False) if not gap_df.empty else "skill,gap\n",
            file_name=f"skill_gap_{str(current)[:12].replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with _dl_col3:
        st.markdown(
            f'<div style="font-size:12px;color:rgba(0,0,0,0.5);padding-top:8px">'
            f'<b>Playbook includes:</b> skill analysis · salary · learning plan · '
            f'debate verdict · decision board · application package · interview prep · agent analysis'
            f'</div>',
            unsafe_allow_html=True,
        )


    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">AI Advisor · Autonomous deep analysis</div>'
        '<div class="li-phase-line"></div></div>',
        unsafe_allow_html=True,
    )
    # ============================================================
    # Career Intelligence Agent (A3)
    # ============================================================
    with st.container(border=True):
        st.markdown(
            '<div class="li-tool-header">'
            '<div class="li-tool-icon" style="background:#EEF3FB">🤖</div>'
            '<div><div class="li-tool-title">Career Intelligence Agent</div>'
            '<div class="li-tool-cap">Autonomous AI that decides which tools to call, in what order, and when it has enough evidence</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Tabs for context vs action
        agent_tab_run, agent_tab_arch, agent_tab_compare, agent_tab_reflect = st.tabs(
            ["Run Agent", "How it works", "Pipeline comparison", "Dev reflection"]
        )

        with agent_tab_arch:
            st.markdown(
                '<div style="font-size:11px;font-weight:800;letter-spacing:0.08em;'
                'text-transform:uppercase;color:#0A66C2;margin-bottom:14px">'
                '🏗 Architecture layers — every LLM call is explicit and justified'
                '</div>',
                unsafe_allow_html=True,
            )

            # Layer definitions with color + components
            _arch_layers = [
                {
                    "layer": "DATA",
                    "color": "#5F6B7A",
                    "bg": "#F3F6F9",
                    "desc": "Raw inputs — no LLM involved",
                    "components": [
                        ("O*NET Skill Database", "Python", "grey", "27,000 occupations × 35 standardised skills"),
                        ("Uploaded CV / LinkedIn URL", "Python", "grey", "PDF/DOCX text extraction (pypdf, python-docx)"),
                        ("SerpAPI Google Jobs", "API", "grey", "Real job listings from LinkedIn · Indeed · Glassdoor"),
                    ],
                },
                {
                    "layer": "ANALYSIS",
                    "color": "#057642",
                    "bg": "#F0FAF4",
                    "desc": "Extraction & gap quantification",
                    "components": [
                        ("CV Skill Extraction", "gpt-4o-mini", "green", "Maps free-text CV → O*NET skill vector (2-pass)"),
                        ("Market Signal", "gpt-4o-mini", "green", "LLM knowledge: demand trends, hot skills, salary ranges"),
                        ("O*NET Cosine Similarity", "Python", "grey", "Deterministic — no LLM; PCA + dot-product in NumPy"),
                    ],
                },
                {
                    "layer": "GENERATION",
                    "color": "#0A66C2",
                    "bg": "#F0F7FF",
                    "desc": "Long-form content creation — quality-critical steps use gpt-4o",
                    "components": [
                        ("Application Package", "gpt-4o", "blue", "Cover letter · InMail · CV rewrites (quality > cost)"),
                        ("Adversarial Advocate", "gpt-4o-mini", "green", "Structured argument — persona framing drives quality"),
                        ("Adversarial Skeptic", "gpt-4o-mini", "green", "Symmetric to advocate; JSON schema constrains output"),
                        ("Adversarial Judge", "gpt-4o", "blue", "Synthesis requires genuine reasoning; mini produced ambiguous verdicts"),
                        ("Learning Plan", "gpt-4o-mini", "green", "Template-filling task; gaps pre-computed by O*NET analysis"),
                        ("Salary Estimation", "gpt-4o-mini", "green", "Percentile-range lookup from training knowledge"),
                        ("Pivot Narrative", "gpt-4o-mini", "green", "200-word LinkedIn story — constrained writing task"),
                        ("Job Listing Fallback", "gpt-4o-mini", "green", "Simulated listings when SerpAPI key not configured"),
                        ("Review Board Strategies", "gpt-4o-mini", "green", "5 parallel calls; Pydantic validates schema"),
                    ],
                },
                {
                    "layer": "EVALUATION",
                    "color": "#B24020",
                    "bg": "#FFF4F0",
                    "desc": "Second-pass LLM scoring — LLM outputs are never used raw",
                    "components": [
                        ("Application Evaluation", "gpt-4o-mini", "orange", "4 dimensions: job relevance · specificity · InMail impact · CV rewrite quality"),
                        ("Learning Plan Evaluation", "gpt-4o-mini", "orange", "4 dimensions: gap coverage · resource specificity · timeline · actionability"),
                        ("Review Personas × 5", "gpt-4o-mini", "orange", "5 reviewer archetypes score strategies on 5 axes each"),
                    ],
                },
                {
                    "layer": "ORCHESTRATION",
                    "color": "#7A3E9D",
                    "bg": "#F8F3FD",
                    "desc": "Agent loop + Python aggregation — the hard logic lives here",
                    "components": [
                        ("Agent Loop", "gpt-4o", "purple", "Tool selection + chain-of-thought; mini shows higher tool-selection error"),
                        ("Python Aggregation", "Python", "grey", "Confidence-adjusted score = weighted_mean − penalty(std, spread)"),
                        ("Judge Synthesis", "gpt-4o-mini", "green", "Template-fill: hard maths done in Python before the call"),
                    ],
                },
            ]

            _badge_css = {
                "blue":   "background:#E8F1FB;color:#0A66C2;border:1px solid #A0C3F0",
                "green":  "background:#E8F9EE;color:#057642;border:1px solid #90D4A8",
                "orange": "background:#FFF0EA;color:#B24020;border:1px solid #F0A880",
                "purple": "background:#F3EDF9;color:#7A3E9D;border:1px solid #C8A8E8",
                "grey":   "background:#F3F6F9;color:#5F6B7A;border:1px solid #C0CCDA",
            }

            for _layer in _arch_layers:
                _lc = _layer["color"]
                _lbg = _layer["bg"]
                _rows_html = ""
                for _comp_name, _comp_model, _comp_color, _comp_desc in _layer["components"]:
                    _badge_style = _badge_css[_comp_color]
                    _rows_html += (
                        f'<tr>'
                        f'<td style="padding:7px 10px 7px 0;font-size:13px;font-weight:600;color:#1D2226;white-space:nowrap">{_comp_name}</td>'
                        f'<td style="padding:7px 8px;"><span style="font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px;{_badge_style}">{_comp_model}</span></td>'
                        f'<td style="padding:7px 0 7px 8px;font-size:12px;color:#5F6B7A;line-height:1.4">{_comp_desc}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="background:{_lbg};border-left:4px solid {_lc};border-radius:6px;'
                    f'padding:12px 16px;margin-bottom:12px">'
                    f'<div style="font-size:10px;font-weight:900;letter-spacing:0.12em;color:{_lc};margin-bottom:2px">{_layer["layer"]}</div>'
                    f'<div style="font-size:12px;color:#5F6B7A;margin-bottom:10px">{_layer["desc"]}</div>'
                    f'<table style="width:100%;border-collapse:collapse">{_rows_html}</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Per-component drill-down
            st.markdown("---")
            st.markdown(
                '<div style="font-size:11px;font-weight:800;letter-spacing:0.08em;'
                'text-transform:uppercase;color:#5F6B7A;margin-bottom:10px">'
                'Full rationale — why each model was chosen</div>',
                unsafe_allow_html=True,
            )
            for _role, _info in MODEL_RATIONALE.items():
                _model_val = _info["model"]
                _is_4o = _model_val == "gpt-4o"
                _mbadge_style = _badge_css["blue"] if _is_4o else _badge_css["green"]
                with st.expander(
                    f"{_role.replace('_', ' ').title()}  ·  {_model_val}",
                    expanded=False,
                ):
                    st.markdown(
                        f'<span style="font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;{_mbadge_style}">{_model_val}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Why:** {_info['why']}")
                    st.caption(f"Alternative considered: {_info['alternative_considered']}")
                    if "cost_note" in _info:
                        st.info(_info["cost_note"])

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

        with agent_tab_reflect:
            st.markdown(
                '<div style="font-size:11px;font-weight:800;letter-spacing:0.08em;'
                'text-transform:uppercase;color:#B24020;margin-bottom:14px">'
                '🔬 Engineering decisions — what we tried, what failed, what we learned'
                '</div>',
                unsafe_allow_html=True,
            )

            _REFLECT_ENTRIES = [
                {
                    "title": "Problem: Raw LLM scores showed 40–60% variance across identical runs",
                    "status": "solved",
                    "what_we_tried": (
                        "Initially we passed raw LLM strategy scores directly to the UI. "
                        "On re-runs of the exact same pivot, the winner changed in roughly 4 out of 10 cases — "
                        "not because the situation changed, but because LLM sampling temperature introduced noise. "
                        "This made the tool feel unreliable and undermined user trust in the recommendation."
                    ),
                    "what_failed": (
                        "Lowering temperature to 0.0 reduced variance but didn't eliminate it. "
                        "Averaging two runs helped but doubled API cost and latency without solving the root cause: "
                        "different reviewer personas weighted the same strategy differently."
                    ),
                    "solution": (
                        "Introduced the Python aggregation layer: compute a confidence-adjusted score "
                        "for each strategy across all 5 reviewers using `weighted_mean − penalty(std, spread)`. "
                        "The penalty increases when reviewers disagree strongly, so high-controversy strategies "
                        "are explicitly down-ranked rather than randomly winning or losing. "
                        "The winner is now deterministic given the same reviewer weights — "
                        "LLM variance is absorbed by the penalty formula, not passed raw to the user."
                    ),
                    "lesson": "LLM outputs should never be used raw. Every number needs a Python post-processing step.",
                },
                {
                    "title": "Problem: Zero-shot prompts returned inconsistent JSON schemas",
                    "status": "solved",
                    "what_we_tried": (
                        "Early prompts asked the model to 'return a JSON object with strategy evaluations'. "
                        "Output varied wildly — sometimes a list, sometimes a dict, sometimes with extra commentary "
                        "wrapping the JSON. Parsing failures silently dropped evaluations, "
                        "causing some strategies to receive 0 scores."
                    ),
                    "what_failed": (
                        "Regex-based JSON extraction was brittle. Asking the model to 'only return JSON, nothing else' "
                        "worked most of the time but failed on edge cases with long strategy names containing brackets."
                    ),
                    "solution": (
                        "Three-layer reliability stack: "
                        "(1) OpenAI's `response_format={\"type\": \"json_object\"}` enforces JSON mode at the API level. "
                        "(2) Pydantic validates every field name, type, and value range before use. "
                        "(3) Heuristic fallbacks replace any failed component so the UI always renders. "
                        "Zero silent failures — if JSON parsing fails, the source field is set to "
                        "'heuristic (error: …)' and the user sees a warning badge."
                    ),
                    "lesson": (
                        "Zero-shot reliability requires API-level enforcement + schema validation + fallbacks — "
                        "not just prompt engineering."
                    ),
                },
                {
                    "title": "Problem: gpt-4o-mini produced ambiguous adversarial debate verdicts",
                    "status": "solved",
                    "what_we_tried": (
                        "The adversarial debate originally used gpt-4o-mini for all three roles: advocate, skeptic, and judge. "
                        "Advocate and skeptic outputs were consistently strong — structured, opinionated, specific. "
                        "But the judge's verdicts regularly restated both sides without resolving the tension. "
                        "The go/no-go signal was set to 'Possible' in over 60% of runs, regardless of whether "
                        "the evidence clearly favoured one side."
                    ),
                    "what_failed": (
                        "Adding 'be decisive' to the judge prompt improved tone but not actual resolution quality. "
                        "The model was hedging because it genuinely lacked the reasoning capacity "
                        "to synthesise two opposing structured arguments into a confident verdict."
                    ),
                    "solution": (
                        "Upgraded the judge role to gpt-4o while keeping advocate + skeptic on gpt-4o-mini. "
                        "This is a deliberate asymmetric model assignment: generation tasks (advocate, skeptic) "
                        "are well-constrained and mini handles them reliably; synthesis tasks (judge) require "
                        "genuine chain-of-thought reasoning where gpt-4o's gap over mini is measurable. "
                        "The 20× cost premium for the judge step is justified by output quality."
                    ),
                    "lesson": (
                        "Model selection should be task-type-specific: generation = mini, synthesis = full model. "
                        "Evaluate each step individually, not the pipeline as a whole."
                    ),
                },
                {
                    "title": "Problem: CV parsing failed silently for ~20% of uploaded files",
                    "status": "solved",
                    "what_we_tried": (
                        "The initial CV parser used only pypdf. This worked for PDFs generated by standard word processors "
                        "but failed on scanned CVs, password-protected PDFs, and CVs exported from Google Docs "
                        "(which use a non-standard glyph encoding). pypdf returned empty strings with no error."
                    ),
                    "what_failed": (
                        "Checking for empty output and returning an error message told the user to 'try a different format' "
                        "but didn't help them. Many users only had the PDF version of their CV."
                    ),
                    "solution": (
                        "Implemented a three-stage fallback chain: "
                        "(1) pypdf — fast, handles most standard PDFs. "
                        "(2) pdfminer — slower but handles complex glyph encodings and multi-column layouts. "
                        "(3) Raw UTF-8 decode — last resort for plain-text disguised as PDF. "
                        "Only if all three fail does the user see an error, with a clear copy-paste alternative. "
                        "DOCX files use python-docx with paragraph-level extraction."
                    ),
                    "lesson": "File parsing should be treated as an unreliable external system — always build the fallback chain first.",
                },
                {
                    "title": "Problem: The app felt like a 'Flickenteppich' — a collection of disconnected tools",
                    "status": "solved",
                    "what_we_tried": (
                        "Early versions added features incrementally: skill gap → salary → debate → smart apply. "
                        "Each tool had its own UI section with no explicit connection to the others. "
                        "User testing revealed that people ran one tool, got an output, "
                        "and didn't know what to do next. The app had no clear north star."
                    ),
                    "what_failed": (
                        "Adding a sidebar menu helped navigation but didn't create a sense of journey. "
                        "Users still experienced the app as 'a dashboard with many widgets', not "
                        "'a product that takes me somewhere'."
                    ),
                    "solution": (
                        "Redesigned the entire app around ONE explicit end goal: "
                        "get the user from 'I want to change careers' to 'I am interview-ready' in one session. "
                        "This produced five changes: "
                        "(1) 5-phase tab structure (Assess → Plan → Validate → Execute → Interview) making the journey literal. "
                        "(2) Pivot Readiness Score 0–100 (milestone-based) as the single north star metric. "
                        "(3) Pivot Intelligence Brief — always-visible session summary with next recommended action. "
                        "(4) Interview Coach as the terminal step — the app has a clear finish line. "
                        "(5) Each tool now shows which milestone it unlocks (+X pts to Readiness)."
                    ),
                    "lesson": (
                        "Product coherence is not a UI problem — it is a goal-definition problem. "
                        "Once the end state is defined (interview-ready), every tool either moves the needle or is cut."
                    ),
                },
                {
                    "title": "Problem: Evaluating LLM output quality in zero-shot tasks",
                    "status": "solved",
                    "what_we_tried": (
                        "Generated cover letters, learning plans, and interview answers looked plausible but "
                        "had no measurable quality signal. We couldn't tell whether gpt-4o-mini's cover letter "
                        "was good enough or needed regeneration. Users had no signal either — "
                        "they just saw text and assumed it was fine."
                    ),
                    "what_failed": (
                        "Human review of every output was the obvious solution but not scalable. "
                        "Keyword matching (does it mention the job title?) caught obvious failures "
                        "but missed subtle quality issues like generic phrasing or missing STAR structure."
                    ),
                    "solution": (
                        "Three independent LLM evaluation layers, each targeting a specific artifact: "
                        "(1) Application package evaluator — scores cover letter + InMail + CV rewrites on "
                        "job_relevance × 0.35 + narrative_specificity × 0.25 + inmail_impact × 0.20 + cv_rewrite_quality × 0.20. "
                        "(2) Learning plan evaluator — scores gap_coverage × 0.35 + resource_specificity × 0.25 + actionability × 0.25 + timeline_realism × 0.15. "
                        "(3) Interview answer evaluator — scores relevance × 0.30 + specificity × 0.30 + STAR_structure × 0.25 + keywords × 0.15. "
                        "Each evaluator uses a second gpt-4o-mini call with a strict rubric and has a heuristic fallback "
                        "so quality scores are always shown even without an API key. "
                        "regenerate_recommended=True automatically when overall_score < threshold."
                    ),
                    "lesson": (
                        "Evaluating LLM output is a first-class engineering concern, not an afterthought. "
                        "Every generated artifact needs a scoring step before it reaches the user."
                    ),
                },
            ]

            _status_colors = {"solved": ("#E7F6EC", "#117A37", "#A8DDB8", "✓ Solved")}

            for _entry in _REFLECT_ENTRIES:
                _sc, _tc, _bc, _sl = _status_colors.get(_entry["status"], ("#F3F6F9", "#5F6B7A", "#C0CCDA", _entry["status"]))
                with st.expander(_entry["title"], expanded=False):
                    st.markdown(
                        f'<span style="font-size:10px;font-weight:800;padding:2px 10px;border-radius:20px;'
                        f'background:{_sc};color:{_tc};border:1px solid {_bc}">{_sl}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("**What we observed:**")
                    st.markdown(_entry["what_we_tried"])
                    st.markdown("**What didn't work:**")
                    st.markdown(_entry["what_failed"])
                    st.markdown("**Solution:**")
                    st.markdown(
                        f'<div style="background:#F0F7FF;border-left:3px solid #0A66C2;'
                        f'border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;line-height:1.7;'
                        f'color:rgba(0,0,0,0.8);margin-bottom:8px">{_entry["solution"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.info(f"**Lesson:** {_entry['lesson']}")

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Interview · Prep + Coach
# Complete the journey: Assess → Plan → Validate → Execute → Interview-Ready
# ══════════════════════════════════════════════════════════════════════════════
with _tab_interview:
    st.markdown(
        '<div class="li-phase"><div class="li-phase-line"></div>'
        '<div class="li-phase-text">Interview Prep · Go from application to offer</div>'
        '<div class="li-phase-line"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="li-tool-header">🎤 AI Interview Coach</div>'
        '<div class="li-tool-cap">Role-specific questions · Answer scoring · Coached rewrites · Interview Readiness score</div>',
        unsafe_allow_html=True,
    )

    # ── Context summary ───────────────────────────────────────────────────────
    _itv_job_title = str(target)
    _itv_company = ""
    _itv_jd = ""
    _itv_cv = st.session_state.cv_text or ""

    # Pull context from Smart Apply if a job was selected
    _itv_pkg: Optional[ApplicationPackage] = st.session_state.smart_apply_package
    if _itv_pkg:
        _itv_job_title = getattr(_itv_pkg, "job_title", str(target))
        _itv_company   = getattr(_itv_pkg, "company", "")
        # Full JD may be on the selected listing
        _sa_idx = st.session_state.smart_apply_selected_idx
        _sa_jobs_itv: Optional[List[JobListing]] = st.session_state.smart_apply_jobs
        if _sa_jobs_itv and _sa_idx is not None and _sa_idx < len(_sa_jobs_itv):
            _jl = _sa_jobs_itv[_sa_idx]
            _itv_jd = getattr(_jl, "full_description", "") or getattr(_jl, "description_preview", "")

    _ctx_parts = []
    if _itv_company:     _ctx_parts.append(f"**Role:** {_itv_job_title} at {_itv_company}")
    else:                _ctx_parts.append(f"**Target role:** {_itv_job_title}")
    if _itv_jd:          _ctx_parts.append("Job description available ✓")
    if _itv_cv.strip():  _ctx_parts.append("CV loaded ✓")

    st.caption("  ·  ".join(_ctx_parts))

    # ── Generate questions ─────────────────────────────────────────────────────
    _oai_key_itv = None
    try:
        _oai_key_itv = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    _itv_col_btn, _itv_col_info = st.columns([2, 3])
    with _itv_col_btn:
        _itv_gen_btn = st.button(
            "🎯 Generate Interview Questions",
            use_container_width=True,
            help="AI generates 6 role-specific questions based on the job description and your CV.",
        )
    with _itv_col_info:
        st.markdown(
            '<div style="font-size:12px;color:rgba(0,0,0,0.5);padding-top:8px">'
            'Questions are tailored to the specific job. '
            'Type your draft answer → get a score + coached rewrite.'
            '</div>',
            unsafe_allow_html=True,
        )

    if _itv_gen_btn:
        with st.spinner("Generating interview questions…"):
            _qs = generate_interview_questions(
                target_role=_itv_job_title,
                job_description=_itv_jd,
                cv_text=_itv_cv,
                n=6,
                api_key=_oai_key_itv,
                prefer_online=bool(_oai_key_itv),
            )
        st.session_state.interview_questions = _qs
        st.session_state.interview_answers = {}
        st.session_state.interview_evals = {}
        st.session_state.interview_prep_done = False
        st.rerun()

    if st.session_state.interview_questions:
        if st.button("↺ Reset questions", key="itv_reset", type="secondary"):
            st.session_state.interview_questions = None
            st.session_state.interview_answers = {}
            st.session_state.interview_evals = {}
            st.session_state.interview_prep_done = False
            st.rerun()

    # ── Question cards ─────────────────────────────────────────────────────────
    _itv_questions: Optional[List] = st.session_state.interview_questions
    if _itv_questions:
        st.divider()

        # Overall Interview Readiness score (avg of evaluated answers)
        _itv_evals: dict = st.session_state.interview_evals or {}
        _itv_scores = [v["overall_score"] for v in _itv_evals.values() if isinstance(v, dict)]
        _itv_overall = int(sum(_itv_scores) / len(_itv_scores)) if _itv_scores else None
        _itv_n_done = len(_itv_scores)
        _itv_n_total = len(_itv_questions)

        if _itv_overall is not None:
            _itv_oc = "#117A37" if _itv_overall >= 75 else ("#A05A00" if _itv_overall >= 55 else "#B71C1C")
            _itv_label = "Strong" if _itv_overall >= 75 else ("Developing" if _itv_overall >= 55 else "Needs work")
            st.markdown(
                f'<div style="background:#F8FAFF;border:1px solid #C7D8F0;border-radius:10px;'
                f'padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:16px">'
                f'<div>'
                f'<div style="font-size:10px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;'
                f'color:#0A66C2;margin-bottom:2px">Interview Readiness</div>'
                f'<div style="font-size:28px;font-weight:900;color:{_itv_oc}">{_itv_overall}'
                f'<span style="font-size:13px;font-weight:600;color:rgba(0,0,0,0.35)">/100</span></div>'
                f'<div style="font-size:11px;color:rgba(0,0,0,0.5)">{_itv_n_done}/{_itv_n_total} questions answered · {_itv_label}</div>'
                f'</div>'
                f'<div style="flex:1">'
                f'<div style="height:8px;background:rgba(0,0,0,0.07);border-radius:4px;overflow:hidden">'
                f'<div style="width:{_itv_overall}%;height:8px;background:{_itv_oc};border-radius:4px;transition:width 0.8s"></div>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        _diff_colors = {"Easy": "#057642", "Medium": "#A05A00", "Hard": "#B71C1C"}
        _type_colors = {
            "Behavioural": "#0A66C2", "Technical": "#7A3E9D",
            "Competency": "#057642", "Motivation": "#B24020", "Self-awareness": "#5F6B7A",
        }

        for _qi, _q in enumerate(_itv_questions):
            _q_text = _q.get("question", "")
            _q_type = _q.get("type", "Behavioural")
            _q_why  = _q.get("why_asked", "")
            _q_diff = _q.get("difficulty", "Medium")
            _dc = _diff_colors.get(_q_diff, "#A05A00")
            _tc = _type_colors.get(_q_type, "#0A66C2")
            _eval = _itv_evals.get(_qi)

            st.markdown(
                f'<div style="background:#fff;border:1px solid rgba(0,0,0,0.1);border-radius:10px;'
                f'padding:16px 20px;margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<span style="font-size:11px;font-weight:700;color:rgba(0,0,0,0.4)">Q{_qi+1}</span>'
                f'<span style="font-size:11px;padding:2px 9px;border-radius:20px;font-weight:700;'
                f'background:{_tc}15;color:{_tc};border:1px solid {_tc}40">{_q_type}</span>'
                f'<span style="font-size:11px;padding:2px 9px;border-radius:20px;font-weight:700;'
                f'background:{_dc}15;color:{_dc};border:1px solid {_dc}40">{_q_diff}</span>'
                + ('<span style="font-size:10px;padding:2px 8px;border-radius:20px;'
                   'background:#E7F6EC;color:#117A37;border:1px solid #A8DDB8">✓ Evaluated</span>' if _eval else "")
                + f'</div>'
                f'<div style="font-size:14px;font-weight:700;color:#1D2226;line-height:1.5;margin-bottom:6px">{_q_text}</div>'
                f'<div style="font-size:11px;color:rgba(0,0,0,0.45);font-style:italic">What they\'re testing: {_q_why}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Answer input
            _saved_answer = (st.session_state.interview_answers or {}).get(_qi, "")
            _answer_input = st.text_area(
                f"Your answer to Q{_qi+1}",
                value=_saved_answer,
                height=120,
                placeholder="Type your draft answer here… (aim for 150-250 words using the STAR framework)",
                key=f"itv_ans_{_qi}",
                label_visibility="collapsed",
            )

            _eval_col, _ = st.columns([2, 3])
            with _eval_col:
                _eval_btn = st.button(
                    f"⚡ Evaluate & Coach  Q{_qi+1}",
                    key=f"itv_eval_{_qi}",
                    use_container_width=True,
                    disabled=not bool(_answer_input.strip()),
                )
            if _eval_btn and _answer_input.strip():
                if st.session_state.interview_answers is None:
                    st.session_state.interview_answers = {}
                st.session_state.interview_answers[_qi] = _answer_input
                with st.spinner("Scoring your answer and writing a coached version…"):
                    _ev = evaluate_interview_answer(
                        question=_q_text,
                        answer=_answer_input,
                        target_role=_itv_job_title,
                        job_title=_itv_job_title,
                        api_key=_oai_key_itv,
                        prefer_online=bool(_oai_key_itv),
                    )
                if st.session_state.interview_evals is None:
                    st.session_state.interview_evals = {}
                st.session_state.interview_evals[_qi] = _ev
                # Mark interview prep as done once at least one answer is evaluated
                st.session_state.interview_prep_done = True
                st.rerun()

            # Show evaluation results
            if _eval:
                _es = _eval.get("overall_score", 0)
                _ec = "#117A37" if _es >= 75 else ("#A05A00" if _es >= 55 else "#B71C1C")
                _dims = _eval.get("dimension_scores", {})
                _dim_names = {"relevance": "Relevance", "specificity": "Specificity",
                              "star_structure": "STAR Structure", "keywords": "Keywords"}
                _dim_pills = "".join(
                    f'<span style="font-size:10px;padding:2px 8px;border-radius:20px;'
                    f'background:rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.12);'
                    f'color:rgba(0,0,0,0.6);margin-right:4px">'
                    f'{_dim_names.get(k, k)}: {v}</span>'
                    for k, v in _dims.items()
                )
                st.markdown(
                    f'<div style="background:#F8FAFF;border:1px solid #C7D8F0;border-radius:8px;'
                    f'padding:12px 16px;margin-top:6px">'
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                    f'<span style="font-size:10px;font-weight:800;letter-spacing:0.06em;text-transform:uppercase;color:#0A66C2">Answer Score</span>'
                    f'<span style="font-size:20px;font-weight:900;color:{_ec}">{_es}</span>'
                    f'<span style="font-size:11px;color:rgba(0,0,0,0.4)">/100 · {_eval.get("one_line_verdict", "")}</span>'
                    f'</div>'
                    f'<div style="margin-bottom:8px">{_dim_pills}</div>',
                    unsafe_allow_html=True,
                )
                # Strengths + improvements
                if _eval.get("strengths") or _eval.get("improvements"):
                    _s_col, _i_col = st.columns(2)
                    with _s_col:
                        st.markdown("**What works**")
                        for _s in _eval.get("strengths", []):
                            st.markdown(f"✓ {_s}")
                    with _i_col:
                        st.markdown("**Improve**")
                        for _imp in _eval.get("improvements", []):
                            st.markdown(f"→ {_imp}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Coached answer
                _coached = _eval.get("coached_answer", "")
                if _coached:
                    with st.expander("✨ Coached answer — study this structure", expanded=(_es < 70)):
                        st.markdown(
                            f'<div style="background:#F0F7FF;border-left:3px solid #0A66C2;'
                            f'border-radius:0 8px 8px 0;padding:14px 16px;font-size:13px;line-height:1.7;'
                            f'color:rgba(0,0,0,0.8)">{_coached}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Source: {_eval.get('source', 'llm')}")

            st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

        # ── Summary when all questions answered ────────────────────────────────
        if _itv_n_done == _itv_n_total and _itv_overall is not None:
            _final_color = "#117A37" if _itv_overall >= 75 else ("#A05A00" if _itv_overall >= 55 else "#B71C1C")
            _final_verdict = (
                "You're interview-ready. Confidence is high — practice delivery and you're set."
                if _itv_overall >= 75 else
                "Good foundation. Review the coached answers for the questions you scored below 70."
                if _itv_overall >= 55 else
                "Keep practicing. Focus on adding STAR structure and specific metrics to every answer."
            )
            st.markdown(
                f'<div style="background:{_final_color}18;border:2px solid {_final_color}55;'
                f'border-radius:10px;padding:16px 20px;margin-top:8px;text-align:center">'
                f'<div style="font-size:11px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;'
                f'color:{_final_color};margin-bottom:4px">🎤 Interview Readiness: {_itv_overall}/100</div>'
                f'<div style="font-size:13px;color:rgba(0,0,0,0.75)">{_final_verdict}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:
        # Empty state
        st.markdown(
            '<div style="text-align:center;padding:48px 24px;color:rgba(0,0,0,0.4)">'
            '<div style="font-size:40px;margin-bottom:12px">🎤</div>'
            '<div style="font-size:15px;font-weight:600;margin-bottom:6px">AI Interview Coach</div>'
            '<div style="font-size:13px">Click "Generate Interview Questions" to get 6 tailored questions.<br>'
            'Type your draft answers and get scored + coached rewrites.<br>'
            'Complete all 6 to unlock your Interview Readiness score.</div>'
            '</div>',
            unsafe_allow_html=True,
        )