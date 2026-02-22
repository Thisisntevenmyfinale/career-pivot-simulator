# app.py
from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model_logic import (
    load_runtime_artifacts,
    compute_gap_df,
    compute_match_score_hybrid,
    compute_match_score_cosine,
    compute_skill_contributions,
    generate_learning_plan,
    recommend_similar_roles,
    compute_confidence_score,
    find_pivot_path,
    robustness_analysis,
    compute_group_gap_df,
    filter_missing_skills_by_group,
    format_cluster_theme,
    compute_all_targets_robustness,
    compute_effort_metrics,
    pareto_frontier_flags,
    counterfactual_uplift_greedy,
)

st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")

st.title("🧭 Career Pivot Simulator")
st.caption(
    "Decision-support prototype for career pivots: matching + explainability + planning + robustness + what-if sensitivity. "
    "Artifacts are precomputed offline and loaded at runtime (deploy-ready)."
)


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    return load_runtime_artifacts("artifacts")


try:
    art = load_artifacts()
except Exception as e:
    st.error("Missing or invalid runtime artifacts.")
    st.info("Run: `python scripts/preprocess_dummy.py --input data/skills_long.csv --out artifacts`")
    st.exception(e)
    st.stop()

mat = art.matrix
coords = art.coords
occupations = mat.index.astype(str).tolist()

clusters = getattr(art, "clusters", {}) or {}
cluster_themes = getattr(art, "cluster_themes", {}) or {}
skill_taxonomy = getattr(art, "skill_taxonomy", {}) or {}
group_meta = getattr(art, "group_meta", {}) or {}

# -----------------------------
# Session state
# -----------------------------
if "__has_run__" not in st.session_state:
    st.session_state["__has_run__"] = False
if "__target_override__" not in st.session_state:
    st.session_state["__target_override__"] = None


# -----------------------------
# Helpers (runtime-only)
# -----------------------------
@st.cache_data(show_spinner=False)
def _precompute_map_maxdist(coords_df: pd.DataFrame) -> float:
    all_xy = coords_df[["x", "y"]].to_numpy(dtype=float)
    if len(all_xy) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(all_xy)):
        for j in range(i + 1, len(all_xy)):
            d = float(np.linalg.norm(all_xy[i] - all_xy[j]))
            if d > max_dist:
                max_dist = d
    return float(max_dist)


def _map_proximity(coords_df: pd.DataFrame, occ_a: str, occ_b: str, max_dist: float) -> float:
    a_xy = coords_df.loc[coords_df["occupation"] == occ_a, ["x", "y"]].to_numpy(dtype=float)
    b_xy = coords_df.loc[coords_df["occupation"] == occ_b, ["x", "y"]].to_numpy(dtype=float)
    if len(a_xy) == 0 or len(b_xy) == 0:
        return 0.0
    if max_dist <= 0:
        return 1.0
    dist = float(np.linalg.norm(a_xy[0] - b_xy[0]))
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _hybrid_score_from_vectors(
    a: np.ndarray,
    b: np.ndarray,
    map_part: float,
    w_cosine: float,
    w_map: float,
) -> float:
    cos = max(0.0, _cosine(a, b))  # 0..1
    score = 100.0 * (w_cosine * cos + w_map * map_part)
    return float(np.clip(score, 0.0, 100.0))


@st.cache_data(show_spinner=False)
def _what_if_sweep(
    matrix: pd.DataFrame,
    coords_df: pd.DataFrame,
    current_occ: str,
    grid_w_cosine: tuple[float, float, int],
    grid_noise: tuple[float, float, int],
    mc_per_cell: int,
    seed: int,
) -> pd.DataFrame:
    """
    Sweep across weight and noise to estimate mean hybrid score for ALL target roles.
    Output columns: w_cosine, w_map, noise_std, occupation, mean_score
    """
    rng = np.random.default_rng(seed)
    a0 = matrix.loc[current_occ].astype(float).to_numpy()

    w0, w1, wn = grid_w_cosine
    n0, n1, nn = grid_noise

    w_vals = np.linspace(w0, w1, wn)
    n_vals = np.linspace(n0, n1, nn)

    max_dist = _precompute_map_maxdist(coords_df)

    # Fixed map proximity current->other
    map_part = {}
    for occ in matrix.index.astype(str):
        map_part[occ] = _map_proximity(coords_df, current_occ, occ, max_dist)

    records = []
    for w in w_vals:
        w = float(np.clip(w, 0.0, 1.0))
        w_map = float(1.0 - w)
        for noise in n_vals:
            noise = float(max(0.0, noise))
            for occ in matrix.index.astype(str):
                if occ == current_occ:
                    continue
                b0 = matrix.loc[occ].astype(float).to_numpy()

                scores = []
                for _ in range(mc_per_cell):
                    a = np.clip(a0 + rng.normal(0.0, noise, size=a0.shape), 0.0, None)
                    b = np.clip(b0 + rng.normal(0.0, noise, size=b0.shape), 0.0, None)
                    s = _hybrid_score_from_vectors(a, b, map_part[occ], w, w_map)
                    scores.append(s)

                records.append(
                    {
                        "w_cosine": w,
                        "w_map": w_map,
                        "noise_std": noise,
                        "occupation": occ,
                        "mean_score": float(np.mean(scores)),
                    }
                )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def _ranking_stability(
    sweep_df: pd.DataFrame,
    target_occ: str,
    top_k: int = 3,
) -> dict:
    """
    Stability metrics:
    - target_in_topk_rate
    - distribution of target rank across configs
    - frequency of roles in Top-K across configs
    - most common Top-K set + its share
    """
    group_cols = ["w_cosine", "w_map", "noise_std"]
    ranks_frames = []
    topk_lists = []
    target_ranks = []

    for _, g in sweep_df.groupby(group_cols):
        g2 = g.sort_values("mean_score", ascending=False).reset_index(drop=True)
        g2["rank"] = np.arange(1, len(g2) + 1)
        ranks_frames.append(g2)

        topk = tuple(g2.head(top_k)["occupation"].tolist())
        topk_lists.append(topk)

        tr = g2.loc[g2["occupation"] == target_occ, "rank"]
        if len(tr) == 0:
            target_ranks.append(np.nan)
        else:
            target_ranks.append(int(tr.iloc[0]))

    ranks_df = pd.concat(ranks_frames, ignore_index=True) if ranks_frames else pd.DataFrame()
    target_ranks_arr = np.array([r for r in target_ranks if not (isinstance(r, float) and np.isnan(r))], dtype=int)

    in_topk = float(np.mean(target_ranks_arr <= top_k)) if len(target_ranks_arr) else 0.0

    # frequency of roles appearing in Top-K
    freq = Counter()
    for t in topk_lists:
        for occ in t:
            freq[occ] += 1

    freq_df = (
        pd.DataFrame({"occupation": list(freq.keys()), "count": list(freq.values())})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    freq_df["share"] = freq_df["count"] / max(1, len(topk_lists))

    # mode of Top-K sets using Counter (robust)
    topk_mode = None
    topk_mode_share = 0.0
    if topk_lists:
        set_counts = Counter(topk_lists)
        topk_mode, c = set_counts.most_common(1)[0]
        topk_mode = list(topk_mode)
        topk_mode_share = float(c / len(topk_lists))

    return {
        "target_in_topk_rate": in_topk,
        "target_rank_values": target_ranks_arr.tolist(),
        "topk_freq_df": freq_df,
        "topk_mode": topk_mode,
        "topk_mode_share": topk_mode_share,
        "ranks_df": ranks_df,
        "n_configs": int(len(topk_lists)),
    }


# -----------------------------
# Sidebar: controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    st.write("Pick a pivot → run analysis → interpret plan + risks.")
    st.divider()

    with st.expander("1) Choose your pivot", expanded=True):
        current = st.selectbox("Current occupation", options=occupations, index=0)
        default_target_idx = 1 if len(occupations) > 1 else 0
        target = st.selectbox("Target occupation", options=occupations, index=default_target_idx)

        if st.session_state["__target_override__"]:
            target = st.session_state["__target_override__"]

    with st.expander("2) Scoring knobs (optional)", expanded=False):
        st.caption("Default is fine. Only change to explore trade-offs.")
        w_cosine = st.slider("Skill similarity weight", 0.0, 1.0, 0.65, 0.05)
        w_map = st.slider("Map proximity weight", 0.0, 1.0, 0.35, 0.05)
        s = w_cosine + w_map
        if s == 0:
            w_cosine, w_map = 1.0, 0.0
        else:
            w_cosine, w_map = w_cosine / s, w_map / s

    with st.expander("3) Planning & robustness knobs", expanded=False):
        st.subheader("Path planning")
        k_neighbors = st.slider("kNN neighbors", 2, 10, 5, 1)
        max_steps = st.slider("Max steps", 2, 6, 4, 1)

        st.subheader("Robustness")
        n_samples = st.slider("Monte Carlo samples", 50, 600, 200, 50)
        noise_std = st.slider("Noise level (std)", 0.00, 0.20, 0.05, 0.01)

    with st.expander("Advanced visualization", expanded=False):
        show_map = st.toggle("Show PCA map (advanced)", value=False)
        color_by_cluster = st.toggle("Color by clusters", value=bool(clusters))

    st.divider()
    run = st.button("🚀 Run pivot analysis", use_container_width=True)
    if run:
        st.session_state["__has_run__"] = True

    st.divider()
    st.subheader("Dataset snapshot")
    st.metric("Occupations", mat.shape[0])
    st.metric("Skills", mat.shape[1])


# -----------------------------
# Empty-state: onboarding + AI transparency
# -----------------------------
if not st.session_state["__has_run__"]:
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.subheader("What can I do here?")
        st.markdown(
            """
This prototype evaluates a career pivot with **AI-driven decision support**:

1) **Match score** – role similarity in skill space (vector similarity)  
2) **Explainability** – transferable vs missing skills  
3) **Pivot path** – stepping-stone roles via graph shortest-path  
4) **Robustness** – uncertainty via Monte Carlo simulation  
5) **What-if lab** – stability of rankings across parameter/noise variations

👉 Choose Current + Target in the sidebar and click **Run pivot analysis**.
            """
        )

        st.markdown("### Quick experiments")
        st.markdown(
            """
- Near pivot: *Data Analyst → Data Scientist*  
- Hard pivot: *Cybersecurity Analyst → UX Designer*  
- Increase noise to see when the score becomes unstable
            """
        )

    with right:
        st.subheader("Where is AI/ML used?")
        st.markdown(
            """
This uses classic ML techniques (no deep learning required):

- **Feature representation:** occupation = skill-importance vector  
- **Similarity model:** cosine similarity (embedding-style matching)  
- **Dimensionality reduction:** PCA (optional map)  
- **Clustering:** KMeans (career neighborhoods)  
- **Planning:** kNN graph + Dijkstra shortest path  
- **Uncertainty:** Monte Carlo perturbation (stability metrics)  
- **Sensitivity:** what-if sweeps (ranking stability across configs)
            """
        )
        st.info("The value comes from the model — without it, the app has no purpose.")

    st.stop()


# -----------------------------
# Core AI outputs
# -----------------------------
gap_df = compute_gap_df(mat, current, target)
match_score = compute_match_score_hybrid(mat, coords, current, target, w_cosine=w_cosine, w_map=w_map)
cosine_score = compute_match_score_cosine(mat, current, target)
contrib = compute_skill_contributions(gap_df)
plan = generate_learning_plan(gap_df)
conf = compute_confidence_score(mat, art.pca_meta, current, target)

path_out = find_pivot_path(mat, current, target, k_neighbors=k_neighbors, max_steps=max_steps)
rob = robustness_analysis(
    mat,
    coords,
    current,
    target,
    w_cosine=w_cosine,
    w_map=w_map,
    n_samples=n_samples,
    noise_std=noise_std,
    seed=42,
)

group_gap_df = compute_group_gap_df(mat, skill_taxonomy, group_meta, current, target)


# -----------------------------
# Overview
# -----------------------------
st.subheader("Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Match score (hybrid)", f"{match_score:.0f}/100")
m2.metric("Confidence (coverage)", f"{conf['confidence_score']:.0f}/100")
m3.metric("Robust mean", f"{rob['mean']:.1f}")
m4.metric("Robust 95% CI width", f"{(rob['ci95_high'] - rob['ci95_low']):.1f}")

ci_width = float(rob["ci95_high"] - rob["ci95_low"])
if match_score >= 70:
    pivot_label = "Likely smooth pivot"
elif match_score >= 40:
    pivot_label = "Moderate pivot (skill build required)"
else:
    pivot_label = "Hard pivot (expect significant gaps)"

if ci_width <= 8:
    stability_label = "Stable score"
elif ci_width <= 18:
    stability_label = "Moderately sensitive"
else:
    stability_label = "Sensitive / unstable"

st.success(f"**{pivot_label}** • **{stability_label}**")

with st.expander("Under the hood: score breakdown (transparent)", expanded=False):
    max_dist = _precompute_map_maxdist(coords)
    map_part = _map_proximity(coords, current, target, max_dist)  # 0..1

    st.markdown("### Components")
    st.write(f"- Cosine-based score: **{cosine_score:.1f}/100** (vector similarity)")
    st.write(f"- Map proximity (0..1): **{map_part:.2f}** (from PCA coords; weak prior)")
    st.write(f"- Weights: `w_cosine={w_cosine:.2f}`, `w_map={w_map:.2f}`")
    st.code("score = 100 * (w_cosine * max(0, cosine_similarity) + w_map * map_proximity)")

    st.markdown("### Why your score is high/low")
    top_missing = gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False).head(5)
    top_transfer = gap_df.copy()
    top_transfer["overlap"] = np.minimum(top_transfer["current_importance"], top_transfer["target_importance"])
    top_transfer = top_transfer.sort_values("overlap", ascending=False).head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top transferable skills (overlap)**")
        st.dataframe(
            top_transfer[["skill", "current_importance", "target_importance", "overlap"]],
            hide_index=True,
            use_container_width=True,
        )
    with c2:
        st.markdown("**Top missing skills (gaps)**")
        if top_missing.empty:
            st.success("No missing skills in this dataset.")
        else:
            st.dataframe(
                top_missing[["skill", "current_importance", "target_importance", "gap"]],
                hide_index=True,
                use_container_width=True,
            )


# -----------------------------
# Main: Neighborhood + Plan
# -----------------------------
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.subheader("Career neighborhood (actionable)")
    st.caption("Closest roles to your current occupation + one-click stepping-stones.")

    rec_df = recommend_similar_roles(mat, coords, current, top_k=6, w_cosine=w_cosine, w_map=w_map)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown("**Use a recommended role as new target:**")
    for _, row in rec_df.iterrows():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{row['occupation']}** — {row['match_score']:.0f}/100")
        with c2:
            if st.button("Use", key=f"use_{row['occupation']}"):
                st.session_state["__target_override__"] = row["occupation"]
                st.session_state["__has_run__"] = True
                st.rerun()

    if clusters:
        st.divider()
        st.markdown("**Career neighborhood theme (cluster)**")
        cur_cluster = clusters.get(current, None)
        if cur_cluster is None:
            st.info("No cluster found for current role.")
        else:
            st.write(f"Cluster **{cur_cluster}** theme skills: {format_cluster_theme(cur_cluster, cluster_themes)}")

    st.divider()
    st.subheader("Confidence (coverage)")
    st.progress(int(round(conf["confidence_score"])))
    st.caption(
        f"Overlap Jaccard: {conf['signals']['overlap_jaccard']:.2f} • "
        f"Matrix density: {conf['signals']['matrix_density']:.2f} • "
        f"PCA EVR(2D): {conf['signals']['pca_evr_2d']:.2f}"
    )

with right:
    st.subheader("Plan: Pivot Path + Learning Plan")
    st.caption("This is the decision-support core: route + learning phases.")

    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.markdown("### Pivot Path Finder")
        if not path_out.get("reachable"):
            st.warning(path_out.get("notes", "Target not reachable. Try increasing kNN neighbors."))
        else:
            path = path_out["path"]
            step_costs = path_out.get("step_costs", [])

            st.markdown("**Suggested route**")
            for i, p in enumerate(path):
                if i == 0:
                    st.write(f"Start: **{p}**")
                elif i == len(path) - 1:
                    st.write(f"Target: **{p}**")
                else:
                    st.write(f"Step {i}: **{p}**")

            if step_costs:
                df_steps = pd.DataFrame({"from": path[:-1], "to": path[1:], "transition_cost": step_costs})
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Learning plan (3 phases)")
        st.caption("Derived from the largest missing skill gaps.")

        st.markdown("**Foundations**")
        for b in plan["Foundations"]:
            st.write("- " + b)
        st.markdown("**Intermediate**")
        for b in plan["Intermediate"]:
            st.write("- " + b)
        st.markdown("**Advanced**")
        for b in plan["Advanced"]:
            st.write("- " + b)


# -----------------------------
# Tabs (deep dives)
# -----------------------------
st.divider()
tabs = st.tabs(
    ["Decision Brief", "What-If Lab", "Explain", "Skill Groups", "Robustness", "Skill Gap", "Radar", "Export", "Map (advanced)"]
)

# ---- Decision Brief
with tabs[0]:
    st.subheader("Decision Brief (risk-aware, tradeoffs, actionable)")
    st.caption(
        "This panel turns model outputs into decision support: robust recommendations, Pareto tradeoffs, and minimal skill lifts."
    )

    colA, colB, colC = st.columns([1.1, 1.1, 1.0], gap="large")

    with colA:
        risk_options = ["Risk-averse (CVaR)", "Balanced", "Risk-seeking (Mean)"]
        try:
            risk_profile = st.segmented_control("Risk profile", options=risk_options, default=risk_options[1])
        except Exception:
            risk_profile = st.radio("Risk profile", options=risk_options, index=1)

    with colB:
        goal_options = ["Threshold", "Top-K"]
        try:
            goal_mode_ui = st.segmented_control("Counterfactual goal", options=goal_options, default=goal_options[0])
        except Exception:
            goal_mode_ui = st.radio("Counterfactual goal", options=goal_options, index=0)

        goal_mode = "threshold" if goal_mode_ui == "Threshold" else "topk"
        score_threshold = st.slider("Score threshold", 40.0, 90.0, 60.0, 1.0)
        topk_goal = st.slider("Top-K", 1, 5, 3, 1)

    with colC:
        uplift_budget = st.slider("Max uplift skills", 1, 6, 3, 1)
        min_conf = st.slider("Min confidence gate", 0, 100, 25, 5)

        with st.popover("What is CVaR / Pareto / Counterfactual?"):
            st.markdown(
                """
- **CVaR(5%)**: average score in the worst 5% outcomes (downside robustness).
- **Pareto frontier**: roles that are not dominated on (Match ↑, Risk ↑, Effort ↓).
- **Counterfactual uplift**: minimal set of missing skills to unlock a goal (Top-K or score threshold).
                """
            )

    @st.cache_data(show_spinner=False)
    def _decision_brief_table(
        matrix: pd.DataFrame,
        coords_df: pd.DataFrame,
        current_occ: str,
        w_cos: float,
        w_m: float,
        n_samp: int,
        noise: float,
        k_neigh: int,
        max_steps_: int,
    ) -> pd.DataFrame:
        base = compute_all_targets_robustness(
            matrix,
            coords_df,
            current_occ,
            w_cosine=w_cos,
            w_map=w_m,
            n_samples=int(n_samp),
            noise_std=float(noise),
            seed=42,
        )

        rows = []
        for _, r in base.iterrows():
            occ = str(r["occupation"])
            po = find_pivot_path(matrix, current_occ, occ, k_neighbors=int(k_neigh), max_steps=int(max_steps_))
            pc = po.get("total_cost", np.nan) if po.get("reachable") else np.nan

            em = compute_effort_metrics(matrix, current_occ, occ, path_cost=pc)
            out = dict(r)
            out.update(em)
            out["reachable"] = bool(po.get("reachable"))
            rows.append(out)

        df = pd.DataFrame(rows)

        df["pareto_frontier"] = pareto_frontier_flags(
            df,
            maximize_cols=["mean", "cvar05"],
            minimize_cols=["effort_mix"],
        ).astype(bool)

        return df.sort_values(["pareto_frontier", "mean"], ascending=[False, False]).reset_index(drop=True)

    df = _decision_brief_table(
        mat,
        coords,
        current,
        float(w_cosine),
        float(w_map),
        int(n_samples),
        float(noise_std),
        int(k_neighbors),
        int(max_steps),
    )

    gated = df.copy()
    # simple per-target confidence proxy: overlap jaccard between current and each candidate target
    @st.cache_data(show_spinner=False)
    def _confidence_table(matrix: pd.DataFrame, current_occ: str) -> pd.DataFrame:
        a = matrix.loc[current_occ].astype(float).values
        a_nz = a > 0
        rows = []
        for occ in matrix.index.astype(str):
            if occ == current_occ:
                continue
            b = matrix.loc[occ].astype(float).values
            b_nz = b > 0
            inter = float((a_nz & b_nz).mean()) if len(a_nz) else 0.0
            uni = float((a_nz | b_nz).mean()) if len(a_nz) else 0.0
            jacc = 0.0 if uni == 0 else inter / uni
            rows.append({"occupation": occ, "confidence": 100.0 * jacc})
        return pd.DataFrame(rows)

    conf_df = _confidence_table(mat, current)
    gated = gated.merge(conf_df, on="occupation", how="left")
    gated["confidence_gate_ok"] = gated["confidence"].fillna(0.0) >= float(min_conf)

    if risk_profile.startswith("Risk-averse"):
        cand = gated[gated["confidence_gate_ok"]].copy()
        if cand.empty:
            cand = gated.copy()
        chosen = cand.sort_values(["cvar05", "mean"], ascending=False).head(1)
        rationale = "Chosen by **downside robustness (CVaR 5%)** with confidence gate."
    elif risk_profile.startswith("Risk-seeking"):
        cand = gated[gated["confidence_gate_ok"]].copy()
        if cand.empty:
            cand = gated.copy()
        chosen = cand.sort_values(["mean", "cvar05"], ascending=False).head(1)
        rationale = "Chosen by **highest expected score (mean)** with confidence gate."
    else:
        cand = gated[gated["confidence_gate_ok"] & gated["pareto_frontier"]].copy()
        if cand.empty:
            cand = gated[gated["confidence_gate_ok"]].copy()
        if cand.empty:
            cand = gated.copy()
        chosen = cand.sort_values(["mean", "cvar05"], ascending=False).head(1)
        rationale = "Chosen from **Pareto frontier**, then by mean + CVaR."

    st.markdown("### Recommended target (decision-grade)")
    if chosen.empty:
        st.warning("No targets available.")
    else:
        rec = chosen.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended", str(rec["occupation"]))
        c2.metric("Mean", f"{rec['mean']:.1f}")
        c3.metric("CVaR 5%", f"{rec['cvar05']:.1f}")
        c4.metric("Effort (mix)", f"{rec['effort_mix']:.1f}")
        st.info(rationale)

        if st.button("Use recommended as target", key="use_decision_brief_reco"):
            st.session_state["__target_override__"] = str(rec["occupation"])
            st.session_state["__has_run__"] = True
            st.rerun()

    st.markdown("### Tradeoff map (Pareto frontier)")
    plot_df = df.copy()
    plot_df["frontier_label"] = plot_df["pareto_frontier"].apply(lambda x: "Frontier" if x else "Dominated")

    fig = px.scatter(
        plot_df,
        x="effort_mix",
        y="mean",
        size="cvar05",
        hover_name="occupation",
        color="frontier_label",
        title="Higher = better score (y), lower = less effort (x), bubble size = CVaR(5%) robustness",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Decision table (sortable)")
    show_cols = [
        "occupation",
        "mean",
        "cvar05",
        "q05",
        "std",
        "effort_mix",
        "gap_effort",
        "path_cost",
        "reachable",
        "pareto_frontier",
    ]
    st.dataframe(
        df[show_cols].sort_values(["pareto_frontier", "mean"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Counterfactual: minimal skill uplift to unlock your target")
    chosen_target = st.selectbox(
        "Pick a target to unlock",
        options=df["occupation"].astype(str).tolist(),
        index=0,
        key="cf_target_select",
    )

    if goal_mode == "threshold":
        cf = counterfactual_uplift_greedy(
            mat,
            coords,
            current,
            chosen_target,
            w_cosine=float(w_cosine),
            w_map=float(w_map),
            goal_mode="threshold",
            score_threshold=float(score_threshold),
            max_skills=int(uplift_budget),
        )
        st.caption(f"Goal: reach score ≥ {score_threshold:.0f} for **{chosen_target}**")
    else:
        cf = counterfactual_uplift_greedy(
            mat,
            coords,
            current,
            chosen_target,
            w_cosine=float(w_cosine),
            w_map=float(w_map),
            goal_mode="topk",
            top_k=int(topk_goal),
            max_skills=int(uplift_budget),
        )
        st.caption(f"Goal: make **{chosen_target}** appear in Top-{topk_goal}")

    if cf.get("selected_skills"):
        st.success(
            "Suggested uplift skills ("
            + str(len(cf["selected_skills"]))
            + "): **"
            + "**, **".join(cf["selected_skills"])
            + "**"
        )
    else:
        st.info("No uplift skills suggested (either goal already met or dataset too sparse for this target).")

    st.write(
        f"Before score: **{cf.get('before_score', np.nan):.1f}**  → After score: **{cf.get('after_score', np.nan):.1f}**"
    )
    st.caption(str(cf.get("notes", "")))

# ---- What-If Lab
with tabs[1]:
    st.subheader("What-If Sensitivity Lab (overkill mode)")
    st.caption(
        "We sweep across weight settings and noise levels to measure how stable your target recommendation is. "
        "This is the prototyping mindset: accuracy + uncertainty + decision robustness."
    )

    with st.expander("Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            w_min = st.slider("w_cosine min", 0.0, 1.0, 0.40, 0.05)
            w_max = st.slider("w_cosine max", 0.0, 1.0, 0.90, 0.05)
            w_steps = st.slider("w_cosine steps", 3, 9, 6, 1)
        with c2:
            n_min = st.slider("noise min", 0.00, 0.20, 0.00, 0.01)
            n_max = st.slider("noise max", 0.00, 0.20, 0.12, 0.01)
            n_steps = st.slider("noise steps", 3, 9, 6, 1)
        with c3:
            mc = st.slider("MC per cell (speed vs fidelity)", 5, 60, 20, 5)
            topk = st.slider("Top-K stability", 1, 5, 3, 1)
            seed = st.number_input("Seed", value=42, step=1)

        if w_max < w_min:
            st.warning("w_cosine max < min. Swapping.")
            w_min, w_max = w_max, w_min
        if n_max < n_min:
            st.warning("noise max < min. Swapping.")
            n_min, n_max = n_max, n_min

    sweep = _what_if_sweep(
        mat,
        coords,
        current,
        grid_w_cosine=(float(w_min), float(w_max), int(w_steps)),
        grid_noise=(float(n_min), float(n_max), int(n_steps)),
        mc_per_cell=int(mc),
        seed=int(seed),
    )

    stab = _ranking_stability(sweep, target_occ=target, top_k=int(topk))

    st.markdown("### Stability summary")
    a, b, c = st.columns(3)
    a.metric(f"Target in Top-{topk}", f"{100*stab['target_in_topk_rate']:.0f}%")

    if stab["topk_mode"] is None:
        b.metric("Most common Top-K set", "n/a")
        c.metric("Top-K set stability", "n/a")
    else:
        b.metric("Most common Top-K set", " • ".join(stab["topk_mode"]))
        c.metric("Top-K set stability", f"{100*stab['topk_mode_share']:.0f}%")

    freq_df = stab["topk_freq_df"].copy()
    if not freq_df.empty:
        st.markdown("### Robust alternatives (if target is unstable)")
        st.caption("These roles appear in Top-K across many configurations — i.e. more robust recommendations.")
        show_n = min(6, len(freq_df))
        best = freq_df.head(show_n).copy()

        for _, r in best.iterrows():
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.write(f"**{r['occupation']}**")
            with c2:
                st.write(f"{100*float(r['share']):.0f}%")
            with c3:
                if st.button("Use", key=f"whatif_use_{r['occupation']}"):
                    st.session_state["__target_override__"] = r["occupation"]
                    st.session_state["__has_run__"] = True
                    st.rerun()

    st.markdown("### Target score sensitivity (heatmap)")
    target_df = sweep[sweep["occupation"] == target].copy()
    if target_df.empty:
        st.warning("Target not found in sweep output.")
    else:
        pivot = target_df.pivot_table(index="noise_std", columns="w_cosine", values="mean_score", aggfunc="mean")
        fig = px.imshow(
            pivot.to_numpy(),
            x=[f"{x:.2f}" for x in pivot.columns.tolist()],
            y=[f"{y:.2f}" for y in pivot.index.tolist()],
            labels={"x": "w_cosine", "y": "noise_std", "color": "mean score"},
            title="Mean hybrid score for selected target across weight/noise sweep",
            aspect="auto",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Target rank distribution (robust decision view)")
    ranks = stab["target_rank_values"]
    if len(ranks) == 0:
        st.info("No rank values computed.")
    else:
        fig = px.histogram(
            pd.DataFrame({"rank": ranks}),
            x="rank",
            nbins=min(10, len(set(ranks))),
            title="Rank of selected target across configurations",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Stable recommendations (who appears in Top-{topk} most often?)")
    if freq_df.empty:
        st.info("No Top-K frequency computed.")
    else:
        show_n = min(10, len(freq_df))
        fig = px.bar(freq_df.head(show_n), x="occupation", y="share", title=f"Share of configs where role appears in Top-{topk}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(freq_df.head(show_n), use_container_width=True, hide_index=True)

    st.info(
        f"Interpretation: If your target is rarely in Top-{topk} across plausible configurations, it's a risky choice. "
        f"If it stays Top-{topk} across most configs, it's robust."
    )

# ---- Explain
with tabs[2]:
    st.subheader("Explainability (what drives vs blocks the pivot)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Match drivers (overlap)**")
        st.dataframe(
            contrib["match_drivers"][["skill", "current_importance", "target_importance", "overlap", "match_driver_score"]],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("**Missing drivers (priority gaps)**")
        md = contrib["missing_drivers"]
        if md.empty:
            st.success("No missing skills detected in this dataset.")
        else:
            st.dataframe(
                md[["skill", "current_importance", "target_importance", "gap", "missing_priority"]],
                use_container_width=True,
                hide_index=True,
            )

    with c3:
        st.markdown("**Surplus skills (less relevant in target)**")
        ss = contrib["surplus_skills"]
        if ss.empty:
            st.info("No surplus skills detected.")
        else:
            st.dataframe(
                ss[["skill", "current_importance", "target_importance", "gap", "surplus_magnitude"]],
                use_container_width=True,
                hide_index=True,
            )

# ---- Skill Groups
with tabs[3]:
    st.subheader("Skill Groups (taxonomy view)")
    if group_gap_df.empty:
        st.info("No taxonomy artifacts loaded.")
    else:
        left2, right2 = st.columns([1.2, 1.0], gap="large")

        with left2:
            tmp = group_gap_df.copy().sort_values("gap", ascending=False)
            fig = px.bar(tmp, x="group", y="gap", title="Group gap (target − current)")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                tmp[["group", "current_importance", "target_importance", "gap"]],
                use_container_width=True,
                hide_index=True,
            )

        with right2:
            categories = group_gap_df["group"].tolist()
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(r=group_gap_df["current_importance"], theta=categories, fill="toself", name="Current")
            )
            fig.add_trace(
                go.Scatterpolar(r=group_gap_df["target_importance"], theta=categories, fill="toself", name="Target")
            )
            fig.update_layout(height=420, polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Drilldown: missing skills in a group")
            chosen_group = st.selectbox("Select group", options=categories, index=0)
            miss_in_group = filter_missing_skills_by_group(gap_df, skill_taxonomy, chosen_group, top_n=20)
            if miss_in_group.empty:
                st.success("No missing skills in this group (for this pivot).")
            else:
                st.dataframe(
                    miss_in_group[["skill", "current_importance", "target_importance", "gap"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ---- Robustness
with tabs[4]:
    st.subheader("Robustness (stability under noise)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean score", f"{rob['mean']:.1f}")
    m2.metric("Std dev", f"{rob['std']:.1f}")
    m3.metric("95% CI", f"[{rob['ci95_low']:.1f}, {rob['ci95_high']:.1f}]")
    fig = px.histogram(pd.DataFrame({"score": rob["scores"]}), x="score", nbins=20, title="Score distribution (Monte Carlo)")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# ---- Skill Gap
with tabs[5]:
    st.subheader("Skill Gap (target minus current)")
    top_missing = gap_df[gap_df["gap"] > 0].copy()
    top_missing = top_missing.sort_values(["gap", "target_importance"], ascending=False).head(15)
    if top_missing.empty:
        st.success("No positive gaps found.")
    else:
        st.dataframe(
            top_missing[["skill", "current_importance", "target_importance", "gap"]],
            use_container_width=True,
            hide_index=True,
        )

# ---- Radar
with tabs[6]:
    st.subheader("Radar chart (current vs target)")
    radar_df = gap_df.sort_values("target_importance", ascending=False).head(12).copy()
    categories = radar_df["skill"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=radar_df["current_importance"], theta=categories, fill="toself", name="Current"))
    fig.add_trace(go.Scatterpolar(r=radar_df["target_importance"], theta=categories, fill="toself", name="Target"))
    fig.update_layout(height=520, polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

# ---- Export
with tabs[7]:
    st.subheader("Export")
    export_df = gap_df.copy()
    export_df.insert(0, "current_occupation", current)
    export_df.insert(1, "target_occupation", target)
    export_df.insert(2, "match_score", round(match_score, 2))
    export_df.insert(3, "confidence_score", round(conf["confidence_score"], 2))
    export_df.insert(4, "robust_mean_score", round(rob["mean"], 2))
    export_df.insert(5, "robust_ci95_low", round(rob["ci95_low"], 2))
    export_df.insert(6, "robust_ci95_high", round(rob["ci95_high"], 2))

    if not group_gap_df.empty:
        for _, r in group_gap_df.iterrows():
            key = f"group_gap__{str(r['group']).replace(' ', '_').lower()}"
            export_df[key] = round(float(r["gap"]), 4)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download pivot CSV",
        data=csv_bytes,
        file_name=f"pivot_{current}_to_{target}.csv".replace(" ", "_").lower(),
        mime="text/csv",
    )

# ---- Map (advanced)
with tabs[8]:
    st.subheader("PCA map (advanced)")
    st.caption("PCA axes are not semantic. Use only for rough neighborhood intuition.")
    if not show_map:
        st.info("Enable **Show PCA map (advanced)** in the sidebar to display this.")
    else:
        coords_plot = coords.copy()
        coords_plot["selected"] = "Other"
        coords_plot.loc[coords_plot["occupation"] == current, "selected"] = "Current"
        coords_plot.loc[coords_plot["occupation"] == target, "selected"] = "Target"

        if clusters:
            coords_plot["cluster_id"] = coords_plot["occupation"].map(lambda o: clusters.get(str(o), -1))
            coords_plot["cluster_theme"] = coords_plot["cluster_id"].map(lambda c_: format_cluster_theme(c_, cluster_themes))
        else:
            coords_plot["cluster_id"] = -1
            coords_plot["cluster_theme"] = ""

        if color_by_cluster and clusters:
            fig = px.scatter(
                coords_plot,
                x="x",
                y="y",
                hover_name="occupation",
                color="cluster_id",
                symbol="selected",
                hover_data={"cluster_id": True, "cluster_theme": True, "x": False, "y": False},
                title="Clustered PCA map (career neighborhoods).",
            )
        else:
            fig = px.scatter(
                coords_plot,
                x="x",
                y="y",
                hover_name="occupation",
                color="selected",
                title="PCA map (unclustered).",
            )

        if path_out.get("reachable") and len(path_out.get("path", [])) >= 2:
            path = path_out["path"]
            sub = coords.set_index("occupation").loc[path][["x", "y"]].reset_index()
            fig.add_trace(
                go.Scatter(
                    x=sub["x"],
                    y=sub["y"],
                    mode="lines+markers",
                    name="Pivot path",
                    hovertext=sub["occupation"],
                )
            )

        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Engineering note: runtime loads precomputed artifacts (matrix + PCA coords + clusters + taxonomy). "
    "This prototype adds decision robustness via what-if sensitivity sweeps."
)