from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.model_logic import (
    load_runtime_artifacts,
    compute_gap_df,
    compute_skill_contributions,
    generate_learning_plan,
    compute_confidence_score,
    find_pivot_path,
    compute_group_gap_df,
    filter_missing_skills_by_group,
    format_cluster_theme,
    compute_effort_metrics,
    pareto_frontier_flags,
    counterfactual_uplift_greedy,
)

st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")

st.title("🧭 Career Pivot Simulator")
st.caption(
    "Decision-support prototype for career pivots: matching + explainability + planning + robustness + what-if sensitivity. "
    "Artifacts are precomputed offline and loaded at runtime."
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_data(show_spinner=False)
def load_artifacts() -> Any:
    return load_runtime_artifacts("artifacts")


try:
    art = load_artifacts()
except Exception as e:
    st.error("Missing or invalid runtime artifacts.")
    st.info("Run: `python scripts/preprocess_onet.py` to generate artifacts.")
    st.exception(e)
    st.stop()

mat: pd.DataFrame = art.matrix
coords_pca: pd.DataFrame = art.coords

coords_umap: pd.DataFrame = art.umap_coords if hasattr(art, "umap_coords") else pd.DataFrame(columns=["occupation", "x", "y"])
umap_meta: dict[str, Any] = art.umap_meta if hasattr(art, "umap_meta") and isinstance(art.umap_meta, dict) else {}

occupations = mat.index.astype(str).tolist()

clusters: dict[str, int] = art.clusters if hasattr(art, "clusters") and isinstance(art.clusters, dict) else {}
cluster_themes: dict[str, Any] = (
    art.cluster_themes if hasattr(art, "cluster_themes") and isinstance(art.cluster_themes, dict) else {}
)
skill_taxonomy: dict[str, str] = (
    art.skill_taxonomy if hasattr(art, "skill_taxonomy") and isinstance(art.skill_taxonomy, dict) else {}
)
group_meta: dict[str, Any] = art.group_meta if hasattr(art, "group_meta") and isinstance(art.group_meta, dict) else {}

# -----------------------------
# Session state
# -----------------------------
if "__has_run__" not in st.session_state:
    st.session_state["__has_run__"] = False
if "__target_override__" not in st.session_state:
    st.session_state["__target_override__"] = None

# cache buckets for heavy computations
if "__cache__" not in st.session_state:
    st.session_state["__cache__"] = {
        "robustness": {},     # key -> rob dict
        "decision_brief": {}, # key -> df
        "whatif": {},         # key -> (sweep_df, stab_dict)
    }


# -----------------------------
# Helpers
# -----------------------------
def _coords_is_valid(df: pd.DataFrame) -> bool:
    return (
        isinstance(df, pd.DataFrame)
        and (not df.empty)
        and {"occupation", "x", "y"}.issubset(df.columns)
        and df["occupation"].astype(str).nunique() >= 2
    )


@st.cache_data(show_spinner=False)
def _precompute_map_context(coords_df: pd.DataFrame) -> dict[str, Any]:
    """
    Fast map normalization:
    max_dist = bounding box diagonal (O(n)).
    Also builds lookup occupation -> (x,y).
    """
    df = coords_df.copy()
    df["occupation"] = df["occupation"].astype(str)
    df = df.dropna(subset=["occupation", "x", "y"])
    df = df.drop_duplicates(subset=["occupation"], keep="first").reset_index(drop=True)

    xy = df[["x", "y"]].to_numpy(dtype=float)
    if xy.shape[0] < 2:
        max_dist = 0.0
    else:
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)
        max_dist = float(np.linalg.norm([xmax - xmin, ymax - ymin]))

    occ_to_xy: dict[str, tuple[float, float]] = {}
    for occ, x, y in df[["occupation", "x", "y"]].itertuples(index=False, name=None):
        occ_to_xy[str(occ)] = (float(x), float(y))

    return {"df": df, "xy": xy, "max_dist": float(max_dist), "occ_to_xy": occ_to_xy}


@st.cache_data(show_spinner=False)
def _map_parts_from_current(coords_df: pd.DataFrame, current_occ: str) -> dict[str, float]:
    ctx = _precompute_map_context(coords_df)
    df = ctx["df"]
    xy = ctx["xy"]
    max_dist = float(ctx["max_dist"])

    if df.empty or xy.shape[0] == 0:
        return {}

    occs = df["occupation"].astype(str).to_numpy()
    cur_mask = occs == str(current_occ)
    if not np.any(cur_mask):
        return {}

    cur_xy = xy[np.argmax(cur_mask)]
    if max_dist <= 0.0:
        return {str(o): 1.0 for o in occs}

    d = np.linalg.norm(xy - cur_xy, axis=1)
    prox = np.clip(1.0 - d / max_dist, 0.0, 1.0)
    return {str(o): float(p) for o, p in zip(occs, prox)}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _hybrid_score_from_vectors(a: np.ndarray, b: np.ndarray, map_part: float, w_cosine: float, w_map: float) -> float:
    cos = max(0.0, _cosine(a, b))  # 0..1
    return float(np.clip(100.0 * (w_cosine * cos + w_map * float(map_part)), 0.0, 100.0))


@st.cache_data(show_spinner=False)
def _recommend_similar_roles_fast(
    matrix: pd.DataFrame,
    coords_df: pd.DataFrame,
    current_occ: str,
    *,
    top_k: int,
    w_cosine: float,
    w_map: float,
) -> pd.DataFrame:
    a = matrix.loc[current_occ].astype(float).to_numpy()
    map_parts = _map_parts_from_current(coords_df, current_occ)

    rows: list[tuple[str, float]] = []
    for occ in matrix.index.astype(str):
        if occ == current_occ:
            continue
        b = matrix.loc[occ].astype(float).to_numpy()
        mp = float(map_parts.get(occ, 0.0))
        s = _hybrid_score_from_vectors(a, b, mp, float(w_cosine), float(w_map))
        rows.append((occ, s))

    return (
        pd.DataFrame(rows, columns=["occupation", "match_score"])
        .sort_values("match_score", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )


def _pick_coords(choice: str) -> tuple[pd.DataFrame, str]:
    if choice == "UMAP" and _coords_is_valid(coords_umap):
        return coords_umap.copy(), "UMAP"
    return coords_pca.copy(), "PCA"


def _cache_key(*parts: Any) -> str:
    return "|".join([str(p) for p in parts])


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")
    st.write("Pick a pivot → run analysis → explore deep dives on demand.")
    st.divider()

    with st.expander("1) Choose your pivot", expanded=True):
        current = st.selectbox("Current occupation", options=occupations, index=0)
        default_target_idx = 1 if len(occupations) > 1 else 0
        target = st.selectbox("Target occupation", options=occupations, index=default_target_idx)

        if st.session_state["__target_override__"]:
            target = st.session_state["__target_override__"]

        if current == target:
            st.warning("Current and Target are identical. Pick a different target to see meaningful results.")

    with st.expander("2) Scoring knobs (optional)", expanded=False):
        st.caption("Default is fine. Only change to explore trade-offs.")
        w_cosine = st.slider("Skill similarity weight", 0.0, 1.0, 0.65, 0.05)
        w_map = st.slider("Map proximity weight", 0.0, 1.0, 0.35, 0.05)
        s = float(w_cosine + w_map)
        if s <= 0:
            w_cosine, w_map = 1.0, 0.0
        else:
            w_cosine, w_map = float(w_cosine / s), float(w_map / s)

    with st.expander("3) Planning & robustness knobs", expanded=False):
        st.subheader("Path planning")
        k_neighbors = st.slider("kNN neighbors", 2, 10, 5, 1)
        max_steps = st.slider("Max steps", 2, 6, 4, 1)

        st.subheader("Robustness (runs on-demand)")
        n_samples = st.slider("Monte Carlo samples", 50, 600, 200, 50)
        noise_std = st.slider("Noise level (std)", 0.00, 0.20, 0.05, 0.01)

    with st.expander("Advanced visualization", expanded=False):
        embed_options = ["PCA"]
        if _coords_is_valid(coords_umap):
            embed_options.append("UMAP")

        embedding_choice = st.selectbox("Embedding for map + proximity", options=embed_options, index=0)
        show_map = st.toggle("Show map tab (advanced)", value=False)
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
# Empty-state
# -----------------------------
if not st.session_state["__has_run__"]:
    st.info("Choose Current + Target in the sidebar and click **Run pivot analysis**.")
    st.stop()

coords, coords_label = _pick_coords(embedding_choice)

# -----------------------------
# Fast core outputs (no heavy Monte Carlo here)
# -----------------------------
gap_df = compute_gap_df(mat, current, target)
contrib = compute_skill_contributions(gap_df)
plan = generate_learning_plan(gap_df)
conf = compute_confidence_score(mat, art.pca_meta, current, target)
path_out = find_pivot_path(mat, current, target, k_neighbors=int(k_neighbors), max_steps=int(max_steps))
group_gap_df = compute_group_gap_df(mat, skill_taxonomy, group_meta, current, target)

# fast score (no loops over all pairs)
a_vec = mat.loc[current].astype(float).to_numpy()
b_vec = mat.loc[target].astype(float).to_numpy()
map_parts_current = _map_parts_from_current(coords, current)
map_part_ct = float(map_parts_current.get(str(target), 0.0))
match_score = _hybrid_score_from_vectors(a_vec, b_vec, map_part_ct, float(w_cosine), float(w_map))
cosine_score = float(np.clip(max(0.0, _cosine(a_vec, b_vec)) * 100.0, 0.0, 100.0))

# -----------------------------
# Overview (FAST)
# -----------------------------
st.subheader("Overview")

# robustness placeholder (computed on-demand)
rob_key = _cache_key("rob", coords_label, current, target, w_cosine, w_map, n_samples, noise_std)
rob_cached = st.session_state["__cache__"]["robustness"].get(rob_key)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Match score (hybrid)", f"{match_score:.0f}/100")
m2.metric("Confidence (coverage)", f"{conf['confidence_score']:.0f}/100")

if isinstance(rob_cached, dict):
    m3.metric("Robust mean", f"{rob_cached['mean']:.1f}")
    m4.metric("Robust 95% CI width", f"{(rob_cached['ci95_high'] - rob_cached['ci95_low']):.1f}")
else:
    m3.metric("Robust mean", "—")
    m4.metric("Robust 95% CI width", "—")

st.success(f"Map embedding: **{coords_label}** • Map proximity: **{map_part_ct:.2f}**")

with st.expander("Under the hood: score breakdown (transparent)", expanded=False):
    st.write(f"- Cosine-based score: **{cosine_score:.1f}/100**")
    st.write(f"- Map proximity (0..1): **{map_part_ct:.2f}** (from {coords_label})")
    st.write(f"- Weights: `w_cosine={w_cosine:.2f}`, `w_map={w_map:.2f}`")
    st.code("score = 100 * (w_cosine * max(0, cosine_similarity) + w_map * map_proximity)")

# -----------------------------
# Main: Neighborhood + Plan (FAST)
# -----------------------------
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.subheader("Career neighborhood (actionable)")
    rec_df = _recommend_similar_roles_fast(mat, coords, current, top_k=6, w_cosine=float(w_cosine), w_map=float(w_map))
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown("**Use a recommended role as new target:**")
    for _, row in rec_df.iterrows():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**{row['occupation']}** — {row['match_score']:.0f}/100")
        with c2:
            if st.button("Use", key=f"use_{row['occupation']}"):
                st.session_state["__target_override__"] = str(row["occupation"])
                st.session_state["__has_run__"] = True
                st.rerun()

    if clusters:
        st.divider()
        st.markdown("**Career neighborhood theme (cluster)**")
        cur_cluster = clusters.get(str(current), None)
        if cur_cluster is None:
            st.info("No cluster found for current role.")
        else:
            st.write(f"Cluster **{cur_cluster}** theme skills: {format_cluster_theme(cur_cluster, cluster_themes)}")

with right:
    st.subheader("Plan: Pivot Path + Learning Plan")
    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.markdown("### Pivot Path Finder")
        if not path_out.get("reachable"):
            st.warning(path_out.get("notes", "Target not reachable. Try increasing kNN neighbors."))
        else:
            path = path_out["path"]
            step_costs = path_out.get("step_costs", [])
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
# Tabs
# -----------------------------
st.divider()
tabs = st.tabs(
    ["Explain", "Skill Groups", "Robustness (on-demand)", "Decision Brief (on-demand)", "What-If Lab (on-demand)", "Map (advanced)", "Export"]
)

# ---- Explain
with tabs[0]:
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
        st.markdown("**Surplus skills**")
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
with tabs[1]:
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
            fig.add_trace(go.Scatterpolar(r=group_gap_df["current_importance"], theta=categories, fill="toself", name="Current"))
            fig.add_trace(go.Scatterpolar(r=group_gap_df["target_importance"], theta=categories, fill="toself", name="Target"))
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

# ---- Robustness (on-demand)
with tabs[2]:
    st.subheader("Robustness (stability under noise) — on-demand")
    st.caption("This is expensive. Click to run Monte Carlo and cache results for this configuration.")

    if st.button("Run robustness (Monte Carlo)", key="run_robustness"):
        with st.spinner("Running Monte Carlo robustness..."):
            rng = np.random.default_rng(42)
            a0 = mat.loc[current].astype(float).to_numpy()
            b0 = mat.loc[target].astype(float).to_numpy()
            mp = float(map_parts_current.get(str(target), 0.0))

            scores: list[float] = []
            for _ in range(int(n_samples)):
                a = np.clip(a0 + rng.normal(0.0, float(noise_std), size=a0.shape), 0.0, None)
                b = np.clip(b0 + rng.normal(0.0, float(noise_std), size=b0.shape), 0.0, None)
                s = _hybrid_score_from_vectors(a, b, mp, float(w_cosine), float(w_map))
                scores.append(s)

            arr = np.array(scores, dtype=float)
            rob = {
                "n_samples": int(n_samples),
                "noise_std": float(noise_std),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci95_low": float(np.quantile(arr, 0.025)),
                "ci95_high": float(np.quantile(arr, 0.975)),
                "scores": scores,
            }
            st.session_state["__cache__"]["robustness"][rob_key] = rob
            rob_cached = rob

    if not isinstance(rob_cached, dict):
        st.info("Not computed yet for this pivot/config.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean score", f"{rob_cached['mean']:.1f}")
        m2.metric("Std dev", f"{rob_cached['std']:.1f}")
        m3.metric("95% CI", f"[{rob_cached['ci95_low']:.1f}, {rob_cached['ci95_high']:.1f}]")

        fig = px.histogram(pd.DataFrame({"score": rob_cached["scores"]}), x="score", nbins=20, title=f"Score distribution — map={coords_label}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---- Decision Brief (on-demand)
with tabs[3]:
    st.subheader("Decision Brief — on-demand (very expensive)")
    st.caption("This computes robust stats for all targets + path costs. Use only when needed.")

    risk_options = ["Risk-averse (CVaR)", "Balanced", "Risk-seeking (Mean)"]
    try:
        risk_profile = st.segmented_control("Risk profile", options=risk_options, default=risk_options[1])
    except Exception:
        risk_profile = st.radio("Risk profile", options=risk_options, index=1)

    uplift_budget = st.slider("Max uplift skills", 1, 6, 3, 1)
    min_conf = st.slider("Min confidence gate", 0, 100, 25, 5)

    db_key = _cache_key("db", coords_label, current, w_cosine, w_map, n_samples, noise_std, k_neighbors, max_steps)
    df_cached = st.session_state["__cache__"]["decision_brief"].get(db_key)

    if st.button("Compute Decision Brief", key="run_decision_brief"):
        with st.spinner("Computing Decision Brief (robust all-target ranking + effort + pareto)..."):
            rng = np.random.default_rng(42)
            a0 = mat.loc[current].astype(float).to_numpy()
            map_parts = map_parts_current

            rows: list[dict[str, Any]] = []
            for occ in mat.index.astype(str):
                if occ == current:
                    continue

                b0 = mat.loc[occ].astype(float).to_numpy()
                mp = float(map_parts.get(occ, 0.0))

                scores = []
                for _ in range(int(n_samples)):
                    a = np.clip(a0 + rng.normal(0.0, float(noise_std), size=a0.shape), 0.0, None)
                    b = np.clip(b0 + rng.normal(0.0, float(noise_std), size=b0.shape), 0.0, None)
                    s = _hybrid_score_from_vectors(a, b, mp, float(w_cosine), float(w_map))
                    scores.append(s)

                arr = np.array(scores, dtype=float)
                q05 = float(np.quantile(arr, 0.05))
                tail = arr[arr <= q05]
                cvar05 = float(np.mean(tail)) if tail.size else q05

                po = find_pivot_path(mat, current, occ, k_neighbors=int(k_neighbors), max_steps=int(max_steps))
                pc = po.get("total_cost", np.nan) if po.get("reachable") else np.nan

                em = compute_effort_metrics(mat, current, occ, path_cost=pc)

                rows.append(
                    {
                        "occupation": occ,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "q05": q05,
                        "q95": float(np.quantile(arr, 0.95)),
                        "cvar05": cvar05,
                        "map_part": float(mp),
                        "reachable": bool(po.get("reachable")),
                        **em,
                    }
                )

            df = pd.DataFrame(rows)
            df["pareto_frontier"] = pareto_frontier_flags(
                df,
                maximize_cols=["mean", "cvar05"],
                minimize_cols=["effort_mix"],
            ).astype(bool)
            df = df.sort_values(["pareto_frontier", "mean"], ascending=[False, False]).reset_index(drop=True)

            st.session_state["__cache__"]["decision_brief"][db_key] = df
            df_cached = df

    if df_cached is None or not isinstance(df_cached, pd.DataFrame) or df_cached.empty:
        st.info("Not computed yet for this configuration.")
    else:
        # confidence gating (cheap)
        a = mat.loc[current].astype(float).values
        a_nz = a > 0
        conf_rows = []
        for occ in mat.index.astype(str):
            if occ == current:
                continue
            b = mat.loc[occ].astype(float).values
            b_nz = b > 0
            inter = float((a_nz & b_nz).mean()) if len(a_nz) else 0.0
            uni = float((a_nz | b_nz).mean()) if len(a_nz) else 0.0
            jacc = 0.0 if uni == 0 else inter / uni
            conf_rows.append({"occupation": occ, "confidence": 100.0 * jacc})
        conf_df = pd.DataFrame(conf_rows)

        gated = df_cached.merge(conf_df, on="occupation", how="left")
        gated["confidence_gate_ok"] = gated["confidence"].fillna(0.0) >= float(min_conf)

        if risk_profile.startswith("Risk-averse"):
            cand = gated[gated["confidence_gate_ok"]].copy()
            if cand.empty:
                cand = gated.copy()
            chosen = cand.sort_values(["cvar05", "mean"], ascending=False).head(1)
            rationale = "Chosen by downside robustness (CVaR 5%) with confidence gate."
        elif risk_profile.startswith("Risk-seeking"):
            cand = gated[gated["confidence_gate_ok"]].copy()
            if cand.empty:
                cand = gated.copy()
            chosen = cand.sort_values(["mean", "cvar05"], ascending=False).head(1)
            rationale = "Chosen by highest expected score (mean) with confidence gate."
        else:
            cand = gated[gated["confidence_gate_ok"] & gated["pareto_frontier"]].copy()
            if cand.empty:
                cand = gated[gated["confidence_gate_ok"]].copy()
            if cand.empty:
                cand = gated.copy()
            chosen = cand.sort_values(["mean", "cvar05"], ascending=False).head(1)
            rationale = "Chosen from Pareto frontier, then by mean + CVaR."

        st.markdown("### Recommended target (decision-grade)")
        if not chosen.empty:
            rec = chosen.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended", str(rec["occupation"]))
            c2.metric("Mean", f"{rec['mean']:.1f}")
            c3.metric("CVaR 5%", f"{rec['cvar05']:.1f}")
            c4.metric("Effort (mix)", f"{rec['effort_mix']:.1f}")
            st.info(rationale)

        st.markdown("### Tradeoff map (Pareto)")
        plot_df = df_cached.copy()
        plot_df["frontier_label"] = plot_df["pareto_frontier"].apply(lambda x: "Frontier" if x else "Dominated")
        fig = px.scatter(
            plot_df,
            x="effort_mix",
            y="mean",
            size="cvar05",
            hover_name="occupation",
            color="frontier_label",
            title="Higher = better score (y), lower = less effort (x), bubble size = CVaR(5%)",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Counterfactual uplift (your selected target)")
        goal_mode = st.radio("Goal mode", options=["threshold", "topk"], index=0, horizontal=True)
        score_threshold = st.slider("Score threshold", 40.0, 90.0, 60.0, 1.0)
        topk_goal = st.slider("Top-K", 1, 5, 3, 1)

        if goal_mode == "threshold":
            cf = counterfactual_uplift_greedy(
                mat,
                coords,
                current,
                target,
                w_cosine=float(w_cosine),
                w_map=float(w_map),
                goal_mode="threshold",
                score_threshold=float(score_threshold),
                max_skills=int(uplift_budget),
            )
        else:
            cf = counterfactual_uplift_greedy(
                mat,
                coords,
                current,
                target,
                w_cosine=float(w_cosine),
                w_map=float(w_map),
                goal_mode="topk",
                top_k=int(topk_goal),
                max_skills=int(uplift_budget),
            )

        if cf.get("selected_skills"):
            st.success("Suggested uplift skills: **" + "**, **".join(cf["selected_skills"]) + "**")
        else:
            st.info("No uplift skills suggested (goal already met or sparse).")

        st.write(f"Before: **{cf.get('before_score', np.nan):.1f}** → After: **{cf.get('after_score', np.nan):.1f}**")

# ---- What-If Lab (on-demand)
with tabs[4]:
    st.subheader("What-If Lab — on-demand (extremely expensive)")
    st.caption("This can take long. Keep grid small for demo. Results are cached per configuration.")

    c1, c2, c3 = st.columns(3)
    with c1:
        w_min = st.slider("w_cosine min", 0.0, 1.0, 0.40, 0.05, key="wmin")
        w_max = st.slider("w_cosine max", 0.0, 1.0, 0.90, 0.05, key="wmax")
        w_steps = st.slider("w_cosine steps", 3, 7, 5, 1, key="wsteps")
    with c2:
        n_min = st.slider("noise min", 0.00, 0.20, 0.00, 0.01, key="nmin")
        n_max = st.slider("noise max", 0.00, 0.20, 0.10, 0.01, key="nmax")
        n_steps = st.slider("noise steps", 3, 7, 5, 1, key="nsteps")
    with c3:
        mc = st.slider("MC per cell", 5, 30, 15, 5, key="mc")
        topk = st.slider("Top-K stability", 1, 5, 3, 1, key="topk")
        seed = st.number_input("Seed", value=42, step=1, key="seed")

    if w_max < w_min:
        w_min, w_max = w_max, w_min
    if n_max < n_min:
        n_min, n_max = n_max, n_min

    wf_key = _cache_key("wf", coords_label, current, target, w_min, w_max, w_steps, n_min, n_max, n_steps, mc, seed)
    cached = st.session_state["__cache__"]["whatif"].get(wf_key)

    if st.button("Run What-If Sweep", key="run_whatif"):
        with st.spinner("Running sweep (this is heavy)..."):
            rng = np.random.default_rng(int(seed))
            a0 = mat.loc[current].astype(float).to_numpy()
            map_parts = map_parts_current

            w_vals = np.linspace(float(w_min), float(w_max), int(w_steps))
            n_vals = np.linspace(float(n_min), float(n_max), int(n_steps))

            records = []
            for w in w_vals:
                w = float(np.clip(w, 0.0, 1.0))
                w_map_local = float(1.0 - w)
                for noise in n_vals:
                    noise = float(max(0.0, noise))
                    for occ in mat.index.astype(str):
                        if occ == current:
                            continue
                        b0 = mat.loc[occ].astype(float).to_numpy()
                        mp = float(map_parts.get(occ, 0.0))

                        scores = []
                        for _ in range(int(mc)):
                            a = np.clip(a0 + rng.normal(0.0, noise, size=a0.shape), 0.0, None)
                            b = np.clip(b0 + rng.normal(0.0, noise, size=b0.shape), 0.0, None)
                            s = _hybrid_score_from_vectors(a, b, mp, w, w_map_local)
                            scores.append(s)

                        records.append(
                            {"w_cosine": w, "w_map": w_map_local, "noise_std": noise, "occupation": occ, "mean_score": float(np.mean(scores))}
                        )

            sweep_df = pd.DataFrame(records)

            # stability
            group_cols = ["w_cosine", "w_map", "noise_std"]
            topk_lists = []
            target_ranks = []

            for _, g in sweep_df.groupby(group_cols):
                g2 = g.sort_values("mean_score", ascending=False).reset_index(drop=True)
                topk_set = tuple(g2.head(int(topk))["occupation"].tolist())
                topk_lists.append(topk_set)

                tr = g2.index[g2["occupation"] == target]
                target_ranks.append(int(tr[0] + 1) if len(tr) else np.nan)

            ranks_arr = np.array([r for r in target_ranks if not (isinstance(r, float) and np.isnan(r))], dtype=int)
            in_topk_rate = float(np.mean(ranks_arr <= int(topk))) if len(ranks_arr) else 0.0

            freq = Counter()
            for t in topk_lists:
                for occ in t:
                    freq[occ] += 1
            freq_df = pd.DataFrame({"occupation": list(freq.keys()), "count": list(freq.values())}).sort_values("count", ascending=False)
            freq_df["share"] = freq_df["count"] / max(1, len(topk_lists))

            stab = {
                "target_in_topk_rate": in_topk_rate,
                "target_rank_values": ranks_arr.tolist(),
                "topk_freq_df": freq_df.reset_index(drop=True),
                "n_configs": int(len(topk_lists)),
            }

            st.session_state["__cache__"]["whatif"][wf_key] = (sweep_df, stab)
            cached = (sweep_df, stab)

    if cached is None:
        st.info("Not computed yet. Click **Run What-If Sweep**.")
    else:
        sweep_df, stab = cached
        st.metric(f"Target in Top-{topk}", f"{100*stab['target_in_topk_rate']:.0f}%")

        target_df = sweep_df[sweep_df["occupation"] == target].copy()
        if not target_df.empty:
            pivot = target_df.pivot_table(index="noise_std", columns="w_cosine", values="mean_score", aggfunc="mean")
            fig = px.imshow(
                pivot.to_numpy(),
                x=[f"{x:.2f}" for x in pivot.columns.tolist()],
                y=[f"{y:.2f}" for y in pivot.index.tolist()],
                labels={"x": "w_cosine", "y": "noise_std", "color": "mean score"},
                title=f"Mean hybrid score for target across sweep ({coords_label})",
                aspect="auto",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

        freq_df = stab["topk_freq_df"]
        if not freq_df.empty:
            show_n = min(10, len(freq_df))
            fig = px.bar(freq_df.head(show_n), x="occupation", y="share", title=f"Share of configs where role appears in Top-{topk}")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(freq_df.head(show_n), use_container_width=True, hide_index=True)

# ---- Map (advanced)
with tabs[5]:
    st.subheader(f"Map (advanced) — {coords_label}")
    if not show_map:
        st.info("Enable **Show map tab (advanced)** in the sidebar to display this.")
    else:
        try:
            coords_plot = coords.copy()
            coords_plot["occupation"] = coords_plot["occupation"].astype(str)

            coords_plot["selected"] = "Other"
            coords_plot.loc[coords_plot["occupation"] == str(current), "selected"] = "Current"
            coords_plot.loc[coords_plot["occupation"] == str(target), "selected"] = "Target"

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
                    title=f"Clustered {coords_label} map.",
                )
            else:
                fig = px.scatter(
                    coords_plot,
                    x="x",
                    y="y",
                    hover_name="occupation",
                    color="selected",
                    title=f"{coords_label} map.",
                )

            # Path overlay
            if path_out.get("reachable") and len(path_out.get("path", [])) >= 2:
                path = [str(p) for p in path_out.get("path", [])]
                coords_idx = coords_plot.set_index("occupation")
                if all(p in coords_idx.index for p in path):
                    sub = coords_idx.loc[path][["x", "y"]].reset_index()
                    fig.add_trace(go.Scatter(x=sub["x"], y=sub["y"], mode="lines+markers", name="Pivot path", hovertext=sub["occupation"]))

            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Map rendering failed (the rest of the app is still usable).")
            with st.expander("Details"):
                st.exception(e)

# ---- Export
with tabs[6]:
    st.subheader("Export")
    export_df = gap_df.copy()
    export_df.insert(0, "current_occupation", current)
    export_df.insert(1, "target_occupation", target)
    export_df.insert(2, "match_score", round(match_score, 2))
    export_df.insert(3, "confidence_score", round(conf["confidence_score"], 2))
    export_df.insert(4, "map_embedding", coords_label)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download pivot CSV",
        data=csv_bytes,
        file_name=f"pivot_{current}_to_{target}.csv".replace(" ", "_").lower(),
        mime="text/csv",
    )

st.caption("Note: Heavy computations are on-demand to keep the prototype fast and usable.")