from __future__ import annotations

from dataclasses import dataclass
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
    compute_confidence_score,
    compute_group_gap_df,
    filter_missing_skills_by_group,
    format_cluster_theme,
    compute_effort_metrics,
    pareto_frontier_flags,
    counterfactual_uplift_greedy,
)
from src.ai_coach import generate_learning_plan_markdown


# ============================================================
# Page config + Branding
# ============================================================
st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")

st.title("🧭 Career Pivot Simulator")
st.caption(
    "AI-first decision-support prototype for career pivots: match scoring + explainability + stepping-stone paths + robustness + what-if exploration. "
    "Artifacts are precomputed offline and loaded at runtime (deploy-friendly)."
)

# ============================================================
# Load artifacts (fast + stable)
# ============================================================
@st.cache_data(show_spinner=False)
def _load_artifacts_cached() -> Any:
    return load_runtime_artifacts("artifacts")


try:
    art = _load_artifacts_cached()
except Exception as e:
    st.error("Missing or invalid runtime artifacts.")
    st.info("Run: `python scripts/preprocess_onet.py` to generate artifacts.")
    st.exception(e)
    st.stop()

mat: pd.DataFrame = art.matrix
coords_pca: pd.DataFrame = art.coords

coords_umap: pd.DataFrame = (
    art.umap_coords
    if hasattr(art, "umap_coords") and isinstance(art.umap_coords, pd.DataFrame)
    else pd.DataFrame(columns=["occupation", "x", "y"])
)

clusters: dict[str, int] = art.clusters if hasattr(art, "clusters") and isinstance(art.clusters, dict) else {}
cluster_themes: dict[str, Any] = (
    art.cluster_themes if hasattr(art, "cluster_themes") and isinstance(art.cluster_themes, dict) else {}
)
skill_taxonomy: dict[str, str] = (
    art.skill_taxonomy if hasattr(art, "skill_taxonomy") and isinstance(art.skill_taxonomy, dict) else {}
)
group_meta: dict[str, Any] = art.group_meta if hasattr(art, "group_meta") and isinstance(art.group_meta, dict) else {}

occupations: list[str] = mat.index.astype(str).tolist()

# ============================================================
# Session state
# ============================================================
if "__has_run__" not in st.session_state:
    st.session_state["__has_run__"] = False

if "__target_override__" not in st.session_state:
    st.session_state["__target_override__"] = None

if "__guided_k_neighbors__" not in st.session_state:
    st.session_state["__guided_k_neighbors__"] = 6  # slightly higher than 5 -> fewer "unreachable"

if "__guided_max_steps__" not in st.session_state:
    st.session_state["__guided_max_steps__"] = 5

# predictable, session-local cache
if "__cache__" not in st.session_state:
    st.session_state["__cache__"] = {
        "map_ctx": {},        # label -> context dict
        "knn_graph": {},      # (k, use_idf) -> graph
        "robustness": {},     # key -> rob dict
        "decision_brief": {}, # key -> dict payload
        "whatif": {},         # key -> dict payload
        "score_dist": {},     # key -> dict scores
    }

# ============================================================
# Helpers
# ============================================================
def _coords_is_valid(df: pd.DataFrame) -> bool:
    return (
        isinstance(df, pd.DataFrame)
        and not df.empty
        and {"occupation", "x", "y"}.issubset(df.columns)
        and df["occupation"].astype(str).nunique() >= 2
    )


def _pick_coords(choice: str) -> tuple[pd.DataFrame, str]:
    if choice == "UMAP" and _coords_is_valid(coords_umap):
        return coords_umap.copy(), "UMAP"
    return coords_pca.copy(), "PCA"


def _cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def _ensure_core_arrays() -> None:
    """
    Precompute fast numeric structures once per session:
      - X (n_roles, n_skills)
      - cosine matrices (raw + IDF-weighted)
      - IDF weights
      - occupation list + index map
    """
    if "__core__" in st.session_state:
        return

    X = mat.to_numpy(dtype=float)
    occs = mat.index.astype(str).tolist()
    occ_to_idx = {o: i for i, o in enumerate(occs)}

    # Unweighted cosine
    norms = np.linalg.norm(X, axis=1)
    norms_safe = np.where(norms == 0.0, 1.0, norms)
    Xn = X / norms_safe[:, None]
    cos_mat = np.clip(Xn @ Xn.T, -1.0, 1.0)

    # IDF weights (downweight common skills)
    df = np.sum(X > 0.0, axis=0).astype(float)
    N = float(X.shape[0])
    idf = np.log((N + 1.0) / (1.0 + df)) + 1.0
    idf = np.clip(idf, 1.0, None)

    # Weighted cosine
    Xw = X * idf[None, :]
    norms_w = np.linalg.norm(Xw, axis=1)
    norms_w_safe = np.where(norms_w == 0.0, 1.0, norms_w)
    Xwn = Xw / norms_w_safe[:, None]
    cos_mat_idf = np.clip(Xwn @ Xwn.T, -1.0, 1.0)

    st.session_state["__core__"] = {
        "X": X,
        "Xn": Xn,
        "cos_mat": cos_mat,
        "idf": idf,
        "Xw": Xw,
        "Xwn": Xwn,
        "cos_mat_idf": cos_mat_idf,
        "occs": occs,
        "occ_to_idx": occ_to_idx,
    }


def _get_map_ctx(coords_df: pd.DataFrame, label: str) -> dict[str, Any]:
    cache = st.session_state["__cache__"]["map_ctx"]
    if label in cache:
        return cache[label]

    df = coords_df.copy()
    df = df.dropna(subset=["occupation", "x", "y"]).copy()
    df["occupation"] = df["occupation"].astype(str)
    df = df.drop_duplicates(subset=["occupation"], keep="first")
    df = df.sort_values("occupation").reset_index(drop=True)

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

    ctx = {"df": df, "xy": xy, "max_dist": float(max_dist), "occ_to_xy": occ_to_xy}
    cache[label] = ctx
    return ctx


def _map_parts_from_current(ctx: dict[str, Any], current_occ: str) -> dict[str, float]:
    df: pd.DataFrame = ctx["df"]
    xy: np.ndarray = ctx["xy"]
    max_dist: float = float(ctx["max_dist"])

    if df.empty or xy.size == 0:
        return {}

    occs = df["occupation"].astype(str).to_numpy()
    cur_mask = occs == str(current_occ)
    if not np.any(cur_mask):
        return {}

    cur_xy = xy[int(np.argmax(cur_mask))]
    if max_dist <= 0.0:
        return {str(o): 1.0 for o in occs}

    d = np.linalg.norm(xy - cur_xy, axis=1)
    prox = np.clip(1.0 - d / max_dist, 0.0, 1.0)
    return {str(o): float(p) for o, p in zip(occs, prox)}


def _hybrid_score(cos: float, map_part: float, w_cosine: float, w_map: float) -> float:
    cos01 = max(0.0, float(cos))
    return float(np.clip(100.0 * (float(w_cosine) * cos01 + float(w_map) * float(map_part)), 0.0, 100.0))


def _jaccard_overlap_from_matrix(current_occ: str, target_occ: str) -> float:
    try:
        a = mat.loc[str(current_occ)].astype(float).to_numpy()
        b = mat.loc[str(target_occ)].astype(float).to_numpy()
    except Exception:
        return 0.0
    a_nz = a > 0.0
    b_nz = b > 0.0
    inter = float(np.mean(a_nz & b_nz)) if a_nz.size else 0.0
    uni = float(np.mean(a_nz | b_nz)) if a_nz.size else 0.0
    return 0.0 if uni <= 0 else float(inter / uni)


def _percentile_rank_high_is_good(values: np.ndarray, x: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    less = float(np.sum(v < float(x)))
    eq = float(np.sum(v == float(x)))
    pct = 100.0 * (less + 0.5 * eq) / float(v.size)
    return float(np.clip(pct, 0.0, 100.0))


def _apply_overlap_gate(score: float, overlap_jaccard: float, min_overlap: float) -> tuple[float, float]:
    mo = float(max(0.0, min_overlap))
    ov = float(max(0.0, overlap_jaccard))
    if mo <= 0.0:
        return float(score), 1.0
    if ov >= mo:
        return float(score), 1.0
    mult = float(np.clip(ov / mo, 0.0, 1.0))
    return float(score) * mult, mult


def _get_score_distribution(
    *,
    current_occ: str,
    coords_label: str,
    coords_df: pd.DataFrame,
    use_idf: bool,
    w_cosine: float,
    w_map: float,
) -> dict[str, Any]:
    cache = st.session_state["__cache__"]["score_dist"]
    key = _cache_key(
        "dist",
        coords_label,
        bool(use_idf),
        str(current_occ),
        round(float(w_cosine), 6),
        round(float(w_map), 6),
    )
    if key in cache:
        return cache[key]

    _ensure_core_arrays()
    core = st.session_state["__core__"]
    occs: list[str] = core["occs"]
    occ_to_idx: dict[str, int] = core["occ_to_idx"]
    cos_mat: np.ndarray = core["cos_mat_idf"] if bool(use_idf) else core["cos_mat"]

    if str(current_occ) not in occ_to_idx:
        out = {"scores": np.array([], dtype=float), "by_occ": {}, "occs": occs}
        cache[key] = out
        return out

    ctx = _get_map_ctx(coords_df, coords_label)
    map_parts = _map_parts_from_current(ctx, str(current_occ))

    i = occ_to_idx[str(current_occ)]
    cos_row = cos_mat[i].copy()

    scores = np.zeros(len(occs), dtype=float)
    by_occ: dict[str, float] = {}

    for j, occ in enumerate(occs):
        if occ == str(current_occ):
            scores[j] = np.nan
            continue
        mp = float(map_parts.get(occ, 0.0))
        s = _hybrid_score(float(cos_row[j]), mp, float(w_cosine), float(w_map))
        scores[j] = float(s)
        by_occ[occ] = float(s)

    out = {"scores": scores[np.isfinite(scores)], "by_occ": by_occ, "occs": occs}
    cache[key] = out
    return out


def _recommend_similar_roles_fast(
    current_occ: str,
    coords_label: str,
    coords_df: pd.DataFrame,
    *,
    use_idf: bool,
    top_k: int,
    w_cosine: float,
    w_map: float,
    min_overlap: float,
) -> pd.DataFrame:
    dist = _get_score_distribution(
        current_occ=str(current_occ),
        coords_label=coords_label,
        coords_df=coords_df,
        use_idf=bool(use_idf),
        w_cosine=float(w_cosine),
        w_map=float(w_map),
    )
    by_occ: dict[str, float] = dist["by_occ"]
    scores_arr = dist["scores"]

    if not by_occ or scores_arr.size == 0:
        return pd.DataFrame(columns=["occupation", "raw_score_gated", "percentile_gated", "overlap_jaccard"])

    rows = []
    for occ, raw in by_occ.items():
        ov = _jaccard_overlap_from_matrix(str(current_occ), str(occ))
        raw_g, _ = _apply_overlap_gate(float(raw), float(ov), float(min_overlap))
        pct = _percentile_rank_high_is_good(scores_arr, float(raw))
        pct_g, _ = _apply_overlap_gate(float(pct), float(ov), float(min_overlap))
        rows.append((occ, float(raw_g), float(pct_g), float(ov)))

    df = pd.DataFrame(rows, columns=["occupation", "raw_score_gated", "percentile_gated", "overlap_jaccard"])
    df = df.sort_values("raw_score_gated", ascending=False).head(int(top_k)).reset_index(drop=True)
    return df


def _build_knn_graph(k_neighbors: int, use_idf: bool) -> dict[str, list[tuple[str, float]]]:
    cache = st.session_state["__cache__"]["knn_graph"]
    key = (int(k_neighbors), bool(use_idf))
    if key in cache:
        return cache[key]

    _ensure_core_arrays()
    core = st.session_state["__core__"]
    occs: list[str] = core["occs"]
    cos_mat: np.ndarray = core["cos_mat_idf"] if bool(use_idf) else core["cos_mat"]

    n = len(occs)
    kk = max(1, min(int(k_neighbors), max(1, n - 1)))

    graph: dict[str, list[tuple[str, float]]] = {}
    for i, occ in enumerate(occs):
        row = cos_mat[i].copy()
        row[i] = -np.inf

        cand = np.argpartition(row, -kk)[-kk:]
        cand = cand[np.argsort(row[cand])[::-1]]

        edges: list[tuple[str, float]] = []
        for j in cand:
            sim = float(row[int(j)])
            cost = 1.0 - max(0.0, sim)
            edges.append((occs[int(j)], float(np.clip(cost, 0.0, 1.0))))
        graph[occ] = edges

    cache[key] = graph
    return graph


def _find_pivot_path_fast(
    start_occ: str,
    target_occ: str,
    *,
    k_neighbors: int,
    max_steps: int,
    use_idf: bool,
) -> dict[str, Any]:
    start = str(start_occ)
    target = str(target_occ)

    if start == target:
        return {"path": [start], "reachable": True, "notes": "Start equals target."}

    graph = _build_knn_graph(int(k_neighbors), bool(use_idf))

    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str] = {}
    visited: set[str] = set()

    while True:
        candidates = [(node, d) for node, d in dist.items() if node not in visited]
        if not candidates:
            break
        node, dmin = min(candidates, key=lambda x: x[1])

        if node == target:
            break

        visited.add(node)

        for neigh, cost in graph.get(node, []):
            nd = float(dmin) + float(cost)
            if neigh not in dist or nd < dist[neigh]:
                dist[neigh] = nd
                prev[neigh] = node

    if target not in dist:
        return {
            "path": [start],
            "reachable": False,
            "notes": "Target not reachable in kNN graph. Increase connectivity (kNN neighbors) or allow more steps.",
        }

    path = [target]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()

    truncated = False
    if len(path) > int(max_steps):
        path = path[: int(max_steps)]
        truncated = True

    if truncated and path[-1] != target:
        return {
            "path": path,
            "reachable": False,
            "truncated": True,
            "k_neighbors": int(k_neighbors),
            "max_steps": int(max_steps),
            "step_costs": [],
            "total_cost": float(dist.get(target, np.nan)),
            "notes": "A path exists, but it was truncated by max_steps before reaching target. Increase max_steps.",
        }

    step_costs: list[float] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        cost = None
        for neigh, c in graph.get(u, []):
            if neigh == v:
                cost = c
                break
        step_costs.append(float(cost) if cost is not None else np.nan)

    return {
        "path": path,
        "reachable": True,
        "truncated": truncated,
        "k_neighbors": int(k_neighbors),
        "max_steps": int(max_steps),
        "step_costs": step_costs,
        "total_cost": float(dist.get(target, np.nan)),
        "notes": "Edge cost is 1 - max(0, cosine_sim). Lower cost ≈ easier transition.",
    }


def _robustness_monte_carlo(
    current_occ: str,
    target_occ: str,
    *,
    use_idf: bool,
    map_part: float,
    w_cosine: float,
    w_map: float,
    n_samples: int,
    noise_std: float,
    seed: int = 42,
) -> dict[str, Any]:
    _ensure_core_arrays()
    core = st.session_state["__core__"]
    occ_to_idx: dict[str, int] = core["occ_to_idx"]
    X = core["X"]
    idf = core["idf"]

    ci = occ_to_idx.get(str(current_occ))
    ti = occ_to_idx.get(str(target_occ))
    if ci is None or ti is None:
        return {
            "n_samples": int(n_samples),
            "noise_std": float(noise_std),
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "scores": [],
            "notes": "Missing occupations in matrix.",
        }

    a0 = X[int(ci)].astype(float)
    b0 = X[int(ti)].astype(float)
    rng = np.random.default_rng(int(seed))

    scores: list[float] = []
    for _ in range(int(n_samples)):
        a = np.clip(a0 + rng.normal(0.0, float(noise_std), size=a0.shape), 0.0, None)
        b = np.clip(b0 + rng.normal(0.0, float(noise_std), size=b0.shape), 0.0, None)

        if use_idf:
            aw = a * idf
            bw = b * idf
            denom = float(np.linalg.norm(aw) * np.linalg.norm(bw) + 1e-12)
            cos = float(np.dot(aw, bw) / denom)
        else:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            cos = float(np.dot(a, b) / denom)

        s = _hybrid_score(cos, float(map_part), float(w_cosine), float(w_map))
        scores.append(float(s))

    arr = np.array(scores, dtype=float)
    return {
        "n_samples": int(n_samples),
        "noise_std": float(noise_std),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "ci95_low": float(np.quantile(arr, 0.025)) if arr.size else float("nan"),
        "ci95_high": float(np.quantile(arr, 0.975)) if arr.size else float("nan"),
        "scores": scores,
        "notes": "Monte Carlo stability on skill vectors (Gaussian noise). Map part fixed (artifact).",
    }


# ============================================================
# Sidebar: UX-first Controls
# ============================================================
with st.sidebar:
    st.header("Controls")

    # Mode switch (uses segmented_control if available)
    if hasattr(st, "segmented_control"):
        view_mode = st.segmented_control("Mode", options=["Guided", "Research"], default="Guided")
    else:
        view_mode = st.radio("Mode", options=["Guided", "Research"], index=0)

    guided = view_mode == "Guided"

    st.caption("Pick a pivot → run analysis → explore deep dives as needed.")
    st.divider()

    with st.expander("1) Pivot selection", expanded=True):
        current = st.selectbox("Current occupation", options=occupations, index=0)

        default_target_idx = 1 if len(occupations) > 1 else 0
        target = st.selectbox("Target occupation", options=occupations, index=default_target_idx)

        if st.session_state.get("__target_override__"):
            target = st.session_state["__target_override__"]

        if current == target:
            st.warning("Pick a different target.")

    with st.expander("2) Scoring (recommended)", expanded=True):
        use_idf = st.toggle("Downweight common skills (IDF)", value=True)

        score_mode = st.radio(
            "Score shown in Overview",
            options=["Calibrated percentile (recommended)", "Raw hybrid (transparent)"],
            index=0,
        )

        min_overlap = st.slider("Plausibility gate (min Jaccard)", 0.00, 0.30, 0.08, 0.01)

        if guided:
            w_cosine = st.slider("Skill similarity weight", 0.0, 1.0, 0.65, 0.05)
            w_map = 1.0 - w_cosine
        else:
            w_cosine = st.slider("Skill similarity weight", 0.0, 1.0, 0.65, 0.05)
            w_map = st.slider("Map proximity weight", 0.0, 1.0, 0.35, 0.05)

        s = float(w_cosine + w_map)
        if s <= 0.0:
            w_cosine, w_map = 1.0, 0.0
        else:
            w_cosine, w_map = float(w_cosine / s), float(w_map / s)

    if guided:
        k_neighbors = int(st.session_state.get("__guided_k_neighbors__", 6))
        max_steps = int(st.session_state.get("__guided_max_steps__", 5))
        n_samples = 200
        noise_std = 0.05
        embedding_choice = "PCA"
        show_map = False
        color_by_cluster = False
    else:
        with st.expander("3) Planning & robustness", expanded=False):
            st.subheader("Path planning")
            k_neighbors = st.slider("kNN neighbors", 2, 20, 6, 1)
            max_steps = st.slider("Max steps", 2, 10, 5, 1)

            st.subheader("Robustness")
            n_samples = st.slider("Monte Carlo samples", 50, 800, 250, 50)
            noise_std = st.slider("Noise level (std)", 0.00, 0.20, 0.05, 0.01)

        with st.expander("4) Visualization", expanded=False):
            embed_options = ["PCA"]
            if _coords_is_valid(coords_umap):
                embed_options.append("UMAP")
            embedding_choice = st.selectbox("Embedding", options=embed_options, index=0)
            show_map = st.toggle("Enable Map tab", value=False)
            color_by_cluster = st.toggle("Color by clusters", value=bool(clusters))

    st.divider()
    run = st.button("🚀 Run pivot analysis", use_container_width=True)
    if run:
        st.session_state["__has_run__"] = True

    st.divider()
    st.subheader("Dataset snapshot")
    st.metric("Occupations", int(mat.shape[0]))
    st.metric("Skills", int(mat.shape[1]))


# ============================================================
# Empty-state (Guided onboarding)
# ============================================================
if not st.session_state["__has_run__"]:
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.subheader("What this prototype does")
        st.markdown(
            """
This is a **decision-support** prototype (not “just a dashboard”):

1) **Match score** — similarity in skill space (cosine) + optional map proximity  
2) **Explainability** — what transfers vs what’s missing  
3) **Stepping stones** — shortest path through similar roles (kNN graph)  
4) **Robustness** — uncertainty under noisy skill estimates  
5) **What-If lab** — sensitivity to scoring choices

👉 Choose **Current** + **Target** in the sidebar and click **Run pivot analysis**.
            """
        )

    with right:
        st.subheader("Why the score is shown as a percentile")
        st.markdown(
            """
Raw similarity can look high because many roles share generic skills.

So we show a **calibrated percentile** by default:
“How strong is this pivot compared to all other targets from your current role?”
            """
        )
        st.info("You can switch to Raw hybrid for full transparency.")

    st.stop()


# ============================================================
# Core computations (fast)
# ============================================================
coords, coords_label = _pick_coords(embedding_choice)
_ensure_core_arrays()

dist = _get_score_distribution(
    current_occ=str(current),
    coords_label=coords_label,
    coords_df=coords,
    use_idf=bool(use_idf),
    w_cosine=float(w_cosine),
    w_map=float(w_map),
)
scores_all: np.ndarray = dist["scores"]
raw_by_occ: dict[str, float] = dist["by_occ"]

raw_target = float(raw_by_occ.get(str(target), 0.0))
cal_target = _percentile_rank_high_is_good(scores_all, raw_target)

overlap_j = _jaccard_overlap_from_matrix(str(current), str(target))
raw_target_gated, gate_mult = _apply_overlap_gate(raw_target, overlap_j, float(min_overlap))
cal_target_gated, _ = _apply_overlap_gate(cal_target, overlap_j, float(min_overlap))

show_calibrated = score_mode.startswith("Calibrated")
match_score_display = float(cal_target_gated if show_calibrated else raw_target_gated)

core = st.session_state["__core__"]
occ_to_idx: dict[str, int] = core["occ_to_idx"]
cos_mat: np.ndarray = core["cos_mat_idf"] if bool(use_idf) else core["cos_mat"]
i_cur = occ_to_idx.get(str(current))
i_tgt = occ_to_idx.get(str(target))
cos_ct = float(cos_mat[int(i_cur), int(i_tgt)]) if (i_cur is not None and i_tgt is not None) else 0.0
cosine_score = float(np.clip(max(0.0, cos_ct) * 100.0, 0.0, 100.0))

map_ctx = _get_map_ctx(coords, coords_label)
map_parts_current = _map_parts_from_current(map_ctx, str(current))
map_ct = float(map_parts_current.get(str(target), 0.0))

gap_df = compute_gap_df(mat, current, target)
contrib = compute_skill_contributions(gap_df)
conf = compute_confidence_score(mat, art.pca_meta, current, target)
group_gap_df = compute_group_gap_df(mat, skill_taxonomy, group_meta, current, target)

path_out = _find_pivot_path_fast(
    current,
    target,
    k_neighbors=int(k_neighbors),
    max_steps=int(max_steps),
    use_idf=bool(use_idf),
)

rob_key = _cache_key(
    "rob",
    coords_label,
    bool(use_idf),
    current,
    target,
    round(float(w_cosine), 4),
    round(float(w_map), 4),
    int(n_samples),
    round(float(noise_std), 4),
)
rob_cached = st.session_state["__cache__"]["robustness"].get(rob_key)

# ============================================================
# Overview: High-signal + Next action
# ============================================================
st.subheader("Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Match", f"{match_score_display:.0f}/100")
m2.metric("Confidence", f"{conf['confidence_score']:.0f}/100")
m3.metric("Overlap (Jaccard)", f"{overlap_j:.3f}")
m4.metric("Gate multiplier", f"x{gate_mult:.2f}")

st.success(
    f"Embedding: **{coords_label}** • "
    f"Cosine ({'IDF' if use_idf else 'raw'}): **{cosine_score:.1f}/100** • "
    f"Map proximity: **{map_ct:.2f}** • "
    f"Primary score: **{'percentile' if show_calibrated else 'raw hybrid'}**"
)

# Next action guidance (Guided UX)
def _next_action_label(score: float, conf_score: float, gate: float) -> tuple[str, str]:
    # heuristic tiers (prototype-friendly)
    if conf_score < 45:
        return ("Data coverage is low", "Treat the score as uncertain. Use stepping stones + focus on skill evidence.")
    if gate < 0.7:
        return ("Pivot is likely a stretch", "Try a stepping-stone target from your neighborhood or increase connectivity.")
    if score >= 75:
        return ("Strong pivot candidate", "Start with a portfolio artifact and prep interview stories.")
    if score >= 55:
        return ("Promising with gaps", "Use the learning plan + one stepping-stone role if needed.")
    return ("Hard pivot", "Try stepping stones, tighten target role, and focus on 2–3 must-have skills first.")

title_na, body_na = _next_action_label(match_score_display, float(conf["confidence_score"]), float(gate_mult))
st.info(f"**Next action:** {title_na} — {body_na}")

if not guided:
    with st.expander("Under the hood (transparent)", expanded=False):
        st.write(f"- Raw hybrid (before gate): **{raw_target:.2f}**")
        st.write(f"- Percentile (before gate): **{cal_target:.2f}**")
        st.write(f"- Overlap gate: Jaccard={overlap_j:.3f}, min={min_overlap:.3f} → x{gate_mult:.2f}")
        st.code("raw_hybrid = 100 * (w_cosine * max(0, cosine) + w_map * map_proximity)")

# ============================================================
# Main area: Neighborhood + Path + Plan
# ============================================================
left, right = st.columns([1.05, 1.45], gap="large")

with left:
    st.subheader("Career neighborhood")
    st.caption("Closest roles to your current occupation, ranked by the same scoring settings.")

    rec_df = _recommend_similar_roles_fast(
        current,
        coords_label,
        coords,
        use_idf=bool(use_idf),
        top_k=10 if guided else 15,
        w_cosine=float(w_cosine),
        w_map=float(w_map),
        min_overlap=float(min_overlap),
    )

    if rec_df.empty:
        st.info("No neighborhood recommendations available.")
    else:
        pretty = rec_df.rename(
            columns={
                "raw_score_gated": "raw score (gated)",
                "percentile_gated": "percentile (gated)",
                "overlap_jaccard": "overlap",
            }
        )
        st.dataframe(
            pretty[["occupation", "percentile (gated)", "raw score (gated)", "overlap"]],
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Use a stepping-stone target:**")
        options = []
        label_to_occ = {}
        for _, row in rec_df.iterrows():
            occ = str(row["occupation"])
            label = f"{occ}  —  pct {row['percentile_gated']:.0f}/100 • raw {row['raw_score_gated']:.0f}/100 • ov {row['overlap_jaccard']:.2f}"
            options.append(label)
            label_to_occ[label] = occ

        pick = st.selectbox("Recommended targets", options=options, index=0, label_visibility="collapsed")
        c_apply, c_note = st.columns([1, 2])
        with c_apply:
            if st.button("Use as target", use_container_width=True):
                st.session_state["__target_override__"] = label_to_occ[pick]
                st.session_state["__has_run__"] = True
                st.rerun()
        with c_note:
            st.caption("Use a neighbor as an intermediate step if the direct pivot is hard/unreachable.")

    if clusters:
        st.divider()
        st.markdown("**Neighborhood theme (cluster)**")
        cur_cluster = clusters.get(str(current))
        if cur_cluster is None:
            st.info("No cluster label for the current role.")
        else:
            st.write(f"Cluster **{cur_cluster}**: {format_cluster_theme(cur_cluster, cluster_themes)}")

with right:
    st.subheader("Path + Learning plan")
    st.caption("Stepping stones + a 3-phase plan you can execute.")

    c1, c2 = st.columns([1.08, 0.92], gap="large")

    with c1:
        st.markdown("### Pivot path (stepping stones)")

        if not path_out.get("reachable"):
            st.warning(path_out.get("notes", "Target not reachable."))

            if guided:
                with st.popover("Fix path connectivity"):
                    st.caption("If the graph is too sparse, the target can be unreachable.")
                    k_try = st.slider("kNN neighbors", 2, 20, int(k_neighbors), 1, key="guided_k_try")
                    steps_try = st.slider("Max steps", 2, 10, int(max_steps), 1, key="guided_steps_try")
                    if st.button("Apply + recompute", use_container_width=True):
                        st.session_state["__guided_k_neighbors__"] = int(k_try)
                        st.session_state["__guided_max_steps__"] = int(steps_try)
                        st.session_state["__has_run__"] = True
                        st.rerun()
        else:
            path = path_out["path"]
            step_costs = path_out.get("step_costs", [])

            st.markdown("**Suggested route**")
            if len(path) == 1:
                st.write(f"Start/Target: **{path[0]}**")
            else:
                st.write(" → ".join([f"**{p}**" for p in path]))

            if step_costs:
                df_steps = pd.DataFrame({"from": path[:-1], "to": path[1:], "transition_cost": step_costs})
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

            st.caption("Lower transition cost ≈ easier step (based on cosine similarity edges).")

    with c2:
        st.markdown("### Learning plan (3 phases)")
        st.caption("AI Coach tries OpenAI if available; otherwise uses a deterministic offline fallback. Output is **English-only**.")

        with st.expander("🤖 Generate plan", expanded=True):
            if st.button("Generate learning plan", key="ai_generate_plan", use_container_width=True):
                md = generate_learning_plan_markdown(
                    current_role=str(current),
                    target_role=str(target),
                    gap_df=gap_df,
                    language="en",           # ENGLISH ONLY requirement
                    model="gpt-4o-mini",
                    max_missing=6,
                    prefer_online=True,
                )
                st.session_state["ai_learning_plan_md"] = md

            if "ai_learning_plan_md" in st.session_state and str(st.session_state["ai_learning_plan_md"]).strip():
                st.markdown(st.session_state["ai_learning_plan_md"])
            else:
                st.info("Click **Generate learning plan** to create a 3-phase plan (online if possible, otherwise offline).")


# ============================================================
# Deep dives
# ============================================================
st.divider()

# --- Compact explain for Guided; Full tabs for Research
if guided:
    st.subheader("Explain (compact)")
    st.caption("High-signal view: what transfers vs what blocks this pivot.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("**Top transferable skills**")
        top_transfer = gap_df.copy()
        top_transfer["overlap"] = np.minimum(top_transfer["current_importance"], top_transfer["target_importance"])
        top_transfer = top_transfer.sort_values("overlap", ascending=False).head(12)
        st.dataframe(
            top_transfer[["skill", "current_importance", "target_importance", "overlap"]],
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("**Top missing skills**")
        top_missing = gap_df[gap_df["gap"] > 0].sort_values(["gap", "target_importance"], ascending=False).head(12)
        if top_missing.empty:
            st.success("No missing skills detected in this dataset.")
        else:
            st.dataframe(
                top_missing[["skill", "current_importance", "target_importance", "gap"]],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Export (CSV)", expanded=False):
        export_df = gap_df.copy()
        export_df.insert(0, "current_occupation", current)
        export_df.insert(1, "target_occupation", target)
        export_df.insert(2, "match_score_display", round(match_score_display, 2))
        export_df.insert(3, "score_mode", "percentile" if show_calibrated else "raw_hybrid")
        export_df.insert(4, "raw_hybrid_score_gated", round(raw_target_gated, 4))
        export_df.insert(5, "calibrated_percentile_gated", round(cal_target_gated, 4))
        export_df.insert(6, "raw_hybrid_score_before_gate", round(raw_target, 4))
        export_df.insert(7, "calibrated_percentile_before_gate", round(cal_target, 4))
        export_df.insert(8, "overlap_jaccard", round(overlap_j, 6))
        export_df.insert(9, "overlap_gate_multiplier", round(gate_mult, 6))
        export_df.insert(10, "min_overlap", round(float(min_overlap), 6))
        export_df.insert(11, "cosine_score", round(cosine_score, 4))
        export_df.insert(12, "use_idf", bool(use_idf))
        export_df.insert(13, "map_embedding", coords_label)
        export_df.insert(14, "map_proximity", round(map_ct, 6))
        export_df.insert(15, "w_cosine", round(float(w_cosine), 6))
        export_df.insert(16, "w_map", round(float(w_map), 6))

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download pivot CSV",
            data=csv_bytes,
            file_name=f"pivot_{current}_to_{target}.csv".replace(" ", "_").lower(),
            mime="text/csv",
            use_container_width=True,
        )

else:
    tabs = st.tabs(
        [
            "Explain",
            "Skill Groups",
            "Robustness",
            "Decision Brief",
            "What-If Lab",
            "Map",
            "Export",
        ]
    )

    # ---- Explain
    with tabs[0]:
        st.subheader("Explainability")
        st.caption("What drives the match, what blocks it, and what looks surplus.")

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.markdown("**Match drivers (overlap)**")
            st.dataframe(
                contrib["match_drivers"][
                    ["skill", "current_importance", "target_importance", "overlap", "match_driver_score"]
                ],
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
    with tabs[1]:
        st.subheader("Skill Groups (taxonomy view)")
        if group_gap_df.empty:
            st.info("No taxonomy artifacts loaded.")
        else:
            left2, right2 = st.columns([1.15, 1.0], gap="large")

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
                    go.Scatterpolar(
                        r=group_gap_df["current_importance"],
                        theta=categories,
                        fill="toself",
                        name="Current",
                    )
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=group_gap_df["target_importance"],
                        theta=categories,
                        fill="toself",
                        name="Target",
                    )
                )
                fig.update_layout(height=420, polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Drilldown: missing skills inside one group")
                chosen_group = st.selectbox("Group", options=categories, index=0)
                miss_in_group = filter_missing_skills_by_group(gap_df, skill_taxonomy, chosen_group, top_n=25)
                if miss_in_group.empty:
                    st.success("No missing skills in this group (for this pivot).")
                else:
                    st.dataframe(
                        miss_in_group[["skill", "current_importance", "target_importance", "gap"]],
                        use_container_width=True,
                        hide_index=True,
                    )

    # ---- Robustness
    with tabs[2]:
        st.subheader("Robustness (Monte Carlo)")
        st.caption("Runs on demand and caches results for this pivot + config. Uses raw hybrid scores (before percentile calibration).")

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if st.button("Run robustness simulation", key="run_robustness", use_container_width=True):
                with st.spinner("Running Monte Carlo..."):
                    rob = _robustness_monte_carlo(
                        current,
                        target,
                        use_idf=bool(use_idf),
                        map_part=float(map_ct),
                        w_cosine=float(w_cosine),
                        w_map=float(w_map),
                        n_samples=int(n_samples),
                        noise_std=float(noise_std),
                        seed=42,
                    )
                    st.session_state["__cache__"]["robustness"][rob_key] = rob

        rob_cached_local = st.session_state["__cache__"]["robustness"].get(rob_key)
        if not isinstance(rob_cached_local, dict):
            st.info("Not computed yet for this pivot/config.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean (raw hybrid)", f"{rob_cached_local['mean']:.1f}")
            m2.metric("Std dev", f"{rob_cached_local['std']:.1f}")
            m3.metric("95% CI", f"[{rob_cached_local['ci95_low']:.1f}, {rob_cached_local['ci95_high']:.1f}]")

            fig = px.histogram(pd.DataFrame({"score": rob_cached_local["scores"]}), x="score", nbins=24)
            fig.update_layout(height=420, title="Raw hybrid score distribution under noise")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Decision Brief (on-demand, but optimized)
    with tabs[3]:
        st.subheader("Decision Brief")
        st.caption("A quick decision-grade shortlist: best alternatives, effort estimates, and a Pareto view.")

        brief_k = st.slider("Shortlist size", 5, 30, 12, 1)
        brief_mode = st.radio(
            "Objective",
            ["Max score, min effort (Pareto)", "Max score only"],
            index=0,
            horizontal=True,
        )

        brief_key = _cache_key(
            "brief",
            coords_label,
            bool(use_idf),
            str(current),
            round(float(w_cosine), 4),
            round(float(w_map), 4),
            round(float(min_overlap), 4),
            int(brief_k),
            brief_mode,
        )

        if st.button("Compute decision brief", use_container_width=True):
            with st.spinner("Computing shortlist..."):
                # 1) candidates from full distribution
                dist2 = _get_score_distribution(
                    current_occ=str(current),
                    coords_label=coords_label,
                    coords_df=coords,
                    use_idf=bool(use_idf),
                    w_cosine=float(w_cosine),
                    w_map=float(w_map),
                )
                by_occ2 = dist2["by_occ"]
                scores_arr2 = dist2["scores"]

                rows = []
                for occ, raw in by_occ2.items():
                    if occ == str(current):
                        continue
                    ov = _jaccard_overlap_from_matrix(str(current), str(occ))
                    raw_g, _ = _apply_overlap_gate(float(raw), float(ov), float(min_overlap))
                    pct = _percentile_rank_high_is_good(scores_arr2, float(raw))
                    pct_g, _ = _apply_overlap_gate(float(pct), float(ov), float(min_overlap))
                    rows.append((occ, float(raw_g), float(pct_g), float(ov)))

                cand = pd.DataFrame(rows, columns=["occupation", "raw_score_gated", "percentile_gated", "overlap_jaccard"])
                cand = cand.sort_values("raw_score_gated", ascending=False).head(int(max(50, brief_k * 3))).reset_index(drop=True)

                # 2) effort metrics (from provided helper)
                effort = compute_effort_metrics(gap_df, top_n=20)
                # effort is for current->target; for alternates we recompute quickly for top candidates only
                effort_rows = []
                for occ in cand["occupation"].astype(str).tolist()[: int(brief_k)]:
                    gdf = compute_gap_df(mat, current, occ)
                    em = compute_effort_metrics(gdf, top_n=20)
                    effort_rows.append(
                        (
                            occ,
                            float(em.get("effort_score", np.nan)),
                            float(em.get("missing_count", np.nan)),
                            float(em.get("avg_gap", np.nan)),
                        )
                    )
                eff_df = pd.DataFrame(effort_rows, columns=["occupation", "effort_score", "missing_count", "avg_gap"])

                out = cand.merge(eff_df, on="occupation", how="left")
                out = out.sort_values(["raw_score_gated"], ascending=False).head(int(brief_k)).reset_index(drop=True)

                # 3) Pareto frontier (maximize score, minimize effort_score)
                # pareto_frontier_flags expects arrays, we’ll guard NaNs
                score_arr = out["raw_score_gated"].to_numpy(dtype=float)
                effort_arr = out["effort_score"].to_numpy(dtype=float)
                eff_safe = np.where(np.isfinite(effort_arr), effort_arr, np.nanmax(effort_arr[np.isfinite(effort_arr)]) + 1.0)
                pareto = pareto_frontier_flags(score_arr, eff_safe, maximize_x=True, minimize_y=True)
                out["pareto"] = pareto

                st.session_state["__cache__"]["decision_brief"][brief_key] = {
                    "df": out,
                    "notes": "Pareto: high score + low effort. Effort is estimated from missing skill gaps.",
                }

        payload = st.session_state["__cache__"]["decision_brief"].get(brief_key)
        if not payload:
            st.info("Click **Compute decision brief** to generate a shortlist.")
        else:
            dfb = payload["df"]
            st.success(payload.get("notes", ""))

            st.dataframe(
                dfb[["occupation", "raw_score_gated", "percentile_gated", "effort_score", "missing_count", "pareto"]],
                use_container_width=True,
                hide_index=True,
            )

            fig = px.scatter(
                dfb,
                x="effort_score",
                y="raw_score_gated",
                hover_name="occupation",
                symbol="pareto",
                title="Score vs Effort (Pareto highlighted)",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Counterfactual: which skills give the biggest uplift?")
            st.caption("Greedy uplift: ‘If you improved a small set of missing skills, which ones would increase the score most?’")
            if st.button("Run counterfactual uplift", use_container_width=True):
                with st.spinner("Computing uplift..."):
                    uplift_df = counterfactual_uplift_greedy(mat, current, target, top_n=10)
                    st.session_state["__cf_uplift__"] = uplift_df

            if "__cf_uplift__" in st.session_state and isinstance(st.session_state["__cf_uplift__"], pd.DataFrame):
                st.dataframe(st.session_state["__cf_uplift__"], use_container_width=True, hide_index=True)

    # ---- What-If Lab
    with tabs[4]:
        st.subheader("What-If Lab (sensitivity)")
        st.caption("Test how stable the decision is across scoring choices. Keep the grid small for demos.")

        colA, colB, colC = st.columns(3)
        with colA:
            w_grid = st.multiselect(
                "Skill weight values (w_cosine)",
                options=[0.2, 0.35, 0.5, 0.65, 0.8, 0.9],
                default=[0.5, 0.65, 0.8],
            )
        with colB:
            gate_grid = st.multiselect(
                "Min overlap gate values",
                options=[0.00, 0.05, 0.08, 0.12, 0.18],
                default=[0.05, 0.08, 0.12],
            )
        with colC:
            topN = st.slider("Track top-N neighbors", 3, 15, 8, 1)

        what_key = _cache_key(
            "whatif",
            coords_label,
            bool(use_idf),
            str(current),
            str(target),
            tuple(w_grid),
            tuple(gate_grid),
            int(topN),
        )

        if st.button("Run What-If sweep", use_container_width=True):
            with st.spinner("Sweeping…"):
                rows = []
                top_hit = 0
                for wc in w_grid:
                    wm = 1.0 - float(wc)
                    for mo in gate_grid:
                        dist3 = _get_score_distribution(
                            current_occ=str(current),
                            coords_label=coords_label,
                            coords_df=coords,
                            use_idf=bool(use_idf),
                            w_cosine=float(wc),
                            w_map=float(wm),
                        )
                        by_occ3 = dist3["by_occ"]
                        scores_arr3 = dist3["scores"]

                        # score for target
                        raw_t = float(by_occ3.get(str(target), 0.0))
                        ov_t = _jaccard_overlap_from_matrix(str(current), str(target))
                        raw_t_g, _ = _apply_overlap_gate(raw_t, ov_t, float(mo))
                        pct_t = _percentile_rank_high_is_good(scores_arr3, raw_t)
                        pct_t_g, _ = _apply_overlap_gate(pct_t, ov_t, float(mo))

                        # topN set stability
                        # compute gated raw for all, then topN
                        tmp = []
                        for occ, raw in by_occ3.items():
                            if occ == str(current):
                                continue
                            ov = _jaccard_overlap_from_matrix(str(current), str(occ))
                            raw_g, _ = _apply_overlap_gate(float(raw), float(ov), float(mo))
                            tmp.append((occ, raw_g))
                        tmp.sort(key=lambda x: x[1], reverse=True)
                        top_set = {x[0] for x in tmp[: int(topN)]}
                        if str(target) in top_set:
                            top_hit += 1

                        rows.append(
                            {
                                "w_cosine": float(wc),
                                "w_map": float(wm),
                                "min_overlap": float(mo),
                                "target_raw_gated": float(raw_t_g),
                                "target_pct_gated": float(pct_t_g),
                                "target_in_topN": bool(str(target) in top_set),
                            }
                        )

                sweep_df = pd.DataFrame(rows)
                stability = {
                    "runs": int(len(rows)),
                    "target_in_topN_rate": float(top_hit / max(1, len(rows))),
                }
                st.session_state["__cache__"]["whatif"][what_key] = {"df": sweep_df, "stability": stability}

        wpayload = st.session_state["__cache__"]["whatif"].get(what_key)
        if not wpayload:
            st.info("Choose a grid and click **Run What-If sweep**.")
        else:
            sdf = wpayload["df"]
            stab = wpayload["stability"]

            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.metric("Runs", stab["runs"])
            with c2:
                st.metric("Target in top-N rate", f"{100.0 * stab['target_in_topN_rate']:.0f}%")

            st.dataframe(sdf, use_container_width=True, hide_index=True)

            fig = px.line(
                sdf.sort_values(["min_overlap", "w_cosine"]),
                x="w_cosine",
                y="target_pct_gated",
                color="min_overlap",
                markers=True,
                title="Target percentile (gated) across weights & gate",
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Map
    with tabs[5]:
        st.subheader(f"Map ({coords_label})")
        st.caption("For intuition only — axes are not semantic.")

        if not show_map:
            st.info("Enable **Map tab** in the sidebar to display this.")
        else:
            mdf = coords.copy()
            mdf["occupation"] = mdf["occupation"].astype(str)

            # coloring
            if color_by_cluster and clusters:
                mdf["cluster"] = mdf["occupation"].map(lambda x: clusters.get(x, -1))
                color_col = "cluster"
            else:
                mdf["cluster"] = "role"
                color_col = "cluster"

            fig = px.scatter(
                mdf,
                x="x",
                y="y",
                color=color_col,
                hover_name="occupation",
                title=f"Embedding: {coords_label}",
            )

            # highlight current + target + path
            def _add_marker(occ: str, name: str):
                xy = map_ctx["occ_to_xy"].get(str(occ))
                if xy:
                    fig.add_trace(
                        go.Scatter(
                            x=[xy[0]],
                            y=[xy[1]],
                            mode="markers+text",
                            text=[name],
                            textposition="top center",
                            name=name,
                        )
                    )

            _add_marker(current, "Current")
            _add_marker(target, "Target")

            if path_out.get("reachable") and len(path_out.get("path", [])) >= 2:
                xs, ys = [], []
                for p in path_out["path"]:
                    xy = map_ctx["occ_to_xy"].get(str(p))
                    if xy:
                        xs.append(xy[0])
                        ys.append(xy[1])
                if len(xs) >= 2:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Path"))

            fig.update_layout(height=560, legend_title_text="Legend")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Export
    with tabs[6]:
        st.subheader("Export")
        export_df = gap_df.copy()
        export_df.insert(0, "current_occupation", current)
        export_df.insert(1, "target_occupation", target)
        export_df.insert(2, "match_score_display", round(match_score_display, 2))
        export_df.insert(3, "score_mode", "percentile" if show_calibrated else "raw_hybrid")
        export_df.insert(4, "raw_hybrid_score_gated", round(raw_target_gated, 4))
        export_df.insert(5, "calibrated_percentile_gated", round(cal_target_gated, 4))
        export_df.insert(6, "raw_hybrid_score_before_gate", round(raw_target, 4))
        export_df.insert(7, "calibrated_percentile_before_gate", round(cal_target, 4))
        export_df.insert(8, "overlap_jaccard", round(overlap_j, 6))
        export_df.insert(9, "overlap_gate_multiplier", round(gate_mult, 6))
        export_df.insert(10, "min_overlap", round(float(min_overlap), 6))
        export_df.insert(11, "cosine_score", round(cosine_score, 4))
        export_df.insert(12, "use_idf", bool(use_idf))
        export_df.insert(13, "map_embedding", coords_label)
        export_df.insert(14, "map_proximity", round(map_ct, 6))
        export_df.insert(15, "w_cosine", round(float(w_cosine), 6))
        export_df.insert(16, "w_map", round(float(w_map), 6))

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download pivot CSV",
            data=csv_bytes,
            file_name=f"pivot_{current}_to_{target}.csv".replace(" ", "_").lower(),
            mime="text/csv",
            use_container_width=True,
        )


