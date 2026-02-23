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
# Page config
# ============================================================
st.set_page_config(page_title="Career Pivot Simulator", page_icon="🧭", layout="wide")

st.title("🧭 Career Pivot Simulator")
st.caption(
    "Decision-support prototype for career pivots: matching + explainability + planning + robustness + what-if sensitivity. "
    "Artifacts are precomputed offline and loaded at runtime (deploy-ready)."
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

# Optional UMAP
coords_umap: pd.DataFrame = (
    art.umap_coords
    if hasattr(art, "umap_coords") and isinstance(art.umap_coords, pd.DataFrame)
    else pd.DataFrame(columns=["occupation", "x", "y"])
)

# Optional “overkill” artifacts
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

# All runtime caches in one place (keeps things predictable)
if "__cache__" not in st.session_state:
    st.session_state["__cache__"] = {
        "map_ctx": {},        # label -> context dict
        "knn_graph": {},      # (k, use_idf) -> graph
        "robustness": {},     # key -> rob dict
        "decision_brief": {}, # key -> df
        "whatif": {},         # key -> (sweep_df, stab_dict)
        "score_dist": {},     # (coords_label, use_idf, current, w_cosine, w_map) -> dict
    }

# ============================================================
# Helpers (robust, fast, deterministic)
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
      - normalized X (unweighted) + cosine matrix
      - IDF weights + normalized weighted X + cosine matrix
      - occupation list + index map
    """
    if "__core__" in st.session_state:
        return

    X = mat.to_numpy(dtype=float)
    occs = mat.index.astype(str).tolist()
    occ_to_idx = {o: i for i, o in enumerate(occs)}

    # --- Unweighted cosine
    norms = np.linalg.norm(X, axis=1)
    norms_safe = np.where(norms == 0.0, 1.0, norms)
    Xn = X / norms_safe[:, None]
    cos_mat = np.clip(Xn @ Xn.T, -1.0, 1.0)

    # --- IDF weights: downweight “common” skills
    df = np.sum(X > 0.0, axis=0).astype(float)
    N = float(X.shape[0])
    idf = np.log((N + 1.0) / (1.0 + df)) + 1.0
    idf = np.clip(idf, 1.0, None)

    # --- Weighted cosine
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
    """
    Build a lightweight map context and cache in session_state (NOT st.cache_data).
    Avoids hashing huge DataFrames and keeps reruns snappy.
    """
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
    """
    Compute proximity (0..1) from current occupation to every other occupation on the map.
    Vectorized, fast.
    """
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
    """
    Jaccard overlap on (skill > 0) supports, using the matrix.
    """
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
    """
    Percentile rank in [0,100] where higher values are better.
    Uses "average rank" for ties:
      pct = 100 * (count_less + 0.5*count_equal) / n
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    less = float(np.sum(v < float(x)))
    eq = float(np.sum(v == float(x)))
    pct = 100.0 * (less + 0.5 * eq) / float(v.size)
    return float(np.clip(pct, 0.0, 100.0))


def _apply_overlap_gate(score: float, overlap_jaccard: float, min_overlap: float) -> tuple[float, float]:
    """
    Soft plausibility gate:
      - if overlap < min_overlap: multiply score by overlap/min_overlap
      - returns (gated_score, multiplier)
    """
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
    """
    For a given current occupation and config, compute raw hybrid scores for ALL targets.
    Cached in session_state because it's quick but we want it once per rerun.
    """
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
) -> pd.DataFrame:
    """
    Recommend top-K similar roles using cached raw hybrid scores.
    """
    dist = _get_score_distribution(
        current_occ=str(current_occ),
        coords_label=coords_label,
        coords_df=coords_df,
        use_idf=bool(use_idf),
        w_cosine=float(w_cosine),
        w_map=float(w_map),
    )
    by_occ: dict[str, float] = dist["by_occ"]

    if not by_occ:
        return pd.DataFrame(columns=["occupation", "raw_score", "percentile"])

    scores_arr = dist["scores"]
    rows = []
    for occ, raw in by_occ.items():
        pct = _percentile_rank_high_is_good(scores_arr, float(raw))
        rows.append((occ, float(raw), float(pct)))

    df = pd.DataFrame(rows, columns=["occupation", "raw_score", "percentile"])
    df = df.sort_values("raw_score", ascending=False).head(int(top_k)).reset_index(drop=True)
    return df


def _build_knn_graph(k_neighbors: int, use_idf: bool) -> dict[str, list[tuple[str, float]]]:
    """
    Build kNN graph using cosine matrix. Cached in session_state by (k, use_idf).
    Edge cost = 1 - max(0, cosine).
    """
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
    """
    Dijkstra on cached kNN graph.
    Truncates path if longer than max_steps.
    """
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
            "notes": "Target not reachable in kNN graph. Increase k_neighbors.",
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
            "notes": "Path exists but was truncated by max_steps before reaching target. Increase max_steps.",
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
        "notes": "Edge cost is 1 - max(0, cosine_sim). Lower cost means easier transition.",
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
    """
    Monte Carlo robustness on skill vectors.
    Uses either unweighted or IDF-weighted cosine internally.
    """
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
# Sidebar: Controls
# ============================================================
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

    with st.expander("2) Scoring knobs (recommended defaults)", expanded=True):
        st.caption("We show a calibrated decision score by default (percentile). Raw score stays available for transparency.")
        w_cosine = st.slider("Skill similarity weight", 0.0, 1.0, 0.65, 0.05)
        w_map = st.slider("Map proximity weight", 0.0, 1.0, 0.35, 0.05)
        s = float(w_cosine + w_map)
        if s <= 0:
            w_cosine, w_map = 1.0, 0.0
        else:
            w_cosine, w_map = float(w_cosine / s), float(w_map / s)

        use_idf = st.toggle("Downweight common skills (IDF)", value=True)
        score_mode = st.radio(
            "Score mode (shown in Overview)",
            options=["Calibrated percentile (recommended)", "Raw hybrid"],
            index=0,
        )

        st.caption("Plausibility gate: if two roles share too few non-zero skills, the final score is reduced.")
        min_overlap = st.slider("Min overlap (Jaccard) for full score", 0.00, 0.30, 0.08, 0.01)

    with st.expander("3) Planning & robustness knobs", expanded=False):
        st.subheader("Path planning (fast)")
        k_neighbors = st.slider("kNN neighbors", 2, 12, 5, 1)
        max_steps = st.slider("Max steps", 2, 8, 4, 1)

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

        if embedding_choice == "UMAP" and not _coords_is_valid(coords_umap):
            st.warning("UMAP not available (missing artifacts). Falling back to PCA.")

    st.divider()
    run = st.button("🚀 Run pivot analysis", use_container_width=True)
    if run:
        st.session_state["__has_run__"] = True

    st.divider()
    st.subheader("Dataset snapshot")
    st.metric("Occupations", mat.shape[0])
    st.metric("Skills", mat.shape[1])

# ============================================================
# Empty-state
# ============================================================
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
    with right:
        st.subheader("Why scores can look “too high”")
        st.markdown(
            """
Raw similarity scores are **not calibrated**: many roles share generic skills.

✅ We therefore show a **calibrated score (percentile)** by default:
“How strong is this target compared to all other targets from your current role?”
            """
        )
        st.info("You can switch back to raw hybrid score in the sidebar for full transparency.")
    st.stop()

# ============================================================
# Core computations (FAST)
# ============================================================
coords, coords_label = _pick_coords(embedding_choice)
_ensure_core_arrays()

# distribution of raw hybrid scores for this current role + config
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
cal_target = _percentile_rank_high_is_good(scores_all, raw_target)  # 0..100 percentile

# plausibility overlap gate (soft)
overlap_j = _jaccard_overlap_from_matrix(str(current), str(target))
raw_target_gated, gate_mult = _apply_overlap_gate(raw_target, overlap_j, float(min_overlap))
cal_target_gated, _ = _apply_overlap_gate(cal_target, overlap_j, float(min_overlap))

# Choose what we show as primary match score
show_calibrated = score_mode.startswith("Calibrated")
match_score_display = float(cal_target_gated if show_calibrated else raw_target_gated)

# Keep a transparent cosine-only score for explanation (0..100)
# We can approximate cosine-only using hybrid with w_map=0 (same precomputed cos + clip)
# But we already have raw hybrid; cosine-only is useful so we compute directly from cos matrix row:
core = st.session_state["__core__"]
occ_to_idx: dict[str, int] = core["occ_to_idx"]
cos_mat: np.ndarray = core["cos_mat_idf"] if bool(use_idf) else core["cos_mat"]
i_cur = occ_to_idx.get(str(current))
i_tgt = occ_to_idx.get(str(target))
cos_ct = float(cos_mat[int(i_cur), int(i_tgt)]) if (i_cur is not None and i_tgt is not None) else 0.0
cosine_score = float(np.clip(max(0.0, cos_ct) * 100.0, 0.0, 100.0))

# Map proximity for transparency
map_ctx = _get_map_ctx(coords, coords_label)
map_parts_current = _map_parts_from_current(map_ctx, str(current))
map_ct = float(map_parts_current.get(str(target), 0.0))

# Explainability artifacts
gap_df = compute_gap_df(mat, current, target)
contrib = compute_skill_contributions(gap_df)
conf = compute_confidence_score(mat, art.pca_meta, current, target)
group_gap_df = compute_group_gap_df(mat, skill_taxonomy, group_meta, current, target)

# Path planning (fast)
path_out = _find_pivot_path_fast(
    current,
    target,
    k_neighbors=int(k_neighbors),
    max_steps=int(max_steps),
    use_idf=bool(use_idf),
)

# ============================================================
# Overview (robustness is on-demand)
# ============================================================
st.subheader("Overview")

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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Match score", f"{match_score_display:.0f}/100")
m2.metric("Confidence (coverage)", f"{conf['confidence_score']:.0f}/100")
if isinstance(rob_cached, dict):
    m3.metric("Robust mean", f"{rob_cached['mean']:.1f}")
    m4.metric("Robust 95% CI width", f"{(rob_cached['ci95_high'] - rob_cached['ci95_low']):.1f}")
else:
    m3.metric("Robust mean", "—")
    m4.metric("Robust 95% CI width", "—")

st.success(
    f"Embedding: **{coords_label}** • "
    f"Cosine ({'IDF' if use_idf else 'raw'}): **{cosine_score:.1f}/100** • "
    f"Map proximity: **{map_ct:.2f}** • "
    f"Overlap gate x{gate_mult:.2f}"
)

with st.expander("Under the hood (transparent + calibrated)", expanded=False):
    st.markdown("### What’s shown as the main score?")
    if show_calibrated:
        st.write(
            f"**Calibrated percentile score (recommended)**: your target is at about the **{cal_target:.0f}th percentile** "
            "among all possible targets from your current role (then gated by overlap)."
        )
    else:
        st.write("**Raw hybrid score**: direct weighted combination of cosine similarity and map proximity (then gated by overlap).")

    st.markdown("### Components (raw)")
    st.write(f"- Raw hybrid score (before gate): **{raw_target:.1f}/100**")
    st.write(f"- Calibrated percentile (before gate): **{cal_target:.1f}/100**")
    st.write(f"- Gate multiplier from overlap: **x{gate_mult:.2f}** (Jaccard overlap = {overlap_j:.3f}, min={min_overlap:.3f})")

    st.markdown("### Model ingredients")
    st.write(f"- Cosine-only score: **{cosine_score:.1f}/100** ({'IDF-weighted' if use_idf else 'raw'})")
    st.write(f"- Map proximity: **{map_ct:.2f}** (from {coords_label} coords)")
    st.write(f"- Weights: `w_cosine={w_cosine:.2f}`, `w_map={w_map:.2f}`")
    st.code("raw_hybrid = 100 * (w_cosine * max(0, cosine) + w_map * map_proximity)")

    st.markdown("### Why calibration helps")
    st.write(
        "Raw similarity is not a probability and not calibrated. "
        "Percentile converts it into a decision-grade signal: how strong this target is relative to alternatives."
    )

    st.markdown("### Quick skill sanity check")
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

# ============================================================
# Main: Neighborhood + Plan
# ============================================================
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.subheader("Career neighborhood (actionable)")
    st.caption("Closest roles to your current occupation + one-click stepping-stones.")
    st.caption("Table shows raw hybrid for ranking + percentile for interpretability.")

    rec_df = _recommend_similar_roles_fast(
        current,
        coords_label,
        coords,
        use_idf=bool(use_idf),
        top_k=8,
        w_cosine=float(w_cosine),
        w_map=float(w_map),
    )

    # Apply overlap gate to displayed columns for recs (optional but consistent)
    if not rec_df.empty:
        gated_scores = []
        gated_pct = []
        for occ, raw, pct in rec_df[["occupation", "raw_score", "percentile"]].itertuples(index=False, name=None):
            ov = _jaccard_overlap_from_matrix(str(current), str(occ))
            raw_g, _m = _apply_overlap_gate(float(raw), ov, float(min_overlap))
            pct_g, _m2 = _apply_overlap_gate(float(pct), ov, float(min_overlap))
            gated_scores.append(raw_g)
            gated_pct.append(pct_g)
        rec_df["raw_score_gated"] = gated_scores
        rec_df["percentile_gated"] = gated_pct

    show_cols = ["occupation", "raw_score_gated", "percentile_gated"]
    pretty = rec_df.copy()
    if not pretty.empty:
        pretty = pretty.rename(
            columns={
                "raw_score_gated": "raw_score (gated)",
                "percentile_gated": "percentile (gated)",
            }
        )
        st.dataframe(pretty[["occupation", "raw_score (gated)", "percentile (gated)"]], use_container_width=True, hide_index=True)
    else:
        st.info("No recommendations available.")

    st.markdown("**Use a recommended role as new target:**")
    if not rec_df.empty:
        for _, row in rec_df.iterrows():
            occ = str(row["occupation"])
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{occ}** — raw {row['raw_score_gated']:.0f}/100 • pct {row['percentile_gated']:.0f}/100")
            with c2:
                if st.button("Use", key=f"use_{occ}"):
                    st.session_state["__target_override__"] = occ
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
    st.caption("Route + learning phases. Heavy analyses are on-demand in tabs below.")

    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.markdown("### Pivot Path Finder")
        if not path_out.get("rseachable"):
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
                df_steps = pd.DataFrame(
                    {"from": path[:-1], "to": path[1:], "transition_cost": step_costs}
                )
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Learning plan (3 phases)")
        st.caption("Single pipeline: AI Coach (OpenAI if available) with deterministic offline fallback.")

        with st.expander("🤖 Learning Plan (AI Coach + Offline Fallback)", expanded=False):
            st.caption(
                "On click: tries OpenAI. If unavailable (no key/quota/network/dependency), returns a deterministic offline plan."
            )

            if st.button("Generate learning plan", key="ai_generate_plan"):
                md = generate_learning_plan_markdown(
                    current_role=str(current),
                    target_role=str(target),
                    gap_df=gap_df,
                    language="en",
                    model="gpt-4o-mini",
                    max_missing=6,
                    prefer_online=True,
                )
                st.session_state["ai_learning_plan_md"] = md

            if "ai_learning_plan_md" in st.session_state and str(st.session_state["ai_learning_plan_md"]).strip():
                st.markdown(st.session_state["ai_learning_plan_md"])
            else:
                st.info(
                    "Click **Generate learning plan** to create the plan (online if possible, otherwise offline fallback)."
                )
                

# ============================================================
# Tabs (deep dives) — heavy tabs on-demand
# ============================================================
st.divider()
tabs = st.tabs(
    [
        "Explain",
        "Skill Groups",
        "Robustness (on-demand)",
        "Decision Brief (on-demand)",
        "What-If Lab (on-demand)",
        "Map (advanced)",
        "Export",
    ]
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
    st.caption("Runs Monte Carlo for this pivot/config and caches the raw-hybrid result (before calibration).")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Run robustness (Monte Carlo)", key="run_robustness"):
            with st.spinner("Running Monte Carlo robustness..."):
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
                rob_cached = rob

    with c2:
        st.info("Tip: increase noise to test worst-case stability. This is decision-support, not just a score.")

    if not isinstance(rob_cached, dict):
        st.info("Not computed yet for this pivot/config.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean (raw hybrid)", f"{rob_cached['mean']:.1f}")
        m2.metric("Std dev", f"{rob_cached['std']:.1f}")
        m3.metric("95% CI", f"[{rob_cached['ci95_low']:.1f}, {rob_cached['ci95_high']:.1f}]")

        fig = px.histogram(
            pd.DataFrame({"score": rob_cached["scores"]}),
            x="score",
            nbins=20,
            title=f"Raw hybrid score distribution — embedding={coords_label}, cosine={'IDF' if use_idf else 'raw'}",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---- Decision Brief (on-demand)
with tabs[3]:
    st.subheader("Decision Brief — on-demand (very expensive)")
    st.caption("Computes robust stats for all targets + path costs. Use only when needed.")

    risk_options = ["Risk-averse (CVaR)", "Balanced", "Risk-seeking (Mean)"]
    try:
        risk_profile = st.segmented_control("Risk profile", options=risk_options, default=risk_options[1])
    except Exception:
        risk_profile = st.radio("Risk profile", options=risk_options, index=1)

    uplift_budget = st.slider("Max uplift skills", 1, 6, 3, 1)
    min_conf = st.slider("Min confidence gate", 0, 100, 25, 5)

    db_key = _cache_key(
        "db",
        coords_label,
        bool(use_idf),
        current,
        round(float(w_cosine), 4),
        round(float(w_map), 4),
        int(n_samples),
        round(float(noise_std), 4),
        int(k_neighbors),
        int(max_steps),
    )
    df_cached = st.session_state["__cache__"]["decision_brief"].get(db_key)

    if st.button("Compute Decision Brief", key="run_decision_brief"):
        with st.spinner("Computing Decision Brief... (this may take a while)"):
            _ensure_core_arrays()
            core = st.session_state["__core__"]
            occ_to_idx = core["occ_to_idx"]
            X = core["X"]
            idf = core["idf"]

            ci = occ_to_idx.get(str(current))
            if ci is None:
                st.error("Current occupation not found in matrix.")
            else:
                a0 = X[int(ci)].astype(float)
                map_parts = map_parts_current
                rng = np.random.default_rng(42)
                rows: list[dict[str, Any]] = []

                for occ in mat.index.astype(str):
                    if occ == str(current):
                        continue
                    ti = occ_to_idx.get(occ)
                    if ti is None:
                        continue
                    b0 = X[int(ti)].astype(float)
                    mp = float(map_parts.get(occ, 0.0))

                    scores = []
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

                        scores.append(_hybrid_score(cos, mp, float(w_cosine), float(w_map)))

                    arr = np.array(scores, dtype=float)
                    q05 = float(np.quantile(arr, 0.05))
                    tail = arr[arr <= q05]
                    cvar05 = float(np.mean(tail)) if tail.size else q05

                    po = _find_pivot_path_fast(
                        current,
                        occ,
                        k_neighbors=int(k_neighbors),
                        max_steps=int(max_steps),
                        use_idf=bool(use_idf),
                    )
                    pc = po.get("total_cost", np.nan) if po.get("reachable") else np.nan
                    em = compute_effort_metrics(mat, current, occ, path_cost=pc)

                    rows.append(
                        {
                            "occupation": occ,
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                            "q05": float(np.quantile(arr, 0.05)),
                            "q95": float(np.quantile(arr, 0.95)),
                            "cvar05": cvar05,
                            "map_part": float(mp),
                            "reachable": bool(po.get("reachable")),
                            **em,
                        }
                    )

                df = pd.DataFrame(rows)
                if not df.empty:
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
        if chosen.empty:
            st.warning("No targets available.")
        else:
            rec = chosen.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended", str(rec["occupation"]))
            c2.metric("Mean (raw hybrid)", f"{rec['mean']:.1f}")
            c3.metric("CVaR 5%", f"{rec['cvar05']:.1f}")
            c4.metric("Effort (mix)", f"{rec['effort_mix']:.1f}")
            st.info(rationale)

            if st.button("Use recommended as target", key="use_decision_brief_reco"):
                st.session_state["__target_override__"] = str(rec["occupation"])
                st.session_state["__has_run__"] = True
                st.rerun()

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
            title="Higher = better score (y), lower = less effort (x), bubble size = CVaR(5%) robustness",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Counterfactual uplift (uses raw hybrid score)")
        st.caption("Counterfactual uses the model-logic raw hybrid scoring (not percentile-calibrated). This is intentional for transparency.")
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
            st.caption(f"Goal: reach raw hybrid score ≥ {score_threshold:.0f} for **{target}**")
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
            st.caption(f"Goal: make **{target}** appear in raw Top-{topk_goal}")

        if cf.get("selected_skills"):
            st.success("Suggested uplift skills: **" + "**, **".join(cf["selected_skills"]) + "**")
        else:
            st.info("No uplift skills suggested (goal already met or sparse).")

        st.write(f"Before: **{cf.get('before_score', np.nan):.1f}** → After: **{cf.get('after_score', np.nan):.1f}**")

# ---- What-If Lab (on-demand)
with tabs[4]:
    st.subheader("What-If Lab — on-demand (extremely expensive)")
    st.caption("Keep the grid small for demos. Results are cached per configuration. (Raw hybrid, then interpret stability.)")

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

    wf_key = _cache_key(
        "wf",
        coords_label,
        bool(use_idf),
        current,
        target,
        round(float(w_min), 3),
        round(float(w_max), 3),
        int(w_steps),
        round(float(n_min), 3),
        round(float(n_max), 3),
        int(n_steps),
        int(mc),
        int(seed),
    )
    cached = st.session_state["__cache__"]["whatif"].get(wf_key)

    if st.button("Run What-If Sweep", key="run_whatif"):
        with st.spinner("Running sweep (heavy)..."):
            _ensure_core_arrays()
            core = st.session_state["__core__"]
            occ_to_idx = core["occ_to_idx"]
            X = core["X"]
            idf = core["idf"]

            ci = occ_to_idx.get(str(current))
            if ci is None:
                st.error("Current occupation not found in matrix.")
            else:
                a0 = X[int(ci)].astype(float)
                map_parts = map_parts_current

                rng = np.random.default_rng(int(seed))
                w_vals = np.linspace(float(w_min), float(w_max), int(w_steps))
                n_vals = np.linspace(float(n_min), float(n_max), int(n_steps))

                records: list[dict[str, Any]] = []
                for w in w_vals:
                    w = float(np.clip(w, 0.0, 1.0))
                    w_map_local = float(1.0 - w)
                    for noise in n_vals:
                        noise = float(max(0.0, noise))
                        for occ in mat.index.astype(str):
                            if occ == str(current):
                                continue
                            ti = occ_to_idx.get(occ)
                            if ti is None:
                                continue
                            b0 = X[int(ti)].astype(float)
                            mp = float(map_parts.get(occ, 0.0))

                            scores = []
                            for _ in range(int(mc)):
                                a = np.clip(a0 + rng.normal(0.0, noise, size=a0.shape), 0.0, None)
                                b = np.clip(b0 + rng.normal(0.0, noise, size=b0.shape), 0.0, None)

                                if use_idf:
                                    aw = a * idf
                                    bw = b * idf
                                    denom = float(np.linalg.norm(aw) * np.linalg.norm(bw) + 1e-12)
                                    cos = float(np.dot(aw, bw) / denom)
                                else:
                                    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                                    cos = float(np.dot(a, b) / denom)

                                scores.append(_hybrid_score(cos, mp, w, w_map_local))

                            records.append(
                                {
                                    "w_cosine": w,
                                    "w_map": w_map_local,
                                    "noise_std": noise,
                                    "occupation": occ,
                                    "mean_score": float(np.mean(scores)),
                                }
                            )

                sweep_df = pd.DataFrame(records)

                group_cols = ["w_cosine", "w_map", "noise_std"]
                topk_lists: list[tuple[str, ...]] = []
                target_ranks: list[float] = []

                for _, g in sweep_df.groupby(group_cols):
                    g2 = g.sort_values("mean_score", ascending=False).reset_index(drop=True)
                    topk_set = tuple(g2.head(int(topk))["occupation"].astype(str).tolist())
                    topk_lists.append(topk_set)

                    tr = g2.index[g2["occupation"] == str(target)]
                    target_ranks.append(float(tr[0] + 1) if len(tr) else float("nan"))

                ranks_arr = np.array([r for r in target_ranks if np.isfinite(r)], dtype=int)
                in_topk_rate = float(np.mean(ranks_arr <= int(topk))) if ranks_arr.size else 0.0

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

                stab = {
                    "target_in_topk_rate": in_topk_rate,
                    "target_rank_values": ranks_arr.tolist(),
                    "topk_freq_df": freq_df,
                    "n_configs": int(len(topk_lists)),
                }

                st.session_state["__cache__"]["whatif"][wf_key] = (sweep_df, stab)
                cached = (sweep_df, stab)

    if cached is None:
        st.info("Not computed yet. Click **Run What-If Sweep**.")
    else:
        sweep_df, stab = cached
        st.metric(f"Target in Top-{topk}", f"{100 * stab['target_in_topk_rate']:.0f}%")

        target_df = sweep_df[sweep_df["occupation"] == str(target)].copy()
        if not target_df.empty:
            pivot = target_df.pivot_table(index="noise_std", columns="w_cosine", values="mean_score", aggfunc="mean")
            fig = px.imshow(
                pivot.to_numpy(),
                x=[f"{x:.2f}" for x in pivot.columns.tolist()],
                y=[f"{y:.2f}" for y in pivot.index.tolist()],
                labels={"x": "w_cosine", "y": "noise_std", "color": "mean score"},
                title=f"Mean raw-hybrid score for target across sweep ({coords_label})",
                aspect="auto",
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Target not found in sweep output (unexpected).")

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
    st.caption("The map is for intuition only. Axes are not semantic.")

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
                    title=f"Clustered {coords_label} map (career neighborhoods).",
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
    )

st.caption(
    "Engineering note: the main score is calibrated (percentile) by default + plausibility-gated by shared-skill overlap. "
    "Raw hybrid is preserved for transparency and reproducibility."
)