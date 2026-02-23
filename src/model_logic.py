from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuntimeArtifacts:
    """Bundle of precomputed, runtime-ready artifacts loaded from disk.

    The app treats these as immutable inputs. Keeping this structure stable makes it easier
    to evolve the preprocessing pipeline without changing runtime code paths.
    """

    matrix: pd.DataFrame
    coords: pd.DataFrame
    pca_meta: dict[str, Any]
    quality: dict[str, Any]
    clusters: dict[str, int]
    cluster_meta: dict[str, Any]
    cluster_themes: dict[str, Any]
    skill_taxonomy: dict[str, str]
    group_meta: dict[str, Any]
    umap_coords: pd.DataFrame
    umap_meta: dict[str, Any]


def _read_json_if_exists(path: Path, default: Any) -> Any:
    """Read a JSON file if it exists, otherwise return a provided default.

    This is intentionally tolerant: missing optional metadata should not break the app.
    """
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def load_runtime_artifacts(artifact_dir: str | Path = "artifacts") -> RuntimeArtifacts:
    """Load the full runtime artifact set from a directory.

    The parquet files are required for core functionality; JSON metadata is optional and
    defaults safely when missing.
    """
    artifact_dir = Path(artifact_dir)

    matrix_path = artifact_dir / "occupation_skill_matrix.parquet"
    coords_path = artifact_dir / "pca_coords.parquet"
    pca_meta_path = artifact_dir / "pca_meta.json"
    quality_path = artifact_dir / "data_quality.json"

    clusters_path = artifact_dir / "clusters.json"
    cluster_meta_path = artifact_dir / "cluster_meta.json"
    cluster_themes_path = artifact_dir / "cluster_themes.json"
    taxonomy_path = artifact_dir / "skill_taxonomy.json"
    group_meta_path = artifact_dir / "group_meta.json"

    umap_coords_path = artifact_dir / "umap_coords.parquet"
    umap_meta_path = artifact_dir / "umap_meta.json"

    # These are hard requirements: the app cannot score or display the core map without them.
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {matrix_path}. Run preprocessing: python scripts/preprocess_dummy.py or preprocess_onet.py"
        )
    if not coords_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {coords_path}. Run preprocessing: python scripts/preprocess_dummy.py or preprocess_onet.py"
        )

    matrix = pd.read_parquet(matrix_path)
    # Support both "occupation" column and already-indexed parquet formats.
    if "occupation" in matrix.columns:
        matrix = matrix.set_index("occupation")

    coords = pd.read_parquet(coords_path)

    pca_meta = _read_json_if_exists(pca_meta_path, {})
    quality = _read_json_if_exists(quality_path, {})

    clusters = _read_json_if_exists(clusters_path, {})
    cluster_meta = _read_json_if_exists(cluster_meta_path, {})
    cluster_themes = _read_json_if_exists(cluster_themes_path, {})
    skill_taxonomy = _read_json_if_exists(taxonomy_path, {})
    group_meta = _read_json_if_exists(group_meta_path, {})

    # UMAP artifacts are optional; treat them as a best-effort enhancement.
    umap_coords = pd.DataFrame(columns=["occupation", "x", "y"])
    if umap_coords_path.exists():
        try:
            tmp = pd.read_parquet(umap_coords_path)
            if "occupation" in tmp.columns and {"x", "y"}.issubset(tmp.columns):
                umap_coords = tmp[["occupation", "x", "y"]].copy()
        except Exception:
            # Be resilient to schema changes or partial writes during artifact generation.
            umap_coords = pd.DataFrame(columns=["occupation", "x", "y"])

    umap_meta = _read_json_if_exists(umap_meta_path, {})

    # Validate early: downstream code assumes these columns exist.
    if "occupation" not in coords.columns or not {"x", "y"}.issubset(coords.columns):
        raise ValueError("pca_coords.parquet must contain columns: occupation, x, y")

    # Sorting ensures deterministic UI ordering and stable results across runs.
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    coords = coords.sort_values("occupation").reset_index(drop=True)

    if not umap_coords.empty:
        umap_coords = umap_coords.sort_values("occupation").reset_index(drop=True)

    return RuntimeArtifacts(
        matrix=matrix,
        coords=coords,
        pca_meta=pca_meta if isinstance(pca_meta, dict) else {},
        quality=quality if isinstance(quality, dict) else {},
        clusters={str(k): int(v) for k, v in clusters.items()} if isinstance(clusters, dict) else {},
        cluster_meta=cluster_meta if isinstance(cluster_meta, dict) else {},
        cluster_themes=cluster_themes if isinstance(cluster_themes, dict) else {},
        skill_taxonomy={str(k): str(v) for k, v in skill_taxonomy.items()} if isinstance(skill_taxonomy, dict) else {},
        group_meta=group_meta if isinstance(group_meta, dict) else {},
        umap_coords=umap_coords,
        umap_meta=umap_meta if isinstance(umap_meta, dict) else {},
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with a safe zero-vector guard."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        # If either vector has no magnitude, treat similarity as zero rather than NaN/inf.
        return 0.0
    return float(np.dot(a, b) / denom)


def _precompute_map_maxdist(coords: pd.DataFrame) -> float:
    """Compute the maximum pairwise distance in the embedding space.

    Used to normalize distances into a [0, 1] proximity score without hardcoding a scale.
    """
    all_xy = coords[["x", "y"]].to_numpy(dtype=float)
    if len(all_xy) < 2:
        return 0.0
    max_dist = 0.0
    # O(n^2) but typically small enough for embedding-sized datasets; kept explicit for transparency.
    for i in range(len(all_xy)):
        diffs = all_xy[i + 1 :] - all_xy[i]
        if len(diffs) == 0:
            continue
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        md = float(np.max(dists))
        if md > max_dist:
            max_dist = md
    return float(max_dist)


def _map_proximity(coords: pd.DataFrame, occ_a: str, occ_b: str, max_dist: float) -> float:
    """Convert 2D embedding distance into a normalized proximity score in [0, 1]."""
    a_xy = coords.loc[coords["occupation"] == occ_a, ["x", "y"]].to_numpy(dtype=float)
    b_xy = coords.loc[coords["occupation"] == occ_b, ["x", "y"]].to_numpy(dtype=float)
    if len(a_xy) == 0 or len(b_xy) == 0:
        # Missing coordinates should not crash scoring; treat as no proximity signal.
        return 0.0
    if max_dist <= 0.0:
        # Degenerate embedding (single point or identical coords): treat as maximally close.
        return 1.0
    dist = float(np.linalg.norm(a_xy[0] - b_xy[0]))
    return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))


def compute_gap_df(matrix: pd.DataFrame, current_occ: str, target_occ: str) -> pd.DataFrame:
    """Compute per-skill gaps and transfer proxies between two occupations."""
    if current_occ not in matrix.index:
        raise ValueError(f"Current occupation not found: {current_occ}")
    if target_occ not in matrix.index:
        raise ValueError(f"Target occupation not found: {target_occ}")

    current = matrix.loc[current_occ].astype(float)
    target = matrix.loc[target_occ].astype(float)
    gap = target - current

    df = pd.DataFrame(
        {
            "skill": matrix.columns.astype(str),
            "current_importance": current.values,
            "target_importance": target.values,
            "gap": gap.values,
        }
    )
    # Keep both signed and absolute gap forms; different panels rank on different notions of "need".
    df["abs_gap"] = df["gap"].abs()
    # Simple interaction term: highlights skills that matter in both roles (transfer leverage).
    df["transfer_strength"] = df["current_importance"] * df["target_importance"]
    return df


def compute_match_score_cosine(matrix: pd.DataFrame, current_occ: str, target_occ: str) -> float:
    """Cosine-based match score in [0, 100], clipped to non-negative similarity."""
    a = matrix.loc[current_occ].astype(float).values
    b = matrix.loc[target_occ].astype(float).values
    sim = _cosine_similarity(a, b)
    return float(np.clip(max(0.0, sim) * 100.0, 0.0, 100.0))


def compute_match_score_hybrid(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    target_occ: str,
    w_cosine: float = 0.65,
    w_map: float = 0.35,
) -> float:
    """Hybrid score mixing skill-vector similarity with embedding proximity."""
    cosine_part = compute_match_score_cosine(matrix, current_occ, target_occ) / 100.0
    max_dist = _precompute_map_maxdist(coords)
    map_part = _map_proximity(coords, current_occ, target_occ, max_dist)
    score = 100.0 * (w_cosine * cosine_part + w_map * map_part)
    return float(np.clip(score, 0.0, 100.0))


def compute_skill_contributions(gap_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive compact, high-signal tables explaining match drivers and gaps."""
    df = gap_df.copy()

    # Overlap is the shared mass between profiles; useful as an intuitive transfer signal.
    df["overlap"] = np.minimum(df["current_importance"], df["target_importance"])
    # Weight overlap by importance to emphasize skills that matter in both roles.
    df["match_driver_score"] = df["overlap"] * (df["target_importance"] + df["current_importance"]) / 2.0
    match_drivers = df.sort_values("match_driver_score", ascending=False).head(10)

    missing = df[df["gap"] > 0].copy()
    # Prioritize skills that are both missing and important for the target role.
    missing["missing_priority"] = missing["gap"] * missing["target_importance"]
    missing_drivers = missing.sort_values(["missing_priority", "gap"], ascending=False).head(10)

    surplus = df[df["gap"] < 0].copy()
    # Surplus can be used as a narrative tool (strengths), not necessarily a penalty.
    surplus["surplus_magnitude"] = (-surplus["gap"]) * surplus["current_importance"]
    surplus_skills = surplus.sort_values(["surplus_magnitude"], ascending=False).head(10)

    return {
        "match_drivers": match_drivers,
        "missing_drivers": missing_drivers,
        "surplus_skills": surplus_skills,
    }


def generate_learning_plan(gap_df: pd.DataFrame) -> dict[str, list[str]]:
    """Generate a simple three-phase plan from missing skills.

    This is a deterministic fallback used when an external plan generator is unavailable.
    """
    missing = gap_df[gap_df["gap"] > 0].copy()
    if missing.empty:
        return {
            "Foundations": [
                "The target skill profile is already covered in this dataset. Focus on portfolio work and interview narratives."
            ],
            "Intermediate": ["Deepen real-world projects that demonstrate your strongest overlap skills."],
            "Advanced": ["Specialize into domain-specific variants of your most transferable strengths."],
        }

    missing["priority"] = missing["gap"] * missing["target_importance"]
    missing = missing.sort_values("priority", ascending=False)

    skills = missing["skill"].tolist()
    n = len(skills)
    # Split into roughly equal thirds while ensuring each phase has at least one item when possible.
    a = max(1, n // 3)
    b = max(1, n // 3)

    foundations = skills[:a]
    intermediate = skills[a : a + b]
    advanced = skills[a + b :]

    def bullets(phase_skills: list[str], label: str) -> list[str]:
        # Keep messaging consistent even when a phase ends up empty due to small n.
        if not phase_skills:
            return [f"No additional {label.lower()} skills identified in this dataset."]
        return [
            f"Build {label.lower()} competence in **{s}** through a small project, spaced practice, and a measurable deliverable."
            for s in phase_skills
        ]

    return {
        "Foundations": bullets(foundations, "Foundational"),
        "Intermediate": bullets(intermediate, "Intermediate"),
        "Advanced": bullets(advanced, "Advanced"),
    }


def _get_group_order(group_meta: dict[str, Any]) -> list[str]:
    """Extract an explicit group ordering list when provided by metadata."""
    order = group_meta.get("group_order", [])
    if isinstance(order, list) and all(isinstance(x, str) for x in order):
        return order
    return []


def compute_group_gap_df(
    matrix: pd.DataFrame,
    skill_taxonomy: dict[str, str],
    group_meta: dict[str, Any],
    current_occ: str,
    target_occ: str,
) -> pd.DataFrame:
    """Aggregate per-skill gaps into taxonomy groups for higher-level reporting."""
    if not skill_taxonomy:
        # Without a taxonomy, grouping would be misleading; return an empty frame with expected columns.
        return pd.DataFrame(columns=["group", "current_importance", "target_importance", "gap"])

    current = matrix.loc[current_occ].astype(float)
    target = matrix.loc[target_occ].astype(float)

    skills = matrix.columns.astype(str).tolist()
    # Default to "Other" to keep unknown skills visible rather than dropping them.
    groups = [skill_taxonomy.get(s, "Other") for s in skills]

    cur_df = pd.DataFrame({"group": groups, "value": current.values})
    tgt_df = pd.DataFrame({"group": groups, "value": target.values})

    cur_agg = cur_df.groupby("group", as_index=False)["value"].sum().rename(columns={"value": "current_importance"})
    tgt_agg = tgt_df.groupby("group", as_index=False)["value"].sum().rename(columns={"value": "target_importance"})

    # Outer merge ensures groups present in only one side still appear with 0 on the other side.
    out = cur_agg.merge(tgt_agg, on="group", how="outer").fillna(0.0)
    out["gap"] = out["target_importance"] - out["current_importance"]

    order = _get_group_order(group_meta)
    if order:
        # Keep a stable, metadata-driven ordering when available; unknown groups go last.
        out["__ord__"] = out["group"].apply(lambda g: order.index(g) if g in order else 10_000)
        out = out.sort_values(["__ord__", "group"]).drop(columns=["__ord__"])
    else:
        # Default ordering emphasizes where the target role differs most from the current role.
        out = out.sort_values("gap", ascending=False)

    return out.reset_index(drop=True)


def filter_missing_skills_by_group(
    gap_df: pd.DataFrame,
    skill_taxonomy: dict[str, str],
    group_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top missing skills within a specific taxonomy group."""
    if gap_df.empty or not skill_taxonomy:
        return pd.DataFrame(columns=gap_df.columns)

    df = gap_df.copy()
    df["group"] = df["skill"].astype(str).map(lambda s: skill_taxonomy.get(s, "Other"))
    missing = df[(df["gap"] > 0) & (df["group"] == group_name)].copy()
    missing = missing.sort_values(["gap", "target_importance"], ascending=False).head(int(top_n))
    return missing.reset_index(drop=True)


def format_cluster_theme(cluster_id: int | str, cluster_themes: dict[str, Any]) -> str:
    """Format a concise, user-facing theme string for a cluster."""
    key = str(cluster_id)
    info = cluster_themes.get(key, {})
    if isinstance(info, dict):
        skills = info.get("top_skills", [])
        if isinstance(skills, list) and skills:
            # Keep the label compact; it is typically used in UI badges/tooltips.
            return ", ".join([str(s) for s in skills[:6]])
    return ""


def recommend_similar_roles(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    top_k: int = 5,
    w_cosine: float = 0.65,
    w_map: float = 0.35,
) -> pd.DataFrame:
    """Recommend roles similar to the current occupation using the hybrid score."""
    max_dist = _precompute_map_maxdist(coords)
    a = matrix.loc[current_occ].astype(float).values

    roles: list[tuple[str, float]] = []
    for occ in matrix.index.astype(str):
        if occ == current_occ:
            continue
        b = matrix.loc[occ].astype(float).values
        sim = max(0.0, _cosine_similarity(a, b))
        map_part = _map_proximity(coords, current_occ, occ, max_dist)
        score = 100.0 * (w_cosine * sim + w_map * map_part)
        roles.append((occ, float(np.clip(score, 0.0, 100.0))))

    return (
        pd.DataFrame(roles, columns=["occupation", "match_score"])
        .sort_values("match_score", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )


def compute_confidence_score(
    matrix: pd.DataFrame, pca_meta: dict[str, Any], current_occ: str, target_occ: str
) -> dict[str, Any]:
    """Compute a heuristic confidence score for a pivot result.

    This is intentionally not a probability: it combines coverage overlap, dataset density,
    and how well the 2D embedding explains variance, to provide an at-a-glance reliability cue.
    """
    X = matrix.to_numpy(dtype=float)
    density = float((X > 0).mean()) if X.size else 0.0

    a = matrix.loc[current_occ].astype(float).values
    b = matrix.loc[target_occ].astype(float).values

    # Binary support overlap approximates how comparable the profiles are given sparse skill signals.
    a_nz = a > 0
    b_nz = b > 0
    overlap = float((a_nz & b_nz).mean()) if len(a_nz) else 0.0
    union = float((a_nz | b_nz).mean()) if len(a_nz) else 0.0
    jacc = 0.0 if union == 0.0 else overlap / union

    # The embedding EVR acts as a proxy for how meaningful "map distance" is in this dataset.
    evr = pca_meta.get("explained_variance_ratio", [0.0, 0.0])
    evr2 = float(sum(evr[:2])) if isinstance(evr, list) else 0.0

    score_0_1 = np.clip(0.45 * jacc + 0.25 * density + 0.30 * evr2, 0.0, 1.0)
    score = float(score_0_1 * 100.0)

    return {
        "confidence_score": score,
        "signals": {
            "overlap_jaccard": float(jacc),
            "matrix_density": float(density),
            "pca_evr_2d": float(evr2),
        },
        "notes": "Heuristic confidence: overlap + density + PCA 2D EVR. Not a calibrated probability.",
    }


def _top_k_neighbors_by_cosine(matrix: pd.DataFrame, occ: str, k: int) -> List[Tuple[str, float]]:
    """Return the k nearest neighbors by cosine similarity (descending)."""
    a = matrix.loc[occ].astype(float).values
    sims: List[Tuple[str, float]] = []
    for other in matrix.index:
        if other == occ:
            continue
        b = matrix.loc[other].astype(float).values
        sim = _cosine_similarity(a, b)
        sims.append((str(other), float(sim)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[: int(k)]


def build_transition_graph(matrix: pd.DataFrame, k_neighbors: int = 5) -> Dict[str, List[Tuple[str, float]]]:
    """Build a directed kNN graph where edge weights represent transition "cost"."""
    graph: Dict[str, List[Tuple[str, float]]] = {}
    for occ in matrix.index.astype(str):
        neigh = _top_k_neighbors_by_cosine(matrix, occ, k_neighbors)
        edges: List[Tuple[str, float]] = []
        for n, sim in neigh:
            # Convert similarity into a bounded cost to support shortest-path style search.
            cost = 1.0 - max(0.0, sim)
            edges.append((n, float(np.clip(cost, 0.0, 1.0))))
        graph[occ] = edges
    return graph


def find_pivot_path(
    matrix: pd.DataFrame,
    start_occ: str,
    target_occ: str,
    k_neighbors: int = 5,
    max_steps: int = 4,
) -> dict[str, Any]:
    """Find a stepping-stone route on the kNN graph using a Dijkstra-like search.

    The graph is sparse (k neighbors per node), so this approach is usually fast enough
    without additional priority-queue machinery.
    """
    if start_occ not in matrix.index:
        raise ValueError(f"Start occupation not found: {start_occ}")
    if target_occ not in matrix.index:
        raise ValueError(f"Target occupation not found: {target_occ}")

    graph = build_transition_graph(matrix, k_neighbors=k_neighbors)

    start = str(start_occ)
    target = str(target_occ)

    dist: Dict[str, float] = {start: 0.0}
    prev: Dict[str, str] = {}
    visited: set[str] = set()

    while True:
        # Select the unvisited node with the smallest current distance (Dijkstra step).
        candidates = [(node, d) for node, d in dist.items() if node not in visited]
        if not candidates:
            break
        node, dmin = min(candidates, key=lambda x: x[1])

        if node == target:
            break

        visited.add(node)

        for neigh, cost in graph.get(node, []):
            nd = dmin + cost
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
        # This assumes 'prev' is complete along the best-known path to target.
        path.append(prev[path[-1]])
    path.reverse()

    truncated = len(path) > int(max_steps)
    if truncated:
        # Truncation is a UI/product constraint, not a graph constraint.
        path = path[: int(max_steps)]

    if truncated and path[-1] != target:
        return {
            "path": path,
            "reachable": False,
            "truncated": True,
            "k_neighbors": int(k_neighbors),
            "max_steps": int(max_steps),
            "step_costs": [],
            "total_cost": float(dist.get(target, np.nan)),
            "notes": "Path exists but was truncated before reaching target. Increase max_steps.",
        }

    step_costs: list[float] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        cost = np.nan
        # Edge weights are stored on outgoing adjacency lists; scan to recover step costs.
        for neigh, c in graph.get(u, []):
            if neigh == v:
                cost = c
                break
        step_costs.append(float(cost))

    return {
        "path": path,
        "reachable": True,
        "truncated": bool(truncated),
        "k_neighbors": int(k_neighbors),
        "max_steps": int(max_steps),
        "step_costs": step_costs,
        "total_cost": float(dist.get(target, np.nan)),
        "notes": "Edge cost is 1 - max(0, cosine_sim). Lower cost indicates an easier transition.",
    }


def robustness_analysis(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    target_occ: str,
    w_cosine: float,
    w_map: float,
    n_samples: int = 200,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate score stability under Gaussian noise on skill vectors.

    The map component is held fixed to isolate sensitivity to the skill profile inputs.
    """
    rng = np.random.default_rng(seed)

    a0 = matrix.loc[current_occ].astype(float).values
    b0 = matrix.loc[target_occ].astype(float).values

    max_dist = _precompute_map_maxdist(coords)
    map_part = _map_proximity(coords, current_occ, target_occ, max_dist)

    scores: list[float] = []
    for _ in range(int(n_samples)):
        # Clip at 0 to avoid creating negative "skill weights" after perturbation.
        a = np.clip(a0 + rng.normal(0.0, noise_std, size=a0.shape), 0.0, None)
        b = np.clip(b0 + rng.normal(0.0, noise_std, size=b0.shape), 0.0, None)

        sim = max(0.0, _cosine_similarity(a, b))
        score = 100.0 * (w_cosine * sim + w_map * map_part)
        scores.append(float(np.clip(score, 0.0, 100.0)))

    arr = np.array(scores, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))

    return {
        "n_samples": int(n_samples),
        "noise_std": float(noise_std),
        "mean": mean,
        "std": std,
        "ci95_low": lo,
        "ci95_high": hi,
        "scores": scores,
        "notes": "Monte Carlo stability on skill vectors (Gaussian noise). Map component held fixed.",
    }


def _hybrid_score_from_vectors(
    a: np.ndarray,
    b: np.ndarray,
    map_part: float,
    w_cosine: float,
    w_map: float,
) -> float:
    """Compute the hybrid score directly from vectors and a precomputed map component."""
    sim = max(0.0, _cosine_similarity(a, b))
    score = 100.0 * (w_cosine * sim + w_map * float(map_part))
    return float(np.clip(score, 0.0, 100.0))


def _cvar_left_tail(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Compute left-tail CVaR (expected value in the worst alpha-quantile)."""
    if scores.size == 0:
        return float("nan")
    a = float(alpha)
    a = min(max(a, 1e-6), 1.0)
    q = float(np.quantile(scores, a))
    tail = scores[scores <= q]
    return float(np.mean(tail)) if tail.size else float(q)


def compute_all_targets_robustness(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    *,
    w_cosine: float,
    w_map: float,
    n_samples: int,
    noise_std: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Run robustness simulation for all targets from a single current occupation."""
    rng = np.random.default_rng(seed)
    a0 = matrix.loc[current_occ].astype(float).values

    max_dist = _precompute_map_maxdist(coords)

    rows: list[dict[str, Any]] = []
    for occ in matrix.index.astype(str):
        if occ == current_occ:
            continue

        b0 = matrix.loc[occ].astype(float).values
        map_part = _map_proximity(coords, current_occ, occ, max_dist)

        scores: list[float] = []
        for _ in range(int(n_samples)):
            a = np.clip(a0 + rng.normal(0.0, noise_std, size=a0.shape), 0.0, None)
            b = np.clip(b0 + rng.normal(0.0, noise_std, size=b0.shape), 0.0, None)
            scores.append(_hybrid_score_from_vectors(a, b, map_part, w_cosine, w_map))

        arr = np.array(scores, dtype=float)
        rows.append(
            {
                "occupation": occ,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q95": float(np.quantile(arr, 0.95)),
                "cvar05": float(_cvar_left_tail(arr, alpha=0.05)),
                "map_part": float(map_part),
            }
        )

    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def compute_effort_metrics(
    matrix: pd.DataFrame,
    current_occ: str,
    target_occ: str,
    *,
    skill_gap_weight: float = 1.0,
    path_cost_weight: float = 1.0,
    path_cost: float | None = None,
) -> dict[str, float]:
    """Compute simple effort proxies from skill gaps and (optional) path cost."""
    gap_df = compute_gap_df(matrix, current_occ, target_occ)
    missing = gap_df[gap_df["gap"] > 0].copy()
    gap_effort = (
        float(np.sum(missing["gap"].values * missing["target_importance"].values)) if not missing.empty else 0.0
    )

    # Keep NaN path costs explicit; callers may want to distinguish "unknown" from "zero".
    pc = (
        float(path_cost)
        if path_cost is not None and not (isinstance(path_cost, float) and np.isnan(path_cost))
        else float("nan")
    )
    mix = float(skill_gap_weight * gap_effort + path_cost_weight * (pc if np.isfinite(pc) else gap_effort))

    return {"gap_effort": gap_effort, "path_cost": pc, "effort_mix": mix}


def pareto_frontier_flags(
    df: pd.DataFrame,
    *,
    maximize_cols: list[str],
    minimize_cols: list[str],
) -> pd.Series:
    """Return a boolean mask indicating Pareto-efficient rows.

    A row is marked inefficient if another row is at least as good in all objectives and
    strictly better in at least one, with special care for non-finite minimize objectives.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    Xmax = df[maximize_cols].to_numpy(dtype=float) if maximize_cols else np.zeros((len(df), 0))
    Xmin = df[minimize_cols].to_numpy(dtype=float) if minimize_cols else np.zeros((len(df), 0))

    n = len(df)
    efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not efficient[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if not efficient[i]:
                break

            cond_max = True
            cond_min = True
            strict = False

            if Xmax.shape[1] > 0:
                if np.any(Xmax[j] < Xmax[i]):
                    cond_max = False
                if np.any(Xmax[j] > Xmax[i]):
                    strict = True

            if Xmin.shape[1] > 0:
                # Non-finite values are treated as incomparable rather than dominating.
                if np.any(~np.isfinite(Xmin[j])) or np.any(~np.isfinite(Xmin[i])):
                    cond_min = False
                else:
                    if np.any(Xmin[j] > Xmin[i]):
                        cond_min = False
                    if np.any(Xmin[j] < Xmin[i]):
                        strict = True

            if cond_max and cond_min and strict:
                efficient[i] = False

    return pd.Series(efficient, index=df.index)


def counterfactual_uplift_greedy(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    target_occ: str,
    *,
    w_cosine: float,
    w_map: float,
    goal_mode: str,
    score_threshold: float = 60.0,
    top_k: int = 3,
    max_skills: int = 3,
) -> dict[str, Any]:
    """Greedy counterfactual: raise a small set of missing skills to target levels.

    This produces a "what would I need to improve?" narrative under the vector scoring model.
    """
    a_base = matrix.loc[current_occ].astype(float).values.copy()
    b_target = matrix.loc[target_occ].astype(float).values.copy()
    skills = matrix.columns.astype(str).tolist()

    max_dist = _precompute_map_maxdist(coords)
    # Cache per-occupation map parts to avoid recomputing distances inside scoring loops.
    map_parts = {
        occ: _map_proximity(coords, current_occ, str(occ), max_dist)
        for occ in matrix.index.astype(str)
        if str(occ) != current_occ
    }

    gap_df = compute_gap_df(matrix, current_occ, target_occ)
    candidates = gap_df[gap_df["gap"] > 0].copy()
    if candidates.empty:
        return {"selected_skills": [], "achieved": True, "notes": "No missing skills for this pivot in this dataset."}

    cand_idx = candidates.index.tolist()

    def _score_for_target(a_vec: np.ndarray) -> float:
        return _hybrid_score_from_vectors(a_vec, b_target, map_parts[target_occ], w_cosine, w_map)

    def _target_in_topk(a_vec: np.ndarray) -> bool:
        rows: list[tuple[str, float]] = []
        for occ in matrix.index.astype(str):
            if occ == current_occ:
                continue
            b = matrix.loc[occ].astype(float).values
            s = _hybrid_score_from_vectors(a_vec, b, map_parts[occ], w_cosine, w_map)
            rows.append((occ, s))
        rows.sort(key=lambda x: x[1], reverse=True)
        top = [o for o, _ in rows[: int(top_k)]]
        return target_occ in top

    def _goal_met(a_vec: np.ndarray) -> bool:
        if goal_mode == "threshold":
            return _score_for_target(a_vec) >= float(score_threshold)
        if goal_mode == "topk":
            return _target_in_topk(a_vec)
        raise ValueError("goal_mode must be 'threshold' or 'topk'")

    a = a_base.copy()
    selected: list[str] = []

    if _goal_met(a):
        return {
            "selected_skills": [],
            "achieved": True,
            "before_score": float(_score_for_target(a_base)),
            "after_score": float(_score_for_target(a)),
            "notes": "Goal already met under the current dataset and scoring model.",
        }

    for _ in range(int(max_skills)):
        best_skill: str | None = None
        best_gain = -1e18
        best_a: np.ndarray | None = None

        base_score = float(_score_for_target(a))

        for idx in cand_idx:
            skill = str(gap_df.loc[idx, "skill"])
            if skill in selected:
                continue

            # The model vector is aligned to matrix columns; update by column index.
            j = skills.index(skill)
            a_try = a.copy()
            a_try[j] = max(a_try[j], b_target[j])

            s_try = float(_score_for_target(a_try))
            gain = s_try - base_score

            # For ranking goals, crossing into top-k is treated as a qualitative "win".
            if goal_mode == "topk" and (not _target_in_topk(a)) and _target_in_topk(a_try):
                gain += 1000.0

            if gain > best_gain:
                best_gain = gain
                best_skill = skill
                best_a = a_try

        if best_skill is None or best_a is None:
            break

        selected.append(best_skill)
        a = best_a

        if _goal_met(a):
            break

    achieved = _goal_met(a)
    return {
        "selected_skills": selected,
        "achieved": bool(achieved),
        "before_score": float(_score_for_target(a_base)),
        "after_score": float(_score_for_target(a)),
        "notes": "Greedy uplift: selected skills are raised to target levels in the vector model.",
    }