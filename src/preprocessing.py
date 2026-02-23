# src/preprocessing.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# -----------------------------
# Core preprocessing
# -----------------------------
def build_occupation_matrix(skills_long_path: str | Path) -> pd.DataFrame:
    """
    Load long-format skills data and return a wide occupation x skill matrix.
    Expected columns: occupation, skill, importance
    """
    skills_long_path = Path(skills_long_path)
    if not skills_long_path.exists():
        raise FileNotFoundError(f"Could not find data file: {skills_long_path}")

    df = pd.read_csv(skills_long_path)

    required = {"occupation", "skill", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}. Found: {list(df.columns)}")

    df = df.copy()
    df["occupation"] = df["occupation"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")

    df = df.dropna(subset=["occupation", "skill", "importance"])

    # average duplicates
    df = df.groupby(["occupation", "skill"], as_index=False)["importance"].mean()

    mat = df.pivot(index="occupation", columns="skill", values="importance").fillna(0.0)
    mat = mat.sort_index(axis=0).sort_index(axis=1)

    return mat


def compute_pca_coords(
    matrix: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute PCA coords (2D) from standardized skill matrix.
    Returns coords df + pca metadata.
    """
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 occupations to compute PCA coords.")
    if matrix.shape[1] < 2:
        raise ValueError("Need at least 2 skills to compute 2D PCA coords.")

    X = matrix.to_numpy(dtype=float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    coords_arr = pca.fit_transform(X_scaled)

    coords = pd.DataFrame(
        {
            "occupation": matrix.index.astype(str),
            "x": coords_arr[:, 0],
            "y": coords_arr[:, 1],
        }
    )

    meta = {
        "n_components": n_components,
        "random_state": random_state,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "notes": "PCA run on standardized skill features (StandardScaler).",
    }

    return coords, meta

def compute_umap_coords(
    matrix: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 20,
    min_dist: float = 0.15,
    metric: str = "cosine",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute UMAP coords (2D) from standardized skill matrix.
    Returns coords df + umap metadata.

    Requires: umap-learn
    """
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 occupations to compute UMAP coords.")
    if matrix.shape[1] < 2:
        raise ValueError("Need at least 2 skills to compute 2D UMAP coords.")

    try:
        import umap  # type: ignore
    except Exception as e:
        raise ImportError(
            "UMAP requires `umap-learn`. Install via: pip install umap-learn"
        ) from e

    X = matrix.to_numpy(dtype=float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    emb = reducer.fit_transform(X_scaled)

    coords = pd.DataFrame(
        {
            "occupation": matrix.index.astype(str),
            "x": emb[:, 0],
            "y": emb[:, 1],
        }
    )

    meta = {
        "n_components": int(n_components),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "random_state": int(random_state),
        "notes": "UMAP run on standardized skill features (StandardScaler).",
    }
    return coords, meta 


def compute_data_quality(matrix: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight data quality / coverage signals for confidence display.
    """
    X = matrix.to_numpy(dtype=float)
    n_occ, n_skills = X.shape

    nonzero = (X > 0).sum()
    density = float(nonzero / (n_occ * n_skills)) if n_occ * n_skills else 0.0

    # per-occupation coverage: fraction of skills with >0
    occ_cov = (X > 0).mean(axis=1) if n_skills else np.array([])
    skill_cov = (X > 0).mean(axis=0) if n_occ else np.array([])

    quality = {
        "n_occupations": int(n_occ),
        "n_skills": int(n_skills),
        "matrix_density": density,
        "occupation_coverage_mean": float(np.mean(occ_cov)) if len(occ_cov) else 0.0,
        "occupation_coverage_min": float(np.min(occ_cov)) if len(occ_cov) else 0.0,
        "skill_coverage_mean": float(np.mean(skill_cov)) if len(skill_cov) else 0.0,
        "skill_coverage_min": float(np.min(skill_cov)) if len(skill_cov) else 0.0,
        "notes": "Coverage is based on non-zero importance values in the matrix.",
    }
    return quality


# -----------------------------
# Feature A: Role clustering
# -----------------------------
def compute_role_clusters_kmeans(
    matrix: pd.DataFrame,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    KMeans clustering on standardized occupation skill vectors.
    If n_clusters is None -> choose k by silhouette score.

    Returns:
      - clusters: occupation -> cluster_id
      - meta: cluster sizes + inertia + silhouette + chosen_k + notes
    """
    n_occ = int(matrix.shape[0])
    if n_occ < 2:
        clusters = {str(matrix.index[0]): 0} if n_occ == 1 else {}
        meta = {"n_clusters": 1 if n_occ == 1 else 0, "notes": "Not enough occupations to cluster."}
        return clusters, meta

    X = matrix.to_numpy(dtype=float)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    # --- Choose k by silhouette if not provided
    sil_scores: dict[int, float] = {}
    chosen_k: int

    if n_clusters is None:
        k_min = 2
        k_max = min(14, n_occ - 1)  # keep it sane / fast
        best_k = k_min
        best_s = -1.0

        for k in range(k_min, k_max + 1):
            km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels_tmp = km_tmp.fit_predict(X_scaled)

            # silhouette requires at least 2 clusters and no empty clusters
            if len(np.unique(labels_tmp)) < 2:
                continue

            s = float(silhouette_score(X_scaled, labels_tmp))
            sil_scores[int(k)] = s
            if s > best_s:
                best_s = s
                best_k = k

        chosen_k = int(best_k)
    else:
        chosen_k = int(max(2, min(int(n_clusters), n_occ)))

    km = KMeans(n_clusters=chosen_k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_scaled)

    clusters = {str(occ): int(lbl) for occ, lbl in zip(matrix.index.astype(str).tolist(), labels)}

    unique, counts = np.unique(labels, return_counts=True)
    size_map = {int(u): int(c) for u, c in zip(unique, counts)}

    meta = {
        "n_clusters": int(chosen_k),
        "random_state": int(random_state),
        "inertia": float(km.inertia_),
        "cluster_sizes": size_map,
        "silhouette_scores": sil_scores,  # empty if user forced k
        "chosen_by": "silhouette" if n_clusters is None else "manual",
        "notes": "KMeans on standardized occupation skill vectors (StandardScaler). Auto-k uses silhouette.",
    }
    return clusters, meta

def compute_cluster_themes(
    matrix: pd.DataFrame,
    clusters: dict[str, int],
    top_n_skills: int = 6,
) -> dict[str, Any]:
    """
    For each cluster, compute a simple theme: top skills by average importance in that cluster.
    Returns cluster_themes dict:
      cluster_id -> {top_skills: [...], mean_vector_top: [...]}
    """
    if matrix.empty or not clusters:
        return {}

    df = matrix.copy()
    df.index = df.index.astype(str)

    # group occupations by cluster id
    inv: dict[int, list[str]] = {}
    for occ, cid in clusters.items():
        inv.setdefault(int(cid), []).append(str(occ))

    themes: dict[str, Any] = {}
    for cid, occs in inv.items():
        sub = df.loc[df.index.intersection(occs)]
        if sub.empty:
            continue
        mean_vec = sub.mean(axis=0).sort_values(ascending=False)
        top_skills = mean_vec.head(top_n_skills).index.astype(str).tolist()
        themes[str(cid)] = {
            "top_skills": top_skills,
            "notes": "Top skills by mean importance within cluster (simple, interpretable theme).",
        }
    return themes


# -----------------------------
# Feature B: Skill taxonomy groups (dummy, but future-proof)
# -----------------------------
def build_skill_taxonomy_dummy(skills: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Deterministic, lightweight skill->group mapping.
    For dummy dataset we use keyword heuristics. Later can be swapped with O*NET taxonomy offline.

    Returns:
      - taxonomy: skill -> group
      - group_meta: order + descriptions
    """
    def _group_for(skill: str) -> str:
        s = skill.strip().lower()

        # Security
        if any(k in s for k in ["security", "incident", "risk assessment"]):
            return "Security & Risk"

        # Design / UX
        if any(k in s for k in ["ux", "ui", "user research", "prototyp"]):
            return "Design & UX"

        # Product / Strategy
        if any(k in s for k in ["roadmap", "stakeholder", "strategy", "product"]):
            return "Product & Strategy"

        # Engineering
        if any(k in s for k in ["git", "docker", "software design", "engineering"]):
            return "Engineering"

        # Analytics / Data
        if any(k in s for k in ["sql", "excel", "statistics", "machine learning", "data", "python", "visualization"]):
            return "Data & Analytics"

        return "Other"

    taxonomy = {str(skill): _group_for(str(skill)) for skill in skills}

    # Stable order for nicer charts
    group_order = [
        "Data & Analytics",
        "Engineering",
        "Product & Strategy",
        "Design & UX",
        "Security & Risk",
        "Other",
    ]
    group_descriptions = {
        "Data & Analytics": "Analysis, modeling, data tooling, quantitative reasoning.",
        "Engineering": "Software building blocks, tooling, development practices.",
        "Product & Strategy": "Planning, alignment, stakeholder work, roadmap & prioritization.",
        "Design & UX": "Research, prototyping, UI/UX craft and iteration.",
        "Security & Risk": "Security controls, incident response, risk management basics.",
        "Other": "Uncategorized / future taxonomy expansion.",
    }

    group_meta = {
        "group_order": group_order,
        "group_descriptions": group_descriptions,
        "notes": "Heuristic dummy taxonomy. Replace offline with O*NET/ESCO mapping later.",
    }
    return taxonomy, group_meta


# -----------------------------
# Saving artifacts
# -----------------------------
def save_artifacts(
    out_dir: Path,
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    pca_meta: dict[str, Any],
    quality: dict[str, Any],
    *,
    clusters: dict[str, int] | None = None,
    cluster_meta: dict[str, Any] | None = None,
    cluster_themes: dict[str, Any] | None = None,
    skill_taxonomy: dict[str, str] | None = None,
    group_meta: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet is robust & fast on Streamlit Cloud (pyarrow already included)
    matrix_path = out_dir / "occupation_skill_matrix.parquet"
    coords_path = out_dir / "pca_coords.parquet"

    matrix.to_parquet(matrix_path, index=True)
    coords.to_parquet(coords_path, index=False)

    (out_dir / "occupations.json").write_text(json.dumps(matrix.index.tolist(), indent=2), encoding="utf-8")
    (out_dir / "skills.json").write_text(json.dumps(matrix.columns.tolist(), indent=2), encoding="utf-8")
    (out_dir / "pca_meta.json").write_text(json.dumps(pca_meta, indent=2), encoding="utf-8")
    (out_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    # New: optional "overkill" artifacts (safe to ignore at runtime)
    if clusters is not None:
        (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2), encoding="utf-8")
    if cluster_meta is not None:
        (out_dir / "cluster_meta.json").write_text(json.dumps(cluster_meta, indent=2), encoding="utf-8")
    if cluster_themes is not None:
        (out_dir / "cluster_themes.json").write_text(json.dumps(cluster_themes, indent=2), encoding="utf-8")
    if skill_taxonomy is not None:
        (out_dir / "skill_taxonomy.json").write_text(json.dumps(skill_taxonomy, indent=2), encoding="utf-8")
    if group_meta is not None:
        (out_dir / "group_meta.json").write_text(json.dumps(group_meta, indent=2), encoding="utf-8")