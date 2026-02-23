from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_occupation_matrix(skills_long_path: str | Path) -> pd.DataFrame:
    """Load long-format skill data and return a wide matrix (occupation × skill).

    Expected columns: occupation, skill, importance
    """
    skills_long_path = Path(skills_long_path)
    if not skills_long_path.exists():
        raise FileNotFoundError(f"Data file not found: {skills_long_path}")

    df = pd.read_csv(skills_long_path)

    required = {"occupation", "skill", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df.copy()
    df["occupation"] = df["occupation"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    # Coerce importance values to numeric; invalid values are dropped as unusable signals.
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["occupation", "skill", "importance"])

    # If duplicates exist, averaging avoids overweighting repeated rows while keeping the signal.
    df = df.groupby(["occupation", "skill"], as_index=False)["importance"].mean()

    mat = df.pivot(index="occupation", columns="skill", values="importance").fillna(0.0)
    # Sorting yields deterministic artifacts and stable downstream displays.
    mat = mat.sort_index(axis=0).sort_index(axis=1)
    return mat


def compute_pca_coords(
    matrix: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute PCA coordinates from a standardized occupation-skill matrix.

    Returns:
        - coords: DataFrame with columns [occupation, x, y]
        - meta: PCA metadata (explained variance ratio, singular values, parameters)
    """
    if matrix.shape[0] < 2:
        raise ValueError("At least 2 occupations are required to compute PCA coordinates.")
    if matrix.shape[1] < 2:
        raise ValueError("At least 2 skills are required to compute 2D PCA coordinates.")

    X = matrix.to_numpy(dtype=float)
    # Standardization is required so high-variance skills do not dominate PCA purely by scale.
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

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
        "n_components": int(n_components),
        "random_state": int(random_state),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
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
    """Compute UMAP coordinates from a standardized occupation-skill matrix.

    Requires: umap-learn
    """
    if matrix.shape[0] < 2:
        raise ValueError("At least 2 occupations are required to compute UMAP coordinates.")
    if matrix.shape[1] < 2:
        raise ValueError("At least 2 skills are required to compute 2D UMAP coordinates.")

    try:
        import umap  # type: ignore
    except Exception as e:
        # Fail with a direct installation hint; UMAP is an optional artifact.
        raise ImportError("UMAP requires `umap-learn`. Install with: pip install umap-learn") from e

    X = matrix.to_numpy(dtype=float)
    # Match PCA preprocessing to keep embeddings comparable and reduce scale artifacts.
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

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
    }
    return coords, meta


def compute_data_quality(matrix: pd.DataFrame) -> dict[str, Any]:
    """Compute lightweight coverage metrics for the occupation-skill matrix."""
    X = matrix.to_numpy(dtype=float)
    n_occ, n_skills = X.shape

    denom = int(n_occ * n_skills)
    # Density is a quick proxy for sparsity; many downstream heuristics depend on it.
    density = float((X > 0).sum() / denom) if denom else 0.0

    occ_cov = (X > 0).mean(axis=1) if n_skills else np.array([])
    skill_cov = (X > 0).mean(axis=0) if n_occ else np.array([])

    return {
        "n_occupations": int(n_occ),
        "n_skills": int(n_skills),
        "matrix_density": float(density),
        "occupation_coverage_mean": float(np.mean(occ_cov)) if occ_cov.size else 0.0,
        "occupation_coverage_min": float(np.min(occ_cov)) if occ_cov.size else 0.0,
        "skill_coverage_mean": float(np.mean(skill_cov)) if skill_cov.size else 0.0,
        "skill_coverage_min": float(np.min(skill_cov)) if skill_cov.size else 0.0,
    }


def compute_role_clusters_kmeans(
    matrix: pd.DataFrame,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Cluster occupations using KMeans on standardized skill vectors.

    If n_clusters is None, selects k by maximizing silhouette score over a bounded range.

    Returns:
        - clusters: mapping occupation -> cluster_id
        - meta: clustering metadata (sizes, inertia, silhouette scores)
    """
    n_occ = int(matrix.shape[0])
    if n_occ < 2:
        # Preserve a consistent output shape even for tiny datasets.
        clusters = {str(matrix.index[0]): 0} if n_occ == 1 else {}
        meta = {"n_clusters": 1 if n_occ == 1 else 0}
        return clusters, meta

    X = matrix.to_numpy(dtype=float)
    # Standardization is required for distance-based clustering on heterogeneous skill scales.
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    sil_scores: dict[int, float] = {}
    if n_clusters is None:
        # Keep the search bounded to avoid overfitting and reduce runtime on larger datasets.
        k_min = 2
        k_max = min(14, n_occ - 1)

        best_k = k_min
        best_s = -1.0

        for k in range(k_min, k_max + 1):
            km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels_tmp = km_tmp.fit_predict(X_scaled)
            if len(np.unique(labels_tmp)) < 2:
                # Silhouette is undefined for a single cluster; skip degenerate solutions.
                continue

            s = float(silhouette_score(X_scaled, labels_tmp))
            sil_scores[int(k)] = s
            if s > best_s:
                best_s = s
                best_k = k

        chosen_k = int(best_k)
        chosen_by = "silhouette"
    else:
        # Ensure k is valid for KMeans while respecting the caller's intent.
        chosen_k = int(max(2, min(int(n_clusters), n_occ)))
        chosen_by = "manual"

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
        "silhouette_scores": sil_scores,
        "chosen_by": chosen_by,
    }
    return clusters, meta


def compute_cluster_themes(
    matrix: pd.DataFrame,
    clusters: dict[str, int],
    top_n_skills: int = 6,
) -> dict[str, Any]:
    """Summarize each cluster by listing the top skills by mean importance."""
    if matrix.empty or not clusters:
        return {}

    df = matrix.copy()
    df.index = df.index.astype(str)

    # Group occupations by cluster id; clustering output is stored as a plain mapping.
    by_cluster: dict[int, list[str]] = {}
    for occ, cid in clusters.items():
        by_cluster.setdefault(int(cid), []).append(str(occ))

    themes: dict[str, Any] = {}
    for cid, occs in by_cluster.items():
        sub = df.loc[df.index.intersection(occs)]
        if sub.empty:
            continue
        # Mean importance provides an interpretable "prototype" for the cluster.
        mean_vec = sub.mean(axis=0).sort_values(ascending=False)
        top_skills = mean_vec.head(int(top_n_skills)).index.astype(str).tolist()
        themes[str(cid)] = {"top_skills": top_skills}

    return themes


def build_skill_taxonomy(skills: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign skills to high-level groups using keyword rules.

    This is a lightweight default for small datasets and can be replaced with a richer taxonomy.
    """

    def group_for(skill: str) -> str:
        s = skill.strip().lower()

        # Keyword rules are intentionally simple and biased toward precision over recall.
        if any(k in s for k in ["security", "incident", "risk assessment"]):
            return "Security & Risk"
        if any(k in s for k in ["ux", "ui", "user research", "prototyp"]):
            return "Design & UX"
        if any(k in s for k in ["roadmap", "stakeholder", "strategy", "product"]):
            return "Product & Strategy"
        if any(k in s for k in ["git", "docker", "software design", "engineering"]):
            return "Engineering"
        if any(k in s for k in ["sql", "excel", "statistics", "machine learning", "data", "python", "visualization"]):
            return "Data & Analytics"

        return "Other"

    taxonomy = {str(skill): group_for(str(skill)) for skill in skills}

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
        "Product & Strategy": "Planning, alignment, stakeholder work, roadmap and prioritization.",
        "Design & UX": "Research, prototyping, UI/UX craft and iteration.",
        "Security & Risk": "Security controls, incident response, risk management basics.",
        "Other": "Uncategorized.",
    }

    group_meta = {"group_order": group_order, "group_descriptions": group_descriptions}
    return taxonomy, group_meta


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
    """Persist preprocessing outputs to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "occupation_skill_matrix.parquet"
    coords_path = out_dir / "pca_coords.parquet"

    matrix.to_parquet(matrix_path, index=True)
    coords.to_parquet(coords_path, index=False)

    # JSON files are intended for human inspection and lightweight downstream integrations.
    (out_dir / "occupations.json").write_text(json.dumps(matrix.index.tolist(), indent=2), encoding="utf-8")
    (out_dir / "skills.json").write_text(json.dumps(matrix.columns.tolist(), indent=2), encoding="utf-8")
    (out_dir / "pca_meta.json").write_text(json.dumps(pca_meta, indent=2), encoding="utf-8")
    (out_dir / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    # Optional artifacts are written only when provided to keep the output directory minimal.
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