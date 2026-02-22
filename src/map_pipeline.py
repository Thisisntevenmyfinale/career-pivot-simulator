# src/map_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MapArtifacts:
    """Artifacts produced by the map pipeline."""
    matrix: pd.DataFrame           # occupation x skill
    coords: pd.DataFrame           # occupation, x, y (PCA coords)
    explained_variance_ratio: np.ndarray


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

    # Basic cleanup
    df = df.copy()
    df["occupation"] = df["occupation"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")

    df = df.dropna(subset=["occupation", "skill", "importance"])

    # If duplicates exist, average them (safe default)
    df = df.groupby(["occupation", "skill"], as_index=False)["importance"].mean()

    # Pivot to wide
    mat = df.pivot(index="occupation", columns="skill", values="importance").fillna(0.0)

    # Sort for stable output
    mat = mat.sort_index(axis=0).sort_index(axis=1)

    return mat


def compute_map_pca(matrix: pd.DataFrame, n_components: int = 2, random_state: int = 42) -> MapArtifacts:
    """
    Compute a 2D PCA embedding of occupation vectors.
    - Standardizes features to prevent high-variance skills dominating.
    """
    if matrix.shape[0] < 2:
        raise ValueError("Need at least 2 occupations to compute a map.")
    if matrix.shape[1] < 2:
        raise ValueError("Need at least 2 skills to compute a 2D PCA map.")

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

    return MapArtifacts(
        matrix=matrix,
        coords=coords,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_match_and_gap(
    matrix: pd.DataFrame,
    current_occ: str,
    target_occ: str,
) -> tuple[float, pd.DataFrame]:
    """
    Returns:
      - match_score: 0..100 based on cosine similarity of skill vectors
      - gap_df: per-skill breakdown with current/target importance + gap
    """
    if current_occ not in matrix.index:
        raise ValueError(f"Current occupation not in matrix: {current_occ}")
    if target_occ not in matrix.index:
        raise ValueError(f"Target occupation not in matrix: {target_occ}")

    current = matrix.loc[current_occ].astype(float)
    target = matrix.loc[target_occ].astype(float)

    # gap: positive => missing for target, negative => surplus vs target
    gap = (target - current)

    gap_df = pd.DataFrame({
        "skill": matrix.columns.astype(str),
        "current_importance": current.values,
        "target_importance": target.values,
        "gap": gap.values,
    })

    # cosine similarity for match score
    sim = cosine_similarity(current.values, target.values)  # [-1,1]
    match_score = max(0.0, sim) * 100.0  # clamp negative to 0, scale to 0..100

    # helpful derived columns
    gap_df["abs_gap"] = gap_df["gap"].abs()
    gap_df = gap_df.sort_values(["gap", "target_importance"], ascending=[False, False])

    return match_score, gap_df

def generate_explanation(
    current_occ: str,
    target_occ: str,
    match_score: float,
    gap_df: pd.DataFrame,
) -> str:
    """
    Generates a short human-readable explanation of the pivot.
    """

    top_gaps = (
        gap_df[gap_df["gap"] > 0]
        .sort_values(["gap", "target_importance"], ascending=False)
        .head(3)
    )

    top_transfer = (
        gap_df.sort_values(
            gap_df["current_importance"] * gap_df["target_importance"],
            ascending=False
        )
        .head(3)
    )

    explanation = f"The similarity between {current_occ} and {target_occ} is {match_score:.0f}/100. "

    if match_score < 30:
        explanation += "The roles are structurally quite different. "
    elif match_score < 60:
        explanation += "There is moderate overlap between the roles. "
    else:
        explanation += "The roles are strongly related. "

    if not top_gaps.empty:
        explanation += "Key missing skills include: "
        explanation += ", ".join(top_gaps["skill"].tolist()) + ". "

    explanation += "Strong transferable skills include: "
    explanation += ", ".join(top_transfer["skill"].tolist()) + "."

    return explanation

def compute_match_score_hybrid(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    target_occ: str,
    w_cosine: float = 0.6,
    w_map: float = 0.4,
) -> float:
    """
    Hybrid score: cosine similarity (skill space) + normalized distance (map space).
    Returns 0..100.
    """
    # Cosine part
    current = matrix.loc[current_occ].astype(float).values
    target = matrix.loc[target_occ].astype(float).values
    sim = cosine_similarity(current, target)  # [-1,1]
    cosine_part = max(0.0, sim)              # 0..1

    # Map distance part (PCA coordinates)
    c_xy = coords.loc[coords["occupation"] == current_occ, ["x", "y"]].to_numpy()
    t_xy = coords.loc[coords["occupation"] == target_occ, ["x", "y"]].to_numpy()
    if len(c_xy) == 0 or len(t_xy) == 0:
        map_part = 0.0
    else:
        dist = float(np.linalg.norm(c_xy[0] - t_xy[0]))

        # normalize by max pairwise distance in current dataset
        all_xy = coords[["x", "y"]].to_numpy()
        max_dist = 0.0
        for i in range(len(all_xy)):
            for j in range(i + 1, len(all_xy)):
                d = float(np.linalg.norm(all_xy[i] - all_xy[j]))
                if d > max_dist:
                    max_dist = d

        if max_dist == 0:
            map_part = 1.0
        else:
            map_part = 1.0 - (dist / max_dist)  # 0..1 (close => high)

    score = 100.0 * (w_cosine * cosine_part + w_map * map_part)
    return float(np.clip(score, 0.0, 100.0))

def get_top_similar_roles(
    matrix: pd.DataFrame,
    coords: pd.DataFrame,
    current_occ: str,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Returns top_k most similar roles to current_occ based on hybrid score.
    """

    from math import isnan

    roles = []
    for occ in matrix.index:
        if occ == current_occ:
            continue

        score = compute_match_score_hybrid(matrix, coords, current_occ, occ)

        if not isnan(score):
            roles.append((occ, score))

    roles_df = (
        pd.DataFrame(roles, columns=["occupation", "match_score"])
        .sort_values("match_score", ascending=False)
        .head(top_k)
    )

    return roles_df