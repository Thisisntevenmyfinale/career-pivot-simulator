# scripts/preprocess_onet.py
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# make project root importable (so "import src..." works when running from /scripts)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    compute_pca_coords,
    compute_role_clusters_kmeans,
)

# Optional: UMAP only if installed + function exists
try:
    from src.preprocessing import compute_umap_coords  # type: ignore
    HAS_UMAP_FN = True
except Exception:
    HAS_UMAP_FN = False


DATA_DIR = Path("data/onet_raw")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)


def load_onet_file(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, sep="\t", dtype=str)


def clean_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_matrix() -> pd.DataFrame:
    print("Loading O*NET files...")

    skills = load_onet_file("Skills.txt")
    knowledge = load_onet_file("Knowledge.txt")
    abilities = load_onet_file("Abilities.txt")
    occupations = load_onet_file("Occupation Data.txt")

    # Keep only importance ratings
    skills = skills[skills["Scale ID"] == "IM"]
    knowledge = knowledge[knowledge["Scale ID"] == "IM"]
    abilities = abilities[abilities["Scale ID"] == "IM"]

    skills = clean_numeric(skills, "Data Value")
    knowledge = clean_numeric(knowledge, "Data Value")
    abilities = clean_numeric(abilities, "Data Value")

    print("Merging skill domains...")

    all_skills = pd.concat(
        [
            skills[["O*NET-SOC Code", "Element Name", "Data Value"]],
            knowledge[["O*NET-SOC Code", "Element Name", "Data Value"]],
            abilities[["O*NET-SOC Code", "Element Name", "Data Value"]],
        ],
        ignore_index=True,
    )

    # Pivot to Occupation × Skill matrix
    matrix = all_skills.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean",
    ).fillna(0.0)

    print("Attaching occupation titles...")

    occ_titles = occupations[["O*NET-SOC Code", "Title"]].drop_duplicates()
    matrix = matrix.merge(
        occ_titles,
        left_index=True,
        right_on="O*NET-SOC Code",
        how="left",
    )

    matrix = matrix.set_index("Title").drop(columns=["O*NET-SOC Code"])

    # Safety: stable ordering
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    return matrix


def main() -> None:
    matrix = build_matrix()

    print("Saving matrix...")
    matrix.to_parquet(OUT_DIR / "occupation_skill_matrix.parquet")
    print("Done.")
    print(f"Shape: {matrix.shape}")

    # -----------------------------
    # PCA artifacts (needed for Map tab)
    # -----------------------------
    print("Computing PCA coords...")
    coords, pca_meta = compute_pca_coords(matrix)
    coords.to_parquet(OUT_DIR / "pca_coords.parquet", index=False)
    (OUT_DIR / "pca_meta.json").write_text(json.dumps(pca_meta, indent=2), encoding="utf-8")
    print("Saved PCA artifacts:", coords.shape)
    

        # -----------------------------
    # UMAP coords (optional "overkill")
    # -----------------------------
    try:
        print("Computing UMAP coords...")
        from src.preprocessing import compute_umap_coords

        umap_coords, umap_meta = compute_umap_coords(
            matrix,
            n_components=2,
            n_neighbors=25,
            min_dist=0.10,
            metric="cosine",
            random_state=42,
        )

        umap_coords.to_parquet(OUT_DIR / "umap_coords.parquet", index=False)
        (OUT_DIR / "umap_meta.json").write_text(json.dumps(umap_meta, indent=2), encoding="utf-8")

        print("Saved UMAP artifacts:", umap_coords.shape)
    except Exception as e:
        print("UMAP skipped (install or runtime issue):", repr(e))

    # -----------------------------
    # Clustering artifacts
    # -----------------------------
    print("Computing clusters (KMeans)...")
    clusters, cluster_meta = compute_role_clusters_kmeans(matrix, n_clusters=4, random_state=42)
    (OUT_DIR / "clusters.json").write_text(json.dumps(clusters, indent=2), encoding="utf-8")
    (OUT_DIR / "cluster_meta.json").write_text(json.dumps(cluster_meta, indent=2), encoding="utf-8")
    print("Saved clusters artifacts:", len(clusters), "roles")
    print("cluster sizes:", cluster_meta.get("cluster_sizes"))

    # -----------------------------
    # Optional UMAP (only if function exists)
    # -----------------------------
    if HAS_UMAP_FN:
        print("Computing UMAP coords...")
        umap_coords, umap_meta = compute_umap_coords(matrix)
        umap_coords.to_parquet(OUT_DIR / "umap_coords.parquet", index=False)
        (OUT_DIR / "umap_meta.json").write_text(json.dumps(umap_meta, indent=2), encoding="utf-8")
        print("Saved UMAP artifacts:", umap_coords.shape)
    else:
        print("UMAP skipped (compute_umap_coords not found in src.preprocessing).")


if __name__ == "__main__":
    main()