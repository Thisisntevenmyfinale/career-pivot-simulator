# scripts/preprocess_dummy.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so `import src...` works reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (  # noqa: E402
    build_occupation_matrix,
    build_skill_taxonomy_dummy,
    compute_cluster_themes,
    compute_data_quality,
    compute_pca_coords,
    compute_role_clusters_kmeans,
    save_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dummy skills CSV into runtime artifacts.")
    parser.add_argument("--input", type=str, default="data/skills_long.csv", help="Path to long-format CSV")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory for artifacts")
    parser.add_argument("--n-clusters", type=int, default=0, help="KMeans clusters (0 = auto)")
    args = parser.parse_args()

    in_path = (REPO_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out_dir = (REPO_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_occupation_matrix(in_path)
    coords, pca_meta = compute_pca_coords(matrix, n_components=2, random_state=42)
    quality = compute_data_quality(matrix)

    # --- Feature A: Role clustering
    n_clusters = None if args.n_clusters <= 0 else int(args.n_clusters)
    clusters, cluster_meta = compute_role_clusters_kmeans(matrix, n_clusters=n_clusters, random_state=42)
    cluster_themes = compute_cluster_themes(matrix, clusters, top_n_skills=6)

    # --- Feature B: Skill taxonomy groups
    skills = matrix.columns.astype(str).tolist()
    skill_taxonomy, group_meta = build_skill_taxonomy_dummy(skills)

    save_artifacts(
        out_dir=out_dir,
        matrix=matrix,
        coords=coords,
        pca_meta=pca_meta,
        quality=quality,
        clusters=clusters,
        cluster_meta=cluster_meta,
        cluster_themes=cluster_themes,
        skill_taxonomy=skill_taxonomy,
        group_meta=group_meta,
    )

    print("✅ Preprocessing complete.")
    print(f"- Matrix: {out_dir / 'occupation_skill_matrix.parquet'}")
    print(f"- Coords:  {out_dir / 'pca_coords.parquet'}")
    print(f"- Meta:    {out_dir / 'pca_meta.json'}")
    print(f"- Quality: {out_dir / 'data_quality.json'}")
    print(f"- Lists:   {out_dir / 'occupations.json'} / {out_dir / 'skills.json'}")
    print(f"- Clusters: {out_dir / 'clusters.json'} / {out_dir / 'cluster_meta.json'} / {out_dir / 'cluster_themes.json'}")
    print(f"- Taxonomy: {out_dir / 'skill_taxonomy.json'} / {out_dir / 'group_meta.json'}")


if __name__ == "__main__":
    main()