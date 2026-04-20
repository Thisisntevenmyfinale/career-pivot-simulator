"""
Pivot Space Navigator
=====================
Transforms the O*NET occupation-skill matrix into a navigable career map.

Given two occupations (current + target), computes:
  - Skill-vector cosine distance (how different are the actual skill profiles?)
  - Cluster journey (e.g. "Data & Technical Systems → Engineering & Applied Science")
  - Bridge occupations — the 3 closest occupations that sit between current and target
    in skill-vector space, ranked by path-alignment score
  - Pivot distance grade (A=nearly identical → F=fundamentally different)
  - Visualization data for the interactive Plotly scatter

Why cosine distance (not Euclidean):
  Skill vectors are high-dimensional and sparse. Cosine similarity measures
  the *angle* between skill profiles, capturing "same shape of skills,
  different intensity" vs "completely different skill composition."

Bridge occupations:
  Bridge = occupations whose skill vector projects close to the geodesic between
  current and target. These are real stepping-stone roles: "Go from Marketing Manager
  to Digital Product Manager before going full PM" — not invented, pulled from the matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading (cached per session via caller's session_state)
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifacts() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]]:
    """Load matrix, UMAP coords, clusters, cluster themes from disk."""
    try:
        mat   = pd.read_parquet(ARTIFACTS_DIR / "occupation_skill_matrix.parquet")
        umap  = pd.read_parquet(ARTIFACTS_DIR / "umap_coords.parquet")
        with open(ARTIFACTS_DIR / "clusters.json")       as f: clusters = json.load(f)
        with open(ARTIFACTS_DIR / "cluster_themes.json") as f: themes   = json.load(f)
        return mat, umap, clusters, themes
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pivot_path(
    current_occ: str,
    target_occ: str,
    cluster_names: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Full pivot path analysis between two O*NET occupations.

    Args:
        current_occ:   occupation name (must be in the matrix index)
        target_occ:    occupation name (must be in the matrix index)
        cluster_names: {cluster_id: {name, tagline, ...}} from cluster_evaluator

    Returns dict with:
        pivot_distance         float 0-1 (0=identical, 1=maximum distance)
        pivot_distance_grade   "A"–"F"
        pivot_distance_label   human-readable ("Very Similar" → "Fundamentally Different")
        shared_skills          list of top shared high-importance skills
        missing_skills         top skills in target NOT in current (top 8)
        excess_skills          top skills current has that target doesn't need (top 5)
        bridge_occupations     list of 3 dicts {occupation, cluster, distance_from_current, distance_to_target, alignment_score}
        cluster_journey        "Data & Technical Systems → Engineering & Applied Science"
        current_cluster_id     str
        target_cluster_id      str
        umap_data              dict for Plotly visualization
        current_found          bool
        target_found           bool
    """
    arts = _load_artifacts()
    if arts is None:
        return None

    mat, umap_df, clusters, themes = arts

    # ── Fuzzy-match occupation names ─────────────────────────────────────────
    all_occs = mat.index.tolist()
    cur_match = _fuzzy_match(current_occ, all_occs)
    tgt_match = _fuzzy_match(target_occ,  all_occs)

    current_found = cur_match is not None
    target_found  = tgt_match is not None

    if not current_found or not target_found:
        return {
            "current_found": current_found,
            "target_found":  target_found,
            "current_occ":   current_occ,
            "target_occ":    target_occ,
            "umap_data":     _build_umap_data(umap_df, clusters, cluster_names, None, None),
        }

    cur_vec = mat.loc[cur_match].to_numpy(dtype=float)
    tgt_vec = mat.loc[tgt_match].to_numpy(dtype=float)

    # ── Cosine distance ───────────────────────────────────────────────────────
    cos_sim = _cosine_sim(cur_vec, tgt_vec)
    pivot_distance = float(1.0 - cos_sim)

    # ── Shared / missing / excess skills ─────────────────────────────────────
    skills = mat.columns.tolist()
    shared_skills  = _shared_skills(cur_vec, tgt_vec, skills)
    missing_skills = _missing_skills(cur_vec, tgt_vec, skills)
    excess_skills  = _missing_skills(tgt_vec, cur_vec, skills)  # flipped = excess

    # ── Cluster journey ───────────────────────────────────────────────────────
    cur_cluster = str(clusters.get(cur_match, "?"))
    tgt_cluster = str(clusters.get(tgt_match, "?"))

    def _cluster_name(cid: str) -> str:
        if cluster_names and cid in cluster_names:
            return cluster_names[cid]["name"]
        return themes.get(cid, {}).get("top_skills", [cid])[0] if cid != "?" else "Unknown"

    cluster_journey = f"{_cluster_name(cur_cluster)} → {_cluster_name(tgt_cluster)}"

    # ── Bridge occupations ───────────────────────────────────────────────────
    bridge_occs = _find_bridge_occupations(
        cur_vec, tgt_vec, cur_match, tgt_match, mat, clusters, cluster_names or {}, n=3
    )

    # ── UMAP visualization data ───────────────────────────────────────────────
    umap_data = _build_umap_data(
        umap_df, clusters, cluster_names, cur_match, tgt_match,
        bridge_names=[b["occupation"] for b in bridge_occs],
    )

    return {
        "current_occ":         cur_match,
        "target_occ":          tgt_match,
        "current_found":       True,
        "target_found":        True,
        "pivot_distance":      round(pivot_distance, 3),
        "pivot_distance_grade": _distance_grade(pivot_distance),
        "pivot_distance_label": _distance_label(pivot_distance),
        "cos_similarity":      round(float(cos_sim), 3),
        "shared_skills":       shared_skills[:8],
        "missing_skills":      missing_skills[:8],
        "excess_skills":       excess_skills[:5],
        "bridge_occupations":  bridge_occs,
        "cluster_journey":     cluster_journey,
        "current_cluster_id":  cur_cluster,
        "target_cluster_id":   tgt_cluster,
        "umap_data":           umap_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bridge occupation computation
# ─────────────────────────────────────────────────────────────────────────────

def _find_bridge_occupations(
    cur_vec: np.ndarray,
    tgt_vec: np.ndarray,
    cur_name: str,
    tgt_name: str,
    mat: pd.DataFrame,
    clusters: Dict,
    cluster_names: Dict,
    n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find occupations that lie 'between' current and target in skill-vector space.

    Alignment score = how well an occupation projects onto the cur→tgt direction.
    We want occupations that are:
      (a) closer to current than to target (reachable step)
      (b) closer to target than current occupation is (actual progress)
      (c) maximally aligned with the cur→tgt direction vector
    """
    direction = tgt_vec - cur_vec
    dir_norm  = np.linalg.norm(direction)
    if dir_norm < 1e-9:
        return []

    all_vecs = mat.to_numpy(dtype=float)
    all_occs = mat.index.tolist()

    results = []
    for i, occ in enumerate(all_occs):
        if occ in (cur_name, tgt_name):
            continue

        v = all_vecs[i]

        # Projection of (v - cur) onto (tgt - cur)
        rel = v - cur_vec
        proj = float(np.dot(rel, direction) / (dir_norm ** 2))

        # Only consider occupations that are "between" (proj 0.15 to 0.85)
        if proj < 0.15 or proj > 0.85:
            continue

        # Perpendicular distance from the line (how far off the direct path)
        perp_vec = rel - proj * direction
        perp_dist = float(np.linalg.norm(perp_vec))

        # Distance to current and target
        d_cur = float(1.0 - _cosine_sim(v, cur_vec))
        d_tgt = float(1.0 - _cosine_sim(v, tgt_vec))

        # Alignment score: reward being on-path, penalize perpendicular drift
        alignment = proj * (1.0 - min(perp_dist / (dir_norm + 1e-9), 1.0))

        cid = str(clusters.get(occ, "?"))
        results.append({
            "occupation":            occ,
            "cluster_id":            cid,
            "cluster_name":          cluster_names.get(cid, {}).get("name", cid) if cluster_names else cid,
            "distance_from_current": round(d_cur, 3),
            "distance_to_target":    round(d_tgt, 3),
            "alignment_score":       round(alignment, 3),
            "path_position":         round(proj, 2),   # 0=at current, 1=at target
        })

    results.sort(key=lambda x: x["alignment_score"], reverse=True)
    return results[:n]


# ─────────────────────────────────────────────────────────────────────────────
# UMAP visualization data builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_umap_data(
    umap_df: pd.DataFrame,
    clusters: Dict,
    cluster_names: Optional[Dict],
    current_occ: Optional[str],
    target_occ: Optional[str],
    bridge_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build Plotly-ready scatter data for the Occupation Space visualization."""
    bridge_names = set(bridge_names or [])
    special = {current_occ, target_occ} | bridge_names

    # Build cluster color map
    cluster_colors = {
        "0": "#4F8EF7",   # blue   — Data & Technical
        "1": "#F7A84F",   # orange — Skilled Trades
        "2": "#4FD196",   # green  — Engineering
        "3": "#B96FF7",   # purple — Service & Creative
        "?": "#888888",
    }

    def _name(cid: str) -> str:
        if cluster_names and cid in cluster_names:
            return cluster_names[cid]["name"]
        return f"Cluster {cid}"

    # Separate occupations into layers for Plotly (background → highlights)
    background, bridges, highlighted = [], [], []

    for _, row in umap_df.iterrows():
        occ = str(row["occupation"])
        cid = str(clusters.get(occ, "?"))
        color = cluster_colors.get(cid, "#888888")
        point = {
            "occ":   occ,
            "x":     float(row["x"]),
            "y":     float(row["y"]),
            "cid":   cid,
            "cname": _name(cid),
            "color": color,
        }
        if occ in (current_occ, target_occ):
            highlighted.append(point)
        elif occ in bridge_names:
            bridges.append(point)
        else:
            background.append(point)

    return {
        "background":  background,
        "bridges":     bridges,
        "highlighted": highlighted,
        "current_occ": current_occ,
        "target_occ":  target_occ,
        "cluster_colors": cluster_colors,
        "cluster_names": {
            cid: _name(cid) for cid in cluster_colors if cid != "?"
        },
    }


def get_all_occupations() -> List[str]:
    """Return sorted list of all occupation names in the matrix."""
    try:
        mat = pd.read_parquet(ARTIFACTS_DIR / "occupation_skill_matrix.parquet")
        return sorted(mat.index.tolist())
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _shared_skills(a: np.ndarray, b: np.ndarray, skills: List[str], threshold: float = 0.5) -> List[str]:
    """Skills where both vectors have importance >= threshold, sorted by min(a,b)."""
    shared = [
        (skills[i], min(a[i], b[i]))
        for i in range(len(skills))
        if a[i] >= threshold and b[i] >= threshold
    ]
    shared.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in shared]


def _missing_skills(a: np.ndarray, b: np.ndarray, skills: List[str], threshold: float = 0.5) -> List[str]:
    """Skills where b is high (>= threshold) but a is low (< threshold/2)."""
    missing = [
        (skills[i], b[i])
        for i in range(len(skills))
        if b[i] >= threshold and a[i] < threshold / 2
    ]
    missing.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in missing]


def _fuzzy_match(query: str, options: List[str]) -> Optional[str]:
    """Case-insensitive prefix/substring match. Returns None if no match."""
    q = query.strip().lower()
    # Exact match first
    for o in options:
        if o.lower() == q:
            return o
    # Prefix match
    for o in options:
        if o.lower().startswith(q):
            return o
    # Substring match
    for o in options:
        if q in o.lower():
            return o
    return None


def _distance_grade(d: float) -> str:
    if d < 0.10: return "A"
    if d < 0.25: return "B"
    if d < 0.40: return "C"
    if d < 0.60: return "D"
    return "F"


def _distance_label(d: float) -> str:
    if d < 0.10: return "Near-identical skill profiles"
    if d < 0.25: return "Very similar — smooth pivot"
    if d < 0.40: return "Moderately different — gaps manageable"
    if d < 0.60: return "Substantially different — 6-12 months to bridge"
    return "Fundamentally different — major reskilling required"
