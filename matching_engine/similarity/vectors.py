"""
similarity/vectors.py
=====================
IDF normalization, sparse cosine similarity, and bio similarity.
"""

from __future__ import annotations

import math


def _base_cluster_id(dimension_key: str) -> str:
    """
    Extract the cluster_id from a dimension key.

    Handles both formats:
      New: "cluster_id|WD_2"  → "cluster_id"
      Legacy: "cluster_id"    → "cluster_id"
    """
    return dimension_key.split("|", 1)[0]


def idf_normalize_sparse(
    V: dict[str, float],
    idf_map: dict[str, float],
) -> dict[str, float]:
    """
    Apply IDF weighting then unit-normalise a sparse vector.

    Step 1 — IDF:   V*[k] = V[k] · IDF(base_cluster_id(k))
    Step 2 — Norm:  V̂*[k] = V*[k] / ‖V*‖

    Dimension keys may be "cluster_id" (legacy) or "cluster_id|time_bin" (v5).
    IDF is always looked up on the base cluster_id so that all time-bin
    dimensions of the same cluster share the same IDF weight.

    Result: all components bounded [0, 1], dot product = cosine similarity.
    """
    if not V:
        return {}
    V_star = {
        k: v * idf_map.get(_base_cluster_id(k), 0.0)
        for k, v in V.items() if v > 0
    }
    norm = math.sqrt(sum(v * v for v in V_star.values()))
    if norm < 1e-10:
        return {}
    return {k: v / norm for k, v in V_star.items()}


def cosine_sparse(
    V_a: dict[str, float],
    V_b: dict[str, float],
    idf_map: dict[str, float],
) -> float:
    """
    Cosine similarity between two sparse vectors after IDF normalisation.

    Since both are unit-normalised, similarity = dot product over shared keys.
    Returns 0 if either vector is empty (cold start / no data for this stream).

    Returns
    -------
    float ∈ [0, 1]
    """
    a_norm = idf_normalize_sparse(V_a, idf_map)
    b_norm = idf_normalize_sparse(V_b, idf_map)
    if not a_norm or not b_norm:
        return 0.0
    # Dot product over intersection of keys
    shared = set(a_norm.keys()) & set(b_norm.keys())
    dot = sum(a_norm[k] * b_norm[k] for k in shared)
    return float(min(max(dot, 0.0), 1.0))


def bio_similarity(tags_a: list[str], tags_b: list[str]) -> float:
    """
    Jaccard overlap of stated interest tag sets.

    Sim_prior(A, B) = |T_A ∩ T_B| / |T_A ∪ T_B|

    Fallback signal at low ping count. Authority decays via w_bio schedule.

    Returns
    -------
    float ∈ [0, 1]
    """
    s_a, s_b = set(tags_a), set(tags_b)
    union = s_a | s_b
    if not union:
        return 0.0
    return float(len(s_a & s_b) / len(union))
