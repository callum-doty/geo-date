"""
similarity/transitions.py
=========================
Edge similarity — cosine similarity over sparse transition matrices.

The transition matrix E encodes behavioral pathways between clusters.
Comparing two users' E matrices reveals structural movement similarity:
"gym → coffee on weekday mornings" matched against the same pattern.

Edge keys use the → separator defined in pipeline/transitions.py.
IDF is applied at the base cluster level of the from-node.
"""

from __future__ import annotations

import math


def edge_similarity(
    E_a: dict[str, float],
    E_b: dict[str, float],
    idf_map: dict[str, float] | None = None,
) -> float:
    """
    Cosine similarity between two sparse transition matrices.

    Each matrix is treated as a flat vector over edge keys.
    IDF weighting is applied on the from-node's base cluster_id,
    so rare behavioral transitions contribute more than common ones
    (e.g. "home → jazz_club" outweighs "home → coffee_shop").

    Parameters
    ----------
    E_a, E_b  : sparse transition dicts {"from_key→to_key": weight}
    idf_map   : optional cluster-level IDF weights; if None, unweighted

    Returns
    -------
    float ∈ [0, 1]
    """
    if not E_a or not E_b:
        return 0.0

    def _idf(edge_key: str) -> float:
        if idf_map is None:
            return 1.0
        from_key = edge_key.split("→", 1)[0]
        base_cluster = from_key.split("|", 1)[0]
        return idf_map.get(base_cluster, 1.0)

    # IDF-weight and unit-normalise E_a
    a_star = {k: v * _idf(k) for k, v in E_a.items() if v > 0}
    b_star = {k: v * _idf(k) for k, v in E_b.items() if v > 0}

    norm_a = math.sqrt(sum(v * v for v in a_star.values()))
    norm_b = math.sqrt(sum(v * v for v in b_star.values()))

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    a_norm = {k: v / norm_a for k, v in a_star.items()}
    b_norm = {k: v / norm_b for k, v in b_star.items()}

    shared = set(a_norm.keys()) & set(b_norm.keys())
    dot = sum(a_norm[k] * b_norm[k] for k in shared)
    return float(min(max(dot, 0.0), 1.0))
