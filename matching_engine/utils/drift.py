"""
utils/drift.py
==============
Match Invalidation — Drift Detection.

Determines whether a user's vector has changed enough to invalidate
cached match scores. Compares IDF-normalised Euclidean distance between
old and new vectors.
"""

from __future__ import annotations

import math

from matching_engine.similarity.vectors import idf_normalize_sparse


def vector_has_drifted(
    V_old: dict[str, float],
    V_new: dict[str, float],
    idf_map: dict[str, float],
    drift_threshold: float = 0.05,
) -> bool:
    """
    True if ‖V̂*_old - V̂*_new‖ > drift_threshold.

    If cached G scores for this user should be recomputed, returns True.

    Parameters
    ----------
    V_old, V_new : dict[str, float]
        Raw sparse vectors before and after the latest ping.
    idf_map : dict[str, float]
        Current IDF snapshot from ClusterRegistry.
    drift_threshold : float
        Default 0.05. Recommended range: [0.03, 0.10].
        Lower = more aggressive recomputation (fresher scores, more compute).
        Higher = more caching tolerance (staler scores, less compute).

    Returns
    -------
    bool
    """
    old_norm = idf_normalize_sparse(V_old, idf_map)
    new_norm = idf_normalize_sparse(V_new, idf_map)

    all_keys = set(old_norm) | set(new_norm)
    if not all_keys:
        return False

    sq_dist = sum(
        (new_norm.get(k, 0.0) - old_norm.get(k, 0.0)) ** 2
        for k in all_keys
    )
    return math.sqrt(sq_dist) > drift_threshold