"""
similarity/proximity.py
=======================
Proximity scoring and geometric median calculation.
"""

from __future__ import annotations

import math
import numpy as np


def proximity_score(commute_minutes: float, sigma: float = 25.0) -> float:
    """
    RBF kernel on commute time.

    Sim_log(d) = exp(-d² / 2σ²)

    d = average commute time (minutes) between the two users' HomeBases.
    Commute time, not Euclidean distance. 10 miles in KC ≠ 10 miles in NYC.

    City-specific sigma (commute minutes):
        Kansas City : σ = 25
        New York    : σ = 12
        Los Angeles : σ = 30

    Returns
    -------
    float ∈ (0, 1]
    """
    return float(math.exp(-(commute_minutes ** 2) / (2.0 * sigma ** 2)))


def geometric_median(
    coords: list[tuple[float, float]],
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute the geometric median of a set of (lat, lng) coordinates
    using Weiszfeld's algorithm.

    More robust than the arithmetic mean — a single airport ping does not
    relocate a user's HomeBase. Only V_rhythm ping coords are passed here;
    identity (festival) pings are excluded.

    Parameters
    ----------
    coords : list[tuple[float, float]]
        List of (lat, lng) pairs.
    max_iter : int
        Maximum iterations (default 300).
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple[float, float]
        (lat, lng) of the geometric median.
    """
    if not coords:
        return (0.0, 0.0)
    if len(coords) == 1:
        return coords[0]

    pts = np.array(coords, dtype=float)

    # Initialise at arithmetic mean
    median = pts.mean(axis=0)

    for _ in range(max_iter):
        dists = np.linalg.norm(pts - median, axis=1)
        # Avoid division by zero for points exactly at median
        nonzero = dists > 1e-10
        if not np.any(nonzero):
            break
        safe_dists = np.where(nonzero, dists, 1.0)          # avoid division by zero
        weights = np.where(nonzero, 1.0 / safe_dists, 0.0)
        new_median = (pts[nonzero] * weights[nonzero, np.newaxis]).sum(axis=0) \
                     / weights[nonzero].sum()
        if np.linalg.norm(new_median - median) < tol:
            median = new_median
            break
        median = new_median

    return (float(median[0]), float(median[1]))
