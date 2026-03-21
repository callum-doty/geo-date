"""
pipeline/transitions.py
=======================
Transition matrix update logic.

The transition matrix E captures how users move between behavioral clusters.
Keys use the → separator: "from_dim_key→to_dim_key"
e.g. "v1_3|WD_2→v1_7|WD_2"  (gym → coffee, weekday morning)

This encodes behavioral pathways — not just where users go, but how they
move between places — enabling matching on structural movement patterns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig

# Separator between from/to keys in transition edge keys
TRANSITION_SEP = "→"


def _transition_key(from_key: str, to_key: str) -> str:
    return f"{from_key}{TRANSITION_SEP}{to_key}"


def decay_transitions(
    E: dict[str, float],
    delta_t: float,
    lambda_rate: float,
) -> dict[str, float]:
    """
    Apply exponential decay to all edges in the transition matrix.

    Edges below the sparsity floor are pruned to keep E compact.
    """
    if delta_t <= 0:
        return dict(E)
    factor = math.exp(-lambda_rate * delta_t)
    return {k: v * factor for k, v in E.items() if v * factor > 1e-6}


def update_transition_matrix(
    E: dict[str, float],
    from_key: str,
    to_key: str,
    weight: float,
    delta_t: float,
    cfg: "MatchConfig",
) -> dict[str, float]:
    """
    Decay E then record a new from→to transition.

    E[from→to](t) = E[from→to](t-1) · exp(-λ_transition · Δt) + weight

    Parameters
    ----------
    E        : current transition matrix (mutated copy returned)
    from_key : dimension key of the preceding cluster
    to_key   : dimension key of the current cluster
    weight   : dwell-based weight of the current ping
    delta_t  : days since last ping (for decay)
    cfg      : MatchConfig

    Returns
    -------
    Updated transition matrix dict.
    """
    E_decayed = decay_transitions(E, delta_t, cfg.lambda_transition)
    edge = _transition_key(from_key, to_key)
    E_decayed[edge] = E_decayed.get(edge, 0.0) + weight
    return E_decayed
