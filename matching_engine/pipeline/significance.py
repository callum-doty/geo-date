"""
pipeline/significance.py
========================
Significance Multiplier computation and stream routing logic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.cluster import LocationCluster
    from matching_engine.models.ping import Ping, StreamAssignment

from matching_engine.models.ping import Stream, StreamAssignment


def _dwell_weight(dwell_minutes: float) -> float:
    """
    Log-scaled dwell weight.
    w = log10(1 + dwell_minutes / 30)
    30 min → 0.30  |  60 min → 0.48  |  300 min → 1.0
    """
    return math.log10(1.0 + max(dwell_minutes, 0) / 30.0)


def compute_significance(
    ping: "Ping",
    cluster: "LocationCluster",
    cfg: "MatchConfig",
    n_total_users: int = 1000,
) -> float:
    """
    Compute the Significance Multiplier S for a ping event.

    S = venue_rarity × event_duration_weight × dwell_intensity

    All three components are emergent — no manual tier assignment.
    """
    # Component 1 — Venue rarity
    n_window     = max(ping.n_users_in_window, 1)
    venue_rarity = math.log(max(n_total_users, 2) / n_window)

    # Component 2 — Event duration weight
    event_duration_weight = cluster.event_duration_weight()

    # Component 3 — Dwell intensity
    dwell_h = ping.dwell_hours()
    dwell_intensity = math.log10(1.0 + dwell_h / max(cluster.activation_days, 1.0))

    return venue_rarity * event_duration_weight * dwell_intensity


def route_ping(
    ping: "Ping",
    cluster: "LocationCluster",
    cfg: "MatchConfig",
    n_total_users: int = 1000,
) -> StreamAssignment:
    """
    Compute S and assign the ping to the correct stream.

    Parameters
    ----------
    ping : Ping
        Post-buffer, cluster-resolved ping.
    cluster : LocationCluster
        The resolved cluster.
    cfg : MatchConfig
    n_total_users : int
        Pass ClusterRegistry.n_total_users here.

    Returns
    -------
    StreamAssignment
    """
    S = compute_significance(ping, cluster, cfg, n_total_users=n_total_users)
    stream = Stream.IDENTITY if S >= cfg.s_threshold else Stream.RHYTHM
    weight = _dwell_weight(ping.dwell_minutes)
    return StreamAssignment(
        cluster_id=cluster.cluster_id,
        stream=stream,
        S=S,
        weight=weight,
    )
