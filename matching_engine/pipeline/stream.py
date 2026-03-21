"""
pipeline/stream.py
==================
Vector decay, stream update, and pinning logic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.cluster import LocationCluster
    from matching_engine.models.user import PinnedWeight


def decay_stream(
    V: dict[str, float],
    delta_t: float,
    lambda_rate: float,
) -> dict[str, float]:
    """
    Apply exponential decay to all non-zero dimensions of a sparse vector.

    V[k](t) = V[k](t-1) · exp(-λ · Δt)

    Zero-valued dimensions are pruned.
    """
    if delta_t <= 0:
        return dict(V)
    factor = math.exp(-lambda_rate * delta_t)
    return {k: v * factor for k, v in V.items() if v * factor > 1e-6}


def apply_rhythm_ping(
    V_rhythm: dict[str, float],
    cluster_id: str,
    weight: float,
    delta_t: float,
    cfg: "MatchConfig",
    time_bin: str = "",
) -> dict[str, float]:
    """
    Decay V_rhythm then add a new rhythm ping.

    Dimension key: "cluster_id|time_bin" when time_bin is set,
                   "cluster_id" for legacy/no-time-bin paths.

    V_rhythm[k](t) = V_rhythm[k](t-1) · exp(-λ_rhythm · Δt) + weight
    """
    from matching_engine.models.ping import make_dimension_key
    V = decay_stream(V_rhythm, delta_t, cfg.lambda_rhythm)
    dim_key = make_dimension_key(cluster_id, time_bin)
    V[dim_key] = V.get(dim_key, 0.0) + weight
    return V


def apply_identity_ping(
    V_identity: dict[str, float],
    cluster_id: str,
    weight: float,
    t_ping: float,
    cfg: "MatchConfig",
    cluster: Optional["LocationCluster"] = None,
    pins: Optional[dict[str, "PinnedWeight"]] = None,
    time_bin: str = "",
) -> dict[str, float]:
    """
    Decay V_identity then add a new identity ping, with pinning logic.

    For event clusters (activation_days ≤ 7), the peak weight is pinned
    at pin_floor for pin_duration_days to prevent seasonal washout.

    Pins are keyed by the full dimension key so that time-bucketed identity
    events are pinned independently.
    """
    from matching_engine.models.ping import make_dimension_key
    V = decay_stream(V_identity, 0, cfg.lambda_identity)  # decay handled externally
    dim_key = make_dimension_key(cluster_id, time_bin)
    new_val = V.get(dim_key, 0.0) + weight
    V[dim_key] = new_val

    # Apply identity pin for concentrated events
    if cluster is not None and pins is not None:
        from matching_engine.models.user import PinnedWeight
        is_event_cluster = cluster.is_event or cluster.activation_days <= 7
        if is_event_cluster:
            pins[dim_key] = PinnedWeight(
                peak_weight=new_val,
                pinned_at=t_ping,
                pin_duration=cfg.pin_duration_days,
                pin_floor=cfg.pin_floor,
            )
    return V


def enforce_pins(
    V_identity: dict[str, float],
    pins: dict[str, "PinnedWeight"],
) -> dict[str, float]:
    """
    Enforce pin floors on V_identity after any decay step.

    For each active pin, ensure V_identity[k] ≥ pin.floor_value().
    Expired pins are removed from the pin registry.
    """
    V = dict(V_identity)
    expired = []
    for cluster_id, pin in pins.items():
        if pin.is_active():
            floor_val = pin.floor_value()
            V[cluster_id] = max(V.get(cluster_id, 0.0), floor_val)
        else:
            expired.append(cluster_id)
    for cid in expired:
        del pins[cid]
    return V
