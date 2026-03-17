"""
models/user.py
==============
UserProfile and PinnedWeight data structures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.cluster import LocationCluster, ClusterRegistry
    from matching_engine.models.ping import Ping, StreamAssignment, Stream


@dataclass
class PinnedWeight:
    """
    Tracks the identity pin for a single cluster dimension.

    When a high-S ping lands in V_identity and the cluster is an event
    (activation_days ≤ 7), the weight is pinned above pin_floor for
    pin_duration_days, regardless of decay.
    """
    peak_weight:    float       # weight at the time the pin was set
    pinned_at:      float       # unix timestamp
    pin_duration:   float       # days
    pin_floor:      float       # fraction of peak_weight held as minimum

    def floor_value(self) -> float:
        return self.peak_weight * self.pin_floor

    def is_active(self) -> bool:
        elapsed_days = (time.time() - self.pinned_at) / 86400
        return elapsed_days <= self.pin_duration


@dataclass
class UserProfile:
    """
    A user's complete dual-stream behavioral state.
    """
    user_id:            str
    V_rhythm:           dict[str, float] = field(default_factory=dict)
    V_identity:         dict[str, float] = field(default_factory=dict)
    pins:               dict[str, PinnedWeight] = field(default_factory=dict)
    n_rhythm:           int = 0
    n_identity:         int = 0
    tags:               list[str] = field(default_factory=list)
    home_base_lat:      Optional[float] = None
    home_base_lng:      Optional[float] = None
    home_base_commute:  float = 20.0
    rhythm_ping_coords: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_rhythm + self.n_identity

    def add_ping(
        self,
        assignment: "StreamAssignment",
        ping: "Ping",
        cluster: "LocationCluster",
        cfg: "MatchConfig",
        registry: Optional["ClusterRegistry"] = None,
    ) -> None:
        """
        Apply a routed ping to the appropriate stream vector.
        Mutates the profile in-place.
        """
        from matching_engine.models.ping import Stream
        from matching_engine.pipeline.stream import apply_rhythm_ping, apply_identity_ping
        from matching_engine.similarity.proximity import geometric_median

        if assignment.stream == Stream.RHYTHM:
            self.V_rhythm = apply_rhythm_ping(
                self.V_rhythm, assignment.cluster_id,
                assignment.weight, ping.delta_t_days, cfg
            )
            self.n_rhythm += 1
            # Track coords for HomeBase (rhythm only)
            self.rhythm_ping_coords.append((ping.lat, ping.lng))
            # Recompute HomeBase every 10 rhythm pings
            if self.n_rhythm % 10 == 0 and len(self.rhythm_ping_coords) >= 3:
                lat, lng = geometric_median(self.rhythm_ping_coords)
                self.home_base_lat  = lat
                self.home_base_lng  = lng

        elif assignment.stream == Stream.IDENTITY:
            self.V_identity = apply_identity_ping(
                self.V_identity, assignment.cluster_id,
                assignment.weight, ping.timestamp, cfg,
                cluster=cluster, pins=self.pins
            )
            self.n_identity += 1
