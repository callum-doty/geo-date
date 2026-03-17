"""
models/cluster.py
=================
LocationCluster, ClusterObservation, and ClusterRegistry data structures.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig


@dataclass
class LocationCluster:
    """
    An emergent geographic place — the fundamental unit of the vector space.

    In v4.0 clusters are also self-registering venue candidates. No separate
    Venue catalog exists. A cluster graduates to venue-eligible status when
    it clears the corpus-level gate (n_users ≥ 5, n_total_visits ≥ 10) and
    its Bayesian suitability posterior is above a meaningful floor.
    """
    cluster_id:             str
    centroid_lat:           float
    centroid_lng:           float
    soft_label:             str   = "Unknown Place"
    places_category:        str   = "unknown"
    n_users:                int   = 1
    n_total_visits:         int   = 1
    activation_days:        float = 365.0
    is_event:               bool  = False
    created_at:             float = field(default_factory=time.time)

    # Behavioral suitability accumulators
    evening_visits:         int   = 0
    cooccurrence_visits:    int   = 0
    date_dwell_visits:      int   = 0

    def idf(self, n_total_users: int) -> float:
        if self.n_users == 0 or n_total_users == 0:
            return 0.0
        return math.log(max(n_total_users, 1) / max(self.n_users, 1))

    def event_duration_weight(self) -> float:
        return 1.0 / math.log(1.0 + max(self.activation_days, 1.0))


@dataclass
class ClusterObservation:
    """
    A single behavioral observation used to update a cluster's suitability.
    """
    cluster_id:             str
    timestamp:              float
    dwell_minutes:          float
    concurrent_user_ids:    list[str] = field(default_factory=list)


def _hour_from_timestamp(ts: float) -> int:
    """Extract local hour (0–23) from a Unix timestamp. Uses UTC as proxy."""
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).hour


@dataclass
class ClusterRegistry:
    """
    The live dynamic cluster space.

    In production this is backed by a database.
    In-memory dict keyed by cluster_id here.
    """
    clusters:               dict[str, LocationCluster] = field(default_factory=dict)
    n_total_users:          int = 1000
    idf_snapshot:           dict[str, float] = field(default_factory=dict)
    venue_eligible_ids:     set[str] = field(default_factory=set)

    def add_cluster(self, cluster: LocationCluster) -> None:
        self.clusters[cluster.cluster_id] = cluster

    def get(self, cluster_id: str) -> Optional[LocationCluster]:
        return self.clusters.get(cluster_id)

    def refresh_idf(self) -> None:
        self.idf_snapshot = {
            cid: c.idf(self.n_total_users)
            for cid, c in self.clusters.items()
        }

    def idf_for(self, cluster_id: str) -> float:
        if cluster_id in self.idf_snapshot:
            return self.idf_snapshot[cluster_id]
        c = self.clusters.get(cluster_id)
        return c.idf(self.n_total_users) if c else 0.0

    def refresh_eligibility(self, cfg: "MatchConfig") -> None:
        """
        Recompute the set of venue-eligible cluster IDs.

        A cluster is eligible as a date venue recommendation when:
            n_users        ≥ cfg.venue_min_users   (not a private location)
            n_total_visits ≥ cfg.venue_min_visits  (sufficient evidence)
        """
        self.venue_eligible_ids = {
            cid for cid, c in self.clusters.items()
            if c.n_users >= cfg.venue_min_users
            and c.n_total_visits >= cfg.venue_min_visits
        }

    def record_observation(
        self,
        obs: ClusterObservation,
        cfg: "MatchConfig",
    ) -> None:
        """
        Incrementally update a cluster's behavioral suitability accumulators.
        """
        c = self.clusters.get(obs.cluster_id)
        if c is None:
            return
        c.n_total_visits += 1
        # Temporal: is this an evening visit?
        local_hour = _hour_from_timestamp(obs.timestamp)
        if cfg.evening_start_hour <= local_hour < cfg.evening_end_hour:
            c.evening_visits += 1
        # Co-occurrence: did another user show up within the window?
        if obs.concurrent_user_ids:
            c.cooccurrence_visits += 1
        # Dwell shape: is this a date-length visit?
        if cfg.dwell_date_min <= obs.dwell_minutes <= cfg.dwell_date_max:
            c.date_dwell_visits += 1
