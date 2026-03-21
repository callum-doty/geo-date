"""
models/cluster.py
=================
LocationCluster, H3CellRecord, ClusterObservation, and ClusterRegistry.

v5.0 — Dual-Layer Clustering
------------------------------
Layer 1 (Spatial): H3 hexagonal grid provides a universal spatial primitive.
Layer 2 (Behavioral): LocationCluster is a behavioral archetype derived by
    clustering H3 cell behavior signatures across the entire user population.

Pipeline:
    GPS → H3 cell (on-device)
        → H3CellRecord.signature updated (server, aggregate)
        → cluster_id assigned via ClusterRegistry.cell_mapping (server)
        → atomic event: (cluster_id, time_bin, S)

Cluster IDs are version-stamped ("v1_42") so that periodic reclustering
does not corrupt active user vectors — old dimensions decay out naturally
via the existing half-life mechanism.
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
    A behavioral archetype — the fundamental unit of the vector space.

    Identity comes from the behavior signature centroid, not geography.
    centroid_lat/lng are retained as approximate display coordinates only;
    they play no role in matching or cluster identity.

    cluster_id format: "v{version}_{archetype_id}" e.g. "v1_3"
    Legacy string IDs (e.g. "gym_crossroads") remain valid for backward compat.
    """
    cluster_id:             str
    centroid_lat:           float = 0.0   # display only
    centroid_lng:           float = 0.0   # display only
    soft_label:             str   = "Unknown Place"
    places_category:        str   = "unknown"
    n_users:                int   = 1
    n_total_visits:         int   = 1
    activation_days:        float = 365.0
    is_event:               bool  = False
    created_at:             float = field(default_factory=time.time)
    version:                int   = 1

    # Behavior signature centroid (populated by clustering engine)
    centroid_signature:     dict  = field(default_factory=dict)
    semantic_label:         str   = ""    # "coffee_shop", "gym", "home", …

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
class H3CellRecord:
    """
    Aggregated behavior signature for a single H3 cell (resolution 10).

    This is NOT user-level data — it is computed across all users who have
    visited this cell, making it privacy-safe to store server-side.

    signature components:
        visit_time_histogram  — 24-bin hourly distribution of visits
        dwell_mean/std        — typical dwell time shape
        repeat_rate           — fraction of visits from repeat visitors (locals)
        user_entropy          — diversity of distinct visitors (Shannon entropy)
    """
    h3_index:               int
    visit_time_histogram:   list  = field(default_factory=lambda: [0] * 24)
    dwell_mean:             float = 0.0
    dwell_std:              float = 0.0
    repeat_rate:            float = 0.0   # 0 = all one-offs, 1 = all regulars
    user_entropy:           float = 0.0
    total_visits:           int   = 0
    cluster_id:             str   = ""    # assigned by ClusterRegistry
    cluster_version:        int   = 1
    last_updated:           float = field(default_factory=time.time)

    def signature_vector(self) -> list:
        """
        Flatten the signature into a feature vector for clustering.
        Format: [*time_histogram (24), dwell_mean, dwell_std,
                  repeat_rate, user_entropy]
        """
        return list(self.visit_time_histogram) + [
            self.dwell_mean,
            self.dwell_std,
            self.repeat_rate,
            self.user_entropy,
        ]

    def update(
        self,
        hour: int,
        dwell_minutes: float,
        is_repeat: bool,
        n_distinct_users: int,
    ) -> None:
        """Incrementally update the signature with a new observation."""
        self.total_visits += 1
        if 0 <= hour < 24:
            self.visit_time_histogram[hour] += 1
        # Running mean/std approximation (Welford's)
        n = float(self.total_visits)
        delta = dwell_minutes - self.dwell_mean
        self.dwell_mean += delta / n
        self.dwell_std = math.sqrt(
            max(0.0, self.dwell_std ** 2 + delta * (dwell_minutes - self.dwell_mean))
        )
        # Repeat rate: exponential moving average
        self.repeat_rate = 0.95 * self.repeat_rate + 0.05 * float(is_repeat)
        # User entropy: log(n_distinct) / log(total_visits + 1)
        if n_distinct_users > 1 and self.total_visits > 1:
            self.user_entropy = (
                math.log(n_distinct_users) / math.log(self.total_visits + 1)
            )
        self.last_updated = time.time()


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
    The behavioral ontology layer.

    Responsibilities:
      1. Maintain the catalog of behavioral archetypes (LocationCluster).
      2. Map H3 cell indices → cluster_ids (cell_mapping).
      3. Maintain per-cell behavior signatures (cell_signatures).
      4. Expose IDF snapshot and venue eligibility for matching.

    In production this is backed by a database.
    In-memory dicts here for tests and prototyping.
    """
    clusters:               dict[str, LocationCluster] = field(default_factory=dict)
    n_total_users:          int = 1000
    idf_snapshot:           dict[str, float] = field(default_factory=dict)
    venue_eligible_ids:     set[str] = field(default_factory=set)
    current_version:        int = 1

    # H3 → cluster_id mapping (Layer 1 → Layer 2 bridge)
    cell_mapping:           dict[int, str] = field(default_factory=dict)
    # H3 → H3CellRecord (aggregated, privacy-safe)
    cell_signatures:        dict[int, H3CellRecord] = field(default_factory=dict)

    def add_cluster(self, cluster: LocationCluster) -> None:
        self.clusters[cluster.cluster_id] = cluster

    def get(self, cluster_id: str) -> Optional[LocationCluster]:
        return self.clusters.get(cluster_id)

    def register_cell(self, h3_index: int, cluster_id: str) -> None:
        """Map an H3 cell to a behavioral cluster."""
        self.cell_mapping[h3_index] = cluster_id

    def get_cluster_for_cell(self, h3_index: int) -> Optional[LocationCluster]:
        """Resolve an H3 cell index to its behavioral cluster."""
        cluster_id = self.cell_mapping.get(h3_index)
        if cluster_id is None:
            return None
        return self.clusters.get(cluster_id)

    def get_or_create_cell_record(self, h3_index: int) -> H3CellRecord:
        """Return existing H3CellRecord or create a new one."""
        if h3_index not in self.cell_signatures:
            self.cell_signatures[h3_index] = H3CellRecord(h3_index=h3_index)
        return self.cell_signatures[h3_index]

    def refresh_idf(self) -> None:
        self.idf_snapshot = {
            cid: c.idf(self.n_total_users)
            for cid, c in self.clusters.items()
        }

    def idf_for(self, cluster_id: str) -> float:
        # Strip time_bin suffix if present ("cluster_id|WD_2" → "cluster_id")
        base_id = cluster_id.split("|", 1)[0]
        if base_id in self.idf_snapshot:
            return self.idf_snapshot[base_id]
        c = self.clusters.get(base_id)
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
