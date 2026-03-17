"""
tests/fixtures.py
=================
Shared test fixtures: registry, clusters, and synthetic users.

Both the smoke test and full suite import from here, keeping cluster
definitions in one place.
"""

from __future__ import annotations

from matching_engine.config.match_config import MatchConfig
from matching_engine.models.cluster import LocationCluster, ClusterRegistry


def build_test_registry(cfg: MatchConfig) -> ClusterRegistry:
    """Build the shared in-memory registry used across all tests."""
    registry = ClusterRegistry(n_total_users=500)

    # ── Rhythm clusters (permanent venues, well-visited) ─────────────────────
    gym = LocationCluster(
        "gym_crossroads", 39.0820, -94.5870,
        soft_label="Centric Fitness", places_category="gym",
        n_users=80, n_total_visits=400, activation_days=365,
        evening_visits=60, cooccurrence_visits=40, date_dwell_visits=90,
    )
    coffee = LocationCluster(
        "coffee_monarch", 39.0890, -94.5820,
        soft_label="Monarch Coffee", places_category="coffee_shop",
        n_users=200, n_total_visits=900, activation_days=365,
        evening_visits=200, cooccurrence_visits=350, date_dwell_visits=500,
    )
    park = LocationCluster(
        "park_loose", 39.0450, -94.5910,
        soft_label="Loose Park", places_category="park",
        n_users=150, n_total_visits=600, activation_days=365,
        evening_visits=180, cooccurrence_visits=200, date_dwell_visits=250,
    )
    bar = LocationCluster(
        "bar_recordbar", 39.0870, -94.5790,
        soft_label="RecordBar KC", places_category="music_venue",
        n_users=60, n_total_visits=300, activation_days=365,
        evening_visits=270, cooccurrence_visits=240, date_dwell_visits=200,
    )
    vinyl = LocationCluster(
        "shop_vinyl", 39.0910, -94.5800,
        soft_label="Mills Record Company", places_category="record_store",
        n_users=25, n_total_visits=80, activation_days=365,
        evening_visits=20, cooccurrence_visits=30, date_dwell_visits=50,
    )

    # ── Identity cluster (festival — rare, concentrated event) ────────────────
    festival = LocationCluster(
        "fest_boulevardia", 39.1020, -94.5850,
        soft_label="Boulevardia Festival", places_category="festival_grounds",
        n_users=12, n_total_visits=36, activation_days=3, is_event=True,
        evening_visits=30, cooccurrence_visits=36, date_dwell_visits=36,
    )

    # ── Ineligible cluster (private office — fails eligibility gate) ──────────
    office = LocationCluster(
        "office_private", 39.0800, -94.5850,
        soft_label="Private Office", places_category="office_building",
        n_users=1, n_total_visits=200, activation_days=365,
    )

    # ── Gas station (filtered by low suitability posterior) ───────────────────
    gas = LocationCluster(
        "gas_quiktrip", 39.0760, -94.5900,
        soft_label="QuikTrip", places_category="gas_station",
        n_users=300, n_total_visits=2000, activation_days=365,
        evening_visits=400, cooccurrence_visits=200, date_dwell_visits=50,
    )

    for c in [gym, coffee, park, bar, vinyl, festival, office, gas]:
        registry.add_cluster(c)

    registry.refresh_idf()
    registry.refresh_eligibility(cfg)
    return registry