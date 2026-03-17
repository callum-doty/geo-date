"""
venue/ranking.py
================
Pillar 5 — Fully Dynamic Venue Recommendation.

No static venue catalog. Clusters self-register as venue candidates.
A cluster graduates to venue-eligible status when it clears the
corpus-level gate (n_users ≥ min_users, n_total_visits ≥ min_visits)
and its Bayesian suitability posterior is above a meaningful floor.

Venue score formula:
    V(k) = α_r · rhythm_intersection(k)
          + α_i · identity_intersection(k)
          + 0.20 · suitability_posterior(k)
          - γ   · travel_penalty(k)

Hard gate: max(rhythm_intersection, identity_intersection) < θ → None
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.cluster import LocationCluster, ClusterRegistry
    from matching_engine.models.user import UserProfile
    from matching_engine.models.results import VenueResult, Venue

from matching_engine.similarity.vectors import idf_normalize_sparse
from matching_engine.venue.suitability import is_venue_eligible, suitability_posterior


def _haversine_minutes(
    lat1: float, lng1: float,
    lat2: float, lng2: float,
    avg_speed_kph: float = 30.0,
) -> float:
    """
    Estimate travel time in minutes using haversine distance.

    Assumes avg_speed_kph for urban driving (default 30 kph).
    In production, replace with Google Maps commute time API.
    """
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    km = R * 2 * math.asin(math.sqrt(a))
    return (km / avg_speed_kph) * 60.0


def _cluster_as_venue(cluster: "LocationCluster") -> "Venue":
    """
    Wrap a LocationCluster in a Venue for VenueResult compatibility.

    In a fully dynamic system, Venue is a thin UI view over a cluster.
    """
    from matching_engine.models.results import Venue
    return Venue(
        venue_id=         cluster.cluster_id,
        name=             cluster.soft_label,
        cluster_id=       cluster.cluster_id,
        is_partner=       False,
        travel_minutes_a= 0.0,   # computed dynamically below
        travel_minutes_b= 0.0,
        soft_label=       cluster.soft_label,
    )


def score_venue_dynamic(
    user_a: "UserProfile",
    user_b: "UserProfile",
    cluster: "LocationCluster",
    cfg: "MatchConfig",
    idf: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """
    Score a cluster as a venue for a matched pair.

    No static venue catalog. Clusters self-register as venue candidates.
    Score combines pair-level intersection (do they share signal here?)
    with cluster-level suitability posterior (is this a good date place?).

    Formula
    -------
    V(k) = α_r · rhythm_intersection(k)
          + α_i · identity_intersection(k)
          + 0.20 · suitability_posterior(k)
          - γ   · travel_penalty(k)

    Hard gate: max(rhythm_intersection, identity_intersection) < θ → None
    Eligibility gate: cluster must pass is_venue_eligible() → None if not

    Parameters
    ----------
    user_a, user_b : UserProfile
    cluster : LocationCluster
    cfg : MatchConfig
    idf : dict[str, float] | None
        Pass registry.idf_snapshot for efficiency.

    Returns
    -------
    float  if both gates cleared
    None   if hard-gated or ineligible
    """
    if not is_venue_eligible(cluster, cfg):
        return None

    _idf = idf or {}
    a_r = idf_normalize_sparse(user_a.V_rhythm,   _idf)
    b_r = idf_normalize_sparse(user_b.V_rhythm,   _idf)
    a_i = idf_normalize_sparse(user_a.V_identity, _idf)
    b_i = idf_normalize_sparse(user_b.V_identity, _idf)

    cid     = cluster.cluster_id
    r_inter = a_r.get(cid, 0.0) * b_r.get(cid, 0.0)
    i_inter = a_i.get(cid, 0.0) * b_i.get(cid, 0.0)

    # Hard gate — pair must share meaningful signal in at least one stream
    if max(r_inter, i_inter) < cfg.theta:
        return None

    suit = suitability_posterior(cluster, cfg)

    # Travel penalty: symmetric average, normalised to [0, 1]
    # In production, computed via Maps API from HomeBase coords.
    travel_a = _haversine_minutes(
        user_a.home_base_lat or cluster.centroid_lat,
        user_a.home_base_lng or cluster.centroid_lng,
        cluster.centroid_lat, cluster.centroid_lng,
    )
    travel_b = _haversine_minutes(
        user_b.home_base_lat or cluster.centroid_lat,
        user_b.home_base_lng or cluster.centroid_lng,
        cluster.centroid_lat, cluster.centroid_lng,
    )
    avg_travel  = (travel_a + travel_b) / 2.0
    travel_norm = min(avg_travel / cfg.p_max, 1.0)

    return float(
        cfg.alpha_rhythm   * r_inter
        + cfg.alpha_identity * i_inter
        + 0.20               * suit
        - cfg.gamma          * travel_norm
    )


def rank_venues_dynamic(
    user_a: "UserProfile",
    user_b: "UserProfile",
    registry: "ClusterRegistry",
    cfg: "MatchConfig",
    top_n: int = 10,
) -> list["VenueResult"]:
    """
    Rank all venue-eligible clusters for a matched pair.

    No venue list is passed in — the registry IS the venue catalog.
    Eligible clusters are those in registry.venue_eligible_ids.

    Hard-gated clusters (low intersection) are included after the
    top_n recommended results so callers can surface meaningful
    "no venues nearby" messaging rather than an empty list.

    Parameters
    ----------
    user_a, user_b : UserProfile
    registry : ClusterRegistry
    cfg : MatchConfig
    top_n : int
        Maximum recommended venues to return (excludes gated).

    Returns
    -------
    list[VenueResult]
        Recommended venues sorted desc by score, then gated venues.
    """
    from matching_engine.models.results import VenueResult

    idf = registry.idf_snapshot
    a_r = idf_normalize_sparse(user_a.V_rhythm,   idf)
    b_r = idf_normalize_sparse(user_b.V_rhythm,   idf)
    a_i = idf_normalize_sparse(user_a.V_identity, idf)
    b_i = idf_normalize_sparse(user_b.V_identity, idf)

    recommended: list[VenueResult] = []
    gated:       list[VenueResult] = []

    for cid in registry.venue_eligible_ids:
        cluster = registry.clusters.get(cid)
        if cluster is None:
            continue

        r_inter = a_r.get(cid, 0.0) * b_r.get(cid, 0.0)
        i_inter = a_i.get(cid, 0.0) * b_i.get(cid, 0.0)
        raw     = score_venue_dynamic(user_a, user_b, cluster, cfg, idf=idf)

        result = VenueResult(
            venue=                _cluster_as_venue(cluster),
            score=                raw,
            gated=                (raw is None),
            rhythm_intersection=  r_inter,
            identity_intersection=i_inter,
        )
        if raw is not None:
            recommended.append(result)
        else:
            gated.append(result)

    recommended.sort(key=lambda r: r.score or 0, reverse=True)
    gated.sort(
        key=lambda r: max(r.rhythm_intersection, r.identity_intersection),
        reverse=True,
    )

    return recommended[:top_n] + gated