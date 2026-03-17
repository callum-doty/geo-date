"""
venue/suitability.py
====================
Pillar 5a — Bayesian Venue Suitability.

Computes how date-appropriate a cluster is, blending a Places API category
prior with behavioral evidence that accumulates over time.

posterior(k) = (1 - α(n_k)) · prior(k)  +  α(n_k) · behavioral(k)

At zero observations: posterior = prior (Places API category drives all)
As observations grow:  behavioral evidence increasingly overrides the prior
At high n:             posterior ≈ behavioral (grounded in reality)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.cluster import LocationCluster


def alpha_confidence(n_observations: int, cfg: "MatchConfig") -> float:
    """
    Confidence weight governing how much behavioral data overrides the prior.

    α(n) = 1 - exp(-β · n)

    At n=0:   α=0.0  — prior has full authority (cold city / new cluster)
    At n=35:  α≈0.50 — equal weight between prior and behavioral
    At n=115: α≈0.90 — behavioral nearly dominant
    At n=230: α≈0.99 — prior almost irrelevant

    Parameters
    ----------
    n_observations : int
        n_total_visits for the cluster.
    cfg : MatchConfig

    Returns
    -------
    float ∈ [0, 1)
    """
    return 1.0 - math.exp(-cfg.beta_suitability * max(n_observations, 0))


def category_prior(places_category: str, cfg: "MatchConfig") -> float:
    """
    Look up the suitability prior for a Places API category string.

    The category string is normalised to lowercase with underscores before
    lookup. If not found, returns cfg.default_category_prior.

    This is the only human judgment call in the venue model — it is
    explicitly a prior that behavioral data will correct over time.

    Parameters
    ----------
    places_category : str
        e.g. "Music Venue", "coffee_shop", "Gym".
    cfg : MatchConfig

    Returns
    -------
    float ∈ [0, 1]
    """
    key = places_category.lower().replace(" ", "_").replace("-", "_")
    return cfg.category_prior_table.get(key, cfg.default_category_prior)


def suitability_behavioral(
    cluster: "LocationCluster",
    cfg: "MatchConfig",
) -> float:
    """
    Compute the behavioral suitability score from aggregate ping signals.

    suitability_behavioral = w_t · temporal_score
                           + w_c · cooccurrence_score
                           + w_d · dwell_score

    Components
    ----------
    temporal_score :
        Fraction of visits in evening hours [cfg.evening_start, cfg.evening_end).
    cooccurrence_score :
        Fraction of visits where ≥1 other user was detected within
        cfg.cooccurrence_window_mins. Higher = more social.
    dwell_score :
        Fraction of visits with dwell in [cfg.dwell_date_min, cfg.dwell_date_max]
        minutes — captures coffee dates (45–90 min) and dinner dates (90–180 min).

    Returns 0.0 if the cluster has no visits recorded yet.

    Returns
    -------
    float ∈ [0, 1]
    """
    n = cluster.n_total_visits
    if n == 0:
        return 0.0

    temporal     = cluster.evening_visits      / n
    cooccurrence = cluster.cooccurrence_visits / n
    dwell        = cluster.date_dwell_visits   / n

    return (
        cfg.w_temporal     * temporal
        + cfg.w_cooccurrence * cooccurrence
        + cfg.w_dwell        * dwell
    )


def suitability_posterior(
    cluster: "LocationCluster",
    cfg: "MatchConfig",
) -> float:
    """
    Bayesian suitability posterior for a cluster.

    posterior(k) = (1 - α(n_k)) · prior(k)  +  α(n_k) · behavioral(k)

    High posterior = this is a good place for a date (cluster-level).
    High intersection = this pair specifically shares signal here (pair-level).
    Both are required for a venue to be recommended.

    Returns
    -------
    float ∈ [0, 1]
    """
    α          = alpha_confidence(cluster.n_total_visits, cfg)
    prior      = category_prior(cluster.places_category, cfg)
    behavioral = suitability_behavioral(cluster, cfg)
    return (1.0 - α) * prior + α * behavioral


def is_venue_eligible(
    cluster: "LocationCluster",
    cfg: "MatchConfig",
) -> bool:
    """
    Corpus-level gate: can this cluster appear as a date venue recommendation?

    Criteria (all must be satisfied):
        n_users        ≥ cfg.venue_min_users   (not a private/solo location)
        n_total_visits ≥ cfg.venue_min_visits  (sufficient behavioral evidence)

    This gate is necessary but not sufficient. A cluster also needs
    sufficient intersection with a matched pair to be surfaced.

    Prevents:
        - Private offices (1 user, many visits)
        - One-time events that only 2 people attended
        - Transit nodes that everyone passes through briefly

    Returns
    -------
    bool
    """
    return (
        cluster.n_users        >= cfg.venue_min_users
        and cluster.n_total_visits >= cfg.venue_min_visits
    )