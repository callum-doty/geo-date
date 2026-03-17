"""
similarity/match.py
===================
Adaptive weights and holistic match scoring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.user import UserProfile
    from matching_engine.models.cluster import ClusterRegistry
    from matching_engine.models.results import WeightSet, MatchResult

from matching_engine.models.results import WeightSet, MatchResult
from matching_engine.similarity.vectors import cosine_sparse, bio_similarity
from matching_engine.similarity.proximity import proximity_score


def _match_phase(n_r_eff: int, n_i_eff: int) -> str:
    """Determine behavioral phase for UI labelling."""
    if n_i_eff >= 10:
        return "Identity-Rich"
    if n_r_eff >= 10 and n_i_eff >= 3:
        return "Dual-Stream"
    if n_r_eff >= 10:
        return "Rhythm-Active"
    return "Discovery"


def adaptive_weights(
    user_a: "UserProfile",
    user_b: "UserProfile",
    cfg: "MatchConfig",
) -> WeightSet:
    """
    Compute the four-way adaptive weight set for a matched pair.

    Uses pair-effective ping counts:
        n_r_eff = min(n_rhythm_A, n_rhythm_B)
        n_i_eff = min(n_identity_A, n_identity_B)

    Weight schedule:
        w_bio(n_total)  = w_bio_max · exp(-μ_bio · n_total)
        w_identity(n_i) = w_identity_max · (1 - exp(-μ_id · n_i_eff))
        w_log           = fixed (cfg.w_log_fixed)
        w_rhythm        = 1 - w_log - w_bio - w_identity  (residual)

    Returns
    -------
    WeightSet
    """
    n_r_eff = min(user_a.n_rhythm,   user_b.n_rhythm)
    n_i_eff = min(user_a.n_identity, user_b.n_identity)
    n_total = n_r_eff + n_i_eff

    w_bio      = cfg.w_bio_max * math.exp(-cfg.mu_bio * n_total)
    w_identity = cfg.w_identity_max * (1.0 - math.exp(-cfg.mu_identity * n_i_eff))
    w_log      = cfg.w_log_fixed
    w_rhythm   = max(0.0, 1.0 - w_log - w_bio - w_identity)

    # Renormalise to guarantee sum = 1.0
    total = w_rhythm + w_identity + w_log + w_bio
    return WeightSet(
        w_rhythm=   round(w_rhythm   / total, 8),
        w_identity= round(w_identity / total, 8),
        w_log=      round(w_log      / total, 8),
        w_bio=      round(w_bio      / total, 8),
        n_r_eff=    n_r_eff,
        n_i_eff=    n_i_eff,
    )


def match_users(
    user_a: "UserProfile",
    user_b: "UserProfile",
    registry: "ClusterRegistry",
    cfg: "MatchConfig",
    sigma: Optional[float] = None,
) -> MatchResult:
    """
    Compute the holistic match score G(A, B).

    G = w_rhythm   · Sim_rhythm(A, B)
      + w_identity · Sim_identity(A, B)
      + w_log      · Sim_log(d)
      + w_bio      · Sim_prior(A, B)

    Parameters
    ----------
    user_a, user_b : UserProfile
    registry : ClusterRegistry
        Provides the IDF snapshot for normalisation.
    cfg : MatchConfig
    sigma : float | None
        City-specific proximity sigma. Defaults to cfg.default_sigma.

    Returns
    -------
    MatchResult
    """
    _sigma  = sigma if sigma is not None else cfg.default_sigma
    weights = adaptive_weights(user_a, user_b, cfg)
    idf     = registry.idf_snapshot

    sim_rhythm   = cosine_sparse(user_a.V_rhythm,   user_b.V_rhythm,   idf)
    sim_identity = cosine_sparse(user_a.V_identity, user_b.V_identity, idf)
    sim_prior    = bio_similarity(user_a.tags, user_b.tags)

    avg_commute = (user_a.home_base_commute + user_b.home_base_commute) / 2.0
    sim_prox    = proximity_score(avg_commute, sigma=_sigma)

    G = (
        weights.w_rhythm   * sim_rhythm
        + weights.w_identity * sim_identity
        + weights.w_log      * sim_prox
        + weights.w_bio      * sim_prior
    )

    return MatchResult(
        user_a_id=    user_a.user_id,
        user_b_id=    user_b.user_id,
        G=            float(min(max(G, 0.0), 1.0)),
        sim_rhythm=   sim_rhythm,
        sim_identity= sim_identity,
        sim_prox=     sim_prox,
        sim_prior=    sim_prior,
        weights=      weights,
        phase=        _match_phase(weights.n_r_eff, weights.n_i_eff),
    )
