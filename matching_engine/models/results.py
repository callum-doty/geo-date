"""
models/results.py
=================
MatchResult, VenueResult, WeightSet, and Venue data structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeightSet:
    """Six-way adaptive weight output."""
    w_rhythm:       float
    w_identity:     float
    w_log:          float
    w_bio:          float
    w_edge:         float = 0.0   # transition similarity
    w_copresence:   float = 0.0   # co-presence signal
    n_r_eff:        int   = 0
    n_i_eff:        int   = 0

    def __post_init__(self) -> None:
        total = (self.w_rhythm + self.w_identity + self.w_log
                 + self.w_bio + self.w_edge + self.w_copresence)
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            # Renormalise silently to guard float drift
            self.w_rhythm       /= total
            self.w_identity     /= total
            self.w_log          /= total
            self.w_bio          /= total
            self.w_edge         /= total
            self.w_copresence   /= total

    def as_dict(self) -> dict:
        return {
            "w_rhythm":     round(self.w_rhythm,     4),
            "w_identity":   round(self.w_identity,   4),
            "w_log":        round(self.w_log,         4),
            "w_bio":        round(self.w_bio,         4),
            "w_edge":       round(self.w_edge,        4),
            "w_copresence": round(self.w_copresence,  4),
            "n_r_eff":      self.n_r_eff,
            "n_i_eff":      self.n_i_eff,
        }


@dataclass
class MatchResult:
    """Complete output of match_users()."""
    user_a_id:      str
    user_b_id:      str
    G:              float
    sim_rhythm:     float
    sim_identity:   float
    sim_prox:       float
    sim_prior:      float
    weights:        WeightSet
    phase:          str           # Discovery / Rhythm-Active / Dual-Stream / Identity-Rich
    sim_edge:       float = 0.0   # transition matrix similarity
    sim_copresence: float = 0.0   # co-presence score

    def as_dict(self) -> dict:
        return {
            "user_a": self.user_a_id,
            "user_b": self.user_b_id,
            "G": round(self.G, 4),
            "phase": self.phase,
            "components": {
                "rhythm":       round(self.sim_rhythm,      4),
                "identity":     round(self.sim_identity,    4),
                "proximity":    round(self.sim_prox,        4),
                "bio_prior":    round(self.sim_prior,       4),
                "edge":         round(self.sim_edge,        4),
                "copresence":   round(self.sim_copresence,  4),
            },
            "weights": self.weights.as_dict(),
        }

    def ui_headline(self) -> str:
        """Plain-language match headline for the match card."""
        if self.phase == "Identity-Rich":
            return "You share something rare"
        if self.phase == "Dual-Stream":
            return "Same rhythm, same soul"
        if self.phase == "Rhythm-Active":
            return "You both live like this"
        return "Based on your interests"


@dataclass
class Venue:
    """
    A candidate date venue in the recommendation pool.
    """
    venue_id:           str
    name:               str
    cluster_id:         str
    is_partner:         bool  = False
    travel_minutes_a:   float = 15.0
    travel_minutes_b:   float = 15.0
    extra_cluster_ids:  list[str] = field(default_factory=list)
    soft_label:         str   = ""

    @property
    def all_cluster_ids(self) -> list[str]:
        return [self.cluster_id] + self.extra_cluster_ids


@dataclass
class VenueResult:
    """Output of rank_venues() — one entry per candidate venue."""
    venue:              Venue
    score:              Optional[float]
    gated:              bool
    rhythm_intersection:float
    identity_intersection: float

    @property
    def recommended(self) -> bool:
        return not self.gated

    def as_dict(self) -> dict:
        return {
            "venue_id":              self.venue.venue_id,
            "name":                  self.venue.name,
            "score":                 round(self.score, 4) if self.score is not None else None,
            "gated":                 self.gated,
            "rhythm_intersection":   round(self.rhythm_intersection, 4),
            "identity_intersection": round(self.identity_intersection, 4),
            "is_partner":            self.venue.is_partner,
        }
