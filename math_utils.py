"""
math_util.py  —  v4.0
=====================
Behavioral Matching Engine · Core Mathematics

Architecture (v4.0):
    - Fully dynamic venue model: clusters self-register as venue candidates
    - No static venue catalog. Venues ARE clusters that meet eligibility criteria.
    - Bayesian suitability: Places API category prior → behavioral posterior
    - Corpus-level eligibility gate: n_users ≥ 5 AND n_total_visits ≥ 10
    - Three behavioral suitability signals: temporal, co-occurrence, dwell shape
    - Cold-city bootstrap: Places API seeds prior; behavior corrects it over time

    Carries forward from v3.0:
    - Dynamic cluster space (no fixed 8-category vector)
    - Dual-stream model: V_rhythm (fast decay) + V_identity (slow / pinned)
    - Significance Multiplier S routes pings to the correct stream
    - Four-way adaptive weight schedule in holistic score G
    - Sparse vector representation throughout
    - Pending buffer transit filter + velocity check
    - Geometric median HomeBase (rhythm pings only)

Public API
----------
# Config
    MatchConfig               — all tunable parameters incl. category prior table

# Data structures
    LocationCluster           — emergent place with Bayesian suitability state
    ClusterRegistry           — dynamic cluster space with eligibility filtering
    ClusterObservation        — a single behavioral signal update to a cluster
    Ping                      — a single venue attendance event (pre-buffer)
    BufferedPing              — a ping that cleared the pending buffer
    UserProfile               — a user's full dual-stream state
    WeightSet                 — four-way adaptive weight set
    MatchResult               — output of match_users()
    VenueResult               — output of rank_venues()

# Bayesian suitability (Pillar 5a)
    category_prior(places_category, cfg)               → float [0,1]
    update_cluster_observations(cluster, obs)          → None  (mutates)
    suitability_behavioral(cluster)                    → float [0,1]
    suitability_posterior(cluster, cfg)                → float [0,1]
    alpha_confidence(n_observations, cfg)              → float [0,1]

# Cluster self-registration
    is_venue_eligible(cluster, cfg)                    → bool

# Pillar 1 — Cluster pipeline
    compute_significance(ping, cluster, cfg, n_total)  → float
    route_ping(ping, cluster, cfg, n_total)            → StreamAssignment
    apply_rhythm_ping(V, cluster_id, weight, dt, cfg)  → dict
    apply_identity_ping(V, cluster_id, weight, t, cfg) → dict
    decay_stream(V, delta_t, lambda_rate)              → dict
    enforce_pins(V_identity, pins)                     → dict
    geometric_median(coords)                           → (lat, lng)

# Pillar 2 — Similarity
    idf_normalize_sparse(V, idf_map)                   → dict
    cosine_sparse(V_a, V_b, idf_map)                  → float [0,1]
    bio_similarity(tags_a, tags_b)                     → float [0,1]

# Pillar 3 — Proximity
    proximity_score(commute_minutes, sigma)            → float (0,1]

# Pillar 4 — Match score
    adaptive_weights(user_a, user_b, cfg)              → WeightSet
    match_users(user_a, user_b, registry, cfg, sigma)  → MatchResult

# Pillar 5 — Venue recommendation (fully dynamic)
    score_venue_dynamic(user_a, user_b, cluster, cfg)  → float | None
    rank_venues_dynamic(user_a, user_b, registry, cfg) → list[VenueResult]

# Utilities
    PendingBuffer                                      — transit filter
    vector_has_drifted(V_old, V_new, idf_map)         → bool
    profile_strength(user, cfg)                        → dict
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchConfig:
    """
    All tunable system parameters.

    Significance Multiplier
    -----------------------
    S = venue_rarity × event_duration_weight × dwell_intensity
    s_threshold : pings with S >= s_threshold route to V_identity (default 0.40)

    Decay rates
    -----------
    lambda_rhythm   : per-cluster decay rate for rhythm stream (per day)
                      Range [0.05, 0.15]. Habit fades in 2-3 weeks of absence.
    lambda_identity : per-cluster decay rate for identity stream (per day)
                      Range [0.001, 0.005]. Half-life 200-700 days.

    Identity pinning
    ----------------
    pin_floor          : minimum weight floor during pin period (fraction of peak)
    pin_duration_days  : how long the floor is held (default 365 days)

    Adaptive weights
    ----------------
    w_log_fixed      : proximity weight — fixed, geography is stable (0.12)
    w_bio_max        : bio weight at zero pings for both streams (0.55)
    w_identity_max   : maximum identity weight when fully built (0.30)
    mu_bio           : bio decay rate (halves at n_total ≈ 28 pings)
    mu_identity      : identity weight growth rate per high-S ping

    Pending buffer
    --------------
    buffer_min_visits      : visits required before a location graduates
    buffer_min_dwell_mins  : total cumulative dwell required (minutes)
    buffer_window_days     : rolling window for buffer evaluation
    buffer_radius_m        : radius (metres) within which visits count as same location

    Venue scoring
    -------------
    alpha_rhythm   : weight on rhythm intersection in venue score
    alpha_identity : weight on identity intersection in venue score
    beta           : partner incentive boost
    gamma          : travel penalty weight
    theta          : hard-gate threshold (product of unit-norm components)
    p_max          : commute ceiling for normalisation (minutes)

    Proximity
    ---------
    default_sigma : city-specific RBF drop-off (commute minutes)
    """

    # Significance
    s_threshold:            float = 0.40

    # Decay
    lambda_rhythm:          float = 0.08    # per day (mid-range default)
    lambda_identity:        float = 0.002   # per day (~350-day half-life)

    # Pinning
    pin_floor:              float = 0.20    # fraction of peak weight
    pin_duration_days:      float = 365.0

    # Adaptive weights
    w_log_fixed:            float = 0.12
    w_bio_max:              float = 0.55
    w_identity_max:         float = 0.30
    mu_bio:                 float = 0.025
    mu_identity:            float = 0.15

    # Pending buffer
    buffer_min_visits:      int   = 2
    buffer_min_dwell_mins:  float = 20.0
    buffer_window_days:     float = 30.0
    buffer_radius_m:        float = 60.0

    # Venue scoring
    alpha_rhythm:           float = 0.45
    alpha_identity:         float = 0.35
    beta:                   float = 0.08
    gamma:                  float = 0.12
    theta:                  float = 0.25
    p_max:                  float = 60.0

    # Proximity
    default_sigma:          float = 25.0

    # ── Bayesian Suitability ──────────────────────────────────────────────────
    # alpha(n) = 1 - exp(-beta_suitability * n)
    # beta = 0.02 → behavioral reaches 50% authority at ~35 observations,
    #                                   90% authority at ~115 observations
    beta_suitability:       float = 0.02

    # Corpus-level eligibility gate — cluster must clear both thresholds
    # before it can appear as a date venue recommendation.
    # Prevents private offices, one-off locations, and solo-use spots.
    venue_min_users:        int   = 5
    venue_min_visits:       int   = 10

    # Behavioral suitability component weights
    # temporal + cooccurrence + dwell must sum to 1.0
    w_temporal:             float = 0.40   # evening-weighted pings
    w_cooccurrence:         float = 0.35   # multi-user co-visits
    w_dwell:                float = 0.25   # dwell shape (date-length visits)

    # Dwell shape thresholds (minutes) — visits in this range score highest
    dwell_date_min:         float = 45.0
    dwell_date_max:         float = 180.0

    # Co-occurrence: two users within this many minutes = social visit
    cooccurrence_window_mins: float = 120.0

    # Evening hours for temporal score (inclusive, 24h)
    evening_start_hour:     int   = 18   # 6 PM
    evening_end_hour:       int   = 24   # midnight

    # Default suitability prior for unknown/unmapped Places API categories
    default_category_prior: float = 0.50

    # Places API category → suitability prior [0, 1]
    # Based on how date-appropriate each category type is.
    # This is the ONLY place in the system where a human makes a judgment call.
    # It is explicitly labeled as a prior — behavioral data corrects it.
    category_prior_table: dict = field(default_factory=lambda: {
        # High suitability — strong date venues
        "music_venue":          0.90,
        "jazz_club":            0.92,
        "wine_bar":             0.88,
        "cocktail_bar":         0.85,
        "rooftop_bar":          0.87,
        "speakeasy":            0.89,
        "comedy_club":          0.83,
        "art_gallery":          0.80,
        "restaurant":           0.80,
        "fine_dining":          0.85,
        "bistro":               0.82,
        "cafe":                 0.78,
        "coffee_shop":          0.75,
        "tea_house":            0.73,
        "bookstore":            0.72,
        "record_store":         0.74,
        "bowling_alley":        0.76,
        "arcade":               0.74,
        "mini_golf":            0.72,
        "escape_room":          0.78,
        "board_game_cafe":      0.80,
        "movie_theater":        0.70,
        "live_music_venue":     0.88,
        "concert_hall":         0.85,
        "theater":              0.82,
        "comedy_venue":         0.80,
        "festival_grounds":     0.85,
        "park":                 0.70,
        "botanical_garden":     0.78,
        "waterfront":           0.75,
        "rooftop":              0.80,
        "hiking_trail":         0.68,
        "beach":                0.72,
        "museum":               0.75,
        "aquarium":             0.73,
        "planetarium":          0.77,
        "brewery":              0.80,
        "winery":               0.82,
        "distillery":           0.78,
        "taproom":              0.76,
        "pub":                  0.74,
        "bar":                  0.72,
        "nightclub":            0.55,   # high energy, harder for first dates
        "lounge":               0.78,
        "karaoke":              0.74,
        "dance_studio":         0.65,
        "cooking_class":        0.82,
        "pottery_studio":       0.78,
        "climbing_gym":         0.65,
        "yoga_studio":          0.55,
        "fitness_center":       0.35,
        "gym":                  0.30,

        # Low suitability — utilitarian / solo-use venues
        "grocery_store":        0.05,
        "supermarket":          0.05,
        "convenience_store":    0.08,
        "gas_station":          0.02,
        "pharmacy":             0.05,
        "bank":                 0.03,
        "atm":                  0.02,
        "laundromat":           0.04,
        "dry_cleaner":          0.04,
        "car_wash":             0.03,
        "parking_lot":          0.02,
        "office_building":      0.10,
        "hospital":             0.02,
        "clinic":               0.03,
        "dentist":              0.02,
        "airport":              0.05,
        "bus_station":          0.05,
        "train_station":        0.08,
        "warehouse":            0.02,
        "storage_facility":     0.01,
        "fast_food":            0.25,   # possible but not ideal
        "food_court":           0.20,
        "mall":                 0.30,
        "department_store":     0.15,
        "hardware_store":       0.08,
    })

    def validate(self) -> None:
        assert 0 < self.s_threshold < 1,      "s_threshold must be in (0,1)"
        assert self.lambda_rhythm > 0,         "lambda_rhythm must be positive"
        assert self.lambda_identity >= 0,      "lambda_identity must be non-negative"
        assert 0 < self.pin_floor < 1,         "pin_floor must be in (0,1)"
        assert self.pin_duration_days > 0,     "pin_duration_days must be positive"
        total_max = self.w_log_fixed + self.w_bio_max + self.w_identity_max
        assert total_max <= 1.05,              "w_log + w_bio_max + w_identity_max should be ≤ 1"
        assert 0 < self.theta < 1,             "theta must be in (0,1)"
        assert self.alpha_rhythm + self.alpha_identity + self.beta <= 1.05, \
            "alpha_r + alpha_i + beta should sum ≤ 1"
        assert self.beta_suitability > 0,      "beta_suitability must be positive"
        assert self.venue_min_users >= 1,      "venue_min_users must be >= 1"
        assert self.venue_min_visits >= 1,     "venue_min_visits must be >= 1"
        assert abs(self.w_temporal + self.w_cooccurrence + self.w_dwell - 1.0) < 1e-6, \
            "w_temporal + w_cooccurrence + w_dwell must sum to 1.0"
        assert 0 <= self.default_category_prior <= 1, \
            "default_category_prior must be in [0,1]"


DEFAULT_CFG = MatchConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAM ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class Stream(Enum):
    RHYTHM   = auto()   # V_rhythm  — daily life, fast decay
    IDENTITY = auto()   # V_identity — defining traits, slow/pinned decay
    BUFFER   = auto()   # pending — not yet graduated


# ═══════════════════════════════════════════════════════════════════════════════
# LOCATION CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocationCluster:
    """
    An emergent geographic place — the fundamental unit of the vector space.

    In v4.0 clusters are also self-registering venue candidates. No separate
    Venue catalog exists. A cluster graduates to venue-eligible status when
    it clears the corpus-level gate (n_users ≥ 5, n_total_visits ≥ 10) and
    its Bayesian suitability posterior is above a meaningful floor.

    Suitability State (Bayesian)
    ----------------------------
    suitability is computed as:
        posterior = (1 - α) · prior  +  α · behavioral
    where α = alpha_confidence(n_total_visits) grows with observations.

    Behavioral suitability is derived from three aggregate signals:
        temporal_score      : fraction of pings in evening hours (6pm–midnight)
        cooccurrence_score  : fraction of visits with ≥2 users in same window
        dwell_score         : fraction of visits in date-appropriate dwell range

    Attributes
    ----------
    cluster_id : str
    centroid_lat, centroid_lng : float
    soft_label : str
        Human-readable name from Places API. UI only — never used in math.
    places_category : str
        Normalized Places API category string. Keys into cfg.category_prior_table.
    n_users : int
        Distinct users who have visited this cluster.
    n_total_visits : int
        Total ping-level visits (all users combined).
    activation_days : float
        Distinct days with elevated ping density.
    is_event : bool
        True for time-limited events (festivals, conventions).
    created_at : float

    Behavioral suitability accumulators (updated per observation):
    evening_visits : int        — pings in evening hours
    cooccurrence_visits : int   — visits with ≥2 users in time window
    date_dwell_visits : int     — visits with dwell in [45, 180] min
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

    Generated when a graduated ping is processed. The registry collects these
    to incrementally update temporal, co-occurrence, and dwell accumulators
    on the corresponding LocationCluster.

    Attributes
    ----------
    cluster_id : str
    timestamp : float
        Unix timestamp of the visit — used for temporal_score.
    dwell_minutes : float
        Duration of visit — used for dwell_score.
    concurrent_user_ids : list[str]
        Other user_ids detected at the same cluster within the co-occurrence
        window. In production, populated by the session-proximity service.
        Empty list = solo visit.
    """
    cluster_id:             str
    timestamp:              float
    dwell_minutes:          float
    concurrent_user_ids:    list[str] = field(default_factory=list)


@dataclass
class ClusterRegistry:
    """
    The live dynamic cluster space.

    In production this is backed by a database.
    In-memory dict keyed by cluster_id here.

    v4.0 additions:
        - venue_eligible_ids: pre-filtered set of clusters that meet the
          corpus-level gate (n_users ≥ cfg.venue_min_users AND
          n_total_visits ≥ cfg.venue_min_visits).
          Recomputed when refresh_eligibility() is called.
        - record_observation(): incremental update to a cluster's
          behavioral suitability accumulators.
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

        Call this after any batch of cluster updates (e.g. nightly job).
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

        Called every time a graduated ping is processed. Updates:
            evening_visits       if ping falls in evening hours
            cooccurrence_visits  if concurrent_user_ids is non-empty
            date_dwell_visits    if dwell is in date-appropriate range
            n_total_visits       always incremented

        Does not recompute IDF or eligibility — those run on nightly jobs.
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


# ═══════════════════════════════════════════════════════════════════════════════
# PING & BUFFER DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ping:
    """
    A single raw GPS venue-attendance event, pre-buffer.

    Attributes
    ----------
    lat, lng : float
        GPS coordinates at time of ping.
    dwell_minutes : float
        Time spent at this location.
    timestamp : float
        Unix timestamp of ping.
    delta_t_days : float
        Days elapsed since the user's previous ping.
    resolved_cluster_id : str | None
        Set after Tier 1 / Tier 2 resolution. None if still in buffer.
    n_users_in_window : int
        How many users visited the same cluster in a ±7-day window.
        Used for venue_rarity in Significance computation.
    """
    lat:                    float
    lng:                    float
    dwell_minutes:          float
    timestamp:              float = field(default_factory=time.time)
    delta_t_days:           float = 0.0
    resolved_cluster_id:    Optional[str] = None
    n_users_in_window:      int = 1

    def dwell_hours(self) -> float:
        return self.dwell_minutes / 60.0


@dataclass
class BufferedPing:
    """
    A collection of raw pings at the same location, pending buffer graduation.

    Attributes
    ----------
    location_key : str
        Coarse location identifier (e.g. rounded lat/lng at ~60m resolution).
    pings : list[Ping]
        All raw pings at this location within the buffer window.
    first_seen : float
        Timestamp of first ping at this location.
    """
    location_key:   str
    pings:          list[Ping] = field(default_factory=list)
    first_seen:     float = field(default_factory=time.time)

    @property
    def visit_count(self) -> int:
        return len(self.pings)

    @property
    def total_dwell_minutes(self) -> float:
        return sum(p.dwell_minutes for p in self.pings)

    def has_graduated(self, cfg: MatchConfig = DEFAULT_CFG) -> bool:
        """
        True if this location has cleared the pending buffer.
        All three criteria must be satisfied.
        """
        window_ok = (time.time() - self.first_seen) / 86400 <= cfg.buffer_window_days
        return (
            window_ok
            and self.visit_count >= cfg.buffer_min_visits
            and self.total_dwell_minutes >= cfg.buffer_min_dwell_mins
        )


@dataclass
class StreamAssignment:
    """Output of route_ping() — where a ping goes and why."""
    cluster_id: str
    stream:     Stream
    S:          float       # Significance score
    weight:     float       # dwell-time weight to add to the vector


# ═══════════════════════════════════════════════════════════════════════════════
# PINNED WEIGHT RECORD  (for V_identity)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# USER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """
    A user's complete dual-stream behavioral state.

    Attributes
    ----------
    user_id : str
    V_rhythm : dict[str, float]
        Sparse rhythm vector. Keys are cluster_ids, values are raw weights.
        Fast-decaying. Represents daily lifestyle patterns.
    V_identity : dict[str, float]
        Sparse identity vector. Keys are cluster_ids, values are raw weights.
        Slow-decaying with pinning. Represents core identity traits.
    pins : dict[str, PinnedWeight]
        Active identity pins per cluster. Enforces pin_floor during pin period.
    n_rhythm : int
        Count of rhythm pings received.
    n_identity : int
        Count of identity pings received.
    tags : list[str]
        Stated preference tags from onboarding (bio similarity fallback).
    home_base_lat, home_base_lng : float | None
        Geometric median of V_rhythm ping coordinates.
        Excludes identity pings (festival travel ≠ home relocation).
    home_base_commute : float
        Estimated commute time (minutes) from HomeBase to city centre.
    rhythm_ping_coords : list[tuple[float, float]]
        Running list of (lat, lng) for HomeBase recomputation.
        Only rhythm-stream pings are stored here.
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
        assignment: StreamAssignment,
        ping: Ping,
        cluster: LocationCluster,
        cfg: MatchConfig = DEFAULT_CFG,
        registry: Optional["ClusterRegistry"] = None,
    ) -> None:
        """
        Apply a routed ping to the appropriate stream vector.
        Mutates the profile in-place.
        Pass registry so route_ping can access n_total_users correctly.
        """
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


# ═══════════════════════════════════════════════════════════════════════════════
# VENUE & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Venue:
    """
    A candidate date venue in the recommendation pool.

    Attributes
    ----------
    venue_id : str
    name : str
    cluster_id : str
        The cluster this venue maps to. This is how it enters the math.
    extra_cluster_ids : list[str]
        Secondary clusters for multi-category venues.
    is_partner : bool
        Triggers beta incentive boost — only if intersection clears theta.
    travel_minutes_a, travel_minutes_b : float
        Commute time from each user's HomeBase to this venue.
    soft_label : str
        Human-readable label for UI display.
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
class WeightSet:
    """Four-way adaptive weight output."""
    w_rhythm:   float
    w_identity: float
    w_log:      float
    w_bio:      float
    n_r_eff:    int
    n_i_eff:    int

    def __post_init__(self) -> None:
        total = self.w_rhythm + self.w_identity + self.w_log + self.w_bio
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            # Renormalise silently to guard float drift
            self.w_rhythm   /= total
            self.w_identity /= total
            self.w_log      /= total
            self.w_bio      /= total

    def as_dict(self) -> dict:
        return {
            "w_rhythm":   round(self.w_rhythm,   4),
            "w_identity": round(self.w_identity, 4),
            "w_log":      round(self.w_log,       4),
            "w_bio":      round(self.w_bio,       4),
            "n_r_eff":    self.n_r_eff,
            "n_i_eff":    self.n_i_eff,
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
    phase:          str     # Discovery / Rhythm-Active / Dual-Stream / Identity-Rich

    def as_dict(self) -> dict:
        return {
            "user_a": self.user_a_id,
            "user_b": self.user_b_id,
            "G": round(self.G, 4),
            "phase": self.phase,
            "components": {
                "rhythm":   round(self.sim_rhythm,   4),
                "identity": round(self.sim_identity, 4),
                "proximity":round(self.sim_prox,     4),
                "bio_prior":round(self.sim_prior,    4),
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


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 5a — BAYESIAN VENUE SUITABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _hour_from_timestamp(ts: float) -> int:
    """Extract local hour (0–23) from a Unix timestamp. Uses UTC as proxy."""
    import datetime
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).hour


def alpha_confidence(n_observations: int, cfg: MatchConfig = DEFAULT_CFG) -> float:
    """
    Confidence weight that governs how much behavioral data overrides the prior.

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


def category_prior(places_category: str, cfg: MatchConfig = DEFAULT_CFG) -> float:
    """
    Look up the suitability prior for a Places API category string.

    The category string is normalised to lowercase with underscores before
    lookup. If not found in the table, returns cfg.default_category_prior.

    This is the only human judgment in the venue model. It is explicitly
    a prior — behavioral data will correct it over time via alpha_confidence.

    Parameters
    ----------
    places_category : str
        Places API category (e.g. "Music Venue", "coffee_shop", "Gym").
    cfg : MatchConfig

    Returns
    -------
    float ∈ [0, 1]
    """
    key = places_category.lower().replace(" ", "_").replace("-", "_")
    return cfg.category_prior_table.get(key, cfg.default_category_prior)


def suitability_behavioral(
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
) -> float:
    """
    Compute the behavioral suitability score from aggregate ping signals.

    suitability_behavioral = w_t · temporal_score
                           + w_c · cooccurrence_score
                           + w_d · dwell_score

    All three components are derived from the cluster's behavioral accumulators.
    Returns 0.0 if the cluster has no visits recorded yet.

    Components
    ----------
    temporal_score :
        Fraction of visits that occurred in evening hours [cfg.evening_start,
        cfg.evening_end). Higher = more evening activity = more date-like.

    cooccurrence_score :
        Fraction of visits where ≥1 other user was detected at the same
        cluster within cfg.cooccurrence_window_mins. Higher = more social.

    dwell_score :
        Fraction of visits with dwell in [cfg.dwell_date_min,
        cfg.dwell_date_max] minutes. This range captures coffee dates
        (45–90 min) and dinner dates (90–180 min) while excluding
        quick errand stops and gym sessions.

    Parameters
    ----------
    cluster : LocationCluster
    cfg : MatchConfig

    Returns
    -------
    float ∈ [0, 1]
    """
    n = cluster.n_total_visits
    if n == 0:
        return 0.0

    temporal      = cluster.evening_visits      / n
    cooccurrence  = cluster.cooccurrence_visits / n
    dwell         = cluster.date_dwell_visits   / n

    return (
        cfg.w_temporal      * temporal
        + cfg.w_cooccurrence  * cooccurrence
        + cfg.w_dwell         * dwell
    )


def suitability_posterior(
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
) -> float:
    """
    Bayesian suitability posterior for a cluster.

    posterior(k) = (1 - α(n_k)) · prior(k)  +  α(n_k) · behavioral(k)

    At zero observations: posterior = prior (Places API category drives all)
    As observations grow: behavioral evidence increasingly overrides the prior
    At high n: posterior ≈ behavioral (system is grounded in reality)

    This is the score that governs whether a cluster appears as a
    date recommendation — independent of pair-level intersection.
    High posterior = this is a good place for a date.
    High intersection = this pair specifically shares signal here.
    Both are required for a venue to be recommended.

    Parameters
    ----------
    cluster : LocationCluster
    cfg : MatchConfig

    Returns
    -------
    float ∈ [0, 1]
    """
    α = alpha_confidence(cluster.n_total_visits, cfg)
    prior      = category_prior(cluster.places_category, cfg)
    behavioral = suitability_behavioral(cluster, cfg)
    return (1.0 - α) * prior + α * behavioral


def is_venue_eligible(
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
) -> bool:
    """
    Corpus-level gate: can this cluster appear as a date venue recommendation?

    Criteria (all must be satisfied):
        n_users        ≥ cfg.venue_min_users   (not a private/solo location)
        n_total_visits ≥ cfg.venue_min_visits  (sufficient behavioral evidence)

    Note: this gate is necessary but not sufficient. A cluster also needs
    sufficient intersection with a matched pair to be surfaced.
    The gate prevents:
        - Private offices (1 user, many visits)
        - One-time events that only 2 people attended
        - Transit nodes that everyone passes through briefly

    Parameters
    ----------
    cluster : LocationCluster
    cfg : MatchConfig

    Returns
    -------
    bool
    """
    return (
        cluster.n_users >= cfg.venue_min_users
        and cluster.n_total_visits >= cfg.venue_min_visits
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — SIGNIFICANCE MULTIPLIER & STREAM ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def _dwell_weight(dwell_minutes: float) -> float:
    """
    Log-scaled dwell weight.
    w = log10(1 + dwell_minutes / 30)
    30 min → 0.30  |  60 min → 0.48  |  300 min → 1.0
    """
    return math.log10(1.0 + max(dwell_minutes, 0) / 30.0)


def compute_significance(
    ping: Ping,
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
    n_total_users: int = 1000,
) -> float:
    """
    Compute the Significance Multiplier S for a ping event.

    S = venue_rarity × event_duration_weight × dwell_intensity

    All three components are emergent — no manual tier assignment.

    Parameters
    ----------
    ping : Ping
        The graduated ping (post-buffer, cluster resolved).
    cluster : LocationCluster
        The resolved cluster for this ping.
    cfg : MatchConfig
    n_total_users : int
        Total active users in the registry (denominator for venue_rarity).
        Must be passed from ClusterRegistry.n_total_users — NOT from
        cluster.n_users, which would collapse rarity to log(1) = 0
        when n_users_in_window == cluster.n_users.

    Returns
    -------
    float
        S ∈ [0, ∞).  Practically bounded ~[0, 8].
        S < cfg.s_threshold  →  rhythm stream
        S ≥ cfg.s_threshold  →  identity stream

    Examples (n_total=500):
        Walmart visit (n_window≈400, act_days=365): S ≈ 0.000  → RHYTHM
        Local coffee (n_window≈200, act_days=365):  S ≈ 0.001  → RHYTHM
        Niche bar    (n_window≈60,  act_days=365):  S ≈ 0.001  → RHYTHM
        Jazz festival(n_window≈18,  act_days=3):    S ≈ 2.9    → IDENTITY
        Underground rave (n_window≈5, act_days=1):  S ≈ 6.3    → IDENTITY
    """
    # Component 1 — Venue rarity
    # Compares event attendance against the TOTAL active user base, not the
    # cluster's all-time visitor count. This is the key fix: rarity must
    # reflect how unusual it is to attend *this event* among all active users.
    n_window     = max(ping.n_users_in_window, 1)
    venue_rarity = math.log(max(n_total_users, 2) / n_window)

    # Component 2 — Event duration weight
    # Shorter activation window = higher signal.
    # 1-day event: 1/log(2) ≈ 1.44  |  365-day venue: 1/log(366) ≈ 0.17
    event_duration_weight = cluster.event_duration_weight()

    # Component 3 — Dwell intensity
    # Log-scaled hours attended relative to event length.
    dwell_h = ping.dwell_hours()
    dwell_intensity = math.log10(1.0 + dwell_h / max(cluster.activation_days, 1.0))

    return venue_rarity * event_duration_weight * dwell_intensity


def route_ping(
    ping: Ping,
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
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
        Pass ClusterRegistry.n_total_users here. Critical for correct
        venue_rarity computation in compute_significance().

    Returns
    -------
    StreamAssignment
        Contains cluster_id, stream (RHYTHM or IDENTITY), S, and weight.
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


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — VECTOR UPDATE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def decay_stream(
    V: dict[str, float],
    delta_t: float,
    lambda_rate: float,
) -> dict[str, float]:
    """
    Apply exponential decay to all non-zero dimensions of a sparse vector.

    V[k](t) = V[k](t-1) · exp(-λ · Δt)

    Zero-valued dimensions are pruned (saves memory in large cluster spaces).

    Parameters
    ----------
    V : dict[str, float]
        Sparse vector to decay.
    delta_t : float
        Days elapsed.
    lambda_rate : float
        Decay rate (use cfg.lambda_rhythm or cfg.lambda_identity).

    Returns
    -------
    dict[str, float]
        Decayed sparse vector with near-zero entries pruned.
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
    cfg: MatchConfig = DEFAULT_CFG,
) -> dict[str, float]:
    """
    Decay V_rhythm then add a new rhythm ping.

    V_rhythm[k](t) = V_rhythm[k](t-1) · exp(-λ_rhythm · Δt) + weight

    Parameters
    ----------
    V_rhythm : dict[str, float]
        Current rhythm vector.
    cluster_id : str
        Dimension to increment.
    weight : float
        Dwell-time weight from _dwell_weight().
    delta_t : float
        Days since last ping for this user.
    cfg : MatchConfig

    Returns
    -------
    dict[str, float]
        Updated rhythm vector.
    """
    V = decay_stream(V_rhythm, delta_t, cfg.lambda_rhythm)
    V[cluster_id] = V.get(cluster_id, 0.0) + weight
    return V


def apply_identity_ping(
    V_identity: dict[str, float],
    cluster_id: str,
    weight: float,
    t_ping: float,
    cfg: MatchConfig = DEFAULT_CFG,
    cluster: Optional[LocationCluster] = None,
    pins: Optional[dict[str, PinnedWeight]] = None,
) -> dict[str, float]:
    """
    Decay V_identity then add a new identity ping, with pinning logic.

    For event clusters (activation_days ≤ 7), the peak weight is pinned
    at pin_floor for pin_duration_days to prevent seasonal washout.

    Parameters
    ----------
    V_identity : dict[str, float]
        Current identity vector.
    cluster_id : str
        Dimension to increment.
    weight : float
        Dwell-time weight from _dwell_weight().
    t_ping : float
        Unix timestamp of the ping (for pin record).
    cfg : MatchConfig
    cluster : LocationCluster | None
        Resolved cluster — needed for pinning decision.
    pins : dict[str, PinnedWeight] | None
        Mutable pin registry from UserProfile.

    Returns
    -------
    dict[str, float]
        Updated identity vector.
    """
    V = decay_stream(V_identity, 0, cfg.lambda_identity)  # decay handled externally
    new_val = V.get(cluster_id, 0.0) + weight
    V[cluster_id] = new_val

    # Apply identity pin for concentrated events
    if cluster is not None and pins is not None:
        is_event_cluster = cluster.is_event or cluster.activation_days <= 7
        if is_event_cluster:
            pins[cluster_id] = PinnedWeight(
                peak_weight=new_val,
                pinned_at=t_ping,
                pin_duration=cfg.pin_duration_days,
                pin_floor=cfg.pin_floor,
            )
    return V


def enforce_pins(
    V_identity: dict[str, float],
    pins: dict[str, PinnedWeight],
) -> dict[str, float]:
    """
    Enforce pin floors on V_identity after any decay step.

    For each active pin, ensure V_identity[k] ≥ pin.floor_value().
    Expired pins are removed from the pin registry.

    Parameters
    ----------
    V_identity : dict[str, float]
    pins : dict[str, PinnedWeight]
        Mutated in-place: expired pins removed.

    Returns
    -------
    dict[str, float]
        V_identity with pin floors applied.
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


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — HOMEBASE (GEOMETRIC MEDIAN)
# ═══════════════════════════════════════════════════════════════════════════════

def geometric_median(
    coords: list[tuple[float, float]],
    max_iter: int = 300,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute the geometric median of a set of (lat, lng) coordinates
    using Weiszfeld's algorithm.

    More robust than the arithmetic mean — a single airport ping does not
    relocate a user's HomeBase. Only V_rhythm ping coords are passed here;
    identity (festival) pings are excluded.

    Parameters
    ----------
    coords : list[tuple[float, float]]
        List of (lat, lng) pairs.
    max_iter : int
        Maximum iterations (default 300).
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple[float, float]
        (lat, lng) of the geometric median.
    """
    if not coords:
        return (0.0, 0.0)
    if len(coords) == 1:
        return coords[0]

    pts = np.array(coords, dtype=float)

    # Initialise at arithmetic mean
    median = pts.mean(axis=0)

    for _ in range(max_iter):
        dists = np.linalg.norm(pts - median, axis=1)
        # Avoid division by zero for points exactly at median
        nonzero = dists > 1e-10
        if not np.any(nonzero):
            break
        safe_dists = np.where(nonzero, dists, 1.0)          # avoid division by zero
        weights = np.where(nonzero, 1.0 / safe_dists, 0.0)
        new_median = (pts[nonzero] * weights[nonzero, np.newaxis]).sum(axis=0) \
                     / weights[nonzero].sum()
        if np.linalg.norm(new_median - median) < tol:
            median = new_median
            break
        median = new_median

    return (float(median[0]), float(median[1]))


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — SPARSE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def idf_normalize_sparse(
    V: dict[str, float],
    idf_map: dict[str, float],
) -> dict[str, float]:
    """
    Apply IDF weighting then unit-normalise a sparse vector.

    Step 1 — IDF:   V*[k] = V[k] · IDF(k)
    Step 2 — Norm:  V̂*[k] = V*[k] / ‖V*‖

    Result: all components bounded [0, 1], dot product = cosine similarity.

    Parameters
    ----------
    V : dict[str, float]
        Raw sparse vector (V_rhythm or V_identity).
    idf_map : dict[str, float]
        IDF values per cluster_id (from ClusterRegistry.idf_snapshot).

    Returns
    -------
    dict[str, float]
        IDF-weighted, unit-normalised sparse vector.
        Empty dict if V is empty or all-zero after IDF.
    """
    if not V:
        return {}
    V_star = {k: v * idf_map.get(k, 0.0) for k, v in V.items() if v > 0}
    norm = math.sqrt(sum(v * v for v in V_star.values()))
    if norm < 1e-10:
        return {}
    return {k: v / norm for k, v in V_star.items()}


def cosine_sparse(
    V_a: dict[str, float],
    V_b: dict[str, float],
    idf_map: dict[str, float],
) -> float:
    """
    Cosine similarity between two sparse vectors after IDF normalisation.

    Since both are unit-normalised, similarity = dot product over shared keys.
    Returns 0 if either vector is empty (cold start / no data for this stream).

    Parameters
    ----------
    V_a, V_b : dict[str, float]
        Raw sparse vectors (pre-normalisation).
    idf_map : dict[str, float]
        IDF snapshot from ClusterRegistry.

    Returns
    -------
    float ∈ [0, 1]
    """
    a_norm = idf_normalize_sparse(V_a, idf_map)
    b_norm = idf_normalize_sparse(V_b, idf_map)
    if not a_norm or not b_norm:
        return 0.0
    # Dot product over intersection of keys
    shared = set(a_norm.keys()) & set(b_norm.keys())
    dot = sum(a_norm[k] * b_norm[k] for k in shared)
    return float(min(max(dot, 0.0), 1.0))


def bio_similarity(tags_a: list[str], tags_b: list[str]) -> float:
    """
    Jaccard overlap of stated interest tag sets.

    Sim_prior(A, B) = |T_A ∩ T_B| / |T_A ∪ T_B|

    Fallback signal at low ping count. Authority decays via w_bio schedule.

    Returns
    -------
    float ∈ [0, 1]
    """
    s_a, s_b = set(tags_a), set(tags_b)
    union = s_a | s_b
    if not union:
        return 0.0
    return float(len(s_a & s_b) / len(union))


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — PROXIMITY
# ═══════════════════════════════════════════════════════════════════════════════

def proximity_score(commute_minutes: float, sigma: float = 25.0) -> float:
    """
    RBF kernel on commute time.

    Sim_log(d) = exp(-d² / 2σ²)

    d = average commute time (minutes) between the two users' HomeBases.
    Commute time, not Euclidean distance. 10 miles in KC ≠ 10 miles in NYC.

    City-specific sigma (commute minutes):
        Kansas City : σ = 25
        New York    : σ = 12
        Los Angeles : σ = 30

    Returns
    -------
    float ∈ (0, 1]
    """
    return float(math.exp(-(commute_minutes ** 2) / (2.0 * sigma ** 2)))


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — ADAPTIVE WEIGHTS & HOLISTIC SCORE
# ═══════════════════════════════════════════════════════════════════════════════

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
    user_a: UserProfile,
    user_b: UserProfile,
    cfg: MatchConfig = DEFAULT_CFG,
) -> WeightSet:
    """
    Compute the four-way adaptive weight set for a matched pair.

    Uses pair-effective ping counts:
        n_r_eff = min(n_rhythm_A, n_rhythm_B)
        n_i_eff = min(n_identity_A, n_identity_B)

    This prevents a veteran user from claiming false confidence against
    a newcomer who hasn't yet built a profile in either stream.

    Weight schedule:
        w_bio(n_total)  = w_bio_max · exp(-μ_bio · n_total)
        w_identity(n_i) = w_identity_max · (1 - exp(-μ_id · n_i_eff))
        w_log           = fixed (cfg.w_log_fixed)
        w_rhythm        = 1 - w_log - w_bio - w_identity  (residual)

    Parameters
    ----------
    user_a, user_b : UserProfile
    cfg : MatchConfig

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
    user_a: UserProfile,
    user_b: UserProfile,
    registry: ClusterRegistry,
    cfg: MatchConfig = DEFAULT_CFG,
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


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 — FULLY DYNAMIC VENUE RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

def score_venue_dynamic(
    user_a: UserProfile,
    user_b: UserProfile,
    cluster: LocationCluster,
    cfg: MatchConfig = DEFAULT_CFG,
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
         + γ_s · suitability_posterior(k)
         - δ   · travel_penalty(k)

    Hard gate: max(rhythm_intersection, identity_intersection) < θ → None
    Eligibility gate: cluster must pass is_venue_eligible() → None if not

    Parameters
    ----------
    user_a, user_b : UserProfile
    cluster : LocationCluster
        Evaluated directly — no Venue wrapper needed.
    cfg : MatchConfig
    idf : dict[str, float] | None
        IDF snapshot. Pass registry.idf_snapshot for efficiency.

    Returns
    -------
    float  if both gates cleared
    None   if hard-gated or ineligible
    """
    # Corpus eligibility gate
    if not is_venue_eligible(cluster, cfg):
        return None

    _idf = idf or {}
    a_r = idf_normalize_sparse(user_a.V_rhythm,   _idf)
    b_r = idf_normalize_sparse(user_b.V_rhythm,   _idf)
    a_i = idf_normalize_sparse(user_a.V_identity, _idf)
    b_i = idf_normalize_sparse(user_b.V_identity, _idf)

    cid = cluster.cluster_id
    r_inter = a_r.get(cid, 0.0) * b_r.get(cid, 0.0)
    i_inter = a_i.get(cid, 0.0) * b_i.get(cid, 0.0)

    # Hard gate — pair must share meaningful signal
    if max(r_inter, i_inter) < cfg.theta:
        return None

    suit = suitability_posterior(cluster, cfg)

    # Travel penalty: symmetric average, normalised to [0,1]
    # In production, travel_minutes computed via Maps API from HomeBase coords.
    # Here we use a rough haversine estimate.
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
    avg_travel   = (travel_a + travel_b) / 2.0
    travel_norm  = min(avg_travel / cfg.p_max, 1.0)

    return float(
        cfg.alpha_rhythm   * r_inter
        + cfg.alpha_identity * i_inter
        + 0.20              * suit          # suitability posterior
        - cfg.gamma         * travel_norm
    )


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
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    km = R * 2 * math.asin(math.sqrt(a))
    return (km / avg_speed_kph) * 60.0


def rank_venues_dynamic(
    user_a: UserProfile,
    user_b: UserProfile,
    registry: ClusterRegistry,
    cfg: MatchConfig = DEFAULT_CFG,
    top_n: int = 10,
) -> list[VenueResult]:
    """
    Rank all venue-eligible clusters for a matched pair.

    No venue list is passed in. The registry IS the venue catalog.
    Eligible clusters are those in registry.venue_eligible_ids.

    Scoring:
        Each eligible cluster is scored via score_venue_dynamic().
        Hard-gated clusters (low intersection) are included at the
        bottom of the results so callers can surface meaningful
        "no venues nearby" messaging rather than an empty list.

    Parameters
    ----------
    user_a, user_b : UserProfile
    registry : ClusterRegistry
    cfg : MatchConfig
    top_n : int
        Maximum recommended venues to return (excludes gated).
        Gated venues are always appended after the top_n.

    Returns
    -------
    list[VenueResult]
        Recommended venues sorted desc by score, then gated venues.
    """
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
        raw = score_venue_dynamic(user_a, user_b, cluster, cfg, idf=idf)

        result = VenueResult(
            venue=_cluster_as_venue(cluster),
            score=raw,
            gated=(raw is None),
            rhythm_intersection=r_inter,
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


def _cluster_as_venue(cluster: LocationCluster) -> "Venue":
    """
    Shim: wrap a LocationCluster in a Venue for VenueResult compatibility.
    In a fully dynamic system, Venue is just a thin UI view over a cluster.
    """
    return Venue(
        venue_id=           cluster.cluster_id,
        name=               cluster.soft_label,
        cluster_id=         cluster.cluster_id,
        is_partner=         False,
        travel_minutes_a=   0.0,   # computed dynamically in score_venue_dynamic
        travel_minutes_b=   0.0,
        soft_label=         cluster.soft_label,
    )
    """
    Score a venue for a matched pair using both streams.

    V(v) = α_r · (V̂*_rhythm_A[c] · V̂*_rhythm_B[c])
          + α_i · (V̂*_identity_A[c] · V̂*_identity_B[c])
          + β   · I_v
          - γ   · P_travel(v)

    Hard gate: max(rhythm_intersection, identity_intersection) must exceed
    cfg.theta before scoring proceeds. This prevents partner venues from
    appearing for incompatible pairs regardless of payment.

    For multi-cluster venues, uses the maximum intersection across all
    cluster IDs (primary + extra_cluster_ids).

    Parameters
    ----------
    user_a, user_b : UserProfile
    venue : Venue
    registry : ClusterRegistry
    cfg : MatchConfig

    Returns
    -------
    float   if intersection clears the hard gate
    None    if hard-gated (excluded)
    """
    idf = registry.idf_snapshot

    a_r = idf_normalize_sparse(user_a.V_rhythm,   idf)
    b_r = idf_normalize_sparse(user_b.V_rhythm,   idf)
    a_i = idf_normalize_sparse(user_a.V_identity, idf)
    b_i = idf_normalize_sparse(user_b.V_identity, idf)

    best_r_intersection = 0.0
    best_i_intersection = 0.0

    for cid in venue.all_cluster_ids:
        r_inter = a_r.get(cid, 0.0) * b_r.get(cid, 0.0)
        i_inter = a_i.get(cid, 0.0) * b_i.get(cid, 0.0)
        best_r_intersection = max(best_r_intersection, r_inter)
        best_i_intersection = max(best_i_intersection, i_inter)

    # Hard gate — either stream can clear it
    if max(best_r_intersection, best_i_intersection) < cfg.theta:
        return None

    incentive = 1.0 if venue.is_partner else 0.0
    avg_travel = (venue.travel_minutes_a + venue.travel_minutes_b) / 2.0
    p_travel = min(avg_travel / cfg.p_max, 1.0)

    return float(
        cfg.alpha_rhythm   * best_r_intersection
        + cfg.alpha_identity * best_i_intersection
        + cfg.beta           * incentive
        - cfg.gamma          * p_travel
    )





class PendingBuffer:
    """
    Per-user buffer that gates raw GPS pings before they enter any stream.

    The buffer is the transit filter — it distinguishes a red light or
    gas station pause from a genuine behavioral signal.

    Graduation criteria (all must be met):
        1. min_visits     : ≥ 2 visits to the same location key
        2. min_dwell_mins : ≥ 20 min cumulative dwell at that location
        3. window_days    : all qualifying visits within a 30-day window
        4. velocity_check : no two consecutive visits within 60 seconds

    Location identity uses a rounded coordinate key at ~60m resolution.
    In production this is replaced by Tier 1 venue footprint matching;
    the rounded key is the Tier 2 / DBSCAN fallback.
    """

    VELOCITY_MIN_GAP_SECS = 60   # consecutive pings closer than this are transit

    def __init__(self, cfg: MatchConfig = DEFAULT_CFG):
        self.cfg = cfg
        # {location_key: BufferedPing}
        self._buffer: dict[str, BufferedPing] = {}

    @staticmethod
    def _location_key(lat: float, lng: float, precision: int = 4) -> str:
        """
        Round coordinates to ~11m precision (4 decimal places).
        Groups pings at the same place into a single buffer entry.
        """
        return f"{round(lat, precision)},{round(lng, precision)}"

    def ingest(self, ping: Ping) -> Optional[BufferedPing]:
        """
        Add a raw ping to the buffer.

        Returns the BufferedPing record if this ping caused graduation,
        else None (still accumulating).

        Velocity check: if the previous ping at this location was within
        VELOCITY_MIN_GAP_SECS, this ping is treated as transit and dropped.
        """
        key = self._location_key(ping.lat, ping.lng)

        if key not in self._buffer:
            self._buffer[key] = BufferedPing(location_key=key,
                                             first_seen=ping.timestamp)

        bucket = self._buffer[key]

        # Velocity check — drop transit pauses
        if bucket.pings:
            last_ts = bucket.pings[-1].timestamp
            if (ping.timestamp - last_ts) < self.VELOCITY_MIN_GAP_SECS:
                return None  # transit noise, discard

        bucket.pings.append(ping)

        if bucket.has_graduated(self.cfg):
            del self._buffer[key]
            return bucket
        return None

    def expire_stale(self) -> int:
        """
        Remove buffer entries whose 30-day window has expired.
        Returns count of entries purged.
        """
        now = time.time()
        window_secs = self.cfg.buffer_window_days * 86400
        stale = [k for k, b in self._buffer.items()
                 if (now - b.first_seen) > window_secs]
        for k in stale:
            del self._buffer[k]
        return len(stale)

    def __len__(self) -> int:
        return len(self._buffer)

    def pending_locations(self) -> list[str]:
        return list(self._buffer.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# MATCH INVALIDATION — DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def vector_has_drifted(
    V_old: dict[str, float],
    V_new: dict[str, float],
    idf_map: dict[str, float],
    drift_threshold: float = 0.05,
) -> bool:
    """
    Determine whether a user's vector has changed enough to invalidate
    cached match scores.

    Compares the IDF-normalised Euclidean distance between old and new
    vectors. If ‖V̂*_old - V̂*_new‖ > drift_threshold, cached G scores
    for this user should be recomputed.

    Parameters
    ----------
    V_old, V_new : dict[str, float]
        Raw sparse vectors before and after the latest ping.
    idf_map : dict[str, float]
        Current IDF snapshot from ClusterRegistry.
    drift_threshold : float
        Default 0.05. Recommended range: [0.03, 0.10].
        Lower = more aggressive recomputation (fresher scores, more compute).
        Higher = more caching tolerance (staler scores, less compute).

    Returns
    -------
    bool
        True if cached scores should be invalidated and recomputed.
    """
    old_norm = idf_normalize_sparse(V_old, idf_map)
    new_norm = idf_normalize_sparse(V_new, idf_map)

    all_keys = set(old_norm) | set(new_norm)
    if not all_keys:
        return False

    sq_dist = sum(
        (new_norm.get(k, 0.0) - old_norm.get(k, 0.0)) ** 2
        for k in all_keys
    )
    return math.sqrt(sq_dist) > drift_threshold


def profile_strength(user: UserProfile, cfg: MatchConfig = DEFAULT_CFG) -> dict:
    """
    Compute the profile strength metrics shown in the app UI.

    Returns a dict suitable for the 'Profile Strength' progress card,
    which makes the data collection mechanic legible to the user.

    Phases match the behavioral phase labels in MatchResult:
        Discovery     (0–9 rhythm pings,  0 identity pings)
        Rhythm-Active (≥10 rhythm pings,  0–2 identity pings)
        Dual-Stream   (≥10 rhythm,        3–9 identity pings)
        Identity-Rich (≥10 rhythm,        ≥10 identity pings)

    Returns
    -------
    dict with keys:
        phase           : current behavioral phase string
        rhythm_pct      : rhythm stream completion 0–100
        identity_pct    : identity stream completion 0–100
        overall_pct     : weighted composite 0–100
        rhythm_pings    : raw count
        identity_pings  : raw count
        active_clusters : non-zero dimensions across both streams
        next_milestone  : plain-language description of what unlocks next
    """
    RHYTHM_TARGET   = 20   # pings to reach full rhythm authority
    IDENTITY_TARGET = 10   # pings to reach full identity authority

    r_pct = min(user.n_rhythm   / RHYTHM_TARGET,   1.0) * 100
    i_pct = min(user.n_identity / IDENTITY_TARGET, 1.0) * 100
    overall = (r_pct * 0.6 + i_pct * 0.4)

    phase = _match_phase(user.n_rhythm, user.n_identity)

    if phase == "Discovery":
        next_milestone = f"Check in {max(0, 10 - user.n_rhythm)} more times to unlock behavioral matching"
    elif phase == "Rhythm-Active":
        next_milestone = f"Attend a rare event or festival to activate your Identity stream"
    elif phase == "Dual-Stream":
        next_milestone = f"Build {max(0, 10 - user.n_identity)} more identity signals to unlock the deepest matches"
    else:
        next_milestone = "Your profile is fully built — matches now reflect both your rhythm and your soul"

    active = len([v for v in user.V_rhythm.values()   if v > 1e-4]) + \
             len([v for v in user.V_identity.values() if v > 1e-4])

    return {
        "phase":           phase,
        "rhythm_pct":      round(r_pct,   1),
        "identity_pct":    round(i_pct,   1),
        "overall_pct":     round(overall, 1),
        "rhythm_pings":    user.n_rhythm,
        "identity_pings":  user.n_identity,
        "active_clusters": active,
        "next_milestone":  next_milestone,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST + COMPREHENSIVE TEST SUITE  (python math_util.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_test_registry(cfg) -> "ClusterRegistry":
    """Shared registry used across smoke test and full suite."""
    registry = ClusterRegistry(n_total_users=500)

    # ── Rhythm clusters (permanent venues, well-visited) ──────────────────────
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

    # ── Gas station (should be filtered by low suitability posterior) ─────────
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


def _smoke_test(cfg, registry) -> None:
    """Quick end-to-end smoke test printed at startup."""
    print("─" * 70)
    print("math_util.py  v4.0 — smoke test")
    print("─" * 70)

    cfg.validate()
    print("✓ Config validated")
    print(f"✓ Registry: {len(registry.clusters)} clusters  "
          f"eligible: {len(registry.venue_eligible_ids)}")

    # Suitability posteriors
    coffee_c  = registry.get("coffee_monarch")
    bar_c     = registry.get("bar_recordbar")
    gas_c     = registry.get("gas_quiktrip")
    office_c  = registry.get("office_private")

    print("\nSuitability posteriors:")
    for c in [bar_c, coffee_c, gas_c, office_c]:
        α   = alpha_confidence(c.n_total_visits, cfg)
        pri = category_prior(c.places_category, cfg)
        beh = suitability_behavioral(c, cfg)
        pos = suitability_posterior(c, cfg)
        elig = "✓ eligible" if is_venue_eligible(c, cfg) else "✗ ineligible"
        print(f"  {c.soft_label:<26} prior={pri:.2f} beh={beh:.2f} "
              f"α={α:.2f} posterior={pos:.2f}  {elig}")

    # Build users
    jordan = UserProfile("jordan", tags=["jazz","fitness","vinyl","outdoors"],
                         home_base_commute=12.0,
                         home_base_lat=39.082, home_base_lng=-94.587)
    riley  = UserProfile("riley",  tags=["jazz","fitness","coffee","hiking"],
                         home_base_commute=18.0,
                         home_base_lat=39.089, home_base_lng=-94.582)

    rhythm_schedule = [
        ("gym_crossroads", 55, 1.0), ("coffee_monarch", 35, 0.5),
        ("gym_crossroads", 60, 1.0), ("park_loose",     45, 0.5),
        ("gym_crossroads", 50, 1.0), ("coffee_monarch", 30, 0.5),
        ("bar_recordbar",  90, 1.0),
    ]
    for cid, dwell, dt in rhythm_schedule:
        cluster = registry.get(cid)
        p = Ping(lat=cluster.centroid_lat, lng=cluster.centroid_lng,
                 dwell_minutes=dwell, delta_t_days=dt,
                 resolved_cluster_id=cid, n_users_in_window=cluster.n_users)
        for user in [jordan, riley]:
            a = route_ping(p, cluster, cfg, registry.n_total_users)
            user.add_ping(a, p, cluster, cfg)

    # Festival identity pings
    fest = registry.get("fest_boulevardia")
    for user in [jordan, riley]:
        p = Ping(lat=fest.centroid_lat, lng=fest.centroid_lng,
                 dwell_minutes=2700, resolved_cluster_id=fest.cluster_id,
                 n_users_in_window=12)
        a = route_ping(p, fest, cfg, registry.n_total_users)
        user.add_ping(a, p, fest, cfg)
        user.V_identity = enforce_pins(user.V_identity, user.pins)

    print(f"\n✓ Jordan — rhythm:{jordan.n_rhythm} identity:{jordan.n_identity}")

    result = match_users(jordan, riley, registry, cfg)
    print(f"\n✓ Match [{result.phase}]  G={result.G:.4f}")
    print(f"  rhythm={result.sim_rhythm:.4f}(w={result.weights.w_rhythm:.3f})  "
          f"identity={result.sim_identity:.4f}(w={result.weights.w_identity:.3f})  "
          f"prox={result.sim_prox:.4f}(w={result.weights.w_log:.3f})  "
          f"bio={result.sim_prior:.4f}(w={result.weights.w_bio:.3f})")
    print(f"  \"{result.ui_headline()}\"")

    ranked = rank_venues_dynamic(jordan, riley, registry, cfg)
    print(f"\n✓ Venues (fully dynamic — no static list):")
    for vr in ranked[:8]:
        s = f"{vr.score:.4f}" if not vr.gated else "GATED  "
        print(f"  {s}  r={vr.rhythm_intersection:.3f} "
              f"i={vr.identity_intersection:.3f}  {vr.venue.name}")

    ps = profile_strength(jordan, cfg)
    print(f"\n✓ Profile strength: {ps['overall_pct']}% — {ps['phase']}")
    print(f"  {ps['next_milestone']}")
    return jordan, riley


def _full_test_suite(cfg, registry) -> None:
    passed = failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ FAIL: {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    # ── [1] Category prior table ──────────────────────────────────────────────
    print("\n[1] Category Prior Table")
    check("music_venue > gym",
          category_prior("music_venue", cfg) > category_prior("gym", cfg))
    check("gas_station near 0",
          category_prior("gas_station", cfg) < 0.10)
    check("coffee_shop in (0.7, 0.9)",
          0.7 < category_prior("coffee_shop", cfg) < 0.9)
    check("Unknown category → default",
          category_prior("unmapped_category_xyz", cfg) == cfg.default_category_prior)
    check("Case-insensitive lookup",
          category_prior("Music Venue", cfg) == category_prior("music_venue", cfg))
    check("Hyphen-normalised",
          category_prior("fine-dining", cfg) == category_prior("fine_dining", cfg))

    # ── [2] Alpha confidence schedule ─────────────────────────────────────────
    print("\n[2] Alpha Confidence Schedule")
    check("α(0) = 0.0",              alpha_confidence(0,   cfg) == 0.0)
    check("α grows monotonically",   alpha_confidence(50, cfg) > alpha_confidence(10, cfg))
    check("α(10) < α(1000)",          alpha_confidence(10, cfg) < alpha_confidence(1000, cfg))
    check("α(35) ≈ 0.50",            abs(alpha_confidence(35, cfg) - 0.50) < 0.05)
    check("α(115) ≈ 0.90",           abs(alpha_confidence(115, cfg) - 0.90) < 0.05)

    # ── [3] Behavioral suitability ────────────────────────────────────────────
    print("\n[3] Behavioral Suitability")
    bar_c    = registry.get("bar_recordbar")
    gym_c    = registry.get("gym_crossroads")
    gas_c    = registry.get("gas_quiktrip")
    coffee_c = registry.get("coffee_monarch")

    beh_bar  = suitability_behavioral(bar_c,    cfg)
    beh_gym  = suitability_behavioral(gym_c,    cfg)
    beh_gas  = suitability_behavioral(gas_c,    cfg)
    beh_cof  = suitability_behavioral(coffee_c, cfg)

    check("Bar behavioral > gym behavioral",   beh_bar > beh_gym,
          f"bar={beh_bar:.3f} gym={beh_gym:.3f}")
    check("Coffee behavioral > gas behavioral", beh_cof > beh_gas,
          f"coffee={beh_cof:.3f} gas={beh_gas:.3f}")
    check("Behavioral in [0,1]",               all(0 <= v <= 1 for v in [beh_bar, beh_gym, beh_gas]))
    empty_c = LocationCluster("empty_test", 0, 0, n_total_visits=0)
    check("Zero visits → behavioral = 0",      suitability_behavioral(empty_c, cfg) == 0.0)

    # ── [4] Bayesian posterior ────────────────────────────────────────────────
    print("\n[4] Bayesian Posterior")
    fest_c   = registry.get("fest_boulevardia")
    office_c = registry.get("office_private")

    # Cold cluster: α≈0, posterior ≈ prior
    cold = LocationCluster("cold_test", 0, 0,
                           places_category="music_venue", n_total_visits=0)
    post_cold = suitability_posterior(cold, cfg)
    prior_mv  = category_prior("music_venue", cfg)
    check("Cold cluster posterior ≈ prior",
          abs(post_cold - prior_mv) < 0.01,
          f"posterior={post_cold:.3f} prior={prior_mv:.3f}")

    # Warm cluster: behavioral has authority
    post_bar = suitability_posterior(bar_c,  cfg)
    post_gas = suitability_posterior(gas_c,  cfg)
    check("Bar posterior > gas posterior",  post_bar > post_gas,
          f"bar={post_bar:.3f} gas={post_gas:.3f}")
    check("Posterior in [0,1]",
          all(0 <= suitability_posterior(c, cfg) <= 1
              for c in [bar_c, gym_c, gas_c, coffee_c, fest_c]))

    # ── [5] Eligibility gate ──────────────────────────────────────────────────
    print("\n[5] Eligibility Gate")
    check("bar eligible",      is_venue_eligible(bar_c,    cfg))
    check("coffee eligible",   is_venue_eligible(coffee_c, cfg))
    check("office ineligible", not is_venue_eligible(office_c, cfg))

    too_few_users  = LocationCluster("tfu", 0, 0, n_users=3, n_total_visits=50)
    too_few_visits = LocationCluster("tfv", 0, 0, n_users=10, n_total_visits=5)
    check("Too few users → ineligible",  not is_venue_eligible(too_few_users,  cfg))
    check("Too few visits → ineligible", not is_venue_eligible(too_few_visits, cfg))

    check("Registry eligible set populated",  len(registry.venue_eligible_ids) > 0)
    check("Office not in eligible set",
          "office_private" not in registry.venue_eligible_ids)

    # ── [6] Cluster observation recording ────────────────────────────────────
    print("\n[6] Cluster Observation Recording")
    import datetime
    # Evening timestamp: 9 PM UTC
    evening_ts = datetime.datetime(2025, 8, 10, 21, 0, 0).timestamp()
    morning_ts = datetime.datetime(2025, 8, 10,  9, 0, 0).timestamp()

    test_cluster = LocationCluster(
        "obs_test", 39.085, -94.582, places_category="bar",
        n_users=10, n_total_visits=20, activation_days=365,
    )
    registry.add_cluster(test_cluster)

    pre_evening = test_cluster.evening_visits
    pre_visits  = test_cluster.n_total_visits
    pre_cooc    = test_cluster.cooccurrence_visits

    obs_evening = ClusterObservation(
        cluster_id="obs_test",
        timestamp=evening_ts,
        dwell_minutes=90,
        concurrent_user_ids=["user_x"],
    )
    registry.record_observation(obs_evening, cfg)

    check("n_total_visits incremented",   test_cluster.n_total_visits == pre_visits + 1)
    check("evening_visits incremented",   test_cluster.evening_visits == pre_evening + 1)
    check("cooccurrence incremented",     test_cluster.cooccurrence_visits == pre_cooc + 1)
    check("date_dwell_visits incremented",test_cluster.date_dwell_visits == 1)

    # Morning observation — no evening increment
    pre_ev2 = test_cluster.evening_visits
    obs_morning = ClusterObservation("obs_test", morning_ts, 20.0)
    registry.record_observation(obs_morning, cfg)
    check("Morning ping doesn't increment evening_visits",
          test_cluster.evening_visits == pre_ev2)

    # ── [7] Dynamic venue scoring ─────────────────────────────────────────────
    print("\n[7] Dynamic Venue Scoring")
    # Build minimal users for scoring
    u_a = UserProfile("score_a",
                      home_base_lat=39.082, home_base_lng=-94.587,
                      home_base_commute=10.0)
    u_b = UserProfile("score_b",
                      home_base_lat=39.089, home_base_lng=-94.582,
                      home_base_commute=15.0)

    idf = registry.idf_snapshot
    # Give both users strong signal at bar_recordbar (rhythm)
    u_a.V_rhythm["bar_recordbar"] = 2.0
    u_b.V_rhythm["bar_recordbar"] = 1.8
    # Give both users festival identity signal
    u_a.V_identity["fest_boulevardia"] = 1.5
    u_b.V_identity["fest_boulevardia"] = 1.4

    score_bar  = score_venue_dynamic(u_a, u_b, bar_c,  cfg, idf=idf)
    score_gas  = score_venue_dynamic(u_a, u_b, gas_c,  cfg, idf=idf)
    score_fest = score_venue_dynamic(u_a, u_b, fest_c, cfg, idf=idf)
    score_off  = score_venue_dynamic(u_a, u_b, office_c, cfg, idf=idf)

    check("Bar scores (shared rhythm signal)",     score_bar is not None,
          f"score={score_bar}")
    check("Festival scores (shared identity)",     score_fest is not None,
          f"score={score_fest}")
    check("Gas station gated (low intersection)",  score_gas is None)
    check("Office ineligible → None",              score_off is None)
    if score_bar and score_fest:
        check("Bar score in reasonable range",
              -0.5 < score_bar < 1.5, f"score={score_bar:.4f}")

    # ── [8] Dynamic ranking ───────────────────────────────────────────────────
    print("\n[8] Dynamic Venue Ranking")
    ranked = rank_venues_dynamic(u_a, u_b, registry, cfg)
    rec    = [r for r in ranked if not r.gated]
    gated  = [r for r in ranked if r.gated]

    check("Ranking returns results",      len(ranked) > 0)
    check("Recommended venues exist",     len(rec) > 0)
    check("Recommended sorted desc",
          all(rec[i].score >= rec[i+1].score for i in range(len(rec)-1)))
    check("Gated venues after recommended",
          all(r.gated for r in ranked[len(rec):]))
    check("Office not in results (ineligible)",
          all(r.venue.venue_id != "office_private" for r in ranked))

    # ── [9] Haversine travel estimate ─────────────────────────────────────────
    print("\n[9] Haversine Travel Estimate")
    # Same point → 0 minutes
    same = _haversine_minutes(39.082, -94.587, 39.082, -94.587)
    check("Same point → 0 min",    same == 0.0)
    # ~1km apart → ~2 min at 30kph
    near = _haversine_minutes(39.082, -94.587, 39.091, -94.587)
    check("~1km → roughly 2 min",  1.0 < near < 4.0, f"got {near:.2f}")
    # Monotone: closer = less travel
    far = _haversine_minutes(39.082, -94.587, 40.000, -95.000)
    check("Further = more time",   far > near)

    # ── [10] Full pipeline: similar vs dissimilar pair ────────────────────────
    print("\n[10] Full Pipeline Discrimination")

    def build_user(uid, rhythm_cids, identity_cids, lat, lng, tags):
        u = UserProfile(uid, tags=tags,
                        home_base_lat=lat, home_base_lng=lng,
                        home_base_commute=15.0)
        for cid, w in rhythm_cids:
            u.V_rhythm[cid] = w
        for cid, w in identity_cids:
            u.V_identity[cid] = w
        u.n_rhythm   = sum(1 for _ in rhythm_cids)
        u.n_identity = sum(1 for _ in identity_cids)
        return u

    # Similar pair: same rhythm clusters, same festival
    ua = build_user("ua",
        rhythm_cids=[("bar_recordbar",1.8), ("park_loose",0.9)],
        identity_cids=[("fest_boulevardia",1.4)],
        lat=39.082, lng=-94.587, tags=["jazz","outdoors"])
    ub = build_user("ub",
        rhythm_cids=[("bar_recordbar",1.6), ("park_loose",0.8)],
        identity_cids=[("fest_boulevardia",1.3)],
        lat=39.089, lng=-94.582, tags=["jazz","hiking"])

    # Dissimilar pair: different rhythm, no shared identity
    uc = build_user("uc",
        rhythm_cids=[("coffee_monarch",2.0), ("gym_crossroads",1.5)],
        identity_cids=[],
        lat=39.070, lng=-94.600, tags=["coffee","fitness"])

    registry.refresh_idf()
    r_ab = match_users(ua, ub, registry, cfg)
    r_ac = match_users(ua, uc, registry, cfg)

    check("Similar pair G > dissimilar G",
          r_ab.G > r_ac.G,
          f"G_AB={r_ab.G:.4f}  G_AC={r_ac.G:.4f}")
    check("Shared festival boosts identity sim",
          r_ab.sim_identity > 0,
          f"sim_identity={r_ab.sim_identity:.4f}")
    check("No shared festival → identity=0",
          r_ac.sim_identity == 0.0)
    check("G always in [0,1]",
          all(0 <= r.G <= 1 for r in [r_ab, r_ac]))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {passed} passed  |  {failed} failed  |  {passed+failed} total")
    if failed == 0:
        print("  ✓ All tests passed")
    else:
        print(f"  ✗ {failed} test(s) FAILED")
    print(f"{'─'*70}")


if __name__ == "__main__":
    cfg = MatchConfig()
    registry = _build_test_registry(cfg)
    jordan, riley = _smoke_test(cfg, registry)
    print(f"\n{'═'*70}")
    print("FULL TEST SUITE")
    print(f"{'═'*70}")
    _full_test_suite(cfg, registry)