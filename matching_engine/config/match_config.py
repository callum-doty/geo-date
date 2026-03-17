"""
config/match_config.py
======================
All tunable system parameters for the behavioral matching engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    beta_suitability:       float = 0.02

    # Corpus-level eligibility gate
    venue_min_users:        int   = 5
    venue_min_visits:       int   = 10

    # Behavioral suitability component weights
    w_temporal:             float = 0.40   # evening-weighted pings
    w_cooccurrence:         float = 0.35   # multi-user co-visits
    w_dwell:                float = 0.25   # dwell shape (date-length visits)

    # Dwell shape thresholds (minutes)
    dwell_date_min:         float = 45.0
    dwell_date_max:         float = 180.0

    # Co-occurrence window
    cooccurrence_window_mins: float = 120.0

    # Evening hours for temporal score (inclusive, 24h)
    evening_start_hour:     int   = 18   # 6 PM
    evening_end_hour:       int   = 24   # midnight

    # Default suitability prior
    default_category_prior: float = 0.50

    # Places API category → suitability prior [0, 1]
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
        "nightclub":            0.55,
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
        "fast_food":            0.25,
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
