"""
matching_engine
===============
Public API for the geo-date behavioral matching engine.
"""

# Configuration
from matching_engine.config.match_config import MatchConfig, DEFAULT_CFG

# Models
from matching_engine.models.ping import (
    Ping,
    BufferedPing,
    Stream,
    StreamAssignment,
    compute_time_bin,
    make_dimension_key,
)
from matching_engine.models.cluster import (
    LocationCluster,
    H3CellRecord,
    ClusterObservation,
    ClusterRegistry,
)
from matching_engine.models.user import UserProfile, PinnedWeight
from matching_engine.models.results import WeightSet, MatchResult, Venue, VenueResult

# Pipeline
from matching_engine.pipeline.significance import compute_significance, route_ping
from matching_engine.pipeline.stream import (
    decay_stream,
    apply_rhythm_ping,
    apply_identity_ping,
    enforce_pins,
)
from matching_engine.pipeline.buffer import PendingBuffer
from matching_engine.pipeline.transitions import update_transition_matrix

# Similarity
from matching_engine.similarity.match import adaptive_weights, match_users
from matching_engine.similarity.vectors import (
    idf_normalize_sparse,
    cosine_sparse,
    bio_similarity,
)
from matching_engine.similarity.transitions import edge_similarity
from matching_engine.similarity.proximity import proximity_score, geometric_median

# Co-occurrence
from matching_engine.cooccurrence.bloom import (
    BloomFilter,
    generate_token,
    copresence_score,
    symmetric_copresence_score,
    daily_salt,
)

# Venue
from matching_engine.venue.ranking import rank_venues_dynamic
from matching_engine.venue.suitability import suitability_posterior

# Utilities
from matching_engine.utils.drift import vector_has_drifted
from matching_engine.utils.profile import profile_strength

__all__ = [
    # config
    "MatchConfig", "DEFAULT_CFG",
    # models
    "Ping", "BufferedPing", "Stream", "StreamAssignment",
    "compute_time_bin", "make_dimension_key",
    "LocationCluster", "H3CellRecord", "ClusterObservation", "ClusterRegistry",
    "UserProfile", "PinnedWeight",
    "WeightSet", "MatchResult", "Venue", "VenueResult",
    # pipeline
    "compute_significance", "route_ping",
    "decay_stream", "apply_rhythm_ping", "apply_identity_ping", "enforce_pins",
    "PendingBuffer", "update_transition_matrix",
    # similarity
    "adaptive_weights", "match_users",
    "idf_normalize_sparse", "cosine_sparse", "bio_similarity",
    "edge_similarity",
    "proximity_score", "geometric_median",
    # cooccurrence
    "BloomFilter", "generate_token", "copresence_score",
    "symmetric_copresence_score", "daily_salt",
    # venue
    "rank_venues_dynamic", "suitability_posterior",
    # utils
    "vector_has_drifted", "profile_strength",
]
