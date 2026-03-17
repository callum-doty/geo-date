matching_engine/
├── __init__.py               ← full public API re-exported (drop-in for old imports)
├── config/
│   └── match_config.py       ← MatchConfig + DEFAULT_CFG
├── models/
│   ├── cluster.py            ← LocationCluster, ClusterObservation, ClusterRegistry
│   ├── ping.py               ← Ping, BufferedPing, StreamAssignment, Stream
│   ├── user.py               ← UserProfile, PinnedWeight
│   └── results.py            ← WeightSet, MatchResult, VenueResult, Venue
├── pipeline/
│   ├── significance.py       ← compute_significance(), route_ping()
│   ├── stream.py             ← decay_stream(), apply_*_ping(), enforce_pins()
│   └── buffer.py             ← PendingBuffer
├── similarity/
│   ├── vectors.py            ← idf_normalize_sparse(), cosine_sparse(), bio_similarity()
│   ├── proximity.py          ← proximity_score(), geometric_median()
│   └── match.py              ← adaptive_weights(), match_users()
├── venue/
│   ├── suitability.py        ← Bayesian suitability (Pillar 5a)
│   └── ranking.py            ← score_venue_dynamic(), rank_venues_dynamic()
├── utils/
│   ├── drift.py              ← vector_has_drifted()
│   └── profile.py            ← profile_strength()
└── tests/
    ├── fixtures.py           ← shared registry (single source of truth)
    ├── smoke_test.py         ← end-to-end pipeline walkthrough
    ├── test_suite.py         ← 47 assertions across 10 groups
    └── run_all.py            ← entry point: python -m matching_engine.tests.run_all