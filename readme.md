# geo-date

A behavioral matching engine for a dating app that matches people on how they actually live — not how they describe themselves.

---

## The Idea

Most dating apps match on self-reported interests. geo-date matches on real-world behavior: where you go, when you go, how long you stay, and how you move between places. A person who goes to the gym every weekday morning and then a coffee shop is expressing a behavioral structure. Someone else with the same structure — regardless of city — is a meaningful match candidate.

> *"Locations are not features. Distributions of behavior over locations are features."*

---

## How It Works

### Two Behavioral Streams

Every user has two sparse vectors built from their location history:

| Stream | What it captures | Decay half-life |
|---|---|---|
| **V_rhythm** | Daily habits — gym, coffee, lunch spots, parks | ~9 days |
| **V_identity** | Rare, meaningful events — concerts, festivals, unique bars | ~350 days |

Routine behavior fades quickly if you stop going. Defining experiences linger for close to a year.

### The Pipeline

```
GPS ping (on-device)
  → H3 hexagonal cell index (~15m precision)
  → Visit detection + dwell measurement
  → Significance score: S = venue_rarity × event_duration × dwell_intensity
  → Buffer (must clear min visits + dwell before graduating)
  → Stream routing: S ≥ 0.40 → V_identity, else → V_rhythm
  → Vector update with exponential decay
  → Transition matrix update (from_cluster → to_cluster)
  → Encrypted vector delta synced to server
```

Raw GPS coordinates never leave the device. The server receives only behavioral vector updates.

### Dual-Layer Clustering

Location identity is resolved in two stages:

**Layer 1 — Spatial:** H3 hexagonal grid provides a universal, globally-consistent spatial primitive. Resolution 10 (~15m) for identity stream; resolution 8 (~460m) for rhythm stream.

**Layer 2 — Behavioral:** H3 cells are aggregated across all users into behavior signatures (visit time distribution, dwell shape, repeat rate, visitor diversity). These signatures are clustered globally into behavioral archetypes: `C1 = coffee shop behavior`, `C2 = gym behavior`, `C3 = evening bar`, etc.

A coffee shop in Kansas City and a coffee shop in Chicago both resolve to the same archetype. Users match on structural behavioral similarity even across different cities.

Vector dimension keys encode both location and time: `"cluster_id|WD_2"` — behavioral archetype, weekday 06:00–09:00. "Coffee at 7am on weekdays" and "coffee at 2pm on weekends" are distinct dimensions.

### The Match Score

```
G(A,B) = w_rhythm     · cosine(V_rhythm_A,   V_rhythm_B)
        + w_identity   · cosine(V_identity_A, V_identity_B)
        + w_log        · proximity(avg_commute)
        + w_bio        · bio_similarity(tags_A, tags_B)
        + w_edge       · edge_similarity(E_A, E_B)
        + w_copresence · copresence(A, B)
```

**Weights are adaptive** — they shift automatically as behavioral profiles mature:

| Stage | Dominant weight |
|---|---|
| No pings | Bio similarity (fallback to stated interests) |
| 10+ rhythm pings | Rhythm similarity enters |
| 3+ identity pings | Identity similarity grows |
| 10+ identity pings | Identity-weighted matching |

### Transition Matrix

Each user maintains a sparse transition matrix `E` capturing behavioral pathways:

```
E["gym|WD_2→coffee|WD_2"] = 0.74   # gym → coffee, weekday mornings
E["work|WD_4→bar|WD_5"]   = 0.51   # work → bar, weekday evenings
```

`edge_similarity(E_A, E_B)` matches users on movement structure, not just place overlap. Two people whose Friday evenings consistently move from work to a bar to late dinner are matched on that behavioral process.

### Co-presence Detection

Detects when two users were at the same venue at the same time — without the server ever knowing which venue.

```
token = HMAC-SHA256(rotating_daily_salt, H3_cell || time_window)
```

Tokens are stored in a Bloom filter per user. Intersection detection reveals co-presence without revealing location. The rotating salt (daily, device-held) prevents cross-period correlation.

---

## Privacy Architecture

| Data | Where it lives |
|---|---|
| Raw GPS coordinates | Device only, never transmitted |
| H3 cell indices | Transmitted as integers, not coordinates |
| Behavior signatures | Aggregated across all users, not user-level |
| Vector dimensions | `cluster_id\|time_bin` — archetypes, not places |
| Co-presence tokens | HMAC-hashed — server cannot reverse to a location |
| Matching | Runs on behavioral vectors with no geographic metadata |

---

## Profile Phases

| Phase | Condition | What drives matching |
|---|---|---|
| Discovery | < 10 rhythm pings | Bio similarity only |
| Rhythm-Active | ≥ 10 rhythm, < 3 identity | Rhythm + bio |
| Dual-Stream | ≥ 10 rhythm + ≥ 3 identity | Full behavioral matching |
| Identity-Rich | ≥ 10 identity pings | Identity-weighted |

---

## Venue Recommendations

Venues are not curated from a static list. They emerge from the cluster registry when a cluster clears the eligibility gate (`n_users ≥ 5`, `n_total_visits ≥ 10`). Each eligible cluster is scored for a matched pair:

```
V(k) = α_r · rhythm_intersection(k)
      + α_i · identity_intersection(k)
      + 0.20 · suitability_posterior(k)
      - γ   · travel_penalty(k)
```

Suitability is Bayesian — a category prior (coffee shop = 0.75, jazz club = 0.92, gas station = 0.02) is updated with behavioral evidence: evening visits, co-occurrence patterns, and date-length dwell times.

---

## Project Structure

```
matching_engine/
├── __init__.py                    # Full public API
├── config/
│   └── match_config.py           # All tunable parameters + DEFAULT_CFG
├── models/
│   ├── ping.py                   # Ping, BufferedPing, StreamAssignment, Stream
│   │                             #   compute_time_bin(), make_dimension_key()
│   ├── cluster.py                # LocationCluster, H3CellRecord, ClusterRegistry
│   ├── user.py                   # UserProfile, PinnedWeight
│   └── results.py                # WeightSet, MatchResult, Venue, VenueResult
├── pipeline/
│   ├── buffer.py                 # PendingBuffer — transit filter for raw pings
│   ├── significance.py           # compute_significance(), route_ping()
│   ├── stream.py                 # decay_stream(), apply_*_ping(), enforce_pins()
│   └── transitions.py            # update_transition_matrix()
├── similarity/
│   ├── vectors.py                # idf_normalize_sparse(), cosine_sparse(), bio_similarity()
│   ├── transitions.py            # edge_similarity()
│   ├── match.py                  # adaptive_weights(), match_users()
│   └── proximity.py              # proximity_score(), geometric_median()
├── cooccurrence/
│   └── bloom.py                  # BloomFilter, generate_token(), copresence_score()
├── venue/
│   ├── suitability.py            # Bayesian suitability posterior
│   └── ranking.py                # score_venue_dynamic(), rank_venues_dynamic()
├── utils/
│   ├── drift.py                  # vector_has_drifted() — match cache invalidation
│   └── profile.py                # profile_strength() — UI phase + progress
└── tests/
    ├── fixtures.py               # Shared registry (single source of truth)
    ├── smoke_test.py             # End-to-end pipeline walkthrough
    ├── test_suite.py             # 47 assertions across 10 groups
    └── run_all.py                # python3 -m matching_engine.tests.run_all
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `s_threshold` | 0.40 | Significance cutoff for identity routing |
| `lambda_rhythm` | 0.08 /day | Rhythm decay (~9-day half-life) |
| `lambda_identity` | 0.002 /day | Identity decay (~350-day half-life) |
| `lambda_transition` | 0.04 /day | Transition edge decay (~17-day half-life) |
| `pin_floor` | 0.20 | Minimum weight held during identity pin |
| `pin_duration_days` | 365 | How long an identity pin is active |
| `w_bio_max` | 0.55 | Bio weight at zero behavioral pings |
| `w_identity_max` | 0.30 | Max identity weight at full maturity |
| `w_edge_fixed` | 0.08 | Fixed contribution of transition similarity |
| `w_copresence_fixed` | 0.05 | Fixed contribution of co-presence signal |
| `h3_resolution_identity` | 10 | H3 resolution for identity stream (~15m cells) |
| `h3_resolution_rhythm` | 8 | H3 resolution for rhythm stream (~460m cells) |
| `time_bin_hours` | 3 | Time slot width (yields 16 bins: 8 × weekday/weekend) |
| `buffer_min_visits` | 2 | Visits required before a location graduates |
| `buffer_min_dwell_mins` | 20.0 | Cumulative dwell required (minutes) |
| `transition_window_hours` | 4.0 | Max time gap to record a behavioral transition |
| `bloom_error_rate` | 0.01 | Bloom filter false-positive rate |

---

## Running Tests

```bash
python3 -m matching_engine.tests.run_all
```

47 assertions across 10 test groups: category priors, alpha confidence schedule, behavioral suitability, Bayesian posteriors, eligibility gates, cluster observation recording, dynamic venue scoring, venue ranking, haversine travel estimation, and full pipeline discrimination.

---

## Dependencies

- Python 3.13
- `numpy`
- `scipy`
- `matplotlib`

The core matching engine and co-occurrence module have no external dependencies beyond the standard library.
