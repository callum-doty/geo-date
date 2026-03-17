"""
tests/test_suite.py
===================
Full regression test suite — 10 groups, ~40 assertions.

    python -m matching_engine.tests.test_suite
    python -m matching_engine.tests.run_all      # smoke + suite together
"""

from __future__ import annotations

import datetime

from matching_engine.config.match_config import MatchConfig
from matching_engine.models.cluster import LocationCluster, ClusterObservation
from matching_engine.models.user import UserProfile
from matching_engine.models.ping import Ping
from matching_engine.pipeline.significance import route_ping
from matching_engine.pipeline.stream import enforce_pins
from matching_engine.similarity.match import match_users
from matching_engine.similarity.vectors import idf_normalize_sparse
from matching_engine.venue.ranking import score_venue_dynamic, rank_venues_dynamic, _haversine_minutes
from matching_engine.venue.suitability import (
    alpha_confidence, category_prior,
    suitability_behavioral, suitability_posterior, is_venue_eligible,
)
from matching_engine.tests.fixtures import build_test_registry


def run(cfg: MatchConfig = None) -> tuple[int, int]:
    cfg = cfg or MatchConfig()
    registry = build_test_registry(cfg)

    passed = failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
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
    check("α(10) < α(1000)",         alpha_confidence(10, cfg) < alpha_confidence(1000, cfg))
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

    check("Bar behavioral > gym behavioral",    beh_bar > beh_gym,
          f"bar={beh_bar:.3f} gym={beh_gym:.3f}")
    check("Coffee behavioral > gas behavioral", beh_cof > beh_gas,
          f"coffee={beh_cof:.3f} gas={beh_gas:.3f}")
    check("Behavioral in [0,1]",
          all(0 <= v <= 1 for v in [beh_bar, beh_gym, beh_gas]))
    empty_c = LocationCluster("empty_test", 0, 0, n_total_visits=0)
    check("Zero visits → behavioral = 0",
          suitability_behavioral(empty_c, cfg) == 0.0)

    # ── [4] Bayesian posterior ────────────────────────────────────────────────
    print("\n[4] Bayesian Posterior")
    cold = LocationCluster(
        "cold_test", 0, 0,
        places_category="music_venue", n_total_visits=0,
    )
    post_cold = suitability_posterior(cold, cfg)
    prior_mv  = category_prior("music_venue", cfg)
    check("Cold cluster posterior ≈ prior",
          abs(post_cold - prior_mv) < 0.01,
          f"posterior={post_cold:.3f} prior={prior_mv:.3f}")

    post_bar = suitability_posterior(bar_c,  cfg)
    post_gas = suitability_posterior(gas_c,  cfg)
    check("Bar posterior > gas posterior",  post_bar > post_gas,
          f"bar={post_bar:.3f} gas={post_gas:.3f}")
    fest_c   = registry.get("fest_boulevardia")
    office_c = registry.get("office_private")
    check("Posterior in [0,1]",
          all(0 <= suitability_posterior(c, cfg) <= 1
              for c in [bar_c, gym_c, gas_c, coffee_c, fest_c]))

    # ── [5] Eligibility gate ──────────────────────────────────────────────────
    print("\n[5] Eligibility Gate")
    check("bar eligible",      is_venue_eligible(bar_c,    cfg))
    check("coffee eligible",   is_venue_eligible(coffee_c, cfg))
    check("office ineligible", not is_venue_eligible(office_c, cfg))

    too_few_users  = LocationCluster("tfu", 0, 0, n_users=3,  n_total_visits=50)
    too_few_visits = LocationCluster("tfv", 0, 0, n_users=10, n_total_visits=5)
    check("Too few users → ineligible",  not is_venue_eligible(too_few_users,  cfg))
    check("Too few visits → ineligible", not is_venue_eligible(too_few_visits, cfg))
    check("Registry eligible set populated",
          len(registry.venue_eligible_ids) > 0)
    check("Office not in eligible set",
          "office_private" not in registry.venue_eligible_ids)

    # ── [6] Cluster observation recording ────────────────────────────────────
    print("\n[6] Cluster Observation Recording")
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

    check("n_total_visits incremented",    test_cluster.n_total_visits == pre_visits + 1)
    check("evening_visits incremented",    test_cluster.evening_visits == pre_evening + 1)
    check("cooccurrence incremented",      test_cluster.cooccurrence_visits == pre_cooc + 1)
    check("date_dwell_visits incremented", test_cluster.date_dwell_visits == 1)

    pre_ev2 = test_cluster.evening_visits
    obs_morning = ClusterObservation("obs_test", morning_ts, 20.0)
    registry.record_observation(obs_morning, cfg)
    check("Morning ping doesn't increment evening_visits",
          test_cluster.evening_visits == pre_ev2)

    # ── [7] Dynamic venue scoring ─────────────────────────────────────────────
    print("\n[7] Dynamic Venue Scoring")
    u_a = UserProfile("score_a",
                      home_base_lat=39.082, home_base_lng=-94.587,
                      home_base_commute=10.0)
    u_b = UserProfile("score_b",
                      home_base_lat=39.089, home_base_lng=-94.582,
                      home_base_commute=15.0)

    idf = registry.idf_snapshot
    u_a.V_rhythm["bar_recordbar"]    = 2.0
    u_b.V_rhythm["bar_recordbar"]    = 1.8
    u_a.V_identity["fest_boulevardia"] = 1.5
    u_b.V_identity["fest_boulevardia"] = 1.4

    score_bar  = score_venue_dynamic(u_a, u_b, bar_c,    cfg, idf=idf)
    score_gas  = score_venue_dynamic(u_a, u_b, gas_c,    cfg, idf=idf)
    score_fest = score_venue_dynamic(u_a, u_b, fest_c,   cfg, idf=idf)
    score_off  = score_venue_dynamic(u_a, u_b, office_c, cfg, idf=idf)

    check("Bar scores (shared rhythm signal)",    score_bar is not None,
          f"score={score_bar}")
    check("Festival scores (shared identity)",    score_fest is not None,
          f"score={score_fest}")
    check("Gas station gated (low intersection)", score_gas is None)
    check("Office ineligible → None",             score_off is None)
    if score_bar is not None:
        check("Bar score in reasonable range",
              -0.5 < score_bar < 1.5, f"score={score_bar:.4f}")

    # ── [8] Dynamic ranking ───────────────────────────────────────────────────
    print("\n[8] Dynamic Venue Ranking")
    ranked = rank_venues_dynamic(u_a, u_b, registry, cfg)
    rec    = [r for r in ranked if not r.gated]
    gated  = [r for r in ranked if r.gated]

    check("Ranking returns results",     len(ranked) > 0)
    check("Recommended venues exist",    len(rec) > 0)
    check("Recommended sorted desc",
          all(rec[i].score >= rec[i + 1].score for i in range(len(rec) - 1)))
    check("Gated venues after recommended",
          all(r.gated for r in ranked[len(rec):]))
    check("Office not in results (ineligible)",
          all(r.venue.venue_id != "office_private" for r in ranked))

    # ── [9] Haversine travel estimate ─────────────────────────────────────────
    print("\n[9] Haversine Travel Estimate")
    same = _haversine_minutes(39.082, -94.587, 39.082, -94.587)
    check("Same point → 0 min", same == 0.0)
    near = _haversine_minutes(39.082, -94.587, 39.091, -94.587)
    check("~1km → roughly 2 min", 1.0 < near < 4.0, f"got {near:.2f}")
    far  = _haversine_minutes(39.082, -94.587, 40.000, -95.000)
    check("Further = more time", far > near)

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
        u.n_rhythm   = len(rhythm_cids)
        u.n_identity = len(identity_cids)
        return u

    ua = build_user("ua",
        rhythm_cids=[("bar_recordbar", 1.8), ("park_loose", 0.9)],
        identity_cids=[("fest_boulevardia", 1.4)],
        lat=39.082, lng=-94.587, tags=["jazz", "outdoors"])
    ub = build_user("ub",
        rhythm_cids=[("bar_recordbar", 1.6), ("park_loose", 0.8)],
        identity_cids=[("fest_boulevardia", 1.3)],
        lat=39.089, lng=-94.582, tags=["jazz", "hiking"])
    uc = build_user("uc",
        rhythm_cids=[("coffee_monarch", 2.0), ("gym_crossroads", 1.5)],
        identity_cids=[],
        lat=39.070, lng=-94.600, tags=["coffee", "fitness"])

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
    print(f"\n{'─' * 70}")
    print(f"  {passed} passed  |  {failed} failed  |  {passed + failed} total")
    if failed == 0:
        print("  ✓ All tests passed")
    else:
        print(f"  ✗ {failed} test(s) FAILED")
    print(f"{'─' * 70}")

    return passed, failed


if __name__ == "__main__":
    run()