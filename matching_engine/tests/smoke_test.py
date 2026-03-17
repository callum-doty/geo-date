"""
tests/smoke_test.py
===================
Quick end-to-end smoke test: builds two users, runs the full pipeline,
prints suitability posteriors, match result, venue rankings, and profile
strength. Intended to be run manually during development.

    python -m matching_engine.tests.smoke_test
"""

from __future__ import annotations

from matching_engine.config.match_config import MatchConfig
from matching_engine.models.user import UserProfile
from matching_engine.models.ping import Ping
from matching_engine.pipeline.significance import route_ping
from matching_engine.pipeline.stream import enforce_pins
from matching_engine.similarity.match import match_users
from matching_engine.venue.ranking import rank_venues_dynamic
from matching_engine.venue.suitability import (
    alpha_confidence, category_prior,
    suitability_behavioral, suitability_posterior, is_venue_eligible,
)
from matching_engine.utils.profile import profile_strength
from matching_engine.tests.fixtures import build_test_registry


def run(cfg: MatchConfig = None) -> tuple[UserProfile, UserProfile]:
    cfg = cfg or MatchConfig()
    cfg.validate()

    registry = build_test_registry(cfg)

    print("─" * 70)
    print("matching_engine  — smoke test")
    print("─" * 70)
    print(f"✓ Config validated")
    print(f"✓ Registry: {len(registry.clusters)} clusters  "
          f"eligible: {len(registry.venue_eligible_ids)}")

    # Suitability posteriors
    coffee_c = registry.get("coffee_monarch")
    bar_c    = registry.get("bar_recordbar")
    gas_c    = registry.get("gas_quiktrip")
    office_c = registry.get("office_private")

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
    jordan = UserProfile(
        "jordan", tags=["jazz", "fitness", "vinyl", "outdoors"],
        home_base_commute=12.0,
        home_base_lat=39.082, home_base_lng=-94.587,
    )
    riley = UserProfile(
        "riley", tags=["jazz", "fitness", "coffee", "hiking"],
        home_base_commute=18.0,
        home_base_lat=39.089, home_base_lng=-94.582,
    )

    rhythm_schedule = [
        ("gym_crossroads", 55, 1.0), ("coffee_monarch", 35, 0.5),
        ("gym_crossroads", 60, 1.0), ("park_loose",     45, 0.5),
        ("gym_crossroads", 50, 1.0), ("coffee_monarch", 30, 0.5),
        ("bar_recordbar",  90, 1.0),
    ]
    for cid, dwell, dt in rhythm_schedule:
        cluster = registry.get(cid)
        p = Ping(
            lat=cluster.centroid_lat, lng=cluster.centroid_lng,
            dwell_minutes=dwell, delta_t_days=dt,
            resolved_cluster_id=cid, n_users_in_window=cluster.n_users,
        )
        for user in [jordan, riley]:
            a = route_ping(p, cluster, cfg, registry.n_total_users)
            user.add_ping(a, p, cluster, cfg)

    # Festival identity pings
    fest = registry.get("fest_boulevardia")
    for user in [jordan, riley]:
        p = Ping(
            lat=fest.centroid_lat, lng=fest.centroid_lng,
            dwell_minutes=2700, resolved_cluster_id=fest.cluster_id,
            n_users_in_window=12,
        )
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


if __name__ == "__main__":
    run()