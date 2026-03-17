"""
utils/profile.py
================
Profile Strength — UI progress card metrics.

Makes the data collection mechanic legible to the user by surfacing
their current behavioral phase and what they need to unlock next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.user import UserProfile


def profile_strength(user: "UserProfile", cfg: "MatchConfig") -> dict:
    """
    Compute the profile strength metrics shown in the app UI.

    Phases (matching MatchResult.phase labels):
        Discovery     : 0–9 rhythm pings,  0 identity pings
        Rhythm-Active : ≥10 rhythm pings,  0–2 identity pings
        Dual-Stream   : ≥10 rhythm,        3–9 identity pings
        Identity-Rich : ≥10 rhythm,        ≥10 identity pings

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
    RHYTHM_TARGET   = 20
    IDENTITY_TARGET = 10

    r_pct   = min(user.n_rhythm   / RHYTHM_TARGET,   1.0) * 100
    i_pct   = min(user.n_identity / IDENTITY_TARGET, 1.0) * 100
    overall = r_pct * 0.6 + i_pct * 0.4

    # Replicate _match_phase logic without importing from similarity.match
    # to avoid a circular import through the models layer.
    n_r, n_i = user.n_rhythm, user.n_identity
    if n_i >= 10:
        phase = "Identity-Rich"
    elif n_r >= 10 and n_i >= 3:
        phase = "Dual-Stream"
    elif n_r >= 10:
        phase = "Rhythm-Active"
    else:
        phase = "Discovery"

    if phase == "Discovery":
        next_milestone = (
            f"Check in {max(0, 10 - user.n_rhythm)} more times "
            f"to unlock behavioral matching"
        )
    elif phase == "Rhythm-Active":
        next_milestone = "Attend a rare event or festival to activate your Identity stream"
    elif phase == "Dual-Stream":
        next_milestone = (
            f"Build {max(0, 10 - user.n_identity)} more identity signals "
            f"to unlock the deepest matches"
        )
    else:
        next_milestone = (
            "Your profile is fully built — "
            "matches now reflect both your rhythm and your soul"
        )

    active = (
        len([v for v in user.V_rhythm.values()   if v > 1e-4])
        + len([v for v in user.V_identity.values() if v > 1e-4])
    )

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