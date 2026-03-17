"""
models/ping.py
==============
Ping, BufferedPing, StreamAssignment, and Stream enum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig


class Stream(Enum):
    RHYTHM   = auto()   # V_rhythm  — daily life, fast decay
    IDENTITY = auto()   # V_identity — defining traits, slow/pinned decay
    BUFFER   = auto()   # pending — not yet graduated


@dataclass
class Ping:
    """
    A single raw GPS venue-attendance event, pre-buffer.
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

    def has_graduated(self, cfg: "MatchConfig") -> bool:
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
