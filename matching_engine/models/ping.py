"""
models/ping.py
==============
Ping, BufferedPing, StreamAssignment, and Stream enum.
"""

from __future__ import annotations

import datetime
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


def compute_time_bin(timestamp: float, bin_hours: int = 3) -> str:
    """
    Discretise a Unix timestamp into a coarse time bin.

    Produces 16 bins: 8 time slots × 2 day types (weekday / weekend).
    With default bin_hours=3: slots 0–7 covering 00:00–23:59.

    Examples (bin_hours=3):
        Monday   06:15  → "WD_2"
        Saturday 21:00  → "WE_7"
        Friday   23:50  → "WD_7"
    """
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    slot = dt.hour // bin_hours
    day_type = "WE" if dt.weekday() >= 5 else "WD"
    return f"{day_type}_{slot}"


def make_dimension_key(cluster_id: str, time_bin: str = "") -> str:
    """
    Compose the vector dimension key.

    New format (time_bin set):  "cluster_id|WD_2"
    Legacy format (no time_bin): "cluster_id"

    The | separator is unambiguous — cluster_ids may contain underscores
    but never a pipe character.
    """
    if time_bin:
        return f"{cluster_id}|{time_bin}"
    return cluster_id


@dataclass
class Ping:
    """
    A single venue-attendance event, pre-buffer.

    On-device: h3_r10 and h3_r8 are computed from lat/lng before transmission.
    Raw lat/lng are retained here for backward compatibility with tests and
    server-side home-base computation, but are never stored server-side.
    """
    lat:                    float
    lng:                    float
    dwell_minutes:          float
    timestamp:              float = field(default_factory=time.time)
    delta_t_days:           float = 0.0
    resolved_cluster_id:    Optional[str] = None
    n_users_in_window:      int = 1
    # H3 spatial indices — set on-device, None when created from raw lat/lng
    h3_r10:                 Optional[int] = None   # ~15 m, identity stream
    h3_r8:                  Optional[int] = None   # ~460 m, rhythm stream
    # Time bin — derived from timestamp on ingest
    time_bin:               str = ""

    def __post_init__(self) -> None:
        if not self.time_bin:
            self.time_bin = compute_time_bin(self.timestamp)

    def dwell_hours(self) -> float:
        return self.dwell_minutes / 60.0

    def dimension_key(self, cluster_id: str) -> str:
        """Compose the full vector dimension key for this ping."""
        return make_dimension_key(cluster_id, self.time_bin)


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
    cluster_id:     str
    stream:         Stream
    S:              float       # Significance score
    weight:         float       # dwell-time weight to add to the vector
    dimension_key:  str = ""    # cluster_id|time_bin composite key

    def __post_init__(self) -> None:
        if not self.dimension_key:
            self.dimension_key = self.cluster_id
