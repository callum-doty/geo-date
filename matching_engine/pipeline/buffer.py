"""
pipeline/buffer.py
==================
PendingBuffer — transit filter for raw GPS pings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matching_engine.config.match_config import MatchConfig
    from matching_engine.models.ping import Ping, BufferedPing


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
    """

    VELOCITY_MIN_GAP_SECS = 60   # consecutive pings closer than this are transit

    def __init__(self, cfg: "MatchConfig" = None):
        from matching_engine.config.match_config import DEFAULT_CFG
        self.cfg = cfg or DEFAULT_CFG
        self._buffer: dict[str, "BufferedPing"] = {}

    @staticmethod
    def _location_key(lat: float, lng: float, precision: int = 4) -> str:
        """
        Round coordinates to ~11m precision (4 decimal places).
        Fallback when H3 index is unavailable.
        """
        return f"{round(lat, precision)},{round(lng, precision)}"

    @staticmethod
    def _key_for_ping(ping: "Ping") -> str:
        """
        Derive the buffer key for a ping.

        Preference order:
          1. H3 r10 index (privacy-preserving, set on-device)
          2. Rounded lat/lng (legacy fallback for tests / server-side creation)
        """
        if ping.h3_r10 is not None:
            return str(ping.h3_r10)
        return PendingBuffer._location_key(ping.lat, ping.lng)

    def ingest(self, ping: "Ping") -> Optional["BufferedPing"]:
        """
        Add a raw ping to the buffer.

        Returns the BufferedPing record if this ping caused graduation,
        else None (still accumulating).

        Velocity check: if the previous ping at this location was within
        VELOCITY_MIN_GAP_SECS, this ping is treated as transit and dropped.
        """
        from matching_engine.models.ping import BufferedPing

        key = self._key_for_ping(ping)

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
