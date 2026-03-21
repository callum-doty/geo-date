"""
cooccurrence/bloom.py
=====================
Privacy-preserving co-presence detection.

Protocol (from architecture design):
    1. On-device: for each visit event (h3_index, time_window):
           token = HMAC-SHA256(key=rotating_salt, msg=h3_index||time_window)
       Tokens are submitted to the server. No plaintext cell IDs transmitted.

    2. Server stores tokens in a Bloom filter per user per time window.

    3. To test co-presence between user A and B:
           score = intersection(bloom_A, tokens_B) / len(tokens_B)
       Server learns only that tokens collided — not which cell they represent.

    4. Rotating salt: salt rotates on a daily schedule, so tokens from
       different days cannot be correlated even if the salt leaks.

Privacy guarantees:
    - Server never sees raw H3 cell IDs.
    - Without the per-device salt, tokens cannot be reversed.
    - False positives possible (rate = bloom_error_rate), never false negatives.
    - Bloom filter is lossy by design — cannot reconstruct the original tokens.

No external dependencies — uses only Python stdlib (hashlib, hmac).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
import time


# ---------------------------------------------------------------------------
# Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Space-efficient probabilistic set membership structure.

    Uses k independent hash functions derived from double-hashing
    (Kirsch-Mitzenmacher technique) to avoid k separate hash computations.
    """

    def __init__(self, capacity: int = 2000, error_rate: float = 0.01) -> None:
        """
        Parameters
        ----------
        capacity   : expected number of elements to insert
        error_rate : target false-positive rate (0, 1)
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < error_rate < 1:
            raise ValueError("error_rate must be in (0, 1)")

        # Optimal bit array size and number of hash functions
        self._m: int = self._optimal_m(capacity, error_rate)
        self._k: int = self._optimal_k(self._m, capacity)
        self._bits: bytearray = bytearray(math.ceil(self._m / 8))
        self._count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, item: str | bytes) -> None:
        """Insert an item into the filter."""
        for i in self._hash_positions(item):
            self._bits[i // 8] |= (1 << (i % 8))
        self._count += 1

    def __contains__(self, item: str | bytes) -> bool:
        """Test membership. May return True for items never inserted (false positive)."""
        return all(
            self._bits[i // 8] & (1 << (i % 8))
            for i in self._hash_positions(item)
        )

    @property
    def count(self) -> int:
        """Number of items inserted (approximate)."""
        return self._count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_positions(self, item: str | bytes) -> list[int]:
        """
        Generate k bit positions using double hashing.
        h_i(x) = (h1(x) + i * h2(x)) % m
        """
        if isinstance(item, str):
            item = item.encode()
        h1 = int(hashlib.sha256(item).hexdigest(), 16)
        h2 = int(hashlib.md5(item).hexdigest(), 16)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        """Optimal bit array size: m = -n·ln(p) / (ln2)²"""
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Optimal hash count: k = (m/n)·ln2"""
        return max(1, int((m / n) * math.log(2)))


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------

def _time_window(timestamp: float, window_hours: float = 1.0) -> int:
    """
    Bucket a timestamp into a coarse time window.
    Default: 1-hour buckets — limits temporal resolution of token collisions.
    """
    return int(timestamp / (window_hours * 3600))


def generate_token(
    h3_index: int,
    timestamp: float,
    rotating_salt: bytes,
    window_hours: float = 1.0,
) -> str:
    """
    Generate a privacy-preserving co-presence token.

    token = HMAC-SHA256(key=rotating_salt, msg=h3_index || time_window)

    The rotating salt should change daily (or per-session) so that tokens
    from different periods cannot be correlated. The salt is held only on
    the device — the server never receives it.

    Parameters
    ----------
    h3_index      : H3 cell index (resolution 10)
    timestamp     : Unix timestamp of the visit event
    rotating_salt : per-device, per-period secret (bytes)
    window_hours  : time window size for bucketing (default 1 hour)

    Returns
    -------
    Hex string token (64 chars)
    """
    window = _time_window(timestamp, window_hours)
    msg = struct.pack(">qI", h3_index, window)   # 8-byte int64 + 4-byte uint32
    return hmac.new(rotating_salt, msg, hashlib.sha256).hexdigest()


def daily_salt(user_id: str, day: int | None = None) -> bytes:
    """
    Derive a daily rotating salt from user_id + day-of-epoch.

    In production the salt should be device-generated and never transmitted.
    This utility is provided for server-side simulation and testing only.

    Parameters
    ----------
    user_id : user identifier
    day     : days since epoch (defaults to today UTC)
    """
    if day is None:
        day = int(time.time() / 86400)
    material = f"{user_id}:{day}".encode()
    return hashlib.sha256(material).digest()


# ---------------------------------------------------------------------------
# Co-presence scoring
# ---------------------------------------------------------------------------

def copresence_score(
    bloom_a: BloomFilter,
    tokens_b: list[str],
) -> float:
    """
    Estimate co-presence between user A and user B.

    Score = number of B's tokens found in A's Bloom filter / len(tokens_b)

    This is an asymmetric probe: A's filter is tested against B's tokens.
    For a symmetric score, average both directions.

    Note: false positives inflate the score slightly — this is acceptable
    and tunable via bloom_error_rate in MatchConfig.

    Parameters
    ----------
    bloom_a  : BloomFilter populated with A's recent tokens
    tokens_b : list of token strings from user B

    Returns
    -------
    float ∈ [0, 1]
    """
    if not tokens_b:
        return 0.0
    hits = sum(1 for t in tokens_b if t in bloom_a)
    return hits / len(tokens_b)


def symmetric_copresence_score(
    bloom_a: BloomFilter,
    tokens_a: list[str],
    bloom_b: BloomFilter,
    tokens_b: list[str],
) -> float:
    """
    Symmetric co-presence score: average of both probe directions.

    Returns
    -------
    float ∈ [0, 1]
    """
    score_a = copresence_score(bloom_a, tokens_b)
    score_b = copresence_score(bloom_b, tokens_a)
    return (score_a + score_b) / 2.0
