import threading
import time
from collections import defaultdict, deque

from app.core.errors import RateLimitExceededError

_SWEEP_ABOVE = 10_000  # tidy up when this many different keys have been seen


class RateLimiter:
    """Sliding window: at most `limit_per_minute` calls per key in any 60 seconds.

    Lives in this process's memory, so with several app copies each has its own count.
    A shared store (Redis) would be needed then.
    """

    def __init__(self, limit_per_minute: int):
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > _SWEEP_ABOVE:
                self._sweep(now)
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self._limit:
                raise RateLimitExceededError("Too many requests. Please wait a moment and try again.")
            hits.append(now)

    def _sweep(self, now: float) -> None:
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > 60]:
            del self._hits[key]
