from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from threading import Lock
from time import perf_counter


def directory_size(path: Path) -> int:
    """Return the size of files below a path without following symlinks."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


class MetricsRegistry:
    """Small process-local metrics registry for the single-user local API."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._request_latency_ms = 0.0
        self._request_paths: Counter[str] = Counter()
        self._request_statuses: Counter[str] = Counter()
        self._search_count = 0
        self._search_latency_ms = 0.0
        self._rag_count = 0
        self._rag_errors = 0
        self._rag_fallbacks = 0
        self._rag_grounded = 0
        self._rag_missing_citations = 0
        self._rag_rate_limited = 0
        self._rag_latency_ms = 0.0

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._request_latency_ms += latency_ms
            self._request_paths[path] += 1
            self._request_statuses[str(status_code)] += 1

    def record_search(self, latency_ms: float) -> None:
        with self._lock:
            self._search_count += 1
            self._search_latency_ms += latency_ms

    def record_rag(self, *, latency_ms: int, fallback_used: bool, grounded: bool) -> None:
        with self._lock:
            self._rag_count += 1
            self._rag_latency_ms += latency_ms
            self._rag_fallbacks += int(fallback_used)
            self._rag_grounded += int(grounded)
            self._rag_missing_citations += int(not grounded)

    def record_rag_error(self) -> None:
        with self._lock:
            self._rag_errors += 1

    def record_rag_rate_limit(self) -> None:
        with self._lock:
            self._rag_rate_limited += 1

    @staticmethod
    def _average(total: float, count: int) -> float:
        return round(total / count, 2) if count else 0.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "requests": {
                    "total": self._request_count,
                    "averageLatencyMs": self._average(self._request_latency_ms, self._request_count),
                    "byPath": dict(self._request_paths),
                    "byStatus": dict(self._request_statuses),
                },
                "search": {
                    "total": self._search_count,
                    "averageLatencyMs": self._average(self._search_latency_ms, self._search_count),
                },
                "rag": {
                    "total": self._rag_count,
                    "errors": self._rag_errors,
                    "fallbacks": self._rag_fallbacks,
                    "grounded": self._rag_grounded,
                    "missingCitations": self._rag_missing_citations,
                    "rateLimited": self._rag_rate_limited,
                    "averageLatencyMs": self._average(self._rag_latency_ms, self._rag_count),
                },
            }


metrics = MetricsRegistry()


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60) -> None:
        self.limit = max(1, limit)
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> bool:
        now = perf_counter()
        with self._lock:
            while self._events and now - self._events[0] >= self.window_seconds:
                self._events.popleft()
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True
