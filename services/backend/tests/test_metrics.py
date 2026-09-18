from paperscope.metrics import SlidingWindowRateLimiter


def test_rate_limiter_allows_limit_and_rejects_the_next_request() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow()
    assert limiter.allow()
    assert not limiter.allow()
