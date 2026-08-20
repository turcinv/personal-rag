"""Bounded inference concurrency for the API.

The Jetson shares 8 GB between CPU and GPU, so several simultaneous embed/rerank/
generate calls can exhaust memory. This limiter caps how many requests run model
inference at once (``api.inference_concurrency``, default 1 — fully serialized).
A request that cannot get a slot within a short timeout gets 503 rather than
piling up and OOM-ing the box.

The limiter lives in ``app.state.rag['inference_limiter']``; routes wrap their
``search()``/``generate()`` call in :func:`guard`. Tests inject a fake state
without a limiter, so ``guard`` is a no-op when one is absent.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from fastapi import HTTPException, status


class InferenceLimiter:
    """A bounded semaphore over model inference with a fail-fast timeout."""

    def __init__(self, max_concurrency: int = 1, acquire_timeout: float = 30.0):
        self.max_concurrency = max(1, int(max_concurrency))
        self.acquire_timeout = float(acquire_timeout)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)

    @contextmanager
    def slot(self):
        acquired = self._semaphore.acquire(timeout=self.acquire_timeout)
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="server is busy (inference concurrency limit reached); retry shortly",
                headers={"Retry-After": "1"},
            )
        try:
            yield
        finally:
            self._semaphore.release()


@contextmanager
def guard(state: dict):
    """Acquire an inference slot if the app configured a limiter; else no-op."""
    limiter = (state or {}).get("inference_limiter")
    if limiter is None:
        yield
        return
    with limiter.slot():
        yield
