"""Lightweight per-request performance timing (all values in milliseconds)."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List

logger = logging.getLogger("perf")


class RequestTimer:
    """Accumulates named stage timings for one request and logs them in ms."""

    def __init__(self, label: str = "request") -> None:
        self.label = label
        self._t0 = time.perf_counter()
        self.stages: Dict[str, float] = {}
        self._order: List[str] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name: str, ms: float) -> None:
        if name not in self.stages:
            self._order.append(name)
            self.stages[name] = 0.0
        self.stages[name] += ms
        logger.info("%s: %.1f ms", name, self.stages[name])

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def finish(self) -> Dict[str, float]:
        total = self.total_ms
        logger.info("Total request time: %.1f ms", total)
        out = {k: round(self.stages[k], 1) for k in self._order}
        out["total"] = round(total, 1)
        return out


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Standalone timer that logs a single stage in ms."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s: %.1f ms", name, (time.perf_counter() - start) * 1000.0)
