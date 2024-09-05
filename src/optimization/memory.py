"""Peak RAM usage context manager using psutil.

Usage:
    with peak_ram_mb() as tracker:
        run_inference()
    print(f"Peak RAM: {tracker.peak_mb:.1f} MB")
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil


@dataclass
class _RamTracker:
    peak_mb: float = 0.0
    _running: bool = field(default=True, repr=False)

    def stop(self) -> None:
        self._running = False


@contextmanager
def peak_ram_mb(poll_interval_s: float = 0.05):
    """Context manager that tracks peak RSS memory in MB during execution.

    Yields a _RamTracker whose .peak_mb attribute is populated on exit.
    """
    tracker = _RamTracker()
    proc = psutil.Process()

    def _poll() -> None:
        while tracker._running:
            mb = proc.memory_info().rss / (1024 ** 2)
            if mb > tracker.peak_mb:
                tracker.peak_mb = mb
            time.sleep(poll_interval_s)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        t.join(timeout=1.0)
