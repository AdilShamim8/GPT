import collections
import math
from typing import Dict, Optional


class MetricTracker:
    """Tracks running averages and windowed statistics for training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history = collections.defaultdict(lambda: collections.deque(maxlen=window_size))
        self._totals = collections.defaultdict(float)
        self._counts = collections.defaultdict(int)

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Update tracker with metric value."""
        self._history[name].append(value)
        self._totals[name] += value * n
        self._counts[name] += n

    def get_window_avg(self, name: str) -> float:
        """Get rolling window average."""
        window = self._history[name]
        return sum(window) / len(window) if window else 0.0

    def get_global_avg(self, name: str) -> float:
        """Get global lifetime average."""
        count = self._counts[name]
        return self._totals[name] / count if count > 0 else 0.0

    def get_perplexity(self, loss_name: str = "loss") -> float:
        """Compute perplexity = exp(loss) safely without overflow."""
        loss = self.get_window_avg(loss_name)
        return math.exp(min(loss, 100.0))

    def summary(self) -> Dict[str, float]:
        """Return dict of current metric window averages."""
        res = {k: self.get_window_avg(k) for k in self._history}
        if "loss" in res:
            res["perplexity"] = self.get_perplexity("loss")
        return res
