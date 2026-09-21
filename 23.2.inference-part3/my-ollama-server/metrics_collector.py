import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict
import time

@dataclass
class MetricsCollector:
    window_seconds: float = 60.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    ttft_samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    token_counts: deque = field(default_factory=lambda: deque(maxlen=10000))
    error_counts: Dict[str, int] = field(default_factory=dict)

    def _prune_old_samples(self, cutoff: float):
        """Removes expired timestamps from the front of the queues."""
        while self.latency_samples and self.latency_samples[0][0] < cutoff:
            self.latency_samples.popleft()
        while self.ttft_samples and self.ttft_samples[0][0] < cutoff:
            self.ttft_samples.popleft()
        while self.token_counts and self.token_counts[0][0] < cutoff:
            self.token_counts.popleft()

    def record_request(self, latency_ms: float, ttft_ms: float, total_tokens: int, error: str = None):
        timestamp = time.time()
        self.latency_samples.append((timestamp, latency_ms))
        self.ttft_samples.append((timestamp, ttft_ms))
        self.token_counts.append((timestamp, total_tokens))
        if error:
            self.error_counts[error] = self.error_counts.get(error, 0) + 1
        self._prune_old_samples(timestamp - self.window_seconds)

    def get_current_metrics(self) -> dict:
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        self._prune_old_samples(cutoff)

        recent_latencies = [v for _, v in self.latency_samples]
        recent_ttft = [v for _, v in self.ttft_samples]
        recent_tokens = [v for _, v in self.token_counts]

        if not recent_latencies:
            return {"status": "no_data_yet", "window_seconds": self.window_seconds}

        return {
            "window_seconds": self.window_seconds,
            "latency_p50_ms": float(np.percentile(recent_latencies, 50)),
            "latency_p95_ms": float(np.percentile(recent_latencies, 95)),
            "latency_p99_ms": float(np.percentile(recent_latencies, 99)),
            "ttft_p50_ms": float(np.percentile(recent_ttft, 50)),
            "ttft_p95_ms": float(np.percentile(recent_ttft, 95)),
            "requests_per_minute": len(recent_latencies),
            "tokens_per_minute": int(sum(recent_tokens)),
            "error_counts": dict(self.error_counts),
        }
