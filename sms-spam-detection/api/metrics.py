"""In-process metrics exposed in Prometheus text format.

Deliberately dependency-free: the counters we need are small enough that
hand-rolling them is cheaper than pulling in prometheus_client.
"""

from __future__ import annotations

import threading
import time
from collections import Counter


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsRegistry:
    """Counters and latency totals for the HTTP layer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._requests: Counter[tuple[str, str, str]] = Counter()
        self._predictions: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._duration_sum = 0.0
        self._duration_count = 0
        self._feedback = 0

    # ---------------------------------------------------------------- writes
    def record_request(self, method: str, path: str, status: int, duration: float) -> None:
        with self._lock:
            self._requests[(method, path, str(status))] += 1
            self._duration_sum += duration
            self._duration_count += 1

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self._predictions[label] += 1

    def record_error(self, kind: str) -> None:
        with self._lock:
            self._errors[kind] += 1

    def record_feedback(self) -> None:
        with self._lock:
            self._feedback += 1

    # ----------------------------------------------------------------- reads
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(self.uptime_seconds, 1),
                "requests_total": sum(self._requests.values()),
                "predictions_total": dict(self._predictions),
                "errors_total": dict(self._errors),
                "feedback_total": self._feedback,
                "mean_latency_ms": round(
                    1000 * self._duration_sum / self._duration_count, 3
                )
                if self._duration_count
                else 0.0,
            }

    def render_prometheus(self) -> str:
        """Prometheus exposition format (text/plain; version=0.0.4)."""
        with self._lock:
            requests = list(self._requests.items())
            predictions = list(self._predictions.items())
            errors = list(self._errors.items())
            duration_sum = self._duration_sum
            duration_count = self._duration_count
            feedback = self._feedback

        lines = [
            "# HELP spam_api_requests_total HTTP requests handled, by route and status.",
            "# TYPE spam_api_requests_total counter",
        ]
        for (method, path, status), count in sorted(requests):
            lines.append(
                "spam_api_requests_total"
                f'{{method="{_escape(method)}",path="{_escape(path)}",status="{status}"}} {count}'
            )

        lines += [
            "# HELP spam_api_request_duration_seconds_sum Total time spent handling requests.",
            "# TYPE spam_api_request_duration_seconds_sum counter",
            f"spam_api_request_duration_seconds_sum {duration_sum:.6f}",
            "# HELP spam_api_request_duration_seconds_count Number of timed requests.",
            "# TYPE spam_api_request_duration_seconds_count counter",
            f"spam_api_request_duration_seconds_count {duration_count}",
            "# HELP spam_api_predictions_total Predicted labels.",
            "# TYPE spam_api_predictions_total counter",
        ]
        for label, count in sorted(predictions):
            lines.append(f'spam_api_predictions_total{{label="{_escape(label)}"}} {count}')

        lines += [
            "# HELP spam_api_errors_total Application errors.",
            "# TYPE spam_api_errors_total counter",
        ]
        for kind, count in sorted(errors):
            lines.append(f'spam_api_errors_total{{kind="{_escape(kind)}"}} {count}')

        lines += [
            "# HELP spam_api_feedback_total Feedback records stored.",
            "# TYPE spam_api_feedback_total counter",
            f"spam_api_feedback_total {feedback}",
            "# HELP spam_api_uptime_seconds Seconds since process start.",
            "# TYPE spam_api_uptime_seconds gauge",
            f"spam_api_uptime_seconds {self.uptime_seconds:.3f}",
        ]
        return "\n".join(lines) + "\n"
