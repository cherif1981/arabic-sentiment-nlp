"""
نظام إحصائيات الـ API
"""
import time
from threading import Lock
from collections import defaultdict


class MetricsCollector:
    """يجمع إحصائيات الاستخدام."""

    def __init__(self):
        self._lock = Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
        self.requests_per_model = defaultdict(int)
        self.latencies = []
        self.max_latency_samples = 1000

    def record_request(self):
        """تسجيل طلب."""
        with self._lock:
            self.total_requests += 1

    def record_prediction(self, model_name: str, latency_ms: float):
        """تسجيل تنبؤ."""
        with self._lock:
            self.total_predictions += 1
            self.requests_per_model[model_name] += 1
            self.latencies.append(latency_ms)

            # نحتفظ بآخر 1000 قياس فقط
            if len(self.latencies) > self.max_latency_samples:
                self.latencies = self.latencies[-self.max_latency_samples:]

    def get_stats(self) -> dict:
        """الإحصائيات الحالية."""
        with self._lock:
            avg_latency = (
                sum(self.latencies) / len(self.latencies)
                if self.latencies else 0
            )
            return {
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "requests_per_model": dict(self.requests_per_model),
                "average_latency_ms": round(avg_latency, 2),
            }


# Singleton
_metrics = None


def get_metrics() -> MetricsCollector:
    """الحصول على جامع الإحصائيات."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics