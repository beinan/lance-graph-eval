from __future__ import annotations

from statistics import mean, stdev
from typing import Dict, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "stdev_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    values = sorted(latencies_ms)
    result = {
        "count": len(values),
        "mean_ms": mean(values),
        "stdev_ms": stdev(values) if len(values) > 1 else 0.0,
        "min_ms": values[0],
        "max_ms": values[-1],
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }
    return result
