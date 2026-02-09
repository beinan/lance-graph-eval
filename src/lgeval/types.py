from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryResult:
    row_count: Optional[float] = None
    payload_bytes: Optional[int] = None


@dataclass
class QuerySample:
    duration_ms: float
    ok: bool
    error: Optional[str] = None
    row_count: Optional[float] = None
    rss_mb: Optional[float] = None
    expect_ok: Optional[bool] = None
    expect_error: Optional[str] = None
