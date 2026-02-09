from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from lgeval.types import QueryResult


class BaseEngine:
    kind = "base"

    def __init__(self, name: str, options: Dict[str, Any]):
        self.name = name
        self.options = options
        self.threadsafe = bool(options.get("threadsafe", True))

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def run_query(self, query_text: str, params: Optional[Dict[str, Any]] = None, fetch: str = "count") -> QueryResult:
        raise NotImplementedError

    def run_setup(self, queries: Iterable[str]) -> None:
        for query in queries:
            if query.strip():
                self.run_query(query, params=None, fetch="none")
