from __future__ import annotations

from typing import Any, Dict, Optional

from lgeval.engines.base import BaseEngine
from lgeval.types import QueryResult


class KuzuEngine(BaseEngine):
    kind = "kuzu"

    def __init__(self, name: str, options: Dict[str, Any]):
        super().__init__(name, options)
        self._db = None
        self._conn = None

    def connect(self) -> None:
        try:
            import kuzu
        except ImportError as exc:
            raise RuntimeError("kuzu driver not installed. Run: pip install kuzu") from exc

        path = self.options.get("path")
        if not path:
            raise ValueError("kuzu requires path in options")

        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)

    def close(self) -> None:
        self._conn = None
        self._db = None

    def run_query(
        self, query_text: str, params: Optional[Dict[str, Any]] = None, fetch: str = "count"
    ) -> QueryResult:
        if self._conn is None:
            raise RuntimeError("kuzu connection not initialized")

        params = params or {}
        result = self._conn.execute(query_text, params)
        if fetch == "none":
            return QueryResult(row_count=None)
        if fetch == "scalar":
            if result.has_next():
                row = result.get_next()
                if isinstance(row, dict):
                    value = next(iter(row.values()))
                elif isinstance(row, (list, tuple)):
                    value = row[0]
                else:
                    value = row
                return QueryResult(row_count=value)
            return QueryResult(row_count=0)
        if fetch == "all":
            rows = []
            while result.has_next():
                rows.append(result.get_next())
            return QueryResult(row_count=len(rows))

        count = 0
        while result.has_next():
            result.get_next()
            count += 1
        return QueryResult(row_count=count)
