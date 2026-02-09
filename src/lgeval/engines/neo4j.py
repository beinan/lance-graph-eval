from __future__ import annotations

from typing import Any, Dict, Optional

from lgeval.engines.base import BaseEngine
from lgeval.types import QueryResult


class Neo4jEngine(BaseEngine):
    kind = "neo4j"

    def __init__(self, name: str, options: Dict[str, Any]):
        super().__init__(name, options)
        self._driver = None

    def connect(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j driver not installed. Run: pip install neo4j") from exc

        uri = self.options.get("uri")
        user = self.options.get("user")
        password = self.options.get("password")
        if not uri or not user or password is None:
            raise ValueError("neo4j requires uri, user, and password in options")

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def run_query(
        self, query_text: str, params: Optional[Dict[str, Any]] = None, fetch: str = "count"
    ) -> QueryResult:
        if self._driver is None:
            raise RuntimeError("neo4j driver not connected")

        database = self.options.get("database")
        params = params or {}
        with self._driver.session(database=database) as session:
            result = session.run(query_text, params)
            if fetch == "none":
                result.consume()
                return QueryResult(row_count=None)
            if fetch == "scalar":
                record = result.single()
                if record is None:
                    return QueryResult(row_count=0)
                value = record[0]
                if isinstance(value, (int, float)):
                    return QueryResult(row_count=value)
                return QueryResult(row_count=value)
            if fetch == "all":
                rows = result.data()
                return QueryResult(row_count=len(rows))
            count = 0
            for _ in result:
                count += 1
            return QueryResult(row_count=count)
