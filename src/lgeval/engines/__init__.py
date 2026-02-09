from __future__ import annotations

from typing import Any, Dict

from lgeval.engines.base import BaseEngine
from lgeval.engines.kuzu import KuzuEngine
from lgeval.engines.lance_graph import LanceGraphEngine
from lgeval.engines.neo4j import Neo4jEngine


def get_engine(kind: str, name: str, options: Dict[str, Any]) -> BaseEngine:
    kind_lower = kind.lower()
    if kind_lower == "neo4j":
        return Neo4jEngine(name, options)
    if kind_lower == "kuzu":
        return KuzuEngine(name, options)
    if kind_lower in {"lance_graph", "lance-graph", "lancegraph"}:
        return LanceGraphEngine(name, options)
    raise ValueError(f"Unsupported engine kind: {kind}")
