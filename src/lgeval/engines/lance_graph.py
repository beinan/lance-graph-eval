from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, Optional

from lgeval.engines.base import BaseEngine
from lgeval.types import QueryResult


class LanceGraphEngine(BaseEngine):
    kind = "lance_graph"

    def __init__(self, name: str, options: Dict[str, Any]):
        super().__init__(name, options)
        self._mode = options.get("mode", "datasets")
        self._command_template = options.get("command_template")
        self._datasets_dir = options.get("datasets_dir")
        self._tables = options.get("tables", {})
        self._graph_spec = options.get("graph", {})
        self._datasets = None
        self._graph_config = None
        self._knowledge_graph = None
        self._engine = None

    def connect(self) -> None:
        if self._mode == "cli":
            if not self._command_template:
                raise ValueError("lance_graph requires command_template in options")
            if "{query}" not in self._command_template:
                raise ValueError("command_template must include {query} placeholder")
            return None

        if self._mode == "knowledge_graph":
            try:
                from knowledge_graph import KnowledgeGraphConfig, LanceKnowledgeGraph
                from knowledge_graph.graph import LanceGraphStore
            except ImportError as exc:
                raise RuntimeError(
                    "knowledge_graph package not installed. Install with: pip install knowledge-graph"
                ) from exc

            root = self.options.get("root")
            if not root:
                raise ValueError("knowledge_graph mode requires options.root")
            config = KnowledgeGraphConfig.from_root(os.fspath(root))
            store = LanceGraphStore(config)
            self._knowledge_graph = LanceKnowledgeGraph(config, store)
            return None

        if self._mode == "datasets":
            try:
                import pyarrow as pa
                import pyarrow.compute as pc
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

            try:
                from lance_graph import CypherEngine, GraphConfig  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "lance-graph v0.5.2+ with CypherEngine is required. "
                    "Run: pip install lance-graph==0.5.2"
                ) from exc

            self._CypherEngine = CypherEngine
            self._GraphConfig = GraphConfig
            self._pa = pa
            self._pc = pc
            self._pq = pq

            self._graph_config = self._build_graph_config(self._graph_spec)
            self._datasets = self._load_datasets()
            self._engine = self._CypherEngine(self._graph_config, self._datasets)
            return None

        raise ValueError(f"Unsupported lance_graph mode: {self._mode}")

    def run_query(
        self, query_text: str, params: Optional[Dict[str, Any]] = None, fetch: str = "count"
    ) -> QueryResult:
        params = params or {}

        if self._mode == "cli":
            return self._run_cli(query_text, params, fetch)
        if self._mode == "knowledge_graph":
            return self._run_knowledge_graph(query_text, params, fetch)
        if self._mode == "datasets":
            return self._run_datasets(query_text, params, fetch)

        raise ValueError(f"Unsupported lance_graph mode: {self._mode}")

    def _run_cli(self, query_text: str, params: Dict[str, Any], fetch: str) -> QueryResult:
        rendered = _render_query(query_text, params)
        cmd = self._command_template.replace("{query}", shlex.quote(rendered))
        completed = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"lance-graph command failed: {stderr}")

        payload_bytes = len(completed.stdout.encode("utf-8"))
        if fetch == "none":
            return QueryResult(row_count=None, payload_bytes=payload_bytes)

        if fetch == "scalar":
            value = _first_nonempty_line(completed.stdout)
            return QueryResult(row_count=_parse_scalar(value), payload_bytes=payload_bytes)

        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return QueryResult(row_count=len(lines), payload_bytes=payload_bytes)

    def _run_knowledge_graph(self, query_text: str, params: Dict[str, Any], fetch: str) -> QueryResult:
        if self._knowledge_graph is None:
            raise RuntimeError("knowledge_graph not initialized")

        rendered = _render_query(query_text, params)
        table = self._knowledge_graph.query(rendered)
        return _table_to_result(table, fetch)

    def _run_datasets(self, query_text: str, params: Dict[str, Any], fetch: str) -> QueryResult:
        if self._graph_config is None or self._datasets is None or self._engine is None:
            raise RuntimeError("lance_graph datasets not initialized")

        rendered = _render_query(query_text, params)
        table = self._engine.execute(rendered)
        return _table_to_result(table, fetch)

    def _build_graph_config(self, graph_spec: Dict[str, Any]):
        nodes = graph_spec.get("nodes", [])
        rels = graph_spec.get("relationships", [])
        if not nodes:
            raise ValueError("lance_graph datasets mode requires graph.nodes")

        builder = self._GraphConfig.builder()
        for node in nodes:
            label = node["label"]
            id_field = node.get("id_field", "id")
            builder = builder.with_node_label(label, id_field)

        for rel in rels:
            name = rel["name"]
            src_field = rel.get("src_field", "src_id")
            dst_field = rel.get("dst_field", "dst_id")
            builder = builder.with_relationship(name, src_field, dst_field)

        return builder.build()

    def _load_datasets(self) -> Dict[str, Any]:
        datasets: Dict[str, Any] = {}
        tables = dict(self._tables)
        graph_spec = self._graph_spec

        nodes = graph_spec.get("nodes", [])
        rels = graph_spec.get("relationships", [])

        for node in nodes:
            label = node["label"]
            table_key = node.get("table", label)
            datasets[label] = self._read_table(table_key)

        edges_cache = {}
        for rel in rels:
            name = rel["name"]
            from_edges = rel.get("from_edges")
            table_key = rel.get("table")
            if from_edges:
                edges_table = edges_cache.get(from_edges)
                if edges_table is None:
                    edges_table = self._read_table(from_edges)
                    edges_cache[from_edges] = edges_table
                type_field = rel.get("type_field", "type")
                filtered = self._filter_edges(edges_table, type_field, name)
                datasets[name] = filtered
            elif table_key:
                datasets[name] = self._read_table(table_key)
            else:
                datasets[name] = self._read_table(name)

        return datasets

    def _read_table(self, key: str):
        path = self._resolve_table_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Table not found: {path}")
        return self._pq.read_table(path)

    def _resolve_table_path(self, key: str) -> str:
        if key in self._tables:
            value = self._tables[key]
            if os.path.isabs(value):
                return value
            if self._datasets_dir:
                return os.path.join(self._datasets_dir, value)
            return value

        if os.path.isabs(key):
            return key
        if self._datasets_dir:
            return os.path.join(self._datasets_dir, f"{key}.parquet")
        return f"{key}.parquet"

    def _filter_edges(self, table, type_field: str, rel_name: str):
        if type_field not in table.schema.names:
            raise ValueError(f"edges table missing type field: {type_field}")
        mask = self._pc.equal(table[type_field], self._pa.scalar(rel_name))
        return table.filter(mask)


def _render_query(query_text: str, params: Dict[str, Any]) -> str:
    rendered = query_text
    for key, value in params.items():
        rendered = rendered.replace(f"${key}", _format_value(value))
    return rendered


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _parse_scalar(value: str):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _table_to_result(table, fetch: str) -> QueryResult:
    if fetch == "none":
        return QueryResult(row_count=None)
    if fetch == "scalar":
        if table.num_rows == 0 or table.num_columns == 0:
            return QueryResult(row_count=0)
        value = table.column(0)[0].as_py()
        return QueryResult(row_count=value)
    return QueryResult(row_count=table.num_rows)


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    text = str(value)
    return "'" + text.replace("'", "''") + "'"
