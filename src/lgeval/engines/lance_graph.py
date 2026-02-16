from __future__ import annotations

import os
import re
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
        self._label_to_id_field: Dict[str, str] = {}
        self._label_to_table_key: Dict[str, str] = {}
        self._lance_datasets: Dict[str, Any] = {}

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

        if self._mode in ("datasets", "lance"):
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
                    "lance-graph v0.5.3+ with CypherEngine is required. "
                    "Run: pip install lance-graph==0.5.3"
                ) from exc

            self._CypherEngine = CypherEngine
            self._GraphConfig = GraphConfig
            self._pa = pa
            self._pc = pc
            self._pq = pq
            self._lance = None
            if self._mode == "lance":
                try:
                    import lance
                except ImportError as exc:
                    raise RuntimeError("lance not installed. Run: pip install lance") from exc
                self._lance = lance

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
        if self._mode in ("datasets", "lance"):
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
        if params.get("vector_rerank"):
            table = self._execute_with_vector_rerank(rendered, params)
            if fetch == "scalar":
                return QueryResult(row_count=table.num_rows)
            return _table_to_result(table, fetch)

        table = self._engine.execute(rendered)
        if fetch == "scalar" and table.num_columns == 1 and table.column(0).type == self._pa.null():
            raise RuntimeError("query returned NULL; check vector rerank or embedding setup")
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
            self._label_to_id_field[label] = id_field
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
            self._label_to_table_key[label] = table_key
            if self._mode == "lance":
                path = self._resolve_table_path(table_key)
                if self._lance is None:
                    raise RuntimeError("lance not initialized")
                dataset = self._lance.dataset(path)
                self._lance_datasets[label] = dataset
                table = dataset.to_table()
            else:
                table = self._read_table(table_key)
            datasets[label] = self._maybe_cast_embeddings(table)

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

        self._maybe_attach_document_embeddings(datasets)
        return datasets

    def _read_table(self, key: str):
        path = self._resolve_table_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Table not found: {path}")
        if self._mode == "lance":
            if self._lance is None:
                raise RuntimeError("lance not initialized")
            dataset = self._lance.dataset(path)
            table = dataset.to_table()
        else:
            table = self._pq.read_table(path)
        return self._maybe_cast_embeddings(table)

    def _maybe_cast_embeddings(self, table):
        if "embedding" not in table.schema.names:
            return table
        return self._ensure_float32_embeddings(table, "embedding")

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
            suffix = "lance" if self._mode == "lance" else "parquet"
            return os.path.join(self._datasets_dir, f"{key}.{suffix}")
        return f"{key}.lance" if self._mode == "lance" else f"{key}.parquet"

    def _filter_edges(self, table, type_field: str, rel_name: str):
        if type_field not in table.schema.names:
            raise ValueError(f"edges table missing type field: {type_field}")
        mask = self._pc.equal(table[type_field], self._pa.scalar(rel_name))
        return table.filter(mask)

    def _maybe_attach_document_embeddings(self, datasets: Dict[str, Any]) -> None:
        if "Document" not in datasets or "Chunk" not in datasets:
            return
        doc_table = datasets["Document"]
        if "embedding" in doc_table.schema.names:
            datasets["Document"] = self._ensure_float32_embeddings(doc_table, "embedding")
            return
        chunk_table = datasets["Chunk"]
        if "embedding" not in chunk_table.schema.names or "document_id" not in chunk_table.schema.names:
            return
        target_type = self._embedding_target_type(chunk_table.schema.field("embedding").type)

        doc_ids = doc_table.column("id").to_pylist()
        chunk_doc_ids = chunk_table.column("document_id").to_pylist()
        chunk_embeddings = chunk_table.column("embedding").to_pylist()

        sums: Dict[str, list] = {}
        counts: Dict[str, int] = {}
        for doc_id, emb in zip(chunk_doc_ids, chunk_embeddings):
            if doc_id is None or emb is None:
                continue
            vec = [float(v) for v in emb]
            if doc_id not in sums:
                sums[doc_id] = [0.0] * len(vec)
                counts[doc_id] = 0
            acc = sums[doc_id]
            for idx, value in enumerate(vec):
                acc[idx] += value
            counts[doc_id] += 1

        embeddings = []
        for doc_id in doc_ids:
            total = sums.get(doc_id)
            count = counts.get(doc_id, 0)
            if total is None or count == 0:
                embeddings.append(None)
            else:
                embeddings.append([value / count for value in total])

        emb_array = self._pa.array(embeddings, type=target_type)
        datasets["Document"] = doc_table.append_column("embedding", emb_array)

    def _embedding_target_type(self, field_type):
        if getattr(self._pa.types, "is_fixed_shape_tensor", None) and self._pa.types.is_fixed_shape_tensor(
            field_type
        ):
            if field_type.value_type == self._pa.float32():
                return field_type
            return field_type
        if self._pa.types.is_fixed_size_list(field_type):
            list_size = field_type.list_size
            if field_type.value_type == self._pa.float32():
                return field_type
            return self._pa.list_(self._pa.float32(), list_size)
        if self._pa.types.is_list(field_type) or self._pa.types.is_large_list(field_type):
            if field_type.value_type == self._pa.float32():
                return field_type
            return self._pa.list_(self._pa.float32())
        return self._pa.list_(self._pa.float32())

    def _ensure_float32_embeddings(self, table, column: str):
        field = table.schema.field(column)
        target_type = self._embedding_target_type(field.type)
        if field.type == target_type:
            return table
        try:
            casted = self._pc.cast(table[column], target_type)
        except Exception:
            values = []
            for row in table[column].to_pylist():
                if row is None:
                    values.append(None)
                else:
                    values.append([float(v) for v in row])
            casted = self._pa.array(values, type=target_type)
        return table.set_column(table.schema.get_field_index(column), column, casted)

    def _execute_with_vector_rerank(self, rendered: str, params: Dict[str, Any]):
        if self._graph_config is None or self._datasets is None:
            raise RuntimeError("lance_graph datasets not initialized")

        try:
            from lance_graph import CypherQuery, VectorSearch, DistanceMetric  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "lance-graph v0.5.3+ with CypherQuery is required. "
                "Run: pip install lance-graph==0.5.3"
            ) from exc

        query_vec = params.get("embedding")
        top_k = params.get("top_k")
        if query_vec is None or top_k is None:
            raise ValueError("vector_rerank requires embedding and top_k params")
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be > 0 for vector_rerank")

        metric_name = str(params.get("metric") or params.get("distance_metric") or "l2").lower()
        if metric_name in ("cosine", "cos"):
            metric = DistanceMetric.Cosine
        elif metric_name == "dot":
            metric = DistanceMetric.Dot
        else:
            metric = DistanceMetric.L2

        vector_column = params.get("vector_column") or "d.embedding"
        if vector_column.endswith(".embedding"):
            if "Document" in self._datasets and "embedding" not in self._datasets["Document"].schema.names:
                raise RuntimeError(
                    "vector_search requires Document.embedding; ensure embeddings exist "
                    "or use a different vector_column."
                )
        else:
            label = vector_column.split(".", 1)[0]
            if label and label in self._datasets:
                if "embedding" not in self._datasets[label].schema.names:
                    raise RuntimeError(
                        f"vector_search requires {label}.embedding; ensure embeddings exist "
                        "or use a different vector_column."
                    )
        if self._mode == "lance" and self.options.get("use_lance_index", True):
            return self._execute_with_lance_index(rendered, params, metric_name)
        vector_search = (
            VectorSearch(vector_column)
            .query_vector([float(v) for v in query_vec])
            .metric(metric)
            .top_k(top_k)
        )
        engine = self._engine
        if engine is not None and hasattr(engine, "execute_with_vector_rerank"):
            return engine.execute_with_vector_rerank(rendered, vector_search)

        query = CypherQuery(rendered).with_config(self._graph_config)
        return query.execute_with_vector_rerank(self._datasets, vector_search)

    def _execute_with_lance_index(self, rendered: str, params: Dict[str, Any], metric_name: str):
        if self._lance is None:
            raise RuntimeError("lance not initialized")
        vector_column = params.get("vector_column") or "d.embedding"
        if "." in vector_column:
            alias, column = vector_column.split(".", 1)
        else:
            alias, column = vector_column, vector_column

        alias_map = self._alias_to_label(rendered)
        label = alias_map.get(alias)
        if label is None and alias in self._label_to_id_field:
            label = alias
        if label is None:
            raise RuntimeError(f"Unable to resolve label for vector column alias '{alias}'")

        id_field = self._label_to_id_field.get(label, "id")
        if label not in self._label_to_table_key:
            raise RuntimeError(f"No table mapping for label {label}")
        dataset = self._lance_datasets.get(label)
        if dataset is None:
            path = self._resolve_table_path(self._label_to_table_key[label])
            dataset = self._lance.dataset(path)

        query_vec = [float(v) for v in params.get("embedding", [])]
        top_k = int(params.get("top_k", 0))
        if not query_vec or top_k <= 0:
            raise ValueError("vector_rerank requires embedding and top_k params")

        nearest = {
            "column": column,
            "q": query_vec,
            "k": top_k,
            "metric": metric_name,
            "use_index": True,
        }
        ids_table = dataset.to_table(
            columns=[id_field],
            nearest=nearest,
            disable_scoring_autoprojection=True,
        )
        ids = ids_table.column(id_field).to_pylist()

        table = self._datasets[label]
        mask = self._pc.is_in(table[id_field], value_set=self._pa.array(ids))
        filtered = table.filter(mask)
        filtered_datasets = dict(self._datasets)
        filtered_datasets[label] = filtered

        from lance_graph import CypherQuery  # type: ignore

        cypher_query = CypherQuery(rendered).with_config(self._graph_config)
        return cypher_query.execute(filtered_datasets)

    @staticmethod
    def _alias_to_label(query_text: str) -> Dict[str, str]:
        pattern = re.compile(r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
        return {match.group(1): match.group(2) for match in pattern.finditer(query_text)}


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
