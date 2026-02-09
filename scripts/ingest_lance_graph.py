#!/usr/bin/env python3
"""Prepare and validate Parquet datasets for lance-graph datasets mode."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List

import yaml


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def maybe_write_parquet(name: str, dataset_path: str, out_dir: str) -> None:
    in_path = os.path.join(dataset_path, name)
    if not os.path.exists(in_path):
        return

    rows = read_jsonl(in_path)
    if not rows:
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    table = pa.Table.from_pylist(rows)
    out_path = os.path.join(out_dir, name.replace(".jsonl", ".parquet"))
    pq.write_table(table, out_path)


def default_graph_spec() -> Dict[str, object]:
    return {
        "nodes": [
            {"label": "Document", "id_field": "id", "table": "documents"},
            {"label": "Chunk", "id_field": "id", "table": "chunks"},
            {"label": "Entity", "id_field": "id", "table": "entities"},
            {"label": "Community", "id_field": "id", "table": "communities"},
        ],
        "relationships": [
            {"name": "HAS_CHUNK", "src_field": "src_id", "dst_field": "dst_id", "from_edges": "edges"},
            {"name": "MENTIONS", "src_field": "src_id", "dst_field": "dst_id", "from_edges": "edges"},
            {"name": "PARENT_OF", "src_field": "src_id", "dst_field": "dst_id", "from_edges": "edges"},
        ],
    }


def load_graph_spec(path: str | None) -> Dict[str, object]:
    if not path:
        return default_graph_spec()
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_tables(datasets_dir: str, graph_spec: Dict[str, object]) -> Dict[str, object]:
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    datasets: Dict[str, object] = {}

    def read_table(name: str):
        path = os.path.join(datasets_dir, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet table not found: {path}")
        return pq.read_table(path)

    nodes = graph_spec.get("nodes", [])
    rels = graph_spec.get("relationships", [])

    for node in nodes:
        label = node["label"]
        table_name = node.get("table", label)
        datasets[label] = read_table(table_name)

    edges_cache: Dict[str, object] = {}
    for rel in rels:
        name = rel["name"]
        from_edges = rel.get("from_edges")
        table_name = rel.get("table", name)
        if from_edges:
            edges = edges_cache.get(from_edges)
            if edges is None:
                edges = read_table(from_edges)
                edges_cache[from_edges] = edges
            type_field = rel.get("type_field", "type")
            if type_field not in edges.schema.names:
                raise ValueError(f"edges table missing type field: {type_field}")
            mask = pc.equal(edges[type_field], pa.scalar(name))
            datasets[name] = edges.filter(mask)
        else:
            datasets[name] = read_table(table_name)

    return datasets


def validate_dataset(datasets_dir: str, graph_spec: Dict[str, object]) -> None:
    try:
        from lance_graph import CypherQuery, GraphConfig
    except ImportError as exc:
        raise RuntimeError("lance-graph not installed. Run: pip install lance-graph") from exc

    builder = GraphConfig.builder()
    for node in graph_spec.get("nodes", []):
        builder = builder.with_node_label(node["label"], node.get("id_field", "id"))
    for rel in graph_spec.get("relationships", []):
        builder = builder.with_relationship(
            rel["name"], rel.get("src_field", "src_id"), rel.get("dst_field", "dst_id")
        )
    config = builder.build()

    datasets = load_tables(datasets_dir, graph_spec)

    for node in graph_spec.get("nodes", []):
        label = node["label"]
        query = f"MATCH (n:{label}) RETURN count(n) AS n"
        table = CypherQuery(query).with_config(config).execute(datasets)
        count = table.column(0)[0].as_py() if table.num_rows else 0
        print(f"{label}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to canonical JSONL dataset")
    parser.add_argument("--out", default=None, help="Output directory for Parquet files")
    parser.add_argument("--graph-spec", default=None, help="YAML graph spec (nodes/relationships)")
    parser.add_argument("--validate", action="store_true", help="Run a small Cypher validation")
    args = parser.parse_args()

    dataset_path = args.dataset
    out_dir = args.out or os.path.join(dataset_path, "parquet")
    os.makedirs(out_dir, exist_ok=True)

    for name in [
        "documents.jsonl",
        "chunks.jsonl",
        "entities.jsonl",
        "communities.jsonl",
        "edges.jsonl",
    ]:
        maybe_write_parquet(name, dataset_path, out_dir)

    print(f"Wrote parquet files to {out_dir}")

    if args.validate:
        graph_spec = load_graph_spec(args.graph_spec)
        validate_dataset(out_dir, graph_spec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
