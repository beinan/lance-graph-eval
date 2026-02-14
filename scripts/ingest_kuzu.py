#!/usr/bin/env python3
"""Ingest canonical JSONL dataset into Kuzu (row-wise)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from typing import Dict, Iterable, List


def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_dataset(dataset_path: str) -> Dict[str, List[Dict[str, object]]]:
    def maybe_load(name: str) -> List[Dict[str, object]]:
        path = os.path.join(dataset_path, name)
        if not os.path.exists(path):
            return []
        return list(read_jsonl(path))

    return {
        "documents": maybe_load("documents.jsonl"),
        "chunks": maybe_load("chunks.jsonl"),
        "entities": maybe_load("entities.jsonl"),
        "communities": maybe_load("communities.jsonl"),
        "edges": maybe_load("edges.jsonl"),
    }


def derive_has_chunk_edges(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    edges = []
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        chunk_id = chunk.get("id")
        if doc_id and chunk_id:
            edges.append(
                {
                    "src_id": doc_id,
                    "src_type": "Document",
                    "dst_id": chunk_id,
                    "dst_type": "Chunk",
                    "type": "HAS_CHUNK",
                }
            )
    return edges


def ensure_schema(conn, embedding_dim: int) -> None:
    statements = [
        "CREATE NODE TABLE Document(id STRING, title STRING, text STRING, metadata_json STRING, PRIMARY KEY(id))",
        f"CREATE NODE TABLE Chunk(id STRING, document_id STRING, text STRING, embedding FLOAT[{embedding_dim}], token_count INT64, metadata_json STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE Entity(id STRING, name STRING, type STRING, metadata_json STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE Community(id STRING, level INT64, summary STRING, metadata_json STRING, PRIMARY KEY(id))",
    ]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except Exception:
            pass


def ensure_rel_table(conn, rel_type: str, src_label: str, dst_label: str) -> None:
    stmt = f"CREATE REL TABLE {rel_type} (FROM {src_label} TO {dst_label}, properties_json STRING)"
    try:
        conn.execute(stmt)
    except Exception:
        pass


def sanitize_rel_type(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not cleaned:
        return "REL"
    if cleaned[0].isdigit():
        cleaned = f"REL_{cleaned}"
    return cleaned.upper()


def insert_node(conn, label: str, row: Dict[str, object]) -> None:
    metadata_json = json.dumps(row.get("metadata", {})) if row.get("metadata") is not None else None
    if label == "Document":
        conn.execute(
            """
            MERGE (n:Document {id: $id})
            SET n.title = $title, n.text = $text, n.metadata_json = $metadata_json
            """,
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "text": row.get("text"),
                "metadata_json": metadata_json,
            },
        )
    elif label == "Chunk":
        conn.execute(
            """
            MERGE (n:Chunk {id: $id})
            SET n.document_id = $document_id,
                n.text = $text,
                n.embedding = $embedding,
                n.token_count = $token_count,
                n.metadata_json = $metadata_json
            """,
            {
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "text": row.get("text"),
                "embedding": row.get("embedding"),
                "token_count": row.get("token_count"),
                "metadata_json": metadata_json,
            },
        )
    elif label == "Entity":
        conn.execute(
            """
            MERGE (n:Entity {id: $id})
            SET n.name = $name, n.type = $type, n.metadata_json = $metadata_json
            """,
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "type": row.get("type"),
                "metadata_json": metadata_json,
            },
        )
    elif label == "Community":
        conn.execute(
            """
            MERGE (n:Community {id: $id})
            SET n.level = $level, n.summary = $summary, n.metadata_json = $metadata_json
            """,
            {
                "id": row.get("id"),
                "level": row.get("level"),
                "summary": row.get("summary"),
                "metadata_json": metadata_json,
            },
        )


def insert_edge(conn, edge: Dict[str, object]) -> None:
    rel_type = sanitize_rel_type(str(edge.get("type", "REL")))
    src_label = str(edge.get("src_type", ""))
    dst_label = str(edge.get("dst_type", ""))
    if not src_label or not dst_label:
        raise ValueError("edges.jsonl requires src_type and dst_type")
    props = edge.get("properties", {}) or {}
    props_json = json.dumps(props) if props else None
    ensure_rel_table(conn, rel_type, src_label, dst_label)
    conn.execute(
        f"""
        MATCH (s:{src_label} {{id: $src_id}})
        MATCH (t:{dst_label} {{id: $dst_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.properties_json = $properties_json
        """,
        {
            "src_id": edge.get("src_id"),
            "dst_id": edge.get("dst_id"),
            "properties_json": props_json,
        },
    )


def _copy_table(conn, table: str, path: str) -> None:
    conn.execute(f'COPY {table} FROM "{path}"')


def bulk_import(dataset_path: str, db_path: str, embedding_dim: int) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    try:
        import kuzu
    except ImportError as exc:
        raise RuntimeError("kuzu driver not installed. Run: pip install kuzu") from exc

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    ensure_schema(conn, embedding_dim)

    with tempfile.TemporaryDirectory() as temp_dir:
        nodes_out_dir = os.path.join(temp_dir, "kuzu_nodes")
        edges_out_dir = os.path.join(temp_dir, "kuzu_edges")
        os.makedirs(nodes_out_dir, exist_ok=True)
        os.makedirs(edges_out_dir, exist_ok=True)

        node_specs = {
            "Document": (
                "documents.jsonl",
                pa.schema(
                    [
                        ("id", pa.string()),
                        ("title", pa.string()),
                        ("text", pa.string()),
                        ("metadata_json", pa.string()),
                    ]
                ),
            ),
            "Chunk": (
            "chunks.jsonl",
            pa.schema(
                [
                    ("id", pa.string()),
                    ("document_id", pa.string()),
                    ("text", pa.string()),
                    ("embedding", pa.list_(pa.float32(), embedding_dim)),
                    ("token_count", pa.int64()),
                    ("metadata_json", pa.string()),
                ]
            ),
            ),
            "Entity": (
                "entities.jsonl",
                pa.schema(
                    [
                        ("id", pa.string()),
                        ("name", pa.string()),
                        ("type", pa.string()),
                        ("metadata_json", pa.string()),
                    ]
                ),
            ),
            "Community": (
                "communities.jsonl",
                pa.schema(
                    [
                        ("id", pa.string()),
                        ("level", pa.int64()),
                        ("summary", pa.string()),
                        ("metadata_json", pa.string()),
                    ]
                ),
            ),
        }

        for label, (filename, schema) in node_specs.items():
            in_path = os.path.join(dataset_path, filename)
            if not os.path.exists(in_path):
                continue
            rows = []
            for row in read_jsonl(in_path):
                metadata_json = json.dumps(row.get("metadata", {})) if row.get("metadata") is not None else None
                if label == "Document":
                    rows.append(
                        {
                            "id": row.get("id"),
                            "title": row.get("title"),
                            "text": row.get("text"),
                            "metadata_json": metadata_json,
                        }
                    )
                elif label == "Chunk":
                    rows.append(
                    {
                        "id": row.get("id"),
                        "document_id": row.get("document_id"),
                        "text": row.get("text"),
                        "embedding": [float(v) for v in (row.get("embedding") or [])],
                        "token_count": row.get("token_count"),
                        "metadata_json": metadata_json,
                    }
                    )
                elif label == "Entity":
                    rows.append(
                        {
                            "id": row.get("id"),
                            "name": row.get("name"),
                            "type": row.get("type"),
                            "metadata_json": metadata_json,
                        }
                    )
                elif label == "Community":
                    rows.append(
                        {
                            "id": row.get("id"),
                            "level": row.get("level"),
                            "summary": row.get("summary"),
                            "metadata_json": metadata_json,
                        }
                    )
            if not rows:
                continue
            table = pa.Table.from_pylist(rows, schema=schema)
            out_path = os.path.join(nodes_out_dir, f"{label.lower()}.parquet")
            pq.write_table(table, out_path)
            _copy_table(conn, label, out_path)

        edges_path = os.path.join(dataset_path, "edges.jsonl")
        if not os.path.exists(edges_path):
            return

        edge_groups: Dict[tuple, List[Dict[str, object]]] = {}
        with open(edges_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                edge = json.loads(line)
                rel_type = sanitize_rel_type(str(edge.get("type", "REL")))
                src_label = str(edge.get("src_type", ""))
                dst_label = str(edge.get("dst_type", ""))
                if not src_label or not dst_label:
                    raise ValueError("edges.jsonl requires src_type and dst_type")
                key = (rel_type, src_label, dst_label)
                props = edge.get("properties", {}) or {}
                edge_groups.setdefault(key, []).append(
                    {
                        "src_id": edge.get("src_id"),
                        "dst_id": edge.get("dst_id"),
                        "properties_json": json.dumps(props) if props else None,
                    }
                )

        for (rel_type, src_label, dst_label), rows in edge_groups.items():
            ensure_rel_table(conn, rel_type, src_label, dst_label)
            table = pa.Table.from_pylist(
                rows,
                schema=pa.schema(
                    [
                        ("src_id", pa.string()),
                        ("dst_id", pa.string()),
                        ("properties_json", pa.string()),
                    ]
                ),
            )
            out_path = os.path.join(edges_out_dir, f"{rel_type}_{src_label}_{dst_label}.parquet")
            pq.write_table(table, out_path)
            _copy_table(conn, rel_type, out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to Kuzu database")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--derive-has-chunk", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--bulk", action="store_true", help="Use COPY FROM with parquet inputs.")
    parser.add_argument("--reset", action="store_true", help="Delete existing DB before ingest.")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(args.db):
            if os.path.isdir(args.db):
                shutil.rmtree(args.db)
            else:
                os.remove(args.db)
        wal_path = f"{args.db}.wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)

    if args.bulk:
        bulk_import(args.dataset, args.db, args.embedding_dim)
        return 0

    try:
        import kuzu
    except ImportError as exc:
        raise RuntimeError("kuzu driver not installed. Run: pip install kuzu") from exc

    payload = load_dataset(args.dataset)
    if args.derive_has_chunk:
        payload["edges"].extend(derive_has_chunk_edges(payload["chunks"]))

    db = kuzu.Database(args.db)
    conn = kuzu.Connection(db)

    ensure_schema(conn, args.embedding_dim)

    for row in payload["documents"]:
        insert_node(conn, "Document", row)
    for row in payload["chunks"]:
        insert_node(conn, "Chunk", row)
    for row in payload["entities"]:
        insert_node(conn, "Entity", row)
    for row in payload["communities"]:
        insert_node(conn, "Community", row)
    for edge in payload["edges"]:
        insert_edge(conn, edge)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
