#!/usr/bin/env python3
"""Ingest canonical JSONL dataset into Neo4j."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batch_iter(items: List[Dict[str, object]], batch_size: int) -> Iterable[List[Dict[str, object]]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def sanitize_rel_type(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    if not cleaned:
        return "REL"
    if cleaned[0].isdigit():
        cleaned = f"REL_{cleaned}"
    return cleaned.upper()


def ensure_constraints(session, labels: List[str]) -> None:
    for label in labels:
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")


def create_indexes(session, create_vector: bool, create_fulltext: bool, embedding_dim: int) -> None:
    if create_vector:
        session.run(
            """
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
            """,
            {"dim": embedding_dim},
        )
    if create_fulltext:
        session.run(
            """
            CREATE FULLTEXT INDEX entity_name IF NOT EXISTS
            FOR (e:Entity) ON EACH [e.name]
            """
        )


def upsert_nodes(session, label: str, rows: List[Dict[str, object]]) -> None:
    for row in rows:
        if isinstance(row.get("metadata"), dict):
            row["metadata_json"] = json.dumps(row["metadata"])
            del row["metadata"]
    query = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id}})
    SET n += row
    """
    session.run(query, {"rows": rows})


def upsert_relationships(
    session,
    rel_type: str,
    src_label: str,
    dst_label: str,
    rows: List[Dict[str, object]],
) -> None:
    query = f"""
    UNWIND $rows AS row
    MATCH (s:{src_label} {{id: row.src_id}})
    MATCH (t:{dst_label} {{id: row.dst_id}})
    MERGE (s)-[r:{rel_type}]->(t)
    SET r += row.props
    """
    session.run(query, {"rows": rows})


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--derive-has-chunk", action="store_true")
    parser.add_argument("--create-vector-index", action="store_true")
    parser.add_argument("--create-fulltext-index", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("neo4j driver not installed. Run: pip install neo4j") from exc

    payload = load_dataset(args.dataset)
    if args.derive_has_chunk:
        payload["edges"].extend(derive_has_chunk_edges(payload["chunks"]))

    labels = ["Document", "Chunk", "Entity", "Community"]

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session(database=args.database) as session:
        if args.reset:
            session.run("MATCH (n) DETACH DELETE n")
        ensure_constraints(session, labels)
        create_indexes(session, args.create_vector_index, args.create_fulltext_index, args.embedding_dim)

        for label, key in [
            ("Document", "documents"),
            ("Chunk", "chunks"),
            ("Entity", "entities"),
            ("Community", "communities"),
        ]:
            rows = payload[key]
            if not rows:
                continue
            for batch in batch_iter(rows, args.batch_size):
                upsert_nodes(session, label, batch)

        edge_groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
        for edge in payload["edges"]:
            rel_type = sanitize_rel_type(str(edge.get("type", "REL")))
            src_label = str(edge.get("src_type", ""))
            dst_label = str(edge.get("dst_type", ""))
            if not src_label or not dst_label:
                raise ValueError("edges.jsonl requires src_type and dst_type")
            key = (rel_type, src_label, dst_label)
            edge_groups.setdefault(key, []).append(
                {
                    "src_id": edge.get("src_id"),
                    "dst_id": edge.get("dst_id"),
                    "props": edge.get("properties", {}) or {},
                }
            )

        for (rel_type, src_label, dst_label), rows in edge_groups.items():
            for batch in batch_iter(rows, args.batch_size):
                upsert_relationships(session, rel_type, src_label, dst_label, batch)

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
