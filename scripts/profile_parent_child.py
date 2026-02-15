#!/usr/bin/env python3
"""Profile parent_child_enrichment steps for lance-graph."""

from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean
from typing import List

from lgeval.config import load_config
from lgeval.engines.lance_graph import LanceGraphEngine


def time_fn(fn, warmups: int, runs: int) -> float:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return mean(samples)


def _ids_literal(ids: List[str]) -> str:
    return "[" + ", ".join("'" + str(i).replace("'", "''") + "'" for i in ids) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphrag_eval.yaml")
    parser.add_argument("--embedding", default=os.getenv("EMBEDDING_JSON", "datasets/embedding.json"))
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    lance_cfg = next((e for e in config.engines if e.kind == "lance_graph"), None)
    if lance_cfg is None:
        raise SystemExit("No lance_graph engine found in config.")

    embedding = json.load(open(args.embedding, "r", encoding="utf-8"))

    engine = LanceGraphEngine("profile_lance", lance_cfg.options)
    engine.connect()

    try:
        from lance_graph import CypherQuery, VectorSearch, DistanceMetric  # type: ignore
    except ImportError as exc:
        raise SystemExit("lance-graph v0.5.2+ required.") from exc

    datasets = engine._datasets
    graph_config = engine._graph_config
    if datasets is None or graph_config is None:
        raise SystemExit("lance_graph datasets not initialized")

    query_vec = [float(v) for v in embedding]

    def vector_only():
        query = CypherQuery("MATCH (c:Chunk) RETURN c.id, c.embedding").with_config(graph_config)
        vector_search = (
            VectorSearch("c.embedding")
            .query_vector(query_vec)
            .metric(DistanceMetric.Cosine)
            .top_k(args.top_k)
        )
        query.execute_with_vector_rerank(datasets, vector_search)

    # Capture top-k ids for join-only timing
    vec_query = CypherQuery("MATCH (c:Chunk) RETURN c.id, c.embedding").with_config(graph_config)
    vec_search = (
        VectorSearch("c.embedding")
        .query_vector(query_vec)
        .metric(DistanceMetric.Cosine)
        .top_k(args.top_k)
    )
    vec_table = vec_query.execute_with_vector_rerank(datasets, vec_search)
    ids = [row["c.id"] for row in vec_table.to_pylist()]

    def join_only():
        ids_literal = _ids_literal(ids)
        query = f"MATCH (c:Chunk)-[:PARENT_OF]->(p:Document) WHERE c.id IN {ids_literal} RETURN count(*) AS n"
        try:
            engine._engine.execute(query)
        except Exception:
            # Fallback: filter relationship table directly
            try:
                import pyarrow.compute as pc
            except ImportError as exc:
                raise SystemExit("pyarrow required for fallback join timing.") from exc
            rel = datasets.get("PARENT_OF")
            if rel is None:
                raise SystemExit("PARENT_OF relationship table missing in datasets.")
            mask = pc.is_in(rel["src_id"], value_set=ids)
            _ = rel.filter(mask).num_rows

    def combined():
        query = CypherQuery(
            "MATCH (c:Chunk)-[:PARENT_OF]->(p:Document) RETURN c.id, c.embedding, p.id"
        ).with_config(graph_config)
        vector_search = (
            VectorSearch("c.embedding")
            .query_vector(query_vec)
            .metric(DistanceMetric.Cosine)
            .top_k(args.top_k)
        )
        query.execute_with_vector_rerank(datasets, vector_search)

    vec_ms = time_fn(vector_only, args.warmups, args.runs)
    join_ms = time_fn(join_only, args.warmups, args.runs)
    comb_ms = time_fn(combined, args.warmups, args.runs)

    print("lance_graph parent_child_enrichment profile (mean ms)")
    print(f"  vector_only: {vec_ms:.3f}")
    print(f"  join_only:   {join_ms:.3f}")
    print(f"  combined:    {comb_ms:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
