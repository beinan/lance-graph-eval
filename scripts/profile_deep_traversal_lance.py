#!/usr/bin/env python3
"""Profile deep_traversal_5hop steps for lance-graph."""

from __future__ import annotations

import argparse
import time
from statistics import mean

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphrag_eval.yaml")
    parser.add_argument("--entity-name", default="About")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    lance_cfg = next((e for e in config.engines if e.kind == "lance_graph"), None)
    if lance_cfg is None:
        raise SystemExit("No lance_graph engine found in config.")

    engine = LanceGraphEngine("profile_lance", lance_cfg.options)
    engine.connect()

    entity = args.entity_name.replace("'", "''")

    queries = [
        (
            "mentions_only",
            f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: '{entity}'}}) RETURN count(*) AS n",
        ),
        (
            "mentions_two_chunks",
            f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: '{entity}'}})"
            f"<-[:MENTIONS]-(c2:Chunk) RETURN count(*) AS n",
        ),
        (
            "with_parent_doc",
            f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: '{entity}'}})"
            f"<-[:MENTIONS]-(c2:Chunk) "
            f"MATCH (c2)-[:PARENT_OF]->(d:Document) RETURN count(*) AS n",
        ),
        (
            "with_has_chunk",
            f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: '{entity}'}})"
            f"<-[:MENTIONS]-(c2:Chunk) "
            f"MATCH (c2)-[:PARENT_OF]->(d:Document)-[:HAS_CHUNK]->(c3:Chunk) "
            f"RETURN count(*) AS n",
        ),
        (
            "full_traversal",
            f"MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {{name: '{entity}'}})"
            f"<-[:MENTIONS]-(c2:Chunk) "
            f"MATCH (c2)-[:PARENT_OF]->(d:Document)-[:HAS_CHUNK]->(c3:Chunk)"
            f"-[:MENTIONS]->(e2:Entity) "
            f"WHERE c.id <> c2.id AND "
            f"(e2.id <> e.id OR (c3.id <> c.id AND c3.id <> c2.id)) "
            f"RETURN count(*) AS n",
        ),
    ]

    def run_query(q: str) -> None:
        engine._engine.execute(q)

    print("lance_graph deep_traversal_5hop profile (mean ms)")
    for label, query in queries:
        ms = time_fn(lambda: run_query(query), args.warmups, args.runs)
        print(f"  {label:18} {ms:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
