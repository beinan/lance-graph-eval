#!/usr/bin/env python3
"""Prepare Lance datasets for lance-graph datasets mode."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_lance_from_table(table, out_path: str) -> None:
    import lance

    lance.write_dataset(table, out_path, mode="overwrite")


def maybe_write_lance(name: str, dataset_path: str, out_dir: str) -> None:
    in_path = os.path.join(dataset_path, name)
    if not os.path.exists(in_path):
        return

    rows = read_jsonl(in_path)
    if not rows:
        return

    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    table = pa.Table.from_pylist(rows)
    out_path = os.path.join(out_dir, name.replace(".jsonl", ".lance"))
    write_lance_from_table(table, out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to canonical JSONL dataset")
    parser.add_argument("--out", default=None, help="Output directory for Lance datasets")
    parser.add_argument(
        "--from-parquet",
        action="store_true",
        help="Read parquet inputs from dataset/parquet instead of JSONL",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    out_dir = args.out or os.path.join(dataset_path, "lance")
    os.makedirs(out_dir, exist_ok=True)

    if args.from_parquet:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

        parquet_dir = os.path.join(dataset_path, "parquet")
        if not os.path.isdir(parquet_dir):
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

        for name in [
            "documents.parquet",
            "chunks.parquet",
            "entities.parquet",
            "communities.parquet",
            "edges.parquet",
        ]:
            in_path = os.path.join(parquet_dir, name)
            if not os.path.exists(in_path):
                continue
            table = pq.read_table(in_path)
            out_path = os.path.join(out_dir, name.replace(".parquet", ".lance"))
            write_lance_from_table(table, out_path)
    else:
        for name in [
            "documents.jsonl",
            "chunks.jsonl",
            "entities.jsonl",
            "communities.jsonl",
            "edges.jsonl",
        ]:
            maybe_write_lance(name, dataset_path, out_dir)

    print(f"Wrote Lance datasets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
