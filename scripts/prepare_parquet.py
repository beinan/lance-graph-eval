#!/usr/bin/env python3
"""Convert canonical JSONL dataset into Parquet files."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def maybe_convert(name: str, dataset_path: str, out_dir: str) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
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
        maybe_convert(name, dataset_path, out_dir)

    print(f"Wrote parquet files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
