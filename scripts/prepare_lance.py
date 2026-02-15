#!/usr/bin/env python3
"""Prepare Lance datasets for lance-graph datasets mode."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_fixed_list_size(array) -> Optional[int]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    if pa.types.is_fixed_size_list(array.type):
        return int(array.type.list_size)

    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return None

    lengths = pc.list_value_length(array)
    max_len = pc.max(lengths).as_py()
    min_len = pc.min(lengths).as_py()
    if max_len is None or min_len is None:
        return None
    if max_len != min_len:
        return None
    return int(max_len)


def _coerce_embedding_column(table, column: str = "embedding") -> Tuple[object, Optional[int]]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    if column not in table.schema.names:
        return table, None

    field = table.schema.field(column)
    list_size = _infer_fixed_list_size(table[column])
    if list_size is not None:
        target_type = pa.list_(pa.float32(), list_size)
    else:
        target_type = pa.list_(pa.float32())

    if field.type == target_type:
        return table, list_size

    try:
        casted = pc.cast(table[column], target_type)
    except Exception:
        values = []
        for row in table[column].to_pylist():
            if row is None:
                values.append(None)
            else:
                values.append([float(v) for v in row])
        casted = pa.array(values, type=target_type)

    idx = table.schema.get_field_index(column)
    return table.set_column(idx, column, casted), list_size


def _vector_index_params(num_rows: int, dim: int) -> Tuple[int, int]:
    # Prefer fewer partitions to avoid empty clusters for small/duplicate-heavy datasets.
    num_partitions = max(1, min(64, int(num_rows / 200)))
    num_sub_vectors = max(1, min(32, dim // 4))
    while num_sub_vectors > 1 and dim % num_sub_vectors != 0:
        num_sub_vectors -= 1
    return num_partitions, num_sub_vectors


def write_lance_from_table(
    table,
    out_path: str,
    index_column: Optional[str] = None,
    index_metric: str = "cosine",
    index_dim: Optional[int] = None,
) -> None:
    import lance

    lance.write_dataset(table, out_path, mode="overwrite")
    if not index_column:
        return
    try:
        dataset = lance.dataset(out_path)
        num_rows = table.num_rows
        dim = index_dim
        if dim is None and index_column in table.schema.names:
            dim = _infer_fixed_list_size(table[index_column])
        if dim is None:
            print(f"Skipping vector index for {out_path} (unknown embedding dimension).")
            return
        num_partitions, num_sub_vectors = _vector_index_params(num_rows, dim)
        dataset.create_index(
            index_column,
            index_type="IVF_PQ",
            metric=index_metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            replace=True,
        )
        print(
            f"Created Lance vector index on {out_path}:{index_column} "
            f"(dim={dim}, partitions={num_partitions}, sub_vectors={num_sub_vectors})."
        )
    except Exception as exc:
        print(f"Failed to create vector index for {out_path}: {exc}")


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
    table, emb_dim = _coerce_embedding_column(table)
    out_path = os.path.join(out_dir, name.replace(".jsonl", ".lance"))
    index_column = "embedding" if name.startswith("chunks") else None
    index_dim = emb_dim if index_column else None
    write_lance_from_table(
        table, out_path, index_column=index_column, index_metric="cosine", index_dim=index_dim
    )


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
            table, emb_dim = _coerce_embedding_column(table)
            out_path = os.path.join(out_dir, name.replace(".parquet", ".lance"))
            index_column = "embedding" if name.startswith("chunks") else None
            index_dim = emb_dim if index_column else None
            write_lance_from_table(
                table, out_path, index_column=index_column, index_metric="cosine", index_dim=index_dim
            )
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
