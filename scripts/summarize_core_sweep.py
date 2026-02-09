#!/usr/bin/env python3
"""Summarize core sweep results into a single CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional


def _parse_cores(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _find_runs(results_dir: Path, prefix: str) -> List[Path]:
    runs = [path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    return sorted(runs, key=lambda p: p.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize core sweep runs into a CSV.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix", default="graphrag_eval_v1_")
    parser.add_argument("--last", type=int, default=None, help="Only include the last N runs by name sort.")
    parser.add_argument("--cores", default=None, help="Comma-separated core limits aligned to run order.")
    parser.add_argument("--output", default="results/core_sweep_summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = _find_runs(results_dir, args.prefix)
    if args.last is not None:
        runs = runs[-args.last :]
    if not runs:
        raise SystemExit(f"No runs found in {results_dir} with prefix {args.prefix}")

    cores = _parse_cores(args.cores)
    if cores is not None and len(cores) != len(runs):
        raise SystemExit(f"--cores count ({len(cores)}) does not match runs ({len(runs)})")

    rows: List[dict] = []
    for idx, run in enumerate(runs):
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        core_limit = cores[idx] if cores is not None else None
        hardware = summary.get("hardware", {})
        for report in summary.get("reports", []):
            stats = report.get("stats", {})
            rows.append(
                {
                    "run_dir": run.name,
                    "core_limit": core_limit,
                    "engine": report.get("engine"),
                    "query": report.get("query"),
                    "count": stats.get("count"),
                    "mean_ms": stats.get("mean_ms"),
                    "p50_ms": stats.get("p50_ms"),
                    "p95_ms": stats.get("p95_ms"),
                    "p99_ms": stats.get("p99_ms"),
                    "min_ms": stats.get("min_ms"),
                    "max_ms": stats.get("max_ms"),
                    "errors": report.get("errors"),
                    "expect_failures": report.get("expect_failures"),
                    "cpu_count": hardware.get("cpu_count"),
                    "memory_total_mb": hardware.get("memory_total_mb"),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
