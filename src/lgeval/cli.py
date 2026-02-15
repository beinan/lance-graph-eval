from __future__ import annotations

import argparse
import os

from lgeval.config import load_config
from lgeval.report import ensure_output_dir, write_report
from lgeval.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG benchmark harness")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML")
    parser.add_argument("--out", default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(f"Loaded config with {len(config.engines)} engines and {len(config.queries)} queries")
        return 0

    out_base = args.out or config.benchmark.output_dir
    metadata = config.benchmark.metadata or {}
    dataset_name = (
        metadata.get("dataset_name")
        or metadata.get("dataset")
        or os.getenv("GRAPHRAG_DATASET_NAME")
        or os.getenv("DATASET_FLAVOR")
    )
    out_dir = ensure_output_dir(out_base, config.benchmark.name, dataset_name)

    runner = BenchmarkRunner(
        settings=config.benchmark,
        engines=config.engines,
        setup=config.setup,
        queries=config.queries,
    )
    report = runner.run()
    paths = write_report(report, out_dir)

    print("Benchmark complete")
    print(f"Summary JSON: {os.path.relpath(paths['summary_json'])}")
    print(f"Summary CSV:  {os.path.relpath(paths['summary_csv'])}")
    print(f"Samples CSV:  {os.path.relpath(paths['samples_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
