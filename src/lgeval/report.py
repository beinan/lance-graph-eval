from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from lgeval.runner import BenchmarkReport, QueryReport


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir(base_dir: str, benchmark_name: str) -> str:
    path = os.path.join(base_dir, f"{benchmark_name}_{_timestamp()}")
    os.makedirs(path, exist_ok=True)
    return path


def write_report(report: BenchmarkReport, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    samples_path = os.path.join(out_dir, "samples.csv")
    summary_csv_path = os.path.join(out_dir, "summary.csv")

    _write_summary_json(report, summary_path)
    _write_samples_csv(report.reports, samples_path)
    _write_summary_csv(report.reports, report.system, summary_csv_path)

    return {
        "summary_json": summary_path,
        "samples_csv": samples_path,
        "summary_csv": summary_csv_path,
    }


def _write_summary_json(report: BenchmarkReport, path: str) -> None:
    payload = {
        "benchmark": asdict(report.benchmark),
        "system": report.system,
        "hardware": report.system,
        "reports": [
            {
                "engine": r.engine,
                "query": r.query,
                "tags": r.tags,
                "stats": r.stats,
                "errors": r.errors,
                "expect_failures": r.expect_failures,
            }
            for r in report.reports
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_samples_csv(reports: List[QueryReport], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "engine",
            "query",
            "run_idx",
            "duration_ms",
            "ok",
            "error",
            "row_count",
            "rss_mb",
            "expect_ok",
            "expect_error",
        ])
        for report in reports:
            for idx, sample in enumerate(report.samples):
                writer.writerow([
                    report.engine,
                    report.query,
                    idx,
                    f"{sample.duration_ms:.3f}",
                    sample.ok,
                    sample.error or "",
                    sample.row_count if sample.row_count is not None else "",
                    f"{sample.rss_mb:.3f}" if sample.rss_mb is not None else "",
                    sample.expect_ok if sample.expect_ok is not None else "",
                    sample.expect_error or "",
                ])


def _write_summary_csv(reports: List[QueryReport], system: Dict[str, object], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "engine",
            "query",
            "tags",
            "platform",
            "python",
            "cpu_count",
            "memory_total_mb",
            "count",
            "mean_ms",
            "stdev_ms",
            "min_ms",
            "max_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "errors",
            "expect_failures",
        ])
        for report in reports:
            stats = report.stats
            writer.writerow([
                report.engine,
                report.query,
                ",".join(report.tags),
                system.get("platform", ""),
                system.get("python", ""),
                system.get("cpu_count", ""),
                f"{system.get('memory_total_mb', ''):.3f}" if isinstance(system.get("memory_total_mb"), (int, float)) else system.get("memory_total_mb", ""),
                stats.get("count", 0),
                f"{stats.get('mean_ms', 0.0):.3f}",
                f"{stats.get('stdev_ms', 0.0):.3f}",
                f"{stats.get('min_ms', 0.0):.3f}",
                f"{stats.get('max_ms', 0.0):.3f}",
                f"{stats.get('p50_ms', 0.0):.3f}",
                f"{stats.get('p95_ms', 0.0):.3f}",
                f"{stats.get('p99_ms', 0.0):.3f}",
                report.errors,
                report.expect_failures,
            ])
