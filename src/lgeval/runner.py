from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from lgeval.config import BenchmarkSettings, EngineConfig, QuerySpec, SetupStep
from lgeval.engines import get_engine
from lgeval.metrics import summarize
from lgeval.types import QuerySample

try:
    import psutil
except ImportError:  # optional
    psutil = None


@dataclass
class QueryReport:
    engine: str
    query: str
    tags: List[str]
    stats: Dict[str, float]
    samples: List[QuerySample]
    errors: int
    expect_failures: int


@dataclass
class BenchmarkReport:
    benchmark: BenchmarkSettings
    reports: List[QueryReport]
    system: Dict[str, object]


def _rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _collect_system_info() -> Dict[str, object]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        info["memory_total_mb"] = psutil.virtual_memory().total / (1024 * 1024)
    return info


def _resolve_params(params: Dict[str, object]) -> Dict[str, object]:
    resolved = dict(params)
    embedding_file = resolved.pop("embedding_file", None)
    if embedding_file and "embedding" not in resolved:
        path = str(embedding_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"embedding_file not found: {path}. Set EMBEDDING_JSON or provide embedding inline."
            )
        with open(path, "r", encoding="utf-8") as handle:
            resolved["embedding"] = json.load(handle)
    return resolved


def _filter_params_for_query(query_text: str, params: Dict[str, object]) -> Dict[str, object]:
    if not params:
        return {}
    used = {}
    for key, value in params.items():
        if f"${key}" in query_text:
            used[key] = value
    return used


def _check_expect(row_count: Optional[float], expect: Dict[str, object]) -> (Optional[bool], Optional[str]):
    if not expect:
        return None, None
    if row_count is None:
        return False, "row_count unavailable for expectation check"

    if "row_count" in expect:
        expected = float(expect["row_count"])
        if float(row_count) != expected:
            return False, f"expected row_count={expected}, got {row_count}"
        return True, None

    min_rows = expect.get("min_rows")
    max_rows = expect.get("max_rows")
    if min_rows is not None and float(row_count) < float(min_rows):
        return False, f"row_count {row_count} < min_rows {min_rows}"
    if max_rows is not None and float(row_count) > float(max_rows):
        return False, f"row_count {row_count} > max_rows {max_rows}"
    return True, None


def _time_query(
    engine,
    query_text: str,
    params: Dict[str, object],
    fetch: str,
    timeout_s: float,
    expect: Dict[str, object],
) -> QuerySample:
    start = time.perf_counter()
    rss_before = _rss_mb()
    try:
        result = engine.run_query(query_text, params=params, fetch=fetch)
        duration_ms = (time.perf_counter() - start) * 1000
        rss_after = _rss_mb()
        expect_ok, expect_error = _check_expect(result.row_count, expect)
        return QuerySample(
            duration_ms=duration_ms,
            ok=True,
            row_count=result.row_count,
            rss_mb=(rss_after - rss_before) if rss_before is not None and rss_after is not None else None,
            expect_ok=expect_ok,
            expect_error=expect_error,
        )
    except Exception as exc:  # pylint: disable=broad-except
        duration_ms = (time.perf_counter() - start) * 1000
        return QuerySample(duration_ms=duration_ms, ok=False, error=str(exc))


class BenchmarkRunner:
    def __init__(
        self,
        settings: BenchmarkSettings,
        engines: List[EngineConfig],
        setup: List[SetupStep],
        queries: List[QuerySpec],
    ) -> None:
        self.settings = settings
        self.engines = engines
        self.setup = setup
        self.queries = queries

    def run(self) -> BenchmarkReport:
        reports: List[QueryReport] = []
        for engine_cfg in self.engines:
            engine = get_engine(engine_cfg.kind, engine_cfg.name, engine_cfg.options)
            engine.connect()

            try:
                self._run_setup(engine, engine_cfg)
                for query in self.queries:
                    query_text = query.texts.get(engine_cfg.kind)
                    if not query_text:
                        continue
                    report = self._run_query_suite(engine, engine_cfg, query, query_text)
                    reports.append(report)
            finally:
                engine.close()

        return BenchmarkReport(
            benchmark=self.settings,
            reports=reports,
            system=_collect_system_info(),
        )

    def _run_setup(self, engine, engine_cfg: EngineConfig) -> None:
        for step in self.setup:
            text = step.per_engine.get(engine_cfg.kind)
            if text:
                if isinstance(text, list):
                    engine.run_setup(text)
                else:
                    engine.run_setup([text])

    def _run_query_suite(self, engine, engine_cfg: EngineConfig, query: QuerySpec, query_text: str) -> QueryReport:
        params = _resolve_params(query.params)
        params = _filter_params_for_query(query_text, params)
        expect = query.expect

        for _ in range(self.settings.warmups):
            _time_query(engine, query_text, params, query.fetch, self.settings.timeout_s, expect)

        samples: List[QuerySample] = []
        latencies: List[float] = []
        errors = 0
        expect_failures = 0

        runs = self.settings.runs
        concurrency = max(1, self.settings.concurrency)
        if not engine.threadsafe:
            concurrency = 1

        if concurrency == 1:
            for _ in range(runs):
                sample = _time_query(engine, query_text, params, query.fetch, self.settings.timeout_s, expect)
                samples.append(sample)
                if sample.ok:
                    latencies.append(sample.duration_ms)
                    if sample.expect_ok is False:
                        expect_failures += 1
                else:
                    errors += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _time_query, engine, query_text, params, query.fetch, self.settings.timeout_s, expect
                    )
                    for _ in range(runs)
                ]
                for future in as_completed(futures):
                    sample = future.result()
                    samples.append(sample)
                    if sample.ok:
                        latencies.append(sample.duration_ms)
                        if sample.expect_ok is False:
                            expect_failures += 1
                    else:
                        errors += 1

        stats = summarize(latencies)
        return QueryReport(
            engine=engine_cfg.name,
            query=query.name,
            tags=query.tags,
            stats=stats,
            samples=samples,
            errors=errors,
            expect_failures=expect_failures,
        )
