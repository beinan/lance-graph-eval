from __future__ import annotations

import json
import os
import platform
import time
from queue import Queue
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


class _EnginePool:
    def __init__(self, engines: List[object]):
        self._queue: Queue[object] = Queue()
        self._size = len(engines)
        for engine in engines:
            self._queue.put(engine)

    @property
    def size(self) -> int:
        return self._size

    def acquire(self):
        return self._queue.get()

    def release(self, engine) -> None:
        self._queue.put(engine)

    def close(self) -> None:
        while not self._queue.empty():
            try:
                engine = self._queue.get_nowait()
            except Exception:
                break
            try:
                engine.close()
            except Exception:
                pass


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


def _apply_vector_search_params(params: Dict[str, object], vector_search: Dict[str, object]) -> Dict[str, object]:
    if not vector_search or not vector_search.get("enabled"):
        return params

    merged = dict(params)
    merged["vector_rerank"] = True

    column = vector_search.get("column")
    if column:
        merged["vector_column"] = column

    metric = vector_search.get("metric")
    if metric:
        merged["distance_metric"] = metric

    if "top_k" in vector_search:
        merged["top_k"] = vector_search.get("top_k")
    else:
        top_k_param = vector_search.get("top_k_param")
        if top_k_param and top_k_param in merged:
            merged["top_k"] = merged.get(top_k_param)

    vector_param = vector_search.get("vector_param")
    if vector_param and vector_param in merged:
        merged["embedding"] = merged.get(vector_param)

    return merged


def _select_vector_search(query: QuerySpec, engine_cfg: EngineConfig) -> Dict[str, object]:
    entry = query.texts.get(engine_cfg.kind)
    if isinstance(entry, dict):
        vs = entry.get("vector_search")
        if isinstance(vs, dict):
            return vs

    if query.vector_search:
        return query.vector_search
    options = engine_cfg.options or {}
    vs = options.get("vector_search")
    if not vs:
        return {}
    if isinstance(vs, dict):
        if "enabled" in vs:
            return vs
        if query.name in vs:
            return vs.get(query.name, {})
        return vs.get("default", {}) or {}
    return {}


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
            pool = self._build_engine_pool(engine_cfg)
            if pool is None:
                engine = get_engine(engine_cfg.kind, engine_cfg.name, engine_cfg.options)
                engine.connect()
                primary_engine = engine
            else:
                primary_engine = pool.acquire()

            try:
                # Setup runs once on the primary engine to avoid duplicate side effects.
                if pool is None:
                    self._run_setup(primary_engine, engine_cfg)
                else:
                    try:
                        self._run_setup(primary_engine, engine_cfg)
                    finally:
                        pool.release(primary_engine)
                for query in self.queries:
                    entry = query.texts.get(engine_cfg.kind)
                    if not entry:
                        continue
                    if isinstance(entry, dict):
                        query_text = entry.get("query") or entry.get("cypher") or entry.get("text")
                    else:
                        query_text = entry
                    if not query_text:
                        continue
                    report = self._run_query_suite(primary_engine, pool, engine_cfg, query, query_text)
                    reports.append(report)
            finally:
                if pool is None:
                    primary_engine.close()
                else:
                    pool.close()

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

    def _run_query_suite(
        self,
        engine,
        pool: Optional[_EnginePool],
        engine_cfg: EngineConfig,
        query: QuerySpec,
        query_text: str,
    ) -> QueryReport:
        params = _resolve_params(query.params)
        vector_search = _select_vector_search(query, engine_cfg)
        if not query.pass_all_params and not vector_search:
            params = _filter_params_for_query(query_text, params)
        if engine_cfg.kind == "lance_graph":
            params = _apply_vector_search_params(params, vector_search)
        expect = query.expect

        def _run_once():
            if pool is None:
                return _time_query(engine, query_text, params, query.fetch, self.settings.timeout_s, expect)
            pooled = pool.acquire()
            try:
                return _time_query(pooled, query_text, params, query.fetch, self.settings.timeout_s, expect)
            finally:
                pool.release(pooled)

        for _ in range(self.settings.warmups):
            _run_once()

        samples: List[QuerySample] = []
        latencies: List[float] = []
        errors = 0
        expect_failures = 0

        runs = self.settings.runs
        concurrency = max(1, self.settings.concurrency)
        if pool is not None:
            concurrency = min(concurrency, pool.size)
        elif not engine.threadsafe:
            concurrency = 1

        if concurrency == 1:
            for _ in range(runs):
                sample = _run_once()
                samples.append(sample)
                if sample.ok:
                    latencies.append(sample.duration_ms)
                    if sample.expect_ok is False:
                        expect_failures += 1
                else:
                    errors += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_run_once) for _ in range(runs)]
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

    def _build_engine_pool(self, engine_cfg: EngineConfig) -> Optional[_EnginePool]:
        options = engine_cfg.options or {}
        if not options.get("engine_pool"):
            return None
        pool_size = int(options.get("pool_size") or self.settings.concurrency or 1)
        if pool_size < 1:
            pool_size = 1
        engines = []
        for _ in range(pool_size):
            engine = get_engine(engine_cfg.kind, engine_cfg.name, options)
            engine.connect()
            engines.append(engine)
        return _EnginePool(engines)
