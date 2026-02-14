from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BenchmarkSettings:
    name: str = "benchmark"
    runs: int = 30
    warmups: int = 5
    concurrency: int = 1
    timeout_s: float = 30.0
    seed: int = 0
    output_dir: str = "results"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    name: str
    kind: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    path: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class SetupStep:
    name: str
    per_engine: Dict[str, str] = field(default_factory=dict)


@dataclass
class QuerySpec:
    name: str
    texts: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    fetch: str = "count"
    expect: Dict[str, Any] = field(default_factory=dict)
    pass_all_params: bool = False
    vector_search: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    benchmark: BenchmarkSettings
    engines: List[EngineConfig]
    datasets: List[DatasetSpec]
    setup: List[SetupStep]
    queries: List[QuerySpec]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    data = _expand(data)

    bench = data.get("benchmark", {})
    benchmark = BenchmarkSettings(
        name=bench.get("name", "benchmark"),
        runs=int(bench.get("runs", 30)),
        warmups=int(bench.get("warmups", 5)),
        concurrency=int(bench.get("concurrency", 1)),
        timeout_s=float(bench.get("timeout_s", 30)),
        seed=int(bench.get("seed", 0)),
        output_dir=bench.get("output_dir", "results"),
        metadata=dict(bench.get("metadata", {})),
    )

    engines = []
    for item in _as_list(data.get("engines")):
        engines.append(
            EngineConfig(
                name=item["name"],
                kind=item["kind"],
                options=dict(item.get("options", {})),
            )
        )

    datasets = []
    for item in _as_list(data.get("datasets")):
        datasets.append(
            DatasetSpec(
                name=item["name"],
                path=item.get("path"),
                notes=item.get("notes"),
            )
        )

    setup = []
    for item in _as_list(data.get("setup")):
        setup.append(
            SetupStep(
                name=item["name"],
                per_engine=dict(item.get("per_engine", {})),
            )
        )

    queries = []
    for item in _as_list(data.get("queries")):
        queries.append(
            QuerySpec(
                name=item["name"],
                texts=dict(item.get("texts", {})),
                params=dict(item.get("params", {})),
                tags=list(item.get("tags", [])),
                fetch=item.get("fetch", "count"),
                expect=dict(item.get("expect", {})),
                pass_all_params=bool(item.get("pass_all_params", False)),
                vector_search=dict(item.get("vector_search", {})),
            )
        )

    return Config(
        benchmark=benchmark,
        engines=engines,
        datasets=datasets,
        setup=setup,
        queries=queries,
    )
