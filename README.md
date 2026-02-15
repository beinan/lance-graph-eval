# lance-graph-eval

Benchmark harness for GraphRAG retrieval workflows across lance-graph, Neo4j, and Kuzu.

## Quick start

1) Create a venv and install deps

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[all]
```

Start Neo4j (optional, for tri-engine runs):

```bash
NEO4J_AUTH=neo4j/neo4j_password docker compose -f docker-compose.neo4j.yml up -d
```

## Fully containerized (fair CPU limits)

To make a fair CPU comparison, run **all engines in containers** and pin the CPU set for both Neo4j and the
benchmark runner. Use the sweep script (uses `docker run` to avoid compose compatibility issues):

```bash
./scripts/run_core_sweep.sh
```

This runs 1/2/4/32/96 core sweeps with the same CPU pinning for Neo4j and the runner.
By default it uses the medical dataset. To sweep a different dataset, set:

```bash
DATASET_FLAVOR=novel ./scripts/run_core_sweep.sh
DATASET_FLAVOR=cs ./scripts/run_core_sweep.sh
```

Summarize sweep results into one CSV:

```bash
python3 scripts/summarize_core_sweep.py --cores 1,2,4,32,96 --last 5
```

Dataset options are tracked in `docs/datasets.md`.

2) Edit the sample config

```bash
cp configs/graphrag_eval.yaml configs/local.yaml
```

3) Set required env vars

```bash
export EMBEDDING_JSON="$(pwd)/datasets/embedding.json"
export GRAPHRAG_MEDICAL_PATH="$(pwd)/datasets/graph/graphrag_bench_medical"
export GRAPHRAG_NOVEL_PATH="$(pwd)/datasets/graph/graphrag_bench_novel"
export GRAPHRAG_LANCE_DATASETS="$(pwd)/datasets/graph/graphrag_bench_medical/parquet"
export KUZU_PATH="$(pwd)/datasets/kuzu_medical.db"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4j_password"
export NEO4J_DATABASE="neo4j"
```

4) Run

```bash
lgeval --config configs/local.yaml --out results
```

## Download GraphRAG-Bench data

This repo uses the GraphRAG-Bench corpus JSON files from Hugging Face and converts them into a canonical
JSONL graph format.

Download the corpus + questions:

```bash
mkdir -p datasets/raw/graphrag_bench
curl -L -o datasets/raw/graphrag_bench/medical.json \\
  https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/medical.json
curl -L -o datasets/raw/graphrag_bench/novel.json \\
  https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Corpus/novel.json
curl -L -o datasets/raw/graphrag_bench/medical_questions.json \\
  https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/medical_questions.json
curl -L -o datasets/raw/graphrag_bench/novel_questions.json \\
  https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench/resolve/main/Datasets/Questions/novel_questions.json
```

Convert to canonical graph JSONL:

```bash
python3 scripts/convert_graphrag_bench.py --corpus datasets/raw/graphrag_bench/medical.json \\
  --out datasets/graph/graphrag_bench_medical --embedding-dim 32
python3 scripts/convert_graphrag_bench.py --corpus datasets/raw/graphrag_bench/novel.json \\
  --out datasets/graph/graphrag_bench_novel --embedding-dim 32
```

### GraphRAG-Bench CS variant (textbooks)

Download the `textbooks/` folder from the Awesome-GraphRAG dataset:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Awesome-GraphRAG/GraphRAG-Bench",
    repo_type="dataset",
    allow_patterns=["textbooks/**"],
    local_dir="datasets/raw/graphrag_bench_cs",
)
PY
```

Convert structured textbook entries to the canonical JSONL layout:

```bash
python3 scripts/convert_graphrag_bench_cs.py \\
  --textbooks-dir datasets/raw/graphrag_bench_cs/textbooks \\
  --out datasets/graph/graphrag_bench_cs --embedding-dim 32
```

Prepare Parquet for lance-graph and ingest Kuzu:

```bash
python3 scripts/prepare_parquet.py --dataset datasets/graph/graphrag_bench_cs
python3 scripts/ingest_kuzu.py --db datasets/kuzu_cs.db --dataset datasets/graph/graphrag_bench_cs --bulk --reset
```

Note: Kuzu vector indexes require fixed-length arrays. Use `--embedding-dim` to match your embedding size.

To run the benchmark against the CS variant, set:

```bash
export GRAPHRAG_DATASET_DIR="$(pwd)/datasets/graph/graphrag_bench_cs"
export GRAPHRAG_LANCE_DATASETS="$(pwd)/datasets/graph/graphrag_bench_cs/parquet"
export KUZU_PATH="$(pwd)/datasets/kuzu_cs.db"
export GRAPHRAG_CS_PATH="$GRAPHRAG_DATASET_DIR"
```

## Config overview

- `benchmark`: run settings (runs, warmups, concurrency, timeout)
- `engines`: connection info per engine
- `setup`: per-engine setup queries (indexes, schema, preload)
- `queries`: per-engine query texts for each benchmark task

See `configs/graphrag_eval.yaml` for the single source-of-truth config. The other YAMLs are kept in sync
as copies for compatibility but are not maintained separately.

## Dataset format

The canonical JSONL layout is documented in `datasets/README.md`. You can convert JSONL to Parquet for columnar
engines with:

```bash
python3 scripts/prepare_parquet.py --dataset /path/to/dataset
```

## Ingestion helpers

Neo4j:

```bash
python3 scripts/ingest_neo4j.py --uri bolt://localhost:7687 --user neo4j --password neo4j_password \\
  --dataset /path/to/dataset --derive-has-chunk --create-vector-index --create-fulltext-index
```

Kuzu (row-wise, slower but simple):

```bash
python3 scripts/ingest_kuzu.py --db /path/to/kuzu.db --dataset /path/to/dataset --derive-has-chunk
```

Lance-graph (lance datasets mode):

1) Convert JSONL or Parquet to Lance datasets:

```bash
python3 scripts/prepare_lance.py --dataset /path/to/dataset --from-parquet
export GRAPHRAG_LANCE_DATASETS=/path/to/dataset/lance
```

2) Ensure the `tables` section in `configs/graphrag_eval.yaml` matches the Lance filenames.

## Notes

- You should tune vector index parameters per engine (or fix them across engines) and keep those values recorded in the config.
- If your queries can return large payloads, consider writing them to return counts only to reduce transfer overhead during benchmarking.
- The sample config uses `${EMBEDDING_JSON}`. You can generate one with:

  ```bash
  python3 scripts/make_embedding.py --dim 32 --out /tmp/embedding.json
  export EMBEDDING_JSON=/tmp/embedding.json
  ```
- Dataset paths in the sample config use `${GRAPHRAG_MEDICAL_PATH}` and `${GRAPHRAG_NOVEL_PATH}`; set those env vars before running. Any string in the config supports `${VAR}` expansion.
- Queries can define `expect` constraints (e.g., `min_rows`) to validate basic correctness. Adjust or remove them if your dataset is sparse.
- Use `fetch: scalar` for queries that return a single numeric value (e.g., `RETURN count(*) AS n`).
- This harness measures end-to-end client latency (driver + server time). If you want server-side timing, add timing functions within each engine.
