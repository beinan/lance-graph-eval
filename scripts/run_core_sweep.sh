#!/usr/bin/env bash
set -euo pipefail

CORES_LIST=(1 2 4 32 96)
IMAGE_NAME="lance-graph-eval-runner"
NEO4J_CONTAINER="lgeval-neo4j"
NEO4J_AUTH_VAL=${NEO4J_AUTH:-neo4j/neo4j_password}
NETWORK_NAME="lgeval-net"

NEO4J_USER_VAL=${NEO4J_USER:-${NEO4J_AUTH_VAL%%/*}}
NEO4J_PASSWORD_VAL=${NEO4J_PASSWORD:-${NEO4J_AUTH_VAL#*/}}
NEO4J_DATABASE_VAL=${NEO4J_DATABASE:-neo4j}

REPO_ROOT="$(pwd)"
DATASETS_ROOT="${REPO_ROOT}/datasets"
HOST_MEDICAL_DATASET_DIR=${GRAPHRAG_MEDICAL_PATH:-${DATASETS_ROOT}/graph/graphrag_bench_medical}
HOST_NOVEL_DATASET_DIR=${GRAPHRAG_NOVEL_PATH:-${DATASETS_ROOT}/graph/graphrag_bench_novel}
HOST_CS_DATASET_DIR=${GRAPHRAG_CS_PATH:-${DATASETS_ROOT}/graph/graphrag_bench_cs}
DATASET_FLAVOR=${DATASET_FLAVOR:-medical}

case "$DATASET_FLAVOR" in
  medical)
    HOST_DATASET_DIR=${GRAPHRAG_DATASET_DIR:-$HOST_MEDICAL_DATASET_DIR}
    HOST_KUZU_PATH=${KUZU_PATH:-${DATASETS_ROOT}/kuzu_medical.db}
    ;;
  novel)
    HOST_DATASET_DIR=${GRAPHRAG_DATASET_DIR:-$HOST_NOVEL_DATASET_DIR}
    HOST_KUZU_PATH=${KUZU_PATH:-${DATASETS_ROOT}/kuzu_novel.db}
    ;;
  cs)
    HOST_DATASET_DIR=${GRAPHRAG_DATASET_DIR:-$HOST_CS_DATASET_DIR}
    HOST_KUZU_PATH=${KUZU_PATH:-${DATASETS_ROOT}/kuzu_cs.db}
    ;;
  *)
    echo "Unsupported DATASET_FLAVOR: $DATASET_FLAVOR (expected: medical|novel|cs)"
    exit 1
    ;;
esac

HOST_LANCE_DATASETS=${GRAPHRAG_LANCE_DATASETS:-${HOST_DATASET_DIR}/parquet}
PYTHON_BIN="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

if [[ "$HOST_DATASET_DIR" != "$DATASETS_ROOT/"* ]]; then
  echo "GRAPHRAG_DATASET_DIR must live under ${DATASETS_ROOT}"
  exit 1
fi
if [[ "$HOST_MEDICAL_DATASET_DIR" != "$DATASETS_ROOT/"* ]]; then
  echo "GRAPHRAG_MEDICAL_PATH must live under ${DATASETS_ROOT}"
  exit 1
fi
if [[ "$HOST_NOVEL_DATASET_DIR" != "$DATASETS_ROOT/"* ]]; then
  echo "GRAPHRAG_NOVEL_PATH must live under ${DATASETS_ROOT}"
  exit 1
fi
if [[ "$HOST_CS_DATASET_DIR" != "$DATASETS_ROOT/"* ]]; then
  echo "GRAPHRAG_CS_PATH must live under ${DATASETS_ROOT}"
  exit 1
fi
if [[ "$HOST_LANCE_DATASETS" != "$DATASETS_ROOT/"* ]]; then
  echo "GRAPHRAG_LANCE_DATASETS must live under ${DATASETS_ROOT}"
  exit 1
fi
if [[ "$HOST_KUZU_PATH" != "$DATASETS_ROOT/"* ]]; then
  echo "KUZU_PATH must live under ${DATASETS_ROOT}"
  exit 1
fi

CONTAINER_DATASET_DIR="/app/datasets/${HOST_DATASET_DIR#${DATASETS_ROOT}/}"
CONTAINER_MEDICAL_DATASET_DIR="/app/datasets/${HOST_MEDICAL_DATASET_DIR#${DATASETS_ROOT}/}"
CONTAINER_NOVEL_DATASET_DIR="/app/datasets/${HOST_NOVEL_DATASET_DIR#${DATASETS_ROOT}/}"
CONTAINER_CS_DATASET_DIR="/app/datasets/${HOST_CS_DATASET_DIR#${DATASETS_ROOT}/}"
CONTAINER_LANCE_DATASETS="/app/datasets/${HOST_LANCE_DATASETS#${DATASETS_ROOT}/}"
CONTAINER_KUZU_PATH="/app/datasets/${HOST_KUZU_PATH#${DATASETS_ROOT}/}"

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 || docker network create "$NETWORK_NAME" >/dev/null

# Build runner image (use FORCE_REBUILD=1 to refresh)
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  docker build -t "$IMAGE_NAME" .
elif ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  docker build -t "$IMAGE_NAME" .
fi

for CORES in "${CORES_LIST[@]}"; do
  if [[ "$CORES" -le 1 ]]; then
    CPUSET="0"
  else
    CPUSET="0-$((CORES-1))"
  fi

  echo "Running sweep with $CORES cores (cpuset: $CPUSET)"

  # Cleanup any existing neo4j container
  docker rm -f "$NEO4J_CONTAINER" >/dev/null 2>&1 || true
  docker volume rm -f neo4j_data neo4j_logs >/dev/null 2>&1 || true

  # Start neo4j with pinned CPUs
  docker run -d --name "$NEO4J_CONTAINER" \
    --network "$NETWORK_NAME" \
    --cpuset-cpus "$CPUSET" --cpus "$CORES" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH="$NEO4J_AUTH_VAL" \
    -e NEO4J_dbms_memory_pagecache_size=2G \
    -e NEO4J_dbms_memory_heap_initial__size=2G \
    -e NEO4J_dbms_memory_heap_max__size=2G \
    -v neo4j_data:/data -v neo4j_logs:/logs \
    neo4j:latest

  # Wait for Neo4j to accept Bolt connections
  READY=0
  for _ in $(seq 1 30); do
    if docker exec "$NEO4J_CONTAINER" cypher-shell -u "$NEO4J_USER_VAL" -p "$NEO4J_PASSWORD_VAL" "RETURN 1" >/dev/null 2>&1; then
      READY=1
      break
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "Neo4j did not become ready in time."
    docker logs "$NEO4J_CONTAINER" | tail -n 200 || true
    docker rm -f "$NEO4J_CONTAINER" >/dev/null 2>&1 || true
    exit 1
  fi

  echo "Ingesting Neo4j dataset..."
  docker run --rm \
    --network "$NETWORK_NAME" \
    --cpuset-cpus "$CPUSET" --cpus "$CORES" \
    -v "$(pwd)/datasets:/app/datasets" \
    --entrypoint python3 \
    "$IMAGE_NAME" /app/scripts/ingest_neo4j.py \
      --uri "bolt://$NEO4J_CONTAINER:7687" \
      --user "$NEO4J_USER_VAL" \
      --password "$NEO4J_PASSWORD_VAL" \
      --database "$NEO4J_DATABASE_VAL" \
      --dataset "$CONTAINER_DATASET_DIR" \
      --reset \
      --embedding-dim 32 \
      --create-vector-index \
      --create-fulltext-index

  echo "Resetting Kuzu dataset..."
  rm -f "$HOST_KUZU_PATH" "$HOST_KUZU_PATH.wal"
  "$PYTHON_BIN" scripts/ingest_kuzu.py \
    --db "$HOST_KUZU_PATH" \
    --dataset "$HOST_DATASET_DIR" \
    --bulk \
    --reset \
    --embedding-dim 32

  # Run benchmark runner container
  docker run --rm \
    --network "$NETWORK_NAME" \
    --cpuset-cpus "$CPUSET" --cpus "$CORES" \
    -e EMBEDDING_JSON=/app/datasets/embedding.json \
    -e DATASET_FLAVOR="$DATASET_FLAVOR" \
    -e GRAPHRAG_DATASET_NAME="$DATASET_FLAVOR" \
    -e GRAPHRAG_MEDICAL_PATH="$CONTAINER_MEDICAL_DATASET_DIR" \
    -e GRAPHRAG_NOVEL_PATH="$CONTAINER_NOVEL_DATASET_DIR" \
    -e GRAPHRAG_CS_PATH="$CONTAINER_CS_DATASET_DIR" \
    -e GRAPHRAG_LANCE_DATASETS="$CONTAINER_LANCE_DATASETS" \
    -e KUZU_PATH="$CONTAINER_KUZU_PATH" \
    -e NEO4J_URI="bolt://$NEO4J_CONTAINER:7687" \
    -e NEO4J_USER="$NEO4J_USER_VAL" \
    -e NEO4J_PASSWORD="$NEO4J_PASSWORD_VAL" \
    -e NEO4J_DATABASE="$NEO4J_DATABASE_VAL" \
    -v "$(pwd)/datasets:/app/datasets" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/configs:/app/configs" \
    "$IMAGE_NAME" --config configs/graphrag_eval.yaml --out results

  # Stop neo4j container
  docker rm -f "$NEO4J_CONTAINER" >/dev/null 2>&1 || true

done
