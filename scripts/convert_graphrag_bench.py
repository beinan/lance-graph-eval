#!/usr/bin/env python3
"""Convert GraphRAG-Bench corpus JSON into canonical JSONL graph format."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from typing import Dict, Iterable, List, Tuple


ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")


def read_corpus(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def chunk_text(text: str, chunk_words: int, overlap: int) -> Iterable[Tuple[int, str]]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append((start, chunk))
        if end == len(words):
            break
    return chunks


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned.lower() or "entity"


def extract_entities(text: str, max_entities: int) -> List[str]:
    found = ENTITY_RE.findall(text)
    uniq = []
    seen = set()
    for name in found:
        norm = " ".join(name.split())
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(norm)
        if len(uniq) >= max_entities:
            break
    return uniq


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def convert_corpus(
    corpus_path: str,
    out_dir: str,
    chunk_words: int,
    overlap: int,
    embedding_dim: int,
    entity_per_chunk: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    docs = []
    chunks = []
    entities: Dict[str, Dict[str, object]] = {}
    communities = []
    edges = []

    corpus = read_corpus(corpus_path)
    for doc_idx, item in enumerate(corpus):
        corpus_name = item.get("corpus_name") or f"doc_{doc_idx}"
        doc_id = f"doc_{doc_idx}"
        context = item.get("context", "")

        docs.append({
            "id": doc_id,
            "title": corpus_name,
            "text": context,
            "metadata": {"corpus_name": corpus_name},
        })

        summary = textwrap.shorten(context, width=280, placeholder="...")
        community_id = f"community_{doc_idx}"
        communities.append({
            "id": community_id,
            "level": 1,
            "summary": summary,
            "metadata": {"corpus_name": corpus_name},
        })
        edges.append({
            "src_id": doc_id,
            "src_type": "Document",
            "dst_id": community_id,
            "dst_type": "Community",
            "type": "IN_COMMUNITY",
        })

        for chunk_idx, (start_word, chunk_text_value) in enumerate(
            chunk_text(context, chunk_words, overlap)
        ):
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunks.append({
                "id": chunk_id,
                "document_id": doc_id,
                "text": chunk_text_value,
                "token_count": len(chunk_text_value.split()),
                "embedding": [1.0] * embedding_dim,
                "metadata": {"start_word": start_word},
            })
            edges.append({
                "src_id": doc_id,
                "src_type": "Document",
                "dst_id": chunk_id,
                "dst_type": "Chunk",
                "type": "HAS_CHUNK",
            })
            edges.append({
                "src_id": chunk_id,
                "src_type": "Chunk",
                "dst_id": doc_id,
                "dst_type": "Document",
                "type": "PARENT_OF",
            })

            for name in extract_entities(chunk_text_value, entity_per_chunk):
                entity_id = f"entity_{slugify(name)}"
                if entity_id not in entities:
                    entities[entity_id] = {
                        "id": entity_id,
                        "name": name,
                        "type": "PROPER_NOUN",
                        "metadata": None,
                    }
                edges.append({
                    "src_id": chunk_id,
                    "src_type": "Chunk",
                    "dst_id": entity_id,
                    "dst_type": "Entity",
                    "type": "MENTIONS",
                })

    write_jsonl(os.path.join(out_dir, "documents.jsonl"), docs)
    write_jsonl(os.path.join(out_dir, "chunks.jsonl"), chunks)
    write_jsonl(os.path.join(out_dir, "entities.jsonl"), entities.values())
    write_jsonl(os.path.join(out_dir, "communities.jsonl"), communities)
    write_jsonl(os.path.join(out_dir, "edges.jsonl"), edges)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to GraphRAG-Bench corpus JSON")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--entities-per-chunk", type=int, default=5)
    args = parser.parse_args()

    convert_corpus(
        args.corpus,
        args.out,
        args.chunk_words,
        args.overlap,
        args.embedding_dim,
        args.entities_per_chunk,
    )
    print(f"Wrote canonical dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
