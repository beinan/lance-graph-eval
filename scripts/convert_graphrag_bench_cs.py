#!/usr/bin/env python3
"""Convert GraphRAG-Bench CS textbooks into canonical JSONL graph format."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import textwrap
from typing import Dict, Iterable, List, Tuple


ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b")


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


def normalize_textbooks_dir(path: str) -> str:
    if os.path.isdir(os.path.join(path, "textbooks")):
        return os.path.join(path, "textbooks")
    return path


def build_title(textbook: str, entry: Dict[str, object], idx: int) -> str:
    parts = []
    for key in ("chapter", "section", "subsection", "subsubsection"):
        value = str(entry.get(key, "")).strip()
        if not value or value.upper() == "N/A":
            continue
        parts.append(value)
    if not parts:
        parts = [f"section_{idx}"]
    return f"{textbook}: " + " / ".join(parts)


def convert_textbooks(
    textbooks_dir: str,
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

    base_dir = normalize_textbooks_dir(textbooks_dir)
    pattern = os.path.join(base_dir, "textbook*", "*_structured.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No structured textbooks found under {base_dir}")

    doc_idx = 0
    for path in files:
        textbook = os.path.basename(path).replace("_structured.json", "")
        with open(path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)

        for entry_idx, entry in enumerate(entries):
            context = str(entry.get("content", "")).strip()
            if not context:
                continue

            doc_id = f"doc_{doc_idx}"
            doc_idx += 1
            title = build_title(textbook, entry, entry_idx)
            docs.append({
                "id": doc_id,
                "title": title,
                "text": context,
                "metadata": {
                    "textbook": textbook,
                    "chapter": entry.get("chapter"),
                    "section": entry.get("section"),
                    "subsection": entry.get("subsection"),
                    "subsubsection": entry.get("subsubsection"),
                },
            })

            summary = textwrap.shorten(context, width=280, placeholder="...")
            community_id = f"community_{doc_id}"
            communities.append({
                "id": community_id,
                "level": 1,
                "summary": summary,
                "metadata": {"textbook": textbook},
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
    parser.add_argument("--textbooks-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--entity-per-chunk", type=int, default=5)
    args = parser.parse_args()

    convert_textbooks(
        textbooks_dir=args.textbooks_dir,
        out_dir=args.out,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        embedding_dim=args.embedding_dim,
        entity_per_chunk=args.entity_per_chunk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
