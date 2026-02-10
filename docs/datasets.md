# Benchmark Datasets

This document tracks larger-scale dataset options for GraphRAG-style benchmarks.

## GraphRAG-Bench (Medical / Novel)

- Source: Hugging Face dataset `GraphRAG-Bench/GraphRAG-Bench`
- Notes: Same corpus format used by the existing `scripts/convert_graphrag_bench.py`.

## GraphRAG-Bench CS Variant (Textbooks)

- Source: Hugging Face dataset `Awesome-GraphRAG/GraphRAG-Bench`, `textbooks/` folder
- Notes: ~7M words from 20 CS textbooks; entries are structured by chapter/section.

## KILT (Wikipedia)

- Source: Hugging Face dataset `corag/kilt-corpus` (Wikipedia text)
- Related: Hugging Face dataset `naist-nlp/kilt` (entity dictionary)
- Notes: Large corpus but requires constructing graph edges.

## WikiKG90Mv2 (OGB-LSC)

- Source: OGB-LSC WikiKG90Mv2 dataset
- Notes: Very large knowledge graph; text enrichment needed for GraphRAG.

## 2WikiMultiHopQA

- Source: Hugging Face dataset `framolfese/2WikiMultihopQA`
- Notes: Multi-hop QA with linked documents.

## MuSiQue

- Source: GitHub dataset `stonybrooknlp/musique`
- Notes: Multi-hop reasoning benchmark; not a full-scale corpus by itself.
