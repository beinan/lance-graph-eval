# Dataset format

This repo expects a simple canonical JSONL layout per dataset directory. Each file is optional, but queries
assume the core nodes and relationships exist.

Required node files (JSONL, one object per line):
- `documents.jsonl`: `{id, title?, text?, metadata?}`
- `chunks.jsonl`: `{id, document_id?, text?, embedding?, token_count?, metadata?}`
- `entities.jsonl`: `{id, name, type?, metadata?}`
- `communities.jsonl`: `{id, level?, summary?, metadata?}`

Relationships (JSONL):
- `edges.jsonl`: `{src_id, src_type, dst_id, dst_type, type, properties?}`
  - `src_type`/`dst_type` must be one of: `Document`, `Chunk`, `Entity`, `Community`.
  - `type` should be a valid relationship name like `HAS_CHUNK`, `MENTIONS`, `PARENT_OF`, `IN_COMMUNITY`.

Notes
- The benchmark queries reference `Chunk.embedding`, `Entity.name`, and `Community.level`.
- If you have `document_id` in `chunks.jsonl`, you can either also emit `HAS_CHUNK` edges or let your ingest
  script derive them.
- In lance-graph datasets mode, relationships are filtered from `edges.parquet` by `type`, so ensure your edges
  include a `type` field with values like `HAS_CHUNK`, `MENTIONS`, `PARENT_OF`.
