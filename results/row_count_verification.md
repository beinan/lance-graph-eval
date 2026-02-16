# Row Count Verification

Each query shows the majority row_count per engine across samples. “OK” means all engines match.

## results/graphrag_eval_v1_medical_index_20260216_044355

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 77960 | 77960 | 77960 | OK |
| global_search | 1 | 1 | 1 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 5 | 5 | 5 | OK |

## results/graphrag_eval_v1_medical_no_index_20260216_044403

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 77960 | 77960 | 77960 | OK |
| global_search | 1 | 1 | 1 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 5 | 5 | 5 | OK |

## results/graphrag_eval_v1_novel_index_20260216_044414

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 215295 | 215295 | 215295 | OK |
| global_search | 20 | 20 | 20 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 16 | 16 | 16 | OK |

## results/graphrag_eval_v1_novel_no_index_20260216_044430

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 215295 | 215295 | 215295 | OK |
| global_search | 20 | 20 | 20 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 16 | 16 | 16 | OK |

## results/graphrag_eval_v1_cs_index_20260216_044509

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 6664 | 6664 | 6664 | OK |
| global_search | 4659 | 4659 | 4659 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 9 | 9 | 9 | OK |

## results/graphrag_eval_v1_cs_no_index_20260216_044532

| query | neo4j | kuzu | lance | status |
| --- | --- | --- | --- | --- |
| deep_traversal_5hop | 6664 | 6664 | 6664 | OK |
| global_search | 4659 | 4659 | 4659 | OK |
| hybrid_search | 10 | 10 | 10 | OK |
| local_search_2hop | 9 | 9 | 9 | OK |
