# Related work and benchmark baselines

This project compares **collaborative trust-aware memory** against published memory architectures and production patterns. Each extra baseline is a **faithful simplification** of the cited idea, implemented in `baselines/` so we can run the same scenarios and LLM tasks.

## Techniques in the benchmark

| Technique | Inspiration | What we implement |
|-----------|-------------|-------------------|
| **collaborative** | This repo (trust fusion, lineage, contradictions) | Full `CollaborativeMemoryRetriever` + `full` ablation |
| **simple_rag** | Classic RAG | FAISS + embeddings only |
| **mem0** | [Mem0](https://github.com/mem0ai/mem0) | Real `mem0ai` SDK, per-`user_id` scope |
| **hybrid_bm25** | INMS / hybrid search (sparse + dense) | BM25 + dense vectors fused with RRF |
| **memgpt_tiered** | [Letta / MemGPT](https://github.com/letta-ai/letta) | Pinned “core” recent slots + archival FAISS |
| **isolated_memory** | [Collaborative Memory for Personalized Multi-User AI](https://arxiv.org/abs/2505.18279) | Author-only retrieval; shared tier invisible |
| **a_mem_linked** | [A-Mem: Agentic Memory](https://arxiv.org/abs/2502.12110) | Semantic seed + tag-overlap neighbor expansion |

## Why these baselines matter for our setting

- **Hybrid BM25+dense** tests whether recall gains come from fusion alone, without trust or lineage.
- **MemGPT tiered** tests whether “always in context” recent pins help multi-hop questions without cross-agent trust.
- **Isolated memory** is the natural ablation of collaborative shared memory: same store, no cross-agent reads.
- **A-Mem linked** tests structured linking (tags) vs our contradiction/lineage graph.
- **Mem0** tests a strong product baseline with **user-scoped** memory (often misses cross-agent shared facts by design).

## Complex LLM evaluation suite

`evaluation/llm_complex_suite.py` defines **10 tasks** on one storyline:

1. Auth migration winner (OAuth/PKCE, not JWT)
2. Verifier-trusted deployment schedule
3. Multi-hop DB pool + RabbitMQ routing
4. Observability p99 SLO
5. Active rollback after supersede
6. Privacy (agent_c must not leak agent_a local)
7. Cross-agent Redis TTL
8. Stale JWT suppression
9. Infra cache + DB in one answer
10. Contradiction rationale (logout vuln → OAuth)

Run:

```bash
python scripts/run_experiments.py llm-eval --suite complex --seed 42
```

That runs **10 tasks × 7 techniques = 70** xAI answer + judge calls. Use `--technique collaborative` to smoke-test one technique first.

## Metrics

- **Judge score** (0–1): xAI JSON judge on rubric fit.
- **Task success** (0–1): required terms in the answer with no forbidden leaks (automatic, comparable across runs).

## Further reading

- Trust in multi-agent systems and calibrated retrieval (project `docs/RESEARCH_SCOPE.md`)
- Ablation presets: `retrieval/ablation.py`
- Architecture: `docs/ARCHITECTURE.md`
