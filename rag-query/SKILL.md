---
name: rag-query
description: >
  Searches indexed technical documentation using BM25 + dense embedding hybrid retrieval
  with cross-encoder reranking, parent-child context expansion, and pseudo-relevance feedback.
  Writes query output to a temp file, reads it, then synthesizes a grounded answer with inline citations.
---

# RAG Query

Hybrid BM25 + dense embedding search over ingested documentation chunks.

## Workflow

### Step 1 — Verify index

Check that the index file exists at `$env:RAG_PATH\out\rag_output.json`. If missing, run ingestion:

```bash
python <SKILL_DIR>/ingest.py --quiet --chunker recursive
```

If ingestion produces no chunks (empty documents, unsupported format, etc.), report this to the user with the specific error.

### Step 2 — Run query to temp file

#### 2a. Clean the search question

Strip logistical/meta instructions from the user's message. The search string should contain **only** the substantive topic — prefer keyword-style queries over full natural-language questions (which contain many stop words that dilute BM25 precision).

**Remove:**
- Action verbs / filler: *"search for"*, *"find"*, *"look up"*, *"please"*, *"can you"*, *"I want to know"*, *"explain"*, *"from indexed data"*
- Diversity / formatting directives: *"give me diverse results"*, *"from multiple sources"*, *"in JSON"*, *"with citations"*
- Temporal / logistical constraints: *"by tomorrow"*, *"as soon as possible"*

**Good:** keyword phrases — *"compressor shock waves types transonic boundary layer"*
**Bad:** full questions — *"what are compressor shockwaves and why are they bad?"* (too many stop words)

> User: *"Please search for how to start the engine and give me diverse results from multiple sources."*
>
> Clean query: *"engine start sequence ignition procedure"*

#### 2b. Choose flags

**Mandatory flags:** `--json`, `--rerank`, `--parent-child`, `--prf`, `--top-k 10`.

| Flag | When to use |
|------|-------------|
| `--filter-source "file.pdf"` | User asks about a specific document |
| `--filter-type "table"` | User needs only tables/figures/etc. |
| `--mmr-lambda 0.5` | **Opt-in only.** User explicitly asks for maximum diversity or wants results spread across all sources. Risk: can promote off-topic material. |

**Deduplication is automatic** (removes near-duplicate chunks silently). No `--mmr-lambda` means no MMR.

#### 2c. Run single query

Always start with a **single query**. Write output to a temp file:

```bash
python <SKILL_DIR>/query.py "search terms" --json --mode hybrid --rerank --parent-child --prf --top-k 10 --out $env:TEMP\opencode\rag_results.json
```

#### 2d. Retry with batch mode (second choice only)

Only if the single-query results in Step 3 are sparse, off-topic, or mostly noise, retry **once** with batch mode. Provide up to 3 keyword variations. The **first** query string MUST be the original question — the reranker uses it as its anchor.

```bash
python <SKILL_DIR>/query.py "original question" "keyword variation 1" "keyword variation 2" --json --mode hybrid --rerank --parent-child --prf --top-k 10 --out $env:TEMP\opencode\rag_results.json
```

### Step 3 — Read results

Read the temp file. Check:

- If empty or `"results": []` — retry once with broader/cleaner keywords (Step 2d). If still empty, tell the user no relevant chunks were found.
- If the script crashes with a FAISS or file-not-found error — run ingestion (Step 1) then retry.
- If **most results are off-topic** (e.g., fuel chemistry appearing for a compressor question) — retry with cleaner, more specific keywords (Step 2d). Long natural-language questions often cause this.
- If results exist and are on-topic — proceed to synthesize.

### Step 4 — Synthesize and cite

Write a single coherent answer grounded strictly in the retrieved text. Every factual claim must end with an inline citation: `[source_document, pg. page_range]`.

If results are sparse or low-quality, say so — don't fabricate information not present in the chunks.

---

## Data schema

Index path: `$env:RAG_PATH\out\rag_output.json`

```json
{
  "chunks": [
    {
      "chunk_id": "uuid",
      "parent_id": "uuid | null",
      "text": "...",
      "element_type": "text | table | heading | formula | list",
      "parent_heading": "section title",
      "document_hierarchy_level": 1,
      "token_count": 123,
      "source_document": "filename.pdf",
      "page_range": "10-12"
    }
  ]
}
```

When querying with `--parent-child`, child chunks are automatically expanded to their parent's full section text after reranking.
