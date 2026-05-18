#!/usr/bin/env python3
"""BM25 query script for the RAG ingestion pipeline output.

Searches the pre-built inverted index in rag_output.json using BM25
scoring (k1=1.5, b=0.75) and returns the top-K ranked chunks.

Usage:
    python query.py "search terms"
    python query.py "nozzle design" --top-k 10
    python query.py "thrust" --index path/to/rag_output.json
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Auto-install missing dependencies & pre-download models on startup
# ---------------------------------------------------------------------------
import importlib as _importlib
import subprocess as _subprocess
import sys as _sys


def _ensure_dependencies():
    """Check for required packages and install any that are missing."""
    deps = {
        "numpy": "numpy",
        "faiss": "faiss-cpu",
        "sentence_transformers": "sentence-transformers",
    }
    missing = []
    for import_name, pip_name in deps.items():
        try:
            _importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[query] Installing missing packages: {', '.join(missing)} ...", flush=True)
        _subprocess.check_call(
            [_sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[query] Packages installed successfully.", flush=True)


def _ensure_model_cached(model_name: str):
    """Download a model via sentence-transformers if not already cached."""
    try:
        from sentence_transformers import SentenceTransformer
        try:
            SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            print(f"[query] Downloading model '{model_name}' (first run, may take a while) ...", flush=True)
            SentenceTransformer(model_name)  # Triggers download
            print(f"[query] Model '{model_name}' cached successfully.", flush=True)
    except ImportError:
        pass  # Will be installed by _ensure_dependencies
    except Exception as e:
        print(f"[query] Warning: could not pre-cache model '{model_name}': {e}", flush=True)


_ensure_dependencies()
_ensure_model_cached("BAAI/bge-small-en-v1.5")
_ensure_model_cached("cross-encoder/ms-marco-MiniLM-L-6-v2")

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── Complete silence: suppress ALL progress bars, warnings, logs ──
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import logging as _logging
for _name in ("sentence_transformers", "transformers", "huggingface_hub", "tqdm"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_rag_path = os.getenv("RAG_PATH")
if not _rag_path:
    raise RuntimeError(
        "RAG_PATH environment variable is not set. "
        "Set it to your RAG workspace directory (containing in/ and out/ folders)."
    )
BASE_DIR: Path = Path(_rag_path)

DEFAULT_INDEX_PATH: Path = BASE_DIR / "out" / "rag_output.json"
DEFAULT_FAISS_PATH: Path = BASE_DIR / "out" / "rag_embeddings.faiss"
DEFAULT_CHUNK_IDS_PATH: Path = BASE_DIR / "out" / "rag_chunk_ids.json"
DEFAULT_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_K1 = 1.5
BM25_B = 0.75

TECHNICAL_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "also", "then", "there", "here", "about",
    "up", "out", "into", "over", "after", "before", "between",
    "under", "again", "further", "once", "any", "if", "while",
    "above", "below", "through", "during", "until", "against",
    "fig", "figure", "table", "eq", "equation", "section", "chapter",
    "note", "example", "see", "et", "al", "ie", "eg", "etc", "ref",
    "using", "used", "via", "per", "null", "true", "false", "none",
}

_PUNCTUATION_RE = re.compile(r"[^\w\s]")

# Minimum token count below which a heading-only chunk is considered noise
DEFAULT_MIN_TOKENS = 15


def is_meaningful_chunk(chunk: dict, min_tokens: int = DEFAULT_MIN_TOKENS) -> bool:
    """Filter out heading-only chunks that are too short to be useful."""
    if chunk.get("element_type") == "heading" and chunk.get("token_count", 0) < min_tokens:
        return False
    return True


def get_parent_ids(index_data: dict) -> Set[str]:
    """Return chunk_ids that are parents (have children pointing to them)."""
    pids: Set[str] = set()
    for c in index_data["chunks"]:
        pid = c.get("parent_id")
        if pid:
            pids.add(pid)
    return pids


def normalize_token(token: str) -> Optional[str]:
    token = _PUNCTUATION_RE.sub("", token.lower())
    if len(token) < 2 or token in TECHNICAL_STOPWORDS:
        return None
    return token


def tokenize_query(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.split():
        ntok = normalize_token(raw)
        if ntok is not None:
            tokens.append(ntok)
    return tokens


def load_index(index_path: Path, silent: bool = False) -> dict:
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if not silent:
            print(json.dumps({"error": f"Index not found at {index_path}. Run ingest.py first."}, indent=2))
        sys.exit(1)
    except json.JSONDecodeError as e:
        if not silent:
            print(json.dumps({"error": f"Invalid JSON in {index_path}: {e}"}, indent=2))
        sys.exit(1)


def bm25_search(
    query: str,
    index_data: dict,
    top_k: int = 5,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    filter_source: Optional[str] = None,
    filter_type: Optional[str] = None,
    parent_ids: Optional[Set[str]] = None,
) -> List[dict]:
    query_tokens = tokenize_query(query)
    if not query_tokens:
        return []

    idx = index_data["inverted_index"]
    posting_index = idx["index"]
    idf_map = idx["idf"]
    doc_lengths = idx["doc_lengths"]
    avg_dl = idx["stats"]["avg_dl"]
    chunks = index_data["chunks"]
    chunk_map = {c["chunk_id"]: c for c in chunks}

    scores: Dict[str, float] = {}

    for token in query_tokens:
        if token not in posting_index:
            continue
        token_idf = idf_map.get(token, 0.0)
        for posting in posting_index[token]:
            cid = posting["chunk_id"]
            chunk = chunk_map.get(cid)
            if chunk is None or not is_meaningful_chunk(chunk, min_tokens):
                continue
            if parent_ids and cid in parent_ids:
                continue
            if filter_source and chunk.get("source_document") != filter_source:
                continue
            if filter_type and chunk.get("element_type") != filter_type:
                continue
            tf = posting["tf"]
            dl = doc_lengths.get(cid, 0)
            tf_norm = (tf * (BM25_K1 + 1)) / (
                tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            )
            scores[cid] = scores.get(cid, 0.0) + token_idf * tf_norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for cid, score in ranked:
        chunk = chunk_map.get(cid)
        if chunk is None:
            continue
        results.append(
            {
                "score": round(score, 4),
                "chunk_id": cid,
                "text": chunk["text"],
                "element_type": chunk["element_type"],
                "parent_heading": chunk["parent_heading"],
                "document_hierarchy_level": chunk["document_hierarchy_level"],
                "parent_id": chunk.get("parent_id"),
                "source_document": chunk["source_document"],
                "page_range": chunk.get("page_range", ""),
                "token_count": chunk["token_count"],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Lazy-loaded cross-encoder reranker
# ---------------------------------------------------------------------------

_RERANKER = None
_RERANKER_AVAILABLE = True  # Set to False if model fails to load


def get_reranker(model_name: str):
    global _RERANKER, _RERANKER_AVAILABLE
    if _RERANKER is None and _RERANKER_AVAILABLE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            _RERANKER_AVAILABLE = False
            return None
        with open(os.devnull, "w") as null:
            old_stderr = sys.stderr
            sys.stderr = null
            try:
                _RERANKER = CrossEncoder(model_name, local_files_only=True)
            except Exception:
                _RERANKER_AVAILABLE = False
                return None
            finally:
                sys.stderr = old_stderr
    return _RERANKER


def apply_reranker(
    query: str,
    candidates: List[dict],
    model_name: str,
    top_k: int,
) -> List[dict]:
    """Re-score candidates with a cross-encoder, return top_k.
    Falls back to returning top_k by hybrid score if model unavailable."""
    reranker = get_reranker(model_name)
    if reranker is None or len(candidates) <= top_k:
        return candidates[:top_k]

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

    # Attach rerank and hybrid scores
    for c, s in zip(candidates, scores):
        c["hybrid_score"] = c["score"]
        c["score"] = round(float(s), 4)
        c["reranked"] = True

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Lazy-loaded embedding model & FAISS index
# ---------------------------------------------------------------------------

_MODEL = None
_FAISS_INDEX = None
_CHUNK_IDS: Optional[List[str]] = None
_CID_TO_POS: Dict[str, int] = {}


def get_model(model_name: str):
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not installed.")
        # Redirect stderr to null during model loading (airtight silence)
        with open(os.devnull, "w") as null:
            old_stderr = sys.stderr
            sys.stderr = null
            try:
                _MODEL = SentenceTransformer(model_name, local_files_only=True)
            finally:
                sys.stderr = old_stderr
    return _MODEL


def get_faiss_index(faiss_path: Path, chunk_ids_path: Path):
    global _FAISS_INDEX, _CHUNK_IDS, _CID_TO_POS
    if _FAISS_INDEX is None:
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed.")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not chunk_ids_path.exists():
            raise FileNotFoundError(f"Chunk IDs mapping not found: {chunk_ids_path}")
        _FAISS_INDEX = faiss.read_index(str(faiss_path))
        with open(chunk_ids_path, "r", encoding="utf-8") as f:
            _CHUNK_IDS = json.load(f)
        _CID_TO_POS = {cid: i for i, cid in enumerate(_CHUNK_IDS)}
    return _FAISS_INDEX, _CHUNK_IDS


def vector_search(
    query: str,
    faiss_path: Path,
    chunk_ids_path: Path,
    model_name: str,
    top_k: int = 20,
    chunk_map: Optional[dict] = None,
    filter_source: Optional[str] = None,
    filter_type: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Dense semantic search via FAISS."""
    model = get_model(model_name)
    index, chunk_ids = get_faiss_index(faiss_path, chunk_ids_path)

    query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Over-fetch for better fusion quality (compensate for post-hoc filtering)
    search_k = min(top_k * 30 if (filter_source or filter_type) else top_k * 3, len(chunk_ids))
    distances, indices = index.search(query_embedding, search_k)

    results: List[Tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunk_ids):
            continue
        cid = chunk_ids[idx]
        if chunk_map and (filter_source or filter_type):
            chunk = chunk_map.get(cid)
            if chunk:
                if filter_source and chunk.get("source_document") != filter_source:
                    continue
                if filter_type and chunk.get("element_type") != filter_type:
                    continue
        results.append((cid, float(dist)))
        if len(results) >= top_k * 3:
            break
    return results


def _get_candidate_embeddings(chunk_ids: List[str]) -> Optional[np.ndarray]:
    """Look up FAISS embeddings for candidate chunk IDs.
    Returns None if FAISS is unavailable or any chunk is missing."""
    global _FAISS_INDEX, _CID_TO_POS
    if _FAISS_INDEX is None or not _CID_TO_POS:
        return None
    try:
        positions = [_CID_TO_POS.get(cid) for cid in chunk_ids]
        if None in positions:
            return None
        return np.array([_FAISS_INDEX.reconstruct(p) for p in positions], dtype=np.float32)
    except Exception:
        return None


def hybrid_fuse(
    bm25_results: List[dict],
    vector_results: List[Tuple[str, float]],
    index_data: dict,
    top_k: int = 5,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    parent_ids: Optional[Set[str]] = None,
) -> List[dict]:
    """Reciprocal Rank Fusion (RRF) of BM25 and dense scores.
    
    Uses standard RRF formula:
    RRF_Score = 1/(k + rank_bm25) + 1/(k + rank_semantic)
    where k = 60.
    """
    K = 60
    
    # Sort inputs to guarantee rank order (though they should already be sorted)
    sorted_bm25 = sorted(bm25_results, key=lambda x: x["score"], reverse=True)
    sorted_vector = sorted(vector_results, key=lambda x: x[1], reverse=True)

    # Build rank dictionaries (1-indexed)
    bm25_ranks: Dict[str, int] = {r["chunk_id"]: i + 1 for i, r in enumerate(sorted_bm25)}
    vector_ranks: Dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(sorted_vector)}

    # Union of all chunk IDs
    all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

    fused: Dict[str, float] = {}
    for cid in all_ids:
        rrf_bm25 = 1.0 / (K + bm25_ranks[cid]) if cid in bm25_ranks else 0.0
        rrf_vector = 1.0 / (K + vector_ranks[cid]) if cid in vector_ranks else 0.0
        fused[cid] = rrf_bm25 + rrf_vector

    # Sort by fused score descending
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # Reconstruct result dicts from BM25 results (they have full metadata)
    bm25_map = {r["chunk_id"]: r for r in bm25_results}
    chunk_map = {c["chunk_id"]: c for c in index_data["chunks"]}

    results = []
    for cid, score in ranked:
        # Skip heading-only chunks below threshold
        if cid in chunk_map and not is_meaningful_chunk(chunk_map[cid], min_tokens):
            continue
        if parent_ids and cid in parent_ids:
            continue
        if len(results) >= top_k:
            break
        if cid in bm25_map:
            r = dict(bm25_map[cid])
            r["score"] = round(score, 6) # RRF scores are small, use 6 precision
            results.append(r)
        elif cid in chunk_map:
            # Vector-only result: populate from chunk map
            c = chunk_map[cid]
            results.append(
                {
                    "score": round(score, 6),
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "element_type": c.get("element_type", "text"),
                    "parent_heading": c.get("parent_heading", ""),
                    "document_hierarchy_level": c.get("document_hierarchy_level", 1),
                    "parent_id": c.get("parent_id"),
                    "source_document": c.get("source_document", ""),
                    "page_range": c.get("page_range", ""),
                    "token_count": c.get("token_count", 0),
                }
            )
        else:
            # Should not happen, but provide a safe fallback
            results.append(
                {
                    "score": round(score, 6),
                    "chunk_id": cid,
                    "text": "",
                    "element_type": "text",
                    "parent_heading": "",
                    "document_hierarchy_level": 1,
                    "parent_id": None,
                    "source_document": "",
                    "page_range": "",
                    "token_count": 0,
                }
            )
    return results


def print_human_results(results: List[dict]) -> None:
    """Prints results in a clean, human-readable format."""
    print(f"\n[RAG] Found {len(results)} relevant technical chunks:\n")
    for i, res in enumerate(results, 1):
        source = res["source_document"]
        pages = res["page_range"]
        score = res["score"]
        heading = res["parent_heading"] or "Main"
        
        # Color-like markers for terminal
        print(f"[{i}] SOURCE: {source} (pg. {pages})")
        print(f"    SCORE: {score} | SECTION: {heading}")
        
        # Snippet: first 200 chars, cleaned up
        snippet = res["text"].strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        print(f"    CONTEXT: {snippet}\n")
    print("-" * 60)


def extract_prf_terms(results: List[dict], index_data: dict, top_n: int = 5, exclude_tokens: Optional[Set[str]] = None) -> List[str]:
    """Extract high-IDF terms from top results for query expansion."""
    idf_map = index_data["inverted_index"]["idf"]
    term_scores: Dict[str, float] = {}
    for r in results:
        tokens = tokenize_query(r["text"])
        for tok in tokens:
            if tok in idf_map and (not exclude_tokens or tok not in exclude_tokens):
                term_scores[tok] = term_scores.get(tok, 0) + idf_map[tok]
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_terms[:top_n]]


def apply_parent_child(results: List[dict], chunk_map: dict) -> List[dict]:
    expanded = []
    seen = set()
    for r in results:
        pid = r.get("parent_id")
        target_id = pid if pid else r["chunk_id"]
        if pid and pid in chunk_map:
            if target_id in seen:
                expanded.append(r)
                continue
            seen.add(target_id)
            parent_chunk = chunk_map[pid]
            r_copy = dict(r)
            r_copy["text"] = parent_chunk.get("text", "")
            r_copy["chunk_id"] = pid
            r_copy["token_count"] = parent_chunk.get("token_count", 0)
            r_copy["parent_id"] = parent_chunk.get("parent_id")
            expanded.append(r_copy)
        else:
            expanded.append(r)
    return expanded


def apply_dedup(
    results: List[dict],
    top_k: int,
    similarity_threshold: float = 0.90,
) -> List[dict]:
    """Remove near-duplicate chunks, keeping the highest-scored first.

    Uses FAISS embedding cosine similarity. Two chunks with cosine
    similarity above threshold are considered duplicates and the
    lower-scored one is skipped. Falls back to returning the first
    top_k if FAISS is unavailable.

    Unlike MMR, this never promotes off-topic content — it only
    removes redundancy within already-relevant candidates.
    """
    n = len(results)
    if n <= top_k:
        return results

    chunk_ids = [r.get("chunk_id", "") for r in results]
    embeddings = _get_candidate_embeddings(chunk_ids)

    if embeddings is not None:
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms < 1e-10, 1.0, emb_norms)
        emb = np.asarray(embeddings / emb_norms, dtype=np.float32)

        kept: List[int] = [0]
        for i in range(1, n):
            sims = np.dot(emb[kept], emb[i])
            if float(np.max(sims)) < similarity_threshold:
                kept.append(i)
            if len(kept) >= top_k:
                break
        return [results[i] for i in kept]

    return results[:top_k]


def apply_mmr(
    results: List[dict],
    top_k: int,
    lambda_param: float = 0.5,
) -> List[dict]:
    """Maximal Marginal Relevance — balance relevance with diversity.

    Uses FAISS embedding cosine similarity for semantic diversity.
    Falls back to Jaccard token-overlap if FAISS is unavailable.

    Scores are min-max normalized to [0,1] so lambda is meaningful
    regardless of score distribution (RRF, cross-encoder logits, etc.).

    Args:
        results: Input list of scored results (must be sorted desc by score).
        top_k: Number of results to return after MMR.
        lambda_param: Relevance vs diversity (1.0 = pure relevance).
    """
    n = len(results)
    if n <= top_k:
        return results

    # ── Min-max normalize scores to [0, 1] ──
    raw_scores = [r.get("score", 0.0) for r in results]
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s > min_s:
        norm_scores = np.array([(s - min_s) / (max_s - min_s) for s in raw_scores], dtype=np.float32)
    else:
        norm_scores = np.ones(n, dtype=np.float32)

    # ── Semantic similarity via FAISS embeddings ──
    chunk_ids = [r.get("chunk_id", "") for r in results]
    embeddings = _get_candidate_embeddings(chunk_ids)

    if embeddings is not None and embeddings.shape[0] == n:
        # Normalize embeddings for cosine similarity (dot product)
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms < 1e-10, 1.0, emb_norms)
        emb = np.asarray(embeddings / emb_norms, dtype=np.float32)

        selected: List[int] = [0]
        max_sim = np.zeros(n, dtype=np.float32)

        for _ in range(top_k - 1):
            remaining = np.ones(n, dtype=bool)
            remaining[selected] = False
            if not remaining.any():
                break

            mmr = lambda_param * norm_scores - (1.0 - lambda_param) * max_sim
            mmr[~remaining] = -np.inf
            best = int(np.argmax(mmr))

            selected.append(best)
            # Update: max similarity of each candidate to any selected item
            new_sim = np.dot(emb[best : best + 1], emb.T).ravel()
            np.maximum(max_sim, new_sim, out=max_sim)

        return [results[i] for i in selected]

    # ── Lexical fallback: Jaccard similarity ──
    token_sets = [set(tokenize_query(r.get("text", ""))) for r in results]

    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    selected = [0]
    flags = [False] * n
    flags[0] = True

    while len(selected) < top_k:
        best_idx = -1
        best_mmr = float("-inf")

        for i in range(n):
            if flags[i]:
                continue
            max_sim = max((_jaccard(token_sets[i], token_sets[j]) for j in selected), default=0.0)
            mmr = float(lambda_param) * float(norm_scores[i]) - (1.0 - float(lambda_param)) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx == -1:
            break
        selected.append(best_idx)
        flags[best_idx] = True

    return [results[i] for i in selected]


def run_single_query(
    query_text: str,
    args,
    index_data,
) -> List[dict]:
    chunk_map = {c["chunk_id"]: c for c in index_data["chunks"]}
    p_ids = get_parent_ids(index_data)
    original_query = query_text  # Preserve clean query for reranker anchor

    fs, ft = getattr(args, "filter_source", None), getattr(args, "filter_type", None)

    # Build query list (PRF appends expanded variant alongside original)
    queries_to_run = [query_text]
    if getattr(args, "prf", False):
        prf_results = bm25_search(query_text, index_data, top_k=3, filter_source=fs, filter_type=ft, parent_ids=p_ids)
        if prf_results:
            exclude = set(tokenize_query(query_text))
            terms = extract_prf_terms(prf_results, index_data, exclude_tokens=exclude)
            queries_to_run.append(query_text + " " + " ".join(terms))

    rerank = getattr(args, "rerank", False)
    min_tokens = getattr(args, "min_tokens", DEFAULT_MIN_TOKENS)
    search_k = args.top_k * 3
    fuse_k = args.top_k * 15 if rerank else args.top_k

    # Run all queries and merge by best score per chunk_id
    all_by_chunk: Dict[str, dict] = {}
    for q in queries_to_run:
        if args.mode == "bm25-only":
            results = bm25_search(q, index_data, top_k=fuse_k, min_tokens=min_tokens, filter_source=fs, filter_type=ft, parent_ids=p_ids)
        elif args.mode == "vector-only":
            vresults = vector_search(q, args.faiss_index, args.chunk_ids, args.model, top_k=fuse_k, chunk_map=chunk_map, filter_source=fs, filter_type=ft)
            results = hybrid_fuse([], vresults, index_data, top_k=fuse_k, min_tokens=min_tokens, parent_ids=p_ids)
        else:
            bm25_results = bm25_search(q, index_data, top_k=search_k, min_tokens=min_tokens, filter_source=fs, filter_type=ft, parent_ids=p_ids)
            vresults = vector_search(q, args.faiss_index, args.chunk_ids, args.model, top_k=search_k, chunk_map=chunk_map, filter_source=fs, filter_type=ft)
            results = hybrid_fuse(bm25_results, vresults, index_data, top_k=fuse_k, min_tokens=min_tokens, parent_ids=p_ids)

        for r in results:
            cid = r["chunk_id"]
            if cid not in all_by_chunk or r["score"] > all_by_chunk[cid]["score"]:
                all_by_chunk[cid] = r

    merged = sorted(all_by_chunk.values(), key=lambda x: x["score"], reverse=True)[:fuse_k]

    # Rerank BEFORE parent-child (cross-encoder works best on small chunks)
    if rerank and len(merged) > args.top_k:
        merged = apply_reranker(original_query, merged, args.rerank_model, fuse_k)

    # Parent-child expansion AFTER reranking
    if getattr(args, "parent_child", False):
        merged = apply_parent_child(merged, chunk_map)

    mmr_lambda = getattr(args, "mmr_lambda", None)
    if mmr_lambda is not None:
        merged = apply_mmr(merged, args.top_k, lambda_param=mmr_lambda)
    else:
        merged = apply_dedup(merged, args.top_k)

    return merged[:args.top_k]


def run_batch_queries(queries: List[str], args, index_data) -> List[dict]:
    chunk_map = {c["chunk_id"]: c for c in index_data["chunks"]}
    p_ids = get_parent_ids(index_data)

    fs, ft = getattr(args, "filter_source", None), getattr(args, "filter_type", None)

    queries_to_run = list(queries)
    if getattr(args, "prf", False) and queries:
        prf_results = bm25_search(queries[0], index_data, top_k=3, filter_source=fs, filter_type=ft, parent_ids=p_ids)
        if prf_results:
            exclude = set(tokenize_query(queries[0]))
            terms = extract_prf_terms(prf_results, index_data, exclude_tokens=exclude)
            queries_to_run.append(queries[0] + " " + " ".join(terms))

    rerank = getattr(args, "rerank", False)
    min_tokens = getattr(args, "min_tokens", DEFAULT_MIN_TOKENS)
    search_k = args.top_k * 3
    fuse_k = args.top_k * 15 if rerank else args.top_k

    all_by_chunk: Dict[str, dict] = {}
    for q in queries_to_run:
        if args.mode == "bm25-only":
            results = bm25_search(q, index_data, top_k=fuse_k, min_tokens=min_tokens, filter_source=fs, filter_type=ft, parent_ids=p_ids)
        elif args.mode == "vector-only":
            vresults = vector_search(q, args.faiss_index, args.chunk_ids, args.model, top_k=fuse_k, chunk_map=chunk_map, filter_source=fs, filter_type=ft)
            results = hybrid_fuse([], vresults, index_data, top_k=fuse_k, min_tokens=min_tokens, parent_ids=p_ids)
        else:
            bm25_results = bm25_search(q, index_data, top_k=search_k, min_tokens=min_tokens, filter_source=fs, filter_type=ft, parent_ids=p_ids)
            vresults = vector_search(q, args.faiss_index, args.chunk_ids, args.model, top_k=search_k, chunk_map=chunk_map, filter_source=fs, filter_type=ft)
            results = hybrid_fuse(bm25_results, vresults, index_data, top_k=fuse_k, min_tokens=min_tokens, parent_ids=p_ids)

        for r in results:
            cid = r["chunk_id"]
            if cid not in all_by_chunk or r["score"] > all_by_chunk[cid]["score"]:
                all_by_chunk[cid] = r

    merged = [r for r in sorted(all_by_chunk.values(), key=lambda x: x["score"], reverse=True)
              if r.get("token_count", 0) >= min_tokens or r.get("element_type") != "heading"]

    # Rerank BEFORE parent-child (cross-encoder works best on small chunks)
    if rerank and len(merged) > args.top_k:
        candidates = merged[:fuse_k]
        merged = apply_reranker(queries[0], candidates, args.rerank_model, fuse_k)

    # Parent-child expansion AFTER reranking
    if getattr(args, "parent_child", False):
        merged = apply_parent_child(merged, chunk_map)

    mmr_lambda = getattr(args, "mmr_lambda", None)
    if mmr_lambda is not None:
        merged = apply_mmr(merged, args.top_k, lambda_param=mmr_lambda)
    else:
        merged = apply_dedup(merged, args.top_k)

    return merged[:args.top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 search over the ingested RAG index.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Search terms. Pass multiple queries for batch mode (results merged & deduplicated).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help=f"Minimum token count for heading chunks to be included (default: {DEFAULT_MIN_TOKENS}). Shorter headings are filtered out as noise.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Path to rag_output.json (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON for programmatic use.",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25-only", "vector-only"],
        default="hybrid",
        help="Search mode (default: hybrid).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=DEFAULT_FAISS_PATH,
        help=f"Path to FAISS index (default: {DEFAULT_FAISS_PATH}).",
    )
    parser.add_argument(
        "--chunk-ids",
        type=Path,
        default=DEFAULT_CHUNK_IDS_PATH,
        help=f"Path to chunk ID mapping (default: {DEFAULT_CHUNK_IDS_PATH}).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranker after hybrid fusion (default: off).",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=DEFAULT_RERANK_MODEL,
        help=f"Cross-encoder model for reranking (default: {DEFAULT_RERANK_MODEL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write results to file instead of stdout (silent operation).",
    )
    parser.add_argument(
        "--parent-child",
        action="store_true",
        help="Enable Small-to-Big retrieval (return parent chunk for high-scoring children).",
    )
    parser.add_argument(
        "--filter-source",
        type=str,
        default=None,
        help="Filter results by exact source_document name.",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        default=None,
        help="Filter results by exact element_type (e.g., table, text, list).",
    )
    parser.add_argument(
        "--prf",
        action="store_true",
        help="Enable Pseudo-Relevance Feedback (offline query expansion using top results).",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=None,
        help="Opt-in MMR diversity reranking. 1.0 = pure relevance, 0.0 = pure diversity. When not set, simple semantic dedup is used instead (safer, no off-topic promotion).",
    )
    args = parser.parse_args()

    try:
        index_data = load_index(args.index, silent=args.out is not None)
    except SystemExit:
        return

    queries = args.query
    if len(queries) == 1:
        results = run_single_query(queries[0], args, index_data)
    else:
        results = run_batch_queries(queries, args, index_data)

    if not results:
        sys.exit(0)

    # Annotate with full source file paths for quick lookup
    source_dir = BASE_DIR / "in"
    for r in results:
        r["file_path"] = str(source_dir / r["source_document"])

    _RERANKER_ACTIVE = _RERANKER is not None
    output = {
        "reranked": _RERANKER_ACTIVE and getattr(args, "rerank", False),
        "results": results,
    }
    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()