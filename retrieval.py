"""
retrieval.py - Two-Stage RAG retrieval layer for the health chatbot.

Pipeline:
  1. Candidate Fetch (Vector Search): Query -> BGE-M3 embedding -> Supabase RPC (Top 10)
  2. Cross-Encoder Re-Ranking: BAAI/bge-reranker-v2-m3 scores (Query, Candidate) pairs -> Top 3
  3. Context Hydration: Top 3 IDs -> Sub-millisecond Redis key lookup -> RetrievedChunk list
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import redis
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from supabase import create_client

load_dotenv()

# Search defaults
CANDIDATE_FETCH_COUNT = int(os.getenv("CANDIDATE_FETCH_COUNT", "10"))  # Vector candidates
FINAL_TOP_K = int(os.getenv("MATCH_COUNT", "3"))                      # Re-ranked output
THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.20"))
MAX_PROMPT_CHARS = 2000
STORE_QUERY_TEXT = os.getenv("STORE_QUERY_TEXT", "false").lower() == "true"

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    section_title: str | None
    subsection_title: str | None
    similarity: float  # Re-ranker cross-encoder score
    text: str


# ---------- Singletons & Connection Pooling ----------

@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    log.info("Loading BAAI/bge-m3 embedding model...")
    return SentenceTransformer("BAAI/bge-m3")


@lru_cache(maxsize=1)
def _reranker_model() -> CrossEncoder:
    log.info("Loading BAAI/bge-reranker-v2-m3 cross-encoder...")
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=1)
def _supabase():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


@lru_cache(maxsize=1)
def _redis():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", None)
    return redis.Redis(
        host=host, port=port, password=password, decode_responses=True
    )


# ---------- Query & Embedding Caching ----------

def read_prompt(raw: str) -> str:
    """Sanitizes user input string."""
    if not raw or not raw.strip():
        raise ValueError("Empty prompt")
    prompt = re.sub(r"\s+", " ", raw).strip()
    return prompt[:MAX_PROMPT_CHARS]


def _query_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.lower().encode("utf-8")).hexdigest()


def _load_cached_embedding(qhash: str) -> list[float] | None:
    try:
        res = (
            _supabase()
            .table("query_embeddings")
            .select("embedding")
            .eq("query_hash", qhash)
            .limit(1)
            .execute()
        )
        if res.data:
            emb = res.data[0]["embedding"]
            return json.loads(emb) if isinstance(emb, str) else emb
    except Exception as e:
        log.warning("Query embedding cache read failed: %s", e)
    return None


def _store_embedding(qhash: str, prompt: str, vec: list[float]) -> None:
    try:
        _supabase().table("query_embeddings").upsert(
            {
                "query_hash": qhash,
                "query_text": prompt if STORE_QUERY_TEXT else None,
                "embedding": vec,
            },
            on_conflict="query_hash",
        ).execute()
    except Exception as e:
        log.warning("Query embedding store failed: %s", e)


def embed_query(raw_prompt: str) -> list[float]:
    """Generate 1024-dim BGE-M3 query vector with Supabase embedding cache."""
    prompt = read_prompt(raw_prompt)
    qhash = _query_hash(prompt)
    cached = _load_cached_embedding(qhash)
    if cached:
        return cached

    vec = _embedding_model().encode(prompt, normalize_embeddings=True).tolist()
    _store_embedding(qhash, prompt, vec)
    return vec


# ---------- Hydration & Re-ranking ----------

def hydrate_from_redis(chunk_ids: list[str]) -> list[str | None]:
    """Sub-millisecond batch text hydration via Redis MGET."""
    if not chunk_ids:
        return []
    try:
        return _redis().mget(chunk_ids)
    except Exception as e:
        log.error("Redis hydration failed: %s", e)
        return [None] * len(chunk_ids)


def rerank_candidates(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
    """Re-scores candidate chunks using BAAI/bge-reranker-v2-m3."""
    if not candidates:
        return []

    reranker = _reranker_model()
    chunk_ids = [c["id"] for c in candidates]
    
    # Batch hydrate candidate raw texts for cross-encoder evaluation
    raw_texts = hydrate_from_redis(chunk_ids)

    valid_pairs = []
    valid_candidates = []
    
    for candidate, text in zip(candidates, raw_texts):
        if text:
            candidate["text"] = text
            valid_pairs.append([query, text])
            valid_candidates.append(candidate)

    if not valid_pairs:
        return []

    # Calculate cross-encoder cross-attention scores
    scores = reranker.predict(valid_pairs)

    # Attach re-ranker scores back to candidates
    for candidate, score in zip(valid_candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort candidates descending by Cross-Encoder score
    reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ---------- Core Public API ----------

def retrieve(
    query: str, 
    fetch_k: int = CANDIDATE_FETCH_COUNT, 
    final_k: int = FINAL_TOP_K, 
    threshold: float = THRESHOLD
) -> list[RetrievedChunk]:
    """
    Executes a 2-Stage Retrieval:
    1. Vector similarity search in Supabase (HNSW Index) -> Fetch top candidate set.
    2. Cross-Encoder Re-Ranking (BGE-Reranker-v2-m3) -> Return final Top K chunks.
    """
    query_vector = embed_query(query)

    # Stage 1: Vector Search candidate recall (Top 10)
    resp = _supabase().rpc(
        "match_document_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": threshold,
            "match_count": fetch_k,
        },
    ).execute()

    matched_candidates = resp.data or []
    if not matched_candidates:
        return []

    # Stage 2: Cross-Encoder re-ranking & Redis text hydration
    top_ranked = rerank_candidates(query, matched_candidates, top_k=final_k)

    # Build final response objects
    results = []
    for item in top_ranked:
        results.append(
            RetrievedChunk(
                id=item["id"],
                document_id=item["document_id"],
                section_title=item.get("section_title"),
                subsection_title=item.get("subsection_title"),
                similarity=item.get("rerank_score", 0.0),
                text=item["text"],
            )
        )

    return results


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the treatment for chloroquine-resistant malaria?"
    print(f"Executing 2-stage retrieval for query: '{q}'\n")

    results = retrieve(q)
    for c in results:
        print(f"--- [Re-rank Score: {c.similarity:.4f}] | {c.id} | {c.section_title} > {c.subsection_title} ---")
        print(c.text)
        print("\n" + "=" * 60 + "\n")