"""
retrieval.py - RAG retrieval layer for the health chatbot.

Flow: query -> BGE-M3 embedding -> Supabase RPC (match_document_chunks)
    -> hydrate raw text from local Markdown -> list of RetrievedChunk

Env vars:
    SUPABASE_URL, SUPABASE_SERVICE_KEY
    KNOWLEDGE_BASE_DIR  (default: ./knowledge_base)
    MATCH_THRESHOLD     (default: 0.25)
    MATCH_COUNT         (default: 3)
"""
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

import yaml
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

KB_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.25"))
TOP_K = int(os.getenv("MATCH_COUNT", "3"))

MAX_PROMPT_CHARS = 2000
# Health queries are sensitive: by default only a hash + vector are stored.
STORE_QUERY_TEXT = os.getenv("STORE_QUERY_TEXT", "false").lower() == "true"
log = logging.getLogger(__name__)

METADATA_RE = re.compile(r"<!--\s*DOCUMENT_METADATA(.*?)-->", re.DOTALL)


@dataclass
class RetrievedChunk:
    id: str
    document_id: str
    section_title: str | None
    subsection_title: str | None
    similarity: float
    text: str


# ---------- Singletons ----------
@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3")


@lru_cache(maxsize=1)
def _supabase():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


# ---------- Local knowledge base ----------
def _parse_doc(path: Path) -> tuple[dict, dict]:
    """Return (metadata, sections) where sections maps (h2, h3|None) -> text."""
    raw = path.read_text(encoding="utf-8")
    m = METADATA_RE.search(raw)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = METADATA_RE.sub("", raw, count=1)

    sections: dict[tuple, list[str]] = {}
    h2 = h3 = None
    for line in body.splitlines():
        if line.startswith("### "):
            h3 = line[4:].strip()
        elif line.startswith("## "):
            h2, h3 = line[3:].strip(), None
        elif line.startswith("# "):
            continue
        else:
            sections.setdefault((h2, h3), []).append(line)
    return meta or {}, {k: "\n".join(v).strip() for k, v in sections.items()}


@lru_cache(maxsize=1)
def _kb_index() -> dict[str, dict]:
    """document_id -> {meta, sections}. Built once from knowledge_base/*.md."""
    index = {}
    for path in KB_DIR.glob("*.md"):
        meta, sections = _parse_doc(path)
        doc_id = meta.get("document_id") or path.stem
        index[str(doc_id)] = {"meta": meta, "sections": sections}
    return index


def _hydrate(document_id: str, section: str | None, subsection: str | None) -> str:
    doc = _kb_index().get(document_id)
    if not doc:
        return ""
    secs = doc["sections"]
    if (section, subsection) in secs:
        text = secs[(section, subsection)]
    else:
        # Fallback: everything under the H2 section
        text = "\n\n".join(t for (h2, _), t in secs.items() if h2 == section)
    disease = doc["meta"].get("disease_name", document_id)
    crumb = " > ".join(p for p in (section, subsection) if p)
    return f"[Disease: {disease} | Section: {crumb}]\n{text}".strip()


# ---------- Public API ----------
def read_prompt(raw: str) -> str:
    """Step 1: read and clean the user's input prompt."""
    if not raw or not raw.strip():
        raise ValueError("Empty prompt")
    prompt = re.sub(r"\s+", " ", raw).strip()
    return prompt[:MAX_PROMPT_CHARS]


def _query_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.lower().encode("utf-8")).hexdigest()


def _load_cached_embedding(qhash: str) -> list[float] | None:
    try:
        res = (_supabase().table("query_embeddings").select("embedding")
                .eq("query_hash", qhash).limit(1).execute())
        if res.data:
            emb = res.data[0]["embedding"]
            return json.loads(emb) if isinstance(emb, str) else emb
    except Exception as e:  # cache failure must never break retrieval
        log.warning("Query embedding cache read failed: %s", e)
    return None


def _store_embedding(qhash: str, prompt: str, vec: list[float]) -> None:
    try:
        _supabase().table("query_embeddings").upsert({
            "query_hash": qhash,
            "query_text": prompt if STORE_QUERY_TEXT else None,
            "embedding": vec,
        }, on_conflict="query_hash").execute()
    except Exception as e:
        log.warning("Query embedding store failed: %s", e)


def embed_query(raw_prompt: str) -> list[float]:
    """Step 2: prompt -> normalized 1024-dim BGE-M3 vector.
    Embeddings are stored in the separate `query_embeddings` table and reused
    on repeat questions."""
    prompt = read_prompt(raw_prompt)
    qhash = _query_hash(prompt)
    cached = _load_cached_embedding(qhash)
    if cached:
        return cached
    vec = _model().encode(prompt, normalize_embeddings=True).tolist()
    _store_embedding(qhash, prompt, vec)
    return vec


def retrieve(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> list[RetrievedChunk]:
    """Step 3: cosine-similarity search (HNSW) in Supabase, then hydrate text."""
    resp = _supabase().rpc(
        "match_document_chunks",
        {"query_embedding": embed_query(query), "match_threshold": threshold, "match_count": top_k},
    ).execute()

    chunks, seen = [], set()
    for row in resp.data or []:
        key = (row["document_id"], row["section_title"], row["subsection_title"])
        if key in seen:  # chunks from the same subsection hydrate to identical text
            continue
        seen.add(key)
        text = _hydrate(*key)
        if text:
            chunks.append(RetrievedChunk(
                id=row["id"], document_id=row["document_id"],
                section_title=row["section_title"], subsection_title=row["subsection_title"],
                similarity=row["similarity"], text=text,
            ))
    return chunks


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the treatment for chloroquine-resistant malaria?"
    for c in retrieve(q):
        print(f"\n--- {c.similarity:.3f} | {c.section_title} > {c.subsection_title} ---")
        print(c.text[:400])
