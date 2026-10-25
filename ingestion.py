#!/usr/bin/env python3
"""
ingest_knowledge_base.py

Reads every disease markdown file in a folder, chunks each one on its
markdown headers, embeds the chunks with BGE-M3 (1024 dims), and inserts
them along with the full chunk text directly into the `document_chunks` 
table in Supabase via the Supabase Python SDK.

Usage:
    python ingest_knowledge_base.py /path/to/knowledge_base_folder

Requires a .env file with:
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
"""

import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MIN_WORDS = 15          # pieces shorter than this always get merged into a neighbor
MAX_WORDS = 230         # ~300 tokens; pieces longer than this get split further
PLACEHOLDER_RATIO = 0.5 # treat as low-value if > 50% sentences are placeholders
PLACEHOLDER_MARKERS = ("not specified in the provided source", "not independently verified")
EMBED_MODEL = "BAAI/bge-m3"

METADATA_BLOCK_RE = re.compile(r"<!--\s*DOCUMENT_METADATA\s*(.*?)-->", re.DOTALL)


# ---------------------------------------------------------------------
# Step 1: Parse the metadata comment block
# ---------------------------------------------------------------------

def parse_metadata(raw_text: str) -> dict:
    """Pulls the <!-- DOCUMENT_METADATA ... --> block out of the file and
    parses its key: value lines. Returns {} if no block is found."""
    match = METADATA_BLOCK_RE.search(raw_text)
    if not match:
        return {}
    block = match.group(1).strip()
    try:
        data = yaml.safe_load(block)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass

    # Fallback: line-by-line parsing
    data = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip()
    return data


def strip_metadata_block(raw_text: str) -> str:
    return METADATA_BLOCK_RE.sub("", raw_text).strip()


def derive_disease_id(metadata: dict) -> str:
    doc_id = str(metadata.get("document_id", "")).strip()
    if doc_id:
        return doc_id
    name = str(metadata.get("disease_name", "")).strip().lower()
    return f"disease_{re.sub(r'[^a-z0-9]+', '_', name).strip('_')}"


# ---------------------------------------------------------------------
# Step 2: Header splitting, placeholder filtering, merging, and cleaning
# ---------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def clean_chunk_text(text: str) -> str:
    """Ensures chunk text ends cleanly on a complete sentence or terminal punctuation."""
    text = text.strip()
    if not text:
        return text
    # If string ends mid-word or without terminal punctuation, trim to last valid sentence boundary
    if text[-1] not in ".!?\":'":
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct > 0:
            return text[: last_punct + 1].strip()
    return text


def is_low_value(text: str) -> bool:
    if word_count(text) < MIN_WORDS:
        return True
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return True
    flagged = sum(1 for s in sentences if any(m in s.lower() for m in PLACEHOLDER_MARKERS))
    return (flagged / len(sentences)) > PLACEHOLDER_RATIO


def header_split(body_text: str):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Disease"), ("##", "Main_Section"), ("###", "Subsection")],
        strip_headers=True,
    )
    return splitter.split_text(body_text)


def split_oversized(text: str) -> list:
    """Splits oversized sections while prioritizing natural sentence endings."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        chunk_size=MAX_WORDS * 6,
        chunk_overlap=30,
    )
    return splitter.split_text(text)


def group_and_merge(docs) -> list:
    groups = []
    current_section = object()
    for doc in docs:
        section = doc.metadata.get("Main_Section", "")
        if section != current_section:
            groups.append({"section": section, "pieces": []})
            current_section = section
        groups[-1]["pieces"].append(doc)

    final_chunks = []
    for group in groups:
        expanded = []
        for doc in group["pieces"]:
            text = doc.page_content.strip()
            if not text:
                continue
            if word_count(text) > MAX_WORDS:
                for sub_text in split_oversized(text):
                    expanded.append({"subsection": doc.metadata.get("Subsection", ""), "text": sub_text})
            else:
                expanded.append({"subsection": doc.metadata.get("Subsection", ""), "text": text})

        buffer = None
        for piece in expanded:
            if buffer is None:
                buffer = {"subsections": [piece["subsection"]] if piece["subsection"] else [], "text": piece["text"]}
                continue
            if is_low_value(buffer["text"]) or is_low_value(piece["text"]):
                buffer["text"] += "\n\n" + piece["text"]
                if piece["subsection"] and piece["subsection"] not in buffer["subsections"]:
                    buffer["subsections"].append(piece["subsection"])
            else:
                final_chunks.append({
                    "section": group["section"],
                    "subsections": buffer["subsections"],
                    "text": clean_chunk_text(buffer["text"]),
                })
                buffer = {"subsections": [piece["subsection"]] if piece["subsection"] else [], "text": piece["text"]}
        if buffer is not None:
            final_chunks.append({
                "section": group["section"],
                "subsections": buffer["subsections"],
                "text": clean_chunk_text(buffer["text"]),
            })

    return final_chunks


# ---------------------------------------------------------------------
# Step 3: Build chunk records matching Supabase document_chunks table
# ---------------------------------------------------------------------

def build_chunk_records(file_path: Path, metadata: dict, chunks: list) -> list:
    doc_id = derive_disease_id(metadata)
    disease_name = str(metadata.get("disease_name", doc_id))

    records = []
    for i, chunk in enumerate(chunks):
        subsections_str = "; ".join(chunk["subsections"]) if chunk["subsections"] else ""

        breadcrumb = f"Disease: {disease_name} | Section: {chunk['section']}"
        if subsections_str:
            breadcrumb += f" > {subsections_str}"
        text_with_breadcrumb = f"[{breadcrumb}]\n\n{chunk['text']}"

        records.append({
            "id": f"{doc_id}_chunk_{i}",
            "document_id": str(doc_id),
            "section_title": str(chunk["section"]),
            "subsection_title": str(subsections_str),
            "text": text_with_breadcrumb,  # Stored directly in Supabase Postgres table
        })
    return records


def insert_records(supabase: Client, records: list, embedder: SentenceTransformer):
    if not records:
        return

    # Generate embeddings locally using BGE-M3
    texts_to_embed = [r["text"] for r in records]
    embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True)

    # Attach vectors to records
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding.tolist()

    # Insert IDs, metadata, text, and embeddings into Supabase
    supabase.table("document_chunks").upsert(records).execute()


# ---------------------------------------------------------------------
# Step 4: Process File (Pure Supabase)
# ---------------------------------------------------------------------

def process_file(file_path: Path, supabase: Client, embedder: SentenceTransformer) -> dict:
    raw_text = file_path.read_text(encoding="utf-8")
    metadata = parse_metadata(raw_text)
    body_text = strip_metadata_block(raw_text)

    docs = header_split(body_text)
    chunks = group_and_merge(docs)
    records = build_chunk_records(file_path, metadata, chunks)

    if not records:
        return {"file": file_path.name, "status": "skipped", "reason": "No valid content chunks extracted"}

    insert_records(supabase, records, embedder)

    unverified = metadata.get("verification_status") == "NOT_INDEPENDENTLY_VERIFIED"
    return {
        "file": file_path.name,
        "status": "ok",
        "chunks": len(records),
        "disease_id": records[0]["document_id"],
        "unverified": unverified,
    }


def main(folder: str):
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in your .env file.")
        sys.exit(1)

    md_files = sorted(Path(folder).glob("*.md"))
    if not md_files:
        print(f"No .md files found in {folder}")
        return

    print("Initializing Supabase client...")
    supabase: Client = create_client(supabase_url, supabase_key)

    print(f"Loading {EMBED_MODEL} model (1024-dim vectors)...")
    embedder = SentenceTransformer(EMBED_MODEL)

    results = []
    for file_path in tqdm(md_files, desc="Ingesting to Supabase"):
        result = process_file(file_path, supabase, embedder)
        results.append(result)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    unverified = [r for r in ok if r.get("unverified")]

    print(f"\nDone: {len(ok)} file(s) ingested into Supabase, {len(skipped)} skipped.")
    print(f"Total chunks stored: {sum(r['chunks'] for r in ok)}")
    if unverified:
        print(
            f"\nWARNING: {len(unverified)} file(s) are NOT_INDEPENDENTLY_VERIFIED "
            f"(still ingested and chunked): {[r['file'] for r in unverified]}"
        )
    if skipped:
        print("\nSkipped:")
        for r in skipped:
            print(f"  {r['file']}: {r['reason']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_knowledge_base.py /path/to/knowledge_base_folder")
        sys.exit(1)
    main(sys.argv[1])