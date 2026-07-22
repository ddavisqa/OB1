#!/usr/bin/env python3
"""
Phase 3a: Seed OB1 memory from the wiki.
Walks ~/wiki/*.md, parses frontmatter, embeds via Ollama, stores in ChromaDB ob1_memory.
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

WIKI = os.path.expanduser("~/wiki")
CHROMA = "http://localhost:8000"
TENANT = "default_tenant"
DATABASE = "default_database"
COLL = "ob1_memory"
COLL_BASE = f"{CHROMA}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
OLLAMA = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
BATCH_SIZE = 10  # docs per batch

SKIP_FILES = {"index.md", "log.md", "SCHEMA.md", "README.md"}
STATS = {"ingested": 0, "skipped": 0, "errors": 0, "total": 0}

def log(msg):
    print(f"  {msg}")

def http_post(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}

def parse_frontmatter(text):
    """Extract YAML frontmatter and body from a .md file."""
    fm = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw = parts[1]
            body = parts[2].strip()
            for line in raw.strip().split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    fm[key] = val
    return fm, body

def strip_markdown(text):
    """Strip markdown formatting to plaintext for embedding."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove wikilinks but keep text
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    # Remove markdown links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"(\*{1,3}|_{1,3})", "", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bullet markers
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def embed_text(text):
    """Get embedding vector via Ollama."""
    # Truncate to 8000 chars for Ollama context limit
    d = json.dumps({"model": OLLAMA_MODEL, "prompt": text[:8000]}).encode()
    req = urllib.request.Request(OLLAMA, data=d, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("embedding", [])

def get_collection_id():
    """Get or create the ob1_memory collection. Returns (base_url, collection_id)."""
    try:
        resp = json.loads(urllib.request.urlopen(
            f"{COLL_BASE}?name={COLL}", timeout=5).read())
        matched = [c for c in (resp if isinstance(resp, list) else [])
                   if c.get("name") == COLL]
        if matched:
            return matched[0]["id"]
    except Exception:
        pass
    # Create
    c = http_post(COLL_BASE, {"name": COLL, "metadata": {"hnsw:space": "cosine"}})
    return c.get("id", "")

def embed_content_type(fm, wiki_type=""):
    """Map wiki type to OB1 content_type."""
    t = fm.get("type", wiki_type).lower()
    mapping = {
        "entity": "entity",
        "concept": "concept",
        "comparison": "comparison",
    }
    return mapping.get(t, "wiki_page")

def main():
    global STATS
    log(f"Seeding OB1 from {WIKI}")
    log("")

    # Resolve collection
    log("Resolving ChromaDB collection...")
    cid = get_collection_id()
    if not cid:
        log("ERROR: Could not create/find collection")
        sys.exit(1)
    log(f"  Collection ID: {cid[:12]}...")

    # Collect all .md files
    files = []
    for root, dirs, fnames in os.walk(WIKI):
        # Skip hidden dirs, raw/, logs/
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("raw", "logs", ".git")]
        for fname in fnames:
            if fname.endswith(".md") and fname not in SKIP_FILES:
                rel = os.path.relpath(os.path.join(root, fname), WIKI)
                files.append((rel, os.path.join(root, fname)))

    STATS["total"] = len(files)
    log(f"\nFound {len(files)} content pages to ingest")
    log("")

    # Batch processing
    batch_embeddings = []
    batch_metadatas = []
    batch_documents = []
    batch_ids = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for idx, (rel_path, full_path) in enumerate(files):
        try:
            text = Path(full_path).read_text(encoding="utf-8")
        except Exception as e:
            log(f"[{idx+1}/{len(files)}] ERROR reading {rel_path}: {e}")
            STATS["errors"] += 1
            continue

        fm, body = parse_frontmatter(text)
        title = fm.get("title", rel_path.replace(".md", "").replace("/", " — "))
        plain = strip_markdown(body)

        if len(plain) < 20:
            log(f"[{idx+1}/{len(files)}] SKIP {rel_path} (too short after stripping)")
            STATS["skipped"] += 1
            continue

        # Determine content type and confidence from frontmatter
        content_type = embed_content_type(fm, os.path.dirname(rel_path))
        try:
            confidence = float(fm.get("confidence", 0.7))
        except ValueError:
            confidence = 0.7

        # Build metadata
        meta = {
            "workspace_id": "ddavis",
            "content_type": content_type,
            "title": title,
            "timestamp": ts,
            "source": f"wiki:{rel_path}",
            "tags": fm.get("tags", ""),
            "original_type": fm.get("type", ""),
            "confidence": confidence,
            "use_policy": "evidence",
            "requires_review": False,
        }

        doc_id = f"wiki_{content_type}_{idx}_{int(time.time())}"

        # Build the document: title + plaintext content (truncated for ChromaDB)
        doc_content = f"# {title}\n\n{plain}"
        if len(doc_content) > 8000:
            doc_content = doc_content[:8000] + "\n\n[truncated]"

        batch_ids.append(doc_id)
        batch_documents.append(doc_content)
        batch_metadatas.append(meta)

        STATS["ingested"] += 1

        # Flush batch when full
        if len(batch_ids) >= BATCH_SIZE or idx == len(files) - 1:
            print(f"\r  Batching {len(batch_ids)} docs...", end="", flush=True)
            try:
                # Get embeddings for the batch
                embeddings = []
                for j, doc in enumerate(batch_documents):
                    print(f"\r  Embedding {idx+1-len(batch_ids)+j+1}/{len(files)}...", end="", flush=True)
                    emb = embed_text(doc)
                    if len(emb) != 768:
                        log(f"\n  WARNING: Bad embedding dim {len(emb)} for {title}, using zeros")
                        emb = [0.0] * 768
                    embeddings.append(emb)

                # Write to ChromaDB
                print(f"\r  Writing {len(batch_ids)} docs to ChromaDB...        ", end="", flush=True)
                http_post(f"{COLL_BASE}/{cid}/add", {
                    "ids": batch_ids,
                    "embeddings": embeddings,
                    "metadatas": batch_metadatas,
                    "documents": batch_documents,
                })
                print(f"\r  ✓ Batch {idx+1}/{len(files)} written ({len(batch_ids)} docs)")
            except Exception as e:
                log(f"\n  ERROR writing batch: {e}")
                STATS["errors"] += len(batch_ids)

            batch_ids = []
            batch_documents = []
            batch_metadatas = []

    print()
    log("=== Ingestion Complete ===")
    log(f"  Total files found:   {STATS['total']}")
    log(f"  Ingested:            {STATS['ingested']}")
    log(f"  Skipped (too short): {STATS['skipped']}")
    log(f"  Errors:              {STATS['errors']}")

    # Verify
    print()
    log("Verifying...")
    try:
        count = int(urllib.request.urlopen(
            f"{COLL_BASE}/{cid}/count", timeout=5).read().decode())
        log(f"  ChromaDB document count: {count}")
        if count >= STATS["ingested"]:
            log("  ✓ Verification passed")
        else:
            log(f"  ⚠ Count ({count}) < ingested ({STATS['ingested']}) — some may be missing")
    except Exception as e:
        log(f"  Error verifying count: {e}")

if __name__ == "__main__":
    main()
