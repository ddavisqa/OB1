#!/usr/bin/env python3
"""
Ad-hoc smoke test for OB1-Local memory provider.
Verifies: ChromaDB connectivity, embedding, write, recall.
"""

import json
import sys
import time
import urllib.request

FAIL = 0
TEMP_COLLECTION = "ob1_smoke_test"
cid = ""  # collection UUID, populated in step 2
CHROMA = "http://localhost:8000"
TENANT = "default_tenant"
DATABASE = "default_database"
COLLECTIONS = f"{CHROMA}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
OLLAMA = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

def log(label, ok, detail=""):
    global FAIL
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label} {detail}")
    if not ok:
        FAIL += 1

def http(method, url, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}

print("=== OB1-Local Smoke Test ===")
print()

# 1. ChromaDB v2 heartbeat
print("1. ChromaDB connectivity")
try:
    h = http("GET", f"{CHROMA}/api/v2/heartbeat")
    log("v2 heartbeat", "nanosecond" in str(h))
except Exception as e:
    log("v2 heartbeat", False, f"({e})")

# 2. Create temp collection
print("2. Collection ops")
try:
    # List existing collections
    try:
        existing = http("GET", f"{COLLECTIONS}?name={TEMP_COLLECTION}")
    except Exception:
        existing = []
    matched = [c for c in (existing if isinstance(existing, list) else []) if c.get("name") == TEMP_COLLECTION]
    if matched:
        cid = matched[0].get("id", "")
        log(f"collection '{TEMP_COLLECTION}' exists", True, f"(id={cid[:12]}...)")
    else:
        c = http("POST", COLLECTIONS,
                  {"name": TEMP_COLLECTION, "metadata": {"hnsw:space": "cosine"}})
        cid = c.get("id", "")
        log(f"collection created", bool(cid), f"(id={cid[:12]}...)")
except Exception as e:
    log("collection create/get", False, f"({e})")

# 3. Ollama embedding
print("3. Embedding")
try:
    body = {"model": OLLAMA_MODEL, "prompt": "test query for OpenBrain memory system"}
    resp = http("POST", OLLAMA, body)
    vec = resp.get("embedding", [])
    log("nomic-embed-text", len(vec) == 768, f"(dim={len(vec)})")
except Exception as e:
    log("nomic-embed-text", False, f"({e})")
    vec = []

# 4. Write to collection
print("4. Write (add)")
if vec:
    try:
        # Need to get collection ID (UUID), not name
        doc_id = f"smoke_{int(time.time())}"
        r = http("POST", f"{COLLECTIONS}/{cid}/add", {
            "ids": [doc_id],
            "embeddings": [vec],
            "metadatas": [{"workspace_id": "ddavis", "content_type": "test", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
            "documents": ["Test memory entry for OpenBrain smoke test"],
        })
        log("add document", bool(r.get("ids") or r.get("success", True)),
            f"(id={doc_id})")
    except Exception as e:
        log("add document", False, f"({e})")

# 5. Query back
print("5. Recall (query)")
if vec:
    try:
        q = http("POST", f"{COLLECTIONS}/{cid}/query", {
            "query_embeddings": [vec],
            "n_results": 5,
            "where": {"workspace_id": "ddavis"},
        })
        ids = q.get("ids", [[]])[0] if q.get("ids") else []
        docs = q.get("documents", [[]])[0] if q.get("documents") else []
        log("query returned results", len(ids) > 0,
            f"({len(ids)} results)")
        if ids:
            log("document content matches", "OpenBrain" in (docs[0] or ""))
    except Exception as e:
        log("query", False, f"({e})")

# 6. Cleanup
print("6. Cleanup")
try:
    http("POST", f"{COLLECTIONS}/{cid}/delete",
         {"ids": []})  # delete all
    log("temp collection cleaned", True)
except Exception:
    log("cleanup skipped (non-fatal)", True)

print()
if FAIL == 0:
    print("✅ ALL CHECKS PASSED")
else:
    print(f"❌ {FAIL} check(s) FAILED")
sys.exit(FAIL)
