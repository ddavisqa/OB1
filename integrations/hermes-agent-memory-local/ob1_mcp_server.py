#!/usr/bin/env python3
"""
OB1-Local MCP Server — Model Context Protocol bridge to ChromaDB memory.
Exposes OB1 memory as MCP tools so Cursor, Claude Desktop, ChatGPT, etc.
can query and write to the same vector store.

Usage:
  python3 ob1_mcp_server.py          # stdio mode (for Claude Desktop/Cursor)
  python3 ob1_mcp_server.py --http   # HTTP mode (for testing)
"""

import json
import hashlib
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# ChromaDB + Ollama config
# ---------------------------------------------------------------------------
CHROMA = os.environ.get("OB1_CHROMA_URL", "http://localhost:8000")
TENANT = "default_tenant"
DATABASE = "default_database"
COLL = os.environ.get("OB1_COLLECTION", "ob1_memory")
WORKSPACE_ID = os.environ.get("OB1_WORKSPACE_ID", "ddavis")
PROJECT_ID = os.environ.get("OB1_PROJECT_ID", "general").strip()
RECALL_PROJECT_IDS = [
    item.strip()
    for item in os.environ.get("OB1_RECALL_PROJECT_IDS", PROJECT_ID).split(",")
    if item.strip()
]
COLL_BASE = f"{CHROMA}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
OLLAMA = os.environ.get("OB1_OLLAMA_URL", "http://localhost:11434/api/embeddings")
OLLAMA_MODEL = os.environ.get("OB1_EMBED_MODEL", "nomic-embed-text")
HTTP_HOST = os.environ.get("OB1_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("OB1_HTTP_PORT", "9100"))
CORS_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("OB1_CORS_ORIGINS", "").split(",")
    if origin.strip()
}
MEMORY_SCHEMA_VERSION = 2


def _is_financial_text(text):
    lowered = (text or "").lower()
    return any(term in lowered for term in (
        "tax", "finance", "financial", "budget", "401k", "401(k)", "ira",
        "mortgage", "income", "revenue", "portfolio", "retirement",
        "brokerage", "dividend", "property tax",
    ))


def _memory_kind(content_type):
    value = str(content_type or "").strip().lower()
    return {
        "fact": "user_fact", "decision": "decision", "lesson": "lesson",
        "constraint": "constraint", "failure": "failure", "output": "output",
        "next_step": "action_item", "mcp_capture": "observation",
    }.get(value, "observation")


def _metadata_in_scope(meta):
    return (
        isinstance(meta, dict)
        and meta.get("workspace_id") == WORKSPACE_ID
        and (not RECALL_PROJECT_IDS or meta.get("project_id") in RECALL_PROJECT_IDS)
    )


def _recallable(meta):
    if not _metadata_in_scope(meta) or meta.get("review_status") == "rejected":
        return False
    expires_at = str(meta.get("expires_at") or "").strip()
    if not expires_at:
        return True
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return expiry > datetime.now(timezone.utc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


def _get_cid():
    try:
        r = json.loads(urllib.request.urlopen(
            f"{COLL_BASE}?name={COLL}", timeout=5).read())
        matched = [c for c in (r if isinstance(r, list) else [])
                   if c.get("name") == COLL]
        if matched:
            return matched[0]["id"]
    except Exception:
        pass
    c = _http("POST", COLL_BASE, {"name": COLL, "metadata": {"hnsw:space": "cosine"}})
    return c.get("id", "")


def _embed(text):
    d = json.dumps({"model": OLLAMA_MODEL, "prompt": text[:8000]}).encode()
    req = urllib.request.Request(OLLAMA, data=d, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("embedding", [])


def _scope_where():
    clauses = [{"workspace_id": WORKSPACE_ID}]
    if len(RECALL_PROJECT_IDS) == 1:
        clauses.append({"project_id": RECALL_PROJECT_IDS[0]})
    elif RECALL_PROJECT_IDS:
        clauses.append({
            "$or": [{"project_id": pid} for pid in RECALL_PROJECT_IDS]
        })
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

CID = None


def tool_search_thoughts(query: str, limit: int = 10) -> dict:
    """Search OpenBrain memories by semantic similarity."""
    global CID
    if not CID:
        CID = _get_cid()
    vec = _embed(query)
    resp = _http("POST", f"{COLL_BASE}/{CID}/query", {
        "query_embeddings": [vec],
        "n_results": min(limit, 50),
        "where": _scope_where(),
    })
    ids = resp.get("ids", [[]])[0]
    docs = resp.get("documents", [[]])[0]
    metas = resp.get("metadatas", [[]])[0]
    dists = resp.get("distances", [[]])[0]

    results = []
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        if not _recallable(meta):
            continue
        results.append({
            "id": ids[i],
            "content": docs[i][:2000] if i < len(docs) else "",
            "metadata": meta,
            "similarity": round(1.0 - dists[i], 4) if i < len(dists) else None,
        })
    return {"results": results, "count": len(results)}


def tool_capture_thought(content: str, tags: str = "",
                         content_type: str = "mcp_capture") -> dict:
    """Capture a new thought/memory into OpenBrain."""
    global CID
    content = str(content or "").strip()
    if not content:
        raise ValueError("content is required")
    if _is_financial_text(content):
        raise ValueError("Financial memory capture is disabled for the general MCP bridge")
    if not CID:
        CID = _get_cid()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc_id = f"mcp_{uuid.uuid4().hex[:12]}_{int(time.time())}"
    vec = _embed(content)
    _http("POST", f"{COLL_BASE}/{CID}/add", {
        "ids": [doc_id],
        "embeddings": [vec],
        "metadatas": [{
            "workspace_id": WORKSPACE_ID,
            "content_type": content_type,
            "timestamp": ts,
            "tags": tags,
            "use_policy": "evidence",
            "confidence": 0.7,
            "source": "mcp_tool",
            "project_id": PROJECT_ID,
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memory_kind": _memory_kind(content_type),
            "provenance_type": "mcp_client",
            "provenance_ref": "local_mcp",
            "capture_method": "mcp_tool",
            "created_at": ts,
            "updated_at": ts,
            "expires_at": "",
            "review_status": "pending",
            "requires_review": True,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }],
        "documents": [content],
    })
    return {"status": "ok", "id": doc_id, "timestamp": ts}


def tool_stats() -> dict:
    """Show memory statistics."""
    global CID
    if not CID:
        CID = _get_cid()
    response = _http("POST", f"{COLL_BASE}/{CID}/get", {
        "where": _scope_where(),
        "limit": 99999,
    })
    metadatas = response.get("metadatas", []) or []
    visible = [meta for meta in metadatas if _metadata_in_scope(meta)]
    review_counts = {}
    for meta in visible:
        status = meta.get("review_status", "legacy_unreviewed")
        review_counts[status] = review_counts.get(status, 0) + 1
    return {
        "total_memories": len(visible),
        "collection": COLL,
        "workspace": WORKSPACE_ID,
        "projects": RECALL_PROJECT_IDS,
        "by_review_status": review_counts,
        "chroma_url": CHROMA,
        "embedding_model": OLLAMA_MODEL,
    }


def tool_recall_context(query: str, limit: int = 5) -> str:
    """Return a formatted <ob1-context> block for system prompt injection."""
    result = tool_search_thoughts(query, limit)
    items = result.get("results", [])
    if not items:
        return ""
    lines = []
    for item in items:
        meta = item.get("metadata", {})
        content = item.get("content", "")[:400]
        lines.append(f"- [{meta.get('content_type', '?')}] {content}")
    body = "\n".join(lines)
    return (
        "<ob1-context>\n"
        "Background context from OpenBrain long-term memory.\n\n"
        f"{body}\n"
        "</ob1-context>"
    )


# ---------------------------------------------------------------------------
# MCP Protocol (JSON-RPC 2.0 over stdio)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "ob1_search",
        "description": "Search OpenBrain memory by semantic similarity. Returns the most relevant memories for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {"type": "integer", "default": 10, "description": "Max results (1-50)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ob1_capture",
        "description": "Capture a new thought, memory, or finding into OpenBrain. The content is embedded and stored for future recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The thought or memory to store"},
                "tags": {"type": "string", "description": "Optional comma-separated tags"},
                "content_type": {"type": "string", "default": "mcp_capture",
                                 "description": "Category: mcp_capture, decision, lesson, fact"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "ob1_stats",
        "description": "Show OpenBrain memory statistics and configuration.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ob1_recall_context",
        "description": "Get a formatted <ob1-context> block for system prompt injection, based on a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall context about"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def handle_mcp_request(msg: dict) -> dict:
    """Handle a single MCP JSON-RPC request."""
    req_id = msg.get("id", 0)
    method = msg.get("method", "")
    params = msg.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},  # we support tools
                },
                "serverInfo": {
                    "name": "ob1-local-mcp",
                    "version": "0.1.0",
                },
            },
        }
    elif method == "notifications/initialized":
        return None  # no response needed
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }
    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if name == "ob1_search":
                result = tool_search_thoughts(
                    args.get("query", ""),
                    args.get("limit", 10),
                )
            elif name == "ob1_capture":
                result = tool_capture_thought(
                    args.get("content", ""),
                    args.get("tags", ""),
                    args.get("content_type", "mcp_capture"),
                )
            elif name == "ob1_stats":
                result = tool_stats()
            elif name == "ob1_recall_context":
                result = tool_recall_context(
                    args.get("query", ""),
                    args.get("limit", 5),
                )
            else:
                return {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"},
                }
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }
    else:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def run_stdio():
    """Run MCP server over stdin/stdout (standard for Claude Desktop, Cursor)."""
    # Initialize
    CID = _get_cid()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_mcp_request(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def run_http(host=HTTP_HOST, port=HTTP_PORT):
    """Run the local MCP bridge over loopback HTTP by default."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    global CID
    CID = _get_cid()

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            origin = self.headers.get("Origin", "")
            if origin and origin in CORS_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            if self.path == "/health":
                try:
                    embedding = _embed("ob1 dependency health check")
                    count = tool_stats()["total_memories"]
                    healthy = bool(embedding) and count >= 0
                    self._respond(200 if healthy else 503, {
                        "status": "ok" if healthy else "degraded",
                        "collection": COLL,
                        "workspace": WORKSPACE_ID,
                        "projects": RECALL_PROJECT_IDS,
                        "memory_count": count,
                        "embedding_dimensions": len(embedding),
                    })
                except Exception as exc:
                    self._respond(503, {
                        "status": "degraded",
                        "error_type": type(exc).__name__,
                    })
            elif self.path == "/tools":
                self._respond(200, {"tools": TOOL_DEFINITIONS})
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode() if length else "{}"
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                self._respond(400, {"error": "invalid JSON"})
                return
            response = handle_mcp_request(msg)
            self._respond(200, response)

        def log_message(self, format, *args):
            pass  # suppress logs

    print(f"OB1 MCP server listening on http://{host}:{port}")
    print(f"  Tools: ob1_search, ob1_capture, ob1_stats, ob1_recall_context")
    print(f"  Collection: {COLL}")
    print(f"  Workspace: {WORKSPACE_ID}; projects: {RECALL_PROJECT_IDS}")
    print(f"  ChromaDB: {CHROMA}")
    print(f"  Embedding: {OLLAMA_MODEL} (Ollama)")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http(HTTP_HOST, HTTP_PORT)
    else:
        run_stdio()
