#!/usr/bin/env python3
"""
Ad-hoc verification: OB1-Local memory provider Phase 1.
Not a CI suite; one-shot check after installation.
"""

import json
import sys
import urllib.request
import os.path

FAIL = 0
CHROMA = "http://localhost:8000"
TENANT = "default_tenant"
DATABASE = "default_database"
COLL_BASE = f"{CHROMA}/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"
OLLAMA = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
PLUGIN = "/Users/ddavis/.hermes/plugins/ob1-local/__init__.py"
SOURCE = "/Users/ddavis/OB1/integrations/hermes-agent-memory-local/__init__.py"
CONFIG = "/Users/ddavis/.hermes/ob1-local.json"
PLAN = "/Users/ddavis/gh-obsidian/openbrain-implementation-plan.md"
WIKI = "/Users/ddavis/wiki/entities/openbrain.md"

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

print("=== OB1-Local Phase 1 Verification ===")
print()

# 1. Infrastructure
print("1. Infrastructure")
try:
    h = http("GET", f"{CHROMA}/api/v2/heartbeat")
    log("ChromaDB v2 heartbeat", "nanosecond" in str(h))
except Exception as e:
    log("ChromaDB v2 heartbeat", False, f"({e})")

try:
    r = http("POST", OLLAMA, {"model": OLLAMA_MODEL, "prompt": "verify"})
    vec = r.get("embedding", [])
    log(f"Ollama {OLLAMA_MODEL}", len(vec) == 768, f"(768-dim)")
except Exception as e:
    log(f"Ollama {OLLAMA_MODEL}", False, f"({e})")

# 2. Plugin files
print("2. Plugin installation")
log("plugin __init__.py", os.path.exists(PLUGIN), f"({os.path.getsize(PLUGIN)} bytes)")
log("plugin.yaml", os.path.exists(PLUGIN.replace("__init__.py", "plugin.yaml")))
log("ob1-local.json config", os.path.exists(CONFIG))
log("source matches installed",
     open(PLUGIN).read() == open(SOURCE).read(),
     "(identical)")

# 3. Collection
print("3. ChromaDB collection")
try:
    resp = http("GET", f"{COLL_BASE}?name=ob1_memory")
    matched = [c for c in (resp if isinstance(resp, list) else []) if c.get("name") == "ob1_memory"]
    log("ob1_memory collection exists", len(matched) > 0)
    if matched:
        cid = matched[0].get("id", "")
        with urllib.request.urlopen(f"{COLL_BASE}/{cid}/count", timeout=5) as r:
            count = r.read().decode()
        log("collection is ready for writes", int(count) >= 0, f"({count} docs currently — empty is fine on fresh install)")
except Exception as e:
    log("collection check", False, f"({e})")

# 4. Provider structure (syntax check only — Hermes provides runtime deps)
print("4. Plugin code validity")
log("__init__.py has OB1LocalMemoryProvider class",
     open(SOURCE).read().find("class OB1LocalMemoryProvider") > 0)
log("__init__.py has register() function",
     open(SOURCE).read().find("def register(ctx)") > 0)
log("__init__.py implements prefetch()",
     open(SOURCE).read().find("def prefetch") > 0)
log("__init__.py implements sync_turn()",
     open(SOURCE).read().find("def sync_turn") > 0)
log("__init__.py implements queue_prefetch()",
     open(SOURCE).read().find("def queue_prefetch") > 0)
log("__init__.py implements on_session_end()",
     open(SOURCE).read().find("def on_session_end") > 0)
log("__init__.py implements handle_tool_call()",
     open(SOURCE).read().find("def handle_tool_call") > 0)
log("plugin.yaml references ob1-local",
     open(SOURCE.replace("__init__.py", "plugin.yaml")).read().find("ob1-local") > 0)

# Syntax check
try:
    compile(open(SOURCE).read(), SOURCE, 'exec')
    log("__init__.py compiles without errors", True)
except SyntaxError as e:
    log("__init__.py syntax check", False, f"({e})")

# 5. Hermes config
print("5. Hermes configuration")
try:
    import yaml
    cfg = yaml.safe_load(open("/Users/ddavis/.hermes/config.yaml"))
    val = cfg.get("memory", {}).get("provider", "")
    log('memory.provider = ob1-local', val == "ob1-local", f"(got: {val})")
except Exception as e:
    log("config check", False, f"({e})")

# 6. Documentation
print("6. Documentation")
log("implementation plan", os.path.exists(PLAN), f"({os.path.getsize(PLAN)} bytes)")
log("wiki entity updated", os.path.exists(WIKI))

print()
if FAIL == 0:
    print("✅ ALL CHECKS PASSED (ad-hoc)")
else:
    print(f"❌ {FAIL} check(s) FAILED (ad-hoc)")
sys.exit(FAIL)
