"""OB1 (OpenBrain) local memory provider for Hermes Agent.

Connects Hermes agents to a local ChromaDB + Ollama/mlx-omni-server backend
instead of the cloud Supabase Edge Functions. All data stays on-device.

Provides:
- Auto-recall before every LLM turn (via prefetch)
- Auto-writeback after every turn (via sync_turn)
- Token / model / agent tracking via on_turn_start kwargs
- Six explicit tools for direct memory ops (recall, writeback, search, list, review, trace)

Config:
- $HERMES_HOME/ob1-local.json — non-secret (chroma_url, collection_name, embedding config, options)
- No env vars required (everything in config)

Install:
- $HERMES_HOME/plugins/ob1-local/__init__.py — this file
- $HERMES_HOME/plugins/ob1-local/plugin.yaml — Hermes plugin metadata
- Then: hermes config set memory.provider ob1-local
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WORKSPACE_ID = "ddavis"
_DEFAULT_PROJECT_ID: Optional[str] = None
_DEFAULT_FINANCIAL_PROJECT_ID: Optional[str] = None
_DEFAULT_MAX_RECALL_RESULTS = 10
_DEFAULT_API_TIMEOUT = 8.0
_MIN_CAPTURE_LENGTH = 10
_DEFAULT_CONFIDENCE = 0.7
_PREFETCH_TTL_SECONDS = 90.0
_MEMORY_SCHEMA_VERSION = 2

_MEMORY_KINDS = {
    "observation", "user_fact", "agent_inference", "decision", "constraint",
    "lesson", "action_item", "failure", "output", "reference",
}
_REVIEW_STATUSES = {"unreviewed", "pending", "confirmed", "rejected"}
_PROVENANCE_TYPES = {
    "user_statement", "mixed_turn", "session_extraction", "pre_compression",
    "explicit_tool", "mcp_client", "source_document", "legacy_import",
}

# Default ChromaDB settings
_DEFAULT_CHROMA_URL = "http://localhost:8000"
_DEFAULT_COLLECTION = "ob1_memory"

# Default embedding: Ollama nomic-embed-text
_DEFAULT_EMBED_URL = "http://localhost:11434/api/embeddings"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Trivial messages we should not write back.
_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$",
    re.IGNORECASE,
)

# Auto-recall deliberately excludes the redirected financial project so sensitive
# records are not silently sent to a cloud model.  A local-only metadata probe may
# still tell the model that relevant restricted evidence EXISTS; it never injects
# document text, ids, values, or project names.
_RESTRICTED_DISCOVERY_RE = re.compile(
    r"\b(?:portfolio|invest(?:ment|ing)?|retire(?:ment)?|tax(?:es|ation)?|401\s*\(?k\)?|"
    r"ira|stock|bond|brokerage|mortgage|rent(?:al)?|lease|insurance|health|medical|"
    r"hsa|fsa|dental|glp-?1|medication|drug|mri|premium|deductible|coinsurance|"
    r"out[- ]of[- ]pocket|property|real estate|contract|legal)\b",
    re.IGNORECASE,
)
_RESTRICTED_DISCOVERY_MIN_SIMILARITY = 0.35

# Strip our own injected recall context before capturing.
_OB1_CONTEXT_STRIP_RE = re.compile(
    r"<ob1-context>[\s\S]*?</ob1-context>\s*", re.DOTALL
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    return {
        "chroma_url": _DEFAULT_CHROMA_URL,
        "chroma_collection": _DEFAULT_COLLECTION,
        "embedding_endpoint": _DEFAULT_EMBED_URL,
        "embedding_model": _DEFAULT_EMBED_MODEL,
        "workspace_id": _DEFAULT_WORKSPACE_ID,
        "project_id": _DEFAULT_PROJECT_ID,
        "recall_project_ids": [],
        "financial_project_id": _DEFAULT_FINANCIAL_PROJECT_ID,
        "capture_financial": True,
        "auto_recall": True,
        "auto_capture": True,
        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,
        "api_timeout": _DEFAULT_API_TIMEOUT,
        "default_confidence": _DEFAULT_CONFIDENCE,
        "require_review_by_default": True,
        "default_ttl_days": 0,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return default


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    result: List[str] = []
    for item in candidates:
        normalized = str(item or "").strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _load_ob1_local_config(hermes_home: str) -> dict:
    """Load config from $HERMES_HOME/ob1-local.json with env-var overrides."""
    config = _default_config()
    config_path = Path(hermes_home) / "ob1-local.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    # Env var overrides
    env_url = os.environ.get("OB1_CHROMA_URL", "").strip()
    if env_url:
        config["chroma_url"] = env_url
    env_workspace = os.environ.get("OB1_WORKSPACE_ID", "").strip()
    if env_workspace:
        config["workspace_id"] = env_workspace
    env_project = os.environ.get("OB1_PROJECT_ID", "").strip()
    if env_project:
        config["project_id"] = env_project

    # Normalize
    config["chroma_url"] = str(config.get("chroma_url") or _DEFAULT_CHROMA_URL).rstrip("/")
    config["chroma_collection"] = str(config.get("chroma_collection") or _DEFAULT_COLLECTION)
    config["workspace_id"] = str(config.get("workspace_id") or _DEFAULT_WORKSPACE_ID)
    project_id = config.get("project_id")
    config["project_id"] = str(project_id) if project_id else None
    financial_project_id = config.get("financial_project_id")
    config["financial_project_id"] = (
        str(financial_project_id) if financial_project_id else None
    )
    recall_project_ids = _as_string_list(config.get("recall_project_ids"))
    if not recall_project_ids and config["project_id"]:
        recall_project_ids = [config["project_id"]]
    config["recall_project_ids"] = recall_project_ids
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    config["capture_financial"] = _as_bool(
        config.get("capture_financial"), True
    )
    config["require_review_by_default"] = _as_bool(config.get("require_review_by_default"), True)

    try:
        config["max_recall_results"] = max(1, min(50, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))))
    except Exception:
        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS
    try:
        config["api_timeout"] = max(1.0, min(30.0, float(config.get("api_timeout", _DEFAULT_API_TIMEOUT))))
    except Exception:
        config["api_timeout"] = _DEFAULT_API_TIMEOUT
    try:
        config["default_confidence"] = max(0.0, min(1.0, float(config.get("default_confidence", _DEFAULT_CONFIDENCE))))
    except Exception:
        config["default_confidence"] = _DEFAULT_CONFIDENCE
    try:
        config["default_ttl_days"] = max(0, min(3650, int(config.get("default_ttl_days", 0))))
    except Exception:
        config["default_ttl_days"] = 0

    return config


def _save_ob1_local_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "ob1-local.json"
    existing: dict = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    config_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_scope_where(
    workspace_id: str,
    project_ids: List[str],
    *conditions: Tuple[str, Any],
) -> dict:
    clauses: List[dict] = [{"workspace_id": workspace_id}]
    if len(project_ids) == 1:
        clauses.append({"project_id": project_ids[0]})
    elif project_ids:
        clauses.append({"$or": [{"project_id": pid} for pid in project_ids]})
    clauses.extend({key: value} for key, value in conditions)
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _clean_text_for_capture(text: str) -> str:
    return _OB1_CONTEXT_STRIP_RE.sub("", text or "").strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _is_financial_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        word in lowered
        for word in (
            "tax", "finance", "financial", "budget", "401k", "401(k)",
            "solo 401", "ira", "mortgage", "rent", "property tax", "1099",
            "income", "revenue", "portfolio", "retirement", "holding",
            "dividend", "brokerage", "vanguard", "empower",
        )
    )


def _memory_kind(value: Any, content_type: str = "") -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "fact": "user_fact", "facts": "user_fact",
        "decisions": "decision", "constraints": "constraint",
        "lessons": "lesson", "next_step": "action_item",
        "next_steps": "action_item", "failures": "failure",
        "outputs": "output", "turn": "agent_inference",
        "entity": "reference", "concept": "reference",
        "comparison": "reference", "query": "reference",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _MEMORY_KINDS:
        return normalized
    fallback = aliases.get(str(content_type or "").strip().lower(), "observation")
    return fallback if fallback in _MEMORY_KINDS else "observation"


def _quality_metadata(
    content: str,
    *,
    timestamp: str,
    memory_kind: str,
    provenance_type: str,
    provenance_ref: str,
    capture_method: str,
    confidence: float,
    require_review: bool = True,
    expires_at: str = "",
) -> Dict[str, Any]:
    kind = _memory_kind(memory_kind)
    provenance = str(provenance_type or "").strip().lower()
    if provenance not in _PROVENANCE_TYPES:
        provenance = "legacy_import"
    review_status = "pending" if require_review else "unreviewed"
    return {
        "schema_version": _MEMORY_SCHEMA_VERSION,
        "memory_kind": kind,
        "provenance_type": provenance,
        "provenance_ref": str(provenance_ref or "")[:500],
        "capture_method": str(capture_method or "unknown")[:100],
        "created_at": timestamp,
        "updated_at": timestamp,
        "expires_at": str(expires_at or ""),
        "review_status": review_status,
        "requires_review": review_status != "confirmed",
        "use_policy": "evidence",
        "confidence": max(0.0, min(1.0, float(confidence))),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _expiry_for_ttl(timestamp: str, ttl_days: int) -> str:
    if ttl_days <= 0:
        return ""
    try:
        created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return (created + timedelta(days=ttl_days)).isoformat()
    except Exception:
        return ""


def _metadata_in_scope(meta: Any, workspace_id: str, project_ids: List[str]) -> bool:
    if not isinstance(meta, dict) or meta.get("workspace_id") != workspace_id:
        return False
    return not project_ids or meta.get("project_id") in project_ids


def _memory_is_recallable(meta: Any, *, now: Optional[datetime] = None) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("review_status") == "rejected":
        return False
    expires_at = str(meta.get("expires_at") or "").strip()
    if not expires_at:
        return True
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        current = now or datetime.now(timezone.utc)
        return expiry > current
    except Exception:
        return False


def _recallable_in_scope(
    meta: Any,
    workspace_id: str,
    project_ids: List[str],
    *,
    now: Optional[datetime] = None,
) -> bool:
    return _metadata_in_scope(meta, workspace_id, project_ids) and _memory_is_recallable(
        meta, now=now
    )


# Finding extraction patterns (preserved from OB1)
_DECISION_PATTERNS = (
    re.compile(r"\b(decided|let'?s use|going with|we'?ll go with|chose|picked)\b", re.IGNORECASE),
    re.compile(r"\b(decision|conclusion):\s", re.IGNORECASE),
)
_LESSON_PATTERNS = (
    re.compile(r"\b(lesson learned|next time|going forward|takeaway|note to self|gotcha)\b", re.IGNORECASE),
    re.compile(r"\b(don'?t|do not|never|avoid)\s+\w+\s+\w+", re.IGNORECASE),
)
_CONSTRAINT_PATTERNS = (
    re.compile(r"\b(must not|cannot|can'?t|limited to|capped at|max(imum)? \d|min(imum)? \d)\b", re.IGNORECASE),
    re.compile(r"\b(constraint|requirement):\s", re.IGNORECASE),
)
_NEXT_STEP_PATTERNS = (
    re.compile(r"\b(TODO|FIXME|next step|follow up|action item)\b", re.IGNORECASE),
    re.compile(r"\b(I'?ll|we'?ll|need to|should)\s+(start|finish|fix|add|remove|update|review|check|build|write|test)\b", re.IGNORECASE),
)
_QUESTION_PATTERNS = (
    re.compile(r"\b(unclear|not sure|unknown|to be determined|TBD|undecided)\b", re.IGNORECASE),
    re.compile(r"\?\s*$"),
)
_FAILURE_PATTERNS = (
    re.compile(r"\b(failed|error|exception|broke|crashed|didn'?t work)\b", re.IGNORECASE),
    re.compile(r"\b(workaround|hack|temporary fix)\b", re.IGNORECASE),
)


def _extract_findings(messages: List[Dict[str, Any]], *, per_category_limit: int = 5) -> Dict[str, List[str]]:
    findings: Dict[str, List[str]] = {
        "decisions": [], "lessons": [], "constraints": [],
        "next_steps": [], "unresolved_questions": [], "failures": [], "outputs": [],
    }
    seen: set = set()
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _clean_text_for_capture(str(msg.get("content", "")))
        if not content:
            continue
        for line in re.split(r"(?<=[.!?])\s+|\n+", content):
            line = line.strip()
            if len(line) < 20 or len(line) > 400:
                continue
            if line in seen:
                continue
            matched_category: Optional[str] = None
            for category, patterns in (
                ("decisions", _DECISION_PATTERNS),
                ("constraints", _CONSTRAINT_PATTERNS),
                ("failures", _FAILURE_PATTERNS),
                ("next_steps", _NEXT_STEP_PATTERNS),
                ("lessons", _LESSON_PATTERNS),
                ("unresolved_questions", _QUESTION_PATTERNS),
            ):
                if any(p.search(line) for p in patterns):
                    matched_category = category
                    break
            if matched_category is None:
                continue
            seen.add(line)
            bucket = findings[matched_category]
            if len(bucket) < per_category_limit:
                bucket.append(line)
    return {k: v for k, v in findings.items() if v}


def _read_hermes_active_model(hermes_home: str) -> Dict[str, str]:
    try:
        from hermes_cli.config import cfg_get
        model = str(cfg_get("model.default") or cfg_get("default_model") or "").strip()
        provider = str(cfg_get("model.provider") or "").strip()
        if model or provider:
            return {"model": model, "provider": provider}
    except Exception:
        pass
    cfg_path = Path(hermes_home) / "config.yaml"
    if not cfg_path.exists():
        return {"model": "", "provider": ""}
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"model": "", "provider": ""}
    model = ""
    provider = ""
    nested = cfg.get("model")
    if isinstance(nested, dict):
        model = str(nested.get("default") or "").strip()
        provider = str(nested.get("provider") or "").strip()
    elif isinstance(nested, str):
        model = nested.strip()
    if not model:
        model = str(cfg.get("default_model") or "").strip()
    if provider.lower() == "auto":
        provider = ""
    return {"model": model, "provider": provider}


def _format_relative_time(iso_ts: str) -> str:
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        if seconds < 1800:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        if seconds < 604800:
            return f"{int(seconds / 86400)}d ago"
        if dt.year == now.year:
            return dt.strftime("%d %b")
        return dt.strftime("%d %b %Y")
    except Exception:
        return ""


def _format_recall_context(memories: List[Dict[str, Any]], max_results: int) -> str:
    if not memories:
        return ""
    lines: List[str] = []
    for item in memories[:max_results]:
        doc = item.get("document") or item.get("content") or ""
        if not doc:
            continue
        bits: List[str] = []
        meta = item.get("metadata") or {}
        ts = meta.get("timestamp", "")
        rel = _format_relative_time(ts)
        if rel:
            bits.append(f"[{rel}]")
        review_status = str(meta.get("review_status") or "unreviewed")
        policy = meta.get("use_policy", "evidence")
        if policy == "instruction" and review_status == "confirmed":
            bits.append("[instruction]")
        else:
            bits.append("[evidence]")
        bits.append(f"[{_memory_kind(meta.get('memory_kind'), meta.get('content_type', ''))}]")
        if review_status != "confirmed":
            bits.append(f"[{review_status}]")
        prefix = " ".join(bits)
        max_doc = doc[:400]
        lines.append(f"- {prefix} {max_doc}".strip())
    if not lines:
        return ""
    intro = (
        "Background context from OpenBrain long-term memory. Use silently when relevant. "
        "Only confirmed [instruction] memories are rules; [evidence] is supporting context only. "
        "Do not force memories into the conversation."
    )
    body = "\n".join(lines)
    return f"<ob1-context>\n{intro}\n\n{body}\n</ob1-context>"


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

class _EmbeddingClient:
    """Call a local embedding model (Ollama or mlx-omni-server)."""

    def __init__(self, endpoint: str, model: str, timeout: float):
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout = timeout

    def embed(self, text: str) -> List[float]:
        """Return embedding vector for a single text string."""
        # Try Ollama format first
        if "/api/embeddings" in self._endpoint:
            body = json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
        else:
            # OpenAI-compatible format (mlx-omni-server)
            body = json.dumps({
                "model": self._model,
                "input": text[:8000],  # truncate to avoid OOM
            }).encode("utf-8")

        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Embedding request failed: {e}") from e

        # Parse response (Ollama vs OpenAI format)
        if "embedding" in data:
            return data["embedding"]
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", [])
        raise RuntimeError(f"Unexpected embedding response format: {list(data.keys())}")


# ---------------------------------------------------------------------------
# ChromaDB v2 HTTP client
# ---------------------------------------------------------------------------

class _ChromaClient:
    """Thin sync HTTP client for ChromaDB v2 REST API (multi-tenant paths)."""

    def __init__(self, base_url: str, timeout: float, *,
                 tenant: str = "default_tenant",
                 database: str = "default_database"):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._tenant = tenant
        self._database = database
        self._collection_id: Optional[str] = None

    def _collection_path(self, collection_id: str) -> str:
        return (f"/api/v2/tenants/{self._tenant}"
                f"/databases/{self._database}"
                f"/collections/{collection_id}")

    def _collections_path(self) -> str:
        return (f"/api/v2/tenants/{self._tenant}"
                f"/databases/{self._database}"
                f"/collections")

    def _request(self, method: str, path: str, *, body: Optional[dict] = None) -> dict:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"ChromaDB {method} {path} failed: HTTP {e.code} {err_body[:300]}") from e

    def health(self) -> bool:
        try:
            resp = self._request("GET", "/api/v2/heartbeat")
            return "nanosecond heartbeat" in resp
        except Exception:
            return False

    def _ensure_collection(self, name: str, *, recreate: bool = False) -> str:
        """Get or create a collection by name. Returns collection UUID."""
        # Try to find existing collection
        try:
            from urllib.parse import quote as _url_quote
            resp = self._request("GET", f"{self._collections_path()}?name={_url_quote(name)}")
            if isinstance(resp, list) and len(resp) > 0:
                for c in resp:
                    if c.get("name") == name:
                        cid = c.get("id", "")
                        if cid:
                            self._collection_id = cid
                            return cid
            elif isinstance(resp, dict) and resp.get("name") == name:
                self._collection_id = resp["id"]
                return resp["id"]
        except Exception:
            pass

        # Create new collection
        try:
            body = {"name": name, "metadata": {"hnsw:space": "cosine"}}
            resp = self._request("POST", self._collections_path(), body=body)
            cid = resp.get("id", "")
            if cid:
                self._collection_id = cid
                return cid
        except Exception as e:
            raise RuntimeError(f"Failed to create collection '{name}': {e}") from e

        raise RuntimeError(f"Could not resolve collection '{name}'")

    def _cid(self, collection_name: str) -> str:
        """Resolve collection ID, cached."""
        if self._collection_id:
            return self._collection_id
        return self._ensure_collection(collection_name)

    def add(self, collection_name: str, ids: List[str],
            embeddings: List[List[float]],
            metadatas: List[dict],
            documents: List[str]) -> dict:
        cid = self._cid(collection_name)
        body = {
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents,
        }
        return self._request("POST", f"{self._collection_path(cid)}/add", body=body)

    def query(self, collection_name: str,
              query_embeddings: List[List[float]],
              n_results: int = 10,
              where: Optional[dict] = None) -> dict:
        cid = self._cid(collection_name)
        body: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where:
            body["where"] = where
        return self._request("POST", f"{self._collection_path(cid)}/query", body=body)

    def get(self, collection_name: str, *,
            ids: Optional[List[str]] = None,
            where: Optional[dict] = None,
            limit: int = 20) -> dict:
        cid = self._cid(collection_name)
        body: dict = {"limit": limit}
        if ids:
            body["ids"] = ids
        if where:
            body["where"] = where
        return self._request("POST", f"{self._collection_path(cid)}/get", body=body)

    def delete(self, collection_name: str, ids: List[str]) -> dict:
        cid = self._cid(collection_name)
        return self._request("POST", f"{self._collection_path(cid)}/delete", body={"ids": ids})

    def update(self, collection_name: str, ids: List[str], metadatas: List[dict]) -> dict:
        cid = self._cid(collection_name)
        return self._request(
            "POST",
            f"{self._collection_path(cid)}/update",
            body={"ids": ids, "metadatas": metadatas},
        )


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "ob1_recall",
    "description": "Recall relevant memories from OpenBrain. Use BEFORE meaningful work.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language query about the task or topic."},
            "limit": {"type": "integer", "description": "Maximum results, 1-50.", "default": 10},
            "content_type": {"type": "string", "description": "Filter by content type (decision, lesson, constraint, etc)."},
        },
        "required": ["query"],
    },
}

WRITEBACK_SCHEMA = {
    "name": "ob1_writeback",
    "description": "Write a memory to OpenBrain. Memory starts as evidence; review/confirmation needed before instruction-grade.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Short summary of the memory (1-2 sentences)."},
            "content": {"type": "string", "description": "Full content of the memory."},
            "memory_type": {"type": "string", "description": "Category: decision, lesson, constraint, output, failure, fact."},
            "confidence": {"type": "number", "description": "Self-reported confidence 0-1.", "default": 0.7},
            "expires_at": {"type": "string", "description": "Optional ISO-8601 expiration timestamp."},
            "metadata": {"type": "object", "description": "Optional extra metadata (token counts, cost, sources, ...)."},
        },
        "required": ["summary", "content"],
    },
}

SEARCH_SCHEMA = {
    "name": "ob1_search",
    "description": "Search OpenBrain memories explicitly (alias for recall with simpler params).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}

LIST_MEMORIES_SCHEMA = {
    "name": "ob1_list_memories",
    "description": "List recent memories optionally filtered by workspace or content type.",
    "parameters": {
        "type": "object",
        "properties": {
            "content_type": {"type": "string", "description": "Filter by content type (optional)."},
            "limit": {"type": "integer", "default": 20},
        },
    },
}

DELETE_MEMORY_SCHEMA = {
    "name": "ob1_delete_memory",
    "description": "Delete a memory by its ChromaDB ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ChromaDB document ID to delete."},
        },
        "required": ["memory_id"],
    },
}

REVIEW_MEMORY_SCHEMA = {
    "name": "ob1_review_memory",
    "description": "Confirm or reject one visible memory after explicit user review. Never call without direct user authorization.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ChromaDB document ID."},
            "status": {"type": "string", "enum": ["confirmed", "rejected"]},
            "reason": {"type": "string", "description": "Short audit reason for the decision."},
            "instruction_grade": {"type": "boolean", "default": False, "description": "Treat a confirmed decision or constraint as an instruction."},
            "user_confirmation": {"type": "string", "description": "Must exactly equal CONFIRM MEMORY REVIEW."},
        },
        "required": ["memory_id", "status", "reason", "user_confirmation"],
    },
}

STATS_SCHEMA = {
    "name": "ob1_stats",
    "description": "Show memory statistics: total count, counts by content type.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class OB1LocalMemoryProvider(MemoryProvider):
    """OpenBrain (OB1) local memory provider — ChromaDB + local embeddings."""

    def __init__(self) -> None:
        self._config = _default_config()
        self._chroma_client: Optional[_ChromaClient] = None
        self._embed_client: Optional[_EmbeddingClient] = None
        self._session_id: str = ""
        self._turn_count: int = 0
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache: Optional[Tuple[str, str, Optional[str], float]] = None
        self._auto_recall: bool = True
        self._auto_capture: bool = True
        self._max_recall_results: int = _DEFAULT_MAX_RECALL_RESULTS
        self._api_timeout: float = _DEFAULT_API_TIMEOUT
        self._default_confidence: float = _DEFAULT_CONFIDENCE
        self._hermes_home: str = ""
        self._write_enabled: bool = True
        self._active: bool = False
        self._last_turn_meta: Dict[str, Any] = {}
        self._last_request_id: Optional[str] = None
        self._runtime_name: str = "hermes"
        self._runtime_version: str = ""
        self._agent_identity: str = "default"
        self._platform: str = "cli"

    @property
    def name(self) -> str:
        return "ob1-local"

    # ---- Availability / setup -----------------------------------------

    def is_available(self) -> bool:
        try:
            from hermes_constants import get_hermes_home
            home = str(get_hermes_home())
        except Exception:
            home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
        cfg = _load_ob1_local_config(home)
        return bool(cfg.get("chroma_url"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "chroma_url", "description": "ChromaDB server URL", "secret": False, "required": True, "default": _DEFAULT_CHROMA_URL},
            {"key": "chroma_collection", "description": "ChromaDB collection name", "secret": False, "required": False, "default": _DEFAULT_COLLECTION},
            {"key": "embedding_endpoint", "description": "Embedding API endpoint", "secret": False, "required": False},
            {"key": "embedding_model", "description": "Embedding model name", "secret": False, "required": False},
            {"key": "workspace_id", "description": "Workspace ID for memory scoping", "secret": False, "required": False},
            {"key": "project_id", "description": "Default project ID (optional)", "secret": False, "required": False},
            {"key": "recall_project_ids", "description": "Project IDs visible to recall", "secret": False, "required": False},
            {"key": "financial_project_id", "description": "Project ID for financial memories", "secret": False, "required": False},
            {"key": "capture_financial", "description": "Allow financial auto-capture in this profile", "secret": False, "required": False},
            {"key": "require_review_by_default", "description": "Mark new memories pending review", "secret": False, "required": False, "default": True},
            {"key": "default_ttl_days", "description": "Optional default expiry in days (0 disables expiry)", "secret": False, "required": False, "default": 0},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        sanitized: Dict[str, Any] = {}
        for k in (
            "chroma_url", "chroma_collection", "embedding_endpoint",
            "embedding_model", "workspace_id", "project_id",
            "financial_project_id",
        ):
            if k in values and values[k] is not None:
                sanitized[k] = str(values[k]).strip()
        if "recall_project_ids" in values:
            sanitized["recall_project_ids"] = _as_string_list(
                values["recall_project_ids"]
            )
        for k in ("auto_recall", "auto_capture", "capture_financial", "require_review_by_default"):
            if k in values:
                sanitized[k] = _as_bool(values[k], True)
        if "max_recall_results" in values:
            try:
                sanitized["max_recall_results"] = int(values["max_recall_results"])
            except Exception:
                pass
        if "default_confidence" in values:
            try:
                sanitized["default_confidence"] = float(values["default_confidence"])
            except Exception:
                pass
        if "default_ttl_days" in values:
            try:
                sanitized["default_ttl_days"] = int(values["default_ttl_days"])
            except Exception:
                pass
        if sanitized:
            _save_ob1_local_config(sanitized, hermes_home)

    # ---- Lifecycle ----------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            from hermes_constants import get_hermes_home
            self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        except Exception:
            self._hermes_home = kwargs.get("hermes_home") or os.environ.get(
                "HERMES_HOME", str(Path.home() / ".hermes")
            )

        self._session_id = session_id
        self._turn_count = 0
        self._last_turn_meta = {}
        self._last_request_id = None

        self._config = _load_ob1_local_config(self._hermes_home)
        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_recall_results = self._config["max_recall_results"]
        self._api_timeout = self._config["api_timeout"]
        self._default_confidence = self._config["default_confidence"]

        self._agent_identity = kwargs.get("agent_identity", "default")
        self._platform = kwargs.get("platform", "cli")

        try:
            from hermes_constants import VERSION as _hermes_version
            self._runtime_version = str(_hermes_version)
        except Exception:
            self._runtime_version = ""

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in ("cron", "flush", "subagent")

        # Initialize ChromaDB client
        chroma_url = self._config["chroma_url"]
        self._chroma_client = _ChromaClient(base_url=chroma_url, timeout=self._api_timeout)

        # Initialize embedding client
        embed_endpoint = self._config["embedding_endpoint"]
        embed_model = self._config["embedding_model"]
        self._embed_client = _EmbeddingClient(
            endpoint=embed_endpoint, model=embed_model, timeout=self._api_timeout
        )

        # Verify connectivity (non-fatal)
        self._active = False
        try:
            if self._chroma_client.health():
                collection = self._config["chroma_collection"]
                try:
                    self._chroma_client._ensure_collection(collection)
                    self._active = True
                except Exception as e:
                    logger.warning("ChromaDB collection setup failed: %s", e)
            else:
                logger.warning("ChromaDB health check failed on %s", chroma_url)
        except Exception as e:
            logger.warning("ChromaDB initialization failed: %s", e)

        if not self._active:
            logger.warning(
                "OB1-Local provider initialized but NOT active — ChromaDB not reachable at %s. "
                "Start it with: chroma run --path ~/.hermes/chroma",
                chroma_url,
            )

    def shutdown(self) -> None:
        for attr in ("_sync_thread", "_prefetch_thread"):
            t = getattr(self, attr, None)
            if t and t.is_alive():
                t.join(timeout=5.0)
            setattr(self, attr, None)
        with self._prefetch_lock:
            self._prefetch_cache = None

    # ---- Per-turn hooks ----------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)
        self._last_turn_meta = {
            "turn_number": turn_number,
            "remaining_tokens": kwargs.get("remaining_tokens"),
            "model": kwargs.get("model"),
            "platform": kwargs.get("platform") or self._platform,
            "tool_count": kwargs.get("tool_count"),
        }
        platform = kwargs.get("platform")
        if platform:
            self._platform = platform

    def _resolve_model_provider(self) -> Dict[str, str]:
        model = self._last_turn_meta.get("model") or ""
        provider = ""
        if not model and self._hermes_home:
            fallback = _read_hermes_active_model(self._hermes_home)
            model = fallback.get("model", "")
            provider = fallback.get("provider", "")
        if model and not provider:
            if "/" in model:
                head = model.split("/", 1)[0].lower()
                if head in ("openrouter", "anthropic", "openai", "google", "meta",
                            "mistralai", "stepfun", "z-ai", "minimax", "xiaomi", "deepseek"):
                    provider = "openrouter" if head == "openrouter" else head
        return {"model": str(model), "provider": str(provider)}

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        recall_projects = ", ".join(self._config["recall_project_ids"]) or "legacy-unscoped"
        return (
            "# OpenBrain (OB1) Local Memory\n"
            "Active. Workspace: "
            f"{self._config['workspace_id']}"
            + f", recall projects: {recall_projects}"
            + ".\n"
            "Tools: ob1_recall, ob1_writeback, ob1_search, ob1_list_memories, "
            "ob1_review_memory, ob1_delete_memory, ob1_stats.\n"
            "Memory discipline: recall before meaningful work; writeback compact, "
            "provenance-labeled summaries after. Treat memories tagged [instruction] "
            "as binding rules; [evidence]-tagged as supporting context only."
        )

    # ---- Embedding helper --------------------------------------------

    def _embed_text(self, text: str, *, is_financial: bool = False) -> List[float]:
        """Embed text with the collection's single configured embedding model."""
        if self._embed_client:
            return self._embed_client.embed(text[:4000])
        raise RuntimeError("No embedding client available")

    def _write_scope_for_text(self, text: str) -> Tuple[Optional[str], bool]:
        is_financial = _is_financial_text(text)
        if is_financial and not self._config["capture_financial"]:
            return None, True
        if is_financial and self._config.get("financial_project_id"):
            return self._config["financial_project_id"], True
        return self._config.get("project_id"), is_financial

    # ---- Auto-recall / auto-capture ----------------------------------

    def _restricted_evidence_notice(self, query: str, query_vec: List[float]) -> str:
        """Return a content-free alert when a sensitive excluded project has a relevant hit.

        The probe and embedding are local.  Only existence is disclosed to the model; the
        explicit recall tool remains the auditable path that can reveal evidence content.
        """
        restricted_project = self._config.get("financial_project_id")
        if (
            not restricted_project
            or restricted_project in self._config.get("recall_project_ids", [])
            or not _RESTRICTED_DISCOVERY_RE.search(query or "")
        ):
            return ""
        try:
            response = self._chroma_client.query(
                collection_name=self._config["chroma_collection"],
                query_embeddings=[query_vec],
                n_results=5,
                where=_build_scope_where(
                    self._config["workspace_id"], [restricted_project]
                ),
            )
        except Exception:
            logger.debug("OB1-Local restricted-evidence probe failed", exc_info=True)
            return ""

        metadatas = response.get("metadatas", [[]])
        metadatas = metadatas[0] if isinstance(metadatas, list) and metadatas else []
        distances = response.get("distances", [[]])
        distances = distances[0] if isinstance(distances, list) and distances else []
        for i, meta in enumerate(metadatas):
            dist = distances[i] if i < len(distances) else None
            similarity = 1.0 - float(dist) if dist is not None else 0.0
            if (
                _recallable_in_scope(
                    meta, self._config["workspace_id"], [restricted_project]
                )
                and similarity >= _RESTRICTED_DISCOVERY_MIN_SIMILARITY
            ):
                return (
                    "## Restricted Memory Signal\n"
                    "Local-only retrieval found relevant evidence in a project excluded from "
                    "automatic prompt injection. No restricted content, identifiers, or values "
                    "were disclosed. Before Tier 2/3 synthesis or persistence, call the explicit "
                    "memory recall tool for this topic and follow any canonical-note, supersession, "
                    "withdrawal, contradiction, or incident pointer it returns."
                )
        return ""

    def _do_recall(self, query: str, *, session_id: str = "") -> Tuple[str, Optional[str]]:
        if not self._active or not self._chroma_client or not self._embed_client:
            return "", None
        q = (query or "").strip()
        if not q:
            return "", None
        try:
            # Embed the query
            query_vec = self._embed_text(q)

            where = _build_scope_where(
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            )

            response = self._chroma_client.query(
                collection_name=self._config["chroma_collection"],
                query_embeddings=[query_vec],
                n_results=self._max_recall_results,
                where=where,
            )
        except Exception:
            logger.debug("OB1-Local recall failed", exc_info=True)
            return "", None

        # ChromaDB returns: {ids: [[...]], distances: [[...]], metadatas: [[...]], documents: [[...]]}
        request_id = f"recall_{int(time.time())}"
        memories: List[Dict[str, Any]] = []
        ids = response.get("ids", [[]])[0] if isinstance(response.get("ids"), list) and response.get("ids") else []
        documents = response.get("documents", [[]])[0] if isinstance(response.get("documents"), list) and response.get("documents") else []
        metadatas = response.get("metadatas", [[]])[0] if isinstance(response.get("metadatas"), list) and response.get("metadatas") else []
        distances = response.get("distances", [[]])[0] if isinstance(response.get("distances"), list) and response.get("distances") else []

        for i, doc_id in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            if not _recallable_in_scope(
                meta,
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            ):
                continue
            dist = distances[i] if i < len(distances) else None
            review_status = str(meta.get("review_status") or "unreviewed")
            memories.append({
                "id": doc_id,
                "document": doc,
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "updated_at": meta.get("updated_at") or meta.get("timestamp", ""),
                "created_at": meta.get("created_at") or meta.get("timestamp", ""),
                "use_policy": {
                    "can_use_as_instruction": (
                        meta.get("use_policy") == "instruction"
                        and review_status == "confirmed"
                    ),
                    "can_use_as_evidence": meta.get("use_policy", "evidence") == "evidence",
                    "requires_user_confirmation": review_status != "confirmed",
                },
            })

        context = _format_recall_context(memories, self._max_recall_results)
        restricted_notice = self._restricted_evidence_notice(q, query_vec)
        if restricted_notice:
            context = "\n\n".join(part for part in (context, restricted_notice) if part)
        return context, request_id

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall:
            return ""
        q = (query or "").strip()
        if not q:
            return ""

        # Consume only a cache entry prepared for this exact semantic query.  queue_prefetch()
        # runs after the previous turn, so an unkeyed cache serves the previous question's
        # memories to a new question for up to _PREFETCH_TTL_SECONDS.
        cached = self._consume_prefetch_cache(q)
        if cached is not None:
            context, request_id = cached
            if request_id:
                self._last_request_id = request_id
            return context

        context, request_id = self._do_recall(q, session_id=session_id)
        if request_id:
            self._last_request_id = request_id
        return context

    @staticmethod
    def _prefetch_query_key(query: str) -> str:
        """Stable key for cache admission; cosmetic case/spacing changes are equivalent."""
        return re.sub(r"\s+", " ", (query or "").strip()).casefold()

    def _consume_prefetch_cache(self, query: str) -> Optional[Tuple[str, Optional[str]]]:
        with self._prefetch_lock:
            entry = self._prefetch_cache
            self._prefetch_cache = None
        if entry is None:
            return None
        cached_query, context, request_id, ts = entry
        if (time.monotonic() - ts) > _PREFETCH_TTL_SECONDS:
            return None
        if self._prefetch_query_key(cached_query) != self._prefetch_query_key(query):
            logger.debug("OB1-Local discarded prefetch cache for a different query")
            return None
        return context, request_id

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_recall or not self._chroma_client:
            return
        q = (query or "").strip()
        if not q:
            return
        existing = self._prefetch_thread
        if existing and existing.is_alive():
            return

        def _run() -> None:
            try:
                context, request_id = self._do_recall(q, session_id=session_id)
                if not context:
                    return
                with self._prefetch_lock:
                    self._prefetch_cache = (q, context, request_id, time.monotonic())
            except Exception:
                logger.debug("OB1-Local queue_prefetch failed", exc_info=True)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="ob1-local-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._chroma_client:
            return

        clean_user = _clean_text_for_capture(user_content)
        clean_assistant = _clean_text_for_capture(assistant_content)
        if not clean_user or not clean_assistant:
            return
        if len(clean_user) < _MIN_CAPTURE_LENGTH or len(clean_assistant) < _MIN_CAPTURE_LENGTH:
            return
        if _is_trivial_message(clean_user):
            return

        summary = (clean_user[:160].rstrip() + "…") if len(clean_user) > 160 else clean_user
        output_line = f"Turn summary — user: {summary} | assistant: {clean_assistant[:300]}"

        project_id, is_financial = self._write_scope_for_text(
            f"{clean_user}\n{clean_assistant}"
        )
        if is_financial and not self._config["capture_financial"]:
            return

        ts = datetime.now(timezone.utc).isoformat()
        doc_id = f"turn_{session_id or self._session_id}_{self._turn_count}_{int(time.time())}"

        mp = self._resolve_model_provider()
        meta: Dict[str, Any] = {
            "workspace_id": self._config["workspace_id"],
            "content_type": "turn",
            "timestamp": ts,
            "model": mp.get("model", ""),
            "provider": mp.get("provider", ""),
            "task_id": session_id or self._session_id or "",
            "flow_id": self._agent_identity,
            "use_policy": "evidence",
            "confidence": self._default_confidence,
            "requires_review": self._config["require_review_by_default"],
            "is_financial": is_financial,
        }
        meta.update(_quality_metadata(
            output_line,
            timestamp=ts,
            memory_kind="agent_inference",
            provenance_type="mixed_turn",
            provenance_ref=session_id or self._session_id or "",
            capture_method="auto_turn",
            confidence=self._default_confidence,
            require_review=self._config["require_review_by_default"],
            expires_at=_expiry_for_ttl(ts, self._config["default_ttl_days"]),
        ))
        # Only include project_id if it has a value (ChromaDB v2 rejects null metadata)
        if project_id:
            meta["project_id"] = project_id

        def _run() -> None:
            try:
                vec = self._embed_text(output_line, is_financial=is_financial)
                self._chroma_client.add(
                    collection_name=self._config["chroma_collection"],
                    ids=[doc_id],
                    embeddings=[vec],
                    metadatas=[meta],
                    documents=[output_line],
                )
            except Exception:
                logger.debug("OB1-Local sync_turn writeback failed", exc_info=True)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(
            target=_run, daemon=True, name="ob1-local-sync"
        )
        self._sync_thread.start()

    # ---- Session-end + pre-compress hooks ----------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._chroma_client:
            return
        if not messages:
            return

        findings = _extract_findings(messages, per_category_limit=10)
        if not findings:
            user_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
            if user_count < 2:
                return
            findings = {"outputs": [
                f"Session completed with {user_count} user messages; no structured "
                "findings extracted by heuristics."
            ]}

        ts = datetime.now(timezone.utc).isoformat()
        mp = self._resolve_model_provider()
        workspace_id = self._config["workspace_id"]

        for category, items in findings.items():
            for i, item in enumerate(items):
                project_id, is_financial = self._write_scope_for_text(item)
                if is_financial and not self._config["capture_financial"]:
                    continue
                doc_id = f"find_{self._session_id}_{category}_{i}_{int(time.time())}"
                meta: Dict[str, Any] = {
                    "workspace_id": workspace_id,
                    "content_type": category,
                    "timestamp": ts,
                    "model": mp.get("model", ""),
                    "provider": mp.get("provider", ""),
                    "task_id": self._session_id or "",
                    "flow_id": self._agent_identity,
                    "use_policy": "evidence",
                    "confidence": self._default_confidence,
                    "requires_review": self._config["require_review_by_default"],
                    "is_financial": is_financial,
                }
                meta.update(_quality_metadata(
                    item,
                    timestamp=ts,
                    memory_kind=_memory_kind(category, category),
                    provenance_type="session_extraction",
                    provenance_ref=self._session_id or "",
                    capture_method="session_end",
                    confidence=self._default_confidence,
                    require_review=self._config["require_review_by_default"],
                    expires_at=_expiry_for_ttl(ts, self._config["default_ttl_days"]),
                ))
                if project_id:
                    meta["project_id"] = project_id
                try:
                    vec = self._embed_text(item)
                    self._chroma_client.add(
                        collection_name=self._config["chroma_collection"],
                        ids=[doc_id],
                        embeddings=[vec],
                        metadatas=[meta],
                        documents=[item],
                    )
                except Exception:
                    logger.debug("OB1-Local on_session_end writeback failed", exc_info=True)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._active or not self._chroma_client or not messages:
            return ""

        findings = _extract_findings(messages, per_category_limit=5)
        if not findings:
            return ""

        if self._auto_capture and self._write_enabled:
            ts = datetime.now(timezone.utc).isoformat()
            mp = self._resolve_model_provider()
            workspace_id = self._config["workspace_id"]

            for category, items in findings.items():
                for i, item in enumerate(items):
                    project_id, is_financial = self._write_scope_for_text(item)
                    if is_financial and not self._config["capture_financial"]:
                        continue
                    doc_id = f"precompress_{self._session_id}_{category}_{i}_{int(time.time())}"
                    meta: Dict[str, Any] = {
                        "workspace_id": workspace_id,
                        "content_type": category,
                        "timestamp": ts,
                        "model": mp.get("model", ""),
                        "provider": mp.get("provider", ""),
                        "task_id": self._session_id or "",
                        "flow_id": self._agent_identity,
                        "use_policy": "evidence",
                        "confidence": self._default_confidence,
                        "requires_review": self._config["require_review_by_default"],
                        "source": "pre_compress",
                        "is_financial": is_financial,
                    }
                    meta.update(_quality_metadata(
                        item,
                        timestamp=ts,
                        memory_kind=_memory_kind(category, category),
                        provenance_type="pre_compression",
                        provenance_ref=self._session_id or "",
                        capture_method="pre_compress",
                        confidence=self._default_confidence,
                        require_review=self._config["require_review_by_default"],
                        expires_at=_expiry_for_ttl(ts, self._config["default_ttl_days"]),
                    ))
                    if project_id:
                        meta["project_id"] = project_id
                    try:
                        vec = self._embed_text(item)
                        self._chroma_client.add(
                            collection_name=self._config["chroma_collection"],
                            ids=[doc_id],
                            embeddings=[vec],
                            metadatas=[meta],
                            documents=[item],
                        )
                    except Exception:
                        pass

        parts: List[str] = []
        for category, items in findings.items():
            if not items:
                continue
            label = category.replace("_", " ").title()
            joined = "\n".join(f"  - {it}" for it in items[:3])
            parts.append(f"{label}:\n{joined}")
        if not parts:
            return ""
        return (
            "Pre-compression findings extracted by OB1-Local memory provider "
            "(preserve in summary):\n" + "\n".join(parts)
        )

    # ---- Tools --------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            RECALL_SCHEMA,
            WRITEBACK_SCHEMA,
            SEARCH_SCHEMA,
            LIST_MEMORIES_SCHEMA,
            REVIEW_MEMORY_SCHEMA,
            DELETE_MEMORY_SCHEMA,
            STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._chroma_client:
            return tool_error("OpenBrain (OB1) local memory provider is not configured")
        try:
            handlers = {
                "ob1_recall": self._tool_recall,
                "ob1_writeback": self._tool_writeback,
                "ob1_search": self._tool_search,
                "ob1_list_memories": self._tool_list_memories,
                "ob1_review_memory": self._tool_review_memory,
                "ob1_delete_memory": self._tool_delete_memory,
                "ob1_stats": self._tool_stats,
            }
            handler = handlers.get(tool_name)
            if handler:
                return handler(args)
        except Exception as exc:
            return tool_error(f"OB1-Local tool '{tool_name}' failed: {exc}")
        return tool_error(f"Unknown tool: {tool_name}")

    # Per-tool implementations ------------------------------------------

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = max(1, min(50, int(args.get("limit", self._max_recall_results) or self._max_recall_results)))
        except Exception:
            limit = self._max_recall_results

        try:
            query_vec = self._embed_text(query)
            content_type = args.get("content_type")
            extra = (
                [("content_type", str(content_type).lower())]
                if content_type else []
            )
            where = _build_scope_where(
                self._config["workspace_id"],
                self._config["recall_project_ids"],
                *extra,
            )

            response = self._chroma_client.query(
                collection_name=self._config["chroma_collection"],
                query_embeddings=[query_vec],
                n_results=limit,
                where=where,
            )
        except Exception as e:
            return tool_error(f"Recall failed: {e}")

        # Format results
        ids = response.get("ids", [[]])[0] or []
        documents = response.get("documents", [[]])[0] or []
        metadatas = response.get("metadatas", [[]])[0] or []
        distances = response.get("distances", [[]])[0] or []

        results = []
        for i, doc_id in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            if not _recallable_in_scope(
                meta,
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            ):
                continue
            dist = distances[i] if i < len(distances) else None
            results.append({
                "id": doc_id,
                "content": doc[:500],
                "metadata": meta,
                "similarity": round(1.0 - dist, 4) if dist is not None else None,
            })

        return json.dumps({"results": results, "count": len(results)}, indent=2)

    def _tool_search(self, args: Dict[str, Any]) -> str:
        merged = dict(args)
        return self._tool_recall(merged)

    def _tool_writeback(self, args: Dict[str, Any]) -> str:
        summary = str(args.get("summary") or "").strip()
        content = str(args.get("content") or "").strip()
        if not summary and not content:
            return tool_error("summary or content is required")
        try:
            confidence = max(0.0, min(1.0, float(args.get("confidence", self._default_confidence))))
        except Exception:
            confidence = self._default_confidence

        line = summary if summary else content
        if summary and content:
            line = f"{summary} — {content[:600]}"
        project_id, is_financial = self._write_scope_for_text(line)
        if is_financial and not self._config["capture_financial"]:
            return tool_error(
                "Financial memory capture is disabled for this profile"
            )

        memory_type = str(args.get("memory_type") or "lesson").lower()
        category_map = {
            "decision": "decisions", "decisions": "decisions",
            "output": "outputs", "outputs": "outputs",
            "lesson": "lessons", "lessons": "lessons",
            "constraint": "constraints", "constraints": "constraints",
            "question": "unresolved_questions", "unresolved_question": "unresolved_questions",
            "next_step": "next_steps", "next_steps": "next_steps",
            "failure": "failures", "failures": "failures",
        }
        category = category_map.get(memory_type, "outputs")

        ts = datetime.now(timezone.utc).isoformat()
        expires_at = str(args.get("expires_at") or "").strip()
        if expires_at:
            try:
                datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except Exception:
                return tool_error("expires_at must be an ISO-8601 timestamp")
        doc_id = f"tool_{self._session_id}_{self._turn_count}_{int(time.time())}_{hashlib.md5(line.encode()).hexdigest()[:8]}"

        meta: Dict[str, Any] = {
            "workspace_id": self._config["workspace_id"],
            "content_type": category,
            "timestamp": ts,
            "model": self._resolve_model_provider().get("model", ""),
            "provider": self._resolve_model_provider().get("provider", ""),
            "task_id": self._session_id or "",
            "flow_id": self._agent_identity,
            "use_policy": "evidence",
            "confidence": confidence,
            "requires_review": self._config["require_review_by_default"],
            "source": "tool_writeback",
            "is_financial": is_financial,
        }
        meta.update(_quality_metadata(
            line,
            timestamp=ts,
            memory_kind=_memory_kind(memory_type, category),
            provenance_type="explicit_tool",
            provenance_ref=self._session_id or "",
            capture_method="tool_writeback",
            confidence=confidence,
            require_review=self._config["require_review_by_default"],
            expires_at=expires_at or _expiry_for_ttl(ts, self._config["default_ttl_days"]),
        ))
        if project_id:
            meta["project_id"] = project_id

        try:
            vec = self._embed_text(line)
            self._chroma_client.add(
                collection_name=self._config["chroma_collection"],
                ids=[doc_id],
                embeddings=[vec],
                metadatas=[meta],
                documents=[line],
            )
            return json.dumps({"status": "ok", "id": doc_id})
        except Exception as e:
            return tool_error(f"Writeback failed: {e}")

    def _tool_list_memories(self, args: Dict[str, Any]) -> str:
        try:
            limit = max(1, min(100, int(args.get("limit", 20))))
        except Exception:
            limit = 20
        content_type = args.get("content_type")
        extra = (
            [("content_type", str(content_type).lower())]
            if content_type else []
        )
        where = _build_scope_where(
            self._config["workspace_id"],
            self._config["recall_project_ids"],
            *extra,
        )

        try:
            response = self._chroma_client.get(
                collection_name=self._config["chroma_collection"],
                where=where,
                limit=limit,
            )
        except Exception as e:
            return tool_error(f"List failed: {e}")

        ids = response.get("ids", []) or []
        documents = response.get("documents", []) or []
        metadatas = response.get("metadatas", []) or []

        results = []
        for i, doc_id in enumerate(ids):
            doc = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            if not _recallable_in_scope(
                meta,
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            ):
                continue
            results.append({
                "id": doc_id,
                "content": (doc[:300] + "...") if len(doc) > 300 else doc,
                "metadata": meta,
            })
        return json.dumps({"results": results, "count": len(results)}, indent=2)

    def _tool_delete_memory(self, args: Dict[str, Any]) -> str:
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("memory_id is required")
        try:
            visible = self._chroma_client.get(
                collection_name=self._config["chroma_collection"],
                ids=[memory_id],
                where=_build_scope_where(
                    self._config["workspace_id"],
                    self._config["recall_project_ids"],
                ),
                limit=1,
            )
            ids = visible.get("ids", []) or []
            metadatas = visible.get("metadatas", []) or []
            if (
                memory_id not in ids
                or not metadatas
                or not _metadata_in_scope(
                    metadatas[0],
                    self._config["workspace_id"],
                    self._config["recall_project_ids"],
                )
            ):
                return tool_error("Memory is not visible to the active profile")
            self._chroma_client.delete(
                collection_name=self._config["chroma_collection"],
                ids=[memory_id],
            )
            return json.dumps({"status": "ok", "deleted": memory_id})
        except Exception as e:
            return tool_error(f"Delete failed: {e}")

    def _tool_review_memory(self, args: Dict[str, Any]) -> str:
        memory_id = str(args.get("memory_id") or "").strip()
        status = str(args.get("status") or "").strip().lower()
        reason = str(args.get("reason") or "").strip()
        confirmation = str(args.get("user_confirmation") or "")
        if not memory_id:
            return tool_error("memory_id is required")
        if status not in {"confirmed", "rejected"}:
            return tool_error("status must be confirmed or rejected")
        if len(reason) < 3:
            return tool_error("reason is required")
        if confirmation != "CONFIRM MEMORY REVIEW":
            return tool_error("Explicit user confirmation is required")
        try:
            visible = self._chroma_client.get(
                collection_name=self._config["chroma_collection"],
                ids=[memory_id],
                where=_build_scope_where(
                    self._config["workspace_id"],
                    self._config["recall_project_ids"],
                ),
                limit=1,
            )
            ids = visible.get("ids", []) or []
            metadatas = visible.get("metadatas", []) or []
            if memory_id not in ids:
                return tool_error("Memory is not visible to the active profile")
            index = ids.index(memory_id)
            meta = dict(metadatas[index] if index < len(metadatas) else {})
            if not _metadata_in_scope(
                meta,
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            ):
                return tool_error("Memory is not visible to the active profile")
            instruction_grade = _as_bool(args.get("instruction_grade"), False)
            kind = _memory_kind(meta.get("memory_kind"), meta.get("content_type", ""))
            if instruction_grade and (status != "confirmed" or kind not in {"decision", "constraint"}):
                return tool_error(
                    "Instruction grade requires a confirmed decision or constraint"
                )
            reviewed_at = datetime.now(timezone.utc).isoformat()
            meta.update({
                "review_status": status,
                "requires_review": status != "confirmed",
                "use_policy": "instruction" if instruction_grade else "evidence",
                "reviewed_at": reviewed_at,
                "reviewed_by": "user_via_hermes",
                "review_reason": reason[:500],
                "updated_at": reviewed_at,
            })
            self._chroma_client.update(
                collection_name=self._config["chroma_collection"],
                ids=[memory_id],
                metadatas=[meta],
            )
            return json.dumps({
                "status": "ok",
                "memory_id": memory_id,
                "review_status": status,
                "instruction_grade": instruction_grade,
                "content_suppressed": True,
            })
        except Exception as e:
            return tool_error(f"Review failed: {e}")

    def _tool_stats(self, args: Dict[str, Any]) -> str:
        try:
            response = self._chroma_client.get(
                collection_name=self._config["chroma_collection"],
                where=_build_scope_where(
                    self._config["workspace_id"],
                    self._config["recall_project_ids"],
                ),
                limit=99999,
            )
        except Exception as e:
            return tool_error(f"Stats failed: {e}")

        ids = response.get("ids", []) or []
        metadatas = response.get("metadatas", []) or []

        content_types: Dict[str, int] = {}
        quality_statuses: Dict[str, int] = {}
        for meta in metadatas:
            if not _metadata_in_scope(
                meta,
                self._config["workspace_id"],
                self._config["recall_project_ids"],
            ):
                continue
            ct = (meta or {}).get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1
            status = (meta or {}).get("review_status", "legacy_unreviewed")
            quality_statuses[status] = quality_statuses.get(status, 0) + 1
        total = sum(quality_statuses.values())

        return json.dumps({
            "total_memories": total,
            "workspace": self._config["workspace_id"],
            "projects": self._config["recall_project_ids"],
            "collection": self._config["chroma_collection"],
            "by_content_type": content_types,
            "by_review_status": quality_statuses,
            "chroma_url": self._config["chroma_url"],
            "embedding_model": self._config["embedding_model"],
        }, indent=2)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_memory_provider(OB1LocalMemoryProvider())
