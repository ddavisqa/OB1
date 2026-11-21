from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
import types
import unittest

HERMES_REPO = Path("/Users/ddavis/.hermes/hermes-agent")
PLUGIN = Path(__file__).resolve().parents[1] / "__init__.py"
sys.path.insert(0, str(HERMES_REPO))
spec = importlib.util.spec_from_file_location("ob1_local_integrity_tested", PLUGIN)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class FakeChroma:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class MemoryIntegrityTests(unittest.TestCase):
    def provider(self):
        p = mod.OB1LocalMemoryProvider()
        p._active = True
        p._auto_recall = True
        return p

    def test_prefetch_cache_matches_case_and_spacing(self):
        p = self.provider()
        p._prefetch_cache = ("Health   Insurance", "cached", "r1", time.monotonic())
        self.assertEqual(p._consume_prefetch_cache(" health insurance "), ("cached", "r1"))

    def test_prefetch_cache_rejects_previous_turn_query(self):
        p = self.provider()
        p._prefetch_cache = ("portfolio allocation", "wrong context", "r1", time.monotonic())
        self.assertIsNone(p._consume_prefetch_cache("health insurance"))
        self.assertIsNone(p._prefetch_cache)

    def test_prefetch_cache_rejects_expired_entry(self):
        p = self.provider()
        p._prefetch_cache = (
            "health insurance", "stale", "r1",
            time.monotonic() - mod._PREFETCH_TTL_SECONDS - 1,
        )
        self.assertIsNone(p._consume_prefetch_cache("health insurance"))

    def test_prefetch_mismatch_runs_fresh_recall(self):
        p = self.provider()
        p._prefetch_cache = ("old query", "old", "r-old", time.monotonic())
        p._do_recall = lambda query, session_id="": (f"fresh:{query}", "r-new")
        self.assertEqual(p.prefetch("new query"), "fresh:new query")
        self.assertEqual(p._last_request_id, "r-new")

    def test_restricted_notice_contains_no_document_content(self):
        p = self.provider()
        p._config = {
            "financial_project_id": "financial",
            "recall_project_ids": ["general"],
            "workspace_id": "ddavis",
            "chroma_collection": "ob1_memory",
        }
        secret = "private portfolio amount 123456"
        p._chroma_client = FakeChroma({
            "metadatas": [[{
                "workspace_id": "ddavis", "project_id": "financial",
                "review_status": "pending", "valid_to": "", "superseded_by": "",
            }]],
            "distances": [[0.20]],
            "documents": [[secret]],
            "ids": [["private-id"]],
        })
        notice = p._restricted_evidence_notice("Should I rebalance my portfolio?", [0.1, 0.2])
        self.assertIn("Restricted Memory Signal", notice)
        self.assertNotIn(secret, notice)
        self.assertNotIn("private-id", notice)
        self.assertNotIn("financial", notice.casefold())

    def test_restricted_probe_skips_nonconsequential_query(self):
        p = self.provider()
        p._config = {
            "financial_project_id": "financial", "recall_project_ids": ["general"],
            "workspace_id": "ddavis", "chroma_collection": "ob1_memory",
        }
        p._chroma_client = FakeChroma({})
        self.assertEqual(p._restricted_evidence_notice("Refactor this parser", [0.1]), "")
        self.assertEqual(p._chroma_client.calls, [])

    def test_restricted_probe_respects_similarity_floor(self):
        p = self.provider()
        p._config = {
            "financial_project_id": "financial", "recall_project_ids": ["general"],
            "workspace_id": "ddavis", "chroma_collection": "ob1_memory",
        }
        p._chroma_client = FakeChroma({
            "metadatas": [[{"workspace_id": "ddavis", "project_id": "financial"}]],
            "distances": [[0.90]],
        })
        self.assertEqual(p._restricted_evidence_notice("portfolio review", [0.1]), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
