import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _stub_modules():
    """Pre-populate sys.modules with lightweight stubs so that
    ``REMSA.agent_orchestrator`` can be imported without the full
    dependency chain (langchain, openai, duckdb …)."""

    stubs = {}

    # --- langchain_core ---
    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.BaseTool = type("BaseTool", (), {"_run": lambda self, x: x})
    stubs["langchain_core"] = lc
    stubs["langchain_core.tools"] = lc_tools

    # --- pydantic (BaseModel, Field) ---
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {"model_fields": {}})
    pydantic.Field = lambda *a, **kw: None
    stubs["pydantic"] = pydantic

    # Stub every REMSA.tools.* and REMSA.config that agent_orchestrator imports
    for mod_name in (
        "REMSA.tools",
        "REMSA.tools.query_parser",
        "REMSA.tools.retrieval_tool",
        "REMSA.tools.ranking_tool",
        "REMSA.tools.explanation_tool",
        "REMSA.tools.clarifier_tool",
        "REMSA.config",
    ):
        m = types.ModuleType(mod_name)
        stubs[mod_name] = m

    # QueryParserTool + QuerySchema used at module level
    qs = type("QuerySchema", (), {"model_fields": {}})
    stubs["REMSA.tools.query_parser"].QueryParserTool = MagicMock
    stubs["REMSA.tools.query_parser"].QuerySchema = qs

    stubs["REMSA.tools.retrieval_tool"].FMDRetrievalTool = MagicMock
    stubs["REMSA.tools.ranking_tool"].RankerTool = MagicMock
    stubs["REMSA.tools.explanation_tool"].ExplanationTool = MagicMock
    stubs["REMSA.tools.clarifier_tool"].ClarifierTool = MagicMock

    stubs["REMSA.config"].config = {
        "EMBEDDING_MODEL_NAME": "test",
        "MAX_CLARIFY": 3,
    }

    # Install stubs only if not already present
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


# Apply stubs before importing the module under test
_stub_modules()

import REMSA.agent_orchestrator as ao  # noqa: E402


class TestCheckBenchmarkEligible(unittest.TestCase):

    def test_benchmark_unavailable_returns_false(self):
        """When _benchmark_available is False, every model returns False."""
        with patch.object(ao, "_benchmark_available", False):
            self.assertFalse(ao._check_benchmark_eligible("prithvi_100m"))
            self.assertFalse(ao._check_benchmark_eligible("prithvi-100m"))
            self.assertFalse(ao._check_benchmark_eligible("unknown"))

    def test_direct_match(self):
        """Model ID that matches the registry directly."""
        mock_registry = MagicMock()
        mock_registry.supports_benchmark.side_effect = lambda mid: mid == "prithvi_100m"

        with patch.object(ao, "_benchmark_available", True), \
             patch.object(ao, "ModelRegistry", mock_registry, create=True):
            self.assertTrue(ao._check_benchmark_eligible("prithvi_100m"))

    def test_hyphen_normalization(self):
        """Model ID with hyphens is normalised to underscores before lookup."""
        mock_registry = MagicMock()
        mock_registry.supports_benchmark.side_effect = lambda mid: mid == "prithvi_100m"

        with patch.object(ao, "_benchmark_available", True), \
             patch.object(ao, "ModelRegistry", mock_registry, create=True):
            self.assertTrue(ao._check_benchmark_eligible("prithvi-100m"))

    def test_unknown_model_returns_false(self):
        """A model not in the registry returns False."""
        mock_registry = MagicMock()
        mock_registry.supports_benchmark.return_value = False

        with patch.object(ao, "_benchmark_available", True), \
             patch.object(ao, "ModelRegistry", mock_registry, create=True):
            self.assertFalse(ao._check_benchmark_eligible("totally_unknown"))

    def test_registry_exception_returns_false(self):
        """If the registry raises, the helper returns False gracefully."""
        mock_registry = MagicMock()
        mock_registry.supports_benchmark.side_effect = RuntimeError("boom")

        with patch.object(ao, "_benchmark_available", True), \
             patch.object(ao, "ModelRegistry", mock_registry, create=True):
            self.assertFalse(ao._check_benchmark_eligible("prithvi_100m"))


if __name__ == "__main__":
    unittest.main()
