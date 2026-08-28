import asyncio
import unittest

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

from tests.common import PROB, PROB_C, history


class DictAdapter:
    def __init__(self):
        self._h = {"toy": history(10), "ctoy": history(10)}

    def problems(self):
        return {"toy": PROB, "ctoy": PROB_C}

    def history(self, name):
        X, Y = self._h[name]
        return X, Y, {"note": f"meta-for-{name}"}


@unittest.skipUnless(HAVE_MCP, "mcp SDK not installed")
class TestScaffold(unittest.TestCase):
    def setUp(self):
        from surrokit.mcp_scaffold import make_server
        self.server = make_server(DictAdapter())

    def test_tool_names(self):
        tools = asyncio.run(self.server.list_tools())
        names = sorted(t.name for t in tools)
        self.assertEqual(names, ["list_problems", "predict", "refit",
                                 "stats", "suggest"])

    def test_call_list_problems(self):
        result = asyncio.run(self.server.call_tool("list_problems", {}))
        # MCPServer.call_tool returns a CallToolResult in this mcp SDK
        # (mcp_types.CallToolResult); the structured payload lives on
        # .structured_content, not a tuple index.
        structured = result.structured_content
        self.assertIn("toy", structured)
        self.assertEqual(structured["toy"]["dims"], 2)
        self.assertEqual(structured["toy"]["n_rows"], 10)

    def test_call_predict(self):
        result = asyncio.run(self.server.call_tool(
            "predict", {"problem": "toy", "points": [[0.5, 5.0]]}))
        structured = result.structured_content
        self.assertEqual(len(structured["mean"]), 1)
        self.assertEqual(len(structured["mean"][0]), 2)
        self.assertEqual(len(structured["sigma"][0]), 2)

    def test_call_suggest(self):
        result = asyncio.run(self.server.call_tool(
            "suggest", {"problem": "toy", "q": 2, "picker": "qlnei",
                        "seed": 0}))
        structured = result.structured_content
        picks = structured["result"] if isinstance(structured, dict) else structured
        self.assertEqual(len(picks), 2)

    def test_call_stats_merges_meta(self):
        result = asyncio.run(self.server.call_tool("stats",
                                                   {"problem": "ctoy"}))
        structured = result.structured_content
        self.assertEqual(structured["n_rows"], 10)
        self.assertEqual(structured["note"], "meta-for-ctoy")

    def test_refit_returns_row_count(self):
        result = asyncio.run(self.server.call_tool("refit",
                                                   {"problem": "toy"}))
        self.assertEqual(result.structured_content["n_rows"], 10)


class TestLazyImport(unittest.TestCase):
    def test_surrokit_import_never_needs_mcp(self):
        import surrokit  # noqa: F401  -- must not raise even without mcp
