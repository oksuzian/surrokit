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


@unittest.skipUnless(HAVE_MCP, "mcp SDK not installed")
class TestScaffoldSuggestHook(unittest.TestCase):
    """An adapter with suggest() takes over the suggest tool."""

    class HookAdapter(DictAdapter):
        def suggest(self, name, q=5, picker=None, round_idx=0,
                    pending=None):
            self.called_with = (name, q, picker, round_idx, pending)
            return [[0.5, 5]] * q

    def setUp(self):
        from surrokit.mcp_scaffold import make_server
        self.adapter = self.HookAdapter()
        self.server = make_server(self.adapter)

    def test_delegates_and_speaks_round_idx(self):
        result = asyncio.run(self.server.call_tool(
            "suggest", {"problem": "toy", "q": 2, "picker": "budget_sob",
                        "round_idx": 3}))
        structured = result.structured_content
        picks = structured["result"] if isinstance(structured, dict) else structured
        self.assertEqual(picks, [[0.5, 5], [0.5, 5]])
        self.assertEqual(self.adapter.called_with,
                         ("toy", 2, "budget_sob", 3, None))

    def test_unknown_problem_still_uniform_error(self):
        with self.assertRaises(Exception) as cm:
            asyncio.run(self.server.call_tool("suggest", {"problem": "nope"}))
        self.assertIn("unknown problem", str(cm.exception))

    def test_schema_has_round_idx_not_seed(self):
        tools = asyncio.run(self.server.list_tools())
        sg = next(t for t in tools if t.name == "suggest")
        props = sg.input_schema["properties"] if hasattr(sg, "input_schema") \
            else sg.inputSchema["properties"]
        self.assertIn("round_idx", props)
        self.assertNotIn("seed", props)


@unittest.skipUnless(HAVE_MCP, "mcp SDK not installed")
class TestScaffoldFitCacheStaleness(unittest.TestCase):
    """The fit cache keys on row COUNT (append-only assumption): an
    in-place history rewrite at the same count serves the STALE model,
    and refit() is the documented escape hatch."""

    def setUp(self):
        from surrokit.mcp_scaffold import make_server
        self.adapter = DictAdapter()
        self.server = make_server(self.adapter)

    def _predict(self):
        result = asyncio.run(self.server.call_tool(
            "predict", {"problem": "toy", "points": [[0.5, 5.0]]}))
        return result.structured_content["mean"]

    def test_same_count_rewrite_is_stale_until_refit(self):
        before = self._predict()
        X, Y = self.adapter._h["toy"]
        self.adapter._h["toy"] = (X, [[y0 + 1.0, y1] for y0, y1 in Y])
        self.assertEqual(self._predict(), before)  # stale cache served
        asyncio.run(self.server.call_tool("refit", {"problem": "toy"}))
        after = self._predict()
        self.assertNotEqual(after, before)
        self.assertAlmostEqual(after[0][0], before[0][0] + 1.0, places=2)
