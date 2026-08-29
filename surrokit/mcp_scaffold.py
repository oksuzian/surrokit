"""MCP server factory: expose any Adapter's problems over the engine.

Import of THIS module requires the optional `mcp` dependency; importing
`surrokit` itself never does.
"""
from __future__ import annotations

from typing import Any, Protocol

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'surrokit.mcp_scaffold requires the "mcp" package: '
        'pip install "mcp>=2.0"') from e

from .gp import fit, predict as gp_predict
from .pickers import PICKER_CHOICES, ask
from .problem import Problem

DEFAULT_INSTRUCTIONS = (
    "Generic GP surrogate engine (surrokit). Problems are named search "
    "spaces with observed history; every output axis is maximized and "
    "axis 0 is the primary objective. predict() gives posterior "
    "mean/sigma per axis; suggest() proposes new points via the "
    f"production pickers {PICKER_CHOICES}. Call list_problems first. "
    "Pure computation -- nothing is submitted or written anywhere."
)


class Adapter(Protocol):
    """problems()/history() are required. An adapter MAY additionally
    provide suggest(name, q, picker, round_idx, pending) -> list[list];
    if it does, the server's suggest tool delegates to it -- the
    adapter's own pick path, vocabulary, and seed derivation replace the
    generic stateless ask() (use this when picks must match what the
    client's production loop would submit)."""
    def problems(self) -> dict[str, Problem]: ...
    def history(self, name: str) -> tuple[list, list, dict]: ...


def make_server(adapter: Adapter, name: str = "surrokit",
                instructions: str = DEFAULT_INSTRUCTIONS,
                middleware=None) -> MCPServer:
    kwargs: dict[str, Any] = {"name": name, "instructions": instructions}
    if middleware is not None:
        kwargs["middleware"] = middleware
    server = MCPServer(**kwargs)

    # problem name -> (model, n_rows_at_fit). Append-only-history
    # assumption: same row count == same data. refit() is the escape
    # hatch when a client rewrites history in place.
    cache: dict[str, tuple[object, int]] = {}

    def _resolve(pname: str) -> tuple[Problem, list, list, dict]:
        problems = adapter.problems()
        if pname not in problems:
            raise ValueError(f"unknown problem {pname!r}; choose from "
                             f"{sorted(problems)}")
        X, Y, meta = adapter.history(pname)
        return problems[pname], X, Y, meta

    def _fitted(pname: str, refresh: bool = False):
        problem, X, Y, meta = _resolve(pname)
        n = len(X)
        c = cache.get(pname)
        if c is not None and c[1] == n and not refresh:
            return c[0], n, meta
        model = fit(problem, X, Y)
        cache[pname] = (model, n)
        return model, n, meta

    @server.tool(structured_output=True)
    def list_problems() -> dict[str, Any]:
        """Named problems: dims, bounds, integer dims, output-axis count,
        history row counts."""
        out: dict[str, Any] = {}
        for pname, prob in sorted(adapter.problems().items()):
            X, Y, _ = adapter.history(pname)
            out[pname] = {
                "dims": prob.dim,
                "bounds_lo": list(prob.bounds_lo),
                "bounds_hi": list(prob.bounds_hi),
                "int_dims": list(prob.int_dims),
                "axes": (len(Y[0]) if Y else None),
                "n_rows": len(X),
                "constrained": prob.constraint is not None,
            }
        return out

    @server.tool(structured_output=True)
    def predict(problem: str, points: list[list[float]]) -> dict[str, Any]:
        """Posterior mean and sigma per output axis at each point."""
        model, _, _ = _fitted(problem)
        mean, sigma = gp_predict(model, points)
        return {"mean": mean.tolist(), "sigma": sigma.tolist()}

    if hasattr(adapter, "suggest"):
        @server.tool(structured_output=True)
        def suggest(problem: str, q: int = 5, picker: str | None = None,
                    round_idx: int = 0,
                    pending: list[list[float]] | None = None,
                    ) -> list[list[float]]:
            """Propose q new points via the adapter's own production pick
            path: picker vocabulary, history policy, and per-round seed
            derivation all match the client's optimization loop, so these
            picks are what a real round would submit. picker=None uses
            the adapter's default. pending: x-rows already in flight."""
            _resolve(problem)  # uniform unknown-problem error
            return adapter.suggest(problem, q=q, picker=picker,
                                   round_idx=round_idx, pending=pending)
    else:
        @server.tool(structured_output=True)
        def suggest(problem: str, q: int = 5, picker: str = "hybrid",
                    seed: int = 0,
                    pending: list[list[float]] | None = None,
                    min_spacing: float = 0.10,
                    hv_frac: float = 0.6) -> list[list[float]]:
            """Propose q new points (stateless ask over the current
            history). pending: x-rows already in flight, to steer picks
            away from."""
            prob, X, Y, _ = _resolve(problem)
            return ask(prob, X, Y, q=q, picker=picker, seed=seed,
                       pending=pending, min_spacing=min_spacing,
                       hv_frac=hv_frac)

    @server.tool(structured_output=True)
    def stats(problem: str) -> dict[str, Any]:
        """History row count plus whatever metadata the adapter serves."""
        _, X, _, meta = _resolve(problem)
        out: dict[str, Any] = {"problem": problem, "n_rows": len(X)}
        out.update(meta)
        return out

    @server.tool(structured_output=True)
    def refit(problem: str) -> dict[str, Any]:
        """Bypass the fit cache and refit on the current history."""
        _, n, _ = _fitted(problem, refresh=True)
        return {"problem": problem, "n_rows": n, "refitted": True}

    return server
