#!/usr/bin/env python
"""GP map of a problem served by any surrokit MCP server.

Connects to a stdio MCP server built with surrokit.mcp_scaffold.make_server,
Sobol-sweeps the problem's box, fetches posterior mean/sigma through the
`predict` tool, and plots the GP's view of the trade-off: axis-1 mean vs
axis-0 mean (both MAXIMIZED, engine units), colored by sigma(axis 0).
Every number crosses the MCP boundary -- this is what an MCP client
(e.g. Claude) does under the hood.

Usage:
    python examples/gp_map.py --problem NAME --cmd PYTHON SERVER.py [args...]
        [--out map.png] [--n 2000] [--raw-x] [--xline V]

    --raw-x    plot 10**(-axis1) on the x axis (for clients that feed the
               engine -log10(metric), per the README budget recipe)
    --xline V  draw a vertical threshold line at V (x-axis units)

Example (Mu2eBO client):
    python examples/gp_map.py --problem foilspf --raw-x --xline 6.85443e-7 \
        --cmd /path/to/python /path/to/Mu2eBO/surrogate/mcp_server.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout

from _quiet import silence_third_party  # noqa: E402

silence_third_party()


async def fetch(cmd, problem, n, chunk=500):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Forward our environment: the MCP stdio client otherwise starts the
    # server with a stripped env (HOME/PATH/SHELL/TERM/USER only), so a
    # PYTHONPATH-based checkout would fail to import surrokit and surface
    # only as an opaque "Connection closed".
    params = StdioServerParameters(command=cmd[0], args=cmd[1:],
                                   env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            probs = (await s.call_tool("list_problems", {})).structured_content
            if problem not in probs:
                raise SystemExit(
                    f"problem {problem!r} not served; have {sorted(probs)}")
            spec = probs[problem]
            meta = (await s.call_tool(
                "stats", {"problem": problem})).structured_content

            from scipy.stats import qmc
            unit = qmc.Sobol(d=spec["dims"], scramble=True, seed=1).random(n)
            lo, hi = spec["bounds_lo"], spec["bounds_hi"]
            pts = [[lo[j] + u[j] * (hi[j] - lo[j])
                    for j in range(spec["dims"])] for u in unit]
            for j in spec["int_dims"]:
                for p in pts:
                    p[j] = round(p[j])

            mean, sigma = [], []
            for i in range(0, len(pts), chunk):
                out = (await s.call_tool(
                    "predict",
                    {"problem": problem,
                     "points": pts[i:i + chunk]})).structured_content
                mean += out["mean"]
                sigma += out["sigma"]
            return meta, mean, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--problem", required=True)
    ap.add_argument("--cmd", nargs="+", required=True,
                    help="server command line, e.g. python my_server.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--raw-x", action="store_true")
    ap.add_argument("--xline", type=float, default=None)
    a = ap.parse_args()
    out = a.out or f"gp_map_{a.problem}.png"

    meta, mean, sigma = asyncio.run(fetch(a.cmd, a.problem, a.n))
    if len(mean[0]) < 2:
        raise SystemExit("gp_map needs a 2-output problem (axis 0 + axis 1)")
    y = [m[0] for m in mean]
    x = [10 ** -m[1] for m in mean] if a.raw_x else [m[1] for m in mean]
    s0 = [s[0] for s in sigma]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(x, y, c=s0, s=6, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=ax, label=r"GP $\sigma$(axis 0)")
    if a.xline is not None:
        ax.axvline(a.xline, color="crimson", ls="--", lw=1.5,
                   label=f"threshold {a.xline:.3g}")
        ax.legend(loc="lower right")
    if a.raw_x:
        ax.set_xscale("log")
        ax.set_xlabel("10^(-axis 1)  [raw client metric]")
    else:
        ax.set_xlabel("GP-predicted axis 1 (maximized)")
    ax.set_ylabel("GP-predicted axis 0 (maximized)")
    ax.set_title(f"GP map via MCP -- problem={a.problem}, "
                 f"n={meta['n_rows']} history rows, {a.n} Sobol probes")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out} ({a.n} points, {meta['n_rows']}-row history)")


if __name__ == "__main__":
    main()
