#!/usr/bin/env python
"""Minimal self-contained MCP server: a toy 2-output problem over synthetic
history. Pairs with examples/gp_map.py for a zero-dependency demo:

    python examples/gp_map.py --problem toy --cmd python examples/toy_server.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout

from _quiet import silence_third_party  # noqa: E402

silence_third_party()
from surrokit import Problem
from surrokit.mcp_scaffold import make_server


class ToyAdapter:
    def problems(self):
        return {"toy": Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
                               noise=(0.01, 0.01))}

    def history(self, name):
        # 30 deterministic rows: axis 0 peaks at (0.7, 3.0); axis 1 is a
        # smooth "cost proxy" (both MAXIMIZED, per the engine contract).
        X = [[i / 29.0, (7 * i) % 30 / 3.0] for i in range(30)]
        Y = [[math.exp(-8.0 * ((x0 - 0.7) ** 2 + ((x1 - 3.0) / 10.0) ** 2)),
              3.0 - 0.2 * x1 + 0.5 * x0] for x0, x1 in X]
        return X, Y, {"objectives": ["toy-f", "toy-cost-proxy"]}


server = make_server(ToyAdapter(), name="surrokit-toy")

if __name__ == "__main__":
    server.run("stdio")
