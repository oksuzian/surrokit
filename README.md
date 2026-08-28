# surrokit

Generic ask/tell GP surrogate engine: botorch `fit` / `predict` / `ask`
with budget-constrained pickers, plus an MCP server scaffold.

The engine sees only numbers. Every output axis is **maximized**; axis 0
is the primary objective. Clients transform on their side (negate to
minimize, log to tame dynamic range) and keep their units to themselves.

## Install

Python >= 3.10.

```bash
pip install "git+https://github.com/oksuzian/surrokit"                # engine
pip install "surrokit[mcp] @ git+https://github.com/oksuzian/surrokit"  # + MCP scaffold
```

Or from a checkout: `pip install -e ".[mcp]"`. (Not on PyPI yet, so the
plain `pip install surrokit` form does not work.)

## Minimize a metric under a budget (5 lines of client code)

Maximize objective `f`, keep metric `g <= budget` — feed the engine
`-log10(g)` and constrain it above `-log10(budget)`:

```python
import math
from surrokit import Problem, Constraint, ask
prob = Problem(bounds_lo=(0, 0), bounds_hi=(1, 10),
               constraint=Constraint(axis=1, min=-math.log10(budget)))
Y = [[f_i, -math.log10(g_i)] for f_i, g_i in observations]
picks = ask(prob, X, Y, q=5, picker="constrained_max", seed=run_seed)
```

## API

- `Problem(bounds_lo, bounds_hi, int_dims=(), noise=None, constraint=None)`
  — search box; `noise` = ABSOLUTE per-axis sigma (pins a fixed-noise GP;
  strongly recommended when you have replicate measurements).
- `fit(problem, X, Y) -> model`; `predict(model, X) -> (mean, sigma)`.
- `ask(problem, X, Y, q=5, picker="hybrid", seed=0, pending=None,
  min_spacing=0.10, pool=16384, hv_frac=0.6)` — **stateless** (refits per
  call). Pickers: `qnehvi | qlnei | qnparego | hybrid | constrained_max`.
  `seed` is used verbatim in every RNG stream. Fewer than 2 rows falls
  back to a Sobol draw.
- `surrokit.mcp_scaffold.make_server(adapter)` — MCP server over any
  `Adapter` (`problems()`, `history(name)`); tools `list_problems`,
  `predict`, `suggest`, `stats`, `refit`; `middleware=` forwards to
  `MCPServer`.

## Serve your problems over MCP

Implement the two-method `Adapter` protocol and hand it to
`make_server` — you get an MCP server whose tools (`list_problems`,
`predict`, `suggest`, `stats`, `refit`) any MCP client (Claude Code,
Claude Desktop, ...) can call:

```python
# my_server.py
from surrokit import Problem
from surrokit.mcp_scaffold import make_server

class MyAdapter:
    def problems(self):
        # name -> search space
        return {"toy": Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0))}

    def history(self, name):
        # observed rows for `name`: X (n, d), Y (n, m) — every axis
        # maximized — plus free-form metadata served by stats()
        X = [[0.2, 3.0], [0.8, 7.0]]
        Y = [[1.1, 2.5], [1.4, 2.1]]
        return X, Y, {"objectives": ["f", "-log10(g)"]}

server = make_server(MyAdapter(), name="my-surrogate")

if __name__ == "__main__":
    server.run("stdio")
```

Register it with an MCP client (e.g. a Claude Code `.mcp.json`):

```json
{
  "mcpServers": {
    "my-surrogate": {
      "command": "python",
      "args": ["/path/to/my_server.py"]
    }
  }
}
```

The client can then ask things like "list the problems", "predict at
x=[0.5, 5]", or "suggest 5 new points with picker qlnei, seed 7" —
all pure computation over the history your adapter serves; nothing is
submitted or written anywhere.

Real-world example: [Mu2eBO](https://github.com/Mu2e/Mu2eBO)'s
`surrogate/adapter.py` + `surrogate/mcp_server.py` serve its BO
leaderboards (7 detector-geometry search spaces) exactly this way.

`examples/gp_map.py` is a ready-made client: point it at any such
server and it Sobol-sweeps the box through the `predict` tool and
renders the GP's trade-off map (axis 1 vs axis 0, colored by sigma).

Fully self-contained demos (no client repo needed):

```bash
python examples/bo_loop.py            # ask/tell loop finds a toy optimum
python examples/gp_map.py --problem toy \
    --cmd python examples/toy_server.py   # GP map over a toy MCP server
```

Library discipline: no prints (logger `"surrokit"`), no env reads, no
`sys.exit` — `InfeasibleError` / `NotEnoughData` / `ValueError` instead.
