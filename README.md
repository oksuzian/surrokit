# surrokit

Generic ask/tell GP surrogate engine: botorch `fit` / `predict` / `ask`
with budget-constrained pickers, plus an MCP server scaffold.

The engine sees only numbers. Every output axis is **maximized**; axis 0
is the primary objective. Clients transform on their side (negate to
minimize, log to tame dynamic range) and keep their units to themselves.

## Install

```bash
pip install "git+https://github.com/oksuzian/surrokit"   # or a checkout
pip install "surrokit[mcp]"                              # + MCP scaffold
```

## Minimize a metric under a budget (5 lines of client code)

Maximize objective `f`, keep metric `g <= budget` — feed the engine
`-log10(g)` and constrain it above `-log10(budget)`:

```python
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

Library discipline: no prints (logger `"surrokit"`), no env reads, no
`sys.exit` — `InfeasibleError` / `NotEnoughData` / `ValueError` instead.
