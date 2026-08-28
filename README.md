# surrokit

Generic ask/tell GP surrogate engine: botorch `fit` / `predict` / `ask`
with budget-constrained pickers, plus an MCP server scaffold.

The engine sees only numbers. Every output axis is **maximized**; axis 0
is the primary objective. Clients transform on their side (negate to
minimize, log to tame dynamic range) and keep their units to themselves.

## Install

Needs Python >= 3.11 (botorch's floor). Check `pip --version` — it names
the interpreter it belongs to; an older one fails misleadingly with
`No matching distribution found for botorch>=0.18`.

```bash
pip install "git+https://github.com/oksuzian/surrokit"                  # engine
pip install "surrokit[mcp] @ git+https://github.com/oksuzian/surrokit"  # + MCP scaffold
pip install --no-deps "git+https://github.com/oksuzian/surrokit"        # env already has torch/botorch
```

Not on PyPI yet, so plain `pip install surrokit` does not work. From a
checkout: `pip install -e ".[mcp]"`.

On Mu2e/FNAL machines the interpreter you want is `ana 2.8.0` (Python
3.12, botorch 0.18.1); the login shell's `python`/`pip` are 3.9 and
cannot install botorch at all:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana 2.8.0     # always name the version -- bare `pyenv ana` is 2.7.0
python -m pip install --no-deps "git+https://github.com/oksuzian/surrokit"
```

In a shell that also launches Mu2e jobs, use the interpreter path
(`/cvmfs/mu2e.opensciencegrid.org/env/ana/2.8.0/bin/python`) instead of
activating: `pyenv`'s `python`/`pip` wrappers re-prepend the env's
site-packages and libs into every child process and shadow muse's ROOT.

## Quickstart

`pip install` ships the package only, so the examples come from a clone
(they run straight from it, installed or not):

```bash
git clone https://github.com/oksuzian/surrokit && cd surrokit
python examples/bo_loop.py         # ask/tell loop climbs to a toy optimum
python examples/stopping_target.py  # the budget recipe on 90 real designs
python examples/gp_map.py --problem toy \
    --cmd python examples/toy_server.py    # GP map over a toy MCP server
```

`stopping_target.py` is the honest end-to-end case: 90 Geant4-simulated
Mu2e stopping-target geometries (10 knobs), maximizing signal-to-
background while holding beam-flash damage under the deployed target's
budget. The best measured design is over budget; `constrained_max`
proposes the highest-signal geometries the GP believes stay under it.

## API

- `Problem(bounds_lo, bounds_hi, int_dims=(), noise=None, constraint=None)`
  — the search box. `noise` = ABSOLUTE per-axis sigma, which pins a
  fixed-noise GP; strongly recommended when you have replicate
  measurements, since a free-noise fit can erase a real optimum.
- `fit(problem, X, Y) -> model`; `predict(model, X) -> (mean, sigma)`.
- `ask(problem, X, Y, q=5, picker="hybrid", seed=0, pending=None,
  min_spacing=0.10, pool=16384, hv_frac=0.6) -> list[list]` —
  **stateless**: refits per call. Pickers: `qnehvi | qlnei | qnparego |
  hybrid | constrained_max`. `seed` is used verbatim in every RNG stream
  (derive per-round seeds on your side). Fewer than 2 rows falls back to
  a Sobol draw. `pending` rows are conditioned on but never returned.
- Errors: `InfeasibleError`, `NotEnoughData`, `ValueError` — the library
  never prints (logger `"surrokit"`), reads the environment, or exits.

### Minimize a metric under a budget

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

## Serve your problems over MCP

`make_server(adapter)` turns any two-method adapter into an MCP server
with the tools `list_problems`, `predict`, `suggest`, `stats`, `refit`
— callable by Claude Code, Claude Desktop, or any MCP client:

```python
# my_server.py
from surrokit import Problem
from surrokit.mcp_scaffold import make_server

class MyAdapter:
    def problems(self):
        # name -> search space
        return {"toy": Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0))}

    def history(self, name):
        # observed rows: X (n, d), Y (n, m) — every axis maximized —
        # plus free-form metadata served by stats()
        X = [[0.2, 3.0], [0.8, 7.0]]
        Y = [[1.1, 2.5], [1.4, 2.1]]
        return X, Y, {"objectives": ["f", "-log10(g)"]}

server = make_server(MyAdapter(), name="my-surrogate")

if __name__ == "__main__":
    server.run("stdio")
```

Register it with a client (e.g. a Claude Code `.mcp.json`):

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

Then ask it to "list the problems", "predict at x=[0.5, 5]", or "suggest
5 points with picker qlnei, seed 7". Everything is pure computation over
the history your adapter serves; nothing is submitted or written.
`make_server` also takes `middleware=`, forwarded to `MCPServer`.

`examples/gp_map.py` is a ready-made client for any such server: it
Sobol-sweeps the box through `predict` and plots the GP's trade-off map.
[Mu2eBO](https://github.com/Mu2e/Mu2eBO) is the real-world adapter —
`surrogate/adapter.py` + `surrogate/mcp_server.py` serve its Bayesian
optimization leaderboards (7 detector-geometry search spaces) this way.
