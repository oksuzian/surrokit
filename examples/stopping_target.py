#!/usr/bin/env python
"""The whole of surrokit in one example, on real physics data.

The Mu2e experiment stops muons in a stack of thin aluminium foils. Two
things fight each other:

  sob   -- signal-to-background, S/sqrt(B), for the mu->e conversion
           search. Bigger is better.
  flash -- energy per proton-on-target that the prompt "beam flash"
           deposits in the tracker. It damages the detector, so the
           deployed design's value is a budget you may not exceed.

Ten knobs describe the foil stack as quadratic profiles: outer radius,
half-thickness and spacing fraction at three control points, plus the
total z extent. The 90 rows in data/ are real evaluations from a
Bayesian-optimization campaign (github.com/Mu2e/Mu2eBO), each one a
Geant4 simulation of millions of protons.

Axis 0 is sob, axis 1 is -log10(flash) -- so BOTH axes are maximized, as
the engine requires, and "flash <= budget" becomes the README's budget
recipe, "axis 1 >= -log10(budget)".

Three acts, in order:

  1. fit / predict  -- the GP over the measured designs
  2. ask            -- what to simulate next, with and without the budget
  3. MCP            -- the same two problems served over stdio and called
                       back by a client, which is what Claude does

Usage:
    python examples/stopping_target.py             # all three acts
    python examples/stopping_target.py --plot      # + write the GP map PNG
    python examples/stopping_target.py --serve     # run only as MCP server

Act 3 can also drive somebody else's surrokit server:
    python examples/stopping_target.py --server-cmd python other_server.py \
        --problem their_problem_name
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout

from _quiet import silence_third_party  # noqa: E402

silence_third_party()

from surrokit import Constraint, InfeasibleError, Problem, ask, fit, predict

DATA = Path(__file__).resolve().parent / "data" / "stopping_target_foilspf.csv"

KNOBS = ["rOut_0", "rOut_1", "rOut_2", "hT_0", "hT_1", "hT_2",
         "f_0", "f_1", "f_2", "extent"]
BOUNDS_LO = (50.0, 50.0, 50.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 400.0)
BOUNDS_HI = (120.0, 120.0, 120.0, 0.15, 0.15, 0.15, 0.95, 0.95, 0.95, 2000.0)

# Replicate-measured sigmas: 0.6% on sob, 0.01 on -log10(flash). Pinning
# them matters -- a free-noise fit once ranked the best-ever design 16th.
NOISE = (0.006, 0.01)

# The deployed target's beam-flash damage budget, MeV/POT.
BUDGET = 6.85443e-7

FREE = Problem(bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI, noise=NOISE)
UNDER_BUDGET = Problem(
    bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI, noise=NOISE,
    constraint=Constraint(axis=1, min=-math.log10(BUDGET), k_sigma=1.0))
PROBLEMS = {"stopping_target": FREE, "stopping_target_budget": UNDER_BUDGET}


def load():
    """(X, Y) in engine space: Y = [sob, -log10(flash)], both maximized."""
    X, Y, flash = [], [], []
    for row in csv.DictReader(DATA.open()):
        X.append([float(row[k]) for k in KNOBS])
        Y.append([float(row["sob"]), -math.log10(float(row["flash_edep"]))])
        flash.append(float(row["flash_edep"]))
    return X, Y, flash


# --------------------------------------------------------------- act 3 server

class StoppingTargetAdapter:
    """Two named problems over one history -- all a server needs to serve."""

    def problems(self):
        return PROBLEMS

    def history(self, name):
        X, Y, _ = load()
        return X, Y, {"objectives": ["sob", "-log10(flash_edep)"],
                      "budget_MeV_per_POT": BUDGET}


def serve():
    # A server shares the client's stderr, and importing `mcp` puts a rich
    # INFO handler on the root logger -- so the engine's per-fit chatter
    # would land in the caller's terminal. Keep the wire quiet.
    import logging
    logging.getLogger("surrokit").setLevel(logging.WARNING)

    from surrokit.mcp_scaffold import make_server
    make_server(StoppingTargetAdapter(), name="surrokit-stopping-target",
                instructions="Mu2e stopping-target geometry. Axis 0 is "
                             "signal-to-background (maximize); axis 1 is "
                             "-log10(beam-flash damage), so larger means "
                             "less damage. Problem stopping_target_budget "
                             "additionally holds flash under the deployed "
                             "target's budget.").run("stdio")


# --------------------------------------------------------------- act 3 client

async def over_mcp(cmd, problem, n_probe):
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
            call = lambda t, a={}: s.call_tool(t, a)  # noqa: E731

            served = (await call("list_problems")).structured_content
            for pname, spec in sorted(served.items()):
                print(f"  {pname}: {spec['dims']}D, {spec['n_rows']} rows, "
                      f"{spec['axes']} axes, "
                      f"constrained={spec['constrained']}")
            if problem not in served:
                raise SystemExit(f"{problem!r} not served by that command; "
                                 f"have {sorted(served)}")

            meta = (await call("stats", {"problem": problem})
                    ).structured_content
            print(f"  stats({problem}): {meta.get('objectives')}")

            X, _, _ = load()
            out = (await call("predict", {"problem": problem,
                                          "points": [X[0]]})
                   ).structured_content
            m = out["mean"][0]
            print(f"  predict at design 0: sob = {m[0]:.3f}, flash = "
                  f"{10 ** -m[1]:.3e} MeV/POT   <-- same GP, over stdio")

            picks = (await call("suggest",
                                {"problem": problem, "q": 2,
                                 "picker": "constrained_max", "seed": 0})
                     ).structured_content["result"]
            pm = (await call("predict", {"problem": problem, "points": picks})
                  ).structured_content["mean"]
            print(f"  suggest(q=2, constrained_max):")
            for p in pm:
                print(f"    sob ~ {p[0]:.2f}  flash ~ {10 ** -p[1]:.2e}")

            if not n_probe:
                return None
            from scipy.stats import qmc
            spec = served[problem]
            unit = qmc.Sobol(d=spec["dims"], scramble=True,
                             seed=1).random(n_probe)
            lo, hi = spec["bounds_lo"], spec["bounds_hi"]
            pts = [[lo[j] + u[j] * (hi[j] - lo[j]) for j in range(spec["dims"])]
                   for u in unit]
            mean, sigma = [], []
            for i in range(0, len(pts), 500):
                out = (await call("predict", {"problem": problem,
                                              "points": pts[i:i + 500]})
                       ).structured_content
                mean += out["mean"]
                sigma += out["sigma"]
            return mean, sigma


def plot(mean, sigma, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter([10 ** -m[1] for m in mean], [m[0] for m in mean],
                    c=[s[0] for s in sigma], s=6, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=ax, label=r"GP $\sigma$(sob)")
    ax.axvline(BUDGET, color="crimson", ls="--", lw=1.5, label="damage budget")
    ax.legend(loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("GP-predicted beam flash [MeV/POT]")
    ax.set_ylabel("GP-predicted sob")
    ax.set_title(f"What the GP believes, sampled through MCP ({len(mean)} pts)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------- acts

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve", action="store_true",
                    help="run as an MCP server on stdio and exit")
    ap.add_argument("--plot", action="store_true",
                    help="also write a GP map PNG (needs matplotlib)")
    ap.add_argument("--out", default="gp_map_stopping_target.png")
    ap.add_argument("--n", type=int, default=2000, help="Sobol probes to plot")
    ap.add_argument("--server-cmd", nargs="+", default=None,
                    help="drive another surrokit server instead of ourselves")
    ap.add_argument("--problem", default="stopping_target_budget")
    a = ap.parse_args()

    if a.serve:
        serve()
        return

    X, Y, flash = load()
    best = max(range(len(Y)), key=lambda i: Y[i][0])
    n_within = sum(1 for g in flash if g <= BUDGET)
    print(f"[1] {len(X)} evaluated designs, {len(KNOBS)} knobs")
    print(f"  best measured sob = {Y[best][0]:.3f} at flash = "
          f"{flash[best]:.3e} MeV/POT "
          f"({'within' if flash[best] <= BUDGET else 'OVER'} budget)")
    print(f"  {n_within}/{len(X)} designs are within the "
          f"{BUDGET:.3e} MeV/POT budget")

    # The GP should reproduce a measured point it was trained on.
    model = fit(FREE, X, Y)
    mean, sigma = predict(model, [X[best]])
    print(f"  GP at the best design: sob = {mean[0][0]:.3f} +- "
          f"{sigma[0][0]:.3f} (measured {Y[best][0]:.3f}), flash = "
          f"{10 ** -mean[0][1]:.3e} MeV/POT")

    print("\n[2] qnehvi -- explore the trade-off front:")
    for x in ask(FREE, X, Y, q=3, picker="qnehvi", seed=0):
        m, _ = predict(model, [x])
        print(f"  sob ~ {m[0][0]:.2f}  flash ~ {10 ** -m[0][1]:.2e}")

    # Deployment-facing: highest sob the GP believes stays under budget,
    # at mean - 1 sigma so a design whose true flash overshoots is not
    # proposed. Raise the budget and the picks move to higher sob.
    print(f"  constrained_max -- flash <= {BUDGET:.3e} at mean - 1 sigma:")
    try:
        for x in ask(UNDER_BUDGET, X, Y, q=3, picker="constrained_max",
                     seed=0):
            m, s = predict(model, [x])
            print(f"  sob ~ {m[0][0]:.2f}  flash ~ {10 ** -m[0][1]:.2e}"
                  f"  (+{s[0][0]:.2f} sob uncertainty)")
    except InfeasibleError as e:
        print(f"  no feasible design: {e}")

    cmd = a.server_cmd or [sys.executable, str(Path(__file__).resolve()),
                           "--serve"]
    print(f"\n[3] the same engine over MCP ({' '.join(cmd[-2:])}):")
    try:
        import mcp  # noqa: F401
    except ImportError:
        print('  skipped -- pip install "mcp>=2.0" to run the server half')
        return
    got = asyncio.run(over_mcp(cmd, a.problem, a.n if a.plot else 0))
    if got:
        plot(*got, a.out)


if __name__ == "__main__":
    main()
