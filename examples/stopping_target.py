#!/usr/bin/env python
"""Real physics: Mu2e stopping-target geometry, 90 simulated designs.

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

This is the README's budget recipe on real data: axis 0 is sob, axis 1
is -log10(flash) -- so BOTH axes are maximized, as the engine requires,
and "flash <= budget" becomes "axis 1 >= -log10(budget)".

Usage:  python examples/stopping_target.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout

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


def load():
    """(X, Y) in engine space: Y = [sob, -log10(flash)], both maximized."""
    X, Y, flash = [], [], []
    for row in csv.DictReader(DATA.open()):
        X.append([float(row[k]) for k in KNOBS])
        Y.append([float(row["sob"]), -math.log10(float(row["flash_edep"]))])
        flash.append(float(row["flash_edep"]))
    return X, Y, flash


def main():
    X, Y, flash = load()
    best = max(range(len(Y)), key=lambda i: Y[i][0])
    n_within = sum(1 for g in flash if g <= BUDGET)
    print(f"{len(X)} evaluated designs, {len(KNOBS)} knobs")
    print(f"  best measured sob = {Y[best][0]:.3f} at flash = "
          f"{flash[best]:.3e} MeV/POT "
          f"({'within' if flash[best] <= BUDGET else 'OVER'} budget)")
    print(f"  {n_within}/{len(X)} designs are within the "
          f"{BUDGET:.3e} MeV/POT budget")

    problem = Problem(bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI, noise=NOISE)

    # The GP should reproduce a measured point it was trained on.
    model = fit(problem, X, Y)
    mean, sigma = predict(model, [X[best]])
    print(f"\nGP at the best design: sob = {mean[0][0]:.3f} +- "
          f"{sigma[0][0]:.3f} (measured {Y[best][0]:.3f}), flash = "
          f"{10 ** -mean[0][1]:.3e} MeV/POT")

    # Unconstrained: what the multi-objective picker proposes next.
    print("\nqnehvi (explore the trade-off front):")
    for x in ask(problem, X, Y, q=3, picker="qnehvi", seed=0):
        m, _ = predict(model, [x])
        print(f"  sob ~ {m[0][0]:.2f}  flash ~ {10 ** -m[0][1]:.2e}")

    # Deployment-facing: highest sob the GP believes stays under budget,
    # at mean - 1 sigma so a design whose true flash overshoots is not
    # proposed. Raise the budget and the picks move to higher sob.
    budget_problem = Problem(
        bounds_lo=BOUNDS_LO, bounds_hi=BOUNDS_HI, noise=NOISE,
        constraint=Constraint(axis=1, min=-math.log10(BUDGET), k_sigma=1.0))
    print(f"\nconstrained_max (flash <= {BUDGET:.3e} at mean - 1 sigma):")
    try:
        for x in ask(budget_problem, X, Y, q=3, picker="constrained_max",
                     seed=0):
            m, s = predict(model, [x])
            print(f"  sob ~ {m[0][0]:.2f}  flash ~ {10 ** -m[0][1]:.2e}"
                  f"  (+{s[0][0]:.2f} sob uncertainty)")
    except InfeasibleError as e:
        print(f"  no feasible design: {e}")


if __name__ == "__main__":
    main()
