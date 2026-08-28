"""Shared fixtures: a tiny 2D problem with deterministic history."""
import math

from surrokit import Constraint, Problem

# 2D box, dim 1 is integer-valued. noise pinned (fixed-noise GP).
PROB = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
               int_dims=(1,), noise=(0.01, 0.01))

# Same box with a budget constraint on axis 1 (feasible: mean1 - k*sig1 >= 2.0).
PROB_C = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0),
                 int_dims=(1,), noise=(0.01, 0.01),
                 constraint=Constraint(axis=1, min=2.0, k_sigma=1.0))


def history(n=10):
    """Deterministic (X, Y): smooth 2-output landscape, both maximized."""
    X = [[i / max(n - 1, 1), float(i % 10)] for i in range(n)]
    Y = [[math.sin(3.0 * x0) + 0.1 * x1, 3.0 - 0.2 * x1 + 0.5 * x0]
         for x0, x1 in X]
    return X, Y


def in_bounds(row, lo=(0.0, 0.0), hi=(1.0, 10.0)):
    return all(lo[i] <= v <= hi[i] for i, v in enumerate(row))
