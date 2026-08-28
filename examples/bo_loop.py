#!/usr/bin/env python
"""Standalone ask/tell BO loop -- no server, no client repo, just surrokit.

Maximizes a known 2D toy function with the qLogNEI picker: ask() proposes
a batch, we "measure" it (analytic function + a little noise would be the
real world; here it's exact), append to history, repeat. Watch best-so-far
climb toward the true optimum f=1.0 at x=(0.7, 3.0).

Usage:  python examples/bo_loop.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo checkout
from surrokit import Problem, ask

PROB = Problem(bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 10.0), noise=(0.01,))


def f(x):
    """True objective (unknown to the engine): peak 1.0 at (0.7, 3.0)."""
    return math.exp(-8.0 * ((x[0] - 0.7) ** 2 + ((x[1] - 3.0) / 10.0) ** 2))


def main():
    X, Y = [], []
    for r in range(6):
        picks = ask(PROB, X, Y, q=3, picker="qlnei", seed=r)
        for x in picks:
            X.append(x)
            Y.append([f(x)])          # tell: append the measurement
        best = max(Y)[0]
        bx = X[max(range(len(Y)), key=lambda i: Y[i][0])]
        print(f"round {r}: n={len(X):2d}  best f={best:.4f} "
              f"at x=({bx[0]:.3f}, {bx[1]:.3f})")
    print(f"true optimum: f=1.0000 at x=(0.700, 3.000)")


if __name__ == "__main__":
    main()
